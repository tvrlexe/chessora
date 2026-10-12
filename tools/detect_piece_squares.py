import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from board_detection import get_board_points
from grid_reconstruction import detect_segments, get_grid
from perspective import get_perspective_matrix
from piece_detection import (
    bottom_mask,
    get_piece_predictions,
    prediction_to_mask,
)
from square_extraction import LAYOUTS
import main as chessora_main


def confidence_of(prediction):
    if isinstance(prediction, dict):
        return float(prediction.get("confidence", 0.0))
    return float(getattr(prediction, "confidence", 0.0))


def grid_positions(lines, name):
    values = np.asarray(lines, dtype=np.float64).reshape(-1)

    if len(values) != 9 or not np.all(np.isfinite(values)):
        raise RuntimeError(
            f"Expected nine numeric {name} grid positions; "
            f"received shape {np.asarray(lines).shape}"
        )

    if not np.all(np.diff(values) > 0):
        raise RuntimeError(f"{name} grid positions are not increasing")

    return values


def prediction_mask(prediction, expected_shape, source):
    """Decode a prediction and verify its image coordinate system."""
    mask = prediction_to_mask(prediction, expected_shape)

    if mask is None:
        return np.zeros(expected_shape[:2], dtype=np.uint8)

    mask = (np.asarray(mask) > 0).astype(np.uint8)

    if mask.shape != expected_shape[:2]:
        raise RuntimeError(
            f"{source} prediction mask has shape {mask.shape}; "
            f"expected {expected_shape[:2]}. Check prediction_to_mask()."
        )

    return mask


def base_mask_and_center(warped_mask):
    """Return the visible base mask and its median point in warped coordinates."""
    base = bottom_mask((warped_mask > 0).astype(np.uint8))

    if base is None:
        return None, None

    base = (np.asarray(base) > 0).astype(np.uint8)

    if base.shape != warped_mask.shape:
        raise RuntimeError(
            f"bottom_mask() returned {base.shape}; "
            f"expected {warped_mask.shape}"
        )

    ys, xs = np.nonzero(base)

    if len(xs) == 0:
        return base, None

    return base, (float(np.median(xs)), float(np.median(ys)))


def square_for_point(point, horizontal, vertical, layout):
    if point is None:
        return None

    x, y = point
    col = int(np.searchsorted(vertical, x, side="right") - 1)
    row = int(np.searchsorted(horizontal, y, side="right") - 1)

    if not (0 <= row < 8 and 0 <= col < 8):
        return None

    return layout[row][col]


def make_candidates(
    predictions,
    source,
    source_shape,
    matrix,
    size,
    horizontal,
    vertical,
    layout,
):
    candidates = []

    for index, prediction in enumerate(predictions, start=1):
        mask = prediction_mask(prediction, source_shape, source)

        if source == "camera":
            warped_mask = cv2.warpPerspective(
                mask,
                matrix,
                (size, size),
                flags=cv2.INTER_NEAREST,
            )
        else:
            warped_mask = mask

        warped_base, point = base_mask_and_center(warped_mask)

        candidates.append({
            "source": source,
            "index": index,
            "confidence": confidence_of(prediction),
            "square": square_for_point(
                point, horizontal, vertical, layout
            ),
            "base_center": point,
            "warped_base": warped_base,
        })

    return candidates


def group_by_square(candidates):
    """
    Select one candidate per square within a view.

    Competing masks are reported separately; they are not counted as
    additional occupied squares.
    """
    grouped = defaultdict(list)
    unmapped = []

    for candidate in candidates:
        if candidate["square"] is None:
            unmapped.append(candidate)
        else:
            grouped[candidate["square"]].append(candidate)

    selected = {}
    competing = {}

    for square, group in grouped.items():
        ordered = sorted(
            group,
            key=lambda candidate: candidate["confidence"],
            reverse=True,
        )
        selected[square] = ordered[0]

        if len(ordered) > 1:
            competing[square] = ordered[1:]

    return selected, competing, unmapped


def draw_base(annotated, candidate, inverse_matrix, color):
    base = candidate["warped_base"]

    if base is None:
        return

    h, w = annotated.shape[:2]
    camera_base = cv2.warpPerspective(
        (base > 0).astype(np.uint8),
        inverse_matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
    )
    annotated[camera_base > 0] = color


def annotate(
    image,
    camera_selected,
    camera_competing,
    warped_only,
    inverse_matrix,
    matrix,
    horizontal,
    vertical,
    orientation,
):
    """Draw camera bases in red and warped-only bases in amber."""
    chessora_main.perspective_matrix = matrix
    chessora_main.horizontal_lines = horizontal
    chessora_main.vertical_lines = vertical
    chessora_main.orientation_square = orientation
    chessora_main.orientation_ready = True

    with chessora_main.frame_lock:
        chessora_main.frame = image.copy()

    annotated = chessora_main.get_live_annotated_frame()

    if annotated is None:
        annotated = image.copy()

    for square, candidate in camera_selected.items():
        # Amber also marks camera squares with competing camera masks.
        color = (
            (0, 190, 255)
            if square in camera_competing
            else (0, 0, 255)
        )
        draw_base(annotated, candidate, inverse_matrix, color)

    for candidate in warped_only.values():
        draw_base(
            annotated,
            candidate,
            inverse_matrix,
            (0, 190, 255),
        )

    merged_count = len(camera_selected) + len(warped_only)

    cv2.rectangle(
        annotated,
        (0, 0),
        (image.shape[1], 34),
        (20, 20, 20),
        -1,
    )
    cv2.putText(
        annotated,
        (
            f"{merged_count} square candidates; "
            f"{len(warped_only)} warped-only"
        ),
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return annotated


def print_competing(source, selected, competing):
    for square, extras in sorted(competing.items()):
        winner = selected[square]
        others = ", ".join(
            f"{item['index']} ({item['confidence']:.3f})"
            for item in extras
        )
        print(
            f"{source} competing masks on {square}: "
            f"selected {winner['index']} "
            f"({winner['confidence']:.3f}); others {others}"
        )


def print_unmapped(source, unmapped):
    for candidate in unmapped:
        print(
            f"{source} unmapped mask {candidate['index']}: "
            f"confidence {candidate['confidence']:.3f}, "
            f"base={candidate['base_center']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Assign chess-piece masks from camera and warped views "
            "to board squares."
        )
    )
    parser.add_argument(
        "image",
        nargs="?",
        default="photos/photo.jpg",
    )
    parser.add_argument(
        "--orientation",
        choices=("a8", "h8", "a1", "h1"),
        default="a8",
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    image = cv2.imread(args.image)

    if image is None:
        raise RuntimeError(f"Could not read image: {args.image}")

    print(
        f"Image: {args.image} "
        f"({image.shape[1]}x{image.shape[0]})"
    )

    board_corners = get_board_points(image)
    segments = detect_segments(image)

    size = int(chessora_main.SIZE)
    matrix = get_perspective_matrix(board_corners, size)
    inverse_matrix = np.linalg.inv(matrix)

    horizontal_lines, vertical_lines = get_grid(
        image,
        board_corners,
        segments,
    )
    horizontal = grid_positions(horizontal_lines, "horizontal")
    vertical = grid_positions(vertical_lines, "vertical")
    layout = LAYOUTS[args.orientation]

    warped_image = cv2.warpPerspective(
        image,
        matrix,
        (size, size),
    )

    print("Detecting pieces on camera image...")
    camera_predictions = get_piece_predictions(image)

    print("Detecting pieces on perspective-corrected board...")
    warped_predictions = get_piece_predictions(warped_image)

    camera_candidates = make_candidates(
        camera_predictions,
        "camera",
        image.shape,
        matrix,
        size,
        horizontal,
        vertical,
        layout,
    )
    warped_candidates = make_candidates(
        warped_predictions,
        "warped",
        warped_image.shape,
        matrix,
        size,
        horizontal,
        vertical,
        layout,
    )

    camera_selected, camera_competing, camera_unmapped = (
        group_by_square(camera_candidates)
    )
    warped_selected, warped_competing, warped_unmapped = (
        group_by_square(warped_candidates)
    )

    # Camera detections take priority. Warped detections fill only squares
    # absent from the camera result, without inventing missing pieces.
    warped_only = {
        square: candidate
        for square, candidate in warped_selected.items()
        if square not in camera_selected
    }

    merged = dict(camera_selected)
    merged.update(warped_only)

    if args.output:
        output_path = Path(args.output)
    else:
        source_path = Path(args.image)
        output_path = source_path.with_name(
            f"{source_path.stem}_piece_squares.jpg"
        )

    annotated = annotate(
        image,
        camera_selected,
        camera_competing,
        warped_only,
        inverse_matrix,
        matrix,
        horizontal,
        vertical,
        args.orientation,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError(f"Could not save {output_path}")

    print()
    print(f"Camera-view raw masks: {len(camera_predictions)}")
    print(f"Warped-view raw masks: {len(warped_predictions)}")
    print(f"Camera-detected squares: {len(camera_selected)}")
    print(f"Warped-only square candidates: {len(warped_only)}")
    print(f"Merged square candidates: {len(merged)}")
    print(
        "Merged squares: "
        + (", ".join(sorted(merged)) or "none")
    )
    print(
        "Warped-only squares needing review: "
        + (", ".join(sorted(warped_only)) or "none")
    )

    print_competing(
        "camera",
        camera_selected,
        camera_competing,
    )
    print_competing(
        "warped",
        warped_selected,
        warped_competing,
    )

    print_unmapped("camera", camera_unmapped)
    print_unmapped("warped", warped_unmapped)
    print(f"Annotated image: {output_path}")


if __name__ == "__main__":
    main()
