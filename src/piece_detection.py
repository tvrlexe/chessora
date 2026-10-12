import os

import cv2
import numpy as np

from board_detection import decode_rle_mask, run_workflow
from square_extraction import LAYOUTS


BOARD_SIZE = 800
BOTTOM_MASK_FRACTION = 0.40
SQUARE_PRESENCE_THRESHOLD = 0.40
CENTER_CELL_BASE_THRESHOLD = 0.08
PIECE_WORKFLOW_URL = os.getenv("ROBOFLOW_PIECE_WORKFLOW_URL")


def _find_prediction_lists(node):
    if isinstance(node, dict):
        if (
            isinstance(node.get("predictions"), list)
        ):
            yield node.get("image"), node["predictions"]
        for value in node.values():
            yield from _find_prediction_lists(value)
    elif isinstance(node, list):
        for value in node:
            yield from _find_prediction_lists(value)


def _board_crop_bounds(outputs):
    for output in outputs:
        if not isinstance(output, dict):
            continue

        board_result = output.get("board_predictions")
        if not isinstance(board_result, dict):
            continue

        for prediction in board_result.get("predictions", []):
            if prediction.get("class", "").lower() != "chessboard":
                continue

            rle_mask = prediction.get("rle_mask")
            if not rle_mask:
                continue

            mask = decode_rle_mask(rle_mask)
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if not contours:
                continue

            x, y, width, height = cv2.boundingRect(
                max(contours, key=cv2.contourArea)
            )
            if width > 0 and height > 0:
                return x, y, width, height

    return None


def get_piece_predictions(image):
    workflow_url = PIECE_WORKFLOW_URL
    if workflow_url is None:
        result = run_workflow(image, "chess piece")
    else:
        result = run_workflow(
            image,
            "chess piece",
            workflow_url=workflow_url,
        )

    try:
        outputs = result["outputs"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            f"Unexpected Workflow response structure: {result}"
        ) from error

    if not isinstance(outputs, list):
        raise RuntimeError(
            f"Unexpected Workflow outputs structure: {result}"
        )

    prediction_batches = list(_find_prediction_lists(outputs))
    if not prediction_batches:
        raise RuntimeError(
            f"Workflow response did not contain predictions: {result}"
        )

    board_bounds = _board_crop_bounds(outputs)
    predictions = []
    seen = set()

    for image_info, prediction_list in prediction_batches:
        source_height, source_width = image.shape[:2]
        if not isinstance(image_info, dict):
            image_info = {"width": source_width, "height": source_height}
        prediction_width = int(image_info["width"])
        prediction_height = int(image_info["height"])
        if (prediction_width, prediction_height) != (source_width, source_height):
            if board_bounds is None:
                raise RuntimeError(
                    "Piece masks use cropped-image coordinates, but the Workflow "
                    "response has no chessboard mask to map them to the camera frame"
                )

        for prediction in prediction_list:
            if not isinstance(prediction, dict):
                continue

            class_name = prediction.get("class", "")

            if (
                not isinstance(class_name, str)
                or "piece" not in class_name.lower()
            ):
                continue

            if "points" not in prediction and "rle_mask" not in prediction:
                raise RuntimeError(
                    f"Piece prediction has no segmentation mask: {prediction}"
                )

            identity = id(prediction)
            if identity not in seen:
                normalized_prediction = dict(prediction)
                normalized_prediction["_inference_image_size"] = (
                    prediction_width,
                    prediction_height,
                )
                normalized_prediction["_board_crop_bounds"] = board_bounds
                predictions.append(normalized_prediction)
                seen.add(identity)

    return predictions


def prediction_to_mask(prediction, image_shape):
    height, width = image_shape[:2]
    inference_size = prediction.get(
        "_inference_image_size",
        (width, height),
    )
    inference_width, inference_height = inference_size
    crop_bounds = prediction.get("_board_crop_bounds")
    if (inference_width, inference_height) != (width, height):
        if crop_bounds is None:
            raise RuntimeError("No board bounds available for cropped piece mask")
        crop_x, crop_y, crop_width, crop_height = crop_bounds
    else:
        crop_x, crop_y, crop_width, crop_height = 0, 0, width, height

    rle_mask = prediction.get("rle_mask")

    if rle_mask:
        mask = decode_rle_mask(rle_mask)
        if mask.shape != (inference_height, inference_width):
            mask = cv2.resize(
                mask,
                (inference_width, inference_height),
                interpolation=cv2.INTER_NEAREST,
            )
        mask = cv2.resize(
            mask,
            (crop_width, crop_height),
            interpolation=cv2.INTER_NEAREST,
        )
        full_mask = np.zeros((height, width), dtype=np.uint8)
        right = min(width, crop_x + crop_width)
        bottom = min(height, crop_y + crop_height)
        if crop_x < right and crop_y < bottom:
            full_mask[crop_y:bottom, crop_x:right] = (
                mask[:bottom - crop_y, :right - crop_x] > 0
            ).astype(np.uint8)
        return full_mask

    points = prediction["points"]
    polygon = np.array(
        [
            [
                round(crop_x + float(point["x"]) * crop_width / inference_width),
                round(crop_y + float(point["y"]) * crop_height / inference_height),
            ]
            for point in points
        ],
        dtype=np.int32,
    )

    if len(polygon) < 3:
        raise RuntimeError("Piece prediction polygon requires at least 3 points")

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    return mask


def bottom_mask(mask, fraction=BOTTOM_MASK_FRACTION):
    if not 0 < fraction <= 1:
        raise ValueError("Bottom-mask fraction must be in the range (0, 1]")

    ys = np.nonzero(mask)[0]
    result = np.zeros_like(mask, dtype=np.uint8)

    if len(ys) == 0:
        return result

    first_row = int(ys.min())
    last_row = int(ys.max()) + 1
    mask_height = last_row - first_row
    cutoff = last_row - max(1, int(np.ceil(mask_height * fraction)))
    result[cutoff:last_row, :] = mask[cutoff:last_row, :]
    return result


def locate_piece_square(
    mask,
    perspective_matrix,
    orientation_square,
    horizontal_lines=None,
    vertical_lines=None,
):
    warped_base = cv2.warpPerspective(
        bottom_mask(mask),
        perspective_matrix,
        (BOARD_SIZE, BOARD_SIZE),
        flags=cv2.INTER_NEAREST,
    )
    base_pixels = int(np.count_nonzero(warped_base))

    if base_pixels == 0:
        return None, 0.0

    horizontal_lines = (
        np.linspace(0, BOARD_SIZE, 9)
        if horizontal_lines is None
        else np.asarray(horizontal_lines, dtype=np.float32)
    )
    vertical_lines = (
        np.linspace(0, BOARD_SIZE, 9)
        if vertical_lines is None
        else np.asarray(vertical_lines, dtype=np.float32)
    )

    if (
        horizontal_lines.shape != (9,)
        or vertical_lines.shape != (9,)
        or np.any(np.diff(horizontal_lines) <= 0)
        or np.any(np.diff(vertical_lines) <= 0)
    ):
        raise ValueError("Grid reconstruction must provide 9 increasing lines per axis")

    full_mask = cv2.warpPerspective(
        mask,
        perspective_matrix,
        (BOARD_SIZE, BOARD_SIZE),
        flags=cv2.INTER_NEAREST,
    )
    full_ys, full_xs = np.nonzero(full_mask)
    if len(full_xs):
        center_x = float(np.mean(full_xs))
        center_y = float(np.mean(full_ys))
        row = int(np.searchsorted(horizontal_lines, center_y, side="right") - 1)
        col = int(np.searchsorted(vertical_lines, center_x, side="right") - 1)

        if 0 <= row < 8 and 0 <= col < 8:
            center_square = square_for_cell(
                row,
                col,
                orientation_square,
            )
            y0 = max(0, int(np.floor(horizontal_lines[row])))
            y1 = min(BOARD_SIZE, int(np.ceil(horizontal_lines[row + 1])))
            x0 = max(0, int(np.floor(vertical_lines[col])))
            x1 = min(BOARD_SIZE, int(np.ceil(vertical_lines[col + 1])))
            center_cell_base = int(
                np.count_nonzero(warped_base[y0:y1, x0:x1])
            )

            center_cell_coverage = center_cell_base / base_pixels
            if center_cell_coverage >= CENTER_CELL_BASE_THRESHOLD:
                return center_square, center_cell_coverage

    best_square = None
    best_coverage = 0.0

    for row in range(8):
        y0 = max(0, int(np.floor(horizontal_lines[row])))
        y1 = min(BOARD_SIZE, int(np.ceil(horizontal_lines[row + 1])))

        for col in range(8):
            x0 = max(0, int(np.floor(vertical_lines[col])))
            x1 = min(BOARD_SIZE, int(np.ceil(vertical_lines[col + 1])))
            if x0 >= x1 or y0 >= y1:
                continue
            overlap = int(np.count_nonzero(warped_base[y0:y1, x0:x1]))
            coverage = overlap / base_pixels

            if coverage > best_coverage:
                best_square = square_for_cell(
                    row,
                    col,
                    orientation_square,
                )
                best_coverage = coverage

    if best_coverage < SQUARE_PRESENCE_THRESHOLD:
        return None, best_coverage

    return best_square, best_coverage


def square_for_cell(row, col, orientation_square):
    if orientation_square not in LAYOUTS:
        raise ValueError(
            "Invalid layout; expected a8, h8, a1, or h1"
        )

    return LAYOUTS[orientation_square][row][col]


def classify_piece_predictions(
    predictions,
    image_shape,
    perspective_matrix,
    orientation_square,
    horizontal_lines=None,
    vertical_lines=None,
):
    detections = []

    for prediction in predictions:
        mask = prediction_to_mask(prediction, image_shape)
        square, coverage = locate_piece_square(
            mask,
            perspective_matrix,
            orientation_square,
            horizontal_lines,
            vertical_lines,
        )
        detections.append(
            {
                "class": prediction.get("class", "piece"),
                "confidence": float(prediction.get("confidence", 0.0)),
                "square": square,
                "coverage": coverage,
            }
        )

    return detections
