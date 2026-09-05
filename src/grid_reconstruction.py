import cv2
import numpy as np


MIN_GAP = 60
MAX_GAP = 120
TOLERANCE = 15
SIZE = 800
GRID_LINES = 9


def merge(lines, threshold=20):

    lines = sorted(lines)

    if not lines:
        return []

    groups = [[lines[0]]]

    for line in lines[1:]:

        if line - np.mean(groups[-1]) <= threshold:
            groups[-1].append(line)

        else:
            groups.append([line])

    return [
        np.mean(group)
        for group in groups
    ]


def find_sequence(lines):

    lines = sorted(lines)

    if len(lines) < 2:
        raise RuntimeError("Not enough lines")

    best = None

    for i in range(len(lines)):

        for j in range(i + 1, len(lines)):

            distance = lines[j] - lines[i]

            for multiple in range(1, GRID_LINES):

                gap = distance / multiple

                if gap < MIN_GAP or gap > MAX_GAP:
                    continue

                for first_index in range(
                    GRID_LINES - multiple
                ):

                    second_index = (
                        first_index + multiple
                    )

                    sequence = {
                        first_index: (
                            lines[i],
                            i
                        ),
                        second_index: (
                            lines[j],
                            j
                        )
                    }

                    used = {i, j}

                    # -------------------------------------------------
                    # Add detected lines that agree with the spacing.
                    # -------------------------------------------------

                    for index in range(GRID_LINES):

                        if index in sequence:
                            continue

                        expected = (
                            lines[i]
                            + (index - first_index) * gap
                        )

                        candidates = []

                        for k, line in enumerate(lines):

                            if k in used:
                                continue

                            error = abs(
                                line - expected
                            )

                            if error <= TOLERANCE:
                                candidates.append(
                                    (error, k)
                                )

                        if candidates:

                            _, best_candidate = min(
                                candidates
                            )

                            sequence[index] = (
                                lines[best_candidate],
                                best_candidate
                            )

                            used.add(best_candidate)

                    if len(sequence) < 3:
                        continue

                    # -------------------------------------------------
                    # Fit one evenly-spaced grid to the detected lines.
                    # -------------------------------------------------

                    indices = np.array(
                        list(sequence.keys()),
                        dtype=np.float64
                    )

                    positions = np.array(
                        [
                            sequence[index][0]
                            for index in sequence
                        ],
                        dtype=np.float64
                    )

                    A = np.column_stack(
                        [
                            np.ones(len(indices)),
                            indices
                        ]
                    )

                    solution, _, _, _ = np.linalg.lstsq(
                        A,
                        positions,
                        rcond=None
                    )

                    origin = solution[0]
                    refined_gap = solution[1]

                    if (
                        refined_gap < MIN_GAP
                        or refined_gap > MAX_GAP
                    ):
                        continue

                    fitted = (
                        origin
                        + indices * refined_gap
                    )

                    errors = np.abs(
                        positions - fitted
                    )

                    mean_error = np.mean(errors)

                    if np.max(errors) > TOLERANCE:
                        continue

                    # -------------------------------------------------
                    # Make sure reconstructed grid fits the board.
                    # -------------------------------------------------

                    grid = (
                        origin
                        + np.arange(GRID_LINES)
                        * refined_gap
                    )

                    if grid[0] < 0:
                        continue

                    if grid[-1] > SIZE - 1:
                        continue

                    # Prefer more detected lines, then lower error.
                    score = (
                        len(sequence),
                        -mean_error
                    )

                    if (
                        best is None
                        or score > best["score"]
                    ):

                        best = {
                            "sequence": sequence.copy(),
                            "origin": origin,
                            "gap": refined_gap,
                            "error": mean_error,
                            "score": score
                        }

    if best is None:
        raise RuntimeError(
            "Could not find a valid line sequence"
        )

    return best


def reconstruct(lines):

    result = find_sequence(lines)

    sequence = result["sequence"]
    gap = result["gap"]
    origin = result["origin"]

    print("Best sequence:")

    for index in sorted(sequence):

        line, _ = sequence[index]

        print(
            f"  {line:.1f} -> index {index}"
        )

    print(
        f"Gap: {gap:.1f}"
    )

    # ---------------------------------------------------------
    # Reconstruct the complete grid from the fitted origin
    # and spacing.
    #
    # The Hough detections determine the grid.
    # The final grid itself is evenly spaced.
    # ---------------------------------------------------------

    grid = (
        origin
        + np.arange(GRID_LINES)
        * gap
    )

    print(
        "Reconstructed:",
        np.round(grid, 1)
    )

    print(
        "Gaps:",
        np.round(np.diff(grid), 1)
    )

    return grid


def get_grid(warped):

    gray = cv2.cvtColor(
        warped,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        20,
        minLineLength=40,
        maxLineGap=50
    )

    if lines is None:
        raise RuntimeError(
            "No Hough lines detected"
        )

    horizontal = []
    vertical = []

    for x1, y1, x2, y2 in lines.reshape(-1, 4):

        angle = abs(
            np.degrees(
                np.arctan2(
                    y2 - y1,
                    x2 - x1
                )
            )
        )

        if angle < 10 or angle > 170:

            horizontal.append(
                (y1 + y2) / 2
            )

        elif 80 < angle < 100:

            vertical.append(
                (x1 + x2) / 2
            )

    horizontal = merge(horizontal)
    vertical = merge(vertical)

    print(
        "Horizontal:",
        np.round(horizontal, 1)
    )

    print(
        "Vertical:",
        np.round(vertical, 1)
    )

    horizontal = reconstruct(horizontal)
    vertical = reconstruct(vertical)

    return horizontal, vertical