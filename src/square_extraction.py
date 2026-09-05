import cv2
import numpy as np


LAYOUTS = {
    "a8": [
        ["a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1"],
        ["a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2"],
        ["a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3"],
        ["a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4"],
        ["a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5"],
        ["a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6"],
        ["a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7"],
        ["a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8"],
    ],

    "h8": [
        ["a8", "a7", "a6", "a5", "a4", "a3", "a2", "a1"],
        ["b8", "b7", "b6", "b5", "b4", "b3", "b2", "b1"],
        ["c8", "c7", "c6", "c5", "c4", "c3", "c2", "c1"],
        ["d8", "d7", "d6", "d5", "d4", "d3", "d2", "d1"],
        ["e8", "e7", "e6", "e5", "e4", "e3", "e2", "e1"],
        ["f8", "f7", "f6", "f5", "f4", "f3", "f2", "f1"],
        ["g8", "g7", "g6", "g5", "g4", "g3", "g2", "g1"],
        ["h8", "h7", "h6", "h5", "h4", "h3", "h2", "h1"],
    ],

    "a1": [
        ["h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8"],
        ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8"],
        ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"],
        ["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8"],
        ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8"],
        ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"],
        ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8"],
        ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8"],
    ],

    "h1": [
        ["h8", "g8", "f8", "e8", "d8", "c8", "b8", "a8"],
        ["h7", "g7", "f7", "e7", "d7", "c7", "b7", "a7"],
        ["h6", "g6", "f6", "e6", "d6", "c6", "b6", "a6"],
        ["h5", "g5", "f5", "e5", "d5", "c5", "b5", "a5"],
        ["h4", "g4", "f4", "e4", "d4", "c4", "b4", "a4"],
        ["h3", "g3", "f3", "e3", "d3", "c3", "b3", "a3"],
        ["h2", "g2", "f2", "e2", "d2", "c2", "b2", "a2"],
        ["h1", "g1", "f1", "e1", "d1", "c1", "b1", "a1"],
    ],
}


def line_intersection(line1, line2):

    x1, y1 = line1[0]
    x2, y2 = line1[1]

    x3, y3 = line2[0]
    x4, y4 = line2[1]

    denominator = (
        (x1 - x2) * (y3 - y4)
        - (y1 - y2) * (x3 - x4)
    )

    if abs(denominator) < 1e-6:
        raise RuntimeError("Grid lines are parallel")

    px = (
        (x1 * y2 - y1 * x2) * (x3 - x4)
        - (x1 - x2) * (x3 * y4 - y3 * x4)
    ) / denominator

    py = (
        (x1 * y2 - y1 * x2) * (y3 - y4)
        - (y1 - y2) * (x3 * y4 - y3 * x4)
    ) / denominator

    return np.array([px, py], dtype=np.float32)


def square_name(row, col, square):

    return LAYOUTS[square][row][col]


def annotate_grid(
    image,
    horizontal_lines,
    vertical_lines,
    square
):

    if square not in LAYOUTS:
        raise ValueError(
            "Invalid square. Use a8, h8, a1, or h1."
        )

    output = image.copy()

    for row in range(8):

        for col in range(8):

            top_left = line_intersection(
                horizontal_lines[row],
                vertical_lines[col]
            )

            bottom_right = line_intersection(
                horizontal_lines[row + 1],
                vertical_lines[col + 1]
            )

            center = (
                (top_left + bottom_right) / 2
            ).astype(int)

            name = square_name(
                row,
                col,
                square
            )

            cv2.putText(
                output,
                name,
                tuple(center),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

    return output

