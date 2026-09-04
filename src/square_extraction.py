import cv2
import numpy as np


A1_CORNER = "top-right"


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


def get_intersections(horizontal, vertical):

    intersections = []

    for row in range(9):

        current_row = []

        for col in range(9):

            point = line_intersection(
                horizontal[row],
                vertical[col]
            )

            current_row.append(point)

        intersections.append(current_row)

    return np.array(intersections, dtype=np.float32)


def square_name(row, col):

    if A1_CORNER == "top-right":
        file_index = row
        rank = 8 - col

    elif A1_CORNER == "top-left":
        file_index = row
        rank = col + 1

    elif A1_CORNER == "bottom-right":
        file_index = 7 - row
        rank = 8 - col

    elif A1_CORNER == "bottom-left":
        file_index = 7 - row
        rank = col + 1

    else:
        raise ValueError("Invalid A1_CORNER")

    file = chr(ord("a") + file_index)

    return f"{file}{rank}"


def extract_squares(image, horizontal, vertical):

    intersections = get_intersections(
        horizontal,
        vertical
    )

    squares = {}

    for row in range(8):

        for col in range(8):

            polygon = np.array([
                intersections[row][col],
                intersections[row][col + 1],
                intersections[row + 1][col + 1],
                intersections[row + 1][col]
            ], dtype=np.float32)

            x, y, w, h = cv2.boundingRect(
                polygon.astype(np.int32)
            )

            crop = image[y:y + h, x:x + w]

            mask = np.zeros(
                (h, w),
                dtype=np.uint8
            )

            local_polygon = polygon - [x, y]

            cv2.fillPoly(
                mask,
                [local_polygon.astype(np.int32)],
                255
            )

            square = cv2.bitwise_and(
                crop,
                crop,
                mask=mask
            )

            name = square_name(row, col)

            squares[name] = square

    return squares


def annotate_grid(
    image,
    horizontal,
    vertical,
    labels=True
):

    output = image.copy()

    intersections = get_intersections(
        horizontal,
        vertical
    )

    for row in range(9):

        points = np.round(
            horizontal[row]
        ).astype(int)

        cv2.line(
            output,
            tuple(points[0]),
            tuple(points[1]),
            (0, 255, 0),
            2
        )

    for col in range(9):

        points = np.round(
            vertical[col]
        ).astype(int)

        cv2.line(
            output,
            tuple(points[0]),
            tuple(points[1]),
            (0, 255, 0),
            2
        )

    if not labels:
        return output

    for row in range(8):

        for col in range(8):

            name = square_name(row, col)

            center = (
                intersections[row][col]
                + intersections[row][col + 1]
                + intersections[row + 1][col + 1]
                + intersections[row + 1][col]
            ) / 4

            center = np.round(
                center
            ).astype(int)

            cv2.putText(
                output,
                name,
                tuple(center),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

    return output