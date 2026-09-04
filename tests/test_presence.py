import sys
import os

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "src"
    )
)

import cv2
import numpy as np

from board_detection import get_board_points
from perspective import get_perspective_matrix, warp_board
from grid_reconstruction import get_grid
from square_extraction import extract_squares, annotate_grid


SIZE = 800


def structure_score(square):

    gray = cv2.cvtColor(
        square,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitude = cv2.magnitude(
        gx,
        gy
    )

    return np.mean(magnitude)


def find_empty_squares(squares):

    scores = {}

    for name, square in squares.items():

        scores[name] = structure_score(square)

    empty = sorted(
        scores,
        key=scores.get
    )[:32]

    return empty, scores


def map_grid_to_camera(horizontal, vertical, matrix):

    inverse = np.linalg.inv(matrix)

    camera_horizontal = []
    camera_vertical = []

    for y in horizontal:

        points = np.array(
            [[[0, y], [SIZE, y]]],
            dtype=np.float32
        )

        mapped = cv2.perspectiveTransform(
            points,
            inverse
        )[0]

        camera_horizontal.append(mapped)

    for x in vertical:

        points = np.array(
            [[[x, 0], [x, SIZE]]],
            dtype=np.float32
        )

        mapped = cv2.perspectiveTransform(
            points,
            inverse
        )[0]

        camera_vertical.append(mapped)

    return camera_horizontal, camera_vertical


def main():

    if len(sys.argv) != 2:

        print(
            "Usage: python tests/test_presence.py image.jpg"
        )

        return

    image_path = sys.argv[1]

    image = cv2.imread(image_path)

    if image is None:

        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    print("Detecting board...")

    points = get_board_points(image)

    print("Calculating perspective...")

    matrix = get_perspective_matrix(
        points,
        SIZE
    )

    print("Warping board for grid detection...")

    warped = warp_board(
        image,
        matrix,
        SIZE
    )

    print("Detecting grid...")

    horizontal, vertical = get_grid(
        warped
    )

    print("Mapping grid to camera...")

    camera_horizontal, camera_vertical = map_grid_to_camera(
        horizontal,
        vertical,
        matrix
    )

    print("Extracting squares...")

    squares = extract_squares(
        image,
        camera_horizontal,
        camera_vertical
    )

    print("Scoring squares...")

    empty, scores = find_empty_squares(
        squares
    )

    print()
    print("32 lowest-scoring squares:")
    print(empty)

    print()
    print("Scores:")

    for name in sorted(
        scores,
        key=lambda x: (
            int(x[1:]),
            x[0]
        )
    ):

        print(
            f"{name}: {scores[name]:.2f}"
        )

    print()
    print("Creating annotation...")

    debug = annotate_grid(
        image,
        camera_horizontal,
        camera_vertical
    )

    output_path = os.path.join(
        os.path.dirname(image_path),
        "presence_debug.jpg"
    )

    cv2.imwrite(
        output_path,
        debug
    )

    print()
    print(
        f"Debug image saved to: {output_path}"
    )


if __name__ == "__main__":
    main()