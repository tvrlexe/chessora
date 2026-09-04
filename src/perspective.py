import cv2
import numpy as np


SIZE = 800


def get_perspective_matrix(points, size=SIZE):

    destination = np.array([
        [0, 0],
        [size, 0],
        [size, size],
        [0, size]
    ], dtype=np.float32)

    return cv2.getPerspectiveTransform(
        points.astype(np.float32),
        destination
    )


def warp_board(image, matrix, size=SIZE):

    return cv2.warpPerspective(
        image,
        matrix,
        (size, size)
    )