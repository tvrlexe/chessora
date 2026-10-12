import cv2
import numpy as np


def grid_to_camera(matrix, horizontal, vertical, size=800):
    inverse_matrix = np.linalg.inv(matrix)

    camera_horizontal = []
    camera_vertical = []

    for y in horizontal:
        points = np.array(
            [[0, y], [size - 1, y]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(
            points,
            inverse_matrix,
        ).reshape(-1, 2)
        camera_horizontal.append((transformed[0], transformed[1]))

    for x in vertical:
        points = np.array(
            [[x, 0], [x, size - 1]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(
            points,
            inverse_matrix,
        ).reshape(-1, 2)
        camera_vertical.append((transformed[0], transformed[1]))

    return camera_horizontal, camera_vertical
