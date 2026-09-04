import cv2

from camera import start_camera, read_frame
from board_detection import get_board_points
from perspective import get_perspective_matrix, warp_board
from grid_reconstruction import get_grid


SIZE = 800


camera = start_camera()

print("Waiting for camera frame...")

frame = read_frame(camera)

if frame is None:
    raise RuntimeError("Could not read camera frame")

print("Frame received")

points = get_board_points(frame)

M = get_perspective_matrix(points, SIZE)

warped = warp_board(
    frame,
    M,
    SIZE
)

horizontal, vertical = get_grid(warped)

print("Horizontal:", horizontal.round(1))
print("Vertical:", vertical.round(1))

camera.terminate()