import cv2
import numpy as np
from itertools import combinations
from chessboard_segmentation import get_board_points


IMAGE = "photos/photo.jpg"
OUTPUT = "photos/new_angle.jpg"
SIZE = 800


# Board + perspective warp

image = cv2.imread(IMAGE)

points = get_board_points(IMAGE).astype(np.float32)

destination = np.array([
    [0, 0],
    [SIZE, 0],
    [SIZE, SIZE],
    [0, SIZE]
], dtype=np.float32)

M = cv2.getPerspectiveTransform(points, destination)
warped = cv2.warpPerspective(image, M, (SIZE, SIZE))


# Hough

gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

lines = cv2.HoughLinesP(
    edges,
    1,
    np.pi / 180,
    40,
    minLineLength=100,
    maxLineGap=30
)

horizontal = []
vertical = []

for x1, y1, x2, y2 in lines.reshape(-1, 4):

    angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

    if angle < 10 or angle > 170:
        horizontal.append((y1 + y2) / 2)

    elif 80 < angle < 100:
        vertical.append((x1 + x2) / 2)


# Merge duplicate detections

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

    return [np.mean(group) for group in groups]


horizontal = merge(horizontal)
vertical = merge(vertical)


# Fit 9 equally spaced grid lines

def reconstruct(detected):

    if len(detected) < 2:
        raise RuntimeError("Not enough lines")

    detected = np.array(sorted(detected))

    best_error = float("inf")
    best_grid = None

    for spacing in np.linspace(60, 140, 801):

        for origin in np.linspace(0, spacing, 50):

            grid = origin + np.arange(9) * spacing

            error = 0

            for line in detected:
                error += np.min(np.abs(grid - line))

            if error < best_error:
                best_error = error
                best_grid = grid

    return best_grid


horizontal = reconstruct(horizontal)
vertical = reconstruct(vertical)


print("Horizontal:", np.round(horizontal, 1))
print("Vertical:", np.round(vertical, 1))


# Draw

result = warped.copy()

for y in horizontal:
    cv2.line(
        result,
        (0, int(round(y))),
        (SIZE, int(round(y))),
        (0, 255, 0),
        2
    )

for x in vertical:
    cv2.line(
        result,
        (int(round(x)), 0),
        (int(round(x)), SIZE),
        (0, 255, 0),
        2
    )


cv2.imwrite(OUTPUT, result)

print("Saved:", OUTPUT)