import os
import base64
import requests
import cv2
import numpy as np
from dotenv import load_dotenv
from pycocotools import mask as mask_utils


load_dotenv("/home/chess/chessora/.env")

API_KEY = os.environ["ROBOFLOW_API_KEY"]
URL = "https://serverless.roboflow.com/blitz-noob/workflows/general-segmentation-api-9"


def order_corners(points):
    points = np.asarray(points, dtype=np.float32)

    # Sum: smallest = top-left, largest = bottom-right
    s = points.sum(axis=1)

    # Difference y - x:
    # smallest = top-right, largest = bottom-left
    d = points[:, 1] - points[:, 0]

    top_left = points[np.argmin(s)]
    bottom_right = points[np.argmax(s)]
    top_right = points[np.argmin(d)]
    bottom_left = points[np.argmax(d)]

    return np.array([
        top_left,
        top_right,
        bottom_right,
        bottom_left
    ], dtype=np.float32)


def get_board_points(image_path):

    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()

    r = requests.post(
        URL,
        json={
            "inputs": {
                "image": {
                    "type": "base64",
                    "value": data
                },
                "classes": "Chess-Board"
            }
        },
        headers={
            "Authorization": f"Bearer {API_KEY}"
        }
    )

    if r.status_code != 200:
        raise RuntimeError(r.text)

    predictions = r.json()["outputs"][0]["predictions"]["predictions"]

    boards = [
        p for p in predictions
        if p["class"] == "Chess-Board"
    ]

    if not boards:
        raise RuntimeError("No Chess-Board detected")

    board = max(
        boards,
        key=lambda p: p["confidence"]
    )

    print("Board confidence:", board["confidence"])

    # --------------------------------------------------
    # 1. Decode segmentation mask
    # --------------------------------------------------

    mask = mask_utils.decode(
        board["rle_mask"]
    ).astype(np.uint8) * 255

    cv2.imwrite(
        "/home/chess/chessora/debug_mask.png",
        mask
    )

    # --------------------------------------------------
    # 2. Find board contour
    # --------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise RuntimeError("No board contour found")

    contour = max(
        contours,
        key=cv2.contourArea
    )

    # --------------------------------------------------
    # 3. Remove concavities / irregularities
    # --------------------------------------------------

    hull = cv2.convexHull(contour)

    # --------------------------------------------------
    # 4. Find a 4-point polygon
    # --------------------------------------------------

    perimeter = cv2.arcLength(
        hull,
        True
    )

    points = None

    for epsilon_ratio in np.linspace(0.001, 0.20, 500):

        epsilon = epsilon_ratio * perimeter

        approx = cv2.approxPolyDP(
            hull,
            epsilon,
            True
        )

        if len(approx) == 4:
            points = approx.reshape(4, 2)
            break

    if points is None:
        raise RuntimeError(
            "Could not find 4 board corners"
        )

    # --------------------------------------------------
    # 5. Order corners
    # --------------------------------------------------

    points = order_corners(points)

    print("Board corners:")
    print(points.astype(int))

    # --------------------------------------------------
    # 6. Debug visualization
    # --------------------------------------------------

    image = cv2.imread(image_path)

    for i, (x, y) in enumerate(points):

        cv2.circle(
            image,
            (int(x), int(y)),
            8,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            image,
            str(i),
            (int(x) + 10, int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    polygon = points.astype(np.int32)

    cv2.polylines(
        image,
        [polygon],
        True,
        (255, 0, 0),
        3
    )

    cv2.imwrite(
        "/home/chess/chessora/debug_board.png",
        image
    )

    return points