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

    s = points.sum(axis=1)

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


def get_board_points(image):

    success, encoded = cv2.imencode(".jpg", image)

    if not success:
        raise RuntimeError("Could not encode camera frame")

    data = base64.b64encode(
        encoded.tobytes()
    ).decode()

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

    mask = mask_utils.decode(
        board["rle_mask"]
    ).astype(np.uint8) * 255

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

    hull = cv2.convexHull(contour)

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

    points = order_corners(points)

    print("Board corners:")
    print(points.astype(int))

    return points