import os
import base64
import requests
import cv2
import numpy as np
from dotenv import load_dotenv
from pycocotools import mask as mask_utils


load_dotenv("/home/chess/chessora/.env")

API_KEY = os.environ["ROBOFLOW_API_KEY"]

URL = (
    "https://serverless.roboflow.com/infer/workflows/"
    "blitz-noob/general-segmentation-api-9"
)


def order_corners(points):
    points = np.asarray(points, dtype=np.float32)

    coordinate_sum = points.sum(axis=1)
    coordinate_difference = points[:, 1] - points[:, 0]

    top_left = points[np.argmin(coordinate_sum)]
    bottom_right = points[np.argmax(coordinate_sum)]
    top_right = points[np.argmin(coordinate_difference)]
    bottom_left = points[np.argmax(coordinate_difference)]

    return np.array(
        [top_left, top_right, bottom_right, bottom_left],
        dtype=np.float32,
    )


def decode_rle_mask(rle_mask):
    """Decode Roboflow's compressed COCO RLE mask."""

    rle = {
        "size": rle_mask["size"],
        "counts": rle_mask["counts"],
    }

    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("ascii")

    mask = mask_utils.decode(rle)

    # Handle a possible H x W x 1 result.
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    return (mask.astype(np.uint8) * 255)


def get_board_points(image):
    if image is None:
        raise ValueError("Input image is None")

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    if not success:
        raise RuntimeError("Could not encode camera frame")

    image_base64 = base64.b64encode(encoded.tobytes()).decode("ascii")

    response = requests.post(
        URL,
        json={
            "inputs": {
                "image": {
                    "type": "base64",
                    "value": image_base64,
                },

                # This input still exists in the Workflow but no longer
                # controls SAM 3 because its prompts are hardcoded.
                "classes": "chessboard",
            }
        },
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        timeout=60,
    )

    if not response.ok:
        raise RuntimeError(
            f"Roboflow request failed with HTTP "
            f"{response.status_code}: {response.text}"
        )

    result = response.json()

    try:
        workflow_output = result["outputs"][0]
        predictions = workflow_output["board_predictions"]["predictions"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(
            f"Unexpected Workflow response structure: {result}"
        ) from error

    boards = [
        prediction
        for prediction in predictions
        if prediction.get("class", "").lower() == "chessboard"
    ]

    if not boards:
        raise RuntimeError("No chessboard detected")

    board = max(
        boards,
        key=lambda prediction: prediction.get("confidence", 0.0),
    )

    print(f"Board confidence: {board['confidence']:.3f}")

    rle_mask = board.get("rle_mask")

    if not rle_mask:
        raise RuntimeError(
            "Chessboard prediction did not contain an RLE mask"
        )

    mask = decode_rle_mask(rle_mask)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        raise RuntimeError("No board contour found")

    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)

    points = None

    # Start with a conventional approximation range.
    for epsilon_ratio in np.linspace(0.005, 0.10, 100):
        approximation = cv2.approxPolyDP(
            hull,
            epsilon_ratio * perimeter,
            True,
        )

        if len(approximation) == 4:
            points = approximation.reshape(4, 2)
            break

    # Fallback when the SAM mask does not simplify cleanly to four points.
    if points is None:
        rectangle = cv2.minAreaRect(hull)
        points = cv2.boxPoints(rectangle)

    points = order_corners(points)

    print("Board corners:")
    print(np.round(points).astype(int))

    return points
