import base64
import os

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from square_extraction import LAYOUTS


load_dotenv("/home/chess/chessora/.env")

API_KEY = os.environ["ROBOFLOW_API_KEY"]
WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "blitz-noob")
WORKFLOW = os.getenv(
    "ROBOFLOW_PIECE_WORKFLOW",
    "general-segmentation-api-9",
)
API_URL = (
    "https://serverless.roboflow.com/"
    f"{WORKSPACE}/workflows/{WORKFLOW}"
)

BOARD_SIZE = 800
BOTTOM_ANCHOR_RATIO = 0.30
ORIENTATIONS = {
    "top-left": "a8",
    "top-right": "h8",
    "bottom-left": "a1",
    "bottom-right": "h1",
    "a8": "a8",
    "h8": "h8",
    "a1": "a1",
    "h1": "h1",
}


def _encode_image(image):
    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    if not success:
        raise RuntimeError("Could not encode board image")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def run_piece_workflow(image, use_cache=True):
    if image is None:
        raise ValueError("Image is None")

    response = requests.post(
        API_URL,
        json={
            "inputs": {
                "image": {
                    "type": "base64",
                    "value": _encode_image(image),
                },
                "classes": "piece",
            }
        },
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        params={"use_cache": str(use_cache).lower()},
        timeout=60,
    )

    if not response.ok:
        raise RuntimeError(
            f"Piece workflow failed with HTTP {response.status_code}: "
            f"{response.text}"
        )

    return response.json()


def _prediction_list(value):
    if isinstance(value, list):
        if value and all(isinstance(item, dict) for item in value):
            if any(
                key in value[0]
                for key in ("class", "label", "x", "bbox", "bounding_box")
            ):
                return value
        for item in value:
            found = _prediction_list(item)
            if found:
                return found

    if isinstance(value, dict):
        for key in ("predictions", "pieces", "detections", "objects"):
            if key in value:
                found = _prediction_list(value[key])
                if found:
                    return found
        for item in value.values():
            found = _prediction_list(item)
            if found:
                return found

    return []


def _box(prediction):
    if all(key in prediction for key in ("x", "y", "width", "height")):
        x = float(prediction["x"])
        y = float(prediction["y"])
        width = float(prediction["width"])
        height = float(prediction["height"])
        return x - width / 2, y - height / 2, x + width / 2, y + height / 2

    if "bbox" in prediction:
        values = prediction["bbox"]
    elif "bounding_box" in prediction:
        values = prediction["bounding_box"]
    else:
        raise ValueError("Prediction has no supported bounding box")

    if isinstance(values, dict):
        left = values.get("x", values.get("left"))
        top = values.get("y", values.get("top"))
        width = values.get("width")
        height = values.get("height")
        if None not in (left, top, width, height):
            return (
                float(left),
                float(top),
                float(left) + float(width),
                float(top) + float(height),
            )

    if len(values) != 4:
        raise ValueError("Unsupported bounding box format")

    left, top, right, bottom = map(float, values)
    return left, top, right, bottom


def bottom_anchor(prediction):
    left, top, right, bottom = _box(prediction)
    height = bottom - top
    return (
        float((left + right) / 2),
        float(bottom - BOTTOM_ANCHOR_RATIO * height),
    )


def square_for_anchor(anchor, horizontal_lines, vertical_lines, orientation="h1"):
    x, y = anchor
    horizontal = np.asarray(horizontal_lines, dtype=np.float32)
    vertical = np.asarray(vertical_lines, dtype=np.float32)

    column = int(np.searchsorted(vertical, x, side="right") - 1)
    row = int(np.searchsorted(horizontal, y, side="right") - 1)

    if not 0 <= row < 8 or not 0 <= column < 8:
        return None

    if orientation not in LAYOUTS:
        raise ValueError("Orientation must be a corner name or a1/a8/h1/h8")

    return LAYOUTS[orientation][row][column]


def detect_pieces(
    image,
    horizontal_lines,
    vertical_lines,
    orientation="h1",
    min_confidence=0.0,
):
    result = run_piece_workflow(image)
    predictions = _prediction_list(result)
    print(
        f"Model predictions: {len(predictions)}"
    )
    pieces = {}

    for prediction in predictions:
        confidence = float(
            prediction.get(
                "confidence",
                prediction.get("score", 0.0),
            )
        )
        if confidence < min_confidence:
            continue

        try:
            anchor = bottom_anchor(prediction)
        except (KeyError, TypeError, ValueError):
            continue

        square = square_for_anchor(
            anchor,
            horizontal_lines,
            vertical_lines,
            orientation,
        )
        if square is None:
            continue

        piece = {
            "class": prediction.get(
                "class",
                prediction.get("label", "unknown"),
            ),
            "confidence": confidence,
            "anchor": anchor,
            "box": _box(prediction),
        }

        current = pieces.get(square)
        if current is None or piece["confidence"] > current["confidence"]:
            pieces[square] = piece

    return pieces
