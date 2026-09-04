import subprocess
import cv2
import numpy as np
from flask import Flask, Response

from board_detection import get_board_points
from perspective import get_perspective_matrix, warp_board
from grid_reconstruction import get_grid


app = Flask(__name__)


WIDTH = 960
HEIGHT = 720
FPS = 30

ROI = "0.28,0.26,0.44,0.48"

SIZE = 800


def start_camera():

    command = [
        "rpicam-vid",
        "--width", str(WIDTH),
        "--height", str(HEIGHT),
        "--framerate", str(FPS),
        "--codec", "mjpeg",
        "--timeout", "0",
        "--nopreview",
        "--sharpness", "1.0",
        "--contrast", "1.0",
        "--brightness", "0.0",
        "--saturation", "1.0",
        "--vflip",
        "--roi", ROI,
        "-o", "-"
    ]

    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )


def read_frame(process, buffer):

    while True:

        chunk = process.stdout.read(4096)

        if not chunk:
            return None, buffer

        buffer += chunk

        start = buffer.find(b"\xff\xd8")
        end = buffer.find(b"\xff\xd9")

        if start != -1 and end != -1 and end > start:

            jpg = buffer[start:end + 2]

            buffer = buffer[end + 2:]

            frame = cv2.imdecode(
                np.frombuffer(jpg, np.uint8),
                cv2.IMREAD_COLOR
            )

            return frame, buffer


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


def draw_grid(image, horizontal, vertical):

    output = image.copy()

    for points in horizontal:

        points = np.round(points).astype(int)

        cv2.line(
            output,
            tuple(points[0]),
            tuple(points[1]),
            (0, 255, 0),
            2
        )

    for points in vertical:

        points = np.round(points).astype(int)

        cv2.line(
            output,
            tuple(points[0]),
            tuple(points[1]),
            (0, 255, 0),
            2
        )

    return output


def generate_frames():

    process = start_camera()
    buffer = b""

    print("Waiting for camera frame...")

    frame, buffer = read_frame(
        process,
        buffer
    )

    if frame is None:

        process.terminate()
        return

    print("Detecting board...")

    points = get_board_points(frame)

    print("Calculating perspective...")

    M = get_perspective_matrix(
        points,
        SIZE
    )

    print("Warping board...")

    warped = warp_board(
        frame,
        M,
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
        M
    )

    print("Grid detected")
    print("Starting live pipeline...")

    while True:

        frame, buffer = read_frame(
            process,
            buffer
        )

        if frame is None:
            break

        output = draw_grid(
            frame,
            camera_horizontal,
            camera_vertical
        )

        success, encoded = cv2.imencode(
            ".jpg",
            output
        )

        if not success:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + encoded.tobytes()
            + b"\r\n"
        )

    process.terminate()


@app.route("/")
def index():

    return """
    <!DOCTYPE html>
    <html>

    <head>

        <title>Chessora Live Pipeline</title>

        <style>

            body {
                background: #111;
                color: white;
                text-align: center;
                font-family: Arial;
            }

            img {
                width: 960px;
                max-width: 95%;
                height: auto;
            }

        </style>

    </head>

    <body>

        <h1>Chessora Live Pipeline</h1>

        <img src="/video">

    </body>

    </html>
    """


@app.route("/video")
def video():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True
    )