import subprocess
import threading

import cv2
import numpy as np
from flask import Flask, Response

from board_detection import get_board_points
from perspective import get_perspective_matrix, warp_board
from grid_reconstruction import get_grid
from square_extraction import LAYOUTS


app = Flask(__name__)

WIDTH = 960
HEIGHT = 720
FPS = 30
SIZE = 800

frame = None
mode = "camera"

matrix = None
horizontal = None
vertical = None
square = None

lock = threading.Lock()


def start_camera():
    return subprocess.Popen(
        [
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
            "--roi", "0.28,0.26,0.44,0.48",
            "-o", "-"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )


def camera_loop(camera):
    global frame

    buffer = b""

    while True:
        buffer += camera.stdout.read(4096)

        start = buffer.find(b"\xff\xd8")
        end = buffer.find(b"\xff\xd9", start + 2)

        if start == -1 or end == -1:
            continue

        jpg = buffer[start:end + 2]
        buffer = buffer[end + 2:]

        image = cv2.imdecode(
            np.frombuffer(jpg, np.uint8),
            cv2.IMREAD_COLOR
        )

        if image is not None:
            with lock:
                frame = image


def annotate_board(board):
    result = board.copy()
    layout = LAYOUTS[square]

    for y in horizontal:
        cv2.line(
            result,
            (0, int(y)),
            (SIZE, int(y)),
            (0, 255, 0),
            2
        )

    for x in vertical:
        cv2.line(
            result,
            (int(x), 0),
            (int(x), SIZE),
            (0, 255, 0),
            2
        )

    for row in range(8):
        for col in range(8):

            x = int(
                (vertical[col] + vertical[col + 1]) / 2
            )

            y = int(
                (horizontal[row] + horizontal[row + 1]) / 2
            )

            cv2.putText(
                result,
                layout[row][col],
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

    return result


def process_frame(image):

    if mode == "camera":
        return image

    board = warp_board(
        image,
        matrix,
        SIZE
    )

    if mode == "warped":
        return board

    if mode == "annotated":
        return annotate_board(board)

    return image


def generate_frames():

    while True:

        with lock:
            if frame is None:
                continue

            image = frame.copy()

        result = process_frame(image)

        ok, encoded = cv2.imencode(
            ".jpg",
            result
        )

        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + encoded.tobytes()
            + b"\r\n"
        )


@app.route("/")
def index():
    return """
    <html>
    <body style="
        margin:0;
        background:#111;
        display:flex;
        justify-content:center;
        align-items:center;
        height:100vh;
    ">
        <img src="/video" style="
            max-width:100%;
            max-height:100vh;
        ">
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

    camera = start_camera()

    threading.Thread(
        target=camera_loop,
        args=(camera,),
        daemon=True
    ).start()

    threading.Thread(
        target=lambda: app.run(
            host="0.0.0.0",
            port=5000,
            threaded=True,
            use_reloader=False
        ),
        daemon=True
    ).start()

    try:

        print()
        print("Chessora")
        print()
        print("Open: http://10.37.33.89:5000/")
        print()

        print("Is the board positioned correctly? (y/n):")

        while True:

            answer = input().strip().lower()

            if answer == "y":
                break

            if answer == "n":
                print("Adjust the board.")
                print("Is the board positioned correctly? (y/n):")
                continue

            print("Please enter y or n.")

        with lock:
            confirmed = frame.copy()

        print()
        print("Detecting board...")

        points = get_board_points(confirmed)

        print("Board points:")
        print(points.astype(int))

        matrix = get_perspective_matrix(
            points,
            SIZE
        )

        with lock:
            mode = "warped"

        print()
        print("Reconstructing grid...")

        horizontal, vertical = get_grid(
            warp_board(
                confirmed,
                matrix,
                SIZE
            )
        )

        print("Horizontal:", np.round(horizontal, 1))
        print("Vertical:", np.round(vertical, 1))

        print()
        print("What square is at the bottom-left corner?")
        print("(a1/a8/h1/h8):")

        while True:

            answer = input().strip().lower()

            if answer in LAYOUTS:
                square = answer
                break

            print("Please enter a1, a8, h1, or h8.")

        with lock:
            mode = "annotated"

        print()
        print("Chessora is ready.")

        while True:
            threading.Event().wait(1)

    except KeyboardInterrupt:

        print("\nStopping Chessora...")

    finally:

        camera.terminate()
        camera.wait()