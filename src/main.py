import subprocess
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, request

from board_detection import get_board_points
from perspective import get_perspective_matrix, warp_board
from grid_reconstruction import detect_segments, get_grid
from piece_detection import detect_pieces
from square_extraction import annotate_grid


app = Flask(__name__)


WIDTH = 960
HEIGHT = 720
FPS = 30
SIZE = 800


camera_process = None
frame = None
frame_lock = threading.Lock()

confirmed_frame = None
warped_board = None

horizontal_lines = None
vertical_lines = None
perspective_matrix = None

processing = False
board_ready = False
orientation_ready = False
orientation_square = None
piece_positions = {}
piece_missing_counts = {}
piece_lock = threading.Lock()

PIECE_UPDATE_INTERVAL = 1.0
PIECE_CROP_MARGIN = 40
PIECE_MISSING_GRACE = 2


def start_camera():

    global camera_process

    camera_process = subprocess.Popen(
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
            "-o", "-"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0
    )


def read_camera():

    global frame

    buffer = b""

    while True:

        chunk = camera_process.stdout.read(4096)

        if not chunk:
            break

        buffer += chunk

        while True:

            start = buffer.find(b"\xff\xd8")

            if start == -1:
                break

            end = buffer.find(
                b"\xff\xd9",
                start + 2
            )

            if end == -1:
                break

            jpg = buffer[start:end + 2]
            buffer = buffer[end + 2:]

            image = cv2.imdecode(
                np.frombuffer(
                    jpg,
                    dtype=np.uint8
                ),
                cv2.IMREAD_COLOR
            )

            if image is not None:

                with frame_lock:
                    frame = image


def draw_grid(
    image,
    horizontal,
    vertical
):

    output = image.copy()

    for y in horizontal:

        cv2.line(
            output,
            (0, int(y)),
            (SIZE - 1, int(y)),
            (0, 255, 0),
            2
        )

    for x in vertical:

        cv2.line(
            output,
            (int(x), 0),
            (int(x), SIZE - 1),
            (255, 0, 0),
            2
        )

    return output


def grid_to_wide(
    matrix,
    horizontal,
    vertical
):

    inverse_matrix = np.linalg.inv(matrix)

    wide_horizontal = []
    wide_vertical = []

    for y in horizontal:

        points = np.array(
            [
                [0, y],
                [SIZE - 1, y]
            ],
            dtype=np.float32
        ).reshape(-1, 1, 2)

        transformed = cv2.perspectiveTransform(
            points,
            inverse_matrix
        ).reshape(-1, 2)

        wide_horizontal.append(
            (
                transformed[0],
                transformed[1]
            )
        )

    for x in vertical:

        points = np.array(
            [
                [x, 0],
                [x, SIZE - 1]
            ],
            dtype=np.float32
        ).reshape(-1, 1, 2)

        transformed = cv2.perspectiveTransform(
            points,
            inverse_matrix
        ).reshape(-1, 2)

        wide_vertical.append(
            (
                transformed[0],
                transformed[1]
            )
        )

    return wide_horizontal, wide_vertical


def get_wide_grid():

    if perspective_matrix is None:
        return None, None

    if horizontal_lines is None:
        return None, None

    if vertical_lines is None:
        return None, None

    return grid_to_wide(
        perspective_matrix,
        horizontal_lines,
        vertical_lines
    )


def piece_detection_loop():

    global piece_positions
    global piece_missing_counts

    while True:

        if (
            not board_ready
            or not orientation_ready
            or perspective_matrix is None
            or horizontal_lines is None
            or vertical_lines is None
        ):
            time.sleep(0.1)
            continue

        with frame_lock:
            current_frame = None if frame is None else frame.copy()

        if current_frame is None:
            time.sleep(0.1)
            continue

        try:

            warped_board = warp_board(
                current_frame,
                perspective_matrix,
                SIZE
            )

            margin = PIECE_CROP_MARGIN
            cropped_board = warped_board[
                margin:SIZE - margin,
                margin:SIZE - margin
            ]
            current_board = cv2.resize(
                cropped_board,
                (SIZE, SIZE),
                interpolation=cv2.INTER_LINEAR
            )
            scale = SIZE / (SIZE - 2 * margin)
            piece_horizontal = (horizontal_lines - margin) * scale
            piece_vertical = (vertical_lines - margin) * scale

            detected = detect_pieces(
                current_board,
                piece_horizontal,
                piece_vertical,
                orientation_square
            )

            with piece_lock:
                updated_positions = dict(detected)
                updated_missing = {}

                for square, previous in piece_positions.items():

                    if square in detected:
                        continue

                    misses = piece_missing_counts.get(square, 0) + 1

                    if misses < PIECE_MISSING_GRACE:
                        updated_positions[square] = previous
                        updated_missing[square] = misses

                for square in detected:
                    updated_missing[square] = 0

                piece_positions = updated_positions
                piece_missing_counts = updated_missing

                tracked_count = len(piece_positions)

            print(
                f"Pieces detected: {len(detected)}; "
                f"pieces tracked: {tracked_count}"
            )

        except Exception as error:

            print(
                f"Piece detection failed: {error}"
            )

        time.sleep(PIECE_UPDATE_INTERVAL)


def process_board():

    global confirmed_frame
    global warped_board
    global horizontal_lines
    global vertical_lines
    global perspective_matrix
    global processing
    global board_ready
    global orientation_ready
    global orientation_square

    processing = True

    try:

        with frame_lock:

            if frame is None:
                print("No camera frame available.")
                return

            confirmed_frame = frame.copy()

        print("Detecting board...")

        segments = detect_segments(
            confirmed_frame
        )

        print(
            f"Hough segments: {len(segments)}"
        )

        points = get_board_points(
            confirmed_frame
        )

        print("Board detected.")

        perspective_matrix = get_perspective_matrix(
            points,
            SIZE
        )

        print("Cropping board zone...")

        warped = warp_board(
            confirmed_frame,
            perspective_matrix,
            SIZE
        )

        print("Reconstructing grid...")

        horizontal_lines, vertical_lines = get_grid(
            confirmed_frame,
            points,
            segments
        )

        print("Grid reconstructed.")

        warped_board = draw_grid(
            warped,
            horizontal_lines,
            vertical_lines
        )

        board_ready = True
        orientation_ready = False
        orientation_square = None

    except Exception as error:

        print(
            f"Board processing failed: {error}"
        )

        board_ready = False

    finally:

        processing = False


def get_live_annotated_frame():

    if not orientation_ready:
        return None

    if orientation_square is None:
        return None

    with frame_lock:

        if frame is None:
            return None

        current_frame = frame.copy()

    wide_horizontal, wide_vertical = get_wide_grid()

    if wide_horizontal is None:
        return None

    output = annotate_grid(
        current_frame,
        wide_horizontal,
        wide_vertical,
        orientation_square
    )

    inverse_matrix = np.linalg.inv(perspective_matrix)

    with piece_lock:
        current_pieces = dict(piece_positions)

    for square, piece in current_pieces.items():

        crop_scale = SIZE / (SIZE - 2 * PIECE_CROP_MARGIN)
        crop_x, crop_y = piece["anchor"]
        board_anchor = (
            crop_x / crop_scale + PIECE_CROP_MARGIN,
            crop_y / crop_scale + PIECE_CROP_MARGIN,
        )
        anchor = np.array(
            [[board_anchor]],
            dtype=np.float32
        )

        camera_anchor = cv2.perspectiveTransform(
            anchor,
            inverse_matrix
        )[0, 0]

        label = f"{square}: {piece['class']}"

        cv2.putText(
            output,
            label,
            tuple(np.round(camera_anchor).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 255),
            1,
            cv2.LINE_AA
        )

    return output


@app.route("/")
def index():

    return """
<!DOCTYPE html>

<html>

<head>

<title>Chessora</title>

<style>

body {
    font-family: Arial, sans-serif;
    text-align: center;
    margin: 20px;
}

img {
    max-width: 960px;
    width: 100%;
    height: auto;
}

button {
    font-size: 22px;
    padding: 14px 25px;
    margin: 8px;
    cursor: pointer;
}

#cornerButtons {
    display: none;
}

#repositionButton {
    display: none;
}

#message {
    font-size: 26px;
    margin: 20px;
}

</style>

</head>

<body>

<h1>Chessora</h1>

<div id="message">
    Position the board
</div>

<div id="view">
    <img id="camera" src="/video">
</div>

<br>

<button
    id="yesButton"
    onclick="confirmBoard()"
>
    YES
</button>

<div id="cornerButtons">

    <div>

        <button onclick="selectCorner('top-left')">
            TOP LEFT
        </button>

        <button onclick="selectCorner('top-right')">
            TOP RIGHT
        </button>

    </div>

    <div>

        <button onclick="selectCorner('bottom-left')">
            BOTTOM LEFT
        </button>

        <button onclick="selectCorner('bottom-right')">
            BOTTOM RIGHT
        </button>

    </div>

</div>

<button
    id="repositionButton"
    onclick="repositionBoard()"
>
    REPOSITION BOARD
</button>


<script>

function confirmBoard() {

    const button =
        document.getElementById("yesButton");

    button.disabled = true;
    button.innerText = "PROCESSING...";

    document.getElementById("message").innerText =
        "Detecting board and reconstructing grid...";

    fetch("/confirm", {
        method: "POST"
    });

    checkStatus();
}


function checkStatus() {

    fetch("/status")
        .then(response => response.json())
        .then(data => {

            if (data.orientation) {

                document.getElementById("view").innerHTML =
                    '<img src="/board?t=' + Date.now() + '">';

                document.getElementById("message").innerText =
                    "Which corner is a1?";

                document.getElementById("yesButton")
                    .style.display = "none";

                document.getElementById("cornerButtons")
                    .style.display = "block";

                document.getElementById("repositionButton")
                    .style.display = "inline-block";

                return;
            }

            if (data.processing) {

                setTimeout(checkStatus, 300);

                return;
            }

            if (!data.ready) {

                document.getElementById("yesButton")
                    .disabled = false;

                document.getElementById("yesButton")
                    .innerText = "YES";

                return;
            }

            setTimeout(checkStatus, 300);
        });
}


function selectCorner(corner) {

    document.getElementById("message").innerText =
        "Annotating board...";

    document.getElementById("cornerButtons")
        .style.display = "none";

    fetch("/orientation", {

        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify({
            corner: corner
        })

    })
    .then(response => response.json())
    .then(data => {

        if (data.status !== "annotated") {

            document.getElementById("message").innerText =
                "Orientation failed";

            return;
        }

        document.getElementById("view").innerHTML =
            '<img src="/annotated?t=' + Date.now() + '">';

        document.getElementById("message").innerText =
            "Board annotated";

    });
}


function repositionBoard() {

    fetch("/reposition", {
        method: "POST"
    })
    .then(() => {

        document.getElementById("view").innerHTML =
            '<img id="camera" src="/video?t=' +
            Date.now() +
            '">';

        document.getElementById("message").innerText =
            "Position the board";

        document.getElementById("yesButton")
            .style.display = "inline-block";

        document.getElementById("yesButton")
            .disabled = false;

        document.getElementById("yesButton")
            .innerText = "YES";

        document.getElementById("cornerButtons")
            .style.display = "none";

        document.getElementById("repositionButton")
            .style.display = "none";

    });
}

</script>

</body>

</html>
"""


@app.route("/video")
def video():

    def generate():

        while not board_ready:

            with frame_lock:

                if frame is None:
                    continue

                output = frame.copy()

            success, encoded = cv2.imencode(
                ".jpg",
                output
            )

            if success:

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + encoded.tobytes()
                    + b"\r\n"
                )

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/board")
def board():

    def generate():

        while board_ready and not orientation_ready:

            with frame_lock:

                if frame is None:
                    continue

                current_frame = frame.copy()

            warped = warp_board(
                current_frame,
                perspective_matrix,
                SIZE
            )

            output = draw_grid(
                warped,
                horizontal_lines,
                vertical_lines
            )

            success, encoded = cv2.imencode(
                ".jpg",
                output
            )

            if success:

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + encoded.tobytes()
                    + b"\r\n"
                )

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/annotated")
def annotated():

    def generate():

        while orientation_ready:

            output = get_live_annotated_frame()

            if output is None:
                continue

            success, encoded = cv2.imencode(
                ".jpg",
                output
            )

            if success:

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + encoded.tobytes()
                    + b"\r\n"
                )

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/confirm", methods=["POST"])
def confirm():

    if not processing and not board_ready:

        threading.Thread(
            target=process_board,
            daemon=True
        ).start()

    return {
        "status": "processing"
    }


@app.route("/orientation", methods=["POST"])
def orientation():

    global orientation_square
    global orientation_ready

    data = request.get_json()

    corner = data["corner"]

    layouts = {
        "top-left": "a8",
        "top-right": "h8",
        "bottom-left": "a1",
        "bottom-right": "h1",
    }

    square = layouts.get(corner)

    if square is None:

        return {
            "status": "error",
            "error": "Invalid corner selection"
        }, 400

    orientation_square = square
    orientation_ready = True

    return {
        "status": "annotated"
    }


@app.route("/reposition", methods=["POST"])
def reposition():

    global board_ready
    global orientation_ready
    global warped_board
    global confirmed_frame
    global horizontal_lines
    global vertical_lines
    global perspective_matrix
    global orientation_square
    global piece_positions
    global piece_missing_counts

    board_ready = False
    orientation_ready = False

    warped_board = None
    confirmed_frame = None

    horizontal_lines = None
    vertical_lines = None
    perspective_matrix = None

    orientation_square = None

    with piece_lock:
        piece_positions = {}
        piece_missing_counts = {}

    return {
        "status": "repositioning"
    }


@app.route("/status")
def status():

    with piece_lock:
        piece_count = len(piece_positions)

    return {
        "ready": board_ready,
        "processing": processing,
        "pieces": piece_count,
        "orientation": (
            board_ready
            and not orientation_ready
        )
    }


if __name__ == "__main__":

    start_camera()

    threading.Thread(
        target=read_camera,
        daemon=True
    ).start()

    threading.Thread(
        target=piece_detection_loop,
        daemon=True
    ).start()

    print("Camera started.")
    print("Open http://chessora-pi:5000")

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True
    )