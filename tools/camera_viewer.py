import subprocess
import cv2
import numpy as np
from flask import Flask, Response

app = Flask(__name__)

WIDTH = 960
HEIGHT = 720
FPS = 30

ROI = "0.28,0.26,0.44,0.48"


def generate_frames():

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

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    buffer = b""

    while True:

        chunk = process.stdout.read(4096)

        if not chunk:
            break

        buffer += chunk

        start = buffer.find(b"\xff\xd8")
        end = buffer.find(b"\xff\xd9")

        if start != -1 and end != -1:

            frame = buffer[start:end + 2]

            buffer = buffer[end + 2:]

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )


@app.route("/")
def index():

    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chessora Camera</title>
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

        <h1>Chessora Camera</h1>

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