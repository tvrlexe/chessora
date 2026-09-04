import subprocess
import cv2
import numpy as np


WIDTH = 960
HEIGHT = 720
FPS = 30
ROI = "0.28,0.26,0.44,0.48"


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
        bufsize=0
    )


def read_frame(process):

    buffer = b""

    while True:

        buffer += process.stdout.read(4096)

        start = buffer.find(b"\xff\xd8")
        end = buffer.find(b"\xff\xd9")

        if start != -1 and end != -1 and end > start:

            jpg = buffer[start:end + 2]
            buffer = buffer[end + 2:]

            frame = cv2.imdecode(
                np.frombuffer(jpg, np.uint8),
                cv2.IMREAD_COLOR
            )

            return frame