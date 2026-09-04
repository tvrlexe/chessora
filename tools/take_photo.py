import subprocess
import sys
import os


WIDTH = 960
HEIGHT = 720
ROI = "0.28,0.26,0.44,0.48"


filename = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "photo.jpg"
)

OUTPUT = os.path.join("photos", filename)


command = [
    "rpicam-still",

    "--width", str(WIDTH),
    "--height", str(HEIGHT),

    "--timeout", "2000",
    "--nopreview",

    "--sharpness", "1.0",
    "--contrast", "1.0",
    "--brightness", "0.0",
    "--saturation", "1.0",

    "--vflip",

    "--roi", ROI,

    "-o", OUTPUT
]


print("Taking photo...")

subprocess.run(
    command,
    check=True
)

print(f"Saved: {OUTPUT}")