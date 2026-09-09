import argparse
import os
import subprocess


WIDTH = 960
HEIGHT = 720
FPS = 30


def main():

    parser = argparse.ArgumentParser(
        description="Record the Chessora camera view."
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Recording duration in seconds.",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="photos/progress/camera_angle_20260910.mjpeg",
        help="Output MJPEG video path.",
    )

    args = parser.parse_args()

    os.makedirs(
        os.path.dirname(args.output) or ".",
        exist_ok=True,
    )

    command = [
        "rpicam-vid",
        "--width", str(WIDTH),
        "--height", str(HEIGHT),
        "--framerate", str(FPS),
        "--codec", "mjpeg",
        "--timeout", str(args.duration * 1000),
        "--nopreview",
        "--sharpness", "1.0",
        "--contrast", "1.0",
        "--brightness", "0.0",
        "--saturation", "1.0",
        "--vflip",
        "-o", args.output,
    ]

    print(
        f"Recording {WIDTH}x{HEIGHT} at {FPS} FPS for "
        f"{args.duration} seconds..."
    )

    subprocess.run(
        command,
        check=True,
    )

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()