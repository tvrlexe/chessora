import cv2
import numpy as np


def calculate_difference(current, reference):

    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    difference = cv2.absdiff(
        current_gray,
        reference_gray
    )

    return np.mean(difference)


def calibrate(squares):

    print("Calibrating empty board...")

    references = {}

    for name, square in squares.items():
        references[name] = square.copy()

    print("Calibration complete.")

    return references


def detect_presence(squares, references, threshold):

    presence = {}

    for name, square in squares.items():

        difference = calculate_difference(
            square,
            references[name]
        )

        presence[name] = difference > threshold

    occupied = sum(presence.values())

    if occupied > 32:
        raise RuntimeError(
            f"Invalid detection: {occupied} occupied squares. "
            "A chessboard cannot have more than 32 pieces."
        )

    return presence