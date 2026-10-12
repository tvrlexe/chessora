import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import main


class CameraCaptureTests(unittest.TestCase):
    def test_still_capture_uses_take_photo_camera_settings(self):
        expected = np.full((720, 960, 3), 120, dtype=np.uint8)
        success, encoded = cv2.imencode(".jpg", expected)
        self.assertTrue(success)

        with patch(
            "main.subprocess.run",
            return_value=SimpleNamespace(stdout=encoded.tobytes()),
        ) as run:
            image = main.capture_still()

        command = run.call_args.args[0]
        self.assertEqual(command[0], "rpicam-still")
        self.assertIn("--width", command)
        self.assertIn("960", command)
        self.assertIn("--height", command)
        self.assertIn("720", command)
        self.assertEqual(command[command.index("--roi") + 1], "0.28,0.26,0.44,0.48")
        self.assertEqual(command[command.index("--sharpness") + 1], "1.0")
        self.assertEqual(image.shape, expected.shape)


if __name__ == "__main__":
    unittest.main()
