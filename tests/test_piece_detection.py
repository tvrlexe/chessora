import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from piece_detection import (
    bottom_mask,
    classify_piece_predictions,
    get_piece_predictions,
    locate_piece_square,
    prediction_to_mask,
    square_for_cell,
)


class PieceDetectionTests(unittest.TestCase):
    def test_bottom_mask_keeps_only_lower_40_percent(self):
        mask = np.zeros((20, 10), dtype=np.uint8)
        mask[2:12, 3:7] = 1

        base = bottom_mask(mask)

        self.assertEqual(int(base.sum()), 16)
        self.assertEqual(int(base[:8].sum()), 0)
        self.assertEqual(int(base[8:].sum()), 16)

    def test_prediction_base_maps_to_square_using_existing_orientation(self):
        image_shape = (800, 800, 3)
        prediction = {
            "class": "chess piece",
            "confidence": 0.9,
            "points": [
                {"x": 220, "y": 210},
                {"x": 280, "y": 210},
                {"x": 280, "y": 275},
                {"x": 220, "y": 275},
            ],
        }

        detections = classify_piece_predictions(
            [prediction],
            image_shape,
            np.eye(3),
            "a8",
        )

        self.assertEqual(detections[0]["square"], "c3")
        self.assertEqual(square_for_cell(2, 2, "a8"), "c3")

    def test_piece_base_uses_reconstructed_grid_lines(self):
        prediction = {
            "class": "chess piece",
            "confidence": 0.9,
            "points": [
                {"x": 220, "y": 210},
                {"x": 280, "y": 210},
                {"x": 280, "y": 275},
                {"x": 220, "y": 275},
            ],
        }
        horizontal_lines = np.array(
            [0, 300, 400, 500, 600, 650, 700, 750, 800],
            dtype=np.float32,
        )
        vertical_lines = np.array(
            [0, 200, 300, 400, 500, 600, 650, 700, 800],
            dtype=np.float32,
        )

        detections = classify_piece_predictions(
            [prediction],
            (800, 800, 3),
            np.eye(3),
            "a8",
            horizontal_lines,
            vertical_lines,
        )

        self.assertEqual(detections[0]["square"], "b1")

    def test_piece_center_keeps_square_when_lower_mask_spills_over_grid_line(self):
        mask = np.zeros((800, 800), dtype=np.uint8)
        mask[400:520, 700:760] = 1

        square, base_coverage = locate_piece_square(
            mask,
            np.eye(3),
            "h8",
            np.arange(0, 801, 100),
            np.arange(0, 801, 100),
        )

        self.assertEqual(square, "e1")
        self.assertGreater(base_coverage, 0)

    def test_all_top_left_corner_orientations_map_correctly(self):
        self.assertEqual(square_for_cell(0, 0, "a8"), "a1")
        self.assertEqual(square_for_cell(0, 7, "a8"), "h1")
        self.assertEqual(square_for_cell(7, 0, "a8"), "a8")
        self.assertEqual(square_for_cell(7, 7, "a8"), "h8")

        self.assertEqual(square_for_cell(0, 0, "h8"), "a8")
        self.assertEqual(square_for_cell(0, 7, "h8"), "a1")
        self.assertEqual(square_for_cell(7, 0, "h8"), "h8")
        self.assertEqual(square_for_cell(7, 7, "h8"), "h1")

        self.assertEqual(square_for_cell(0, 0, "a1"), "h1")
        self.assertEqual(square_for_cell(0, 7, "a1"), "h8")
        self.assertEqual(square_for_cell(7, 0, "a1"), "a1")
        self.assertEqual(square_for_cell(7, 7, "a1"), "a8")

        self.assertEqual(square_for_cell(0, 0, "h1"), "h8")
        self.assertEqual(square_for_cell(0, 7, "h1"), "a8")
        self.assertEqual(square_for_cell(7, 0, "h1"), "h1")
        self.assertEqual(square_for_cell(7, 7, "h1"), "a1")

    def test_piece_prediction_parser_filters_non_piece_outputs(self):
        piece = {
            "class": "chess piece",
            "confidence": 0.9,
            "points": [{"x": 1, "y": 1}, {"x": 8, "y": 1}, {"x": 4, "y": 8}],
        }
        response = {
            "outputs": [
                {
                    "predictions": {
                        "predictions": [
                            piece,
                            {
                                "class": "chessboard",
                                "confidence": 0.99,
                                "points": [{"x": 0, "y": 0}],
                            },
                        ]
                    }
                }
            ]
        }

        with patch("piece_detection.run_workflow", return_value=response):
            predictions = get_piece_predictions(np.zeros((10, 10, 3)))

        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]["class"], piece["class"])
        self.assertEqual(predictions[0]["points"], piece["points"])

    def test_cropped_prediction_is_mapped_back_to_camera_coordinates(self):
        piece = {
            "class": "chess piece",
            "confidence": 0.9,
            "points": [
                {"x": 10, "y": 10},
                {"x": 30, "y": 10},
                {"x": 20, "y": 30},
            ],
        }
        response = {
            "outputs": [
                {
                    "predictions": [
                        {
                            "image": {"width": 60, "height": 40},
                            "predictions": [piece],
                        }
                    ],
                    "board_predictions": {"predictions": []},
                }
            ]
        }

        with (
            patch("piece_detection.run_workflow", return_value=response),
            patch(
                "piece_detection._board_crop_bounds",
                return_value=(20, 30, 60, 40),
            ),
        ):
            prediction = get_piece_predictions(np.zeros((100, 100, 3)))[0]

        mask = prediction_to_mask(prediction, (100, 100, 3))
        self.assertEqual(mask[40, 30], 1)
        self.assertEqual(mask[40, 50], 1)
        self.assertEqual(mask[30, 10], 0)


if __name__ == "__main__":
    unittest.main()
