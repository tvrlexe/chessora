# Chessora

A real-time chess board detection and analysis system that uses computer vision and deep learning to automatically detect chess boards from camera feeds, reconstruct the game grid, and annotate chess squares with standard algebraic notation.

## Overview

Chessora is a Raspberry Pi-based application that combines cutting-edge computer vision techniques with machine learning to:

1. **Detect chess boards** using Roboflow's AI vision API (SAM 3 - Segment Anything Model)
2. **Reconstruct the grid** using Hough line detection and advanced line fitting algorithms
3. **Provide real-time annotation** with chess square coordinates (a1-h8)
4. **Stream live video** through a web-based interface for setup and configuration

## Key Features

- 🎯 **Accurate Board Detection**: Uses Roboflow's Segment Anything Model 3 (SAM 3) for robust chessboard segmentation
- 📐 **Grid Reconstruction**: Intelligent line detection and fitting to reconstruct perfect 8x8 grids from imperfect camera angles
- 🔄 **Perspective Correction**: Applies perspective transforms to normalize viewing angles
- 🌐 **Web Interface**: Clean, responsive web UI for board positioning and orientation setup
- ⚡ **Real-time Processing**: Multi-threaded architecture for smooth streaming and processing
- 🎮 **Interactive Setup**: User-friendly workflow for board positioning and square orientation

## System Architecture

### Core Components

```
src/
├── main.py                 # Flask web server & main application logic
├── board_detection.py      # Roboflow integration for board segmentation
├── grid_reconstruction.py  # Hough line detection & grid fitting
├── perspective.py          # Perspective transform utilities
├── square_extraction.py    # Square annotation and labeling
├── camera.py              # Camera capture utilities
├── live_pipeline.py       # Live processing pipeline
├── presence_detection.py   # Chess piece presence detection
├── tracker.py             # Piece tracking (placeholder)
├── detector.py            # General detection utilities (placeholder)
└── board_state.py         # Board state management (placeholder)
```

### Pipeline Workflow

1. **Camera Capture**: Streams video from Raspberry Pi camera via `rpicam-vid`
2. **Board Detection**: 
   - Encodes frame to JPEG and sends to Roboflow API
   - Receives RLE-encoded segmentation mask
   - Extracts board corners using contour analysis
   - Orders corners consistently (top-left, top-right, bottom-right, bottom-left)

3. **Perspective Normalization**: Applies perspective matrix to normalize the board to a standard 800×800 view

4. **Grid Reconstruction**:
   - Detects line segments using Hough line transform
   - Classifies lines as horizontal or vertical based on angle
   - Merges nearby parallel lines
   - Finds valid 9-line sequences (8 squares + borders) using spatial consistency
   - Fits an evenly-spaced grid to detected lines

5. **Orientation Setup**: User selects which corner is square a1, establishing chess notation mapping

6. **Live Annotation**: Streams annotated video with square coordinates overlaid on camera feed

## Technical Details

### Grid Reconstruction Algorithm

The grid reconstruction uses a sophisticated line-fitting approach:

- **Detection**: Canny edge detection + Hough line transform
- **Merging**: Groups nearby parallel lines within a threshold
- **Sequence Finding**: Exhaustive search for valid 9-line sequences where:
  - Gap between lines is 60-120 pixels
  - Lines maintain consistent spacing (±15 pixels tolerance)
  - Grid fits within the 800×800 board area
- **Fitting**: Linear regression to refine origin and spacing
- **Scoring**: Prefers sequences with more detected lines and lower error

### Perspective Transform

- Uses 4 board corners detected by AI model as source points
- Maps to 800×800 normalized space
- Inverse transform allows converting grid coordinates back to camera space

### Chess Notation Mapping

Based on user's selection of the a1 corner:
- **top-left**: a8 (black's back rank on left)
- **top-right**: h8 (black's back rank on right)
- **bottom-left**: a1 (white's back rank on left)
- **bottom-right**: h1 (white's back rank on right)

## Hardware Requirements

- **Raspberry Pi** (4B or newer recommended)
- **Raspberry Pi Camera** (v3 recommended)
- **Network connectivity** (for Roboflow API)

## Software Requirements

- Python 3.7+
- OpenCV (cv2)
- Flask
- NumPy
- python-dotenv
- pycocotools
- requests

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/tvrlexe/chessora.git
   cd chessora
   ```

2. **Set up environment**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Roboflow API**:
   - Create a `.env` file in the project root:
   ```bash
   ROBOFLOW_API_KEY=your_api_key_here
   ```

4. **Run the application**:
   ```bash
   python src/main.py
   ```

5. **Access the web interface**:
   - Open browser to `http://chessora-pi:5000` (or your Pi's IP address)

## Usage Workflow

1. **Position the Board**: Place chess board in camera view
2. **Confirm Position**: Click "YES" when satisfied with board framing
3. **Wait for Detection**: Application detects board corners and reconstructs grid
4. **Select Orientation**: Click the corner that corresponds to square a1
5. **View Annotation**: Live video stream now shows algebraic notation for each square

To reposition, click "REPOSITION BOARD" to start over.

## API Endpoints

- `GET /` - Main web interface
- `GET /video` - Live camera stream (positioning phase)
- `GET /board` - Perspective-corrected board view (detection phase)
- `GET /annotated` - Annotated board with square labels (final phase)
- `POST /confirm` - Trigger board detection processing
- `POST /orientation` - Set board orientation (corner selection)
- `POST /reposition` - Reset and start over
- `GET /status` - Current processing status

## Configuration Constants

Located in `src/grid_reconstruction.py`:
- `MIN_GAP`: 60 pixels (minimum space between grid lines)
- `MAX_GAP`: 120 pixels (maximum space between grid lines)
- `TOLERANCE`: 15 pixels (line alignment tolerance)
- `SIZE`: 800 pixels (normalized board size)
- `GRID_LINES`: 9 (8 squares + 1 border line)

## Camera Settings

Configured in `src/main.py`:
- **Resolution**: 960×720
- **FPS**: 30
- **Codec**: MJPEG
- **Quality settings**: Sharpness 1.0, Contrast 1.0, Saturation 1.0

## Future Enhancements

- [ ] Piece detection and tracking
- [ ] Move validation
- [ ] PGN export
- [ ] Multi-angle support
- [ ] Improved performance metrics
- [ ] Configuration UI
- [ ] Stockfish integration for analysis

## Testing

Test files are available in the `tests/` directory.

## Tools

Utility scripts available in the `tools/` directory.

## License

[Add your license information here]

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Author

Created by [tvrlexe](https://github.com/tvrlexe)

## References

- [Roboflow Serverless API](https://docs.roboflow.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Segment Anything Model (SAM)](https://segment-anything.com/)

---

**Note**: This project uses Roboflow's API which requires an internet connection and valid API credentials.
