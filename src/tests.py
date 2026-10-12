from ultralytics import YOLO
from pathlib import Path
from urllib.request import urlretrieve

MODEL_URL = "https://huggingface.co/acapitani/chesspiece-detection-yolo/resolve/main/chesspiece-detection-model.pt"
MODEL_PATH = Path("chesspiece-detection-model.pt")

# Download the model if it isn't already present
if not MODEL_PATH.exists():
    print("Downloading chess piece model...")
    urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

# Load model
model = YOLO(str(MODEL_PATH))

# Run inference
results = model("photos/photo.jpg")

# Save result
results[0].save("photos/yolo_result.jpg")

print("Result saved to photos/yolo_result.jpg")