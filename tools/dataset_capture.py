import json
import random
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import chess
import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from board_detection import get_board_points
from perspective import get_perspective_matrix, warp_board


WIDTH = 960
HEIGHT = 720
BOARD_SIZE = 800
BOARD_EXPANSION_PERCENT = 0.30
PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "photos" / "dataset"
METADATA_DIR = PROJECT_DIR / "photos" / "metadata"
PREVIEW_DIR = PROJECT_DIR / "photos" / "previews"

app = Flask(__name__)
camera_process = None
camera_frame = None
frame_lock = threading.Lock()
state_lock = threading.Lock()
position_seed = None
position_board = None
position_mode = "random"
orientation = "a8"
angle_index = 0
preview_path = None
last_capture = None
last_result = {}


PIECE_LABELS = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}
PIECE_SYMBOLS = {
    "white": {"pawn": "♙", "knight": "♘", "bishop": "♗", "rook": "♖", "queen": "♕", "king": "♔"},
    "black": {"pawn": "♟", "knight": "♞", "bishop": "♝", "rook": "♜", "queen": "♛", "king": "♚"},
}


def random_position(seed=None):
    rng = random.Random(seed)
    board = chess.Board()
    for _ in range(rng.randint(12, 45)):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
    return board


def render_position(board, orientation="a8", square_size=96):
    image = np.zeros((square_size * 8, square_size * 8, 3), dtype=np.uint8)
    light = (235, 218, 185)
    dark = (120, 80, 45)

    if orientation == "white-left":
        coordinates = [
            (file_index, rank_index)
            for file_index in range(8)
            for rank_index in range(8)
        ]
    elif orientation == "white-right":
        coordinates = [
            (file_index, rank_index)
            for file_index in range(8)
            for rank_index in range(7, -1, -1)
        ]
    else:
        files = range(8) if orientation in ("a8", "a1") else range(7, -1, -1)
        ranks = range(7, -1, -1) if orientation in ("a8", "h8") else range(8)
        coordinates = [
            (file_index, rank_index)
            for rank_index in ranks
            for file_index in files
        ]

    for display_index, (file_index, rank_index) in enumerate(coordinates):
        row, col = divmod(display_index, 8)
        square = chess.square(file_index, rank_index)
        left = col * square_size
        top = row * square_size
        color = light if (row + col) % 2 == 0 else dark
        cv2.rectangle(image, (left, top), (left + square_size, top + square_size), color, -1)

        piece = board.piece_at(square)
        if piece is None:
            continue

        label = piece.symbol().upper()
        position = (left + square_size // 3, top + int(square_size * 0.70))
        cv2.putText(image, label, position, cv2.FONT_HERSHEY_SIMPLEX, 2.0, (20, 20, 20), 8, cv2.LINE_AA)
        cv2.putText(image, label, position, cv2.FONT_HERSHEY_SIMPLEX, 2.0, (245, 245, 245), 3, cv2.LINE_AA)

    return image


def board_metadata(board):
    return {
        chess.square_name(square): {
            "color": "white" if piece.color else "black",
            "piece": PIECE_LABELS[piece.piece_type],
        }
        for square, piece in board.piece_map().items()
    }


def start_camera():
    global camera_process
    camera_process = subprocess.Popen(
        [
            "rpicam-vid", "--width", str(WIDTH), "--height", str(HEIGHT),
            "--framerate", "30", "--codec", "mjpeg", "--timeout", "0",
            "--nopreview", "--sharpness", "1.0", "--contrast", "1.0",
            "--brightness", "0.0", "--saturation", "1.0", "--vflip", "-o", "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )


def read_camera():
    global camera_frame
    buffer = b""
    while True:
        chunk = camera_process.stdout.read(4096)
        if not chunk:
            return
        buffer += chunk
        while True:
            start = buffer.find(b"\xff\xd8")
            end = buffer.find(b"\xff\xd9", start + 2)
            if start < 0 or end < 0:
                break
            encoded = buffer[start:end + 2]
            buffer = buffer[end + 2:]
            image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                with frame_lock:
                    camera_frame = image


def multipart(image):
    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        return None
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"


def set_position(seed=None, mode="random", selected_orientation=None):
    global position_seed, position_board, position_mode, orientation
    global preview_path, last_capture, last_result
    position_seed = seed if seed is not None else random.randrange(1_000_000_000)
    position_mode = mode
    if selected_orientation is not None:
        orientation = selected_orientation
    position_board = chess.Board() if mode == "starting" else random_position(position_seed)
    preview_path = None
    last_capture = None
    last_result = {}


def process_capture(image, sample_id):
    points = get_board_points(image)
    center = np.mean(points, axis=0, keepdims=True)
    expanded_points = center + (
        points - center
    ) * (1.0 + BOARD_EXPANSION_PERCENT)
    matrix = get_perspective_matrix(expanded_points, BOARD_SIZE)
    warped = warp_board(image, matrix, BOARD_SIZE)

    crop_path = OUTPUT_DIR / f"{sample_id}.jpg"
    cv2.imwrite(str(crop_path), warped)
    return points, expanded_points, crop_path


@app.route("/")
def index():
    return """
<!doctype html><html><head><meta charset="utf-8"><title>Chessora Dataset Capture</title>
<style>
body{margin:0;background:#111820;color:#eef2f3;font-family:Arial,sans-serif}main{max-width:1400px;margin:auto;padding:24px}h1{margin-top:0}.layout{display:grid;grid-template-columns:minmax(320px,1fr) minmax(320px,1fr);gap:24px}.panel{background:#1d2932;border:1px solid #3d4c55;border-radius:8px;padding:18px}.digital-board{display:grid;grid-template-columns:repeat(8,minmax(0,1fr));grid-template-rows:repeat(8,minmax(0,1fr));aspect-ratio:1;width:100%;overflow:hidden;background:#76502f}.square{display:grid;place-items:center;min-width:0;min-height:0;overflow:hidden;font-family:'DejaVu Sans',serif;font-size:clamp(28px,6vw,64px);line-height:1}.square.light{background:#ead9b5}.square.dark{background:#9b6b43}.piece-white{color:#fff;text-shadow:0 2px 2px #111}.piece-black{color:#111;text-shadow:0 1px 1px #fff}img{display:block;width:100%;height:auto;background:#080b0d}.results{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.meta{font-family:monospace;white-space:pre-wrap;color:#b9c9cd;margin:12px 0}.actions{display:flex;gap:12px;flex-wrap:wrap;margin-top:16px}button{border:0;border-radius:5px;padding:13px 20px;font-size:16px;font-weight:bold;cursor:pointer;background:#e1b45b;color:#17130b}button.secondary{background:#536a73;color:white}button:disabled{opacity:.5;cursor:wait}.status{min-height:24px;margin-top:14px;color:#e1b45b}@media(max-width:800px){.layout{grid-template-columns:1fr}.results{grid-template-columns:1fr}main{padding:12px}}
</style></head><body><main><h1>Chessora Dataset Capture</h1><div class="layout"><section class="panel"><h2>Digital Position</h2><div id="digital-board" class="digital-board"></div><div id="meta" class="meta"></div><div class="actions"><button id="new-button" class="secondary">Random Position</button><button id="starting-button" class="secondary">Starting Position</button><button id="capture-button">Take Photo</button></div><h3>Default Starting Setup by Angle</h3><div class="actions"><button data-default-angle="a8">White Bottom</button><button data-default-angle="h8">White Bottom Flipped</button><button data-default-angle="a1">Black Bottom</button><button data-default-angle="h1">Black Bottom Flipped</button><button data-default-angle="white-left">White Left</button><button data-default-angle="white-right">White Right</button></div><div class="actions"><button data-angle="a8">Top Left: a8</button><button data-angle="h8">Top Right: h8</button><button data-angle="a1">Bottom Left: a1</button><button data-angle="h1">Bottom Right: h1</button><button data-angle="white-left">White Left</button><button data-angle="white-right">White Right</button></div><div id="status" class="status"></div></section><section class="panel"><h2>Camera</h2><img src="/video"><p class="meta">Select the physical board angle, reproduce the digital position, then capture.</p></section></div><section id="result" class="panel" style="display:none;margin-top:24px"><h2>Latest Dataset Image</h2><img id="crop"></section></main><script>
const digitalBoard=document.getElementById('digital-board');const meta=document.getElementById('meta');const status=document.getElementById('status');const result=document.getElementById('result');const symbols={white:{pawn:'♙',knight:'♘',bishop:'♗',rook:'♖',queen:'♕',king:'♔'},black:{pawn:'♟',knight:'♞',bishop:'♝',rook:'♜',queen:'♛',king:'♚'}};
function drawBoard(pieces,angle){digitalBoard.innerHTML='';let coordinates=[];if(angle==='white-left'){for(let file=0;file<8;file++)for(let rank=0;rank<8;rank++)coordinates.push([file,rank])}else if(angle==='white-right'){for(let file=0;file<8;file++)for(let rank=7;rank>=0;rank--)coordinates.push([file,rank])}else{const files=(angle==='a8'||angle==='a1'?[0,1,2,3,4,5,6,7]:[7,6,5,4,3,2,1,0]);const ranks=(angle==='a8'||angle==='h8'?[7,6,5,4,3,2,1,0]:[0,1,2,3,4,5,6,7]);for(const rank of ranks)for(const file of files)coordinates.push([file,rank])}for(let i=0;i<64;i++){const [file,rank]=coordinates[i];const row=Math.floor(i/8);const col=i%8;const square=String.fromCharCode(97+file)+(rank+1);const cell=document.createElement('div');cell.className='square '+((row+col)%2?'dark':'light');const piece=pieces[square];if(piece){cell.textContent=symbols[piece.color][piece.piece];cell.classList.add('piece-'+piece.color)}digitalBoard.appendChild(cell)}}
function refresh(){fetch('/state').then(r=>r.json()).then(s=>{drawBoard(s.pieces,s.orientation);meta.textContent='Mode: '+s.mode+'\\nAngle: '+s.orientation+'\\nSeed: '+s.seed+'\\nFEN: '+s.fen+'\\nPieces: '+s.piece_count+(s.last_capture?'\\nLast: '+s.last_capture:'');if(s.result){result.style.display='block';document.getElementById('crop').src='/result/crop?t='+Date.now()}})}
document.getElementById('new-button').onclick=()=>{status.textContent='Generating random position...';fetch('/new',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:'random'})}).then(refresh).then(()=>{status.textContent='Random position ready.'})};document.getElementById('starting-button').onclick=()=>{status.textContent='Loading starting position...';fetch('/new',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:'starting'})}).then(refresh).then(()=>{status.textContent='Starting position ready.'})};document.querySelectorAll('[data-angle]').forEach(button=>button.onclick=()=>{fetch('/angle',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({angle:button.dataset.angle})}).then(refresh)});document.getElementById('capture-button').onclick=()=>{status.textContent='Capturing and processing board...';fetch('/capture',{method:'POST'}).then(r=>r.json()).then(s=>{status.textContent=s.message;refresh()})};refresh();
document.querySelectorAll('[data-default-angle]').forEach(button=>button.onclick=()=>{status.textContent='Loading default starting setup...';fetch('/new',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:'starting',angle:button.dataset.defaultAngle})}).then(refresh).then(()=>{status.textContent='Default starting setup ready.'})});refresh();
</script></body></html>
"""


@app.route("/preview")
def preview():
    if preview_path is not None and preview_path.exists():
        return send_file(preview_path, mimetype="image/png")

    success, encoded = cv2.imencode(
        ".png",
        render_position(position_board, orientation),
    )
    if not success:
        return {"error": "Could not render preview"}, 500
    return Response(encoded.tobytes(), mimetype="image/png")


@app.route("/result/<kind>")
def result_image(kind):
    path = last_result.get(kind)
    if path is None or not Path(path).exists():
        return {"error": "No result image"}, 404
    return send_file(path)


@app.route("/video")
def video():
    def generate():
        while True:
            with frame_lock:
                image = None if camera_frame is None else camera_frame.copy()
            if image is not None:
                packet = multipart(image)
                if packet:
                    yield packet
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/state")
def state():
    return jsonify({
        "seed": position_seed,
        "fen": position_board.fen(),
        "mode": position_mode,
        "orientation": orientation,
        "piece_count": len(position_board.piece_map()),
        "pieces": board_metadata(position_board),
        "last_capture": last_capture,
        "result": bool(last_result),
    })


@app.route("/new", methods=["POST"])
def new_position():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "random")
    selected_orientation = data.get("angle")
    if mode not in ("random", "starting"):
        return {"error": "Invalid position mode"}, 400
    if selected_orientation not in (
        None, "a8", "h8", "a1", "h1", "white-left", "white-right"
    ):
        return {"error": "Invalid board angle"}, 400
    set_position(mode=mode, selected_orientation=selected_orientation)
    return {"status": "ok"}


@app.route("/angle", methods=["POST"])
def angle():
    global orientation, preview_path
    data = request.get_json(silent=True) or {}
    selected = data.get("angle")
    if selected not in ("a8", "h8", "a1", "h1", "white-left", "white-right"):
        return {"error": "Invalid board angle"}, 400
    orientation = selected
    preview_path = None
    return {"status": "ok", "orientation": orientation}


@app.route("/capture", methods=["POST"])
def capture():
    global last_capture, last_result, preview_path
    with frame_lock:
        image = None if camera_frame is None else camera_frame.copy()
    if image is None:
        return {"message": "No camera frame available"}, 503

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_id = f"sample_{timestamp}_{position_seed}"
    metadata_path = METADATA_DIR / f"{sample_id}.json"

    try:
        points, expanded_points, crop_path = process_capture(image, sample_id)
        preview_path = PREVIEW_DIR / f"position_{position_seed}_preview.png"
        cv2.imwrite(
            str(preview_path),
            render_position(position_board, orientation),
        )
        status = "board_processed"
        message = f"Saved {sample_id}; board crop ready for annotation."
        last_result = {
            "crop": str(crop_path),
        }
    except Exception as error:
        points = None
        expanded_points = None
        crop_path = None
        status = f"board_processing_failed: {error}"
        message = status
        last_result = {}

    metadata = {
        "sample_id": sample_id,
        "seed": position_seed,
        "fen": position_board.fen(),
        "position_mode": position_mode,
        "orientation": orientation,
        "pieces": board_metadata(position_board),
        "dataset_image": None if crop_path is None else str(crop_path),
        "board_corners": None if points is None else points.tolist(),
        "expanded_corners": None if expanded_points is None else expanded_points.tolist(),
        "board_expansion_percent": BOARD_EXPANSION_PERCENT,
        "status": status,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    last_capture = sample_id
    return {"message": message, "sample_id": sample_id}


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    set_position()
    start_camera()
    threading.Thread(target=read_camera, daemon=True).start()
    print("Open http://chessora-pi:5001")
    app.run(host="0.0.0.0", port=5001, threaded=True)
