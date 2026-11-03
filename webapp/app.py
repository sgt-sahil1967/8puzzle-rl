"""Flask demo site for the 8-puzzle solver mini project.

Run from anywhere with:   python3 webapp/app.py   [--port 5000] [--debug]

The server only binds to 127.0.0.1.  Uploaded images are processed purely in
memory (Pillow) and returned to the browser as a base64 data URL; nothing is
ever written to disk.

Board convention (same as puzzle/env.py): a board is a list of 9 ints, row by
row, 0 = blank.  Actions move the BLANK: 0 up, 1 down, 2 left, 3 right.
"""
import argparse
import base64
import io
import os
import random
import sys

# Make `import puzzle` work when this file is started as `python webapp/app.py`.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402
from flask import Flask, jsonify, render_template, request, send_from_directory  # noqa: E402
from PIL import Image, ImageOps, UnidentifiedImageError  # noqa: E402
from werkzeug.exceptions import HTTPException  # noqa: E402

from puzzle import astar, env  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_UPLOAD_BYTES = 5 * 1024 * 1024        # image size limit (file itself)
MAX_IMAGE_PIXELS = 40_000_000             # refuse absurdly large decoded images
OUTPUT_SIZE = 600                         # largest side of the returned image
ALLOWED_FORMATS = ("PNG", "JPEG", "WEBP")
ASTAR_NODE_LIMIT = 300_000                # 8-puzzle has only 181,440 states
METHODS = ("qlearning", "dqn", "dqn2", "astar")
MIN_DEPTH, MAX_DEPTH = 3, 30

CHECKPOINT_DIR = os.path.join(ROOT, "puzzle", "checkpoints")
RESULTS_DIR = os.path.join(ROOT, "puzzle", "results")
Q_PATH = os.path.join(CHECKPOINT_DIR, "q_snap_100.pkl")
DQN_PATH = os.path.join(CHECKPOINT_DIR, "dqn_snap_100.pt")
DQN2_PATH = os.path.join(CHECKPOINT_DIR, "dqn2_snap_100.pt")
RESULT_IMAGES = ("solve_by_depth.png", "learning_curves.png", "snapshots_by_depth.png")  # whitelist

app = Flask(__name__)
# Multipart framing adds a little overhead on top of the file, so allow a small
# margin here; the exact 5 MB check on the file itself happens in /api/upload.
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES + 64 * 1024
app.config["JSON_SORT_KEYS"] = False


# ---------------------------------------------------------------------------
# Model loading (once, at startup).  A missing module/checkpoint never crashes.
# ---------------------------------------------------------------------------
MODELS = {
    "qlearning": {"model": None, "solver": None, "message": "model not trained yet"},
    "dqn": {"model": None, "solver": None, "message": "model not trained yet"},
    "dqn2": {"model": None, "solver": None, "message": "model not trained yet"},
}


def load_models():
    """Try to load the Q-table and the DQN policy; record why if we cannot."""
    # --- Q-learning ---
    try:
        if not os.path.exists(Q_PATH):
            raise FileNotFoundError("checkpoint %s not found" % os.path.basename(Q_PATH))
        from puzzle import qlearn
        MODELS["qlearning"]["model"] = qlearn.load_q_table(Q_PATH)
        MODELS["qlearning"]["solver"] = qlearn.solve_q
        MODELS["qlearning"]["message"] = "ready"
    except Exception as exc:  # ImportError, FileNotFoundError, unpickling errors...
        MODELS["qlearning"]["model"] = None
        MODELS["qlearning"]["message"] = "model not trained yet (%s)" % exc
    # --- DQN ---
    try:
        if not os.path.exists(DQN_PATH):
            raise FileNotFoundError("checkpoint %s not found" % os.path.basename(DQN_PATH))
        from puzzle import dqn
        MODELS["dqn"]["model"] = dqn.load_policy(DQN_PATH)
        MODELS["dqn"]["solver"] = dqn.solve_dqn
        MODELS["dqn"]["message"] = "ready"
    except Exception as exc:  # includes torch not installed
        MODELS["dqn"]["model"] = None
        MODELS["dqn"]["message"] = "model not trained yet (%s)" % exc
    # --- DQN v2 (wider/deeper MLP, same API) ---
    try:
        if not os.path.exists(DQN2_PATH):
            raise FileNotFoundError("checkpoint %s not found" % os.path.basename(DQN2_PATH))
        from puzzle import dqn
        MODELS["dqn2"]["model"] = dqn.load_policy(DQN2_PATH)
        MODELS["dqn2"]["solver"] = dqn.solve_dqn
        MODELS["dqn2"]["message"] = "ready"
    except Exception as exc:
        MODELS["dqn2"]["model"] = None
        MODELS["dqn2"]["message"] = "model not trained yet (%s)" % exc


load_models()


# ---------------------------------------------------------------------------
# Error handling: every error on an /api/ path is JSON {"error": "..."}
# ---------------------------------------------------------------------------
class ApiError(Exception):
    """Raised by route code to return a JSON error with a status code."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.message = message
        self.status = status


@app.errorhandler(ApiError)
def handle_api_error(err):
    return jsonify({"error": err.message}), err.status


@app.errorhandler(HTTPException)
def handle_http_error(err):
    # 404 / 405 / 413 etc. Keep default HTML pages for non-API paths.
    if request.path.startswith("/api/"):
        messages = {
            404: "Not found.",
            405: "Method not allowed for this route.",
            413: "Upload too large (max 5 MB).",
        }
        return jsonify({"error": messages.get(err.code, err.description)}), err.code
    return err


@app.errorhandler(Exception)
def handle_unexpected(err):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Internal server error."}), 500
    raise err


@app.after_request
def no_cache_for_api(resp):
    if request.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def get_json_body():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ApiError("Request body must be a JSON object.")
    return data


def parse_board(data):
    """Return the board from a request as a tuple, or raise a 400."""
    board = data.get("board")
    if not isinstance(board, list) or len(board) != 9:
        raise ApiError("board must be a list of 9 integers.")
    for v in board:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ApiError("board must contain only integers.")
    if sorted(board) != list(range(9)):
        raise ApiError("board must be a permutation of 0..8.")
    board = tuple(board)
    if not env.is_solvable(board):
        raise ApiError("board is not solvable.")
    return board


def parse_method(data):
    method = data.get("method")
    if not isinstance(method, str) or method not in METHODS:
        raise ApiError("method must be one of: %s." % ", ".join(METHODS))
    return method


def run_method(method, board):
    """Run one solver on `board` (tuple) and return a JSON-ready dict."""
    if method == "astar":
        res = astar.solve_result(board, node_limit=ASTAR_NODE_LIMIT)
    else:
        slot = MODELS[method]
        if slot["model"] is None:
            raise ApiError("%s: %s" % (method, slot["message"]), 503)
        res = slot["solver"](slot["model"], board, max_steps=100)
    final = tuple(res["final_state"])
    return {
        "method": method,
        "board": list(board),
        "moves": [int(a) for a in res["moves"]],
        "solved": bool(res["solved"]),
        "num_moves": int(res["num_moves"]),
        "time_ms": round(float(res["time_ms"]), 3),
        "reason": str(res["reason"]),
        "final_board": list(final),
    }


# ---------------------------------------------------------------------------
# Pages and status
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    """Which solvers are usable (the UI disables buttons for the others)."""
    return jsonify({
        "astar": {"available": True, "message": "ready"},
        "qlearning": {"available": MODELS["qlearning"]["model"] is not None,
                      "message": MODELS["qlearning"]["message"]},
        "dqn": {"available": MODELS["dqn"]["model"] is not None,
                "message": MODELS["dqn"]["message"]},
        "dqn2": {"available": MODELS["dqn2"]["model"] is not None,
                 "message": MODELS["dqn2"]["message"]},
    })


# ---------------------------------------------------------------------------
# Image upload (in memory only)
# ---------------------------------------------------------------------------
@app.route("/api/upload", methods=["POST"])
def upload():
    file = request.files.get("image")
    if file is None or not file.filename:
        raise ApiError("No image file was sent (form field 'image').")
    raw = file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ApiError("Image is larger than 5 MB.", 413)
    if not raw:
        raise ApiError("The uploaded file is empty.")
    try:
        # First pass: verify() checks the file structure without decoding pixels.
        probe = Image.open(io.BytesIO(raw))
        fmt = probe.format
        if fmt not in ALLOWED_FORMATS:
            raise ApiError("Only PNG, JPEG or WEBP images are allowed.", 415)
        if probe.width * probe.height > MAX_IMAGE_PIXELS:
            raise ApiError("Image dimensions are too large.", 413)
        probe.verify()
        # Second pass: a real decode (verify() leaves the image unusable).
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img)  # respect phone-camera rotation
        img.load()
    except ApiError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError,
            Image.DecompressionBombError):
        raise ApiError("File is not a valid PNG, JPEG or WEBP image.", 400)

    # Flatten transparency onto white, then centre-crop to a square.
    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        img = img.convert("RGB")
    side = min(img.size)
    left = (img.width - side) // 2
    top = (img.height - side) // 2
    img = img.crop((left, top, left + side, top + side))
    if side > OUTPUT_SIZE:
        img = img.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    return jsonify({"data_url": data_url, "width": img.width, "height": img.height,
                    "original_format": fmt})


# ---------------------------------------------------------------------------
# Puzzle API
# ---------------------------------------------------------------------------
@app.route("/api/scramble", methods=["POST"])
def api_scramble():
    depth = get_json_body().get("depth")
    if isinstance(depth, bool) or not isinstance(depth, int) or not MIN_DEPTH <= depth <= MAX_DEPTH:
        raise ApiError("depth must be an integer from %d to %d." % (MIN_DEPTH, MAX_DEPTH))
    rng = random.Random()  # fresh, unseeded generator for every request
    board = env.scramble(depth, rng)
    while board == env.goal_state():  # extremely unlikely, but never hand out a solved board
        board = env.scramble(depth, rng)
    return jsonify({"board": list(board)})


@app.route("/api/solve", methods=["POST"])
def api_solve():
    data = get_json_body()
    board = parse_board(data)
    method = parse_method(data)
    return jsonify(run_method(method, board))


@app.route("/api/compare", methods=["POST"])
def api_compare():
    board = parse_board(get_json_body())
    results = {}
    for method in METHODS:
        try:
            results[method] = run_method(method, board)
        except ApiError as exc:  # model missing: report it, keep the others
            results[method] = {
                "method": method, "board": list(board), "moves": [], "solved": False,
                "num_moves": 0, "time_ms": 0.0, "reason": "model_unavailable",
                "final_board": list(board), "error": exc.message,
            }
    return jsonify({"board": list(board), "results": results})


@app.route("/api/hint", methods=["POST"])
def api_hint():
    board = parse_board(get_json_body())
    moves = astar.solve(board, node_limit=ASTAR_NODE_LIMIT)
    if moves is None:
        raise ApiError("Could not compute a hint (search limit reached).", 500)
    if not moves:  # already solved
        return jsonify({"board": list(board), "action": None, "tile": None, "remaining": 0})
    action = moves[0]
    # The blank moves by `action`; the tile that slides is the one at the target cell.
    blank = board.index(0)
    target = blank + (-3, 3, -1, 1)[action]
    return jsonify({"board": list(board), "action": int(action), "tile": int(board[target]),
                    "remaining": len(moves)})


# ---------------------------------------------------------------------------
# Training results panel
# ---------------------------------------------------------------------------
@app.route("/api/results")
def api_results():
    """Overall rows of comparison.csv plus which whitelisted PNGs exist."""
    rows = None
    csv_path = os.path.join(RESULTS_DIR, "comparison.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, dtype={"depth": str})
            df = df[df["depth"] == "overall"]
            df = df.astype(object).where(df.notna(), None)
            rows = df.to_dict(orient="records")
        except Exception:
            rows = None
    images = {name: os.path.exists(os.path.join(RESULTS_DIR, name)) for name in RESULT_IMAGES}
    return jsonify({"rows": rows, "images": images})


@app.route("/api/results/image/<name>")
def api_result_image(name):
    if name not in RESULT_IMAGES:  # only whitelisted file names are ever served
        raise ApiError("Not found.", 404)
    if not os.path.exists(os.path.join(RESULTS_DIR, name)):
        raise ApiError("Image not generated yet.", 404)
    return send_from_directory(RESULTS_DIR, name, mimetype="image/png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="8-puzzle demo web app")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true", help="enable Flask debug mode")
    args = parser.parse_args()
    # host is fixed to localhost on purpose: this is a local demo only.
    app.run(host="127.0.0.1", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
