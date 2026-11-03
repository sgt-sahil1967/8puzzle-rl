"""Helpers shared by Q-learning and DQN: paths, reward, curriculum, test set,
logging and atomic checkpoint saving."""
import csv
import json
import os
import pickle
import random
from collections import deque
from pathlib import Path

from .env import DEFAULT_SHAPE, goal_state, manhattan, scramble

PKG_DIR = Path(__file__).resolve().parent
LOG_DIR = PKG_DIR / "logs"
CKPT_DIR = PKG_DIR / "checkpoints"
RESULTS_DIR = PKG_DIR / "results"
TEST_SET_PATH = RESULTS_DIR / "test_set.json"
TEST_DEPTHS = [1, 3, 5, 8, 12, 16, 20, 25, 30]
BOARDS_PER_DEPTH = 200
TEST_SEED = 12345

GAMMA = 0.99
GOAL_REWARD = 10.0
STEP_COST = -0.1
SHAPING_SCALE = 0.3  # weight of the Manhattan shaping term ("small")


def ensure_dirs():
    for d in (LOG_DIR, CKPT_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- reward ----
def reward(state, next_state, shape=DEFAULT_SHAPE):
    """+10 on reaching the goal, -0.1 per step, plus potential-based shaping.

    Shaping uses potential phi(s) = -Manhattan(s):
        F = gamma * phi(s') - phi(s) = h(s) - gamma * h(s')
    so a move that lowers the Manhattan distance gives a small bonus. Potential
    based shaping does not change which policy is optimal.
    Returns (reward, done).
    """
    done = next_state == goal_state(shape)
    r = STEP_COST + SHAPING_SCALE * (manhattan(state, shape) - GAMMA * manhattan(next_state, shape))
    if done:
        r += GOAL_REWARD
    return r, done


# ------------------------------------------------------------- curriculum ----
class Curriculum:
    """Scramble-depth range that grows as the agent gets better.

    Boards are scrambled by a random depth in [1, max_depth]. We keep a rolling
    window of recent training outcomes; once the window is full and the solve
    rate reaches `threshold`, max_depth goes up by one and the window restarts.
    """

    def __init__(self, start_depth=3, cap=32, window=300, threshold=0.8):
        self.max_depth, self.cap, self.threshold = start_depth, cap, threshold
        self.window = deque(maxlen=window)

    def sample_depth(self, rng):
        return rng.randint(1, self.max_depth)

    def record(self, solved):
        self.window.append(1 if solved else 0)
        if (len(self.window) == self.window.maxlen and self.max_depth < self.cap
                and sum(self.window) / len(self.window) >= self.threshold):
            self.max_depth += 1
            self.window.clear()

    def state_dict(self):
        return {"max_depth": self.max_depth, "window": list(self.window)}

    def load_state_dict(self, d):
        self.max_depth = d["max_depth"]
        self.window.clear()
        self.window.extend(d["window"])


def step_cap_for(depth):
    """Episode step limit, proportional to the scramble depth."""
    return max(12, 4 * depth)


# -------------------------------------------------------------- test set ----
def load_test_set(path=TEST_SET_PATH):
    """List of dicts {depth, state (tuple), optimal}."""
    with open(path) as f:
        data = json.load(f)
    return [{"depth": b["depth"], "state": tuple(b["state"]), "optimal": b["optimal"]}
            for b in data["boards"]]


def test_state_set(path=TEST_SET_PATH):
    return {b["state"] for b in load_test_set(path)}


def sample_train_board(depth, rng, forbidden, shape=DEFAULT_SHAPE):
    """Scramble a training board that is NOT one of the fixed test boards.

    Shallow depths have so few distinct boards that the test set contains all of
    them; in that case we silently use a slightly deeper scramble instead.
    Returns (board, depth_used).
    """
    for extra in range(0, 8):
        for _ in range(10):
            board = scramble(depth + extra, rng, shape)
            if board not in forbidden:
                return board, depth + extra
    raise RuntimeError("could not find a non-test training board")


def make_validation_set(forbidden, per_depth=40, seed=777, depths=(8, 12, 16, 20, 25, 30)):
    """Boards used to pick the best DQN checkpoint (different seed, never test boards).

    The test set must stay untouched for the final comparison, so model
    selection uses this separate set. Shallow depths are left out because the
    test set already contains (nearly) every shallow board.
    """
    rng = random.Random(seed)
    boards = []
    for d in depths:
        while sum(1 for b in boards if b["depth"] == d) < per_depth:
            state = scramble(d, rng)
            if state not in forbidden:
                boards.append({"depth": d, "state": state, "optimal": 0})
    return boards


def evaluate_policy(solve_fn, boards):
    """Run solve_fn(state)->result dict on test boards.

    Returns a dict with overall solve rate, average moves and average optimal
    length (both over solved boards), and solve rate per depth.
    """
    solved = moves = opt = 0
    per_depth = {}
    for b in boards:
        res = solve_fn(b["state"])
        n, s = per_depth.get(b["depth"], (0, 0))
        per_depth[b["depth"]] = (n + 1, s + (1 if res["solved"] else 0))
        if res["solved"]:
            solved += 1
            moves += res["num_moves"]
            opt += b["optimal"]
    return {
        "solve_rate": solved / len(boards),
        "avg_moves": moves / solved if solved else float("nan"),
        "avg_optimal": opt / solved if solved else float("nan"),
        "per_depth": {d: s / n for d, (n, s) in sorted(per_depth.items())},
    }


# ------------------------------------------------------- logs/checkpoints ----
def epsilon_at(episode, total, start=1.0, end=0.05, decay_frac=0.6):
    """Exponential decay from start to end over the first decay_frac of training."""
    horizon = max(1, decay_frac * total)
    return max(end, start * (end / start) ** (episode / horizon))


def save_atomic(path, writer):
    """Write to a temp file then rename, so an interrupt never corrupts `path`."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        writer(f)
    os.replace(tmp, path)


def save_pickle(path, obj):
    save_atomic(path, lambda f: pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL))


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def write_log(path, rows, fields):
    """Rewrite the whole CSV (it is small); keeps the file consistent on resume."""
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def snapshot_episodes(total, first):
    """Episode numbers at which named snapshots are saved.

    "0" (~0%) is taken at the first log interval (an untrained agent can do
    nothing, so we use the earliest moment it has learned something).
    """
    return {"0": min(first, total), "25": max(1, int(0.25 * total)),
            "50": max(1, int(0.5 * total)), "100": total}


def make_rng(seed):
    return random.Random(seed)
