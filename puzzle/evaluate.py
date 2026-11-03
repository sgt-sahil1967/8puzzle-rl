"""Fixed test set creation and the A* / Q-learning / DQN comparison.

  python -m puzzle.evaluate --make-test-set   (already done once; seed is fixed)
  python -m puzzle.evaluate                    -> results/comparison.csv, comparison.md, per_board.csv
"""
import argparse
import json
import random
import time

from . import astar
import pandas as pd

from . import dqn, qlearn
from .common import (BOARDS_PER_DEPTH, CKPT_DIR, RESULTS_DIR, TEST_DEPTHS, TEST_SEED,
                     TEST_SET_PATH, ensure_dirs, load_test_set)
from .env import scramble


def make_test_set(path=TEST_SET_PATH, seed=TEST_SEED):
    """200 boards for each scramble depth, with the A* optimal length stored."""
    ensure_dirs()
    rng = random.Random(seed)
    boards = []
    t0 = time.time()
    for depth in TEST_DEPTHS:
        for _ in range(BOARDS_PER_DEPTH):
            state = scramble(depth, rng)
            opt = len(astar.solve(state))
            boards.append({"depth": depth, "state": list(state), "optimal": opt})
        print("depth %2d done (%.0fs)" % (depth, time.time() - t0), flush=True)
    with open(path, "w") as f:
        json.dump({"seed": seed, "boards": boards}, f)
    print("wrote", path)


def run_method(name, solve_fn, boards, q_table=None):
    """Solve every test board; one row per board."""
    rows = []
    for b in boards:
        res = solve_fn(b["state"])
        rows.append({
            "method": name, "depth": b["depth"], "state": "".join(map(str, b["state"])),
            "optimal": b["optimal"], "solved": res["solved"], "reason": res["reason"],
            "num_moves": res["num_moves"], "time_ms": res["time_ms"],
            # for Q-learning: was the starting board an entry in the table?
            "seen_in_table": (b["state"] in q_table) if q_table is not None else None})
    return rows


def summarize(per_board):
    """Aggregate per method and depth (plus 'overall') into comparison rows."""
    out = []
    for method, g in per_board.groupby("method", sort=False):
        groups = [(str(d), x) for d, x in g.groupby("depth")] + [("overall", g)]
        for depth, x in groups:
            ok = x[x["solved"]]
            out.append({
                "method": method, "depth": depth, "n": len(x),
                "solve_rate": round(x["solved"].mean(), 4),
                "avg_moves": round(ok["num_moves"].mean(), 3) if len(ok) else float("nan"),
                "avg_optimal": round(ok["optimal"].mean(), 3) if len(ok) else float("nan"),
                # extra moves over A* on boards the method solved
                "optimality_gap": round((ok["num_moves"] - ok["optimal"]).mean(), 3) if len(ok) else float("nan"),
                "avg_time_ms": round(x["time_ms"].mean(), 4)})
    return pd.DataFrame(out)


def to_markdown(df):
    """Small markdown table writer (avoids the optional 'tabulate' package)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-test-set", action="store_true")
    ap.add_argument("--max-steps", type=int, default=100, help="greedy rollout step cap")
    args = ap.parse_args()
    if args.make_test_set:
        make_test_set()
        return
    ensure_dirs()
    boards = load_test_set()
    rows = run_method("astar", astar.solve_result, boards)
    Q = qlearn.load_q_table(CKPT_DIR / "q_snap_100.pkl")
    rows += run_method("qlearning", lambda s: qlearn.solve_q(Q, s, max_steps=args.max_steps), boards, Q)
    policy = dqn.load_policy(CKPT_DIR / "dqn_snap_100.pt")
    rows += run_method("dqn", lambda s: dqn.solve_dqn(policy, s, max_steps=args.max_steps), boards)
    path2 = CKPT_DIR / "dqn2_snap_100.pt"  # improved DQN (double + n-step + wider), optional
    if path2.exists():
        policy2 = dqn.load_policy(path2)
        rows += run_method("dqn2", lambda s: dqn.solve_dqn(policy2, s, max_steps=args.max_steps), boards)
    per_board = pd.DataFrame(rows)
    per_board.to_csv(RESULTS_DIR / "per_board.csv", index=False)
    table = summarize(per_board)
    table.to_csv(RESULTS_DIR / "comparison.csv", index=False)
    (RESULTS_DIR / "comparison.md").write_text(to_markdown(table) + "\n")
    print(to_markdown(table[table["depth"] == "overall"]))
    print("wrote comparison.csv / comparison.md / per_board.csv in", RESULTS_DIR)


if __name__ == "__main__":
    main()
