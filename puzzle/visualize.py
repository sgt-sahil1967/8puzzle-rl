"""Plots (PNG) and solving GIFs.   python -m puzzle.visualize
Needs: logs/q_log.csv, logs/dqn_log.csv, results/comparison.csv, results/per_board.csv
(run evaluate first) and the four snapshots of each method."""
import argparse

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from . import astar, dqn, qlearn
from .common import CKPT_DIR, LOG_DIR, RESULTS_DIR, TEST_DEPTHS, evaluate_policy, load_test_set
from .env import step

COLORS = {"astar": "#555555", "qlearning": "#0072B2", "dqn": "#D55E00", "dqn2": "#009E73"}
LABELS = {"astar": "A*", "qlearning": "Q-learning", "dqn": "DQN", "dqn2": "DQN v2"}
SNAPS = ["0", "25", "50", "100"]


def _style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def plot_curves(column, ylabel, fname, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    for m, f in (("qlearning", "q_log.csv"), ("dqn", "dqn_log.csv"), ("dqn2", "dqn2_log.csv")):
        if not (LOG_DIR / f).exists():
            continue
        df = pd.read_csv(LOG_DIR / f)
        ax.plot(df["episode"], df[column], label=LABELS[m], color=COLORS[m], lw=1.8)
    ax.set_xscale("log")
    _style(ax, "training episode (log scale)", ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / fname, dpi=140)
    plt.close(fig)


def plot_by_depth(comp):
    depths = [str(d) for d in TEST_DEPTHS]
    methods = [m for m in ("qlearning", "dqn", "dqn2", "astar") if m in set(comp["method"])]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        sub = comp[(comp["method"] == m)].set_index("depth")
        ax.bar(np.arange(len(depths)) + (i - (len(methods) - 1) / 2) * w, [sub.loc[d, "solve_rate"] for d in depths],
               w, label=LABELS[m], color=COLORS[m])
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels(depths)
    ax.set_ylim(0, 1.2)
    _style(ax, "scramble depth", "solve rate on fixed test boards")
    ax.set_title("Solve rate by scramble depth")
    ax.legend(ncol=len(methods), loc="upper center")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "solve_by_depth.png", dpi=140)
    plt.close(fig)


def plot_generalization(pb):
    """Q-learning: start board present in the table (seen) vs absent (unseen)."""
    q = pb[pb["method"] == "qlearning"].copy()
    q["seen_in_table"] = q["seen_in_table"].astype(bool)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.38
    for i, (seen, name, col) in enumerate([(True, "start board seen in table", "#0072B2"),
                                           (False, "start board unseen", "#CC79A7")]):
        xs, ys = [], []
        for j, d in enumerate(TEST_DEPTHS):
            g = q[(q["depth"] == d) & (q["seen_in_table"] == seen)]
            xs.append(j + (i - 0.5) * w)
            ys.append(g["solved"].mean() if len(g) else 0)
            ax.text(xs[-1], ys[-1] + 0.01, "n=%d" % len(g), ha="center", fontsize=7)
        ax.bar(xs, ys, w, label=name, color=col)
    ax.set_xticks(range(len(TEST_DEPTHS)))
    ax.set_xticklabels(TEST_DEPTHS)
    ax.set_ylim(0, 1.25)
    _style(ax, "scramble depth", "Q-learning solve rate")
    ax.set_title("Tabular Q-learning: seen vs unseen test boards (unseen = no knowledge, counts as failure)")
    ax.title.set_fontsize(9)
    ax.legend(loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "generalization.png", dpi=140)
    plt.close(fig)


def snapshot_policies():
    """{method: {snap: solve_fn}} for the four snapshots of each method."""
    out = {"qlearning": {}, "dqn": {}, "dqn2": {}}
    for s in SNAPS:
        Q = qlearn.load_q_table(CKPT_DIR / ("q_snap_%s.pkl" % s))
        out["qlearning"][s] = (lambda st, Q=Q: qlearn.solve_q(Q, st))
        for m, prefix in (("dqn", "dqn"), ("dqn2", "dqn2")):
            path = CKPT_DIR / ("%s_snap_%s.pt" % (prefix, s))
            if path.exists():
                P = dqn.load_policy(path)
                out[m][s] = (lambda st, P=P: dqn.solve_dqn(P, st))
    return {m: v for m, v in out.items() if v}


def plot_snapshots(policies, boards):
    fig, axes = plt.subplots(1, len(policies), figsize=(5 * len(policies), 4), sharey=True)
    shades = dict(zip(SNAPS, [0.35, 0.55, 0.78, 1.0]))
    for ax, m in zip(axes, policies):
        for s in SNAPS:
            res = evaluate_policy(policies[m][s], boards)["per_depth"]
            ax.plot(list(res), list(res.values()), marker="o", color=COLORS[m], alpha=shades[s],
                    label="%s%% of training" % s if s != "0" else "~0% (first log)")
        _style(ax, "scramble depth", "solve rate")
        ax.set_title(LABELS[m])
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "snapshots_by_depth.png", dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------ GIFs ----
def _font(size):
    for p in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_board(state, caption, tile=90):
    """One frame: numbered tiles on a 3x3 grid plus a caption line."""
    img = Image.new("RGB", (3 * tile + 20, 3 * tile + 60), "white")
    d = ImageDraw.Draw(img)
    for i, t in enumerate(state):
        if t == 0:
            continue
        r, c = divmod(i, 3)
        x, y = 10 + c * tile, 10 + r * tile
        d.rounded_rectangle([x + 3, y + 3, x + tile - 3, y + tile - 3], 10, fill="#0072B2" if t != i + 1 else "#2a9d8f")
        d.text((x + tile / 2, y + tile / 2), str(t), fill="white", font=_font(40), anchor="mm")
    d.text((10 + 1.5 * tile, 3 * tile + 38), caption, fill="black", font=_font(14), anchor="mm")
    return np.array(img)


def make_gif(start, result, title, path):
    frames, s = [], start
    frames.append(draw_board(s, "%s: start" % title))
    for i, a in enumerate(result["moves"], 1):
        s = step(s, a)
        frames.append(draw_board(s, "%s: move %d" % (title, i)))
    end = "SOLVED in %d moves" % result["num_moves"] if result["solved"] else "FAILED (%s) after %d moves" % (
        result["reason"], result["num_moves"])
    durations = [0.4] * len(frames)
    frames.append(draw_board(s, "%s: %s" % (title, end)))
    durations.append(2.0)
    imageio.mimsave(path, frames, duration=durations, loop=0)


def make_gifs(policies, boards, depth):
    board = next(b for b in boards if b["depth"] == depth)
    gdir = RESULTS_DIR / "gifs"
    gdir.mkdir(exist_ok=True)
    start = board["state"]
    make_gif(start, astar.solve_result(start), "A* (optimal %d)" % board["optimal"], gdir / "astar.gif")
    for m in policies:
        for s, tag in (("0", "early"), ("100", "final")):
            make_gif(start, policies[m][s](start), "%s %s" % (LABELS[m], tag), gdir / ("%s_%s.gif" % (m, tag)))
    print("GIFs for board", start, "(depth %d, optimal %d)" % (depth, board["optimal"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gif-depth", type=int, default=20, help="scramble depth of the GIF board")
    args = ap.parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    comp = pd.read_csv(RESULTS_DIR / "comparison.csv", dtype={"depth": str})
    pb = pd.read_csv(RESULTS_DIR / "per_board.csv")
    boards = load_test_set()
    plot_curves("test_solve_rate", "solve rate on test boards", "learning_curves.png", "Learning curves")
    plot_curves("avg_reward", "average training-episode reward", "avg_reward.png", "Average reward")
    plot_by_depth(comp)
    plot_generalization(pb)
    policies = snapshot_policies()
    plot_snapshots(policies, boards)
    make_gifs(policies, boards, args.gif_depth)
    print("figures written to", RESULTS_DIR)


if __name__ == "__main__":
    main()
