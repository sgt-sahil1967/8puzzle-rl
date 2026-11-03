"""Extra figures for the report: equations, one-hot encoding, pipeline, system architecture, solution strip."""
import sys
sys.path.insert(0, "..")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# ---- equations as images ------------------------------------------------------
EQ = {
 "eq_q": r"$Q(s,a)\leftarrow Q(s,a)+\alpha\,\left[\,r+\gamma\,\max_{a'}Q(s',a')-Q(s,a)\,\right]$",
 "eq_shape": r"$r_t=-0.1+0.3\,\left[h(s_t)-\gamma\,h(s_{t+1})\right]\;(+10\ \mathrm{at\ the\ goal})$",
 "eq_dqn": r"$y=r+\gamma\,\max_{a'}Q_{\mathrm{target}}(s',a')\qquad L=\mathrm{Huber}\,(Q_{\mathrm{online}}(s,a)-y)$",
 "eq_ddqn": r"$y=r+\gamma\,Q_{\mathrm{target}}\!\left(s',\,\arg\max_{a'}Q_{\mathrm{online}}(s',a')\right)$",
 "eq_soft": r"$\theta_{\mathrm{target}}\leftarrow(1-\tau)\,\theta_{\mathrm{target}}+\tau\,\theta_{\mathrm{online}}\qquad\tau=0.002$",
 "eq_huber": r"$L_\delta(x)=\frac{1}{2}x^2\ \ (|x|\leq 1),\qquad L_\delta(x)=|x|-\frac{1}{2}\ \ (|x|>1)$",
 "eq_f": r"$f(n)=g(n)+h(n)\qquad h(n)=\sum_{\mathrm{tiles}}\left(|r-r_{goal}|+|c-c_{goal}|\right)$",
}
for name, tex in EQ.items():
    fig = plt.figure(figsize=(7, 0.9 if "huber" not in name else 1.2))
    fig.text(0.5, 0.5, tex, ha="center", va="center", fontsize=15)
    fig.savefig("figs/%s.png" % name, dpi=200, transparent=False, facecolor="white", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

# ---- one-hot encoding ------------------------------------------------------------
state = (2, 8, 3, 1, 6, 4, 7, 0, 5)
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1, 2.4]})
a.set_xlim(0, 3); a.set_ylim(0, 3); a.invert_yaxis(); a.axis("off"); a.set_aspect("equal")
for i, t in enumerate(state):
    r, c = divmod(i, 3)
    a.add_patch(Rectangle((c, r), 1, 1, fc="#d0d0d0" if t == 0 else "#a8cde8", ec="black"))
    a.text(c + .5, r + .5, "" if t == 0 else str(t), ha="center", va="center", fontsize=18)
    a.text(c + .08, r + .2, str(i), fontsize=7, color="#444")
a.set_title("Board (cell numbers in corners)", fontsize=10)
grid = np.zeros((9, 9))
for i, t in enumerate(state): grid[i, t] = 1
b.imshow(grid, cmap="Blues", aspect="equal", vmin=0, vmax=1.3)
b.set_xticks(range(9)); b.set_yticks(range(9)); b.set_xticklabels(["blank"] + [str(i) for i in range(1, 9)], fontsize=8)
b.set_yticklabels(["cell %d" % i for i in range(9)], fontsize=8); b.xaxis.tick_top()
b.set_title("One-hot matrix (9 x 9) flattened into 81 inputs; exactly nine 1s", fontsize=10, pad=24)
for i in range(10): b.axhline(i - .5, color="#888", lw=.4); b.axvline(i - .5, color="#888", lw=.4)
fig.tight_layout(); fig.savefig("figs/onehot.png", dpi=150); plt.close(fig)

# ---- boxes-and-arrows helper ---------------------------------------------------------
def flow(fname, boxes, arrows, size, colors=None):
    fig, ax = plt.subplots(figsize=size); ax.axis("off")
    ax.set_xlim(0, size[0] * 10); ax.set_ylim(0, size[1] * 10)
    for name, (x, y, w, h, text, col) in boxes.items():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4", fc=col, ec="#333", lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)
    for arr in arrows:
        a_, b_, side = arr[:3]; both = len(arr) > 3
        A, B = boxes[a_], boxes[b_]
        def pt(bx, s):
            x, y, w, h = bx[:4]
            return {"r": (x + w + .4, y + h / 2), "l": (x - .4, y + h / 2), "t": (x + w / 2, y + h + .4), "b": (x + w / 2, y - .4)}[s]
        sa, sb = side
        ax.add_patch(FancyArrowPatch(pt(A, sa), pt(B, sb), arrowstyle="<|-|>" if both else "-|>", mutation_scale=13, color="#222"))
    fig.tight_layout(); fig.savefig("figs/%s.png" % fname, dpi=150); plt.close(fig)

G, B_, O, Y, P = "#cfe8d5", "#cfe0f3", "#f8d9b8", "#fbf0b8", "#e6d4ee"
flow("pipeline", {
  "scr": (1, 21, 18, 11, "Reverse scramble\n(curriculum depth,\nnot a test board)", B_),
  "env": (22, 21, 18, 11, "Environment step\n(move the blank)", B_),
  "rew": (43, 21, 18, 11, "Reward: +10 goal,\n-0.1 per step,\nManhattan shaping", Y),
  "lrn": (64, 21, 18, 11, "Learn: Q-table\nupdate or DQN\ngradient step", O),
  "log": (64, 3, 18, 11, "Every 500 episodes:\ntest-set check,\nwrite CSV log", P),
  "ckp": (43, 3, 18, 11, "Checkpoints and\nsnapshots (resume\nwith --resume)", P),
  "cur": (1, 3, 18, 11, "Curriculum:\nsolve rate >= 80%\n-> depth + 1", G),
}, [("scr", "env", "rl"), ("env", "rew", "rl"), ("rew", "lrn", "rl"), ("lrn", "log", "bt"), ("log", "ckp", "lr"), ("ckp", "cur", "lr"), ("cur", "scr", "tb")], (8.3, 3.5))
flow("system", {
  "usr": (1, 17, 15, 12, "User's browser\n(HTML, CSS,\nJavaScript)", B_),
  "app": (24, 17, 15, 12, "Flask server\napp.py\n127.0.0.1 only", O),
  "q": (50, 36, 19, 8, "Q-table\nq_snap_100.pkl", G),
  "d1": (50, 25, 19, 8, "DQN v1\ndqn_snap_100.pt", G),
  "d2": (50, 14, 19, 8, "DQN v2\ndqn2_snap_100.pt", G),
  "as": (50, 3, 19, 8, "A* search\n(Manhattan)", Y),
  "res": (24, 1, 15, 9, "results/\ncomparison.csv\nPNG plots", P),
}, [("usr", "app", "rl", 1), ("app", "q", "rl"), ("app", "d1", "rl"), ("app", "d2", "rl"), ("app", "as", "rl"), ("app", "res", "bt")], (8.3, 4.6))

# ---- network diagrams (report version, larger text) ---------------------------------
def net(ax, title, hidden, note, col):
    layers = [("Input\n81 units", "one-hot board")] + [("Hidden %d\n%d units" % (i + 1, h), "Linear + ReLU") for i, h in enumerate(hidden)] + [("Output\n4 units", "Q(up) Q(down)\nQ(left) Q(right)"), ("Mask +\nargmax", "illegal moves\n= -inf")]
    w, gap = 1.7, 0.5
    for i, (nm, sub) in enumerate(layers):
        x = i * (w + gap); fc = "#e8e8e8" if i in (0, len(layers) - 1) else col
        ax.add_patch(FancyBboxPatch((x, 0), w, 2.2, boxstyle="round,pad=0.04", fc=fc, ec="#333", lw=1))
        ax.text(x + w / 2, 1.5, nm, ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(x + w / 2, 0.55, sub, ha="center", va="center", fontsize=7.5)
        if i < len(layers) - 1:
            ax.add_patch(FancyArrowPatch((x + w + .04, 1.1), (x + w + gap - .04, 1.1), arrowstyle="-|>", mutation_scale=11, color="#333"))
    ax.set_xlim(-.2, len(layers) * (w + gap)); ax.set_ylim(-1.1, 3.0); ax.axis("off")
    ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left"); ax.text(0, -0.65, note, fontsize=8, va="center")
fig, axes = plt.subplots(2, 1, figsize=(8.3, 4.9))
net(axes[0], "DQN v1: 81 -> 128 -> 128 -> 4  (27,524 weights)", [128, 128], "Hard target copy every 500 steps; Adam 3e-4; batch 128; gradient step every 8 env. steps; Huber loss; clip 10.", "#f6c9a8")
net(axes[1], "DQN v2: 81 -> 256 -> 256 -> 4  (87,812 weights)", [256, 256], "Double DQN; soft target (tau 0.002); Adam 3e-4 decayed to 10%; gradient step every 4 env. steps; best net on validation boards.", "#a8dcc8")
fig.tight_layout(); fig.savefig("figs/networks.png", dpi=170); plt.close(fig)

# ---- solution strip ------------------------------------------------------------------
from puzzle import astar, qlearn, dqn
from puzzle.common import load_test_set, CKPT_DIR
from puzzle.env import step
from puzzle.visualize import draw_board
b = next(x for x in load_test_set() if x["depth"] == 20)
res = astar.solve_result(b["state"])
frames, s = [b["state"]], b["state"]
for a in res["moves"]: s = step(s, a); frames.append(s)
idx = np.linspace(0, len(frames) - 1, 6).astype(int)
fig, axes = plt.subplots(1, 6, figsize=(12, 2.4))
for ax, i in zip(axes, idx):
    ax.imshow(draw_board(frames[i], "after %d moves" % i)); ax.axis("off")
fig.suptitle("A* solving the depth-20 test board (optimal length %d)" % len(res["moves"]), fontsize=10)
fig.tight_layout(); fig.savefig("figs/strip_astar.png", dpi=150); plt.close(fig)
