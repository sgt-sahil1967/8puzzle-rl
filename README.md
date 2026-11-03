# 8-puzzle solver: Q-learning vs DQN vs A*

College deep-learning mini project. Three solvers for the 3x3 sliding puzzle, a fixed test set to
compare them, plots/GIFs, and a Flask demo site. CPU only (torch.device("cpu")).

## Install and run (from this folder, in this order)

Python 3.9+ (tested on 3.9).

| step | command | approx. time (MacBook Air) |
|---|---|---|
| install | `pip install -r requirements.txt` | 1-3 min |
| unit tests | `python -m unittest discover -s tests` | 1 s |
| (test set, already created) | `python -m puzzle.evaluate --make-test-set` | 1 s |
| smoke test Q | `python -m puzzle.train_q --quick` | ~1.5 min |
| smoke test DQN | `python -m puzzle.train_dqn --quick` | ~0.5 min |
| train Q-learning | `python -m puzzle.train_q` | ~3 min (600k episodes) |
| train DQN (v1) | `python -m puzzle.train_dqn` | ~10-12 min (80k episodes) |
| train DQN v2 | `python -m puzzle.train_dqn --name dqn2 --episodes 60000 --hidden 256,256 --double --tau 0.002 --train-freq 4 --lr-decay` | ~20 min |
| evaluate | `python -m puzzle.evaluate` | ~10 s |
| plots + GIFs | `python -m puzzle.visualize` | ~1 min |
| web demo | `python webapp/app.py` then open http://127.0.0.1:5000 | instant |

Quick runs write `*_quick_*` files, so they never overwrite the real models.
Training is resumable: stop any time (Ctrl-C saves a checkpoint; periodic checkpoints are
written atomically anyway) and continue with `--resume`, e.g. `python -m puzzle.train_dqn --resume`.
Without `--resume` a run starts from scratch. Seeds: `--seed N`. `--help` lists all hyper-parameters.

## What each file does

* `puzzle/env.py` - state (tuple of 9, 0 = blank), actions (0 up, 1 down, 2 left, 3 right = blank moves),
  `legal_moves`, `step`, `scramble` (reverse scramble from the goal, so boards are always solvable),
  `is_solvable` (parity), Manhattan distance, and `greedy_solve` (the shared greedy rollout with step cap
  and loop detection - fails honestly, never falls back to A*). Board size is a `shape` parameter (e.g. `(2, 3)`).
* `puzzle/astar.py` - A* with Manhattan distance and a node limit.
* `puzzle/common.py` - helpers shared by both learners: reward (+10 goal, -0.1/step, potential-based
  Manhattan shaping), the depth curriculum, test-set loading, evaluation, logging, atomic checkpoint saving.
* `puzzle/qlearn.py`, `puzzle/train_q.py` - tabular Q-learning (dict state -> 4 values, illegal moves = -inf).
* `puzzle/dqn.py`, `puzzle/train_dqn.py` - DQN: one-hot board (81) -> hidden layers -> 4, replay buffer (100k),
  target network, Adam, Huber loss, gradient clipping, masked actions. v1 = 128,128 with defaults.
  v2 adds optional flags: `--double` (Double DQN), `--tau` (soft target update), `--lr-decay`, `--hidden`,
  `--nstep`. It also saves `<name>_best.pt`, the best network on a separate validation set.
* `puzzle/evaluate.py` - makes the test set and compares the three methods.
* `puzzle/visualize.py` - the PNG plots and GIFs.
* `webapp/` - Flask app (`app.py`, `templates/index.html`, `static/`).
* `tests/` - unit tests (scrambles are solvable, A* replays to the goal, A* length <= scramble depth).
* Outputs: `puzzle/logs/*.csv`, `puzzle/checkpoints/` (latest + snapshots at ~0/25/50/100% of training),
  `puzzle/results/` (test_set.json, comparison.csv/.md, per_board.csv, PNGs, `gifs/`, summary.md).

## How it works (short version)

* **Test set**: 200 boards per scramble depth in [1, 3, 5, 8, 12, 16, 20, 25, 30] with fixed seed 12345 and
  the A* optimal length stored. Training never *starts* from these boards (see caveats in `results/summary.md`).
* **Curriculum**: scramble depth is drawn from 1..max_depth; max_depth starts at 3 and goes up by one whenever the
  solve rate over the last 300 training episodes reaches 80% (cap 32). Episode step cap = max(12, 4 x depth).
* **Greedy solver**: follow argmax Q; stop on goal, step cap (100), or a state visited a third time (loop).
  Q-learning also fails if it reaches a state missing from its table ("unseen_state").
* **Snapshots**: the "~0%" snapshot is taken after the first 500 episodes (an untrained agent can do nothing).

## Results (single run, seed 0)

| method | solve rate | avg moves (solved) | extra moves vs A* | avg time |
|---|---|---|---|---|
| A* | 100.0% | 11.07 | 0.00 | 0.21 ms |
| Q-learning | 96.7% | 12.81 | 2.13 | 0.01 ms |
| DQN (v1) | 79.3% | 9.44* | 0.39 | 0.10 ms |
| DQN v2 | 88.4% | 10.18* | 0.28 | 0.17 ms |

Solve rate by scramble depth:

| depth | 1 | 3 | 5 | 8 | 12 | 16 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|---|---|
| A* | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Q-learning | 100 | 100 | 100 | 100 | 100 | 100 | 95.5 | 91.5 | 83.5 |
| DQN (v1) | 100 | 100 | 100 | 97 | 90 | 73 | 58.5 | 52 | 43 |
| DQN v2 | 100 | 100 | 100 | 100 | 98 | 88.5 | 75 | 72 | 62.5 |

\* lower than A* only because the DQNs solve mostly the easy boards. Details and honest discussion:
`puzzle/results/summary.md`. Full tables: `puzzle/results/comparison.md`.

## Web demo

Upload an image (png/jpg/webp, max 5 MB; processed in memory only, never saved), choose a difficulty,
Scramble, then use **Watch AI**, **Compare** or **Play** (click a tile or use arrow keys; Hint = next A* move).
It loads `puzzle/checkpoints/q_snap_100.pkl`, `dqn_snap_100.pt` and `dqn2_snap_100.pt` at startup; if missing, it says so and A* still works.
Runs on 127.0.0.1 only, debug off (`--debug`, `--port N` available).
