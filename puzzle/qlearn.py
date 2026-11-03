"""Tabular Q-learning for the sliding puzzle, with a growing-depth curriculum.

The Q-table is a dict {state: np.array of 4 values}. Entries for moves that are
illegal in that state are -inf, so max/argmax automatically ignore them
(this is the "masking of illegal actions").
"""
import time

import numpy as np
from tqdm import tqdm

from . import common
from .common import CKPT_DIR, LOG_DIR, Curriculum
from .env import DEFAULT_SHAPE, _tables, greedy_solve, legal_moves, step

ALPHA = 0.3   # learning rate
GAMMA = common.GAMMA
LOG_FIELDS = ["episode", "epsilon", "max_depth", "avg_reward", "table_size",
              "train_solve_rate", "test_solve_rate", "avg_moves", "avg_optimal",
              "elapsed_s"]


def _template(shape):
    """For each blank position: zeros with -inf on the illegal actions."""
    _, legal = _tables(shape)
    out = []
    for pos_legal in legal:
        t = np.full(4, -np.inf, dtype=np.float32)
        t[list(pos_legal)] = 0.0
        out.append(t)
    return out


def get_row(Q, state, templates):
    """Q-values of a state, creating a fresh zero row (illegal = -inf) if new."""
    row = Q.get(state)
    if row is None:
        row = templates[state.index(0)].copy()
        Q[state] = row
    return row


def choose_action(row, state, eps, rng, shape=DEFAULT_SHAPE):
    """Epsilon-greedy over legal actions; greedy ties broken at random."""
    if rng.random() < eps:
        return rng.choice(legal_moves(state, shape))
    best = np.flatnonzero(row == row.max())
    return int(best[0]) if len(best) == 1 else int(rng.choice(list(best)))


# ------------------------------------------------------------ greedy solver --
def solve_q(Q, state, shape=DEFAULT_SHAPE, max_steps=100, max_visits=2):
    """Greedy rollout of a Q-table. Fails honestly (see env.greedy_solve);
    a state that is not in the table counts as failure "unseen_state"."""
    return greedy_solve(state, Q.get, shape, max_steps, max_visits)


def load_q_table(path):
    """Load the Q dict from a snapshot or checkpoint file."""
    return common.load_pickle(path)["Q"]


# ----------------------------------------------------------------- training --
def train(args):
    common.ensure_dirs()
    shape = DEFAULT_SHAPE
    prefix = "q_quick" if args.quick else "q"
    log_path = LOG_DIR / (prefix + "_log.csv")
    latest = CKPT_DIR / (prefix + "_latest.pkl")
    total = args.episodes
    snaps = common.snapshot_episodes(total, args.log_every)

    test_boards = common.load_test_set()
    forbidden = {b["state"] for b in test_boards}  # never start training on these
    templates = _template(shape)

    rng = common.make_rng(args.seed)
    cur = Curriculum(start_depth=3, cap=args.max_depth, window=args.window,
                     threshold=args.threshold)
    Q, episode, elapsed0, rows, snaps_done = {}, 0, 0.0, [], set()

    if args.resume and latest.exists():
        ck = common.load_pickle(latest)
        Q, episode, elapsed0 = ck["Q"], ck["episode"], ck["elapsed"]
        rows, snaps_done = ck["rows"], set(ck["snaps_done"])
        cur.load_state_dict(ck["curriculum"])
        rng.setstate(ck["rng_state"])
        print("resumed from episode %d (%d table entries)" % (episode, len(Q)))
    elif args.resume:
        print("no checkpoint found, starting fresh")

    def save_checkpoint():
        common.save_pickle(latest, {
            "Q": Q, "episode": episode, "elapsed": elapsed0 + time.time() - t_start,
            "rows": rows, "snaps_done": sorted(snaps_done),
            "curriculum": cur.state_dict(), "rng_state": rng.getstate(), "args": vars(args)})

    t_start = time.time()
    reward_sum, train_wins, n_window = 0.0, 0, 0
    bar = tqdm(total=total, initial=episode, desc="Q-learning", unit="ep")
    try:
        while episode < total:
            eps = common.epsilon_at(episode, total, args.eps_start, args.eps_end, args.eps_decay_frac)
            depth = cur.sample_depth(rng)
            state, used_depth = common.sample_train_board(depth, rng, forbidden, shape)
            ep_reward, solved = 0.0, False
            for _ in range(common.step_cap_for(used_depth)):
                row = get_row(Q, state, templates)
                action = choose_action(row, state, eps, rng, shape)
                nxt = step(state, action, shape)
                r, done = common.reward(state, nxt, shape)
                ep_reward += r
                # Q-learning update; no bootstrapping from the terminal goal state
                target = r if done else r + GAMMA * get_row(Q, nxt, templates).max()
                row[action] += ALPHA * (target - row[action])
                state = nxt
                if done:
                    solved = True
                    break
            episode += 1
            bar.update(1)
            cur.record(solved)
            reward_sum += ep_reward
            train_wins += solved
            n_window += 1

            if episode % args.log_every == 0 or episode == total:
                res = common.evaluate_policy(lambda s: solve_q(Q, s, shape), test_boards)
                rows.append({
                    "episode": episode, "epsilon": round(eps, 4), "max_depth": cur.max_depth,
                    "avg_reward": round(reward_sum / n_window, 4), "table_size": len(Q),
                    "train_solve_rate": round(train_wins / n_window, 4),
                    "test_solve_rate": round(res["solve_rate"], 4),
                    "avg_moves": round(res["avg_moves"], 3),
                    "avg_optimal": round(res["avg_optimal"], 3),
                    "elapsed_s": round(elapsed0 + time.time() - t_start, 1)})
                common.write_log(log_path, rows, LOG_FIELDS)
                bar.set_postfix(depth=cur.max_depth, test=res["solve_rate"], table=len(Q))
                reward_sum, train_wins, n_window = 0.0, 0, 0
            for name, ep in snaps.items():
                if episode >= ep and name not in snaps_done:
                    common.save_pickle(CKPT_DIR / ("%s_snap_%s.pkl" % (prefix, name)),
                                       {"Q": Q, "episode": episode, "shape": shape})
                    snaps_done.add(name)
            if episode % args.checkpoint_every == 0:
                save_checkpoint()
    except KeyboardInterrupt:
        print("\ninterrupted - saving checkpoint (resume with --resume)")
    finally:
        save_checkpoint()
        bar.close()
    print("done: %d episodes, %d table entries, max depth %d, %.1f min" % (
        episode, len(Q), cur.max_depth, (elapsed0 + time.time() - t_start) / 60))
