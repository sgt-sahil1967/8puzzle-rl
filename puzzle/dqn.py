"""DQN for the sliding puzzle (CPU only): replay buffer, target network,
Huber loss, gradient clipping, masked actions, same reward/curriculum as Q-learning."""
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from . import common
from .common import CKPT_DIR, LOG_DIR, Curriculum
from .env import DEFAULT_SHAPE, _tables, greedy_solve, legal_moves, step

DEVICE = torch.device("cpu")  # always CPU: never MPS / CUDA
GAMMA = common.GAMMA
LOG_FIELDS = ["episode", "epsilon", "max_depth", "avg_reward", "avg_loss",
              "train_solve_rate", "test_solve_rate", "val_solve_rate", "avg_moves", "avg_optimal",
              "elapsed_s"]
CELLS = DEFAULT_SHAPE[0] * DEFAULT_SHAPE[1]


def _legal_mask(shape=DEFAULT_SHAPE):
    """mask[blank_pos] = bool array of 4: True where the action is legal."""
    _, legal = _tables(shape)
    m = np.zeros((len(legal), 4), dtype=bool)
    for pos, acts in enumerate(legal):
        m[pos, list(acts)] = True
    return m


LEGAL_MASK = _legal_mask()


class QNet(nn.Module):
    """81 (one-hot board) -> hidden layers (default 128, 128) -> 4 action values."""

    def __init__(self, hidden=(128, 128)):
        super().__init__()
        sizes = [CELLS * CELLS] + list(hidden)
        layers = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        layers.append(nn.Linear(sizes[-1], 4))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def one_hot(states):
    """(B, 9) tile values -> (B, 81) floats; unit i*9+tile is 1 if cell i holds tile."""
    return F.one_hot(states.long(), CELLS).reshape(states.shape[0], -1).float()


def masked(q, blank_mask):
    """Set the value of illegal actions to -inf so they are never chosen."""
    return q.masked_fill(~blank_mask, float("-inf"))


class ReplayBuffer:
    """Fixed-size ring buffer of (s, a, r, s', done, legal-mask of s', discount).

    For n-step learning r is the discounted sum of the next n rewards and
    `disc` is gamma^n (the factor applied to the bootstrapped value).
    """

    def __init__(self, capacity):
        self.cap, self.size, self.pos = capacity, 0, 0
        self.s = np.zeros((capacity, CELLS), np.int8)
        self.s2 = np.zeros((capacity, CELLS), np.int8)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.done = np.zeros(capacity, np.float32)
        self.disc = np.full(capacity, GAMMA, np.float32)
        self.mask2 = np.zeros((capacity, 4), bool)

    def add(self, s, a, r, s2, done, disc=GAMMA):
        i = self.pos
        self.s[i], self.a[i], self.r[i], self.s2[i], self.done[i] = s, a, r, s2, done
        self.disc[i] = disc
        self.mask2[i] = LEGAL_MASK[s2.index(0)]
        self.pos = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch, np_rng):
        idx = np_rng.integers(0, self.size, batch)
        t = torch.from_numpy
        return (t(self.s[idx]), t(self.a[idx]), t(self.r[idx]),
                t(self.s2[idx]), t(self.done[idx]), t(self.mask2[idx]), t(self.disc[idx]))


# ------------------------------------------------------------ greedy solver --
class DQNPolicy:
    """Inference-only copy of a trained network, evaluated with plain numpy.

    The first layer applied to a one-hot input just adds up 9 columns of the
    weight matrix, which makes single-board evaluation fast (no torch overhead).
    Works for any number of hidden layers.
    """

    def __init__(self, state_dict):
        w = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in state_dict.items()}
        n_layers = len(w) // 2
        self.weights = [w["layers.%d.weight" % (2 * i)] for i in range(n_layers)]
        self.biases = [w["layers.%d.bias" % (2 * i)] for i in range(n_layers)]
        self.offsets = np.arange(CELLS) * CELLS

    def q_values(self, state):
        """Action values for one board; illegal actions are -inf."""
        h = np.maximum(self.weights[0][:, self.offsets + np.asarray(state)].sum(axis=1) + self.biases[0], 0)
        for W, b in zip(self.weights[1:-1], self.biases[1:-1]):
            h = np.maximum(W @ h + b, 0)
        q = self.weights[-1] @ h + self.biases[-1]
        return np.where(LEGAL_MASK[state.index(0)], q, -np.inf)


def load_policy(path):
    """Load a snapshot / checkpoint file into a DQNPolicy."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return DQNPolicy(ck["model"])


def solve_dqn(policy, state, shape=DEFAULT_SHAPE, max_steps=100, max_visits=2):
    """Greedy rollout of the network; fails honestly, no A* fallback."""
    return greedy_solve(state, policy.q_values, shape, max_steps, max_visits)


# ----------------------------------------------------------------- training --
def _push_nstep(buf, pending, s_next, done, count):
    """Store the `count` oldest pending steps as n-step transitions ending at s_next.

    Each stored return is the discounted sum of the rewards still in `pending`
    (at most n of them); the bootstrap factor is gamma ** (number of steps used).
    """
    for _ in range(count):
        ret, disc = 0.0, 1.0
        for (_, _, r) in pending:
            ret += disc * r
            disc *= GAMMA
        s0, a0, _ = pending.popleft()
        buf.add(s0, a0, ret, s_next, float(done), disc)


def _estimate_minutes(args, net, target, opt, np_rng):
    """Time a few hundred env+learn steps and extrapolate (a rough guess)."""
    buf = ReplayBuffer(5000)
    rng = common.make_rng(0)
    s = common.scramble(10, rng)
    t0 = time.time()
    n = 600
    for i in range(n):
        a = rng.choice(legal_moves(s))
        s2 = step(s, a)
        r, d = common.reward(s, s2)
        buf.add(s, a, r, s2, d)
        with torch.no_grad():
            net(one_hot(torch.tensor([s])))
        s = s2 if not d else common.scramble(10, rng)
        if i % 4 == 0 and buf.size > 200:
            _learn_step(net, target, opt, buf.sample(args.batch, np_rng), args.double)
    per_step = (time.time() - t0) / n
    return per_step * args.episodes * 25 / 60  # assume ~25 steps per episode


def _learn_step(net, target, opt, batch, double=False):
    s, a, r, s2, done, mask2, disc = batch
    q = net(one_hot(s)).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        q2_all = target(one_hot(s2))
        if double:
            # Double DQN: the online net CHOOSES the next action, the target net EVALUATES it
            best = masked(net(one_hot(s2)), mask2).argmax(dim=1, keepdim=True)
            q2 = q2_all.gather(1, best).squeeze(1)
        else:
            q2 = masked(q2_all, mask2).max(dim=1).values
        y = r + disc * (1.0 - done) * q2  # no bootstrap after the goal
    loss = F.smooth_l1_loss(q, y)  # Huber loss
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 10.0)
    opt.step()
    return loss.item()


def train(args):
    common.ensure_dirs()
    torch.set_num_threads(1)
    shape = DEFAULT_SHAPE
    prefix = "dqn_quick" if args.quick else args.name
    log_path = LOG_DIR / (prefix + "_log.csv")
    latest = CKPT_DIR / (prefix + "_latest.pt")
    total = args.episodes
    snaps = common.snapshot_episodes(total, args.log_every)

    test_boards = common.load_test_set()
    test_states = {b["state"] for b in test_boards}
    val_boards = common.make_validation_set(test_states)
    forbidden = test_states | {b["state"] for b in val_boards}  # never start training on these
    best_val, best_episode = -1.0, 0

    rng = common.make_rng(args.seed)
    np_rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    cur = Curriculum(start_depth=3, cap=args.max_depth, window=args.window,
                     threshold=args.threshold)
    hidden = tuple(int(h) for h in args.hidden.split(","))
    net, target = QNet(hidden).to(DEVICE), QNet(hidden).to(DEVICE)
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    buf = ReplayBuffer(args.buffer)
    episode = env_steps = learn_steps = 0
    elapsed0, rows, snaps_done = 0.0, [], set()

    if args.resume and latest.exists():
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"])
        target.load_state_dict(ck["target"])
        opt.load_state_dict(ck["optimizer"])
        buf.__dict__.update(ck["buffer"])  # (checkpoints from before n-step lack "disc": default gamma is kept)
        episode, env_steps, learn_steps = ck["episode"], ck["env_steps"], ck["learn_steps"]
        elapsed0, rows, snaps_done = ck["elapsed"], ck["rows"], set(ck["snaps_done"])
        cur.load_state_dict(ck["curriculum"])
        best_val, best_episode = ck.get("best_val", -1.0), ck.get("best_episode", 0)
        rng.setstate(ck["rng_state"])
        np_rng.bit_generator.state = ck["np_rng_state"]
        torch.set_rng_state(ck["torch_rng_state"])
        print("resumed from episode %d" % episode)
    elif args.resume:
        print("no checkpoint found, starting fresh")
    else:
        # estimate on a throw-away copy so the real run is unaffected
        est_net, est_tgt = QNet(hidden), QNet(hidden)
        est_opt = torch.optim.Adam(est_net.parameters(), lr=args.lr)
        print("rough time estimate for %d episodes: ~%.0f min (progress bar shows the real ETA)" % (
            total, _estimate_minutes(args, est_net, est_tgt, est_opt, np.random.default_rng(0))))

    t_start = time.time()

    def save_checkpoint():
        common.save_atomic(latest, lambda f: torch.save({
            "model": net.state_dict(), "target": target.state_dict(),
            "optimizer": opt.state_dict(), "buffer": dict(buf.__dict__),
            "episode": episode, "env_steps": env_steps, "learn_steps": learn_steps,
            "elapsed": elapsed0 + time.time() - t_start, "rows": rows,
            "snaps_done": sorted(snaps_done), "curriculum": cur.state_dict(),
            "best_val": best_val, "best_episode": best_episode,
            "rng_state": rng.getstate(), "np_rng_state": np_rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(), "args": vars(args)}, f))

    reward_sum, wins, n_window, loss_sum, n_loss = 0.0, 0, 0, 0.0, 0
    bar = tqdm(total=total, initial=episode, desc="DQN", unit="ep")
    try:
        while episode < total:
            eps = common.epsilon_at(episode, total, args.eps_start, args.eps_end, args.eps_decay_frac)
            if args.lr_decay:  # linear decay of the learning rate to 10% by the end
                for g in opt.param_groups:
                    g["lr"] = args.lr * (1.0 - 0.9 * episode / total)
            depth = cur.sample_depth(rng)
            state, used_depth = common.sample_train_board(depth, rng, forbidden, shape)
            ep_reward, solved = 0.0, False
            pending = deque()  # last <= n steps, for n-step returns
            for _ in range(common.step_cap_for(used_depth)):
                if rng.random() < eps:
                    action = rng.choice(legal_moves(state, shape))
                else:
                    with torch.no_grad():
                        q = masked(net(one_hot(torch.tensor([state]))),
                                   torch.from_numpy(LEGAL_MASK[state.index(0)]).unsqueeze(0))
                    action = int(q.argmax())
                nxt = step(state, action, shape)
                r, done = common.reward(state, nxt, shape)
                pending.append((tuple(state), action, r))
                if done:
                    _push_nstep(buf, pending, nxt, True, len(pending))
                elif len(pending) == args.nstep:
                    _push_nstep(buf, pending, nxt, False, 1)
                ep_reward += r
                state = nxt
                env_steps += 1
                if buf.size >= args.learn_start and env_steps % args.train_freq == 0:
                    loss_sum += _learn_step(net, target, opt, buf.sample(args.batch, np_rng), args.double)
                    n_loss += 1
                    learn_steps += 1
                if args.tau > 0:
                    # soft update: target <- (1 - tau) * target + tau * online, every env step
                    with torch.no_grad():
                        for tp, op in zip(target.parameters(), net.parameters()):
                            tp.mul_(1.0 - args.tau).add_(op, alpha=args.tau)
                elif env_steps % args.target_sync == 0:
                    target.load_state_dict(net.state_dict())  # hard copy
                if done:
                    solved = True
                    break
            if pending:  # step cap reached: flush what is left, still bootstrapping
                _push_nstep(buf, pending, state, False, len(pending))
            episode += 1
            bar.update(1)
            cur.record(solved)
            reward_sum += ep_reward
            wins += solved
            n_window += 1

            if episode % args.log_every == 0 or episode == total:
                policy = DQNPolicy(net.state_dict())
                res = common.evaluate_policy(lambda s: solve_dqn(policy, s, shape), test_boards)
                val = common.evaluate_policy(lambda s: solve_dqn(policy, s, shape), val_boards)["solve_rate"]
                if val > best_val:  # keep the best network according to the VALIDATION boards
                    best_val, best_episode = val, episode
                    common.save_atomic(CKPT_DIR / ("%s_best.pt" % prefix),
                                       lambda f: torch.save({"model": net.state_dict(), "episode": episode}, f))
                rows.append({
                    "episode": episode, "epsilon": round(eps, 4), "max_depth": cur.max_depth,
                    "avg_reward": round(reward_sum / n_window, 4),
                    "avg_loss": round(loss_sum / max(1, n_loss), 5),
                    "train_solve_rate": round(wins / n_window, 4),
                    "test_solve_rate": round(res["solve_rate"], 4),
                    "val_solve_rate": round(val, 4),
                    "avg_moves": round(res["avg_moves"], 3),
                    "avg_optimal": round(res["avg_optimal"], 3),
                    "elapsed_s": round(elapsed0 + time.time() - t_start, 1)})
                common.write_log(log_path, rows, LOG_FIELDS)
                bar.set_postfix(depth=cur.max_depth, test=res["solve_rate"], loss=rows[-1]["avg_loss"])
                reward_sum, wins, n_window, loss_sum, n_loss = 0.0, 0, 0, 0.0, 0
            for name, ep in snaps.items():
                if episode >= ep and name not in snaps_done:
                    common.save_atomic(CKPT_DIR / ("%s_snap_%s.pt" % (prefix, name)),
                                       lambda f: torch.save({"model": net.state_dict(), "episode": episode}, f))
                    snaps_done.add(name)
            if episode % args.checkpoint_every == 0:
                save_checkpoint()
    except KeyboardInterrupt:
        print("\ninterrupted - saving checkpoint (resume with --resume)")
    finally:
        save_checkpoint()
        bar.close()
    print("best validation solve rate %.3f at episode %d (saved as %s_best.pt)" % (best_val, best_episode, prefix))
    print("done: %d episodes, max depth %d, %.1f min" % (
        episode, cur.max_depth, (elapsed0 + time.time() - t_start) / 60))
