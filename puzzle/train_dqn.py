"""CLI for DQN.   python -m puzzle.train_dqn --quick
                  python -m puzzle.train_dqn --episodes 80000 --resume"""
import argparse

from . import dqn


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=80_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from the latest checkpoint")
    ap.add_argument("--name", default="dqn", help="file prefix for logs/checkpoints")
    ap.add_argument("--quick", action="store_true", help="~1 minute smoke test (writes dqn_quick_* files)")
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--checkpoint-every", type=int, default=2000)
    ap.add_argument("--eps-start", type=float, default=1.0)
    ap.add_argument("--eps-end", type=float, default=0.05)
    ap.add_argument("--eps-decay-frac", type=float, default=0.6)
    ap.add_argument("--max-depth", type=int, default=32)
    ap.add_argument("--window", type=int, default=300)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--buffer", type=int, default=100_000)
    ap.add_argument("--tau", type=float, default=0.0, help="soft target update rate (0 = hard copy every --target-sync steps)")
    ap.add_argument("--lr-decay", action="store_true", help="decay learning rate linearly to 10%% of its start")
    ap.add_argument("--hidden", default="128,128", help="hidden layer sizes, e.g. 256,256,256")
    ap.add_argument("--nstep", type=int, default=1, help="n-step returns (1 = ordinary DQN)")
    ap.add_argument("--double", action="store_true", help="Double DQN targets")
    ap.add_argument("--learn-start", type=int, default=2000)
    ap.add_argument("--train-freq", type=int, default=8, help="env steps per gradient step")
    ap.add_argument("--target-sync", type=int, default=500, help="env steps between target updates")
    args = ap.parse_args()
    if args.quick:
        args.episodes, args.checkpoint_every, args.log_every = 12_000, 4000, 500
    dqn.train(args)


if __name__ == "__main__":
    main()
