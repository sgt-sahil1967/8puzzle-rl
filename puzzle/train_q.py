"""CLI for tabular Q-learning.   python -m puzzle.train_q --quick
                                  python -m puzzle.train_q --episodes 600000 --resume"""
import argparse

from . import qlearn


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=600_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from the latest checkpoint")
    ap.add_argument("--quick", action="store_true", help="~1 minute smoke test (writes q_quick_* files)")
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--checkpoint-every", type=int, default=10_000)
    ap.add_argument("--eps-start", type=float, default=1.0)
    ap.add_argument("--eps-end", type=float, default=0.05)
    ap.add_argument("--eps-decay-frac", type=float, default=0.6,
                    help="fraction of training over which epsilon decays")
    ap.add_argument("--max-depth", type=int, default=32, help="curriculum depth cap")
    ap.add_argument("--window", type=int, default=300, help="rolling window for curriculum")
    ap.add_argument("--threshold", type=float, default=0.8, help="solve rate needed to deepen")
    args = ap.parse_args()
    if args.quick:
        args.episodes, args.checkpoint_every = 150_000, 25_000
    qlearn.train(args)


if __name__ == "__main__":
    main()
