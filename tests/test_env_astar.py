"""Phase 1 checks. Run from the project root:  python -m unittest discover -s tests -v"""
import itertools
import random
import unittest

from puzzle import astar, env

SHAPES = [(2, 3), (3, 3)]


class TestEnv(unittest.TestCase):
    def test_scrambled_boards_are_solvable(self):
        rng = random.Random(0)
        for shape in SHAPES:
            for depth in range(0, 31):
                for _ in range(20):
                    s = env.scramble(depth, rng, shape)
                    self.assertTrue(env.is_solvable(s, shape))

    def test_parity_check_matches_reachability_2x3(self):
        # brute force: every permutation of a 2x3 board, solvable iff reachable
        shape = (2, 3)
        seen, frontier = {env.goal_state(shape)}, [env.goal_state(shape)]
        while frontier:
            nxt = []
            for s in frontier:
                for a in env.legal_moves(s, shape):
                    t = env.step(s, a, shape)
                    if t not in seen:
                        seen.add(t)
                        nxt.append(t)
            frontier = nxt
        for perm in itertools.permutations(range(6)):
            self.assertEqual(env.is_solvable(perm, shape), perm in seen)
        self.assertEqual(len(seen), 360)

    def test_odd_permutation_is_unsolvable(self):
        self.assertFalse(env.is_solvable((2, 1, 3, 4, 5, 6, 7, 8, 0)))

    def test_step_and_legal_moves(self):
        g = env.goal_state()
        self.assertEqual(set(env.legal_moves(g)), {env.UP, env.LEFT})
        s = env.step(g, env.UP)
        self.assertEqual(s, (1, 2, 3, 4, 5, 0, 7, 8, 6))
        self.assertEqual(env.step(s, env.DOWN), g)
        with self.assertRaises(ValueError):
            env.step(g, env.DOWN)

    def test_scramble_never_undoes_previous_move(self):
        # a depth-2 scramble can never come back to the goal
        rng = random.Random(1)
        for _ in range(200):
            self.assertNotEqual(env.scramble(2, rng), env.goal_state())


class TestAStar(unittest.TestCase):
    def test_solutions_reach_goal_and_are_short(self):
        rng = random.Random(2)
        for shape in SHAPES:
            for depth in (1, 3, 5, 8, 12, 16, 20):
                for _ in range(15):
                    s = env.scramble(depth, rng, shape)
                    moves = astar.solve(s, shape)
                    self.assertIsNotNone(moves)
                    cur = s
                    for a in moves:  # replay the moves
                        cur = env.step(cur, a, shape)
                    self.assertEqual(cur, env.goal_state(shape))
                    self.assertLessEqual(len(moves), depth)

    def test_goal_needs_zero_moves(self):
        self.assertEqual(astar.solve(env.goal_state()), [])

    def test_node_limit_returns_none(self):
        s = env.scramble(30, random.Random(3))
        self.assertIsNone(astar.solve(s, node_limit=5))


if __name__ == "__main__":
    unittest.main()
