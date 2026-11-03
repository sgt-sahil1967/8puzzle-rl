"""Sliding-tile puzzle environment (default 3x3 "8-puzzle").

A state is a tuple of ints, row by row, with 0 meaning the blank.
The goal is (1, 2, ..., n-1, 0): tiles in order, blank at bottom-right.

Actions move the BLANK:  0 = up, 1 = down, 2 = left, 3 = right.
The board size is a parameter (`shape=(rows, cols)`) so everything can be
tested on a small 2x3 board first.
"""
import time
from functools import lru_cache

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_NAMES = ("up", "down", "left", "right")
_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))
OPPOSITE = (DOWN, UP, RIGHT, LEFT)  # the move that undoes each action
DEFAULT_SHAPE = (3, 3)


@lru_cache(maxsize=None)
def _tables(shape):
    """Precompute, for every blank position, where each action sends it.

    Returns (next_blank, legal) where next_blank[pos][action] is the new blank
    index (or -1 if the move would leave the board) and legal[pos] is the tuple
    of legal actions when the blank is at pos.
    """
    rows, cols = shape
    next_blank = []
    for pos in range(rows * cols):
        r, c = divmod(pos, cols)
        row = []
        for dr, dc in _DELTAS:
            nr, nc = r + dr, c + dc
            inside = 0 <= nr < rows and 0 <= nc < cols
            row.append(nr * cols + nc if inside else -1)
        next_blank.append(tuple(row))
    legal = tuple(tuple(a for a in range(4) if nb[a] >= 0) for nb in next_blank)
    return tuple(next_blank), legal


def goal_state(shape=DEFAULT_SHAPE):
    n = shape[0] * shape[1]
    return tuple(range(1, n)) + (0,)


def is_goal(state, shape=DEFAULT_SHAPE):
    return state == goal_state(shape)


def legal_moves(state, shape=DEFAULT_SHAPE):
    """Tuple of actions that can legally be played (blank must stay on board)."""
    return _tables(shape)[1][state.index(0)]


def step(state, action, shape=DEFAULT_SHAPE):
    """Move the blank in direction `action`; return the new state.

    Raises ValueError for an illegal move.
    """
    next_blank, _ = _tables(shape)
    blank = state.index(0)
    target = next_blank[blank][action]
    if target < 0:
        raise ValueError("illegal move %s from %s" % (ACTION_NAMES[action], state))
    board = list(state)
    board[blank], board[target] = board[target], 0  # tile slides into the blank
    return tuple(board)


def scramble(n_moves, rng, shape=DEFAULT_SHAPE):
    """Make a solvable board: start at the goal, play n random legal moves.

    A move is never the immediate undo of the previous one. Random permutations
    are NOT used because half of all permutations cannot be solved.
    `rng` is a random.Random instance (so seeds are explicit).
    """
    state = goal_state(shape)
    prev = None
    for _ in range(n_moves):
        options = [a for a in legal_moves(state, shape) if prev is None or a != OPPOSITE[prev]]
        action = rng.choice(options)
        state = step(state, action, shape)
        prev = action
    return state


def is_solvable(state, shape=DEFAULT_SHAPE):
    """Parity test: can this arrangement reach the goal?

    Count inversions (pairs of tiles out of order, ignoring the blank).
    Odd width: solvable iff inversions are even.
    Even width: a vertical move changes both the inversion count and the blank's
    row by an odd amount, so (inversions + blank_row) keeps a fixed parity.
    """
    rows, cols = shape
    if sorted(state) != list(range(rows * cols)):
        return False
    tiles = [t for t in state if t != 0]
    inversions = sum(1 for i in range(len(tiles)) for j in range(i + 1, len(tiles)) if tiles[i] > tiles[j])
    if cols % 2 == 1:
        return inversions % 2 == 0
    return (inversions + state.index(0) // cols) % 2 == (rows - 1) % 2


@lru_cache(maxsize=None)
def _dist_table(shape):
    """dist[tile][pos] = Manhattan distance from pos to the tile's goal cell."""
    rows, cols = shape
    table = [[0] * (rows * cols) for _ in range(rows * cols)]
    for tile in range(1, rows * cols):
        gr, gc = divmod(tile - 1, cols)
        for pos in range(rows * cols):
            r, c = divmod(pos, cols)
            table[tile][pos] = abs(r - gr) + abs(c - gc)
    return table


def manhattan(state, shape=DEFAULT_SHAPE):
    """Sum over tiles (not the blank) of the grid distance to the goal cell."""
    dist = _dist_table(shape)
    return sum(dist[t][p] for p, t in enumerate(state) if t != 0)


# ---------------------------------------------------------------------------
# Greedy rollout shared by Q-learning and DQN
# ---------------------------------------------------------------------------
def greedy_solve(state, q_fn, shape=DEFAULT_SHAPE, max_steps=100, max_visits=2):
    """Follow argmax of q_fn(state) until solved, or fail honestly.

    q_fn(state) returns a length-4 array of action values with illegal moves
    set to -inf, or None if the agent has no knowledge of the state.
    There is NO fallback to A*: if the policy fails the result says why.

    Failure reasons:  "step_cap"      -> used up max_steps
                      "loop"          -> some state was visited more than
                                         max_visits times (the policy cycles)
                      "unseen_state"  -> q_fn had no values for the state
    Returns a dict: solved, reason, moves (actions played), final_state,
    num_moves, time_ms.
    """
    start = time.perf_counter()
    goal = goal_state(shape)
    visits = {state: 1}
    moves = []
    reason = "step_cap"
    while True:
        if state == goal:
            reason = "solved"
            break
        if len(moves) >= max_steps:
            break
        q = q_fn(state)
        if q is None:
            reason = "unseen_state"
            break
        action = int(q.argmax())  # ties -> lowest action index (deterministic)
        state = step(state, action, shape)
        moves.append(action)
        visits[state] = visits.get(state, 0) + 1
        if visits[state] > max_visits:
            reason = "loop"
            break
    return {
        "solved": reason == "solved",
        "reason": reason,
        "moves": moves,
        "final_state": state,
        "num_moves": len(moves),
        "time_ms": (time.perf_counter() - start) * 1000.0,
    }
