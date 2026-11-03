"""A* search with the Manhattan-distance heuristic (optimal baseline)."""
import heapq
import time

from .env import DEFAULT_SHAPE, _dist_table, _tables, goal_state, manhattan

DEFAULT_NODE_LIMIT = 2_000_000


def solve(state, shape=DEFAULT_SHAPE, node_limit=DEFAULT_NODE_LIMIT):
    """Return the optimal list of actions (blank moves), or None.

    None means the search gave up: node_limit expansions reached (or the board
    is unsolvable and the whole reachable space was exhausted).
    Manhattan distance never overestimates, so the first time the goal is
    popped its path is optimal.
    """
    goal = goal_state(shape)
    if state == goal:
        return []
    next_blank, legal = _tables(shape)
    dist = _dist_table(shape)

    h0 = manhattan(state, shape)
    # heap entries: (f, h, tie-break counter, g, state, blank position)
    counter = 0
    heap = [(h0, h0, counter, 0, state, state.index(0))]
    best_g = {state: 0}
    parent = {state: None}  # state -> (previous state, action)
    expanded = 0
    while heap:
        f, h, _, g, cur, blank = heapq.heappop(heap)
        if cur == goal:
            actions = []
            while parent[cur] is not None:
                cur, action = parent[cur]
                actions.append(action)
            return actions[::-1]
        if g > best_g[cur]:
            continue  # stale heap entry
        expanded += 1
        if expanded > node_limit:
            return None
        for action in legal[blank]:
            nb = next_blank[blank][action]
            tile = cur[nb]
            board = list(cur)
            board[blank], board[nb] = tile, 0
            nxt = tuple(board)
            ng = g + 1
            if ng < best_g.get(nxt, 1 << 30):
                best_g[nxt] = ng
                parent[nxt] = (cur, action)
                # incremental heuristic: only the moved tile changes distance
                nh = h - dist[tile][nb] + dist[tile][blank]
                counter += 1
                heapq.heappush(heap, (ng + nh, nh, counter, ng, nxt, nb))
    return None


def solve_result(state, shape=DEFAULT_SHAPE, node_limit=DEFAULT_NODE_LIMIT):
    """Same result dict as env.greedy_solve, so all three methods compare easily."""
    from .env import step
    start = time.perf_counter()
    moves = solve(state, shape, node_limit)
    elapsed = (time.perf_counter() - start) * 1000.0
    if moves is None:
        return {"solved": False, "reason": "node_limit", "moves": [], "final_state": state,
                "num_moves": 0, "time_ms": elapsed}
    final = state
    for a in moves:
        final = step(final, a, shape)
    return {"solved": True, "reason": "solved", "moves": moves, "final_state": final,
            "num_moves": len(moves), "time_ms": elapsed}
