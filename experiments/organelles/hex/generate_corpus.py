#!/usr/bin/env python3
# Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
# MIT License — see LICENSE file for details.
"""
Generate training corpora for the Hex 7x7 multi-organelle demo.

Hex: 7x7 board, X connects top-bottom, O connects left-right.
Board: 49-char string, '.'=empty, 'X'=black, 'O'=white.
Player output: "RrCc" e.g. "R3C4".

Uses Monte Carlo evaluation to rank moves.
Topological features (xg, xd, og, od, xb) enrich prompts with structural info.
"""

import random
import sys
import os
from multiprocessing import Pool, cpu_count

SIZE = 7
BOARD_SIZE = SIZE * SIZE
EMPTY = '.'
BLACK = 'X'
WHITE = 'O'

# Hex neighbours: 6 directions
HEX_DIRS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]


def idx(r, c):
    return r * SIZE + c


def board_to_str(board):
    return ''.join(board)


def get_neighbours(r, c):
    """Get valid hex neighbours."""
    result = []
    for dr, dc in HEX_DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < SIZE and 0 <= nc < SIZE:
            result.append((nr, nc))
    return result


def get_empty_cells(board):
    return [(r, c) for r in range(SIZE) for c in range(SIZE) if board[idx(r, c)] == EMPTY]


def check_connection(board, player):
    """Check if player has a connected path.
    X: top row to bottom row. O: left col to right col."""
    visited = set()
    queue = []

    if player == BLACK:
        for c in range(SIZE):
            if board[idx(0, c)] == player:
                queue.append((0, c))
                visited.add((0, c))
    else:
        for r in range(SIZE):
            if board[idx(r, 0)] == player:
                queue.append((r, 0))
                visited.add((r, 0))

    while queue:
        r, c = queue.pop(0)
        if player == BLACK and r == SIZE - 1:
            return True
        if player == WHITE and c == SIZE - 1:
            return True
        for nr, nc in get_neighbours(r, c):
            if (nr, nc) not in visited and board[idx(nr, nc)] == player:
                visited.add((nr, nc))
                queue.append((nr, nc))

    return False


def count_groups(board, player):
    """Count connected components (Betti-0) of player's stones."""
    visited = set()
    groups = 0
    for r in range(SIZE):
        for c in range(SIZE):
            if board[idx(r, c)] == player and (r, c) not in visited:
                groups += 1
                # BFS to mark entire component
                q = [(r, c)]
                visited.add((r, c))
                while q:
                    cr, cc = q.pop(0)
                    for nr, nc in get_neighbours(cr, cc):
                        if (nr, nc) not in visited and board[idx(nr, nc)] == player:
                            visited.add((nr, nc))
                            q.append((nr, nc))
    return groups


def shortest_edge_distance(board, player):
    """BFS shortest distance from player's stones to their target edge.
    X targets bottom row (r=SIZE-1). O targets right col (c=SIZE-1).
    Distance counts empty cells to cross. Returns 0 if connected, 99 if no stones."""
    stones = [(r, c) for r in range(SIZE) for c in range(SIZE) if board[idx(r, c)] == player]
    if not stones:
        return 99

    # BFS from target edge backward through empty + friendly cells
    visited = {}
    q = []
    if player == BLACK:  # Target: bottom row
        for c in range(SIZE):
            q.append((SIZE - 1, c, 0))
            visited[(SIZE - 1, c)] = 0
    else:  # Target: right col
        for r in range(SIZE):
            q.append((r, SIZE - 1, 0))
            visited[(r, SIZE - 1)] = 0

    while q:
        r, c, d = q.pop(0)
        for nr, nc in get_neighbours(r, c):
            if (nr, nc) in visited:
                continue
            cell = board[idx(nr, nc)]
            if cell == player:
                visited[(nr, nc)] = d  # No cost to traverse own stone
                q.append((nr, nc, d))
            elif cell == EMPTY:
                visited[(nr, nc)] = d + 1
                q.append((nr, nc, d + 1))

    # Find minimum distance from any player stone to target edge
    min_dist = 99
    for r, c in stones:
        if (r, c) in visited:
            min_dist = min(min_dist, visited[(r, c)])
    return min_dist


def count_bridges(board, player):
    """Count empty cells adjacent to 2+ friendly stones (potential bridge connections)."""
    bridges = 0
    for r in range(SIZE):
        for c in range(SIZE):
            if board[idx(r, c)] == EMPTY:
                friendly = sum(1 for nr, nc in get_neighbours(r, c) if board[idx(nr, nc)] == player)
                if friendly >= 2:
                    bridges += 1
    return bridges


def count_virtual_connections(board, player):
    """Count virtual connections (Hex bridges).
    A bridge: two friendly stones that share exactly 2 common empty hex neighbours.
    If opponent plays one, player responds on the other to maintain connection.
    Returns (num_bridges, num_bridge_cells) — bridge cells are the empty cells in bridges."""
    stones = [(r, c) for r in range(SIZE) for c in range(SIZE) if board[idx(r, c)] == player]
    bridges = 0
    bridge_cells = set()
    seen_pairs = set()
    for i, (r1, c1) in enumerate(stones):
        n1 = set(get_neighbours(r1, c1))
        for r2, c2 in stones[i+1:]:
            pair = (min(idx(r1,c1), idx(r2,c2)), max(idx(r1,c1), idx(r2,c2)))
            if pair in seen_pairs:
                continue
            # Check if stones are at distance 2 (not adjacent) — classic bridge pattern
            n2 = set(get_neighbours(r2, c2))
            common_empty = [(r, c) for r, c in (n1 & n2) if board[idx(r, c)] == EMPTY]
            if len(common_empty) == 2:
                bridges += 1
                bridge_cells.update(common_empty)
                seen_pairs.add(pair)
    return bridges, len(bridge_cells)


def compute_topo_features(board, player):
    """Compute topological features for the board from player's perspective."""
    opp = WHITE if player == BLACK else BLACK
    return {
        'xg': count_groups(board, player),
        'xd': shortest_edge_distance(board, player),
        'og': count_groups(board, opp),
        'od': shortest_edge_distance(board, opp),
        'xb': count_bridges(board, player),
    }


def monte_carlo_eval(board, player, num_sims=50):
    """Evaluate a position by random playout."""
    opp = WHITE if player == BLACK else BLACK
    wins = 0
    empties = get_empty_cells(board)

    for _ in range(num_sims):
        sim_board = list(board)
        shuffled = list(empties)
        random.shuffle(shuffled)
        turn = player
        for r, c in shuffled:
            sim_board[idx(r, c)] = turn
            turn = opp if turn == player else player

        if check_connection(sim_board, player):
            wins += 1

    return wins / num_sims


def find_best_move(board, player, num_sims=100):
    """Find best move using Monte Carlo evaluation."""
    empties = get_empty_cells(board)
    if not empties:
        return None

    best_score = -1
    best_move = empties[0]
    opp = WHITE if player == BLACK else BLACK

    for r, c in empties:
        new_board = list(board)
        new_board[idx(r, c)] = player
        score = monte_carlo_eval(new_board, player, num_sims)
        if score > best_score:
            best_score = score
            best_move = (r, c)

    return best_move


# --- MCTS Implementation ---

import math

class MCTSNode:
    __slots__ = ['move', 'parent', 'children', 'wins', 'visits', 'untried']

    def __init__(self, move=None, parent=None, untried_moves=None):
        self.move = move
        self.parent = parent
        self.children = []
        self.wins = 0.0
        self.visits = 0
        self.untried = list(untried_moves) if untried_moves else []

    def ucb1(self, c=1.41):
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def select_child(self):
        return max(self.children, key=lambda n: n.ucb1())

    def expand(self, board, player):
        if not self.untried:
            return None
        move = self.untried.pop()
        child = MCTSNode(move=move, parent=self,
                         untried_moves=get_empty_cells_after_move(board, move, player))
        self.children.append(child)
        return child

    def backpropagate(self, result):
        node = self
        while node is not None:
            node.visits += 1
            node.wins += result
            node = node.parent


def get_empty_cells_after_move(board, move, player):
    """Get empty cells after applying a move."""
    r, c = move
    new_board = list(board)
    new_board[idx(r, c)] = player
    return get_empty_cells(new_board)


def mcts_rollout(board, player):
    """Random playout to terminal state."""
    sim = list(board)
    empties = get_empty_cells(sim)
    random.shuffle(empties)
    opp = WHITE if player == BLACK else BLACK
    turn = player
    for r, c in empties:
        sim[idx(r, c)] = turn
        turn = opp if turn == player else player
    return 1.0 if check_connection(sim, player) else 0.0


def mcts_search(board, player, num_iterations=200):
    """MCTS with UCB1 selection. Returns best move (r, c)."""
    empties = get_empty_cells(board)
    if not empties:
        return None
    if len(empties) == 1:
        return empties[0]

    root = MCTSNode(untried_moves=empties)

    for _ in range(num_iterations):
        node = root
        sim_board = list(board)
        sim_player = player
        opp = WHITE if player == BLACK else BLACK

        # Selection: walk down the tree via UCB1
        while not node.untried and node.children:
            node = node.select_child()
            sim_board[idx(node.move[0], node.move[1])] = sim_player
            sim_player = opp if sim_player == player else player

        # Expansion: add one child
        if node.untried:
            move = node.untried.pop(random.randrange(len(node.untried)))
            sim_board[idx(move[0], move[1])] = sim_player
            child = MCTSNode(move=move, parent=node,
                             untried_moves=get_empty_cells(sim_board))
            node.children.append(child)
            node = child
            sim_player = opp if sim_player == player else player

        # Rollout
        result = mcts_rollout(sim_board, player)

        # Backpropagation
        node.backpropagate(result)

    # Pick best move by visit count (most robust)
    best = max(root.children, key=lambda n: n.visits)
    return best.move


def _play_one_game(args):
    """Worker: play one self-play game and return (board_str, board, best_move) tuples."""
    gi, num_games, seed = args
    rng = random.Random(seed + gi)
    results = []

    board = [EMPTY] * BOARD_SIZE
    turn = BLACK

    for ply in range(BOARD_SIZE):
        empties = get_empty_cells(board)
        if not empties:
            break

        if check_connection(board, BLACK) or check_connection(board, WHITE):
            break

        board_str = board_to_str(board)
        if turn == BLACK and len(empties) > 5:
            best = mcts_search(board, BLACK, num_iterations=200)
            if best:
                results.append((board_str, list(board), best))

        # Play: mix MCTS and random for diversity
        if gi % 3 == 0 or ply < 3:
            r, c = rng.choice(empties)
        else:
            move = mcts_search(board, turn, num_iterations=100)
            r, c = move if move else rng.choice(empties)

        board[idx(r, c)] = turn
        turn = WHITE if turn == BLACK else BLACK

    return results


def generate_positions(num_games=800):
    """Generate positions via parallel self-play."""
    num_workers = min(cpu_count(), 8)
    print(f"  Using {num_workers} workers for parallel self-play", file=sys.stderr)

    args = [(gi, num_games, 42) for gi in range(num_games)]

    positions = {}
    completed = 0
    with Pool(num_workers) as pool:
        for result_batch in pool.imap_unordered(_play_one_game, args, chunksize=4):
            for board_str, board, best in result_batch:
                if board_str not in positions:
                    positions[board_str] = (board, best)
            completed += 1
            if completed % 50 == 0:
                print(f"  Self-play: {completed}/{num_games}, {len(positions)} positions", file=sys.stderr)

    return positions


def generate_planner_corpus(positions):
    entries = []
    seen = set()
    for board_str, (board, _) in positions.items():
        empties = len(get_empty_cells(board))
        if empties <= 5: plan = "move,check"
        elif empties <= 15: plan = "move,check,move,check"
        else: plan = "move,check,move,check,move,check"
        prompt = f"board={board_str}|empty={empties}"
        if prompt not in seen:
            entries.append(f"{prompt}\ntodo={plan}")
            seen.add(prompt)
    return entries


def generate_player_corpus(positions):
    entries = []
    seen = set()
    for board_str, (board, (br, bc)) in positions.items():
        topo = compute_topo_features(board, BLACK)
        output = f"R{br}C{bc}"
        # Enriched prompt with topological features instead of flat valid-moves list
        prompt = f"board={board_str}|xg={topo['xg']}|xd={topo['xd']}|og={topo['og']}|od={topo['od']}|xb={topo['xb']}"
        if prompt not in seen:
            entries.append(f"{prompt}\n{output}")
            seen.add(prompt)
    return entries


def main():
    print("Generating Hex 7x7 training corpora (MC + topo features)...\n", file=sys.stderr)

    print("Phase 1: Generating positions (800 games, 100 MC sims)...", file=sys.stderr)
    positions = generate_positions(num_games=800)
    print(f"  Total positions: {len(positions)}\n", file=sys.stderr)

    planner_entries = generate_planner_corpus(positions)
    print(f"  Planner: {len(planner_entries)} entries", file=sys.stderr)

    player_entries = generate_player_corpus(positions)
    print(f"  Player: {len(player_entries)} entries", file=sys.stderr)

    print("\nWriting corpus files:", file=sys.stderr)
    for name, entries in [
        ("hex_planner.txt", planner_entries),
        ("hex_player.txt", player_entries),
    ]:
        content = "\n\n".join(entries) + "\n"
        with open(name, "w") as f:
            f.write(content)
        print(f"  {name}: {len(entries)} entries, {len(content)} bytes", file=sys.stderr)

    print("\nDone!", file=sys.stderr)


if __name__ == "__main__":
    main()
