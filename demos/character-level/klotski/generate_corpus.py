#!/usr/bin/env python3
# Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
# MIT License — see LICENSE file for details.
"""
Generate training corpora for the Klotski (simplified) multi-organelle demo.

Simplified Klotski: 4x5 grid.
Blocks: A=2x2 target, B-E=1x2 vertical, F-I=1x1 single, ..=empty.
Goal: Move block A to the bottom-centre (row 3-4, col 1-2).

Board: 20-char string (row-major), each cell holds block ID or '.'.
Player output: "BdD" where B=block ID, D=direction (U/D/L/R).
"""

import random
from collections import deque
import sys

ROWS = 4
COLS = 5
BOARD_SIZE = ROWS * COLS
EMPTY = '.'

# Simplified setup: fewer blocks for tractable BFS
# A = 2x2 target block
# B,C = 1x2 vertical blocks
# D,E,F,G = 1x1 blocks

DIRS = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}


def idx(r, c):
    return r * COLS + c


def board_to_str(board):
    return ''.join(board)


def get_block_cells(board, block_id):
    """Return list of (r,c) cells occupied by block_id."""
    return [(r, c) for r in range(ROWS) for c in range(COLS) if board[idx(r, c)] == block_id]


def get_all_blocks(board):
    """Return set of block IDs."""
    return set(c for c in board if c != EMPTY)


def can_move(board, block_id, direction):
    """Check if block can move in direction."""
    cells = get_block_cells(board, block_id)
    if not cells:
        return False
    dr, dc = DIRS[direction]
    for r, c in cells:
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS:
            return False
        if board[idx(nr, nc)] != EMPTY and board[idx(nr, nc)] != block_id:
            return False
    return True


def apply_move(board, block_id, direction):
    """Apply move, return new board."""
    cells = get_block_cells(board, block_id)
    dr, dc = DIRS[direction]
    new_board = list(board)
    # Clear old positions
    for r, c in cells:
        new_board[idx(r, c)] = EMPTY
    # Set new positions
    for r, c in cells:
        new_board[idx(r + dr, c + dc)] = block_id
    return new_board


def is_goal(board):
    """Check if A block is at bottom-centre."""
    # A should occupy (ROWS-2,1),(ROWS-2,2),(ROWS-1,1),(ROWS-1,2)
    target = [(ROWS - 2, 1), (ROWS - 2, 2), (ROWS - 1, 1), (ROWS - 1, 2)]
    return all(board[idx(r, c)] == 'A' for r, c in target)


def make_initial_board():
    """Create a Klotski-like starting position."""
    board = [EMPTY] * BOARD_SIZE

    # A = 2x2 at top-centre
    for r in range(2):
        for c in range(1, 3):
            board[idx(r, c)] = 'A'

    # B = 1x2 vertical at (0,0)-(1,0)
    board[idx(0, 0)] = 'B'
    board[idx(1, 0)] = 'B'

    # C = 1x2 vertical at (0,3)-(1,3)
    board[idx(0, 3)] = 'C'
    board[idx(1, 3)] = 'C'

    # D,E = 1x1 at (2,0) and (2,3)
    board[idx(2, 0)] = 'D'
    board[idx(2, 3)] = 'E'

    # F,G = 1x1 at (2,1) and (2,2)
    board[idx(2, 1)] = 'F'
    board[idx(2, 2)] = 'G'

    # Empty cells: (3,0)-(3,4) except some filled
    # H = 1x1 at (3,0)
    board[idx(3, 0)] = 'H'

    # Leave (3,1),(3,2),(3,3),(3,4) and row 0 col 4, etc as empty
    # Actually let's keep it simple: (0,4),(1,4),(3,1),(3,2),(3,3),(3,4)

    return board


def bfs_solve(start_board, max_states=50000):
    """BFS to find solution path. Returns list of (board, block_id, dir) or None."""
    start_str = board_to_str(start_board)
    if is_goal(start_board):
        return []

    visited = {start_str}
    queue = deque([(start_board, [])])

    while queue and len(visited) < max_states:
        board, path = queue.popleft()
        blocks = get_all_blocks(board)

        for block_id in sorted(blocks):
            for d in ['U', 'D', 'L', 'R']:
                if can_move(board, block_id, d):
                    new_board = apply_move(board, block_id, d)
                    new_str = board_to_str(new_board)
                    if new_str not in visited:
                        visited.add(new_str)
                        new_path = path + [(board_to_str(board), block_id, d)]
                        if is_goal(new_board):
                            return new_path
                        queue.append((new_board, new_path))

    return None


def generate_positions_from_solutions(num_starts=500):
    """Generate positions by scrambling from goal state and solving."""
    random.seed(42)
    positions = {}

    # Goal state: A at bottom-centre
    goal = [EMPTY] * BOARD_SIZE
    for r in range(ROWS - 2, ROWS):
        for c in range(1, 3):
            goal[idx(r, c)] = 'A'

    # Add some blocks around
    goal[idx(0, 0)] = 'B'; goal[idx(1, 0)] = 'B'
    goal[idx(0, 3)] = 'C'; goal[idx(1, 3)] = 'C'
    goal[idx(0, 1)] = 'D'; goal[idx(0, 2)] = 'E'
    goal[idx(1, 1)] = 'F'; goal[idx(1, 2)] = 'G'

    for si in range(num_starts):
        # Scramble from goal by random moves
        board = list(goal)
        num_scrambles = random.randint(3, 15)

        for _ in range(num_scrambles):
            blocks = sorted(get_all_blocks(board))
            random.shuffle(blocks)
            moved = False
            for bid in blocks:
                dirs = ['U', 'D', 'L', 'R']
                random.shuffle(dirs)
                for d in dirs:
                    if can_move(board, bid, d):
                        board = apply_move(board, bid, d)
                        moved = True
                        break
                if moved:
                    break

        if is_goal(board):
            continue

        board_str = board_to_str(board)
        if board_str in positions:
            continue

        # Solve it
        solution = bfs_solve(board, max_states=20000)
        if solution and len(solution) > 0:
            # First move of solution
            _, block_id, direction = solution[0]
            positions[board_str] = (list(board), block_id, direction, len(solution))

        if (si + 1) % 100 == 0:
            print(f"  Scramble: {si+1}/{num_starts}, {len(positions)} solvable positions", file=sys.stderr)

    return positions


def generate_planner_corpus(positions):
    entries = []
    seen = set()
    for board_str, (board, _, _, sol_len) in positions.items():
        empties = board.count(EMPTY)
        if sol_len <= 3: plan = "slide,check"
        elif sol_len <= 8: plan = "slide,check,slide,check"
        else: plan = "slide,check,slide,check,slide,check"
        prompt = f"board={board_str}|empty={empties}"
        if prompt not in seen:
            entries.append(f"{prompt}\ntodo={plan}")
            seen.add(prompt)
    return entries


def generate_player_corpus(positions):
    entries = []
    seen = set()
    for board_str, (board, block_id, direction, _) in positions.items():
        # Valid moves
        blocks = sorted(get_all_blocks(board))
        valid_parts = []
        for bid in blocks:
            for d in ['U', 'D', 'L', 'R']:
                if can_move(board, bid, d):
                    valid_parts.append(f"{bid}{d}")
        valid_str = ','.join(valid_parts[:20])
        output = f"{block_id}{direction}"

        prompt = f"board={board_str}|valid={valid_str}"
        if len(prompt) > 100:
            prompt = prompt[:100]
        if prompt not in seen:
            entries.append(f"{prompt}\n{output}")
            seen.add(prompt)

    return entries


def main():
    print("Generating Klotski training corpora (BFS solver)...\n", file=sys.stderr)

    positions = generate_positions_from_solutions(num_starts=500)
    print(f"  Positions: {len(positions)}", file=sys.stderr)

    planner_entries = generate_planner_corpus(positions)
    player_entries = generate_player_corpus(positions)
    print(f"  Planner: {len(planner_entries)}, Player: {len(player_entries)}", file=sys.stderr)

    for name, entries in [
        ("klotski_planner.txt", planner_entries),
        ("klotski_player.txt", player_entries),
    ]:
        content = "\n\n".join(entries) + "\n"
        with open(name, "w") as f:
            f.write(content)
        print(f"  {name}: {len(entries)} entries, {len(content)} bytes", file=sys.stderr)

    print("\nDone!", file=sys.stderr)


if __name__ == "__main__":
    main()
