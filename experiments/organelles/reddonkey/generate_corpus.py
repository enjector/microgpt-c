#!/usr/bin/env python3
# Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
# MIT License — see LICENSE file for details.
"""
Generate training corpora for the Red Donkey (Huarong Dao) demo.

Simplified Red Donkey: 4x3 grid (smaller for tractable BFS).
Blocks: A=2x2 (the "donkey"), B=1x1, C=1x1, D=1x1, ..=empty.
Goal: Move A to bottom row, cols 0-1.

Board: 12-char string (row-major). Player output: "BD" (block + direction).

For the full 5x4 classic puzzle, BFS is too expensive (~10^10 states).
This simplified variant demonstrates the same OPA coordination patterns
with a tractable state space.
"""

import random
from collections import deque
import sys

ROWS = 4
COLS = 3
BOARD_SIZE = ROWS * COLS  # 12
EMPTY = '.'

DIRS = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
DIR_LETTERS = ['U', 'D', 'L', 'R']


def idx(r, c):
    return r * COLS + c


def board_to_str(board):
    return ''.join(board)


def get_block_cells(board, block_id):
    return [(r, c) for r in range(ROWS) for c in range(COLS) if board[idx(r, c)] == block_id]


def get_all_blocks(board):
    return sorted(set(c for c in board if c != EMPTY))


def can_move(board, block_id, direction):
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
    cells = get_block_cells(board, block_id)
    dr, dc = DIRS[direction]
    new_board = list(board)
    for r, c in cells:
        new_board[idx(r, c)] = EMPTY
    for r, c in cells:
        new_board[idx(r + dr, c + dc)] = block_id
    return new_board


def is_goal(board):
    """A at bottom-left: rows 2-3, cols 0-1."""
    return (board[idx(2, 0)] == 'A' and board[idx(2, 1)] == 'A' and
            board[idx(3, 0)] == 'A' and board[idx(3, 1)] == 'A')


def make_start():
    """Starting position: A at top-left, small blocks scattered."""
    board = [EMPTY] * BOARD_SIZE
    # A = 2x2 at (0,0)
    board[idx(0, 0)] = 'A'; board[idx(0, 1)] = 'A'
    board[idx(1, 0)] = 'A'; board[idx(1, 1)] = 'A'
    # B at (0,2)
    board[idx(0, 2)] = 'B'
    # C at (1,2)
    board[idx(1, 2)] = 'C'
    # D at (2,2)
    board[idx(2, 2)] = 'D'
    # E at (3,2)
    board[idx(3, 2)] = 'E'
    # Empty: (2,0), (2,1), (3,0), (3,1)
    return board


def bfs_solve(start_board, max_states=100000):
    """BFS solver. Returns path [(board_str, block_id, dir), ...]."""
    start_str = board_to_str(start_board)
    if is_goal(start_board):
        return []

    visited = {start_str}
    queue = deque([(start_board, [])])

    while queue and len(visited) < max_states:
        board, path = queue.popleft()
        blocks = get_all_blocks(board)

        for block_id in blocks:
            for d in DIR_LETTERS:
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


def generate_positions(num_scrambles=1000):
    """Generate positions by scrambling from various starting positions."""
    random.seed(42)
    positions = {}

    base = make_start()

    for si in range(num_scrambles):
        board = list(base)
        num_moves = random.randint(2, 12)

        for _ in range(num_moves):
            blocks = get_all_blocks(board)
            random.shuffle(blocks)
            moved = False
            for bid in blocks:
                random.shuffle(DIR_LETTERS)
                for d in DIR_LETTERS:
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

        solution = bfs_solve(board, max_states=50000)
        if solution is not None and len(solution) > 0:
            _, block_id, direction = solution[0]
            positions[board_str] = (list(board), block_id, direction, len(solution))

        if (si + 1) % 100 == 0:
            print(f"  Scramble: {si+1}/{num_scrambles}, {len(positions)} positions", file=sys.stderr)

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
        blocks = get_all_blocks(board)
        valid_parts = []
        for bid in blocks:
            for d in DIR_LETTERS:
                if can_move(board, bid, d):
                    valid_parts.append(f"{bid}{d}")
        valid_str = ','.join(valid_parts[:20])
        output = f"{block_id}{direction}"

        prompt = f"board={board_str}|valid={valid_str}"
        if prompt not in seen:
            entries.append(f"{prompt}\n{output}")
            seen.add(prompt)

    return entries


def main():
    print("Generating Red Donkey training corpora (simplified 4x3)...\n", file=sys.stderr)

    positions = generate_positions(num_scrambles=1000)
    print(f"  Positions: {len(positions)}", file=sys.stderr)

    planner_entries = generate_planner_corpus(positions)
    player_entries = generate_player_corpus(positions)
    print(f"  Planner: {len(planner_entries)}, Player: {len(player_entries)}", file=sys.stderr)

    for name, entries in [
        ("reddonkey_planner.txt", planner_entries),
        ("reddonkey_player.txt", player_entries),
    ]:
        content = "\n\n".join(entries) + "\n"
        with open(name, "w") as f:
            f.write(content)
        print(f"  {name}: {len(entries)} entries, {len(content)} bytes", file=sys.stderr)

    print("\nDone!", file=sys.stderr)


if __name__ == "__main__":
    main()
