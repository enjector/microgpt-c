#!/usr/bin/env python3
# Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
# MIT License — see LICENSE file for details.
"""
Generate training corpora for the Lights Out multi-organelle demo.

Creates 2 corpus files:
  - lightsout_planner.txt  (board + lit count → plan)
  - lightsout_player.txt   (board + context → cell 0-24)

DESIGN PRINCIPLE: Keep outputs SHORT and DETERMINISTIC.
  - Player outputs just a cell number: "0", "1", ..., "24"
  - Planner outputs "todo=toggle,check,toggle,check"
  - Judge is deterministic (C code) — no corpus needed.

Lights Out: 5×5 grid. Each cell is on (1) or off (0).
Pressing a cell toggles it and its orthogonal neighbours.
Goal: turn all lights off.

Solution method: Gaussian elimination over GF(2).
"""

import random
import sys

SIZE = 5
CELLS = SIZE * SIZE  # 25


def make_board():
    return [0] * CELLS


def board_to_str(board):
    return ''.join(str(c) for c in board)


def neighbours(cell):
    """Return cell + orthogonal neighbours."""
    r, c = divmod(cell, SIZE)
    result = [cell]
    if r > 0: result.append((r - 1) * SIZE + c)
    if r < SIZE - 1: result.append((r + 1) * SIZE + c)
    if c > 0: result.append(r * SIZE + c - 1)
    if c < SIZE - 1: result.append(r * SIZE + c + 1)
    return result


def toggle(board, cell):
    """Toggle cell and its neighbours. Returns new board."""
    new = list(board)
    for n in neighbours(cell):
        new[n] ^= 1
    return new


def count_lit(board):
    return sum(board)


# ---- GF(2) Gaussian Elimination Solver ----

def build_toggle_matrix():
    """Build the 25x25 toggle matrix over GF(2)."""
    mat = []
    for pressed in range(CELLS):
        row = [0] * CELLS
        for n in neighbours(pressed):
            row[n] = 1
        mat.append(row)
    return mat


def solve_lightsout(board):
    """Solve using Gaussian elimination over GF(2).
    Returns list of cells to press, or None if unsolvable."""
    # Augmented matrix [A | b] over GF(2)
    n = CELLS
    aug = []
    for col in range(n):
        row = [0] * (n + 1)
        for pressed in range(n):
            nbs = neighbours(pressed)
            if col in nbs:
                row[pressed] = 1
        row[n] = board[col]
        aug.append(row)

    # Forward elimination
    pivot_cols = []
    row_idx = 0
    for col in range(n):
        # Find pivot
        pivot = -1
        for r in range(row_idx, n):
            if aug[r][col] == 1:
                pivot = r
                break
        if pivot == -1:
            continue
        aug[row_idx], aug[pivot] = aug[pivot], aug[row_idx]
        for r in range(n):
            if r != row_idx and aug[r][col] == 1:
                for j in range(n + 1):
                    aug[r][j] ^= aug[row_idx][j]
        pivot_cols.append(col)
        row_idx += 1

    # Check consistency
    for r in range(row_idx, n):
        if aug[r][n] == 1:
            return None  # unsolvable

    # Extract solution
    solution = [0] * n
    for i, col in enumerate(pivot_cols):
        solution[col] = aug[i][n]

    return [i for i in range(n) if solution[i] == 1]


def generate_solvable_boards(num_boards=5000):
    """Generate solvable boards by starting from solved and pressing random cells."""
    boards = {}
    random.seed(42)

    for _ in range(num_boards * 3):  # oversample to get unique boards
        if len(boards) >= num_boards:
            break

        board = make_board()
        num_presses = random.randint(1, 15)
        pressed_cells = random.sample(range(CELLS), min(num_presses, CELLS))

        for cell in pressed_cells:
            board = toggle(board, cell)

        if count_lit(board) == 0:
            continue  # already solved

        board_str = board_to_str(board)
        if board_str in boards:
            continue

        solution = solve_lightsout(board)
        if solution is not None:
            boards[board_str] = (board, solution)

    return boards


# ---- Corpus Generation ----

def generate_planner_corpus(boards):
    """Planner: board|lit=N → plan."""
    entries = []
    seen = set()

    for board_str, (board, solution) in boards.items():
        lit = count_lit(board)
        steps = len(solution)

        if steps <= 2:
            plan = "toggle,check"
        elif steps <= 5:
            plan = "toggle,check,toggle,check"
        elif steps <= 10:
            plan = "toggle,check,toggle,check,toggle,check"
        else:
            plan = "toggle,check,toggle,check,toggle,check,toggle,check"

        prompt = f"board={board_str}|lit={lit}"
        if prompt not in seen:
            entries.append(f"{prompt}\ntodo={plan}")
            seen.add(prompt)

        # Stalled variant
        if lit >= 4:
            prompt_s = f"board={board_str}|lit={lit}|stalled"
            if prompt_s not in seen:
                entries.append(f"{prompt_s}\ntodo=toggle,check,toggle,check")
                seen.add(prompt_s)

    return entries


def generate_player_corpus(boards):
    """Player: board + valid + context → cell number.

    Output is ONLY a number 0-24.

    Variants:
    1. Base: board|valid=... → optimal cell
    2. Blocked: board|valid=...|blocked=X → next cell from solution
    """
    entries = []
    seen = set()

    for board_str, (board, solution) in boards.items():
        if not solution:
            continue

        optimal_cell = solution[0]
        valid_str = ','.join(str(i) for i in range(CELLS))

        # ---- Type 1: Base ----
        prompt = f"board={board_str}|valid={valid_str}"
        if prompt not in seen:
            entries.append(f"{prompt}\n{optimal_cell}")
            seen.add(prompt)

        # ---- Type 2: Blocked variants ----
        if len(solution) > 1:
            next_cell = solution[1]
            prompt_b = f"board={board_str}|valid={valid_str}|blocked={optimal_cell}"
            if prompt_b not in seen:
                entries.append(f"{prompt_b}\n{next_cell}")
                seen.add(prompt_b)

        # ---- Type 3: After first toggle, new board state ----
        new_board = toggle(board, optimal_cell)
        if count_lit(new_board) > 0 and len(solution) > 1:
            new_board_str = board_to_str(new_board)
            new_solution = solve_lightsout(new_board)
            if new_solution:
                prompt2 = f"board={new_board_str}|valid={valid_str}"
                if prompt2 not in seen:
                    entries.append(f"{prompt2}\n{new_solution[0]}")
                    seen.add(prompt2)

    return entries


# ---- Main ----

def main():
    print("Generating Lights Out training corpora (GF(2) solver)...\n", file=sys.stderr)

    print("Phase 1: Generating solvable boards...", file=sys.stderr)
    boards = generate_solvable_boards(num_boards=5000)
    print(f"  Total unique solvable boards: {len(boards)}\n", file=sys.stderr)

    print("Phase 2: Generating planner corpus...", file=sys.stderr)
    planner_entries = generate_planner_corpus(boards)
    print(f"  Planner: {len(planner_entries)} entries", file=sys.stderr)

    print("Phase 3: Generating player corpus...", file=sys.stderr)
    player_entries = generate_player_corpus(boards)
    print(f"  Player: {len(player_entries)} entries", file=sys.stderr)

    # Write files
    print("\nWriting corpus files:", file=sys.stderr)

    for name, entries in [
        ("lightsout_planner.txt", planner_entries),
        ("lightsout_player.txt", player_entries),
    ]:
        content = "\n\n".join(entries) + "\n"
        with open(name, "w") as f:
            f.write(content)

        doc_lengths = [len(e) for e in entries]
        max_len = max(doc_lengths) if doc_lengths else 0

        print(f"  {name}: {len(entries)} entries, {len(content)} bytes, "
              f"max_doc={max_len} chars", file=sys.stderr)

    print("\nDone! Corpora ready for MicroGPT-C training.", file=sys.stderr)


if __name__ == "__main__":
    main()
