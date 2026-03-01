#!/usr/bin/env python3
# Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
# MIT License — see LICENSE file for details.
"""
Generate training corpora for the Sudoku 4x4 multi-organelle demo.

Creates 2 corpus files:
  - sudoku_planner.txt  (board + empty count → plan)
  - sudoku_player.txt   (board + context → placement)

Sudoku 4x4: digits 1-4, unique in each row, column, and 2x2 box.
Board string: 16 chars, '0' for empty, '1'-'4' for filled.
Player output: compact "RrCcVv" e.g. "R0C1V3" = row 0, col 1, value 3.
"""

import random
import sys

SIZE = 4
BOX = 2
DIGITS = list(range(1, SIZE + 1))


def board_to_str(board):
    return ''.join(str(c) for c in board)


def idx(r, c):
    return r * SIZE + c


def is_valid_placement(board, r, c, val):
    """Check if placing val at (r,c) is valid."""
    # Row
    for cc in range(SIZE):
        if board[idx(r, cc)] == val:
            return False
    # Column
    for rr in range(SIZE):
        if board[idx(rr, c)] == val:
            return False
    # Box
    br, bc = (r // BOX) * BOX, (c // BOX) * BOX
    for rr in range(br, br + BOX):
        for cc in range(bc, bc + BOX):
            if board[idx(rr, cc)] == val:
                return False
    return True


def solve(board):
    """Solve sudoku using backtracking. Returns solved board or None."""
    for i in range(SIZE * SIZE):
        if board[i] == 0:
            r, c = divmod(i, SIZE)
            for val in DIGITS:
                if is_valid_placement(board, r, c, val):
                    board[i] = val
                    result = solve(board)
                    if result is not None:
                        return result
                    board[i] = 0
            return None
    return list(board)


def count_solutions(board, max_count=2):
    """Count solutions (stop at max_count)."""
    count = [0]

    def _solve(b):
        if count[0] >= max_count:
            return
        for i in range(SIZE * SIZE):
            if b[i] == 0:
                r, c = divmod(i, SIZE)
                for val in DIGITS:
                    if is_valid_placement(b, r, c, val):
                        b[i] = val
                        _solve(b)
                        b[i] = 0
                return
        count[0] += 1

    _solve(list(board))
    return count[0]


def generate_solved_grid():
    """Generate a random solved 4x4 Sudoku grid."""
    board = [0] * (SIZE * SIZE)

    # Fill using randomized backtracking
    def _fill(b):
        for i in range(SIZE * SIZE):
            if b[i] == 0:
                r, c = divmod(i, SIZE)
                candidates = list(DIGITS)
                random.shuffle(candidates)
                for val in candidates:
                    if is_valid_placement(b, r, c, val):
                        b[i] = val
                        if _fill(b):
                            return True
                        b[i] = 0
                return False
        return True

    _fill(board)
    return board


def generate_puzzle(solved, num_remove):
    """Remove cells from a solved grid to create a puzzle with unique solution."""
    puzzle = list(solved)
    cells = list(range(SIZE * SIZE))
    random.shuffle(cells)

    removed = 0
    for cell in cells:
        if removed >= num_remove:
            break
        val = puzzle[cell]
        puzzle[cell] = 0
        if count_solutions(puzzle) == 1:
            removed += 1
        else:
            puzzle[cell] = val

    return puzzle


def get_empty_cells(board):
    return [(r, c) for r in range(SIZE) for c in range(SIZE) if board[idx(r, c)] == 0]


def get_valid_values(board, r, c):
    return [v for v in DIGITS if is_valid_placement(board, r, c, v)]


def find_best_move(board, solved):
    """Find the best next move: the empty cell with fewest candidates."""
    empties = get_empty_cells(board)
    if not empties:
        return None

    best = None
    min_candidates = SIZE + 1
    for r, c in empties:
        candidates = get_valid_values(board, r, c)
        if len(candidates) < min_candidates:
            min_candidates = len(candidates)
            best = (r, c, solved[idx(r, c)])

    return best


def generate_puzzles(num_puzzles=3000):
    """Generate puzzle + solution pairs."""
    random.seed(42)
    puzzles = []
    seen = set()

    for _ in range(num_puzzles * 2):
        if len(puzzles) >= num_puzzles:
            break

        solved = generate_solved_grid()
        num_remove = random.randint(4, 10)  # remove 4-10 cells
        puzzle = generate_puzzle(solved, num_remove)

        puzzle_str = board_to_str(puzzle)
        if puzzle_str in seen:
            continue
        seen.add(puzzle_str)

        puzzles.append((puzzle, solved))

    return puzzles


# ---- Corpus Generation ----

def generate_planner_corpus(puzzles):
    entries = []
    seen = set()

    for puzzle, solved in puzzles:
        board_str = board_to_str(puzzle)
        empty_count = puzzle.count(0)

        if empty_count <= 2:
            plan = "fill,check"
        elif empty_count <= 5:
            plan = "fill,check,fill,check"
        else:
            plan = "fill,check,fill,check,fill,check"

        prompt = f"board={board_str}|empty={empty_count}"
        if prompt not in seen:
            entries.append(f"{prompt}\ntodo={plan}")
            seen.add(prompt)

    return entries


def generate_player_corpus(puzzles):
    entries = []
    seen = set()

    for puzzle, solved in puzzles:
        board = list(puzzle)
        board_str = board_to_str(board)

        # Generate entries for each state along the solution path
        while True:
            empties = get_empty_cells(board)
            if not empties:
                break

            move = find_best_move(board, solved)
            if not move:
                break

            r, c, val = move
            curr_str = board_to_str(board)

            # Valid cells string
            valid_parts = []
            for er, ec in empties:
                valid_parts.append(f"R{er}C{ec}")
            valid_str = ','.join(valid_parts)

            output = f"R{r}C{c}V{val}"

            prompt = f"board={curr_str}|valid={valid_str}"
            if prompt not in seen:
                entries.append(f"{prompt}\n{output}")
                seen.add(prompt)

            # Blocked variant
            if len(empties) > 1:
                prompt_b = f"board={curr_str}|valid={valid_str}|blocked=R{r}C{c}"
                if prompt_b not in seen:
                    # Pick another empty cell
                    for er, ec in empties:
                        if (er, ec) != (r, c):
                            alt_val = solved[idx(er, ec)]
                            entries.append(f"{prompt_b}\nR{er}C{ec}V{alt_val}")
                            seen.add(prompt_b)
                            break

            # Apply move and continue
            board[idx(r, c)] = val

    return entries


def main():
    print("Generating Sudoku 4x4 training corpora...\n", file=sys.stderr)

    print("Phase 1: Generating puzzles...", file=sys.stderr)
    puzzles = generate_puzzles(num_puzzles=3000)
    print(f"  Total puzzles: {len(puzzles)}\n", file=sys.stderr)

    print("Phase 2: Generating planner corpus...", file=sys.stderr)
    planner_entries = generate_planner_corpus(puzzles)
    print(f"  Planner: {len(planner_entries)} entries", file=sys.stderr)

    print("Phase 3: Generating player corpus...", file=sys.stderr)
    player_entries = generate_player_corpus(puzzles)
    print(f"  Player: {len(player_entries)} entries", file=sys.stderr)

    print("\nWriting corpus files:", file=sys.stderr)

    for name, entries in [
        ("sudoku_planner.txt", planner_entries),
        ("sudoku_player.txt", player_entries),
    ]:
        content = "\n\n".join(entries) + "\n"
        with open(name, "w") as f:
            f.write(content)
        doc_lengths = [len(e) for e in entries]
        max_len = max(doc_lengths) if doc_lengths else 0
        print(f"  {name}: {len(entries)} entries, {len(content)} bytes, "
              f"max_doc={max_len} chars", file=sys.stderr)

    print("\nDone!", file=sys.stderr)


if __name__ == "__main__":
    main()
