#!/usr/bin/env python3
# Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
# MIT License — see LICENSE file for details.
"""
Generate training corpora for the 8-puzzle multi-organelle demo (v3 — Generalisation).

Creates 3 corpus files:
  - puzzle8_strategist.txt  (md deltas per direction → priority direction)
  - puzzle8_mover.txt       (md deltas per direction + blank → direction)
  - puzzle8_judge.txt       (board + dir → yes/no)

DESIGN PRINCIPLE: Teach STRUCTURE, not memorisation.
  - MD-delta encoding: m=U,D,L,R tells the model the manhattan distance
    AFTER each possible move.  'x' means that direction is illegal.
  - The model learns "pick the direction with the smallest delta" — a
    simple structural rule that generalises across all board positions.
  - Mover outputs a single word: "up", "down", "left", "right"
  - Strategist outputs the priority direction based on md deltas
"""

import random
from collections import deque

GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)
GOAL_STR = "123456780"

DIRS = [(-1, 0, "up"), (1, 0, "down"), (0, -1, "left"), (0, 1, "right")]
DIR_NAMES = ["up", "down", "left", "right"]
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


def board_to_str(board):
    return "".join(str(t) for t in board)


def find_blank(board):
    return board.index(0)


def manhattan_distance(board):
    dist = 0
    for i, tile in enumerate(board):
        if tile == 0:
            continue
        goal_pos = tile - 1
        dist += abs(i // 3 - goal_pos // 3) + abs(i % 3 - goal_pos % 3)
    return dist


def get_neighbors(board):
    blank = find_blank(board)
    r, c = blank // 3, blank % 3
    neighbors = []
    for dr, dc, name in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            ni = nr * 3 + nc
            new_board = list(board)
            new_board[blank], new_board[ni] = new_board[ni], new_board[blank]
            neighbors.append((tuple(new_board), name))
    return neighbors


def get_valid_dirs(board):
    blank = find_blank(board)
    r, c = blank // 3, blank % 3
    valid = []
    for dr, dc, name in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            valid.append(name)
    return valid


def get_invalid_dirs(board):
    valid = set(get_valid_dirs(board))
    return [d for d in DIR_NAMES if d not in valid]


def apply_dir(board, direction):
    blank = find_blank(board)
    r, c = blank // 3, blank % 3
    dr, dc = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}[direction]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < 3 and 0 <= nc < 3):
        return None
    ni = nr * 3 + nc
    new_board = list(board)
    new_board[blank], new_board[ni] = new_board[ni], new_board[blank]
    return tuple(new_board)


def md_delta_str(board):
    """Compute md after each possible move. 'x' = illegal.
    Format: m=U,D,L,R where values are the resulting md or 'x'.
    """
    parts = []
    for d_name in DIR_NAMES:
        result = apply_dir(board, d_name)
        if result is None:
            parts.append("x")
        else:
            parts.append(str(manhattan_distance(result)))
    return "m=" + ",".join(parts)


def bfs_solve(start):
    if start == GOAL:
        return []
    visited = {start}
    queue = deque([(start, [])])
    while queue:
        board, path = queue.popleft()
        for new_board, direction in get_neighbors(board):
            if new_board in visited:
                continue
            new_path = path + [(new_board, direction)]
            if new_board == GOAL:
                return new_path
            visited.add(new_board)
            queue.append((new_board, new_path))
    return None


def generate_random_puzzle(max_steps=25):
    """Generate a puzzle by random walk from goal (avoids back-tracking)."""
    board = list(GOAL)
    last_dir = None
    n_steps = random.randint(2, max_steps)
    for _ in range(n_steps):
        neighbors = get_neighbors(tuple(board))
        candidates = [(b, d) for b, d in neighbors if d != OPPOSITE.get(last_dir)]
        if not candidates:
            candidates = neighbors
        new_board, direction = random.choice(candidates)
        board = list(new_board)
        last_dir = direction
    return tuple(board)


# ---- Strategist Corpus ----

def generate_strategist_corpus(puzzles_with_solutions):
    """Strategist: md_deltas → priority direction.

    The Strategist sees the md after each possible direction and outputs
    the direction that produces the lowest md.  This is a learnable,
    generalising heuristic.
    """
    entries = []
    seen = set()

    for start, solution in puzzles_with_solutions:
        current = start

        for new_board, optimal_dir in solution:
            mds = md_delta_str(current)
            prompt = mds
            if prompt not in seen:
                entries.append(f"{prompt}\n{optimal_dir}")
                seen.add(prompt)
            current = new_board

    return entries


# ---- Mover Corpus ----

def generate_mover_corpus(puzzles_with_solutions):
    """Mover: md_deltas + blank → direction.

    Output is ONLY the direction word.
    Input uses md-delta encoding so the model learns:
    "given the consequences of each move, pick the best one."

    Variants:
    1. Base: m=U,D,L,R|b=N → dir
    2. Blocked: m=U,D,L,R|b=N|x=DIR → alternative dir
    """
    entries = []
    seen = set()

    for start, solution in puzzles_with_solutions:
        current = start

        for step_idx, (new_board, optimal_dir) in enumerate(solution):
            blank = find_blank(current)
            mds = md_delta_str(current)
            valid = get_valid_dirs(current)

            # ---- Type 1: Base ----
            prompt = f"{mds}|b={blank}"
            if prompt not in seen:
                entries.append(f"{prompt}\n{optimal_dir}")
                seen.add(prompt)

            # ---- Type 2: One dir blocked (not the optimal) ----
            for blocked_dir in valid:
                if blocked_dir == optimal_dir:
                    continue
                prompt_b = f"{mds}|b={blank}|x={blocked_dir}"
                if prompt_b not in seen:
                    entries.append(f"{prompt_b}\n{optimal_dir}")
                    seen.add(prompt_b)

            # ---- Type 3: Optimal dir blocked → pick best alternative ----
            remaining = [d for d in valid if d != optimal_dir]
            if remaining:
                best_alt = None
                best_md = 999
                for alt in remaining:
                    alt_board = apply_dir(current, alt)
                    if alt_board is not None:
                        alt_md = manhattan_distance(alt_board)
                        if alt_md < best_md:
                            best_md = alt_md
                            best_alt = alt

                if best_alt:
                    prompt_b = f"{mds}|b={blank}|x={optimal_dir}"
                    if prompt_b not in seen:
                        entries.append(f"{prompt_b}\n{best_alt}")
                        seen.add(prompt_b)

            current = new_board

    return entries


# ---- Judge Corpus ----

def generate_judge_corpus(puzzles_with_solutions):
    """Judge: board|dir → "yes" or "no"."""
    entries = []
    seen = set()

    for start, solution in puzzles_with_solutions:
        current = start

        for new_board, direction in solution:
            board_str = board_to_str(current)
            valid = get_valid_dirs(current)
            invalid = get_invalid_dirs(current)

            for d in valid:
                prompt = f"board={board_str}|dir={d}"
                if prompt not in seen:
                    entries.append(f"{prompt}\nyes")
                    seen.add(prompt)

            for d in invalid:
                prompt = f"board={board_str}|dir={d}"
                if prompt not in seen:
                    entries.append(f"{prompt}\nno")
                    seen.add(prompt)

            current = new_board

    return entries


# ---- Main ----

def main():
    print("Generating 8-puzzle training corpora (v3 — MD-Delta)...\n")

    random.seed(42)
    num_puzzles = 5000

    # Generate unique solvable puzzles with stratified difficulty
    puzzles = set()
    md_counts = {}
    attempts = 0
    max_attempts = num_puzzles * 20

    while len(puzzles) < num_puzzles and attempts < max_attempts:
        attempts += 1
        p = generate_random_puzzle(max_steps=25)
        if p == GOAL or p in puzzles:
            continue
        md = manhattan_distance(p)
        puzzles.add(p)
        md_counts[md] = md_counts.get(md, 0) + 1

    puzzles = list(puzzles)
    print(f"Generated {len(puzzles)} unique solvable puzzles")
    print(f"Manhattan distance distribution:")
    for md in sorted(md_counts.keys()):
        print(f"  md={md:2d}: {md_counts[md]:4d} puzzles")

    # Solve all puzzles
    solved = []
    for p in puzzles:
        solution = bfs_solve(p)
        if solution is not None:
            solved.append((p, solution))

    print(f"\nSolved {len(solved)} puzzles")
    lengths = [len(s) for _, s in solved]
    print(f"Solution lengths: min={min(lengths)}, max={max(lengths)}, "
          f"avg={sum(lengths)/len(lengths):.1f}")

    # Generate corpora
    strategist_entries = generate_strategist_corpus(solved)
    mover_entries = generate_mover_corpus(solved)
    judge_entries = generate_judge_corpus(solved)

    # Write files
    print("\nWriting corpus files:")

    for name, entries in [
        ("puzzle8_strategist.txt", strategist_entries),
        ("puzzle8_mover.txt", mover_entries),
        ("puzzle8_judge.txt", judge_entries),
    ]:
        content = "\n\n".join(entries) + "\n"
        with open(name, "w") as f:
            f.write(content)

        doc_lengths = [len(e) for e in entries]
        max_len = max(doc_lengths)
        over_64 = sum(1 for l in doc_lengths if l > 64)

        print(f"  {name}: {len(entries)} entries, {len(content)} bytes, "
              f"max_doc={max_len}, >64chars={over_64}")

    # Report unique md-delta patterns
    unique_mds = set()
    for start, solution in solved:
        unique_mds.add(md_delta_str(start))
        for new_board, _ in solution:
            unique_mds.add(md_delta_str(new_board))
    print(f"\nUnique md-delta patterns in training: {len(unique_mds)}")

    print("\nDone! Corpora ready for MicroGPT-C training.")


if __name__ == "__main__":
    main()
