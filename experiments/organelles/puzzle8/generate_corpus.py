#!/usr/bin/env python3
# Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
# MIT License — see LICENSE file for details.
"""
Generate training corpora for the 8-puzzle multi-organelle demo (v2 — Kanban).

Creates 3 corpus files with kanban-enriched prompts:
  - puzzle8_planner.txt  (board + kanban → plan)
  - puzzle8_mover.txt    (board + context → direction)
  - puzzle8_judge.txt    (proposed move → valid/invalid)

DESIGN PRINCIPLE: Keep outputs SHORT and DETERMINISTIC.
  - Mover outputs just a direction word: "up", "down", "left", "right"
  - Judge outputs "yes" or "no"
  - Planner outputs "todo=move,check,move,check"

This ensures the ~18K param model can actually memorize the mapping.
The result board computation is done by the C orchestrator, not the model.
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


def manhattan_distance(board):
    dist = 0
    for i, tile in enumerate(board):
        if tile == 0:
            continue
        goal_pos = tile - 1
        dist += abs(i // 3 - goal_pos // 3) + abs(i % 3 - goal_pos % 3)
    return dist


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


def generate_random_puzzle(max_steps=15):
    board = list(GOAL)
    last_dir = None
    for _ in range(random.randint(2, max_steps)):
        neighbors = get_neighbors(tuple(board))
        candidates = [(b, d) for b, d in neighbors if d != OPPOSITE.get(last_dir)]
        if not candidates:
            candidates = neighbors
        new_board, direction = random.choice(candidates)
        board = list(new_board)
        last_dir = direction
    return tuple(board)


# ---- Planner Corpus ----

def generate_planner_corpus(puzzles_with_solutions):
    """Planner: board|md → todo=move,check,..."""
    entries = []
    for start, solution in puzzles_with_solutions:
        board_str = board_to_str(start)
        md = manhattan_distance(start)
        n_moves = len(solution)

        if n_moves <= 2:
            plan = "move,check"
        elif n_moves <= 5:
            plan = "move,check,move,check"
        else:
            plan = "move,check,move,check,move,check"

        # Initial plan
        prompt = f"board={board_str}|md={md}"
        output = f"todo={plan}"
        entries.append(f"{prompt}\n{output}")

    # Re-planning examples (with done/blocked context)
    for start, solution in puzzles_with_solutions:
        if len(solution) < 3:
            continue
        first_board, first_dir = solution[0]
        md = manhattan_distance(first_board)
        board_str = board_to_str(first_board)

        prompt = f"board={board_str}|md={md}|stalled"
        output = f"todo=move,check,move,check"
        entries.append(f"{prompt}\n{output}")

    return entries


# ---- Mover Corpus (SIMPLIFIED) ----

def generate_mover_corpus(puzzles_with_solutions):
    """Mover: board|blank → direction (just the word).

    Output is ONLY the direction: "up", "down", "left", "right".
    No result board, no prefix, no pipe. Just the word.

    Variants:
    1. Base: board|blank → dir
    2. Blocked: board|blank|blocked=X → alternative dir
    """
    entries = []
    seen = set()

    for start, solution in puzzles_with_solutions:
        current = start

        for step_idx, (new_board, optimal_dir) in enumerate(solution):
            blank = find_blank(current)
            board_str = board_to_str(current)
            valid = get_valid_dirs(current)

            # ---- Type 1: Base (no context) ----
            prompt = f"board={board_str}|blank={blank}"
            if prompt not in seen:
                entries.append(f"{prompt}\n{optimal_dir}")
                seen.add(prompt)

            # ---- Type 2: One dir blocked (not the optimal) ----
            for blocked_dir in valid:
                if blocked_dir == optimal_dir:
                    continue
                prompt_b = f"board={board_str}|blank={blank}|blocked={blocked_dir}"
                if prompt_b not in seen:
                    entries.append(f"{prompt_b}\n{optimal_dir}")
                    seen.add(prompt_b)

            # ---- Type 3: Optimal dir blocked → pick best alternative ----
            remaining = [d for d in valid if d != optimal_dir]
            if remaining:
                # Pick the remaining dir that minimizes manhattan distance
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
                    prompt_b = f"board={board_str}|blank={blank}|blocked={optimal_dir}"
                    if prompt_b not in seen:
                        entries.append(f"{prompt_b}\n{best_alt}")
                        seen.add(prompt_b)

            current = new_board

    return entries


# ---- Judge Corpus (SIMPLIFIED) ----

def generate_judge_corpus(puzzles_with_solutions):
    """Judge: board|dir → "yes" or "no".

    Output is just "yes" (legal move) or "no" (illegal/out-of-bounds).
    Simple binary verdict.
    """
    entries = []
    seen = set()

    for start, solution in puzzles_with_solutions:
        current = start

        for new_board, direction in solution:
            board_str = board_to_str(current)
            valid = get_valid_dirs(current)
            invalid = get_invalid_dirs(current)

            # Valid move examples
            for d in valid:
                prompt = f"board={board_str}|dir={d}"
                if prompt not in seen:
                    entries.append(f"{prompt}\nyes")
                    seen.add(prompt)

            # Invalid move examples
            for d in invalid:
                prompt = f"board={board_str}|dir={d}"
                if prompt not in seen:
                    entries.append(f"{prompt}\nno")
                    seen.add(prompt)

            current = new_board

    return entries


# ---- Main ----

def main():
    print("Generating 8-puzzle training corpora (v2 — Kanban, simplified)...\n")

    random.seed(42)
    num_puzzles = 200

    # Generate unique solvable puzzles
    puzzles = set()
    while len(puzzles) < num_puzzles:
        p = generate_random_puzzle()
        if p != GOAL:
            puzzles.add(p)

    puzzles = list(puzzles)
    print(f"Generated {len(puzzles)} unique solvable puzzles")

    # Solve all puzzles
    solved = []
    for p in puzzles:
        solution = bfs_solve(p)
        if solution is not None:
            solved.append((p, solution))

    print(f"Solved {len(solved)} puzzles")
    lengths = [len(s) for _, s in solved]
    print(f"Solution lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")

    # Generate corpora
    planner_entries = generate_planner_corpus(solved)
    mover_entries = generate_mover_corpus(solved)
    judge_entries = generate_judge_corpus(solved)

    # Write files
    print("\nWriting corpus files:")

    for name, entries in [
        ("puzzle8_planner.txt", planner_entries),
        ("puzzle8_mover.txt", mover_entries),
        ("puzzle8_judge.txt", judge_entries),
    ]:
        content = "\n\n".join(entries) + "\n"
        with open(name, "w") as f:
            f.write(content)

        # Analyze doc lengths
        doc_lengths = [len(e) for e in entries]
        max_len = max(doc_lengths)
        over_64 = sum(1 for l in doc_lengths if l > 64)

        print(f"  {name}: {len(entries)} entries, {len(content)} bytes, "
              f"max_doc={max_len}, >64chars={over_64}")

    print("\nDone! Corpora ready for MicroGPT-C training.")


if __name__ == "__main__":
    main()
