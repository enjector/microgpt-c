#!/usr/bin/env python3
# Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
# MIT License — see LICENSE file for details.
"""
Generate training corpora for the 8-puzzle multi-organelle demo (v3b — Greedy/Detour Split).

Creates 5 corpus files:
  - puzzle8_strategist.txt      (md deltas → priority direction, greedy-only)
  - puzzle8_mover.txt           (md deltas + blank → direction, greedy-only)
  - puzzle8_detour_detector.txt (md deltas → "g" or "d", classifies greedy vs detour)
  - puzzle8_detour_mover.txt    (md deltas + blank → direction, detour-only)
  - puzzle8_judge.txt           (board + dir → yes/no)

DESIGN PRINCIPLE: Separate greedy from detour.
  - The greedy corpus is 100% consistent: BFS agrees with "pick smallest md."
  - The detour detector learns to classify positions as greedy or detour.
  - The detour mover learns the specific non-greedy moves from BFS solutions.
  - At inference: detour detector → if "g", use greedy mover; if "d", use detour mover.
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
    """Compute md after each possible move. 'x' = illegal."""
    parts = []
    for d_name in DIR_NAMES:
        result = apply_dir(board, d_name)
        if result is None:
            parts.append("x")
        else:
            parts.append(str(manhattan_distance(result)))
    return "m=" + ",".join(parts)


def md_after_each_dir(board):
    """Return dict of direction → resulting md (None if illegal)."""
    result = {}
    for d_name in DIR_NAMES:
        new_board = apply_dir(board, d_name)
        if new_board is None:
            result[d_name] = None
        else:
            result[d_name] = manhattan_distance(new_board)
    return result


def is_greedy_consistent(board, optimal_dir):
    """True if optimal_dir is among the directions with lowest resulting md."""
    md_map = md_after_each_dir(board)
    opt_md = md_map[optimal_dir]
    if opt_md is None:
        return False  # Should never happen for BFS moves
    # Find minimum md among valid directions
    min_md = min(v for v in md_map.values() if v is not None)
    return opt_md == min_md


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


# ---- Corpus Generation with Greedy/Detour Classification ----

def generate_all_corpora(puzzles_with_solutions):
    """Generate greedy-only Strategist/Mover, detour detector, and detour Mover."""

    strat_greedy = []      # Strategist: greedy-consistent only
    mover_greedy = []      # Mover: greedy-consistent only
    detector_entries = []  # Detour detector: m=...|b=N → "g" or "d"
    mover_detour = []      # Mover: detour-only

    seen_strat = set()
    seen_mover = set()
    seen_detector = set()
    seen_detour = set()

    stats = {"greedy": 0, "tie": 0, "detour": 0}

    for start, solution in puzzles_with_solutions:
        current = start

        for new_board, optimal_dir in solution:
            mds = md_delta_str(current)
            blank = find_blank(current)
            valid = get_valid_dirs(current)
            md_map = md_after_each_dir(current)
            opt_md = md_map[optimal_dir]
            valid_mds = {d: v for d, v in md_map.items() if v is not None}
            min_md = min(valid_mds.values())

            greedy = is_greedy_consistent(current, optimal_dir)

            # Classify for stats
            if greedy:
                if opt_md == min_md and sum(1 for v in valid_mds.values() if v == min_md) > 1:
                    stats["tie"] += 1
                else:
                    stats["greedy"] += 1
            else:
                stats["detour"] += 1

            valid_str = ','.join(valid)

            # ---- Strategist (greedy-only) ----
            if greedy:
                prompt = mds
                if prompt not in seen_strat:
                    strat_greedy.append(f"{prompt}\n{optimal_dir}")
                    seen_strat.add(prompt)

            # ---- Mover (greedy-only, including blocked variants) ----
            if greedy:
                prompt = f"{mds}|b={blank}|valid={valid_str}"
                if prompt not in seen_mover:
                    mover_greedy.append(f"{prompt}\n{optimal_dir}")
                    seen_mover.add(prompt)

                # Blocked variants (not the optimal dir)
                for blocked_dir in valid:
                    if blocked_dir == optimal_dir:
                        continue
                    remaining_valid = [d for d in valid if d != blocked_dir]
                    remaining_str = ','.join(remaining_valid)
                    prompt_b = f"{mds}|b={blank}|valid={remaining_str}|x={blocked_dir}"
                    if prompt_b not in seen_mover:
                        mover_greedy.append(f"{prompt_b}\n{optimal_dir}")
                        seen_mover.add(prompt_b)

                # Optimal blocked → best greedy alternative
                remaining = [d for d in valid if d != optimal_dir]
                if remaining:
                    best_alt = min(remaining, key=lambda d: valid_mds.get(d, 999))
                    remaining_str = ','.join(remaining)
                    prompt_b = f"{mds}|b={blank}|valid={remaining_str}|x={optimal_dir}"
                    if prompt_b not in seen_mover:
                        mover_greedy.append(f"{prompt_b}\n{best_alt}")
                        seen_mover.add(prompt_b)

            # ---- Detour Detector: every position → "g" or "d" ----
            prompt_det = f"{mds}|b={blank}"
            label = "g" if greedy else "d"
            if prompt_det not in seen_detector:
                detector_entries.append(f"{prompt_det}\n{label}")
                seen_detector.add(prompt_det)

            # ---- Detour Mover (detour-only) ----
            if not greedy:
                prompt = f"{mds}|b={blank}|valid={valid_str}"
                if prompt not in seen_detour:
                    mover_detour.append(f"{prompt}\n{optimal_dir}")
                    seen_detour.add(prompt)

            current = new_board

    return strat_greedy, mover_greedy, detector_entries, mover_detour, stats


# ---- Judge Corpus ----

def generate_judge_corpus(puzzles_with_solutions):
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
    print("Generating 8-puzzle training corpora (v3b — Greedy/Detour Split)...\n")

    random.seed(42)
    num_puzzles = 5000

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

    solved = []
    for p in puzzles:
        solution = bfs_solve(p)
        if solution is not None:
            solved.append((p, solution))

    print(f"\nSolved {len(solved)} puzzles")
    lengths = [len(s) for _, s in solved]
    print(f"Solution lengths: min={min(lengths)}, max={max(lengths)}, "
          f"avg={sum(lengths)/len(lengths):.1f}")

    # Generate split corpora
    strat, mover_g, detector, mover_d, stats = generate_all_corpora(solved)
    judge = generate_judge_corpus(solved)

    total_positions = stats["greedy"] + stats["tie"] + stats["detour"]
    print(f"\n--- Greedy/Detour Classification ---")
    print(f"  Total positions:  {total_positions}")
    print(f"  Greedy-optimal:   {stats['greedy']:5d} ({100*stats['greedy']/total_positions:.1f}%)")
    print(f"  Tie-breaking:     {stats['tie']:5d} ({100*stats['tie']/total_positions:.1f}%)")
    print(f"  Non-greedy detour:{stats['detour']:5d} ({100*stats['detour']/total_positions:.1f}%)")

    # Write files
    print("\nWriting corpus files:")
    for name, entries in [
        ("puzzle8_strategist.txt", strat),
        ("puzzle8_mover.txt", mover_g),
        ("puzzle8_detour_detector.txt", detector),
        ("puzzle8_detour_mover.txt", mover_d),
        ("puzzle8_judge.txt", judge),
    ]:
        content = "\n\n".join(entries) + "\n"
        with open(name, "w") as f:
            f.write(content)

        doc_lengths = [len(e) for e in entries]
        max_len = max(doc_lengths) if doc_lengths else 0
        over_64 = sum(1 for l in doc_lengths if l > 64)

        print(f"  {name}: {len(entries)} entries, {len(content)} bytes, "
              f"max_doc={max_len}, >64chars={over_64}")

    print("\nDone! Corpora ready for MicroGPT-C training.")


if __name__ == "__main__":
    main()
