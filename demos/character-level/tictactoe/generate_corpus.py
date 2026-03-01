#!/usr/bin/env python3
# Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
# MIT License — see LICENSE file for details.
"""
Generate training corpora for the Tic-Tac-Toe multi-organelle demo.

Creates 2 corpus files:
  - tictactoe_planner.txt  (board + empties → plan)
  - tictactoe_player.txt   (board + context → position 0-8)

DESIGN PRINCIPLE: Keep outputs SHORT and DETERMINISTIC.
  - Player outputs just a position: "0", "1", ..., "8"
  - Planner outputs "todo=move,check,move,check"
  - Judge is deterministic (C code) — no corpus needed.

Uses minimax to find optimal moves for X against any opponent.
"""

import random
from itertools import product

EMPTY = '_'
X = 'X'
O = 'O'


def board_to_str(board):
    return ''.join(board)


def get_empties(board):
    return [i for i, c in enumerate(board) if c == EMPTY]


def check_winner(board):
    """Return 'X', 'O', or None."""
    lines = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
        (0, 4, 8), (2, 4, 6),              # diags
    ]
    for a, b, c in lines:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    return None


def is_terminal(board):
    return check_winner(board) is not None or len(get_empties(board)) == 0


def minimax(board, is_maximising):
    """Returns (score, best_pos). X is maximising (+1), O is minimising (-1)."""
    winner = check_winner(board)
    if winner == X:
        return (1, -1)
    if winner == O:
        return (-1, -1)
    empties = get_empties(board)
    if not empties:
        return (0, -1)  # draw

    if is_maximising:
        best_score = -2
        best_pos = empties[0]
        for pos in empties:
            new_board = list(board)
            new_board[pos] = X
            score, _ = minimax(tuple(new_board), False)
            if score > best_score:
                best_score = score
                best_pos = pos
        return (best_score, best_pos)
    else:
        best_score = 2
        best_pos = empties[0]
        for pos in empties:
            new_board = list(board)
            new_board[pos] = O
            score, _ = minimax(tuple(new_board), True)
            if score < best_score:
                best_score = score
                best_pos = pos
        return (best_score, best_pos)


def rank_moves(board, player):
    """Return all positions ranked by minimax score for the given player."""
    empties = get_empties(board)
    is_max = (player == X)
    scored = []
    for pos in empties:
        new_board = list(board)
        new_board[pos] = player
        score, _ = minimax(tuple(new_board), not is_max)
        scored.append((score, pos))
    # X wants highest score, O wants lowest
    scored.sort(key=lambda x: x[0], reverse=is_max)
    return scored


def generate_all_game_states():
    """Generate reachable board states via BFS from empty board."""
    start = tuple([EMPTY] * 9)
    visited = set()
    queue = [start]
    visited.add(start)
    states_by_turn = {X: [], O: []}

    while queue:
        board = queue.pop(0)
        if is_terminal(board):
            continue
        # Determine whose turn it is
        x_count = board.count(X)
        o_count = board.count(O)
        turn = X if x_count == o_count else O
        states_by_turn[turn].append(board)

        empties = get_empties(board)
        for pos in empties:
            new_board = list(board)
            new_board[pos] = turn
            new_board = tuple(new_board)
            if new_board not in visited:
                visited.add(new_board)
                queue.append(new_board)

    return states_by_turn


# ---- Planner Corpus ----

def generate_planner_corpus(x_states):
    """Planner: board|empties → plan."""
    entries = []
    seen = set()

    for board in x_states:
        board_str = board_to_str(board)
        empties = len(get_empties(board))

        if empties <= 2:
            plan = "move,check"
        elif empties <= 5:
            plan = "move,check,move,check"
        else:
            plan = "move,check,move,check,move,check"

        prompt = f"board={board_str}|empties={empties}"
        if prompt not in seen:
            entries.append(f"{prompt}\ntodo={plan}")
            seen.add(prompt)

    # Re-planning examples (stalled)
    for board in x_states:
        board_str = board_to_str(board)
        empties = len(get_empties(board))
        if empties < 4:
            continue
        prompt = f"board={board_str}|empties={empties}|stalled"
        if prompt not in seen:
            entries.append(f"{prompt}\ntodo=move,check,move,check")
            seen.add(prompt)

    return entries


# ---- Player Corpus ----

def generate_player_corpus(x_states):
    """Player: board + valid + context → position (just the number).

    Output is ONLY a number 0-8. Nothing else.

    Variants:
    1. Base: board|valid=... → optimal_pos
    2. Blocked: board|valid=...|blocked=X → next-best pos
    """
    entries = []
    seen = set()

    for board in x_states:
        board_str = board_to_str(board)
        ranked = rank_moves(board, X)
        if not ranked:
            continue

        optimal_score, optimal_pos = ranked[0]
        empties = get_empties(board)
        valid_str = ','.join(str(p) for p in empties)

        # ---- Type 1: Base (no blocked) ----
        prompt = f"board={board_str}|valid={valid_str}"
        if prompt not in seen:
            entries.append(f"{prompt}\n{optimal_pos}")
            seen.add(prompt)

        # ---- Type 2: Non-optimal positions blocked ----
        for blocked_pos in empties:
            if blocked_pos == optimal_pos:
                continue
            remaining = [p for p in empties if p != blocked_pos]
            remaining_str = ','.join(str(p) for p in remaining)
            prompt_b = f"board={board_str}|valid={remaining_str}|blocked={blocked_pos}"
            if prompt_b not in seen:
                entries.append(f"{prompt_b}\n{optimal_pos}")
                seen.add(prompt_b)

        # ---- Type 3: Optimal blocked → next-best ----
        if len(ranked) > 1:
            _, next_best = ranked[1]
            remaining = [p for p in empties if p != optimal_pos]
            remaining_str = ','.join(str(p) for p in remaining)
            prompt_b = f"board={board_str}|valid={remaining_str}|blocked={optimal_pos}"
            if prompt_b not in seen:
                entries.append(f"{prompt_b}\n{next_best}")
                seen.add(prompt_b)

            # Two blocked
            if len(ranked) > 2:
                _, third_best = ranked[2]
                remaining2 = [p for p in empties if p != optimal_pos and p != next_best]
                remaining2_str = ','.join(str(p) for p in remaining2)
                prompt_b2 = f"board={board_str}|valid={remaining2_str}|blocked={optimal_pos},{next_best}"
                if prompt_b2 not in seen:
                    entries.append(f"{prompt_b2}\n{third_best}")
                    seen.add(prompt_b2)

    return entries


# ---- Main ----

def main():
    print("Generating Tic-Tac-Toe training corpora (minimax-based)...\n")

    # Generate all reachable game states
    states = generate_all_game_states()
    x_states = states[X]
    print(f"Reachable X-turn states: {len(x_states)}")
    print(f"Reachable O-turn states: {len(states[O])}")

    # Generate corpora
    planner_entries = generate_planner_corpus(x_states)
    player_entries = generate_player_corpus(x_states)

    # Write files
    print("\nWriting corpus files:")

    for name, entries in [
        ("tictactoe_planner.txt", planner_entries),
        ("tictactoe_player.txt", player_entries),
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
