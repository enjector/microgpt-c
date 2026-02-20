#!/usr/bin/env python3
# Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
# MIT License — see LICENSE file for details.
"""
Generate training corpora for the Hex 7x7 multi-organelle demo.

Hex: 7x7 board, X connects top-bottom, O connects left-right.
Board: 49-char string, '.'=empty, 'X'=black, 'O'=white.
Player output: "RrCc" e.g. "R3C4".

Uses Monte Carlo evaluation to rank moves.
"""

import random
import sys

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


def find_best_move(board, player, num_sims=30):
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


def generate_positions(num_games=300):
    """Generate positions via self-play."""
    random.seed(42)
    positions = {}

    for gi in range(num_games):
        board = [EMPTY] * BOARD_SIZE
        turn = BLACK

        for ply in range(BOARD_SIZE):
            empties = get_empty_cells(board)
            if not empties:
                break

            if check_connection(board, BLACK) or check_connection(board, WHITE):
                break

            board_str = board_to_str(board)
            if turn == BLACK and board_str not in positions and len(empties) > 5:
                best = find_best_move(board, BLACK, num_sims=20)
                if best:
                    positions[board_str] = (list(board), best)

            # Play
            if gi % 3 == 0 or ply < 3:
                r, c = random.choice(empties)
            else:
                move = find_best_move(board, turn, num_sims=10)
                r, c = move if move else random.choice(empties)

            board[idx(r, c)] = turn
            turn = WHITE if turn == BLACK else BLACK

        if (gi + 1) % 50 == 0:
            print(f"  Self-play: {gi+1}/{num_games}, {len(positions)} positions", file=sys.stderr)

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
        empties = get_empty_cells(board)
        valid_str = ','.join(f"R{r}C{c}" for r, c in empties[:20])  # Limit length
        output = f"R{br}C{bc}"
        prompt = f"board={board_str}|valid={valid_str}"
        if len(prompt) > 110:
            prompt = prompt[:110]
        if prompt not in seen:
            entries.append(f"{prompt}\n{output}")
            seen.add(prompt)
    return entries


def main():
    print("Generating Hex 7x7 training corpora (Monte Carlo)...\n", file=sys.stderr)

    print("Phase 1: Generating positions...", file=sys.stderr)
    positions = generate_positions(num_games=300)
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
