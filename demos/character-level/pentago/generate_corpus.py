#!/usr/bin/env python3
# Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
# MIT License — see LICENSE file for details.
"""
Generate training corpora for the Pentago multi-organelle demo.

Pentago: 6x6 board, 4 quadrants (3x3 each).
A move = place a marble + rotate one quadrant (CW or CCW).
Win = 5 in a row (any direction).

Board: 36-char string, '.'=empty, 'X'=player, 'O'=opponent.
Player output: "PrrccQqDd" e.g. "P23Q0DC" = place at (2,3), rotate Q0 clockwise.
d: C=clockwise, A=anticlockwise.
"""

import random
import sys

SIZE = 6
QSIZE = 3
BOARD_SIZE = SIZE * SIZE
EMPTY = '.'
BLACK = 'X'
WHITE = 'O'


def idx(r, c):
    return r * SIZE + c


def board_to_str(board):
    return ''.join(board)


def get_quad_cells(q):
    """Get (r,c) coordinates for quadrant q (0=TL, 1=TR, 2=BL, 3=BR)."""
    qr = (q // 2) * QSIZE
    qc = (q % 2) * QSIZE
    return [(qr + r, qc + c) for r in range(QSIZE) for c in range(QSIZE)]


def rotate_quad(board, q, clockwise):
    """Rotate quadrant q. Returns new board."""
    cells = get_quad_cells(q)
    new_board = list(board)
    vals = [board[idx(r, c)] for r, c in cells]
    # 3x3 rotation
    if clockwise:
        # [0,1,2,3,4,5,6,7,8] -> [6,3,0,7,4,1,8,5,2]
        perm = [6, 3, 0, 7, 4, 1, 8, 5, 2]
    else:
        perm = [2, 5, 8, 1, 4, 7, 0, 3, 6]
    for i, (r, c) in enumerate(cells):
        new_board[idx(r, c)] = vals[perm[i]]
    return new_board


def check_five(board, player):
    """Check for 5 in a row."""
    dirs = [(0,1),(1,0),(1,1),(1,-1)]
    for r in range(SIZE):
        for c in range(SIZE):
            for dr, dc in dirs:
                er, ec = r + 4*dr, c + 4*dc
                if 0 <= er < SIZE and 0 <= ec < SIZE:
                    if all(board[idx(r+i*dr, c+i*dc)] == player for i in range(5)):
                        return True
    return False


def get_empties(board):
    return [(r, c) for r in range(SIZE) for c in range(SIZE) if board[idx(r, c)] == EMPTY]


def evaluate(board, player):
    opp = WHITE if player == BLACK else BLACK
    if check_five(board, player): return 10000
    if check_five(board, opp): return -10000
    score = 0
    # Count near-wins
    dirs = [(0,1),(1,0),(1,1),(1,-1)]
    for r in range(SIZE):
        for c in range(SIZE):
            for dr, dc in dirs:
                for length in [4, 3]:
                    er = r + (length-1)*dr
                    ec = c + (length-1)*dc
                    if 0 <= er < SIZE and 0 <= ec < SIZE:
                        window = [board[idx(r+i*dr, c+i*dc)] for i in range(length)]
                        p = window.count(player)
                        o = window.count(opp)
                        e = window.count(EMPTY)
                        if o == 0 and p > 0:
                            score += p * p
                        if p == 0 and o > 0:
                            score -= o * o
    return score


def find_best_move(board, player):
    """Find best move: try all (place, quad, dir) combos with simple eval."""
    empties = get_empties(board)
    if not empties:
        return None

    best_score = -99999
    best_move = None

    # Sample for speed
    sample = random.sample(empties, min(len(empties), 12))

    for r, c in sample:
        new_board = list(board)
        new_board[idx(r, c)] = player
        for q in range(4):
            for cw in [True, False]:
                rotated = rotate_quad(new_board, q, cw)
                score = evaluate(rotated, player)
                if score > best_score:
                    best_score = score
                    best_move = (r, c, q, cw)

    return best_move


def generate_positions(num_games=100):
    random.seed(42)
    positions = {}

    for gi in range(num_games):
        board = [EMPTY] * BOARD_SIZE
        turn = BLACK

        for ply in range(BOARD_SIZE):
            empties = get_empties(board)
            if not empties:
                break
            if check_five(board, BLACK) or check_five(board, WHITE):
                break

            board_str = board_to_str(board)
            if turn == BLACK and board_str not in positions and len(empties) > 10:
                # Quick evaluation: just pick centre-biased move
                r, c = min(empties, key=lambda rc: abs(rc[0]-2.5) + abs(rc[1]-2.5))
                q = random.randint(0, 3)
                cw = random.choice([True, False])
                positions[board_str] = (list(board), (r, c, q, cw))

            # Play randomly for speed
            r, c = random.choice(empties)
            q = random.randint(0, 3)
            cw = random.choice([True, False])

            board[idx(r, c)] = turn
            board = rotate_quad(board, q, cw)
            turn = WHITE if turn == BLACK else BLACK

        if (gi + 1) % 50 == 0:
            print(f"  Self-play: {gi+1}/{num_games}, {len(positions)} positions", file=sys.stderr)

    return positions



def generate_planner_corpus(positions):
    entries = []
    seen = set()
    for board_str, (board, _) in positions.items():
        empties = len(get_empties(board))
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
    for board_str, (board, (pr, pc, pq, pcw)) in positions.items():
        empties = get_empties(board)
        valid_str = ','.join(f"R{r}C{c}" for r, c in empties[:15])
        d = 'C' if pcw else 'A'
        output = f"P{pr}{pc}Q{pq}D{d}"

        prompt = f"board={board_str}|valid={valid_str}"
        if len(prompt) > 110:
            prompt = prompt[:110]
        if prompt not in seen:
            entries.append(f"{prompt}\n{output}")
            seen.add(prompt)
    return entries


def main():
    print("Generating Pentago training corpora...\n", file=sys.stderr)

    positions = generate_positions(num_games=300)
    print(f"  Positions: {len(positions)}", file=sys.stderr)

    planner_entries = generate_planner_corpus(positions)
    player_entries = generate_player_corpus(positions)
    print(f"  Planner: {len(planner_entries)}, Player: {len(player_entries)}", file=sys.stderr)

    for name, entries in [
        ("pentago_planner.txt", planner_entries),
        ("pentago_player.txt", player_entries),
    ]:
        content = "\n\n".join(entries) + "\n"
        with open(name, "w") as f:
            f.write(content)
        print(f"  {name}: {len(entries)} entries, {len(content)} bytes", file=sys.stderr)

    print("\nDone!", file=sys.stderr)


if __name__ == "__main__":
    main()
