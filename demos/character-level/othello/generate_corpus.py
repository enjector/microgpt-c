#!/usr/bin/env python3
# Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
# MIT License — see LICENSE file for details.
"""
Generate training corpora for the Othello/Reversi 6x6 multi-organelle demo.

Creates 2 corpus files:
  - othello_planner.txt  (board + move count → plan)
  - othello_player.txt   (board + valid moves → position)

Othello 6x6: X and O alternate. Flipping opponent pieces.
Board: 36-char string, '.'=empty, 'X'=black, 'O'=white.
Player output: "RrCc" e.g. "R2C3" = row 2, col 3.
"""

import random
import sys

SIZE = 6
BOARD_SIZE = SIZE * SIZE
EMPTY = '.'
BLACK = 'X'
WHITE = 'O'

DIRS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


def idx(r, c):
    return r * SIZE + c


def make_board():
    board = [EMPTY] * BOARD_SIZE
    mid = SIZE // 2
    board[idx(mid-1, mid-1)] = WHITE
    board[idx(mid-1, mid)] = BLACK
    board[idx(mid, mid-1)] = BLACK
    board[idx(mid, mid)] = WHITE
    return board


def board_to_str(board):
    return ''.join(board)


def get_flips(board, r, c, player):
    """Get list of positions flipped by placing player at (r,c)."""
    opp = WHITE if player == BLACK else BLACK
    all_flips = []
    for dr, dc in DIRS:
        flips = []
        cr, cc = r + dr, c + dc
        while 0 <= cr < SIZE and 0 <= cc < SIZE and board[idx(cr, cc)] == opp:
            flips.append((cr, cc))
            cr += dr
            cc += dc
        if flips and 0 <= cr < SIZE and 0 <= cc < SIZE and board[idx(cr, cc)] == player:
            all_flips.extend(flips)
    return all_flips


def get_valid_moves(board, player):
    """Return list of (r, c) where player can place."""
    moves = []
    for r in range(SIZE):
        for c in range(SIZE):
            if board[idx(r, c)] == EMPTY and get_flips(board, r, c, player):
                moves.append((r, c))
    return moves


def apply_move(board, r, c, player):
    new_board = list(board)
    new_board[idx(r, c)] = player
    for fr, fc in get_flips(board, r, c, player):
        new_board[idx(fr, fc)] = player
    return new_board


def count_pieces(board, player):
    return sum(1 for c in board if c == player)


def evaluate(board, player):
    """Simple evaluation: piece difference + corner bonus + mobility."""
    opp = WHITE if player == BLACK else BLACK
    score = count_pieces(board, player) - count_pieces(board, opp)
    # Corner bonus
    corners = [(0,0),(0,SIZE-1),(SIZE-1,0),(SIZE-1,SIZE-1)]
    for r, c in corners:
        if board[idx(r, c)] == player: score += 10
        elif board[idx(r, c)] == opp: score -= 10
    # Mobility
    score += len(get_valid_moves(board, player)) - len(get_valid_moves(board, opp))
    return score


def minimax(board, depth, alpha, beta, maximising, player):
    opp = WHITE if player == BLACK else BLACK
    moves = get_valid_moves(board, player if maximising else opp)
    opp_moves = get_valid_moves(board, opp if maximising else player)

    if depth == 0 or (not moves and not opp_moves):
        return evaluate(board, player), None

    current = player if maximising else opp
    moves = get_valid_moves(board, current)

    if not moves:
        # Pass
        score, _ = minimax(board, depth - 1, alpha, beta, not maximising, player)
        return score, None

    if maximising:
        best = -9999
        best_move = moves[0]
        for r, c in moves:
            new_board = apply_move(board, r, c, current)
            score, _ = minimax(new_board, depth - 1, alpha, beta, False, player)
            if score > best:
                best = score
                best_move = (r, c)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best, best_move
    else:
        best = 9999
        best_move = moves[0]
        for r, c in moves:
            new_board = apply_move(board, r, c, current)
            score, _ = minimax(new_board, depth - 1, alpha, beta, True, player)
            if score < best:
                best = score
                best_move = (r, c)
            beta = min(beta, score)
            if alpha >= beta:
                break
        return best, best_move


def generate_positions(num_games=500):
    """Generate positions via self-play."""
    random.seed(42)
    positions = {}

    for gi in range(num_games):
        board = make_board()
        turn = BLACK

        for _ in range(BOARD_SIZE):
            moves = get_valid_moves(board, turn)
            opp = WHITE if turn == BLACK else BLACK
            opp_moves = get_valid_moves(board, opp)

            if not moves and not opp_moves:
                break
            if not moves:
                turn = opp
                continue

            board_str = board_to_str(board)
            if turn == BLACK and board_str not in positions:
                _, best = minimax(board, 3, -9999, 9999, True, BLACK)
                if best:
                    positions[board_str] = (list(board), best)

            # Play
            if gi % 3 == 0:
                r, c = random.choice(moves)
            else:
                _, best = minimax(board, 2, -9999, 9999, turn == BLACK, BLACK)
                r, c = best if best else random.choice(moves)

            board = apply_move(board, r, c, turn)
            turn = opp

        if (gi + 1) % 100 == 0:
            print(f"  Self-play: {gi+1}/{num_games}, {len(positions)} positions", file=sys.stderr)

    return positions


def generate_planner_corpus(positions):
    entries = []
    seen = set()
    for board_str, (board, move) in positions.items():
        moves = len(get_valid_moves(board, BLACK))
        if moves <= 3: plan = "move,check"
        elif moves <= 8: plan = "move,check,move,check"
        else: plan = "move,check,move,check,move,check"

        prompt = f"board={board_str}|moves={moves}"
        if prompt not in seen:
            entries.append(f"{prompt}\ntodo={plan}")
            seen.add(prompt)
    return entries


def generate_player_corpus(positions):
    entries = []
    seen = set()
    for board_str, (board, (br, bc)) in positions.items():
        moves = get_valid_moves(board, BLACK)
        valid_str = ','.join(f"R{r}C{c}" for r, c in moves)
        output = f"R{br}C{bc}"

        prompt = f"board={board_str}|valid={valid_str}"
        if prompt not in seen:
            entries.append(f"{prompt}\n{output}")
            seen.add(prompt)

        # Blocked variant
        if len(moves) > 1:
            remaining = [(r, c) for r, c in moves if (r, c) != (br, bc)]
            if remaining:
                alt = remaining[0]
                prompt_b = f"board={board_str}|valid={valid_str}|blocked=R{br}C{bc}"
                if prompt_b not in seen:
                    entries.append(f"{prompt_b}\nR{alt[0]}C{alt[1]}")
                    seen.add(prompt_b)

    return entries


def main():
    print("Generating Othello 6x6 training corpora...\n", file=sys.stderr)

    print("Phase 1: Generating positions via self-play...", file=sys.stderr)
    positions = generate_positions(num_games=500)
    print(f"  Total positions: {len(positions)}\n", file=sys.stderr)

    print("Phase 2: Generating planner corpus...", file=sys.stderr)
    planner_entries = generate_planner_corpus(positions)
    print(f"  Planner: {len(planner_entries)} entries", file=sys.stderr)

    print("Phase 3: Generating player corpus...", file=sys.stderr)
    player_entries = generate_player_corpus(positions)
    print(f"  Player: {len(player_entries)} entries", file=sys.stderr)

    print("\nWriting corpus files:", file=sys.stderr)
    for name, entries in [
        ("othello_planner.txt", planner_entries),
        ("othello_player.txt", player_entries),
    ]:
        content = "\n\n".join(entries) + "\n"
        with open(name, "w") as f:
            f.write(content)
        doc_lengths = [len(e) for e in entries]
        max_len = max(doc_lengths) if doc_lengths else 0
        print(f"  {name}: {len(entries)} entries, {len(content)} bytes, max_doc={max_len}", file=sys.stderr)

    print("\nDone!", file=sys.stderr)


if __name__ == "__main__":
    main()
