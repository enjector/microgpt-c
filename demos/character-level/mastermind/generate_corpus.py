#!/usr/bin/env python3
# Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
# MIT License — see LICENSE file for details.
"""
Generate training corpora for the Mastermind multi-organelle demo.

Creates 2 corpus files:
  - mastermind_planner.txt  (guess count + feedback → plan)
  - mastermind_player.txt   (feedback history + context → 4-char guess)

DESIGN PRINCIPLE: Keep outputs SHORT and DETERMINISTIC.
  - Player outputs a 4-char colour code: "ABCD", "BEEF", etc.
  - Planner outputs "todo=guess,check,guess,check"
  - Judge is deterministic (C code) — compute black/white pegs.

Mastermind: 4 pegs, 6 colours (A-F).
Secret code: 4 characters from {A, B, C, D, E, F}.
Feedback: Black pegs (exact match), White pegs (colour match, wrong position).
"""

import random
from itertools import product
import sys

PEGS = 4
COLOURS = ['A', 'B', 'C', 'D', 'E', 'F']
NUM_COLOURS = len(COLOURS)


def all_codes():
    """Generate all possible 4-peg codes."""
    return [''.join(p) for p in product(COLOURS, repeat=PEGS)]


def score(guess, secret):
    """Return (black_pegs, white_pegs)."""
    black = sum(g == s for g, s in zip(guess, secret))
    # Count colour matches
    guess_counts = {c: 0 for c in COLOURS}
    secret_counts = {c: 0 for c in COLOURS}
    for g, s in zip(guess, secret):
        if g != s:
            guess_counts[g] += 1
            secret_counts[s] += 1
    white = sum(min(guess_counts[c], secret_counts[c]) for c in COLOURS)
    return (black, white)


def minimax_first_guess():
    """Knuth's first guess is always AABB."""
    return "AABB"


def filter_remaining(remaining, guess, feedback):
    """Filter remaining codes consistent with the feedback."""
    return [code for code in remaining if score(guess, code) == feedback]


def knuth_play(secret, max_guesses=10):
    """Play a game using a simplified Knuth algorithm.
    Returns list of (guess, black, white) tuples."""
    remaining = all_codes()
    history = []

    guess = minimax_first_guess()

    for turn in range(max_guesses):
        b, w = score(guess, secret)
        history.append((guess, b, w))

        if b == PEGS:
            break  # solved

        remaining = filter_remaining(remaining, guess, (b, w))
        if not remaining:
            break

        # Next guess: pick first remaining code (simple strategy)
        # For more variety, pick based on entropy
        guess = remaining[0]

    return history


def generate_game_traces(num_secrets=500):
    """Generate game traces for training."""
    all_c = all_codes()  # 1296 codes
    random.seed(42)

    # Sample secrets
    if num_secrets >= len(all_c):
        secrets = all_c
    else:
        secrets = random.sample(all_c, num_secrets)

    traces = []
    for secret in secrets:
        history = knuth_play(secret)
        traces.append((secret, history))

    return traces


# ---- Corpus Generation ----

def feedback_to_str(history):
    """Convert history to compact feedback string."""
    parts = []
    for guess, b, w in history:
        parts.append(f"{guess}:B{b}W{w}")
    return ','.join(parts)


def generate_planner_corpus(traces):
    """Planner: guess count + feedback → plan."""
    entries = []
    seen = set()

    for secret, history in traces:
        for turn_idx in range(len(history)):
            guesses_made = turn_idx
            fb = feedback_to_str(history[:turn_idx]) if turn_idx > 0 else "none"

            if guesses_made == 0:
                prompt = "guesses=0|feedback=none"
            else:
                prompt = f"guesses={guesses_made}|feedback={fb}"

            # Truncate prompt if too long (keep under block_size)
            if len(prompt) > 100:
                prompt = prompt[:100]

            remaining = len(history) - turn_idx
            if remaining <= 1:
                plan = "guess,check"
            elif remaining <= 3:
                plan = "guess,check,guess,check"
            else:
                plan = "guess,check,guess,check,guess,check"

            if prompt not in seen:
                entries.append(f"{prompt}\ntodo={plan}")
                seen.add(prompt)

    return entries


def generate_player_corpus(traces):
    """Player: feedback history → next guess.

    Output is ONLY a 4-char code like "ABCD".
    """
    entries = []
    seen = set()

    for secret, history in traces:
        for turn_idx in range(len(history)):
            guess, b, w = history[turn_idx]

            if turn_idx == 0:
                prompt = "feedback=none"
            else:
                fb = feedback_to_str(history[:turn_idx])
                prompt = f"feedback={fb}"

            # Truncate prompt if too long
            if len(prompt) > 100:
                prompt = prompt[:100]

            if prompt not in seen:
                entries.append(f"{prompt}\n{guess}")
                seen.add(prompt)

            # Blocked variant: if we have subsequent guesses
            if turn_idx + 1 < len(history):
                next_guess = history[turn_idx + 1][0]
                fb_so_far = feedback_to_str(history[:turn_idx + 1])
                prompt_b = f"feedback={fb_so_far}|blocked={guess}"
                if len(prompt_b) > 100:
                    prompt_b = prompt_b[:100]
                if prompt_b not in seen:
                    entries.append(f"{prompt_b}\n{next_guess}")
                    seen.add(prompt_b)

    return entries


# ---- Main ----

def main():
    print("Generating Mastermind training corpora (Knuth algorithm)...\n", file=sys.stderr)

    print("Phase 1: Generating game traces...", file=sys.stderr)
    traces = generate_game_traces(num_secrets=800)
    total_turns = sum(len(h) for _, h in traces)
    avg_turns = total_turns / len(traces) if traces else 0
    print(f"  Secrets: {len(traces)}", file=sys.stderr)
    print(f"  Total turns: {total_turns} (avg {avg_turns:.1f}/game)\n", file=sys.stderr)

    print("Phase 2: Generating planner corpus...", file=sys.stderr)
    planner_entries = generate_planner_corpus(traces)
    print(f"  Planner: {len(planner_entries)} entries", file=sys.stderr)

    print("Phase 3: Generating player corpus...", file=sys.stderr)
    player_entries = generate_player_corpus(traces)
    print(f"  Player: {len(player_entries)} entries", file=sys.stderr)

    # Write files
    print("\nWriting corpus files:", file=sys.stderr)

    for name, entries in [
        ("mastermind_planner.txt", planner_entries),
        ("mastermind_player.txt", player_entries),
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
