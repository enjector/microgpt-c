#!/usr/bin/env python3
# Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
# MIT License — see LICENSE file for details.
"""
Generate training corpora for the EuroMillions Lottery multi-organelle demo.

Creates 2 corpus files from euromillions-draw-history.csv:
  - lottery_analyser.txt   (recent draws window → hot/cold analysis)
  - lottery_predictor.txt  (hot/cold analysis → next draw's numbers)

DESIGN PRINCIPLE: Keep outputs SHORT and DETERMINISTIC.
  - Analyser outputs "hot=N,N,N|cold=N,N,N|stars_hot=N,N|stars_cold=N,N"
  - Predictor outputs "B1,B2,B3,B4,B5;S1,S2" (5 balls + 2 lucky stars)
  - Corpus: prompt on line 1, output on line 2 (native doc format)

EuroMillions: 5 balls from 1-50, 2 lucky stars from 1-12.

Usage:
  python3 generate_corpus.py                         # default CSV
  python3 generate_corpus.py my-draw-history.csv     # custom CSV
"""

import csv
import sys
from collections import Counter


CSV_FILE = "euromillions-draw-history.csv"
WINDOWS = [5, 10, 15, 20]   # sliding-window sizes for training variety
TOP_HOT = 5                  # number of hot numbers to report
TOP_COLD = 5                 # number of cold numbers to report
TOP_STAR_HOT = 3
TOP_STAR_COLD = 3
MAX_BALLS = 50               # EuroMillions ball range
MAX_STARS = 12               # EuroMillions lucky star range


def load_draws(csv_path):
    """Load draws from CSV, sorted oldest-first (reversed from file order).

    Returns list of dicts: {balls: [int], stars: [int], date: str, draw_num: int}
    """
    draws = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                balls = sorted([
                    int(row["Ball 1"]), int(row["Ball 2"]), int(row["Ball 3"]),
                    int(row["Ball 4"]), int(row["Ball 5"]),
                ])
                stars = sorted([
                    int(row["Lucky Star 1"]), int(row["Lucky Star 2"]),
                ])
                draw_num = row.get("DrawNumber") or "0"
                draws.append({
                    "balls": balls,
                    "stars": stars,
                    "date": row["DrawDate"],
                    "draw_num": int(draw_num) if draw_num.strip() else 0,
                })
            except (ValueError, KeyError):
                continue  # skip malformed rows

    # Reverse so oldest is first (file has newest first)
    draws.reverse()
    return draws


def compute_hot_cold(draws_window, top_hot, top_cold, max_val, key="balls"):
    """Compute hot and cold numbers from a window of draws.

    Hot = most frequent in window.
    Cold = numbers in range [1, max_val] that appear least (or not at all).
    """
    counter = Counter()
    for d in draws_window:
        for n in d[key]:
            counter[n] += 1

    # All possible numbers
    all_nums = set(range(1, max_val + 1))

    # Hot: most common
    hot = [n for n, _ in counter.most_common(top_hot)]

    # Cold: numbers with lowest frequency (including 0)
    freq_list = [(n, counter.get(n, 0)) for n in all_nums]
    freq_list.sort(key=lambda x: (x[1], x[0]))  # lowest freq first, then by number
    cold = [n for n, _ in freq_list[:top_cold]]

    return sorted(hot), sorted(cold)


def recent_draws_str(draws_window, max_recent=3):
    """Format the most recent draws as compact string."""
    recent = draws_window[-max_recent:] if len(draws_window) > max_recent else draws_window
    parts = []
    for d in reversed(recent):  # most recent first
        balls_str = ",".join(str(b) for b in d["balls"])
        stars_str = ",".join(str(s) for s in d["stars"])
        parts.append(f"{balls_str}|{stars_str}")
    return ";".join(parts)


def draw_to_output(draw):
    """Format a single draw as output string: 'B1,B2,B3,B4,B5;S1,S2'.

    Uses semicolon (not pipe) to separate balls from stars, because
    the predictor prompt uses pipes for hot/cold fields — using a
    different separator avoids format confusion in the small model.
    """
    balls_str = ",".join(str(b) for b in draw["balls"])
    stars_str = ",".join(str(s) for s in draw["stars"])
    return f"{balls_str};{stars_str}"


def generate_analyser_corpus(draws):
    """Analyser: recent draws window → hot/cold analysis.

    For each position in the draw history, use the preceding window to
    compute hot/cold, and output the analysis string.
    """
    entries = []
    seen = set()

    for window_size in WINDOWS:
        for i in range(window_size, len(draws)):
            window = draws[i - window_size:i]

            # Build prompt: window size + recent draws summary
            recent = recent_draws_str(window, max_recent=3)
            prompt = f"window={window_size}|recent={recent}"

            # Truncate if too long
            if len(prompt) > 110:
                prompt = prompt[:110]

            # Compute analysis
            hot_b, cold_b = compute_hot_cold(window, TOP_HOT, TOP_COLD, MAX_BALLS, "balls")
            hot_s, cold_s = compute_hot_cold(window, TOP_STAR_HOT, TOP_STAR_COLD, MAX_STARS, "stars")

            output = (
                f"hot={','.join(str(n) for n in hot_b)}"
                f"|cold={','.join(str(n) for n in cold_b)}"
                f"|stars_hot={','.join(str(n) for n in hot_s)}"
                f"|stars_cold={','.join(str(n) for n in cold_s)}"
            )

            key = f"{prompt}→{output}"
            if key not in seen:
                entries.append(f"{prompt}\n{output}")
                seen.add(key)

    return entries


def generate_predictor_corpus(draws):
    """Predictor: hot/cold analysis → next draw numbers.

    For each draw, compute the hot/cold from the preceding window,
    and use the actual draw as the target output.
    """
    entries = []
    seen = set()

    for window_size in WINDOWS:
        for i in range(window_size, len(draws)):
            window = draws[i - window_size:i]
            target_draw = draws[i]

            # Compute analysis (same as analyser would output)
            hot_b, cold_b = compute_hot_cold(window, TOP_HOT, TOP_COLD, MAX_BALLS, "balls")
            hot_s, cold_s = compute_hot_cold(window, TOP_STAR_HOT, TOP_STAR_COLD, MAX_STARS, "stars")

            prompt = (
                f"hot={','.join(str(n) for n in hot_b)}"
                f"|cold={','.join(str(n) for n in cold_b)}"
                f"|stars_hot={','.join(str(n) for n in hot_s)}"
                f"|stars_cold={','.join(str(n) for n in cold_s)}"
            )

            # Truncate if needed
            if len(prompt) > 110:
                prompt = prompt[:110]

            output = draw_to_output(target_draw)

            key = f"{prompt}|{output}"
            if key not in seen:
                # Two-line format: prompt\noutput (native doc format)
                entries.append(f"{prompt}\n{output}")
                seen.add(key)

    return entries


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else CSV_FILE

    print(f"Loading draws from {csv_path}...", file=sys.stderr)
    draws = load_draws(csv_path)
    print(f"  Loaded {len(draws)} draws", file=sys.stderr)

    if len(draws) < 10:
        print("ERROR: Need at least 10 draws for meaningful corpus generation",
              file=sys.stderr)
        sys.exit(1)

    print(f"  Date range: {draws[0]['date']} → {draws[-1]['date']}", file=sys.stderr)
    print(f"  Draw numbers: {draws[0]['draw_num']} → {draws[-1]['draw_num']}", file=sys.stderr)

    print("\nPhase 1: Generating analyser corpus...", file=sys.stderr)
    analyser_entries = generate_analyser_corpus(draws)
    print(f"  Analyser: {len(analyser_entries)} entries", file=sys.stderr)

    print("Phase 2: Generating predictor corpus...", file=sys.stderr)
    predictor_entries = generate_predictor_corpus(draws)
    print(f"  Predictor: {len(predictor_entries)} entries", file=sys.stderr)

    # Write files
    print("\nWriting corpus files:", file=sys.stderr)
    for name, entries in [
        ("lottery_analyser.txt", analyser_entries),
        ("lottery_predictor.txt", predictor_entries),
    ]:
        content = "\n\n".join(entries) + "\n"
        with open(name, "w") as f:
            f.write(content)

        doc_lengths = [len(e) for e in entries]
        max_len = max(doc_lengths) if doc_lengths else 0
        print(f"  {name}: {len(entries)} entries, {len(content)} bytes, "
              f"max_doc={max_len} chars", file=sys.stderr)

    # Print hot/cold summary for the full history
    print("\n--- Current Analysis (full history) ---", file=sys.stderr)
    hot_b, cold_b = compute_hot_cold(draws, TOP_HOT, TOP_COLD, MAX_BALLS, "balls")
    hot_s, cold_s = compute_hot_cold(draws, TOP_STAR_HOT, TOP_STAR_COLD, MAX_STARS, "stars")
    print(f"  Hot balls:  {hot_b}", file=sys.stderr)
    print(f"  Cold balls: {cold_b}", file=sys.stderr)
    print(f"  Hot stars:  {hot_s}", file=sys.stderr)
    print(f"  Cold stars: {cold_s}", file=sys.stderr)

    print("\nDone! Corpora ready for MicroGPT-C training.", file=sys.stderr)


if __name__ == "__main__":
    main()
