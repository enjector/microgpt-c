#!/usr/bin/env python3
# Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
# MIT License — see LICENSE file for details.
"""
Download EuroMillions draw history from the public API and save as CSV.

Uses: https://github.com/pedro-mealha/euromillions-api
  API: https://euromillions.api.pedromealha.dev/v1/draws

Downloads all draws from 2004 to present, year by year, and writes them
to euromillions-draw-history.csv in the same format as the existing file.

Usage:
  python3 download_results.py                        # download all years (2004–now)
  python3 download_results.py --from 2020            # download from 2020 onwards
  python3 download_results.py --from 2020 --to 2025  # download range

After downloading, re-run generate_corpus.py to regenerate the training corpus.
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime


API_BASE = "https://euromillions.api.pedromealha.dev"
OUTPUT_CSV = "euromillions-draw-history.csv"
FIRST_YEAR = 2004
REQUEST_DELAY = 1.5  # seconds between API calls to avoid rate limiting


def fetch_draws_for_year(year):
    """Fetch all draws for a given year from the API.
    Returns a list of draw dicts or empty list on failure."""

    url = f"{API_BASE}/v1/draws?year={year}"
    print(f"  Fetching {url} ...", end=" ", flush=True)

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "MicroGPT-C-Lottery/1.0")
        req.add_header("Accept", "application/json")

        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        if isinstance(data, list):
            print(f"{len(data)} draws", flush=True)
            return data
        else:
            print(f"unexpected format: {type(data)}", flush=True)
            return []

    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}", flush=True)
        if e.code == 429:
            print("    Rate limited — waiting 10s and retrying...", flush=True)
            time.sleep(10)
            return fetch_draws_for_year(year)
        return []
    except urllib.error.URLError as e:
        print(f"error: {e.reason}", flush=True)
        return []
    except Exception as e:
        print(f"error: {e}", flush=True)
        return []


def parse_draw(draw):
    """Parse an API draw dict into a standardised row dict.

    API v1 format:
      {
        "id": 1,
        "draw_id": 262021,
        "numbers": [1, 5, 24, 42, 50],
        "stars": [3, 12],
        "date": "2024-12-25",
        "has_winner": true,
        "prizes": [...]
      }
    """
    try:
        numbers = [int(n) for n in draw.get("numbers", [])]
        stars = [int(s) for s in draw.get("stars", [])]
        date_str = draw.get("date", "")
        draw_id = draw.get("draw_id", 0)

        if len(numbers) != 5 or len(stars) != 2:
            return None

        # Parse date — could be "2024-12-25" or "Tue, 01 Oct 2024 00:00:00 GMT"
        parsed_date = None
        for fmt in ["%Y-%m-%d", "%a, %d %b %Y %H:%M:%S %Z",
                     "%a, %d %b %Y %H:%M:%S GMT"]:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                break
            except ValueError:
                continue

        if not parsed_date:
            # Try partial parse
            try:
                parsed_date = datetime.fromisoformat(date_str[:10])
            except (ValueError, TypeError):
                return None

        return {
            "date": parsed_date.strftime("%d-%b-%Y"),
            "balls": sorted(numbers),
            "stars": sorted(stars),
            "draw_num": draw_id,
            "dt": parsed_date,
        }

    except (KeyError, TypeError, ValueError):
        return None


def write_csv(rows, output_path):
    """Write rows to CSV in the same format as the existing file.

    Columns: DrawDate,Ball 1,Ball 2,Ball 3,Ball 4,Ball 5,
             Lucky Star 1,Lucky Star 2,UK Millionaire Maker,
             European Millionaire Maker,Ball Set,Machine,DrawNumber
    """
    # Sort newest first
    rows.sort(key=lambda r: r["dt"], reverse=True)

    with open(output_path, "w") as f:
        f.write("DrawDate,Ball 1,Ball 2,Ball 3,Ball 4,Ball 5,"
                "Lucky Star 1,Lucky Star 2,UK Millionaire Maker,"
                "European Millionaire Maker,Ball Set,Machine,DrawNumber\n")

        for row in rows:
            b = row["balls"]
            s = row["stars"]
            f.write(f"{row['date']},{b[0]},{b[1]},{b[2]},{b[3]},{b[4]},"
                    f"{s[0]},{s[1]},\"\",,,{row['draw_num']}\n")

    return len(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Download EuroMillions draw history from API")
    parser.add_argument("--from", dest="from_year", type=int,
                        default=FIRST_YEAR,
                        help=f"Start year (default: {FIRST_YEAR})")
    parser.add_argument("--to", dest="to_year", type=int,
                        default=datetime.now().year,
                        help=f"End year (default: current year)")
    parser.add_argument("--output", default=OUTPUT_CSV,
                        help=f"Output CSV file (default: {OUTPUT_CSV})")
    args = parser.parse_args()

    print(f"EuroMillions Draw History Downloader")
    print(f"  API: {API_BASE}")
    print(f"  Years: {args.from_year}–{args.to_year}")
    print(f"  Output: {args.output}")
    print()

    all_draws = []
    failed_years = []

    for year in range(args.to_year, args.from_year - 1, -1):
        raw_draws = fetch_draws_for_year(year)

        if not raw_draws:
            failed_years.append(year)
            time.sleep(REQUEST_DELAY)
            continue

        parsed = 0
        for d in raw_draws:
            row = parse_draw(d)
            if row:
                all_draws.append(row)
                parsed += 1

        if parsed < len(raw_draws):
            print(f"    Warning: {len(raw_draws) - parsed} draws "
                  f"failed to parse")

        time.sleep(REQUEST_DELAY)

    if not all_draws:
        print("\nERROR: No draws downloaded!", file=sys.stderr)
        sys.exit(1)

    # Deduplicate by date+numbers
    seen = set()
    unique_draws = []
    for row in all_draws:
        key = f"{row['date']}-{row['balls']}-{row['stars']}"
        if key not in seen:
            unique_draws.append(row)
            seen.add(key)

    print(f"\nTotal draws fetched: {len(all_draws)}")
    print(f"Unique draws:       {len(unique_draws)}")

    if failed_years:
        print(f"Failed years:       {failed_years}")

    # Date range
    unique_draws.sort(key=lambda r: r["dt"])
    print(f"Date range:         {unique_draws[0]['date']} → "
          f"{unique_draws[-1]['date']}")

    # Write
    count = write_csv(unique_draws, args.output)
    print(f"\nWritten {count} draws to {args.output}")

    # Stats
    print(f"\n--- Quick Stats ---")
    from collections import Counter
    ball_freq = Counter()
    star_freq = Counter()
    for row in unique_draws:
        for b in row["balls"]:
            ball_freq[b] += 1
        for s in row["stars"]:
            star_freq[s] += 1

    top_balls = ball_freq.most_common(5)
    top_stars = star_freq.most_common(3)
    cold_balls = ball_freq.most_common()[-5:]
    cold_stars = star_freq.most_common()[-3:]

    print(f"  Hot balls:  {[n for n, _ in top_balls]}")
    print(f"  Cold balls: {[n for n, _ in cold_balls]}")
    print(f"  Hot stars:  {[n for n, _ in top_stars]}")
    print(f"  Cold stars: {[n for n, _ in cold_stars]}")

    print(f"\nDone! Now run:")
    print(f"  python3 generate_corpus.py")
    print(f"  # Then delete *.ckpt files and re-run lottery_demo")


if __name__ == "__main__":
    main()
