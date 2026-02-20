#!/usr/bin/env python3
"""
Generate training corpora for the Markets Multi-Organelle Pipeline.

Reads market_data.csv and produces 3 corpus files:
  1. market_analyser.txt   — cross-asset state summarisation
  2. market_regime.txt     — regime classification from analysis
  3. market_rotator.txt    — sector over/underweight from regime

Encoding: Uses APL-inspired compact ASCII to minimise output length.
Each regime is a single character, analyser output is 5 positional chars,
and rotator uses compact sector notation. This dramatically reduces the
number of tokens the char-level model must predict per sample.

Regime labels:
  R = RISK_ON       (VIX < 18 AND SPY 21d return > 0)
  O = RISK_OFF      (VIX > 25 AND SPY 21d return < 0)
  I = INFLATIONARY  (GLD 21d ret > 0 AND TLT 21d ret < 0)
  D = DEFLATIONARY  (GLD 21d ret < 0 AND TLT 21d ret > 0)
  T = TRANSITIONAL  (everything else)

Analyser output (5 positional chars):
  Pos 1: equity   U(p)/D(own)
  Pos 2: bond     U(p)/D(own)
  Pos 3: commodity U(p)/D(own)
  Pos 4: vol      L(ow)/M(ed)/H(igh)
  Pos 5: USD      S(trong)/W(eak)/F(lat)

Rotator output:
  +EF-KU=V  (+ overweight, - underweight, = neutral, sector initials)

Usage:
    python3 generate_corpus.py"""

import sys
import os

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("ERROR: pandas/numpy not installed. Run: pip install pandas numpy")
    sys.exit(1)


DATA_FILE = "market_data.csv"
ANALYSER_OUT = "market_analyser.txt"
REGIME_OUT = "market_regime.txt"
ROTATOR_OUT = "market_rotator.txt"

# Compact regime label mapping
REGIME_CHAR = {
    "RISK_ON": "R", "RISK_OFF": "O", "INFLATIONARY": "I",
    "DEFLATIONARY": "D", "TRANSITIONAL": "T",
}
CHAR_REGIME = {v: k for k, v in REGIME_CHAR.items()}  # reverse lookup

# Sector rotation rules per regime (using sector initials E/F/K/U/V)
SECTOR_RULES = {
    "R": {"over": "KF", "under": "UV", "neutral": "E"},   # RISK_ON
    "O": {"over": "UV", "under": "KF", "neutral": "E"},   # RISK_OFF
    "I": {"over": "EF", "under": "KU", "neutral": "V"},   # INFLATIONARY
    "D": {"over": "UV", "under": "EF", "neutral": "K"},   # DEFLATIONARY
    "T": {"over": "V",  "under": "E",  "neutral": "KFU"}, # TRANSITIONAL
}


def dir_char(val):
    """Return U(p)/D(own) for positive/negative."""
    if pd.isna(val):
        return "?"
    return "U" if val > 0 else "D"


def vol_char(vix):
    """Single-char VIX level: L(ow)/M(ed)/H(igh)."""
    if pd.isna(vix):
        return "?"
    if vix < 20:
        return "L"
    if vix < 30:
        return "M"
    return "H"


def usd_char(uup_ret):
    """Single-char USD direction: S(trong)/W(eak)/F(lat)."""
    if pd.isna(uup_ret):
        return "?"
    if uup_ret > 0.5:
        return "S"
    if uup_ret < -0.5:
        return "W"
    return "F"


def classify_regime(row):
    """Classify market regime from derived features."""
    vix = row.get("VIX_close", np.nan)
    spy_ret = row.get("SP500_ret21d", np.nan)
    gld_ret = row.get("GOLD_ret21d", np.nan)
    tlt_ret = row.get("BOND_20Y_ret21d", np.nan)

    if pd.isna(vix) or pd.isna(spy_ret):
        return "TRANSITIONAL"

    # Risk-on: low volatility + rising equities
    if vix < 18 and spy_ret > 0:
        return "RISK_ON"

    # Risk-off: high volatility + falling equities
    if vix > 25 and spy_ret < 0:
        return "RISK_OFF"

    # Inflationary: commodities rising, bonds falling
    if not pd.isna(gld_ret) and not pd.isna(tlt_ret):
        if gld_ret > 0 and tlt_ret < 0:
            return "INFLATIONARY"
        if gld_ret < 0 and tlt_ret > 0:
            return "DEFLATIONARY"

    return "TRANSITIONAL"


def build_analysis_prompt(row):
    """Build cross-asset analysis prompt from a data row."""
    parts = []

    # Equity direction
    spy_ret = row.get("SP500_ret1d", np.nan)
    qqq_ret = row.get("NASDAQ_ret1d", np.nan)
    parts.append(f"SPY={spy_ret:+.1f}%" if not pd.isna(spy_ret) else "SPY=?")
    parts.append(f"QQQ={qqq_ret:+.1f}%" if not pd.isna(qqq_ret) else "QQQ=?")

    # Bond direction
    tlt_ret = row.get("BOND_20Y_ret1d", np.nan)
    parts.append(f"TLT={tlt_ret:+.1f}%" if not pd.isna(tlt_ret) else "TLT=?")

    # Commodities
    gld_ret = row.get("GOLD_ret1d", np.nan)
    parts.append(f"GLD={gld_ret:+.1f}%" if not pd.isna(gld_ret) else "GLD=?")

    uso_ret = row.get("OIL_ret1d", np.nan)
    parts.append(f"USO={uso_ret:+.1f}%" if not pd.isna(uso_ret) else "USO=?")

    # VIX level
    vix = row.get("VIX_close", np.nan)
    parts.append(f"VIX={vix:.1f}" if not pd.isna(vix) else "VIX=?")

    # USD direction
    uup_ret = row.get("USD_INDEX_ret1d", np.nan)
    parts.append(f"UUP={uup_ret:+.1f}%" if not pd.isna(uup_ret) else "UUP=?")

    return "|".join(parts)


def build_analysis_output(row):
    """Build compact 5-char analysis output: equity/bond/cmdty/vol/usd.

    Example: 'UDUMS' = equity Up, bond Down, cmdty Up, vol Med, usd Strong.
    """
    return (
        dir_char(row.get("SP500_ret21d", np.nan)) +
        dir_char(row.get("BOND_20Y_ret21d", np.nan)) +
        dir_char(row.get("GOLD_ret21d", np.nan)) +
        vol_char(row.get("VIX_close", np.nan)) +
        usd_char(row.get("USD_INDEX_ret21d", np.nan))
    )


def build_regime_prompt(analysis_output):
    """The regime classifier's prompt is the analyser's output."""
    return analysis_output


def build_rotator_prompt(regime_char, analysis_output):
    """Build rotator prompt: regime char + analysis. E.g. 'R:UDUMS'."""
    return f"{regime_char}:{analysis_output}"


def build_rotator_output(regime_char):
    """Compact sector output: +KF-UV=E (overweight/underweight/neutral)."""
    rules = SECTOR_RULES.get(regime_char, SECTOR_RULES["T"])
    return f"+{rules['over']}-{rules['under']}={rules['neutral']}"


def generate_corpora():
    """Generate all 3 training corpora with class-balanced oversampling.

    IMPORTANT: The last HOLDOUT_DAYS are excluded from training so the C
    pipeline's backtest evaluates on genuinely unseen data (generalization,
    not memorization).
    """
    HOLDOUT_DAYS = 60  # must match BACKTEST_DAYS in main.c

    print(f"Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    print(f"  Loaded {len(df)} trading days × {len(df.columns)} columns")

    # Exclude the last HOLDOUT_DAYS from training corpus
    train_df = df.iloc[:-HOLDOUT_DAYS]
    print(f"  Training on first {len(train_df)} days "
          f"(holding out last {HOLDOUT_DAYS} for backtest)")

    # Group entries by regime for balanced sampling
    regime_groups = {r: {"analyser": [], "regime": [], "rotator": []}
                     for r in ["RISK_ON", "RISK_OFF", "INFLATIONARY",
                               "DEFLATIONARY", "TRANSITIONAL"]}
    regime_counts = {}

    for idx, row in train_df.iterrows():
        # Skip rows with too many NaNs
        if pd.isna(row.get("SP500_ret21d")) or pd.isna(row.get("VIX_close")):
            continue

        # Classify regime
        regime = classify_regime(row)
        regime_char = REGIME_CHAR[regime]
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

        # Organelle 1: Cross-Asset Analyser
        analysis_prompt = build_analysis_prompt(row)
        analysis_output = build_analysis_output(row)
        regime_groups[regime]["analyser"].append(
            f"{analysis_prompt}\n{analysis_output}")

        # Organelle 2: Regime Classifier (single-char output)
        regime_prompt = build_regime_prompt(analysis_output)
        regime_groups[regime]["regime"].append(
            f"{regime_prompt}\n{regime_char}")

        # Organelle 3: Sector Rotator (compact output)
        rotator_prompt = build_rotator_prompt(regime_char, analysis_output)
        rotator_output = build_rotator_output(regime_char)
        regime_groups[regime]["rotator"].append(
            f"{rotator_prompt}\n{rotator_output}")

    # ---- Class-balanced oversampling ----
    max_count = max(regime_counts.values())
    print(f"\n--- Raw Regime Distribution ---")
    total_raw = sum(regime_counts.values())
    for regime in ["RISK_ON", "RISK_OFF", "INFLATIONARY", "DEFLATIONARY",
                    "TRANSITIONAL"]:
        count = regime_counts.get(regime, 0)
        pct = count / total_raw * 100 if total_raw > 0 else 0
        print(f"  {regime:>14s}: {count:5d} ({pct:5.1f}%)")

    print(f"\n  Oversampling minority classes to {max_count} each...")

    analyser_entries = []
    regime_entries = []
    rotator_entries = []

    rng = np.random.RandomState(42)  # reproducible

    for regime_name in ["RISK_ON", "RISK_OFF", "INFLATIONARY",
                        "DEFLATIONARY", "TRANSITIONAL"]:
        group = regime_groups[regime_name]
        n = len(group["analyser"])
        if n == 0:
            continue

        if n >= max_count:
            # Use all entries (no oversampling needed)
            analyser_entries.extend(group["analyser"])
            regime_entries.extend(group["regime"])
            rotator_entries.extend(group["rotator"])
        else:
            # Add all originals
            analyser_entries.extend(group["analyser"])
            regime_entries.extend(group["regime"])
            rotator_entries.extend(group["rotator"])

            # Oversample to fill gap
            extra_needed = max_count - n
            extra_indices = rng.choice(n, size=extra_needed, replace=True)
            for i in extra_indices:
                analyser_entries.append(group["analyser"][i])
                regime_entries.append(group["regime"][i])
                rotator_entries.append(group["rotator"][i])

    # Shuffle all entries
    for entries in [analyser_entries, regime_entries, rotator_entries]:
        rng.shuffle(entries)

    # Write corpus files
    print(f"\n--- Balanced Corpus ---")

    for filename, entries, label in [
        (ANALYSER_OUT, analyser_entries, "Analyser"),
        (REGIME_OUT, regime_entries, "Regime Classifier"),
        (ROTATOR_OUT, rotator_entries, "Sector Rotator"),
    ]:
        with open(filename, "w") as f:
            f.write("\n\n".join(entries))
            f.write("\n")

        file_size = os.path.getsize(filename)

        # Find max doc length
        max_doc = max(len(e) for e in entries) if entries else 0

        print(f"  {filename}: {len(entries)} entries, "
              f"{file_size:,} bytes, max_doc={max_doc} chars")

    # Balanced distribution
    print(f"\n  Regime balance: {max_count} entries per class "
          f"× 5 classes = {max_count * 5} total")

    # Sample entries
    print(f"\n--- Sample Entries ---")
    if analyser_entries:
        print(f"\n  Analyser (first):")
        print(f"    {analyser_entries[0]}")
    if regime_entries:
        print(f"\n  Regime Classifier (first):")
        print(f"    {regime_entries[0]}")
    if rotator_entries:
        print(f"\n  Sector Rotator (first):")
        print(f"    {rotator_entries[0]}")

    print(f"\nDone! Class-balanced corpora ready for MicroGPT-C training.")


if __name__ == "__main__":
    generate_corpora()

