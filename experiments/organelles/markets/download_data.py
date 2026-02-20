#!/usr/bin/env python3
"""
Download cross-asset market data via yfinance for the Markets Organelle Pipeline.

Downloads daily OHLCV data for 20 instruments across 6 segments:
  - Equities: SPY, QQQ, IWM
  - Bonds: TLT, IEF, SHY
  - Commodities: GLD, USO, DBA
  - Forex: UUP, FXE, FXY
  - Volatility: ^VIX
  - Sectors: XLK, XLF, XLE, XLV, XLU

Computes derived features per day:
  - Returns: 1-day, 5-day, 21-day
  - Rolling volatility: 21-day
  - RSI: 14-day
  - Yield curve proxy: TLT/SHY ratio change

Saves to market_data.csv — one row per trading day, ~70+ columns.

Usage:
    python3 download_data.py
"""

import sys
import os
from datetime import datetime

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("ERROR: pandas/numpy not installed. Run: pip install pandas numpy")
    sys.exit(1)


# ---- Configuration ----

OUTPUT_FILE = "market_data.csv"
START_DATE = "2014-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# Tickers by segment
TICKERS = {
    # Equities
    "SPY": "SP500",
    "QQQ": "NASDAQ",
    "IWM": "RUSSELL2000",
    # Bonds (duration ladder)
    "TLT": "BOND_20Y",
    "IEF": "BOND_10Y",
    "SHY": "BOND_3Y",
    # Commodities
    "GLD": "GOLD",
    "USO": "OIL",
    "DBA": "AGRICULTURE",
    # Forex
    "UUP": "USD_INDEX",
    "FXE": "EUR",
    "FXY": "JPY",
    # Volatility
    "^VIX": "VIX",
    # Sectors
    "XLK": "TECH",
    "XLF": "FINANCIALS",
    "XLE": "ENERGY",
    "XLV": "HEALTHCARE",
    "XLU": "UTILITIES",
}


def compute_rsi(series, period=14):
    """Compute RSI (Relative Strength Index) for a price series."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def download_and_process():
    """Download market data and compute derived features."""

    print(f"Downloading market data: {START_DATE} → {END_DATE}")
    print(f"  Tickers: {len(TICKERS)}")
    print()

    # Download all tickers at once for efficiency
    ticker_list = list(TICKERS.keys())
    print(f"  Fetching {len(ticker_list)} tickers from Yahoo Finance...")

    raw = yf.download(
        ticker_list,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=True,
    )

    if raw.empty:
        print("ERROR: No data downloaded. Check internet connection.")
        sys.exit(1)

    # Extract close prices for each ticker
    # yfinance returns MultiIndex columns: (field, ticker)
    close_prices = pd.DataFrame()

    for ticker, label in TICKERS.items():
        try:
            if len(ticker_list) > 1:
                col = raw["Close"][ticker]
            else:
                col = raw["Close"]
            close_prices[label] = col
        except KeyError:
            print(f"  WARNING: {ticker} ({label}) not found in download")
            continue

    print(f"\n  Got {len(close_prices)} trading days")
    print(f"  Date range: {close_prices.index[0].strftime('%Y-%m-%d')} → "
          f"{close_prices.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Columns: {len(close_prices.columns)}")

    # Build feature DataFrame
    features = pd.DataFrame(index=close_prices.index)
    features["date"] = close_prices.index.strftime("%Y-%m-%d")

    for label in close_prices.columns:
        price = close_prices[label]

        # Raw close price
        features[f"{label}_close"] = price

        # Returns: 1-day, 5-day, 21-day (percentage)
        features[f"{label}_ret1d"] = price.pct_change(1) * 100
        features[f"{label}_ret5d"] = price.pct_change(5) * 100
        features[f"{label}_ret21d"] = price.pct_change(21) * 100

        # Rolling volatility: 21-day annualised (percentage)
        daily_ret = price.pct_change(1)
        features[f"{label}_vol21d"] = daily_ret.rolling(21).std() * np.sqrt(252) * 100

        # RSI: 14-day
        features[f"{label}_rsi14"] = compute_rsi(price, 14)

    # Derived cross-asset features

    # Yield curve proxy: TLT vs SHY relative performance (steepening/flattening)
    if "TLT_close" in features.columns and "SHY_close" in features.columns:
        tlt_ret = close_prices.get("BOND_20Y", pd.Series(dtype=float)).pct_change(21)
        shy_ret = close_prices.get("BOND_3Y", pd.Series(dtype=float)).pct_change(21)
        features["yield_curve_21d"] = (tlt_ret - shy_ret) * 100

    # Equity-bond correlation: 21-day rolling correlation of SPY vs TLT returns
    if "SP500" in close_prices.columns and "BOND_20Y" in close_prices.columns:
        spy_ret = close_prices["SP500"].pct_change(1)
        tlt_ret = close_prices["BOND_20Y"].pct_change(1)
        features["eq_bond_corr_21d"] = spy_ret.rolling(21).corr(tlt_ret)

    # Gold-USD correlation: 21-day rolling
    if "GOLD" in close_prices.columns and "USD_INDEX" in close_prices.columns:
        gld_ret = close_prices["GOLD"].pct_change(1)
        uup_ret = close_prices["USD_INDEX"].pct_change(1)
        features["gold_usd_corr_21d"] = gld_ret.rolling(21).corr(uup_ret)

    # Note: vix_regime categorisation is done in generate_corpus.py, not here.
    # String columns in the CSV break the C parser.

    # Drop rows with too many NaNs (first ~21 days due to rolling windows)
    min_valid = len(close_prices.columns) * 4  # at least 4 features per ticker valid
    features = features.dropna(thresh=min_valid)

    print(f"\n  Final dataset: {len(features)} rows × {len(features.columns)} columns")

    # Save
    features.to_csv(OUTPUT_FILE, index=False)
    file_size = os.path.getsize(OUTPUT_FILE)
    print(f"\n  Saved to {OUTPUT_FILE} ({file_size:,} bytes)")

    # Summary statistics
    print(f"\n--- Data Summary ---")
    print(f"  Trading days:  {len(features)}")
    print(f"  Features:      {len(features.columns)}")
    print(f"  Date range:    {features['date'].iloc[0]} → {features['date'].iloc[-1]}")

    if "VIX_close" in features.columns:
        vix = features["VIX_close"].dropna()
        print(f"  VIX range:     {vix.min():.1f} → {vix.max():.1f} "
              f"(mean={vix.mean():.1f})")

    print(f"\nDone! Data ready for corpus generation.")


if __name__ == "__main__":
    download_and_process()
