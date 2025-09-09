"""
combined_portfolio_analyzer.py

A unified script that:
1. Takes portfolio holdings (ticker: shares) as input
2. Generates current portfolio allocation donut chart
3. Runs MHRP optimization on the tickers
4. Generates optimized portfolio donut chart
5. Merges both images and saves to ./Images/ folder

Installation:
pip install yahooquery yfinance matplotlib pillow pandas numpy scipy scikit-learn riskfolio-lib

Usage:
python combined_portfolio_analyzer.py
python combined_portfolio_analyzer.py --csv my_portfolio.csv
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import riskfolio as rp
import yfinance as yf
from sklearn.covariance import LedoitWolf
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image
from yahooquery import Ticker
import time
import sys
import os

# === Helper Functions ===

def ensure_directory(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)

def read_csv(path: Path):
    """Return dict {ticker: shares} from a 2-column csv (ticker, shares)."""
    import csv
    holdings = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            ticker, shares = row[0].strip().upper(), float(row[1])
            holdings[ticker] = holdings.get(ticker, 0) + shares
    return holdings

def get_current_prices(tickers):
    """Fetch latest regular-market prices via yahooquery."""
    data = Ticker(list(tickers)).price
    return {
        t: data[t]["regularMarketPrice"]
        for t in tickers
        if data.get(t) and data[t].get("regularMarketPrice") is not None
    }

def compute_weights(holdings, prices):
    """Return (weights, values, total)."""
    values = {t: shares * prices[t] for t, shares in holdings.items() if t in prices}
    total = sum(values.values())
    weights = {t: v / total for t, v in values.items()}
    return weights, values, total

def plot_allocation_donut_to_file(weights, title, filename):
    """Save donut chart to file instead of showing it."""
    if not isinstance(weights, pd.Series):
        weights = pd.Series(weights, name="weights")

    labels = [f"{t} {w*100:.1f}%" for t, w in weights.items()]
    sizes = weights.values

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts = ax.pie(
        sizes,
        labels=labels,
        startangle=90,
        wedgeprops=dict(width=0.35, edgecolor="w"),
    )

    plt.setp(texts, size=9, weight="bold")

    centre_circle = plt.Circle((0, 0), 0.70, fc="white")
    ax.add_artist(centre_circle)
    ax.axis("equal")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()

# === MHRP Functions ===

def get_cluster_volatility(cov, cluster_items):
    """Compute total cluster volatility assuming equal volatility weighting."""
    sub_cov = cov.loc[cluster_items, cluster_items]
    vol = np.sqrt(np.diag(sub_cov))
    inv_vol = 1 / vol
    weights = inv_vol / inv_vol.sum()
    cluster_vol = np.sqrt(np.dot(weights, np.dot(sub_cov.values, weights)))
    return cluster_vol

def recursive_bisection_equal_volatility(cov, sort_ix):
    """Modified Recursive Bisection that uses Equal Volatility inside clusters."""
    weights = pd.Series(1.0, index=sort_ix, dtype='float64')
    clusters = [sort_ix]

    while clusters:
        cluster = clusters.pop(0)
        if len(cluster) <= 1:
            continue

        split = int(len(cluster) / 2)
        left = cluster[:split]
        right = cluster[split:]

        vol_left = get_cluster_volatility(cov, left)
        vol_right = get_cluster_volatility(cov, right)

        alpha = 1 - vol_left / (vol_left + vol_right)
        weights[left] *= alpha
        weights[right] *= 1 - alpha

        clusters += [left, right]

    return weights

def get_quasi_diag(link):
    """Quasi-diagonalization for hierarchical clustering."""
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link.shape[0] + 1
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = (df0.values - num_items).astype(int)
        sort_ix[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df1])
        sort_ix = sort_ix.sort_index()
    return sort_ix.astype(int).tolist()

def fetch_prices(tickers, lookback_days=252, max_retries=3, pause=1.0):
    """Robust download of historical prices."""
    end = datetime.today()
    start = end - timedelta(days=lookback_days * 1.4)
    tries = 0

    while tries < max_retries:
        data = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )

        close = (data["Close"] if isinstance(data.columns, pd.MultiIndex)
                 else data[["Close"]].rename(columns={"Close": tickers[0]}))

        close = close.ffill().dropna(how="all")
        missing_cols = close.columns[close.isna().all()].tolist()

        if not missing_cols:
            break
        else:
            tries += 1
            print(f"⚠️  Missing data for {missing_cols} … retrying ({tries}/{max_retries})")
            tickers = [t for t in tickers if t not in missing_cols]
            time.sleep(pause)

    if missing_cols:
        print(f"❌ Giving up on {missing_cols}. They'll be skipped.", file=sys.stderr)

    prices = close.ffill(limit=2).dropna(axis=0, how="any")
    return prices, missing_cols

def merge_images(left_path, right_path, out_path):
    """Merge two 600x600 images with a 5-pixel black spacer."""
    TARGET_SIZE = (600, 600)
    DIVIDER_W = 5
    DIVIDER_COLOR = (0, 0, 0, 255)

    left = Image.open(left_path).convert("RGBA")
    right = Image.open(right_path).convert("RGBA")

    if left.size != TARGET_SIZE:
        left = left.resize(TARGET_SIZE, Image.LANCZOS)
    if right.size != TARGET_SIZE:
        right = right.resize(TARGET_SIZE, Image.LANCZOS)

    w, h = TARGET_SIZE
    canvas = Image.new("RGBA", (2 * w + DIVIDER_W, h), (255, 255, 255, 0))

    canvas.paste(left, (0, 0))
    divider = Image.new("RGBA", (DIVIDER_W, h), DIVIDER_COLOR)
    canvas.paste(divider, (w, 0))
    canvas.paste(right, (w + DIVIDER_W, 0))

    canvas.save(out_path)
    print(f"✅ Saved merged image → {out_path}")

# === Main Function ===

def main():
    parser = argparse.ArgumentParser(description="Combined portfolio visualizer and MHRP optimizer")
    parser.add_argument("--csv", type=Path, help="CSV file with columns: ticker,shares")
    args = parser.parse_args()

    # Get holdings
    if args.csv:
        holdings = read_csv(args.csv)
    else:
        # Default holdings - EDIT HERE
        holdings = {
            "GLD": 20,
            "VOO": 10,
            "VNOM": 100,
            "FSLR": 50,
            "TSM": 10,
            "QFIN": 300,
            "BIDU": 30,
            "PDD": 15,
            "NVO": 50,
            "KGC": 180,
        }

    # Extract tickers
    tickers = list(holdings.keys())

    # Create Images directory
    ensure_directory("./Images")

    # Generate date string
    date_str = datetime.now().strftime("%m%d")

    # File paths
    current_img = f"./Images/current_{date_str}.png"
    mhrp_img = f"./Images/mhrp_{date_str}.png"
    combined_img = f"./Images/combined_{date_str}.png"

    # === Step 1: Current Portfolio Visualization ===
    print("📊 Generating current portfolio visualization...")

    prices = get_current_prices(tickers)
    if len(prices) != len(holdings):
        missing = set(holdings) - set(prices)
        raise ValueError(f"No price data for: {', '.join(missing)}")

    weights, values, total = compute_weights(holdings, prices)

    print("\n✅ Current Portfolio Weights:")
    print("      weights")
    for t, w in sorted(weights.items()):
        print(f"{t:<5} {w:7.4f}")

    # Save current portfolio chart
    plot_allocation_donut_to_file(weights, f"Current Portfolio - {datetime.now():%Y-%m-%d}", current_img)

    # === Step 2: MHRP Optimization ===
    print("\n🔍 Running MHRP optimization...")

    lookback_days = 180
    prices_hist, dropped = fetch_prices(tickers, lookback_days)
    if dropped:
        print(f"⚠️  Warning: Skipping {dropped} from optimization due to missing data")
        tickers = [t for t in tickers if t not in dropped]

    returns = prices_hist.pct_change().dropna()

    # Set up HCPortfolio
    print("⚙️  Setting up HCPortfolio for MHRP...")
    port = rp.HCPortfolio(returns=returns)

    # Estimate covariance with Ledoit-Wolf
    print("📈 Estimating covariance matrix (Ledoit-Wolf)...")
    lw = LedoitWolf()
    cov_matrix = lw.fit(returns).covariance_
    port.cov = pd.DataFrame(cov_matrix, index=returns.columns, columns=returns.columns)
    port.returns = returns

    # Manual Hierarchical Clustering
    cor = returns.corr(method='spearman')
    distance = np.sqrt(0.5 * (1 - cor))
    link = linkage(squareform(distance), method='ward')
    ticker_order = get_quasi_diag(link)
    sorted_tickers = [returns.columns[i] for i in ticker_order]

    # Apply Equal Volatility Recursive Bisection
    print("⚖️  Applying Equal Volatility Recursive Bisection...")
    mhrp_weights = recursive_bisection_equal_volatility(port.cov, sorted_tickers)
    mhrp_weights = mhrp_weights / mhrp_weights.sum()

    # Reindex to original ticker order for consistency
    mhrp_weights = mhrp_weights.reindex(tickers).dropna()

    print("\n✅ Optimized MHRP Weights:")
    print("      weights")
    for t, w in mhrp_weights.items():
        print(f"{t:<5} {w:7.4f}")

    # Save MHRP portfolio chart
    plot_allocation_donut_to_file(mhrp_weights, f"MHRP Portfolio - {datetime.now():%Y-%m-%d}", mhrp_img)

    # === Step 3: Merge Images ===
    print("\n🎨 Merging images...")
    merge_images(current_img, mhrp_img, combined_img)

    print(f"\n✅ All done! Combined image saved to: {combined_img}")

if __name__ == "__main__":
    main()