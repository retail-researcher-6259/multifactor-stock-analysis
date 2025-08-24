"""
This script builds a Modified Hierarchical Risk Parity (MHRP) portfolio using Riskfolio-Lib.

MHRP is an enhancement of the classic HRP (Hierarchical Risk Parity) method.
It follows three main steps:

1. Hierarchical Clustering:
   - Group assets based on their codependence (e.g., Spearman rank correlation).
   - Perform clustering using a linkage method (e.g., Ward's method).

2. Modified Weighting:
   - Apply Ledoit-Wolf shrinkage to estimate a more robust covariance matrix.
   - Allocate weights within clusters using **equal volatility** (instead of inverse variance).
   - This spreads risk more evenly across assets and clusters.

3. Volatility Targeting (optional):
   - Scale portfolio weights to match a target portfolio volatility.
   - Helps control risk exposure dynamically.

Result:
- A robust portfolio with better diversification, lower turnover, and more stable risk profiles than standard HRP.
- Improved performance especially in volatile markets.

Technical Notes:
- Ledoit-Wolf shrinkage covariance estimation
- Spearman rank correlation as the distance metric
- Ward linkage hierarchical clustering
- Model type: MHRP (Modified Hierarchical Risk Parity)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import riskfolio as rp
import yfinance as yf
from sklearn.covariance import LedoitWolf
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import riskfolio as rp
from datetime import datetime, timedelta
import time, sys

# === Step 0: Define Equal Volatility Weight Calculation ===

def get_cluster_volatility(cov, cluster_items):
    """
    Compute total cluster volatility assuming equal volatility weighting.
    """
    sub_cov = cov.loc[cluster_items, cluster_items]
    vol = np.sqrt(np.diag(sub_cov))  # standard deviation (√variance) for each asset
    inv_vol = 1 / vol
    weights = inv_vol / inv_vol.sum()
    cluster_vol = np.sqrt(np.dot(weights, np.dot(sub_cov.values, weights)))
    return cluster_vol


# === Step 0: Customize Recursive Bisection Using Equal Volatility ===

def recursive_bisection_equal_volatility(cov, sort_ix):
    """
    Modified Recursive Bisection that uses Equal Volatility inside clusters.
    """
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

# === Quasi-Diagonalization ===
def get_quasi_diag(link):
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

def plot_allocation_donut(weights, title="Portfolio Allocation"):
    """
    weights: 1-D array-like, pandas Series or dict (index → weight)
    Produces:
      • Matplotlib donut with ticker + % text
      • riskfolio pie (if riskfolio-lib is available)
    """
    # --- ensure we have a flat Series for both back-ends
    if not isinstance(weights, pd.Series):
        weights = pd.Series(weights, name="weights")

    # ------------- Matplotlib donut -------------
    labels = [f"{t} {w*100:.1f}%" for t, w in weights.items()]
    sizes  = weights.values

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
    plt.show()

    # ------------- riskfolio-lib pie -------------
    if rp is not None:
        try:
            if hasattr(rp, "Plot"):            # older versions
                rp.Plot().plot_pie(weights, title=title, cmap="tab20")
            else:                              # ≥0.4.0
                rp.plot_pie(weights, title=title, cmap="tab20")
        except Exception as e:
            print(f"[riskfolio-lib] pie skipped – {e}")

def fetch_prices(tickers, lookback_days=252, max_retries=3, pause=1.0):
    """
    Robust download:
      • retries tickers that fail individually
      • forward-fills small gaps, drops only if a ticker is still >missing_tol NaNs
    Returns: price DataFrame with *all* surviving tickers, list(missing)
    """
    end   = datetime.today()
    start = end - timedelta(days=lookback_days * 1.4)   # little buffer for non-trading days
    tries = 0

    while tries < max_retries:
        data = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            group_by="column",  # multi-index if >1 ticker
            threads=True,
        )

        # yfinance returns multi-index if >1 ticker
        close = (data["Close"] if isinstance(data.columns, pd.MultiIndex)
                 else data[["Close"]].rename(columns={"Close": tickers[0]}))

        # Forward-fill then drop rows where *all* tickers are still NaN
        close = close.ffill().dropna(how="all")

        # Now see which columns are still entirely NaN
        missing_cols = close.columns[close.isna().all()].tolist()

        if not missing_cols:
            break          # success!
        else:
            tries += 1
            print(f"⚠️  Missing data for {missing_cols} … retrying ({tries}/{max_retries})")
            tickers = [t for t in tickers if t not in missing_cols]
            time.sleep(pause)

    if missing_cols:
        print(f"❌ Giving up on {missing_cols}. They’ll be skipped.", file=sys.stderr)

    # final clean-up: tiny gaps allowed (1–2 trading days)
    prices = close.ffill(limit=2).dropna(axis=0, how="any")   # drop any remaining NaN rows
    return prices, missing_cols

# Step 1: Download price data
tickers = ['GLD', 'NVDA', 'MSFT', 'TSM', 'DUOL', 'PDD', 'UTHR', 'MNST', 'FSLR', 'SEIC', 'TS']
lookback_days = 180

print("🔍 Downloading price data...")
# prices = yf.download(tickers, period=f"{lookback_days}d", interval="1d", auto_adjust=True)["Close"]
# prices = prices.dropna(axis=1)
# returns = prices.pct_change().dropna()

prices, dropped = fetch_prices(tickers, lookback_days)
if dropped:
    raise RuntimeError(f"Cannot continue – data missing for: {dropped}")
returns = prices.pct_change().dropna()

# Step 2: Set up HCPortfolio Object
print("⚙️ Setting up HCPortfolio for MHRP...")
port = rp.HCPortfolio(returns=returns)

# Step 3: Estimate Covariance Matrix with Ledoit-Wolf Shrinkage
print("📈 Estimating covariance matrix (Ledoit-Wolf)...")
lw = LedoitWolf()
cov_matrix = lw.fit(returns).covariance_

port.cov = pd.DataFrame(cov_matrix, index=returns.columns, columns=returns.columns)
port.returns = returns

# Manual Hierarchical Clustering
# Build correlation matrix for clustering
cor = returns.corr(method='spearman')

# Distance matrix
distance = np.sqrt(0.5 * (1 - cor))

# Linkage tree using Ward's method
link = linkage(squareform(distance), method='ward')

# Get ticker order after clustering
ticker_order = get_quasi_diag(link)
sorted_tickers = [returns.columns[i] for i in ticker_order]

# Step 4: Apply Equal Volatility Recursive Bisection (True MHRP)
print("⚖️ Applying Equal Volatility Recursive Bisection...")
mhrp_weights = recursive_bisection_equal_volatility(port.cov, sorted_tickers)
mhrp_weights = mhrp_weights / mhrp_weights.sum()  # normalize

# weights = mhrp_weights.copy()  # use these as final weights

# -- lock chart order / colours to match portfolio_visualizer --
VIS_ORDER = tickers
weights = mhrp_weights.reindex(VIS_ORDER).dropna()        # <- NEW

# Step 5: (Optional) Volatility Targeting
# If you want the portfolio to have a target volatility (e.g., 10%), uncomment below:

target_vol = None  # Set to 0.10 for 10% target volatility, or None to skip scaling
if target_vol is not None:
    # Calculate current annualized portfolio volatility based on weights and covariance matrix.
    port_vol = np.sqrt(np.dot(weights.T, np.dot(port.cov, weights))) * np.sqrt(252)

    # Figure out how much we need to scale all the weights to match the target volatility.
    scaling_factor = target_vol / port_vol

    # Multiply all weights to achieve the desired portfolio volatility.
    weights = weights * scaling_factor

    # If any weight accidentally becomes negative (unlikely, but very defensive), clip it to 0.
    # weights = weights.clip(lower=0)

    # normalize back to 1
    # weights /= weights.sum()

# Step 6: Show results
print("\n✅ Optimized Weights:")
print(weights.round(4))

# Step 7: Plot Pie Chart
print("\n📊 Plotting Portfolio Composition...")
# ax = rp.plot_pie(w=weights,
#                  title='MHRP Portfolio Allocation',
#                  others=0.0,
#                  nrow=25,
#                  cmap="tab20",
#                  height=6,
#                  width=10,
#                  ax=None)
# plt.show()
plot_allocation_donut(weights.squeeze(), "MHRP Portfolio Allocation")