"""
Performance Tracker Widget for MSAS – v3
Fixed:
  - Realized P&L from closed/partial positions now correctly tracked
  - Time-varying portfolio value series (not static final holdings)
  - P&L metrics now independent of lookback (lookback only affects the chart window)
  - Buy/Sell/Dividend dropdown in Transaction Manager
  - Separate Unrealized / Realized / Total P&L in summary
Added (v3):
  - Day Change ($) and Day Change (%) per position and as a portfolio summary metric
  - Multi-period performance table (1W / 1M / 3M / 6M / YTD / 1Y / All Time)
    compared against selected benchmarks
  - P&L contribution horizontal bar chart (per-ticker unrealized + realized)
  - Performance chart now crops to the chosen lookback window;
    full price history is always fetched internally for accurate period metrics
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QGroupBox, QComboBox,
    QProgressBar, QTabWidget, QFileDialog, QMessageBox,
    QHeaderView, QCheckBox, QScrollArea,
    QLineEdit, QAbstractItemView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPixmap

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

TRANSACTIONS_DIR = PROJECT_ROOT / "config" / "Portfolio_Transactions"
OUTPUT_DIR = PROJECT_ROOT / "output" / "Performance_Tracker_Results"

BENCHMARK_MAP = {
    "S&P 500":      "^GSPC",
    "Dow Jones":    "^DJI",
    "NASDAQ":       "^IXIC",
    "Russell 2000": "^RUT",
}

TX_TYPES      = ["Buy", "Sell", "Dividend"]
PERIOD_LABELS = ["1W", "1M", "3M", "6M", "YTD", "1Y", "All Time"]


# ══════════════════════════════════════════════════════
#  Background thread
# ══════════════════════════════════════════════════════

class PerformanceTrackerThread(QThread):
    progress_update = pyqtSignal(int)
    status_update   = pyqtSignal(str)
    result_ready    = pyqtSignal(dict)
    error_occurred  = pyqtSignal(str)

    def __init__(self, transactions, lookback_key, benchmarks, chart_mode,
                 auto_dividends=False):
        super().__init__()
        self.transactions   = transactions    # list of dicts
        self.lookback_key   = lookback_key    # "All Time" | "1 Year" | "YTD" | …
        self.benchmarks     = benchmarks      # list of benchmark names to overlay
        self.chart_mode     = chart_mode      # "Percentage Return" | "Dollar Value"
        self.auto_dividends = auto_dividends  # fetch dividends automatically from yfinance

    # ── helpers ─────────────────────────────────────────────────────────────

    def _compute_holdings_full(self):
        """
        Process ALL transactions in chronological order and return:
          current       – open positions {ticker: {shares, total_cost, avg_cost, total_commission}}
          realized_by_ticker – {ticker: {realized_pnl, commission}} (closed + partial sells + dividends)
          total_realized_pnl   – realized gains/losses from sells (excl. dividends)
          total_dividend_income
          total_commission     – all commissions ever paid
          total_buy_amount     – sum of all buy notional values (denominator for return %)
          all_tickers          – sorted list of every ticker ever touched
        """
        sorted_txs = sorted(self.transactions, key=lambda x: x["date"])

        open_pos    = {}   # ticker -> {shares, total_cost, total_commission}
        real_tk     = {}   # ticker -> {realized_pnl, commission}
        tot_real    = 0.0
        tot_div     = 0.0
        tot_comm    = 0.0
        tot_buy     = 0.0
        all_tickers = set()

        for tx in sorted_txs:
            ticker     = tx["ticker"].strip().upper()
            shares     = float(tx.get("shares", 0))
            price      = float(tx.get("price", 0))
            commission = float(tx.get("commission", 0))
            tx_type    = tx.get("type", "Buy").strip().capitalize()

            all_tickers.add(ticker)
            tot_comm += commission

            if ticker not in open_pos:
                open_pos[ticker] = {"shares": 0.0, "total_cost": 0.0, "total_commission": 0.0}
            if ticker not in real_tk:
                real_tk[ticker] = {"realized_pnl": 0.0, "commission": 0.0}

            if tx_type == "Buy":
                open_pos[ticker]["shares"]           += shares
                open_pos[ticker]["total_cost"]       += shares * price
                open_pos[ticker]["total_commission"] += commission
                tot_buy += shares * price

            elif tx_type == "Sell":
                h = open_pos[ticker]
                if h["shares"] > 1e-9:
                    avg_cost = h["total_cost"] / h["shares"]
                    realized = (price - avg_cost) * shares - commission
                    h["shares"]     -= shares
                    h["total_cost"]  = max(0.0, h["shares"] * avg_cost)
                    h["total_commission"] += commission
                    real_tk[ticker]["realized_pnl"] += realized
                    real_tk[ticker]["commission"]   += commission
                    tot_real += realized

            elif tx_type == "Dividend":
                # price = dividend per share; shares = how many shares held
                dividend_amt = price * shares - commission
                real_tk[ticker]["realized_pnl"] += dividend_amt
                real_tk[ticker]["commission"]   += commission
                tot_div += dividend_amt

        # Keep only tickers with positive remaining shares
        current = {}
        for t, h in open_pos.items():
            if h["shares"] > 1e-9:
                current[t] = {
                    "shares":            h["shares"],
                    "total_cost":        h["total_cost"],
                    "avg_cost":          h["total_cost"] / h["shares"],
                    "total_commission":  h["total_commission"],
                }

        # Keep only realized entries that have some P&L or commission
        realized_by_ticker = {
            t: d for t, d in real_tk.items()
            if abs(d["realized_pnl"]) > 1e-6 or d["commission"] > 0
        }

        return {
            "current":               current,
            "realized_by_ticker":    realized_by_ticker,
            "total_realized_pnl":    tot_real,
            "total_dividend_income": tot_div,
            "total_commission":      tot_comm,
            "total_buy_amount":      tot_buy,
            "all_tickers":           sorted(all_tickers),
        }

    def _resolve_dates(self):
        """Return (chart_start, end_date) for the selected lookback window."""
        end = datetime.today()
        if self.lookback_key == "All Time":
            dates = [datetime.strptime(tx["date"], "%Y-%m-%d") for tx in self.transactions]
            start = min(dates) if dates else end - timedelta(days=365)
        elif self.lookback_key == "1 Year":
            start = end - timedelta(days=365)
        elif self.lookback_key == "YTD":
            start = datetime(end.year, 1, 1)
        elif self.lookback_key == "6 Months":
            start = end - timedelta(days=182)
        elif self.lookback_key == "3 Months":
            start = end - timedelta(days=91)
        elif self.lookback_key == "1 Month":
            start = end - timedelta(days=30)
        else:
            start = end - timedelta(days=365)
        return start, end

    def _fetch_sector_info(self, tickers):
        """
        Fetch sector/industry for each ticker via yfinance (with sleep).
        ETFs (quoteType == 'ETF') use their yfinance 'category' field instead
        of the empty sector/industry fields, e.g.:
          VOO -> "ETF - Large Blend"
          GLD -> "ETF - Commodities Focused"
        """
        info_map = {}
        for t in tickers:
            try:
                info = yf.Ticker(t).info
                if info.get("quoteType", "").upper() == "ETF":
                    category = (info.get("category") or "").strip()
                    info_map[t] = {
                        "sector":   "ETF",
                        "industry": f"ETF \u2013 {category}" if category else "ETF",
                    }
                else:
                    info_map[t] = {
                        "sector":   info.get("sector")   or "Unknown",
                        "industry": info.get("industry") or "Unknown",
                    }
            except Exception:
                info_map[t] = {"sector": "Unknown", "industry": "Unknown"}
            time.sleep(1)
        return info_map

    def _fetch_auto_dividends(self, all_tickers, realized_by_ticker):
        """
        For every ticker that has NO manually-entered Dividend rows, fetch
        dividend history from yfinance and compute dividend income earned
        based on shares actually held at each ex-dividend date.

        Returns (total_auto_div, updated_realized_by_ticker) where
        updated_realized_by_ticker has the auto dividends merged in.
        """
        # Tickers that already have manual Dividend entries -> skip to avoid double-counting
        manual_div_tickers = {
            tx["ticker"].strip().upper()
            for tx in self.transactions
            if tx.get("type", "Buy").strip().capitalize() == "Dividend"
        }

        # Only the non-Dividend transactions, sorted chronologically
        clean_txs = sorted(
            [tx for tx in self.transactions
             if tx.get("type", "Buy").strip().capitalize() != "Dividend"],
            key=lambda x: x["date"]
        )

        total_auto_div = 0.0
        auto_div_by_ticker: dict[str, float] = {}

        for t in all_tickers:
            if t in manual_div_tickers:
                continue  # user has manual entries -> don't auto-fetch
            try:
                divs = yf.Ticker(t).dividends
                if divs is None or len(divs) == 0:
                    time.sleep(0.5)
                    continue

                # Normalise index to timezone-naive date strings
                divs.index = pd.to_datetime(divs.index).tz_localize(None).normalize()

                # Get this ticker's transactions (buy/sell only)
                ticker_txs = [tx for tx in clean_txs if tx["ticker"].strip().upper() == t]
                if not ticker_txs:
                    time.sleep(0.5)
                    continue

                # Walk dividend dates in order, maintaining a running share count
                cs = 0.0   # shares currently held
                cc = 0.0   # cost basis currently held
                tx_i = 0
                ticker_total_div = 0.0

                for div_date, div_per_share in divs.items():
                    div_date_str = div_date.strftime("%Y-%m-%d")

                    # Apply transactions on or before the ex-dividend date
                    while tx_i < len(ticker_txs) and ticker_txs[tx_i]["date"] <= div_date_str:
                        tx      = ticker_txs[tx_i]
                        s       = float(tx.get("shares", 0))
                        p       = float(tx.get("price",  0))
                        tx_type = tx.get("type", "Buy").strip().capitalize()

                        if tx_type == "Buy":
                            cs += s
                            cc += s * p
                        elif tx_type == "Sell":
                            if cs > 1e-9:
                                avg = cc / cs
                                cs  = max(0.0, cs - s)
                                cc  = max(0.0, cs * avg)
                        tx_i += 1

                    if cs > 1e-9 and div_per_share > 0:
                        ticker_total_div += cs * float(div_per_share)

                if ticker_total_div > 0:
                    auto_div_by_ticker[t] = ticker_total_div
                    total_auto_div += ticker_total_div

                time.sleep(0.5)
            except Exception:
                time.sleep(0.5)
                continue

        # Merge into realized_by_ticker
        for t, amt in auto_div_by_ticker.items():
            if t not in realized_by_ticker:
                realized_by_ticker[t] = {"realized_pnl": 0.0, "commission": 0.0}
            realized_by_ticker[t]["realized_pnl"] += amt

        return total_auto_div, auto_div_by_ticker

    def _fetch_price_data(self, tickers, start_date, end_date):
        """Download daily close prices for all given tickers over the given window."""
        if not tickers:
            return pd.DataFrame()
        try:
            data = yf.download(
                tickers,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
            )
            if len(tickers) == 1:
                price_df = data[["Close"]].rename(columns={"Close": tickers[0]})
            elif isinstance(data.columns, pd.MultiIndex):
                price_df = data["Close"]
            else:
                price_df = data

            # Normalise index (drop timezone if present)
            price_df.index = pd.to_datetime(price_df.index).normalize()
            price_df = price_df.ffill()
            return price_df
        except Exception:
            return pd.DataFrame()

    def _build_portfolio_series(self, price_df):
        """
        Build a daily portfolio-value series using TIME-VARYING holdings.

        For each calendar day in price_df, we replay transactions up to that
        date and compute: sum(shares_held_i * price_i).
        """
        if price_df.empty:
            return pd.Series(dtype=float)

        sorted_txs = sorted(self.transactions, key=lambda x: x["date"])

        current_shares: dict[str, float] = {}
        current_cost:   dict[str, float] = {}
        tx_idx = 0

        values = []
        dates  = []

        for date in price_df.index:
            date_str = date.strftime("%Y-%m-%d")

            # Apply every transaction whose date <= today
            while tx_idx < len(sorted_txs) and sorted_txs[tx_idx]["date"] <= date_str:
                tx      = sorted_txs[tx_idx]
                t       = tx["ticker"].strip().upper()
                s       = float(tx.get("shares", 0))
                p       = float(tx.get("price",  0))
                tx_type = tx.get("type", "Buy").strip().capitalize()

                if tx_type == "Buy":
                    current_shares[t] = current_shares.get(t, 0.0) + s
                    current_cost[t]   = current_cost.get(t,   0.0) + s * p

                elif tx_type == "Sell":
                    held = current_shares.get(t, 0.0)
                    if held > 1e-9:
                        avg  = current_cost.get(t, 0.0) / held
                        new  = max(0.0, held - s)
                        current_shares[t] = new
                        current_cost[t]   = max(0.0, new * avg)
                        if new < 1e-9:
                            current_shares.pop(t, None)
                            current_cost.pop(t, None)
                # Dividend doesn't change share count
                tx_idx += 1

            if not current_shares:
                continue

            value = 0.0
            for t, held in current_shares.items():
                if t in price_df.columns:
                    px = price_df.loc[date, t]
                    if pd.notna(px) and px > 0:
                        value += held * px

            if value > 0:
                values.append(value)
                dates.append(date)

        return pd.Series(values, index=pd.DatetimeIndex(dates))

    # ── chart generators ─────────────────────────────────────────────────────

    def _plot_donut_charts(self, holdings, sector_info):
        """Side-by-side donut charts: allocation by ticker and by industry."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tickers = list(holdings.keys())
        values  = {t: holdings[t]["market_value"] for t in tickers}
        total   = sum(values.values())
        if total <= 0:
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=100, facecolor="white")

        # Left: by ticker
        labels_t = [f"{t}\n{values[t]/total*100:.1f}%" for t in tickers if values[t] > 0]
        sizes_t  = [values[t] for t in tickers if values[t] > 0]
        colors_t = plt.cm.Set3(np.linspace(0, 1, len(sizes_t)))
        ax1.pie(sizes_t, labels=labels_t, colors=colors_t, startangle=90,
                wedgeprops=dict(width=0.35, edgecolor="white"),
                textprops=dict(color="black", weight="bold", size=9))
        ax1.add_artist(plt.Circle((0, 0), 0.70, fc="white", linewidth=0))
        ax1.set_title("Allocation by Ticker", fontsize=12, fontweight="bold", color="black")
        ax1.axis("equal")

        # Right: by industry
        ind_vals: dict[str, float] = {}
        for t in tickers:
            ind = sector_info.get(t, {}).get("industry", "Unknown")
            ind_vals[ind] = ind_vals.get(ind, 0.0) + values.get(t, 0.0)
        ind_labels = [f"{ind}\n{v/total*100:.1f}%" for ind, v in ind_vals.items() if v > 0]
        ind_sizes  = [v for v in ind_vals.values() if v > 0]
        colors_i   = plt.cm.Paired(np.linspace(0, 1, len(ind_sizes)))
        ax2.pie(ind_sizes, labels=ind_labels, colors=colors_i, startangle=90,
                wedgeprops=dict(width=0.35, edgecolor="white"),
                textprops=dict(color="black", weight="bold", size=8))
        ax2.add_artist(plt.Circle((0, 0), 0.70, fc="white", linewidth=0))
        ax2.set_title("Allocation by Industry", fontsize=12, fontweight="bold", color="black")
        ax2.axis("equal")

        fig.patch.set_facecolor("white")
        plt.tight_layout()
        filepath = str(OUTPUT_DIR / "allocation_donuts.png")
        fig.savefig(filepath, dpi=100, facecolor="white", edgecolor="none")
        plt.close(fig)
        return filepath

    def _plot_performance_chart(self, portfolio_series, benchmark_data, chart_mode,
                                chart_start=None):
        """
        Line chart: portfolio value (or % return) vs selected benchmarks.
        Both series are cropped to chart_start before plotting, so the user's
        lookback selection controls the display window while the full history
        is retained internally for the multi-period return table.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Crop to the user-selected lookback window for display only
        if chart_start is not None:
            ts = pd.Timestamp(chart_start)
            portfolio_series = portfolio_series[portfolio_series.index >= ts]
            benchmark_data   = {k: v[v.index >= ts] for k, v in benchmark_data.items()}

        fig, ax = plt.subplots(figsize=(12, 5), dpi=100, facecolor="white")
        has_data = False

        if chart_mode == "Percentage Return":
            if len(portfolio_series) > 1:
                port_ret = (portfolio_series / portfolio_series.iloc[0] - 1) * 100
                ax.plot(port_ret.index, port_ret.values,
                        label="My Portfolio", linewidth=2, color="#7c4dff")
                has_data = True
            for name, series in benchmark_data.items():
                if len(series) > 1:
                    series = series[series.index >= portfolio_series.index[0]] if len(portfolio_series) > 0 else series
                    if len(series) > 0:
                        bench_ret = (series / series.iloc[0] - 1) * 100
                        ax.plot(bench_ret.index, bench_ret.values,
                                label=name, linewidth=1.5, alpha=0.75)
                        has_data = True
            ax.set_ylabel("Return (%)", fontsize=11, color="black")
            ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)

        else:  # Dollar Value
            if len(portfolio_series) > 1:
                ax.plot(portfolio_series.index, portfolio_series.values,
                        label="My Portfolio", linewidth=2, color="#7c4dff")
                has_data = True
            ax.set_ylabel("Portfolio Value ($)", fontsize=11, color="black")

        if not has_data:
            ax.text(0.5, 0.5, "Insufficient price data for the selected period",
                    ha="center", va="center", transform=ax.transAxes, color="gray")

        ax.set_xlabel("Date", fontsize=11, color="black")
        ax.set_title("Portfolio Performance", fontsize=14, fontweight="bold", color="black")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("white")
        ax.tick_params(colors="black")
        fig.patch.set_facecolor("white")
        plt.tight_layout()

        filepath = str(OUTPUT_DIR / "performance_chart.png")
        fig.savefig(filepath, dpi=100, facecolor="white", edgecolor="none")
        plt.close(fig)
        return filepath

    def _compute_period_returns(self, portfolio_series, benchmark_data):
        """
        Compute % returns for each standard lookback period.
        Returns: {period_label: {"My Portfolio": pct, benchmark_name: pct, ...}}
        Periods where data doesn't extend far enough return no entry for that source.
        """
        if len(portfolio_series) == 0:
            return {}

        end = portfolio_series.index[-1]
        cutoffs = {
            "1W":      end - pd.Timedelta(weeks=1),
            "1M":      end - pd.DateOffset(months=1),
            "3M":      end - pd.DateOffset(months=3),
            "6M":      end - pd.DateOffset(months=6),
            "YTD":     pd.Timestamp(end.year, 1, 1),
            "1Y":      end - pd.DateOffset(years=1),
            "All Time": portfolio_series.index[0],
        }

        def _pct_return(series, cutoff):
            ts = pd.Timestamp(cutoff)
            s = series[series.index >= ts]
            if len(s) < 2:
                return None
            try:
                base = float(s.iloc[0])
                end  = float(s.iloc[-1])
            except (TypeError, ValueError):
                return None
            return ((end / base) - 1) * 100 if base > 0 else None

        results = {}
        for label in PERIOD_LABELS:
            cutoff = cutoffs[label]
            row = {}
            r = _pct_return(portfolio_series, cutoff)
            if r is not None:
                row["My Portfolio"] = r
            for bname, bseries in benchmark_data.items():
                r = _pct_return(bseries, cutoff)
                if r is not None:
                    row[bname] = r
            if row:
                results[label] = row

        return results

    def _plot_pnl_contribution_chart(self, stock_data, realized_data):
        """
        Horizontal bar chart: total P&L contribution per position.
        Each bar = unrealized P&L (open positions) + realized P&L (closed/partial).
        Sorted ascending (worst at top, best at bottom) for natural reading.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Merge unrealized + realized per ticker
        ticker_pnl: dict[str, float] = {}
        for s in stock_data:
            ticker_pnl[s["ticker"]] = ticker_pnl.get(s["ticker"], 0.0) + s["pnl_dollar"]
        for r in realized_data:
            ticker_pnl[r["ticker"]] = ticker_pnl.get(r["ticker"], 0.0) + r["realized_pnl"]

        if not ticker_pnl:
            return None

        # Sort ascending so best performer is at the bottom of the chart
        sorted_items = sorted(ticker_pnl.items(), key=lambda x: x[1])
        tickers = [x[0] for x in sorted_items]
        pnls    = [x[1] for x in sorted_items]
        colors  = ["#4CAF50" if v >= 0 else "#f44336" for v in pnls]
        max_abs = max(abs(v) for v in pnls) if pnls else 1.0

        fig_h = max(4, len(tickers) * 0.55 + 1.5)
        fig, ax = plt.subplots(figsize=(10, fig_h), dpi=100, facecolor="white")
        bars = ax.barh(tickers, pnls, color=colors, edgecolor="white", height=0.6)

        # Value labels beside each bar
        for bar, val in zip(bars, pnls):
            offset = max_abs * 0.015
            if val >= 0:
                x_pos, ha = val + offset, "left"
            else:
                x_pos, ha = val - offset, "right"
            sign = "+" if val >= 0 else ""
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                    f"{sign}${val:,.0f}", va="center", ha=ha,
                    fontsize=8, color="black")

        ax.axvline(x=0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Total P&L ($)", fontsize=11, color="black")
        ax.set_title("P&L Contribution by Position", fontsize=13, fontweight="bold", color="black")
        ax.set_facecolor("white")
        ax.tick_params(colors="black")
        ax.grid(True, axis="x", alpha=0.3)
        fig.patch.set_facecolor("white")
        plt.tight_layout()

        filepath = str(OUTPUT_DIR / "pnl_contribution.png")
        fig.savefig(filepath, dpi=100, facecolor="white", edgecolor="none")
        plt.close(fig)
        return filepath

    # ── main run ─────────────────────────────────────────────────────────────

    def run(self):
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            # ── Step 1: Compute full P&L ──────────────────────────────────
            self.status_update.emit("Parsing transactions...")
            self.progress_update.emit(5)

            computed            = self._compute_holdings_full()
            current_holdings    = computed["current"]
            realized_by_ticker  = computed["realized_by_ticker"]
            total_realized_pnl  = computed["total_realized_pnl"]
            total_div_income    = computed["total_dividend_income"]
            total_commission    = computed["total_commission"]
            total_buy_amount    = computed["total_buy_amount"]
            all_tickers         = computed["all_tickers"]

            if not all_tickers:
                self.error_occurred.emit("No transactions found.")
                return

            # chart_start = user's lookback selection (for display cropping only)
            # full_start  = earliest transaction date (always used for data fetching)
            chart_start, end_date = self._resolve_dates()
            all_tx_dates = [datetime.strptime(tx["date"], "%Y-%m-%d") for tx in self.transactions]
            full_start   = min(all_tx_dates) if all_tx_dates else chart_start

            # ── Step 2: Sector/industry for open tickers ──────────────────
            self.status_update.emit("Fetching sector/industry data...")
            self.progress_update.emit(10)
            open_tickers = list(current_holdings.keys())
            sector_info  = self._fetch_sector_info(open_tickers) if open_tickers else {}

            # ── Step 2b: Auto-fetch dividends (optional) ─────────────────
            if self.auto_dividends:
                self.status_update.emit("Fetching dividend history from yfinance...")
                self.progress_update.emit(20)
                auto_div_total, _ = self._fetch_auto_dividends(all_tickers, realized_by_ticker)
                total_div_income += auto_div_total

            # ── Step 3: Historical prices – ALWAYS from full_start ────────
            # Fetching from the earliest transaction ensures:
            #   a) _build_portfolio_series has the complete timeline
            #   b) _compute_period_returns can compute all standard periods
            # The performance chart is later cropped to chart_start for display.
            self.status_update.emit("Fetching historical prices (full history)...")
            self.progress_update.emit(30)
            price_df = self._fetch_price_data(all_tickers, full_start, end_date)

            # ── Step 4: Current prices + Day Change -> unrealized P&L ─────
            self.status_update.emit("Computing metrics...")
            self.progress_update.emit(50)

            for t in open_tickers:
                h = current_holdings[t]

                # Prefer the bulk-downloaded series; fall back to a fresh 5-day fetch
                if t in price_df.columns:
                    series_t = price_df[t].dropna()
                else:
                    series_t = pd.Series(dtype=float)

                if len(series_t) == 0:
                    try:
                        d = yf.download(t, period="5d", auto_adjust=True, progress=False)
                        series_t = d["Close"].dropna()
                    except Exception:
                        series_t = pd.Series(dtype=float)

                if len(series_t) >= 2:
                    h["current_price"]     = float(series_t.iloc[-1])
                    prev_close             = float(series_t.iloc[-2])
                    h["day_change_dollar"] = h["current_price"] - prev_close
                    h["day_change_pct"]    = (
                        (h["day_change_dollar"] / prev_close * 100) if prev_close > 0 else 0.0
                    )
                elif len(series_t) == 1:
                    h["current_price"]     = float(series_t.iloc[-1])
                    h["day_change_dollar"] = 0.0
                    h["day_change_pct"]    = 0.0
                else:
                    h["current_price"]     = h["avg_cost"]
                    h["day_change_dollar"] = 0.0
                    h["day_change_pct"]    = 0.0

                h["market_value"] = h["shares"] * h["current_price"]
                h["pnl_dollar"]   = h["market_value"] - h["total_cost"]
                h["pnl_pct"]      = (
                    (h["pnl_dollar"] / h["total_cost"] * 100) if h["total_cost"] > 0 else 0.0
                )

            total_market_value  = sum(h["market_value"] for h in current_holdings.values())
            total_open_cost     = sum(h["total_cost"]   for h in current_holdings.values())
            total_unrealized    = total_market_value - total_open_cost
            total_pnl_dollar    = total_realized_pnl + total_div_income + total_unrealized
            total_pnl_pct       = (
                (total_pnl_dollar / total_buy_amount * 100) if total_buy_amount > 0 else 0.0
            )

            # Portfolio-level day change (sum of per-share change * shares held)
            total_day_change_dollar = sum(
                h.get("day_change_dollar", 0.0) * h["shares"]
                for h in current_holdings.values()
            )
            total_day_change_pct = (
                (total_day_change_dollar / total_market_value * 100)
                if total_market_value > 0 else 0.0
            )

            # Weight %
            for t, h in current_holdings.items():
                h["weight"] = (
                    (h["market_value"] / total_market_value * 100)
                    if total_market_value > 0 else 0.0
                )

            # ── Step 5: Portfolio value time series ───────────────────────
            portfolio_series = self._build_portfolio_series(price_df)

            # Annualized return (full period, earliest tx to today)
            days_held = (end_date - full_start).days
            if days_held > 30 and total_buy_amount > 0:
                ratio = 1.0 + (total_pnl_dollar / total_buy_amount)
                annualized_return = ((max(ratio, 1e-9) ** (365.0 / days_held)) - 1) * 100
            else:
                annualized_return = 0.0

            # Max drawdown (on portfolio value series)
            max_drawdown = 0.0
            if len(portfolio_series) > 1:
                cummax = portfolio_series.cummax()
                valid  = cummax > 0
                if valid.any():
                    dd = (portfolio_series[valid] - cummax[valid]) / cummax[valid]
                    max_drawdown = float(dd.min()) * 100

            # ── Step 6: Benchmark data (full history for period returns) ──
            self.status_update.emit("Fetching benchmark data...")
            self.progress_update.emit(70)

            benchmark_data: dict[str, pd.Series] = {}
            for name in self.benchmarks:
                symbol = BENCHMARK_MAP.get(name)
                if symbol:
                    try:
                        bdata = yf.download(
                            symbol,
                            start=full_start.strftime("%Y-%m-%d"),
                            end=end_date.strftime("%Y-%m-%d"),
                            auto_adjust=True, progress=False,
                        )
                        if len(bdata) > 0:
                            s = bdata["Close"].ffill().dropna()
                            s.index = pd.to_datetime(s.index).normalize()
                            benchmark_data[name] = s
                        time.sleep(1)
                    except Exception:
                        pass

            # ── Step 6b: Multi-period return table ────────────────────────
            period_returns = self._compute_period_returns(portfolio_series, benchmark_data)

            # ── Step 7: Build data lists ──────────────────────────────────
            self.status_update.emit("Generating charts...")
            self.progress_update.emit(85)

            stock_data = [
                {
                    "ticker":            t,
                    "shares":            h["shares"],
                    "avg_cost":          h["avg_cost"],
                    "current_price":     h["current_price"],
                    "market_value":      h["market_value"],
                    "pnl_dollar":        h["pnl_dollar"],
                    "pnl_pct":           h["pnl_pct"],
                    "weight":            h["weight"],
                    "day_change_dollar": h.get("day_change_dollar", 0.0),
                    "day_change_pct":    h.get("day_change_pct",    0.0),
                    "sector":            sector_info.get(t, {}).get("sector",   "Unknown"),
                    "industry":          sector_info.get(t, {}).get("industry", "Unknown"),
                }
                for t, h in current_holdings.items()
            ]

            realized_data = [
                {
                    "ticker":       t,
                    "realized_pnl": d["realized_pnl"],
                    "commission":   d["commission"],
                }
                for t, d in sorted(realized_by_ticker.items())
            ]

            # ── Step 8: Charts ────────────────────────────────────────────
            donut_path = (
                self._plot_donut_charts(current_holdings, sector_info)
                if current_holdings else None
            )
            perf_path = self._plot_performance_chart(
                portfolio_series, benchmark_data, self.chart_mode,
                chart_start=chart_start,
            )
            pnl_path = self._plot_pnl_contribution_chart(stock_data, realized_data)

            self.progress_update.emit(100)
            self.status_update.emit("Done!")

            self.result_ready.emit({
                "stock_data":               stock_data,
                "realized_data":            realized_data,
                "total_market_value":       total_market_value,
                "total_open_cost":          total_open_cost,
                "total_unrealized_pnl":     total_unrealized,
                "total_realized_pnl":       total_realized_pnl,
                "total_dividend_income":    total_div_income,
                "total_pnl_dollar":         total_pnl_dollar,
                "total_pnl_pct":            total_pnl_pct,
                "total_commission":         total_commission,
                "total_buy_amount":         total_buy_amount,
                "total_day_change_dollar":  total_day_change_dollar,
                "total_day_change_pct":     total_day_change_pct,
                "annualized_return":        annualized_return,
                "max_drawdown":             max_drawdown,
                "period_returns":           period_returns,
                "donut_chart_path":         donut_path,
                "performance_chart_path":   perf_path,
                "pnl_contribution_path":    pnl_path,
            })

        except Exception as e:
            import traceback
            self.error_occurred.emit(f"{e}\n\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════
#  Main Widget
# ══════════════════════════════════════════════════════

class PerformanceTrackerWidget(QWidget):
    """Performance Tracker – 6th tab of MSAS."""

    def __init__(self):
        super().__init__()
        self.tracker_thread  = None
        self.current_results = None
        self.transactions:   list      = []
        self.loaded_file:    str | None = None
        self.init_ui()

    # ── Top-level UI ─────────────────────────────────────────────────────────

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)

        title = QLabel("Performance Tracker")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 18, QFont.Bold))
        main_layout.addWidget(title)

        self.sub_tabs = QTabWidget()
        self.sub_tabs.setStyleSheet("QTabBar::tab { padding: 8px 16px; font-size: 11px; }")

        self.tx_tab   = QWidget()
        self.dash_tab = QWidget()
        self._build_transaction_tab()
        self._build_dashboard_tab()
        self.sub_tabs.addTab(self.tx_tab,   "Transaction Manager")
        self.sub_tabs.addTab(self.dash_tab, "Performance Dashboard")

        main_layout.addWidget(self.sub_tabs)
        self.setLayout(main_layout)

    # ══════════════════════════════════════════════════════
    #  Sub-tab 1: Transaction Manager
    # ══════════════════════════════════════════════════════

    def _build_transaction_tab(self):
        layout = QVBoxLayout()

        info = QLabel(
            "Enter transactions below.  Type dropdown: Buy / Sell / Dividend.  "
            "For Dividend rows: 'Price' = dividend per share, 'Shares' = shares you held.\n"
            "Save as JSON (recommended) or CSV.  The lookback period on the Dashboard only "
            "affects the performance chart — all P&L metrics always show all-time figures."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaa; font-size: 10px; padding: 4px;")
        layout.addWidget(info)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Portfolio Name:"))
        self.portfolio_name_edit = QLineEdit("My Portfolio")
        self.portfolio_name_edit.setMaximumWidth(320)
        name_row.addWidget(self.portfolio_name_edit)
        name_row.addStretch()
        layout.addLayout(name_row)

        # Transaction table
        self.tx_table = QTableWidget()
        self.tx_table.setColumnCount(6)
        self.tx_table.setHorizontalHeaderLabels([
            "Ticker", "Type", "Price ($)", "Shares", "Date (YYYY-MM-DD)", "Commission ($)"
        ])
        self.tx_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tx_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tx_table.setAlternatingRowColors(True)
        self.tx_table.setRowCount(0)
        for _ in range(5):
            self._add_row()
        layout.addWidget(self.tx_table)

        # Button row
        btn_row = QHBoxLayout()
        add_btn  = QPushButton("Add Row");       add_btn.clicked.connect(self._add_row);         btn_row.addWidget(add_btn)
        del_btn  = QPushButton("Delete Row(s)"); del_btn.clicked.connect(self._delete_rows);     btn_row.addWidget(del_btn)
        btn_row.addStretch()
        load_btn = QPushButton("Load Transactions")
        load_btn.setStyleSheet("background-color:#2196F3;")
        load_btn.clicked.connect(self._load_transactions)
        btn_row.addWidget(load_btn)
        save_btn = QPushButton("Save Transactions")
        save_btn.setStyleSheet("background-color:#4CAF50;")
        save_btn.clicked.connect(self._save_transactions)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

        self.tx_tab.setLayout(layout)

    # ── Transaction table helpers ─────────────────────────────────────────────

    def _make_type_combo(self, value: str = "Buy") -> QComboBox:
        """Return a styled QComboBox pre-set to `value`."""
        cb = QComboBox()
        cb.addItems(TX_TYPES)
        idx = cb.findText(value, Qt.MatchFixedString)
        if idx >= 0:
            cb.setCurrentIndex(idx)
        cb.setStyleSheet("background-color:#3c3c3c; color:white; padding:2px;")
        return cb

    def _add_row(self, tx_type: str = "Buy"):
        row = self.tx_table.rowCount()
        self.tx_table.insertRow(row)
        self.tx_table.setCellWidget(row, 1, self._make_type_combo(tx_type))

    def _delete_rows(self):
        rows = sorted({idx.row() for idx in self.tx_table.selectedIndexes()}, reverse=True)
        if not rows:
            QMessageBox.information(self, "Delete", "Select at least one row first.")
            return
        for r in rows:
            self.tx_table.removeRow(r)

    def _read_table_transactions(self) -> list:
        """Harvest all non-empty rows from the transaction table."""
        txs = []
        for row in range(self.tx_table.rowCount()):
            ticker_item = self.tx_table.item(row, 0)
            if not ticker_item or not ticker_item.text().strip():
                continue
            combo   = self.tx_table.cellWidget(row, 1)
            tx_type = combo.currentText() if combo else "Buy"

            def _cell(col):
                it = self.tx_table.item(row, col)
                return it.text().strip() if it else ""

            try:
                txs.append({
                    "ticker":     ticker_item.text().strip().upper(),
                    "type":       tx_type,
                    "price":      float(_cell(2)) if _cell(2) else 0.0,
                    "shares":     float(_cell(3)) if _cell(3) else 0.0,
                    "date":       _cell(4) or datetime.today().strftime("%Y-%m-%d"),
                    "commission": float(_cell(5)) if _cell(5) else 0.0,
                })
            except ValueError:
                pass  # skip malformed rows
        return txs

    def _populate_table(self, txs: list):
        """Fill the transaction table from a list of dicts."""
        self.tx_table.setRowCount(0)
        for tx in txs:
            row = self.tx_table.rowCount()
            self.tx_table.insertRow(row)
            self.tx_table.setItem(row, 0, QTableWidgetItem(tx.get("ticker", "")))
            self.tx_table.setCellWidget(row, 1, self._make_type_combo(tx.get("type", "Buy")))
            self.tx_table.setItem(row, 2, QTableWidgetItem(str(tx.get("price", ""))))
            self.tx_table.setItem(row, 3, QTableWidgetItem(str(tx.get("shares", ""))))
            self.tx_table.setItem(row, 4, QTableWidgetItem(tx.get("date", "")))
            self.tx_table.setItem(row, 5, QTableWidgetItem(str(tx.get("commission", 0))))

    def _save_transactions(self):
        txs = self._read_table_transactions()
        if not txs:
            QMessageBox.warning(self, "Save", "No transactions to save."); return

        name = self.portfolio_name_edit.text().strip() or "My Portfolio"
        safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in name)
        default = str(TRANSACTIONS_DIR / f"portfolio_{safe}.json")

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Transactions", default,
            "JSON Files (*.json);;CSV Files (*.csv)"
        )
        if not path:
            return

        if path.endswith(".csv"):
            pd.DataFrame(txs).to_csv(path, index=False)
        else:
            with open(path, "w") as f:
                json.dump({"portfolio_name": name,
                           "created": datetime.today().strftime("%Y-%m-%d"),
                           "transactions": txs}, f, indent=2)

        self.loaded_file  = path
        self.transactions = txs
        self._refresh_portfolio_selector()
        QMessageBox.information(self, "Save", f"Saved to:\n{path}")

    def _load_transactions(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Transactions", str(TRANSACTIONS_DIR),
            "JSON Files (*.json);;CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        try:
            if path.endswith(".csv"):
                txs  = pd.read_csv(path).to_dict(orient="records")
                name = Path(path).stem
            else:
                with open(path) as f:
                    data = json.load(f)
                txs  = data.get("transactions", [])
                name = data.get("portfolio_name", Path(path).stem)

            self.transactions = txs
            self.loaded_file  = path
            self.portfolio_name_edit.setText(name)
            self._populate_table(txs)
            self._refresh_portfolio_selector()
            QMessageBox.information(self, "Load", f"Loaded {len(txs)} transactions from:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load:\n{e}")

    # ══════════════════════════════════════════════════════
    #  Sub-tab 2: Performance Dashboard
    # ══════════════════════════════════════════════════════

    def _build_dashboard_tab(self):
        layout = QVBoxLayout()

        # ── Controls ─────────────────────────────────────────────────────
        ctrl_group  = QGroupBox("Dashboard Controls")
        ctrl_layout = QHBoxLayout()

        # Portfolio selector
        p_col = QVBoxLayout()
        p_col.addWidget(QLabel("Portfolio File:"))
        self.portfolio_combo = QComboBox()
        self.portfolio_combo.setMinimumWidth(200)
        self._refresh_portfolio_selector()
        p_col.addWidget(self.portfolio_combo)
        load_p_btn = QPushButton("Load Portfolio")
        load_p_btn.clicked.connect(self._load_portfolio_from_selector)
        p_col.addWidget(load_p_btn)
        ctrl_layout.addLayout(p_col)

        # Lookback
        lb_col = QVBoxLayout()
        lb_col.addWidget(QLabel("Lookback (chart):"))
        self.lookback_combo = QComboBox()
        self.lookback_combo.addItems(["All Time", "1 Year", "YTD", "6 Months", "3 Months", "1 Month"])
        lb_col.addWidget(self.lookback_combo)
        lb_col.addStretch()
        ctrl_layout.addLayout(lb_col)

        # Benchmarks
        bench_col = QVBoxLayout()
        bench_col.addWidget(QLabel("Benchmarks:"))
        self.bench_checks: dict[str, QCheckBox] = {}
        for name in BENCHMARK_MAP:
            cb = QCheckBox(name)
            if name == "S&P 500":
                cb.setChecked(True)
            bench_col.addWidget(cb)
            self.bench_checks[name] = cb
        ctrl_layout.addLayout(bench_col)

        # Chart mode + dividend option
        mode_col = QVBoxLayout()
        mode_col.addWidget(QLabel("Chart Mode:"))
        self.chart_mode_combo = QComboBox()
        self.chart_mode_combo.addItems(["Percentage Return", "Dollar Value"])
        mode_col.addWidget(self.chart_mode_combo)
        self.auto_div_check = QCheckBox("Auto-fetch dividends\n(from yfinance)")
        self.auto_div_check.setChecked(True)
        self.auto_div_check.setToolTip(
            "Automatically compute dividend income from yfinance for each held period.\n"
            "Tickers with manually entered Dividend rows are skipped to avoid double-counting."
        )
        mode_col.addWidget(self.auto_div_check)
        mode_col.addStretch()
        ctrl_layout.addLayout(mode_col)

        # Run button
        run_col = QVBoxLayout()
        run_col.addStretch()
        self.run_btn = QPushButton("Run Performance Tracker")
        self.run_btn.setStyleSheet("background-color:#7c4dff; font-weight:bold; padding:12px;")
        self.run_btn.clicked.connect(self._run_tracker)
        run_col.addWidget(self.run_btn)
        run_col.addStretch()
        ctrl_layout.addLayout(run_col)

        ctrl_group.setLayout(ctrl_layout)
        layout.addWidget(ctrl_group)

        # ── Summary Metrics (3-row grid) ─────────────────────────────────
        metrics_group = QGroupBox("Summary Metrics")
        mg = QGridLayout()
        mg.setHorizontalSpacing(30)

        self.lbl_total_value    = QLabel("Current Value: --")
        self.lbl_total_value.setFont(QFont("Arial", 10, QFont.Bold))
        self.lbl_open_cost      = QLabel("Open Cost Basis: --")
        self.lbl_unrealized_pnl = QLabel("Unrealized P&L: --")
        self.lbl_realized_pnl   = QLabel("Realized P&L: --")
        self.lbl_dividend       = QLabel("Dividends: --")
        self.lbl_total_pnl      = QLabel("Total P&L: --")
        self.lbl_total_pnl.setFont(QFont("Arial", 10, QFont.Bold))
        self.lbl_total_invested = QLabel("Total Invested: --")
        self.lbl_commission     = QLabel("Commissions: --")
        self.lbl_ann_return     = QLabel("Ann. Return: --")
        self.lbl_max_dd         = QLabel("Max Drawdown: --")
        self.lbl_day_change     = QLabel("Today's Change: --")
        self.lbl_day_change.setFont(QFont("Arial", 10, QFont.Bold))

        for col, lbl in enumerate([self.lbl_total_value, self.lbl_open_cost,
                                    self.lbl_unrealized_pnl, self.lbl_realized_pnl,
                                    self.lbl_dividend]):
            mg.addWidget(lbl, 0, col)

        for col, lbl in enumerate([self.lbl_total_pnl, self.lbl_total_invested,
                                    self.lbl_commission, self.lbl_ann_return,
                                    self.lbl_max_dd]):
            mg.addWidget(lbl, 1, col)

        # "Today's Change" gets its own prominent row, spanning 2 columns
        mg.addWidget(self.lbl_day_change, 2, 0, 1, 2)

        metrics_group.setLayout(mg)
        layout.addWidget(metrics_group)

        # ── Open Positions Table ─────────────────────────────────────────
        open_group  = QGroupBox("Open Positions (Unrealized)")
        open_layout = QVBoxLayout()
        self.holdings_table = QTableWidget()
        self.holdings_table.setColumnCount(10)
        self.holdings_table.setHorizontalHeaderLabels([
            "Ticker", "Shares", "Avg Cost", "Current Price",
            "Market Value", "Unrealized P&L ($)", "Unrealized P&L (%)", "Weight (%)",
            "Day Change ($)", "Day Change (%)",
        ])
        self.holdings_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.holdings_table.setAlternatingRowColors(True)
        self.holdings_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        open_layout.addWidget(self.holdings_table)
        open_group.setLayout(open_layout)
        layout.addWidget(open_group)

        # ── Realized Gains Table ─────────────────────────────────────────
        real_group  = QGroupBox("Realized Gains / Losses (closed & partial sells, dividends)")
        real_layout = QVBoxLayout()
        self.realized_table = QTableWidget()
        self.realized_table.setColumnCount(3)
        self.realized_table.setHorizontalHeaderLabels([
            "Ticker", "Realized P&L ($)", "Commission Paid ($)"
        ])
        self.realized_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.realized_table.setAlternatingRowColors(True)
        self.realized_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.realized_table.setMaximumHeight(200)
        real_layout.addWidget(self.realized_table)
        real_group.setLayout(real_layout)
        layout.addWidget(real_group)

        # ── Multi-Period Performance Table ───────────────────────────────
        period_group  = QGroupBox("Multi-Period Performance (% Return)")
        period_layout = QVBoxLayout()
        period_layout.addWidget(QLabel(
            "Rows: your portfolio + selected benchmarks.  "
            "Periods without sufficient history show N/A.",
            styleSheet="color:#aaa; font-size:10px;"
        ))
        self.period_table = QTableWidget()
        self.period_table.setColumnCount(len(PERIOD_LABELS))
        self.period_table.setHorizontalHeaderLabels(PERIOD_LABELS)
        self.period_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.period_table.setAlternatingRowColors(True)
        self.period_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.period_table.verticalHeader().setVisible(True)
        self.period_table.setMaximumHeight(160)
        period_layout.addWidget(self.period_table)
        period_group.setLayout(period_layout)
        layout.addWidget(period_group)

        # ── Charts (inside scroll area) ──────────────────────────────────
        charts_group  = QGroupBox("Charts")
        charts_layout = QVBoxLayout()

        self.donut_label = QLabel("Allocation charts will appear here after running.")
        self.donut_label.setAlignment(Qt.AlignCenter)
        self.donut_label.setMinimumHeight(180)
        self.donut_label.setStyleSheet("background-color:#1e1e1e; border-radius:5px; padding:8px;")
        charts_layout.addWidget(self.donut_label)

        self.perf_label = QLabel("Performance chart will appear here after running.")
        self.perf_label.setAlignment(Qt.AlignCenter)
        self.perf_label.setMinimumHeight(180)
        self.perf_label.setStyleSheet("background-color:#1e1e1e; border-radius:5px; padding:8px;")
        charts_layout.addWidget(self.perf_label)

        self.pnl_label = QLabel("P&L contribution chart will appear here after running.")
        self.pnl_label.setAlignment(Qt.AlignCenter)
        self.pnl_label.setMinimumHeight(180)
        self.pnl_label.setStyleSheet("background-color:#1e1e1e; border-radius:5px; padding:8px;")
        charts_layout.addWidget(self.pnl_label)

        charts_group.setLayout(charts_layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(charts_group)
        layout.addWidget(scroll)

        # ── Progress ─────────────────────────────────────────────────────
        prog_group  = QGroupBox("Progress")
        prog_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        prog_layout.addWidget(self.progress_bar)
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color:#4CAF50; font-size:10px;")
        prog_layout.addWidget(self.status_label)
        prog_group.setLayout(prog_layout)
        layout.addWidget(prog_group)

        self.dash_tab.setLayout(layout)

    # ── Dashboard actions ─────────────────────────────────────────────────────

    def _refresh_portfolio_selector(self):
        self.portfolio_combo.clear()
        TRANSACTIONS_DIR.mkdir(parents=True, exist_ok=True)
        files = sorted(TRANSACTIONS_DIR.glob("*.json")) + sorted(TRANSACTIONS_DIR.glob("*.csv"))
        for f in files:
            self.portfolio_combo.addItem(f.name, str(f))
        if self.portfolio_combo.count() == 0:
            self.portfolio_combo.addItem("(no saved portfolios)")

    def _load_portfolio_from_selector(self):
        path = self.portfolio_combo.currentData()
        if not path or not Path(path).exists():
            QMessageBox.warning(self, "Load", "No valid file selected."); return
        try:
            if path.endswith(".csv"):
                txs = pd.read_csv(path).to_dict(orient="records")
            else:
                with open(path) as f:
                    data = json.load(f)
                txs = data.get("transactions", [])
            self.transactions = txs
            self.loaded_file  = path
            self.status_label.setText(f"Loaded {len(txs)} transactions from {Path(path).name}")
            self.status_label.setStyleSheet("color:#4CAF50; font-size:10px;")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed:\n{e}")

    def _run_tracker(self):
        # Fall back to reading the Transaction Manager table if no file loaded
        if not self.transactions:
            self.transactions = self._read_table_transactions()
        if not self.transactions:
            QMessageBox.warning(self, "No Data",
                "No transactions loaded.  Please save or load transactions first.")
            return

        lookback       = self.lookback_combo.currentText()
        benchmarks     = [n for n, cb in self.bench_checks.items() if cb.isChecked()]
        chart_mode     = self.chart_mode_combo.currentText()
        auto_dividends = self.auto_div_check.isChecked()

        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting\u2026")
        self.status_label.setStyleSheet("color:#FFC107; font-size:10px;")

        self.tracker_thread = PerformanceTrackerThread(
            transactions=self.transactions,
            lookback_key=lookback,
            benchmarks=benchmarks,
            chart_mode=chart_mode,
            auto_dividends=auto_dividends,
        )
        self.tracker_thread.progress_update.connect(self.progress_bar.setValue)
        self.tracker_thread.status_update.connect(self._on_status)
        self.tracker_thread.result_ready.connect(self._on_results)
        self.tracker_thread.error_occurred.connect(self._on_error)
        self.tracker_thread.start()

    def _on_status(self, msg: str):
        self.status_label.setText(msg)

    def _on_error(self, msg: str):
        self.status_label.setText(f"Error: {msg}")
        self.status_label.setStyleSheet("color:#f44336; font-size:10px;")
        self.run_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", msg[:800])

    def _on_results(self, results: dict):
        self.current_results = results
        self.run_btn.setEnabled(True)
        self.status_label.setText("Tracking complete!")
        self.status_label.setStyleSheet("color:#4CAF50; font-size:10px;")

        def pnl_color(v): return "#4CAF50" if v >= 0 else "#f44336"
        def fmt_pnl(v):   return f"{'+'if v>=0 else ''}${v:,.2f}"

        # ── Summary metrics ──────────────────────────────────────────────
        self.lbl_total_value.setText(f"Current Value: ${results['total_market_value']:,.2f}")
        self.lbl_open_cost.setText(f"Open Cost Basis: ${results['total_open_cost']:,.2f}")

        u = results["total_unrealized_pnl"]
        self.lbl_unrealized_pnl.setText(f"Unrealized P&L: {fmt_pnl(u)}")
        self.lbl_unrealized_pnl.setStyleSheet(f"color:{pnl_color(u)};")

        r = results["total_realized_pnl"]
        self.lbl_realized_pnl.setText(f"Realized P&L: {fmt_pnl(r)}")
        self.lbl_realized_pnl.setStyleSheet(f"color:{pnl_color(r)};")

        dv = results["total_dividend_income"]
        self.lbl_dividend.setText(f"Dividends: {fmt_pnl(dv)}")
        self.lbl_dividend.setStyleSheet(f"color:{pnl_color(dv)};")

        tp = results["total_pnl_dollar"]
        pp = results["total_pnl_pct"]
        self.lbl_total_pnl.setText(f"Total P&L: {fmt_pnl(tp)} ({'+' if pp>=0 else ''}{pp:.2f}%)")
        self.lbl_total_pnl.setStyleSheet(f"color:{pnl_color(tp)}; font-weight:bold;")

        self.lbl_total_invested.setText(f"Total Invested: ${results['total_buy_amount']:,.2f}")
        self.lbl_commission.setText(f"Commissions: ${results['total_commission']:,.2f}")
        self.lbl_ann_return.setText(f"Ann. Return: {results['annualized_return']:+.2f}%")
        self.lbl_max_dd.setText(f"Max Drawdown: {results['max_drawdown']:.2f}%")

        dc = results.get("total_day_change_dollar", 0.0)
        dp = results.get("total_day_change_pct",    0.0)
        self.lbl_day_change.setText(
            f"Today's Change:  {'+'if dc>=0 else ''}${dc:,.2f}"
            f"  ({'+'if dp>=0 else ''}{dp:.2f}%)"
        )
        self.lbl_day_change.setStyleSheet(f"color:{pnl_color(dc)}; font-weight:bold;")

        # ── Open positions table ─────────────────────────────────────────
        sd = results["stock_data"]
        self.holdings_table.setRowCount(len(sd))
        for row, s in enumerate(sd):
            self.holdings_table.setItem(row, 0, QTableWidgetItem(s["ticker"]))
            self.holdings_table.setItem(row, 1, QTableWidgetItem(f"{s['shares']:.4f}"))
            self.holdings_table.setItem(row, 2, QTableWidgetItem(f"${s['avg_cost']:.2f}"))
            self.holdings_table.setItem(row, 3, QTableWidgetItem(f"${s['current_price']:.2f}"))
            self.holdings_table.setItem(row, 4, QTableWidgetItem(f"${s['market_value']:,.2f}"))

            item5 = QTableWidgetItem(f"{'+'if s['pnl_dollar']>=0 else ''}${s['pnl_dollar']:,.2f}")
            item5.setForeground(QColor("#4CAF50") if s["pnl_dollar"] >= 0 else QColor("#f44336"))
            self.holdings_table.setItem(row, 5, item5)

            item6 = QTableWidgetItem(f"{'+'if s['pnl_pct']>=0 else ''}{s['pnl_pct']:.2f}%")
            item6.setForeground(QColor("#4CAF50") if s["pnl_pct"] >= 0 else QColor("#f44336"))
            self.holdings_table.setItem(row, 6, item6)

            self.holdings_table.setItem(row, 7, QTableWidgetItem(f"{s['weight']:.1f}%"))

            dc_d = s.get("day_change_dollar", 0.0)
            item8 = QTableWidgetItem(f"{'+'if dc_d>=0 else ''}${dc_d:,.2f}")
            item8.setForeground(QColor("#4CAF50") if dc_d >= 0 else QColor("#f44336"))
            self.holdings_table.setItem(row, 8, item8)

            dc_p = s.get("day_change_pct", 0.0)
            item9 = QTableWidgetItem(f"{'+'if dc_p>=0 else ''}{dc_p:.2f}%")
            item9.setForeground(QColor("#4CAF50") if dc_p >= 0 else QColor("#f44336"))
            self.holdings_table.setItem(row, 9, item9)

        # ── Realized gains table ─────────────────────────────────────────
        rd = results["realized_data"]
        self.realized_table.setRowCount(len(rd))
        for row, d in enumerate(rd):
            self.realized_table.setItem(row, 0, QTableWidgetItem(d["ticker"]))
            rp = d["realized_pnl"]
            item1 = QTableWidgetItem(f"{'+'if rp>=0 else ''}${rp:,.2f}")
            item1.setForeground(QColor("#4CAF50") if rp >= 0 else QColor("#f44336"))
            self.realized_table.setItem(row, 1, item1)
            self.realized_table.setItem(row, 2, QTableWidgetItem(f"${d['commission']:.2f}"))

        # ── Multi-period returns table ────────────────────────────────────
        period_returns = results.get("period_returns", {})
        if period_returns:
            # Collect all row names; always put "My Portfolio" first
            all_sources: set[str] = set()
            for pd_data in period_returns.values():
                all_sources.update(pd_data.keys())
            row_names = ["My Portfolio"] + sorted(s for s in all_sources if s != "My Portfolio")

            self.period_table.setRowCount(len(row_names))
            self.period_table.setVerticalHeaderLabels(row_names)

            for col, period in enumerate(PERIOD_LABELS):
                period_data = period_returns.get(period, {})
                for row_idx, source in enumerate(row_names):
                    val = period_data.get(source)
                    if val is not None:
                        sign = "+" if val >= 0 else ""
                        item = QTableWidgetItem(f"{sign}{val:.2f}%")
                        item.setTextAlignment(Qt.AlignCenter)
                        item.setForeground(QColor("#4CAF50") if val >= 0 else QColor("#f44336"))
                    else:
                        item = QTableWidgetItem("N/A")
                        item.setTextAlignment(Qt.AlignCenter)
                        item.setForeground(QColor("#888888"))
                    self.period_table.setItem(row_idx, col, item)

        # ── Charts ───────────────────────────────────────────────────────
        chart_map = [
            ("donut_chart_path",       self.donut_label, 1000, 420),
            ("performance_chart_path", self.perf_label,  1000, 420),
            ("pnl_contribution_path",  self.pnl_label,   1000, 500),
        ]
        for path_key, label, max_w, max_h in chart_map:
            img_path = results.get(path_key)
            if img_path and Path(img_path).exists():
                px = QPixmap(img_path)
                label.setPixmap(px.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
