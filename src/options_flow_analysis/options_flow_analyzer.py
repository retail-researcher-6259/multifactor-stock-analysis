"""
Options Flow Analyzer - daily change detection report

Unlike the stock stability analyzer (which rewards persistence), this
hunts DISRUPTION: deviations from each stock's own recent normal.
Measured basis (June-July 2026 history, 24 sessions): the score's sign
alone has no predictive value (IC ~ 0), but abnormal-volume events
(rvol >= 3x, extreme imbalance) showed ~+0.32% next-day drift vs +0.01%
universe, in BOTH directions. All alerts below are therefore attention/
risk flags, not directional buy/sell recommendations, until the
long-horizon backfill validates them.

Report sections:
    1. VOLUME SPIKE events - today's abnormal flow (the validated signal)
    2. OI BUILDS - net new positioning from day-over-day open interest
       (directional by nature; needs accumulated daily-scorer history)
    3. WATCHLIST - events intersected with the MSAS top-N ranked list
    4. EXIT ALERTS - events / put builds on current portfolio holdings

Inputs: options_YYYYMMDD.csv session files from BOTH repos (personal
backfill + automated daemon), deduped by DataDate preferring OI-bearing
files; latest MSAS ranked list of the chosen regime; latest portfolio
JSON from config/Portfolio_Lists.

Usage:
    python options_flow_analyzer.py                       (latest session)
    python options_flow_analyzer.py --regime Steady_Growth --top-n 50
    python options_flow_analyzer.py --date 20260706
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ===== CONFIGURATION =====

PERSONAL_ROOT = Path(__file__).parents[2]
AUTOMATED_ROOT = Path(r"C:\Miscellaneous_Programs\Learnings\multifactor-program-automated")

SESSION_DIRS = [
    PERSONAL_ROOT / "output" / "Options_Ranked_Lists",
    AUTOMATED_ROOT / "output" / "Options_Ranked_Lists",
]
MSAS_DIR = PERSONAL_ROOT / "output" / "Ranked_Lists"
PORTFOLIO_DIR = PERSONAL_ROOT / "config" / "Portfolio_Lists"
OUTPUT_DIR = PERSONAL_ROOT / "output" / "Options_Flow_Analysis"

REGIMES = ["Crisis_Bear", "Steady_Growth", "Strong_Bull"]

# Volume spike events (thresholds from the validated event study)
TRAIL_WINDOW = 10        # sessions for the volume baseline
RVOL_SPIKE = 3.0         # today's volume vs trailing mean
MIN_EVENT_VOLUME = 500   # contracts, absolute floor
SCORE_EXTREME = 0.5      # |Score| floor for an event

# OI builds (positioning changes; OI exists only in daemon-era files)
OI_MIN_BASE = 5000       # total OI floor: ignore tiny bases
OI_REL_1D = 0.25         # |net dOI| / prior total OI, 1-session flag
OI_REL_5D = 0.50         # same over 5 sessions

# ===== END CONFIGURATION =====


def load_history():
    """All session files from both repos, deduped by DataDate.

    When the same session exists twice (backfill + daemon), prefer the
    file that carries open interest.
    """
    by_date = {}
    for folder in SESSION_DIRS:
        if not folder.exists():
            continue
        for f in sorted(folder.glob("options_*.csv")):
            try:
                df = pd.read_csv(f)
            except Exception as e:
                print(f"WARNING: could not read {f.name}: {e}")
                continue
            if "DataDate" not in df.columns or df.empty:
                continue
            d = df["DataDate"].iloc[0]
            has_oi = df["CallOI"].notna().any() if "CallOI" in df.columns else False
            if d in by_date and not has_oi:
                continue
            if d not in by_date or has_oi:
                by_date[d] = df
    return by_date


def build_panel(by_date, value):
    frames = {d: df.set_index("Ticker")[value] for d, df in by_date.items()
              if value in df.columns}
    return pd.DataFrame(frames).sort_index(axis=1)


def load_msas_top(regime, top_n):
    """Tickers of the latest MSAS ranked list for the regime."""
    folder = MSAS_DIR / regime
    files = sorted(folder.glob(f"top_ranked_stocks_{regime}_*.csv"))
    if not files:
        print(f"WARNING: no MSAS ranked list found in {folder}")
        return set(), None
    df = pd.read_csv(files[-1])
    return set(df["Ticker"].head(top_n)), files[-1].name


def load_portfolio():
    """Tickers of the newest portfolio JSON (template excluded)."""
    files = sorted(f for f in PORTFOLIO_DIR.glob("portfolio*.json")
                   if "template" not in f.name)
    if not files:
        print(f"WARNING: no portfolio file found in {PORTFOLIO_DIR}")
        return set(), None
    with open(files[-1]) as f:
        tickers = json.load(f).get("tickers", [])
    return {t.strip().upper() for t in tickers}, files[-1].name


def main():
    parser = argparse.ArgumentParser(
        description="Options flow change-detection report")
    parser.add_argument("--regime", choices=REGIMES, default="Strong_Bull",
                        help="MSAS regime for the watchlist (default Strong_Bull)")
    parser.add_argument("--top-n", type=int, default=100, dest="top_n",
                        help="MSAS top-N for the watchlist (default 100)")
    parser.add_argument("--date", default=None,
                        help="Session to analyze, YYYYMMDD (default: latest)")
    args = parser.parse_args()

    by_date = load_history()
    if not by_date:
        print("ERROR: no session files found.")
        sys.exit(1)
    dates = sorted(by_date)

    if args.date:
        target = f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:]}"
        if target not in by_date:
            print(f"ERROR: no session file for {target}. "
                  f"Available: {dates[0]} .. {dates[-1]}")
            sys.exit(1)
    else:
        target = dates[-1]
    hist_dates = [d for d in dates if d < target]

    print(f"{'=' * 70}")
    print(f"OPTIONS FLOW ANALYZER - session {target} "
          f"({len(dates)} sessions loaded, {len(hist_dates)} history)")
    print(f"{'=' * 70}")
    try:
        import pytz
        from datetime import datetime, time as dt_time
        now_et = datetime.now(pytz.timezone("US/Eastern"))
        if (target == now_et.strftime("%Y-%m-%d")
                and now_et.time() < dt_time(16, 15)):
            print("WARNING: this session is still IN PROGRESS in the U.S.;")
            print("volumes are partial and rvol comparisons are biased low.")
    except Exception:
        pass
    print("NOTE: alerts are attention/risk flags, not directional advice;")
    print("the imbalance sign showed no predictive value on 24 sessions.\n")

    score = build_panel(by_date, "Score")
    vol = build_panel(by_date, "TotalVolume")
    call_oi = build_panel(by_date, "CallOI")
    put_oi = build_panel(by_date, "PutOI")
    status = build_panel(by_date, "Status")
    price = build_panel(by_date, "UnderlyingPrice")

    msas_top, msas_name = load_msas_top(args.regime, args.top_n)
    holdings, port_name = load_portfolio()

    report_rows = []

    # ---- 1. VOLUME SPIKE events -------------------------------------
    trail_cols = hist_dates[-TRAIL_WINDOW:]
    trail_mean = vol[trail_cols].mean(axis=1) if trail_cols else None

    events = pd.DataFrame()
    if trail_mean is not None and target in vol.columns:
        ev = pd.DataFrame({
            "Score": score[target],
            "Volume": vol[target],
            "TrailMean": trail_mean,
            "Price": price[target],
            "Status": status[target],
        })
        ev["Rvol"] = ev["Volume"] / ev["TrailMean"]
        ev["ScoreEMA3"] = score[hist_dates + [target]].T.ewm(span=3).mean().iloc[-1]
        events = ev[(ev["Status"] == "OK")
                    & (ev["Score"].abs() >= SCORE_EXTREME)
                    & (ev["Volume"] >= MIN_EVENT_VOLUME)
                    & (ev["TrailMean"] > 0)
                    & (ev["Rvol"] >= RVOL_SPIKE)].copy()
        events["Direction"] = np.where(events["Score"] > 0,
                                       "BULL FLOW", "BEAR FLOW")
        events = events.sort_values("Rvol", ascending=False)

    print(f"1. VOLUME SPIKE EVENTS (rvol >= {RVOL_SPIKE}x, "
          f"vol >= {MIN_EVENT_VOLUME}, |score| >= {SCORE_EXTREME}): "
          f"{len(events)}")
    print("-" * 70)
    if events.empty:
        print("   none\n")
    else:
        with pd.option_context("display.float_format", "{:.2f}".format):
            print(events[["Direction", "Score", "ScoreEMA3", "Volume",
                          "TrailMean", "Rvol", "Price"]].to_string())
        print()
        for tk, row in events.iterrows():
            report_rows.append({
                "Section": "VOLUME_SPIKE", "Ticker": tk,
                "Label": row["Direction"], "Score": row["Score"],
                "Volume": row["Volume"], "Rvol": row["Rvol"],
                "Detail": f"vol {row['Volume']:.0f} vs {row['TrailMean']:.0f} avg",
            })

    # ---- 2. OI BUILDS ------------------------------------------------
    oi_dates = [d for d in dates if d <= target
                and call_oi.get(d) is not None and call_oi[d].notna().any()]
    print(f"2. OI BUILDS (net new positioning, base OI >= {OI_MIN_BASE}):")
    print("-" * 70)
    if target not in oi_dates or len(oi_dates) < 2:
        print(f"   insufficient OI history ({len(oi_dates)} OI session(s); "
              f"need 2+ for 1-day, 6+ for 5-day). Accumulating daily.\n")
        oi_flags = pd.DataFrame()
    else:
        prev = oi_dates[oi_dates.index(target) - 1]
        base = call_oi[prev] + put_oi[prev]
        d_net = (call_oi[target] - call_oi[prev]) - (put_oi[target] - put_oi[prev])
        oi = pd.DataFrame({
            "dCallOI": call_oi[target] - call_oi[prev],
            "dPutOI": put_oi[target] - put_oi[prev],
            "BaseOI": base,
            "Rel1d": d_net / (base + 1),
            "Score": score[target],
        })
        oi_flags = oi[(oi["BaseOI"] >= OI_MIN_BASE)
                      & (oi["Rel1d"].abs() >= OI_REL_1D)].copy()

        if len(oi_dates) >= 6:
            prev5 = oi_dates[oi_dates.index(target) - 5]
            base5 = call_oi[prev5] + put_oi[prev5]
            net5 = ((call_oi[target] - call_oi[prev5])
                    - (put_oi[target] - put_oi[prev5]))
            rel5 = net5 / (base5 + 1)
            oi_flags["Rel5d"] = rel5.reindex(oi_flags.index)
            more = oi[(oi["BaseOI"] >= OI_MIN_BASE)
                      & (rel5.abs() >= OI_REL_5D)
                      & (~oi.index.isin(oi_flags.index))].copy()
            more["Rel5d"] = rel5.reindex(more.index)
            oi_flags = pd.concat([oi_flags, more])

        oi_flags["Label"] = np.where(oi_flags["Rel1d"] > 0,
                                     "CALL BUILD", "PUT BUILD")
        oi_flags = oi_flags.sort_values("Rel1d", key=abs, ascending=False)
        if oi_flags.empty:
            print("   none\n")
        else:
            print(f"   (caution: 1-day deltas near option expiries include "
                  f"roll-off artifacts)")
            with pd.option_context("display.float_format", "{:.3f}".format):
                print(oi_flags[["Label", "dCallOI", "dPutOI", "BaseOI",
                                "Rel1d", "Score"]].head(15).to_string())
            print()
            for tk, row in oi_flags.iterrows():
                report_rows.append({
                    "Section": "OI_BUILD", "Ticker": tk,
                    "Label": row["Label"], "Score": row["Score"],
                    "Volume": np.nan, "Rvol": np.nan,
                    "Detail": f"dCall {row['dCallOI']:+.0f}, "
                              f"dPut {row['dPutOI']:+.0f}, "
                              f"rel {row['Rel1d']:+.2f}",
                })

    # ---- 3. WATCHLIST (events x MSAS) --------------------------------
    print(f"3. WATCHLIST - events on MSAS {args.regime} top {args.top_n} "
          f"({msas_name}):")
    print("-" * 70)
    watch = events[events.index.isin(msas_top)] if not events.empty else pd.DataFrame()
    oi_watch = (oi_flags[oi_flags.index.isin(msas_top)]
                if not oi_flags.empty else pd.DataFrame())
    if watch.empty and oi_watch.empty:
        print("   none\n")
    else:
        for tk, row in watch.iterrows():
            print(f"   {tk}: {row['Direction']} volume spike "
                  f"(rvol {row['Rvol']:.1f}x, score {row['Score']:+.2f})")
            report_rows.append({
                "Section": "WATCHLIST", "Ticker": tk,
                "Label": row["Direction"], "Score": row["Score"],
                "Volume": row["Volume"], "Rvol": row["Rvol"],
                "Detail": "MSAS-ranked stock with abnormal flow",
            })
        for tk, row in oi_watch.iterrows():
            print(f"   {tk}: {row['Label']} (rel {row['Rel1d']:+.2f})")
            report_rows.append({
                "Section": "WATCHLIST", "Ticker": tk,
                "Label": row["Label"], "Score": row["Score"],
                "Volume": np.nan, "Rvol": np.nan,
                "Detail": "MSAS-ranked stock with OI build",
            })
        print()

    # ---- 4. EXIT ALERTS (holdings) ------------------------------------
    print(f"4. EXIT ALERTS - portfolio holdings ({port_name}):")
    print("-" * 70)
    any_alert = False
    for tk in sorted(holdings):
        notes = []
        if not events.empty and tk in events.index:
            row = events.loc[tk]
            sev = ("EXIT REVIEW" if row["Score"] < 0 else "VOLATILITY NOTE")
            notes.append(f"{sev}: {row['Direction']} spike "
                         f"(rvol {row['Rvol']:.1f}x, score {row['Score']:+.2f})")
        if not oi_flags.empty and tk in oi_flags.index:
            row = oi_flags.loc[tk]
            if row["Label"] == "PUT BUILD":
                notes.append(f"EXIT REVIEW: put OI build "
                             f"(rel {row['Rel1d']:+.2f})")
        for n in notes:
            any_alert = True
            print(f"   {tk}: {n}")
            report_rows.append({
                "Section": "EXIT_ALERT", "Ticker": tk,
                "Label": n.split(":")[0], "Score": np.nan,
                "Volume": np.nan, "Rvol": np.nan, "Detail": n,
            })
    if not any_alert:
        print("   none")
    print()

    # ---- save ---------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"flow_report_{target.replace('-', '')}.csv"
    pd.DataFrame(report_rows).to_csv(out, index=False)
    print(f"Report saved to: {out}")


if __name__ == "__main__":
    main()
