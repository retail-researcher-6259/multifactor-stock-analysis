"""
Leading Factor Recommender v2 - confluence of stability and options flow

Successor to output/Score_Trend_Analysis_Results/leading_factor_recommender.py.
The v1 script blended two single-snapshot factor scores; this version uses
the accumulated options session history on three clocks:

    SLOW    stability_adjusted_score rank   -> WHAT deserves to be owned
    MEDIUM  5-session open-interest trend   -> is conviction building/eroding
    FAST    today's flow events (rvol)      -> is something happening NOW

Selection authority stays with the stability analysis: the universe is its
top-N for the chosen regime, and options signals only accelerate, delay,
or protect. Tiers: STRONG BUY / BUY / HOLD / SELL / STRONG SELL, each with
a descriptive reason. Reminder from the June-July event study: the flow
score's SIGN alone showed no predictive value; tiers therefore lean on OI
positioning (directional by construction) and abnormal-volume events, and
remain hypotheses until the long-horizon backfill validates them.

Inputs:
    - latest stability_analysis_results_<regime>_*.csv of the chosen regime
    - options session history (both repos), deduped by DataDate,
      OI-bearing files preferred
Output:
    output/Options_Flow_Analysis/leading_factor_recommendations_
        <regime>_<session>.csv

Usage:
    python leading_factor_recommender.py                    (interactive)
    python leading_factor_recommender.py --regime Steady_Growth --top-n 50
"""

import argparse
import sys
from datetime import datetime, time as dt_time
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

# ===== CONFIGURATION =====

PERSONAL_ROOT = Path(__file__).parents[2]
AUTOMATED_ROOT = Path(r"C:\Miscellaneous_Programs\Learnings\multifactor-program-automated")

SESSION_DIRS = [
    PERSONAL_ROOT / "output" / "Options_Ranked_Lists",
    AUTOMATED_ROOT / "output" / "Options_Ranked_Lists",
]
STABILITY_DIR = PERSONAL_ROOT / "output" / "Score_Trend_Analysis_Results"
OUTPUT_DIR = PERSONAL_ROOT / "output" / "Options_Flow_Analysis"

REGIMES = ["Crisis_Bear", "Steady_Growth", "Strong_Bull"]

# FAST clock: abnormal-volume event (validated thresholds)
TRAIL_WINDOW = 10
RVOL_SPIKE = 3.0
MIN_EVENT_VOLUME = 500
SCORE_EXTREME = 0.5

# MEDIUM clock: 5-session OI trend (1-session fallback while history short)
OI_SESSIONS = 5
OI_MIN_BASE = 5000
OI_TREND_5D = 0.15          # |net dOI| / base to call a build over 5 sessions
OI_TREND_1D = 0.25          # fallback threshold for a single session

# STRONG tiers require this stability rank or better (absolute, so the
# meaning does not change when the universe is the full list)
STRONG_RANK_MAX = 25

# ===== END CONFIGURATION =====


def load_history():
    """Session files from both repos, deduped by DataDate (prefer OI)."""
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
            if d not in by_date or (has_oi and not (
                    "CallOI" in by_date[d].columns
                    and by_date[d]["CallOI"].notna().any())):
                by_date[d] = df
    return by_date


def build_panel(by_date, value):
    frames = {d: df.set_index("Ticker")[value] for d, df in by_date.items()
              if value in df.columns}
    return pd.DataFrame(frames).sort_index(axis=1)


def load_stability(regime, top_n):
    folder = STABILITY_DIR / regime
    files = sorted(folder.glob(f"stability_analysis_results_{regime}_*.csv"))
    if not files:
        print(f"ERROR: no stability results in {folder}")
        sys.exit(1)
    df = pd.read_csv(files[-1]).copy()
    df["StabilityRank"] = range(1, len(df) + 1)
    if top_n:
        df = df.head(top_n)
    return df, files[-1].name


def prompt_regime():
    print("Select the current market regime (from the regime detection "
          "system):")
    for i, r in enumerate(REGIMES, 1):
        print(f"  {i}. {r}")
    while True:
        c = input(f"Enter 1-{len(REGIMES)}: ").strip()
        if c.isdigit() and 1 <= int(c) <= len(REGIMES):
            return REGIMES[int(c) - 1]
        print("Invalid choice, try again.")


def recommend(row, strong_rank_max):
    """Map the three clocks to a tier + reason. First match wins.

    OI positioning stays valid even when today's volume is thin, so the
    OI branches come before the no-flow fallback.
    """
    bull_ev = row["Event"] == "BULL"
    bear_ev = row["Event"] == "BEAR"
    call_build = row["OITrend"] == "CALL BUILD"
    put_build = row["OITrend"] == "PUT BUILD"
    ema3 = row["ScoreEMA3"] if pd.notna(row["ScoreEMA3"]) else 0.0
    strong_rank = row["StabilityRank"] <= strong_rank_max

    if put_build and bear_ev:
        return "STRONG SELL - Exit signal: put positioning plus bear flow event"
    if put_build and ema3 < -0.2:
        return "SELL - Protect: sustained put positioning with negative flow"
    if put_build:
        return "HOLD - Caution: put OI building against quality rank, watch closely"
    if bear_ev:
        return "HOLD - Delay entry: bear flow event, reassess in 2-3 sessions"
    if call_build and bull_ev and strong_rank:
        return "STRONG BUY - Enter now: OI build and flow event aligned"
    if call_build and strong_rank:
        return "STRONG BUY - Enter: call OI building on top-quality name"
    if call_build:
        return "BUY - Accumulate: call OI building"
    if bull_ev:
        return "BUY - Enter: bull flow event (no OI confirmation yet)"
    if not row["HasFlow"]:
        return ("AMBIGUOUS - Not enough options data to judge "
                "(thin or no options market)")
    if ema3 >= 0.3:
        return "BUY - Accumulate: positive flow tilt"
    if ema3 <= -0.3:
        return "HOLD - Wait: negative flow tilt"
    return "HOLD - Wait: neutral flow"


TIER_ORDER = ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL",
              "AMBIGUOUS"]


def main():
    parser = argparse.ArgumentParser(
        description="Confluence recommender: stability rank x options flow")
    parser.add_argument("--regime", choices=REGIMES,
                        help="Current regime (interactive prompt if omitted)")
    parser.add_argument("--top-n", type=int, default=None,
                        dest="top_n", help="Restrict to the stability "
                        "top-N (default: the complete list)")
    args = parser.parse_args()
    regime = args.regime if args.regime else prompt_regime()

    by_date = load_history()
    if not by_date:
        print("ERROR: no options session files found.")
        sys.exit(1)
    dates = sorted(by_date)
    target = dates[-1]

    # Never recommend off a partial session: if the newest file belongs to
    # a U.S. session still in progress, use the previous one.
    now_et = datetime.now(pytz.timezone("US/Eastern"))
    if (target == now_et.strftime("%Y-%m-%d")
            and now_et.time() < dt_time(16, 15) and len(dates) > 1):
        print(f"NOTE: session {target} still in progress; using "
              f"{dates[-2]} instead.")
        target = dates[-2]
    hist = [d for d in dates if d < target]

    stab, stab_name = load_stability(regime, args.top_n)
    strong_rank_max = STRONG_RANK_MAX

    print(f"{'=' * 70}")
    print(f"LEADING FACTOR RECOMMENDER v2 - {regime}")
    print(f"Options session: {target} ({len(dates)} sessions loaded)")
    print(f"Stability universe: top {len(stab)} of {stab_name}")
    print(f"{'=' * 70}\n")

    score = build_panel(by_date, "Score")
    vol = build_panel(by_date, "TotalVolume")
    call_oi = build_panel(by_date, "CallOI")
    put_oi = build_panel(by_date, "PutOI")
    status = build_panel(by_date, "Status")

    # FAST: today's event state per ticker
    trail = vol[hist[-TRAIL_WINDOW:]].mean(axis=1)
    rvol = vol[target] / trail
    is_event = ((status[target] == "OK")
                & (score[target].abs() >= SCORE_EXTREME)
                & (vol[target] >= MIN_EVENT_VOLUME)
                & (trail > 0) & (rvol >= RVOL_SPIKE))
    event = pd.Series("", index=score.index)
    event[is_event & (score[target] > 0)] = "BULL"
    event[is_event & (score[target] < 0)] = "BEAR"

    ema3 = score[hist + [target]].T.ewm(span=3).mean().iloc[-1]

    # MEDIUM: OI trend over the last OI_SESSIONS sessions
    oi_dates = [d for d in dates if d <= target and call_oi[d].notna().any()]
    oi_trend = pd.Series("", index=score.index)
    oi_rel = pd.Series(np.nan, index=score.index)
    oi_span = 0
    if len(oi_dates) >= 2:
        oi_span = min(OI_SESSIONS, len(oi_dates) - 1)
        prev = oi_dates[oi_dates.index(target) - oi_span]
        base = call_oi[prev] + put_oi[prev]
        net = ((call_oi[target] - call_oi[prev])
               - (put_oi[target] - put_oi[prev]))
        oi_rel = net / (base + 1)
        threshold = OI_TREND_5D if oi_span >= OI_SESSIONS else OI_TREND_1D
        oi_trend[(base >= OI_MIN_BASE) & (oi_rel >= threshold)] = "CALL BUILD"
        oi_trend[(base >= OI_MIN_BASE) & (oi_rel <= -threshold)] = "PUT BUILD"
        print(f"OI trend window: {oi_span} session(s), "
              f"{prev} -> {target}, threshold {threshold}\n")
    else:
        print("OI trend: insufficient OI history, skipped\n")

    # Assemble per stability-universe ticker
    out = stab[[c for c in ("ticker", "CompanyName", "Sector",
                            "stability_adjusted_score", "StabilityRank")
                if c in stab.columns]].copy()
    out = out.rename(columns={"ticker": "Ticker"})
    out = out.set_index("Ticker")
    out["FlowScore"] = score[target]
    out["ScoreEMA3"] = ema3
    out["Rvol"] = rvol
    out["Event"] = event
    out["OITrend"] = oi_trend
    out["OIRel"] = oi_rel
    out["HasFlow"] = (status[target] == "OK").reindex(out.index, fill_value=False)

    out["Recommendation"] = out.apply(
        lambda r: recommend(r, strong_rank_max), axis=1)
    out["Tier"] = out["Recommendation"].str.split(" - ").str[0]
    out["Tier"] = pd.Categorical(out["Tier"], categories=TIER_ORDER,
                                 ordered=True)
    out = out.sort_values(["Tier", "StabilityRank"])

    def print_row(tk, r):
        extras = []
        if r["Event"]:
            extras.append(f"{r['Event']} event rvol {r['Rvol']:.1f}x")
        if r["OITrend"]:
            extras.append(f"{r['OITrend']} {r['OIRel']:+.2f}")
        detail = f" [{', '.join(extras)}]" if extras else ""
        reason = r["Recommendation"].split(" - ", 1)[1]
        print(f"   #{r['StabilityRank']:>4} {tk:<6} {reason}{detail}")

    # Console report: action tiers in full, big neutral tiers condensed
    for tier in TIER_ORDER:
        sub = out[out["Tier"] == tier]
        if sub.empty:
            continue
        if tier == "AMBIGUOUS":
            print(f"AMBIGUOUS ({len(sub)}): not enough options data to "
                  f"judge - full list in the CSV\n")
            continue
        flagged = sub[(sub["Event"] != "") | (sub["OITrend"] != "")]
        if tier in ("HOLD", "BUY") and len(sub) > 40:
            shown = flagged if not flagged.empty else sub.head(10)
            print(f"{tier} ({len(sub)} total, showing {len(shown)} with "
                  f"active flags):")
            for tk, r in shown.iterrows():
                print_row(tk, r)
            print(f"   ... plus {len(sub) - len(shown)} more in the CSV\n")
        else:
            print(f"{tier} ({len(sub)}):")
            for tk, r in sub.iterrows():
                print_row(tk, r)
            print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = (OUTPUT_DIR / f"leading_factor_recommendations_"
                             f"{regime}_{target.replace('-', '')}.csv")
    out.drop(columns=["Tier"]).to_csv(out_file)
    print(f"Saved to: {out_file}")
    print(f"Sources: {stab_name} + options sessions through {target}")


if __name__ == "__main__":
    main()
