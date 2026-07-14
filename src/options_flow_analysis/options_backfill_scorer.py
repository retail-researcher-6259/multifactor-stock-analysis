"""
Historical Options Flow Backfill Scorer

Produces the same daily ranked lists as options_daily_scorer.py, but for
PAST sessions, using Massive.com flat files (S3) instead of chain
snapshots (which are current-day only). One output CSV per U.S. trading
session in the configured range, same schema and folder as the daily
scorer, so the future analysis script can consume one uniform history.

Data sources:
    S3 flat files: us_options_opra/day_aggs_v1/YYYY/MM/YYYY-MM-DD.csv.gz
        (endpoint https://files.massive.com, bucket "flatfiles";
        one file per trading day covering EVERY option contract, columns:
        ticker, volume, open, close, high, low, window_start, transactions)
    REST grouped daily stocks (1 call per session) for underlying closes.

KNOWN LIMITATION: flat files contain no open interest and no vwap, so
for backfilled days ImbalanceOI/CallOI/PutOI are empty and notional uses
the contract's daily close as the price reference. Volume-based Score is
fully equivalent to the daily scorer.

Credentials in config/massive_config.json:
    "api_key"              - REST key (grouped daily)
    "s3_access_key_id"     - from the Massive dashboard (S3 / Flat Files
    "s3_secret_access_key"   section; NOT the same as the REST key)

Date convention: PERIOD_START/PERIOD_END are U.S. SESSION dates. A 06:00
Taipei run on calendar day D captures the U.S. session dated D-1, so a
Taipei-time window "June 1 to July 4, 6 AM" corresponds to sessions
2026-05-29 through 2026-07-02. Trading days are discovered by listing
the S3 bucket (files only exist for sessions), so holidays are skipped
automatically.

Output: output/Options_Ranked_Lists/options_YYYYMMDD.csv (session date;
existing files are skipped unless OVERWRITE = True).
"""

import gzip
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

try:
    import boto3
    from botocore.config import Config as BotoConfig
except ImportError:
    print("ERROR: boto3 is required for flat file access: pip install boto3")
    sys.exit(1)

# ===== CONFIGURATION =====

PROJECT_ROOT = Path(__file__).parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
TICKER_LIST_FILE = CONFIG_DIR / "Buyable_stocks_0901.txt"
API_CONFIG_FILE = CONFIG_DIR / "massive_config.json"
OUTPUT_DIR = PROJECT_ROOT / "output" / "Options_Ranked_Lists"
CACHE_DIR = PROJECT_ROOT / "output" / "Options_Ranked_Lists" / "Flat_Files_Cache"

# U.S. SESSION dates, inclusive (see date convention in the docstring)
PERIOD_START = "2026-06-01"
PERIOD_END = "2026-07-02"

BASE_URL = "https://api.massive.com"
S3_ENDPOINT = "https://files.massive.com"
S3_BUCKET = "flatfiles"
S3_PREFIX = "us_options_opra/day_aggs_v1"

# Scoring parameters (keep identical to options_daily_scorer.py)
NTM_BAND = (0.90, 1.10)
MAX_DTE = 60
MIN_TOTAL_VOLUME = 100

OVERWRITE = False          # True: regenerate existing output CSVs
KEEP_CACHE = True          # False: delete downloaded .csv.gz after use
REQUEST_DELAY_SEC = 0.1

# ===== END CONFIGURATION =====


def load_config():
    if not API_CONFIG_FILE.exists():
        print(f"ERROR: Config file not found: {API_CONFIG_FILE}")
        sys.exit(1)
    with open(API_CONFIG_FILE) as f:
        cfg = json.load(f)
    api_key = cfg.get("api_key", "")
    s3_id = cfg.get("s3_access_key_id", "")
    s3_secret = cfg.get("s3_secret_access_key", "")
    if not api_key or api_key in ("YOUR_API_KEY_HERE", "aaa"):
        print(f"ERROR: No valid api_key in {API_CONFIG_FILE}")
        sys.exit(1)
    if (not s3_id or "YOUR_S3" in s3_id
            or not s3_secret or "YOUR_S3" in s3_secret):
        print(f"ERROR: Missing S3 credentials in {API_CONFIG_FILE}")
        print("Get them from the Massive dashboard (S3 / Flat Files keys)")
        print("and fill in s3_access_key_id / s3_secret_access_key.")
        sys.exit(1)
    return api_key, s3_id, s3_secret


def load_tickers():
    if not TICKER_LIST_FILE.exists():
        print(f"ERROR: Ticker list not found: {TICKER_LIST_FILE}")
        sys.exit(1)
    tickers = []
    with open(TICKER_LIST_FILE) as f:
        for line in f:
            t = line.strip().upper()
            if t and not t.startswith("#"):
                tickers.append(t)
    return set(tickers)


def get_grouped_closes(session, date_str, max_retries=6):
    """{ticker: close} for one session via REST grouped daily.

    The stocks endpoints can be on a stricter rate limit than the
    options ones (5/min free-tier style), so 429 is retried with
    a generous backoff instead of failing the session.
    """
    url = f"{BASE_URL}/v2/aggs/grouped/locale/us/market/stocks/{date_str}"
    for attempt in range(max_retries):
        time.sleep(REQUEST_DELAY_SEC)
        resp = session.get(url, params={"adjusted": "true"}, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            return {bar["T"]: bar["c"] for bar in data.get("results", [])
                    if bar.get("c")}
        if resp.status_code == 429:
            wait = 15 * (attempt + 1)
            print(f"   Rate limited (429) on grouped daily, waiting {wait}s "
                  f"(attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait)
            continue
        print(f"   WARNING: grouped daily HTTP {resp.status_code} "
              f"for {date_str}")
        return {}
    print(f"   WARNING: grouped daily still rate limited for {date_str}")
    return {}


def list_session_dates(s3, start, end):
    """Trading days in [start, end] discovered from S3 file listing."""
    months = sorted({d.strftime("%Y/%m") for d in
                     pd.date_range(start, end, freq="D")})
    dates = []
    for month in months:
        token = None
        while True:
            kwargs = {"Bucket": S3_BUCKET, "Prefix": f"{S3_PREFIX}/{month}/"}
            if token:
                kwargs["ContinuationToken"] = token
            resp = s3.list_objects_v2(**kwargs)
            for obj in resp.get("Contents", []):
                name = obj["Key"].rsplit("/", 1)[-1]       # 2026-06-01.csv.gz
                date_str = name.split(".")[0]
                if start <= date_str <= end:
                    dates.append((date_str, obj["Key"]))
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
    return sorted(dates)


def download_day_file(s3, key, date_str):
    """Download one day-aggregates file to the cache; returns local path."""
    local = CACHE_DIR / f"{date_str}.csv.gz"
    if local.exists() and local.stat().st_size > 0:
        return local
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    s3.download_file(S3_BUCKET, key, str(local))
    return local


def parse_occ_ticker(t):
    """Split an OPRA symbol 'O:ROOTYYMMDDC00123000' into components.

    Returns (root, expiry 'YYYY-MM-DD', 'call'/'put', strike) or None for
    nonstandard symbols (adjusted contracts keep digits in the root and
    simply won't match the buyable list).
    """
    if not t.startswith("O:") or len(t) < 18:
        return None
    tail = t[-15:]
    root = t[2:-15]
    exp_raw, cp, strike_raw = tail[:6], tail[6], tail[7:]
    if cp not in ("C", "P") or not exp_raw.isdigit() or not strike_raw.isdigit():
        return None
    expiry = f"20{exp_raw[:2]}-{exp_raw[2:4]}-{exp_raw[4:6]}"
    return root, expiry, ("call" if cp == "C" else "put"), int(strike_raw) / 1000


def score_session(day_df, closes, wanted, date_str):
    """Aggregate one session's flat file into per-ticker NTM flow rows."""
    parsed = day_df["ticker"].map(parse_occ_ticker)
    mask = parsed.notna()
    day_df = day_df[mask].copy()
    comp = pd.DataFrame(parsed[mask].tolist(),
                        columns=["root", "expiry", "cp", "strike"],
                        index=day_df.index)
    day_df = pd.concat([day_df, comp], axis=1)

    day_df = day_df[day_df["root"].isin(wanted)]

    exp_max = (pd.Timestamp(date_str)
               + pd.Timedelta(days=MAX_DTE)).strftime("%Y-%m-%d")
    day_df = day_df[(day_df["expiry"] >= date_str)
                    & (day_df["expiry"] <= exp_max)]

    day_df["close_u"] = day_df["root"].map(closes)
    day_df = day_df[day_df["close_u"].notna() & (day_df["close_u"] > 0)]
    moneyness = day_df["strike"] / day_df["close_u"]
    day_df = day_df[(moneyness >= NTM_BAND[0]) & (moneyness <= NTM_BAND[1])]

    day_df["notional"] = day_df["volume"] * day_df["close"] * 100

    rows = []
    for root in sorted(wanted):
        close_u = closes.get(root)
        row = {
            "Ticker": root, "DataDate": date_str, "Score": np.nan,
            "ImbalanceNotional": np.nan, "ImbalanceOI": np.nan,
            "CallVolume": 0, "PutVolume": 0, "TotalVolume": 0,
            "CallOI": np.nan, "PutOI": np.nan,
            "UnderlyingPrice": close_u if close_u else np.nan,
            "Contracts": 0, "Status": "OK",
        }
        if not close_u or close_u <= 0:
            row["Status"] = "NO_PRICE"
            rows.append(row)
            continue

        sub = day_df[day_df["root"] == root]
        if sub.empty:
            row["Status"] = "NO_OPTIONS"
            rows.append(row)
            continue

        call = sub[sub["cp"] == "call"]
        put = sub[sub["cp"] == "put"]
        c_vol, p_vol = call["volume"].sum(), put["volume"].sum()
        c_ntl, p_ntl = call["notional"].sum(), put["notional"].sum()
        total = c_vol + p_vol
        row.update({
            "CallVolume": int(c_vol), "PutVolume": int(p_vol),
            "TotalVolume": int(total), "Contracts": len(sub),
        })
        if total < MIN_TOTAL_VOLUME:
            row["Status"] = "LOW_VOLUME"
        else:
            row["Score"] = (c_vol - p_vol) / total
            row["ImbalanceNotional"] = ((c_ntl - p_ntl) / (c_ntl + p_ntl)
                                        if (c_ntl + p_ntl) > 0 else np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values("Score", ascending=False, na_position="last")


def main():
    global PERIOD_START, PERIOD_END
    import argparse
    parser = argparse.ArgumentParser(
        description="Backfill daily options ranked lists from flat files")
    parser.add_argument("--start", default=None,
                        help="First session date YYYY-MM-DD "
                             f"(default {PERIOD_START})")
    parser.add_argument("--end", default=None,
                        help="Last session date YYYY-MM-DD "
                             f"(default {PERIOD_END})")
    args = parser.parse_args()
    if args.start:
        PERIOD_START = args.start
    if args.end:
        PERIOD_END = args.end

    start_time = time.time()
    print(f"{'=' * 70}")
    print(f"OPTIONS FLOW BACKFILL - sessions {PERIOD_START} to {PERIOD_END}")
    print(f"NTM band {NTM_BAND[0]:.0%}-{NTM_BAND[1]:.0%}, max {MAX_DTE} DTE, "
          f"min volume {MIN_TOTAL_VOLUME}")
    print(f"{'=' * 70}\n")

    api_key, s3_id, s3_secret = load_config()
    wanted = load_tickers()
    print(f"Tickers: {len(wanted)}")

    rest = requests.Session()
    rest.headers["Authorization"] = f"Bearer {api_key}"
    s3 = boto3.Session(
        aws_access_key_id=s3_id, aws_secret_access_key=s3_secret,
    ).client("s3", endpoint_url=S3_ENDPOINT,
             config=BotoConfig(signature_version="s3v4"))

    print("Listing available session files on S3...")
    try:
        sessions = list_session_dates(s3, PERIOD_START, PERIOD_END)
    except Exception as e:
        print(f"ERROR: S3 listing failed: {type(e).__name__}: {e}")
        print("Check s3_access_key_id / s3_secret_access_key in the config")
        print("(dashboard S3 keys, not the REST api_key).")
        sys.exit(1)
    if not sessions:
        print("ERROR: No flat files found in the period.")
        sys.exit(1)
    print(f"   {len(sessions)} trading sessions found\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n_done = n_skipped = 0
    for date_str, key in sessions:
        out_file = OUTPUT_DIR / f"options_{date_str.replace('-', '')}.csv"
        if out_file.exists() and not OVERWRITE:
            print(f"{date_str}: output exists, skipping")
            n_skipped += 1
            continue

        print(f"{date_str}: downloading flat file...")
        local = download_day_file(s3, key, date_str)
        size_mb = local.stat().st_size / 1e6
        day_df = pd.read_csv(local, compression="gzip",
                             usecols=["ticker", "volume", "close"])
        closes = get_grouped_closes(rest, date_str)
        if not closes:
            print(f"   WARNING: no underlying closes for {date_str}, skipping")
            continue

        df = score_session(day_df, closes, wanted, date_str)
        df.to_csv(out_file, index=False)
        n_ok = (df["Status"] == "OK").sum()
        print(f"   {len(day_df)} contract rows ({size_mb:.0f} MB) -> "
              f"{n_ok} scored tickers -> {out_file.name}")
        n_done += 1

        if not KEEP_CACHE:
            local.unlink(missing_ok=True)

    print(f"\n{'-' * 70}")
    print(f"Backfill complete: {n_done} sessions written, "
          f"{n_skipped} skipped (existing), "
          f"elapsed {(time.time() - start_time) / 60:.1f} min")
    print(f"Output folder: {OUTPUT_DIR}")
    if KEEP_CACHE:
        print(f"Flat file cache kept at: {CACHE_DIR}")


if __name__ == "__main__":
    main()
