"""
Daily Options Flow Scorer

Scores every ticker in the buyable list by near-the-money options flow
imbalance and writes a daily ranked list, mirroring the Ranked_Lists
pipeline convention.

Score (per ticker, near-the-money contracts only):

    Score = (call_volume - put_volume) / (call_volume + put_volume)

Bounded in [-1, +1]: +1 = all-call flow, -1 = all-put flow, 0 = balanced.
Also computed (not used for ranking yet):
    ImbalanceNotional - same equation on dollar volume (vol * vwap * 100)
    ImbalanceOI       - same equation on open interest

Data source: Massive.com (formerly Polygon.io)
    GET /v2/aggs/grouped/locale/us/market/stocks/{date}  (1 call: closes
        for the whole U.S. market, used for the strike band; the chain
        snapshot's underlying_asset.price is not populated on all plans)
    GET /v3/snapshot/options/{ticker}  (1 strike-filtered call per ticker)
Snapshot `day` data reflects the MOST RECENT trading session (it
persists over weekends/holidays), so the DataDate column records which
session the volumes belong to; do not treat two runs with the same
DataDate as independent observations. Run after the U.S. market close
for final volumes. Rolling averages / z-scores are left to a separate
analysis script; this one only produces the daily raw list.

Output: output/Options_Ranked_Lists/options_YYYYMMDD.csv
"""

import json
import sys
import time
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import pytz
import requests

# ===== CONFIGURATION =====

PROJECT_ROOT = Path(__file__).parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
TICKER_LIST_FILE = CONFIG_DIR / "Buyable_stocks_0901.txt"
API_CONFIG_FILE = CONFIG_DIR / "massive_config.json"
OUTPUT_DIR = PROJECT_ROOT / "output" / "Options_Ranked_Lists"

BASE_URL = "https://api.massive.com"

NTM_BAND = (0.90, 1.10)   # strike band around underlying price
MAX_DTE = 60              # only expirations within N days
MIN_TOTAL_VOLUME = 100    # below this NTM volume: LOW_VOLUME, no score

REQUEST_DELAY_SEC = 0.1   # 12.5 for the free tier (5 calls/min)
MAX_TICKERS = None        # cap for quick tests, e.g. 20; None = full list

# ===== END CONFIGURATION =====


def load_api_key():
    if not API_CONFIG_FILE.exists():
        print(f"ERROR: Config file not found: {API_CONFIG_FILE}")
        sys.exit(1)
    with open(API_CONFIG_FILE) as f:
        key = json.load(f).get("api_key", "")
    if not key or key in ("YOUR_API_KEY_HERE", "aaa"):
        print(f"ERROR: No valid api_key in {API_CONFIG_FILE}")
        sys.exit(1)
    return key


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
    # de-duplicate, preserve order
    return list(dict.fromkeys(tickers))


class MassiveClient:
    """Minimal REST client with rate limiting and 429 retry."""

    def __init__(self, api_key):
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {api_key}"
        self.n_calls = 0
        self.n_403 = 0

    def get(self, url, params=None, max_retries=5):
        for attempt in range(max_retries):
            if REQUEST_DELAY_SEC > 0:
                time.sleep(REQUEST_DELAY_SEC)
            resp = self.session.get(url, params=params, timeout=30)
            self.n_calls += 1

            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 401:
                print("ERROR: 401 Unauthorized - check your API key.")
                sys.exit(1)
            if resp.status_code == 403:
                self.n_403 += 1
                return None
            if resp.status_code == 404:
                return None
            if resp.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"   Rate limited (429), waiting {wait}s...")
                time.sleep(wait)
                continue
            return None
        return None


def get_session_date():
    """The U.S. session the snapshot's `day` data belongs to, from the
    NYSE calendar (options exchanges follow it) in U.S. Eastern time.

    Grouped-daily availability must NOT decide this: right after the
    close, the just-finished session's EOD file may not be published
    yet, and walking back would silently mislabel the data.

    Returns (session_date 'YYYY-MM-DD', in_progress bool).
    """
    now_et = datetime.now(pytz.timezone("US/Eastern"))
    nyse = mcal.get_calendar("NYSE")
    valid = [d.date() for d in nyse.valid_days(
        start_date=now_et.date() - timedelta(days=10),
        end_date=now_et.date())]
    if not valid:
        print("ERROR: No NYSE trading days found in the last 10 days.")
        sys.exit(1)
    in_progress = False
    if valid[-1] == now_et.date():
        if now_et.time() < dt_time(9, 30):
            valid = valid[:-1]      # today's session has not started yet
        elif now_et.time() < dt_time(16, 15):
            in_progress = True      # intraday run: volumes are partial
    if not valid:
        print("ERROR: No completed NYSE session found.")
        sys.exit(1)
    return valid[-1].strftime("%Y-%m-%d"), in_progress


def get_price_map(client, session_date):
    """Closing prices for the whole U.S. market from grouped daily bars.

    Tries the session itself first; if its EOD file is not published
    yet, walks back to earlier days for the strike bands only (the
    session date labeling is NOT affected).
    Returns ({ticker: close}, closes_date_str).
    """
    start = datetime.strptime(session_date, "%Y-%m-%d")
    for days_back in range(8):
        date = (start - timedelta(days=days_back)).strftime("%Y-%m-%d")
        url = f"{BASE_URL}/v2/aggs/grouped/locale/us/market/stocks/{date}"
        data = client.get(url, params={"adjusted": "true"})
        if data and data.get("resultsCount"):
            prices = {bar["T"]: bar["c"] for bar in data["results"]
                      if bar.get("c")}
            return prices, date
    print("ERROR: No grouped daily data found in the last 8 days.")
    sys.exit(1)


def get_chain_ntm(client, ticker, price):
    """Fetch the near-the-money chain for one ticker (1+ calls)."""
    base = f"{BASE_URL}/v3/snapshot/options/{ticker}"
    today = datetime.now().strftime("%Y-%m-%d")
    exp_max = (datetime.now() + timedelta(days=MAX_DTE)).strftime("%Y-%m-%d")
    params = {
        "strike_price.gte": round(NTM_BAND[0] * price, 2),
        "strike_price.lte": round(NTM_BAND[1] * price, 2),
        "expiration_date.gte": today,
        "expiration_date.lte": exp_max,
        "limit": 250,
    }

    contracts = []
    url = base
    while url:
        data = client.get(url, params=params)
        if data is None:
            break
        contracts.extend(data.get("results", []))
        url = data.get("next_url")  # already carries the cursor
        params = None
    return contracts


def safe_imbalance(call_val, put_val):
    total = call_val + put_val
    return (call_val - put_val) / total if total > 0 else np.nan


def score_ticker(client, ticker, price, session_start_ns):
    """Compute NTM flow metrics for one ticker. Returns a result dict."""
    row = {
        "Ticker": ticker, "DataDate": "", "Score": np.nan,
        "ImbalanceNotional": np.nan, "ImbalanceOI": np.nan,
        "CallVolume": 0, "PutVolume": 0, "TotalVolume": 0,
        "CallOI": 0, "PutOI": 0,
        "UnderlyingPrice": np.nan, "Contracts": 0, "Status": "OK",
    }

    if price is None or price <= 0:
        row["Status"] = "NO_PRICE"
        return row
    row["UnderlyingPrice"] = price

    contracts = get_chain_ntm(client, ticker, price)

    call_vol = put_vol = 0
    call_notional = put_notional = 0.0
    call_oi = put_oi = 0
    n = 0
    for c in contracts:
        details = c.get("details") or {}
        ctype = details.get("contract_type")
        if ctype not in ("call", "put"):
            continue
        day = c.get("day") or {}
        vol = day.get("volume") or 0
        # Snapshot `day` persists the contract's LAST-TRADED session, so
        # an illiquid contract untraded this session still shows old
        # volume. Count volume only when last_updated falls inside the
        # session being scored (missing timestamp: keep, benefit of the
        # doubt). OI is current positioning and stays valid regardless.
        lu = day.get("last_updated")
        if vol and lu and lu < session_start_ns:
            vol = 0
        price_ref = day.get("vwap") or day.get("close") or 0
        oi = c.get("open_interest") or 0
        n += 1
        if ctype == "call":
            call_vol += vol
            call_notional += vol * price_ref * 100
            call_oi += oi
        else:
            put_vol += vol
            put_notional += vol * price_ref * 100
            put_oi += oi

    total_vol = call_vol + put_vol
    row.update({
        "CallVolume": int(call_vol), "PutVolume": int(put_vol),
        "TotalVolume": int(total_vol),
        "CallOI": int(call_oi), "PutOI": int(put_oi),
        "Contracts": n,
    })

    if n == 0:
        row["Status"] = "NO_OPTIONS"
        return row
    if total_vol < MIN_TOTAL_VOLUME:
        row["Status"] = "LOW_VOLUME"
        return row

    row["Score"] = safe_imbalance(call_vol, put_vol)
    row["ImbalanceNotional"] = safe_imbalance(call_notional, put_notional)
    row["ImbalanceOI"] = safe_imbalance(call_oi, put_oi)
    return row


def main():
    start_time = time.time()
    run_date = datetime.now().strftime("%Y%m%d")

    print(f"{'=' * 70}")
    print(f"DAILY OPTIONS FLOW SCORER - {run_date}")
    print(f"NTM band {NTM_BAND[0]:.0%}-{NTM_BAND[1]:.0%}, "
          f"max {MAX_DTE} DTE, min volume {MIN_TOTAL_VOLUME}")
    print(f"{'=' * 70}\n")

    api_key = load_api_key()
    client = MassiveClient(api_key)
    tickers = load_tickers()
    if MAX_TICKERS is not None:
        tickers = tickers[:MAX_TICKERS]
        print(f"NOTE: capped to first {MAX_TICKERS} tickers (MAX_TICKERS)\n")
    print(f"Tickers to score: {len(tickers)}")
    print(f"Estimated API calls: ~{len(tickers) + 2} "
          f"(~{len(tickers) * max(REQUEST_DELAY_SEC, 0.05) / 60:.0f} min "
          f"at current REQUEST_DELAY_SEC)\n")

    session_date, in_progress = get_session_date()
    session_start_ns = int(pd.Timestamp(
        session_date, tz="US/Eastern").value)
    print(f"Scoring U.S. session: {session_date}")
    if in_progress:
        print("   WARNING: session is still in progress - volumes are "
              "partial. Prefer running after the U.S. close.")

    print("Fetching market-wide closing prices (grouped daily)...")
    price_map, closes_date = get_price_map(client, session_date)
    print(f"   {len(price_map)} closes as of {closes_date}")
    if closes_date != session_date:
        print(f"   NOTE: session {session_date} closes not published yet; "
              f"using {closes_date} closes for strike bands only.")
    print()

    rows = []
    for i, ticker in enumerate(tickers, 1):
        try:
            row = score_ticker(client, ticker, price_map.get(ticker),
                               session_start_ns)
            row["DataDate"] = session_date
            rows.append(row)
        except Exception as e:
            print(f"   ERROR scoring {ticker}: {e}")
            rows.append({"Ticker": ticker, "Status": "ERROR"})

        if i % 25 == 0 or i == len(tickers):
            elapsed = time.time() - start_time
            n_ok = sum(1 for r in rows if r.get("Status") == "OK")
            print(f"   {i}/{len(tickers)} scored "
                  f"({n_ok} OK, {client.n_calls} calls, {elapsed:.0f}s)")

        # Abort early if the plan clearly lacks snapshot access
        if i == 5 and client.n_403 >= 5:
            print("\nERROR: First 5 tickers all returned 403 Forbidden.")
            print("Your Massive plan may not include option chain snapshots.")
            sys.exit(1)

    df = pd.DataFrame(rows)
    df = df.sort_values("Score", ascending=False, na_position="last")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Name by SESSION date (DataDate), not run date, so weekend runs
    # overwrite the same session file instead of duplicating it and the
    # filename always matches the data inside (backfill convention).
    output_file = OUTPUT_DIR / f"options_{session_date.replace('-', '')}.csv"
    df.to_csv(output_file, index=False)

    scored = df[df["Status"] == "OK"]
    print(f"\n{'-' * 70}")
    print(f"Status counts: {df['Status'].value_counts().to_dict()}")

    n_candidates = (df["Status"].isin(["OK", "LOW_VOLUME"])).sum()
    if n_candidates > 0 and (df["TotalVolume"].fillna(0) > 0).sum() == 0:
        print("\nWARNING: Every ticker returned zero day volume.")
        print("The U.S. market is likely closed today (weekend/holiday);")
        print("snapshot day data only exists during/after a trading session.")
        print("Re-run after the next U.S. market close.")
    if not scored.empty:
        with pd.option_context("display.float_format", "{:.3f}".format):
            print(f"\nTop 20 by NTM volume imbalance:")
            print(scored.head(20)[
                ["Ticker", "Score", "ImbalanceNotional", "ImbalanceOI",
                 "TotalVolume", "UnderlyingPrice"]
            ].to_string(index=False))
            print(f"\nBottom 10 (most put-skewed):")
            print(scored.tail(10)[
                ["Ticker", "Score", "ImbalanceNotional", "ImbalanceOI",
                 "TotalVolume", "UnderlyingPrice"]
            ].to_string(index=False))

    print(f"\nTotal API calls: {client.n_calls}, "
          f"elapsed {time.time() - start_time:.0f}s")
    print(f"Ranked list saved to: {output_file}")


if __name__ == "__main__":
    main()
