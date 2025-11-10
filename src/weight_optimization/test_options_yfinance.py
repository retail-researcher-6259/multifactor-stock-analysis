"""
Test script for Options Data Fetching - yfinance version
Compares yahooquery vs yfinance for options data retrieval
"""

import yfinance as yf
from yahooquery import Ticker as YQTicker
from curl_cffi.requests import Session as CurlSession
import yahooquery.utils as yq_utils
import types
from pathlib import Path
from datetime import datetime
import json
import pandas as pd

# Set up curl_cffi session for yahooquery
class CurlCffiSessionWrapper(CurlSession):
    """Wrapper to make curl_cffi compatible with yahooquery's expectations"""
    def mount(self, *_):
        pass

_global_curl_session = CurlCffiSessionWrapper(
    impersonate="chrome110",
    headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "en-US,en;q=0.9",
    },
    timeout=30
)

curl_module = types.ModuleType("curl_requests")
curl_module.Session = CurlCffiSessionWrapper
yq_utils.requests = curl_module
yq_utils._COOKIE = {"B": "dummy"}
yq_utils._CRUMB = "dummy"
yq_utils.get_crumb = lambda *_: "dummy"

# Set up cache directory
CACHE_DIR = Path("cache")
options_cache_dir = CACHE_DIR / "options_oi"
options_cache_dir.mkdir(parents=True, exist_ok=True)

def test_yahooquery(ticker):
    """Test yahooquery options data"""
    print(f"\n{'='*60}")
    print(f"YAHOOQUERY Test for {ticker}")
    print(f"{'='*60}")

    try:
        t = YQTicker(ticker, session=_global_curl_session)
        options_chain = t.option_chain

        print(f"Return type: {type(options_chain)}")
        print(f"Return value: {options_chain}")

        if isinstance(options_chain, pd.DataFrame):
            print(f"\n📊 DataFrame shape: {options_chain.shape}")
            print(f"📊 Columns: {list(options_chain.columns)}")
            print(f"📊 Index: {options_chain.index}")

            if len(options_chain) > 0:
                print(f"\n✅ Has {len(options_chain)} rows")
                print(f"\nFirst few rows:")
                print(options_chain.head())
                return True, len(options_chain)
            else:
                print(f"\n❌ DataFrame is empty")
                return False, 0

        elif isinstance(options_chain, dict):
            total_contracts = 0
            for exp_date, contracts in options_chain.items():
                if isinstance(contracts, pd.DataFrame):
                    total_contracts += len(contracts)

            print(f"\n✅ Dict with {len(options_chain)} expirations")
            print(f"✅ Total contracts: {total_contracts}")
            return True, total_contracts
        else:
            print(f"\n❌ Unexpected type: {type(options_chain)}")
            return False, 0

    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False, 0

def test_yfinance(ticker):
    """Test yfinance options data"""
    print(f"\n{'='*60}")
    print(f"YFINANCE Test for {ticker}")
    print(f"{'='*60}")

    try:
        stock = yf.Ticker(ticker)

        # Get available expiration dates
        expirations = stock.options
        print(f"📅 Available expirations: {len(expirations)}")

        if len(expirations) == 0:
            print(f"❌ No expiration dates available")
            return False, 0

        print(f"📅 First few dates: {expirations[:5]}")

        # Get options chain for first expiration
        first_exp = expirations[0]
        opt_chain = stock.option_chain(first_exp)

        calls = opt_chain.calls
        puts = opt_chain.puts

        print(f"\n📊 Calls shape: {calls.shape}")
        print(f"📊 Puts shape: {puts.shape}")
        print(f"📊 Calls columns: {list(calls.columns)}")

        # Show sample data
        if len(calls) > 0:
            print(f"\n✅ Sample CALL contract:")
            sample = calls.iloc[0]
            print(f"   Strike: {sample.get('strike', 'N/A')}")
            print(f"   Last Price: {sample.get('lastPrice', 'N/A')}")
            print(f"   Open Interest: {sample.get('openInterest', 'N/A')}")
            print(f"   Volume: {sample.get('volume', 'N/A')}")
            print(f"   Implied Volatility: {sample.get('impliedVolatility', 'N/A')}")

        total_contracts = len(calls) + len(puts)
        print(f"\n✅ Total contracts (first exp): {total_contracts}")

        # Test caching with yfinance data
        print(f"\n3. Testing cache with yfinance data...")
        date = datetime.now().strftime('%Y-%m-%d')
        cache_file = options_cache_dir / f"{ticker}_yf_{date}.json"

        oi_data = {}

        # Process calls
        for _, contract in calls.iterrows():
            strike = contract.get('strike')
            exp_date = first_exp
            key = f"{strike}_C_{exp_date}"

            oi_data[key] = {
                'strike': float(strike),
                'type': 'CALL',
                'expiration': str(exp_date),
                'openInterest': int(contract.get('openInterest', 0)),
                'volume': int(contract.get('volume', 0)),
                'lastPrice': float(contract.get('lastPrice', 0)),
                'impliedVolatility': float(contract.get('impliedVolatility', 0))
            }

        # Process puts
        for _, contract in puts.iterrows():
            strike = contract.get('strike')
            exp_date = first_exp
            key = f"{strike}_P_{exp_date}"

            oi_data[key] = {
                'strike': float(strike),
                'type': 'PUT',
                'expiration': str(exp_date),
                'openInterest': int(contract.get('openInterest', 0)),
                'volume': int(contract.get('volume', 0)),
                'lastPrice': float(contract.get('lastPrice', 0)),
                'impliedVolatility': float(contract.get('impliedVolatility', 0))
            }

        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump({
                'ticker': ticker,
                'date': date,
                'data': oi_data,
                'source': 'yfinance'
            }, f, indent=2)

        print(f"   ✅ Cached {len(oi_data)} contracts to: {cache_file}")
        print(f"   ✅ Cache file size: {cache_file.stat().st_size / 1024:.2f} KB")

        return True, total_contracts

    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False, 0

def compare_providers(tickers):
    """Compare yahooquery vs yfinance for multiple tickers"""
    print(f"\n{'#'*60}")
    print(f"# Comparing Options Data Providers")
    print(f"{'#'*60}")

    results = {
        'yahooquery': {},
        'yfinance': {}
    }

    for ticker in tickers:
        print(f"\n\n{'='*60}")
        print(f"Testing {ticker}")
        print(f"{'='*60}")

        # Test yahooquery
        yq_success, yq_contracts = test_yahooquery(ticker)
        results['yahooquery'][ticker] = (yq_success, yq_contracts)

        # Test yfinance
        yf_success, yf_contracts = test_yfinance(ticker)
        results['yfinance'][ticker] = (yf_success, yf_contracts)

    # Summary
    print(f"\n\n{'='*60}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*60}")

    print(f"\nYahooquery Results:")
    print(f"{'Ticker':<10} {'Success':<10} {'Contracts':<10}")
    print(f"{'-'*30}")
    for ticker, (success, contracts) in results['yahooquery'].items():
        status = "✅" if success else "❌"
        print(f"{ticker:<10} {status:<10} {contracts:<10}")

    print(f"\nyfinance Results:")
    print(f"{'Ticker':<10} {'Success':<10} {'Contracts':<10}")
    print(f"{'-'*30}")
    for ticker, (success, contracts) in results['yfinance'].items():
        status = "✅" if success else "❌"
        print(f"{ticker:<10} {status:<10} {contracts:<10}")

    # Recommendation
    yq_total = sum(1 for ticker, (success, _) in results['yahooquery'].items() if success)
    yf_total = sum(1 for ticker, (success, _) in results['yfinance'].items() if success)

    print(f"\n{'='*60}")
    print(f"RECOMMENDATION")
    print(f"{'='*60}")

    if yf_total > yq_total:
        print(f"✅ Use yfinance ({yf_total}/{len(tickers)} successful)")
    elif yq_total > yf_total:
        print(f"✅ Use yahooquery ({yq_total}/{len(tickers)} successful)")
    elif yf_total == 0 and yq_total == 0:
        print(f"❌ Neither provider working - disable options_flow factor")
    else:
        print(f"⚠️ Both providers tied - use yfinance (more mature)")

    return results

if __name__ == "__main__":
    test_tickers = [
        "AAPL",  # Apple
    ]

    print("Options Data Provider Comparison")
    print("=" * 60)
    print(f"Testing: yahooquery vs yfinance")
    print(f"Tickers: {', '.join(test_tickers)}")

    results = compare_providers(test_tickers)

    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)
