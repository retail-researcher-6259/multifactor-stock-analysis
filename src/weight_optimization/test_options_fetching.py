"""
Test script for Options Data Fetching
Tests the options flow data collection and caching mechanism
"""

from curl_cffi.requests import Session as CurlSession
import yahooquery.utils as yq_utils
import types
from yahooquery import Ticker
from pathlib import Path
from datetime import datetime
import json
import pandas as pd

# Set up curl_cffi session (same as main script)
class CurlCffiSessionWrapper(CurlSession):
    """Wrapper to make curl_cffi compatible with yahooquery's expectations"""
    def mount(self, *_):  # dummy for compatibility
        pass

# Configure global session with browser impersonation
_global_curl_session = CurlCffiSessionWrapper(
    impersonate="chrome110",
    headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "en-US,en;q=0.9",
    },
    timeout=30
)

# Build a dummy module that exposes Session for yahooquery
curl_module = types.ModuleType("curl_requests")
curl_module.Session = CurlCffiSessionWrapper
yq_utils.requests = curl_module

# Monkey-patch yahooquery's requests module
yq_utils._COOKIE = {"B": "dummy"}
yq_utils._CRUMB = "dummy"
yq_utils.get_crumb = lambda *_: "dummy"

# Set up cache directory
CACHE_DIR = Path("cache")
options_cache_dir = CACHE_DIR / "options_oi"
options_cache_dir.mkdir(parents=True, exist_ok=True)

def test_single_ticker_options(ticker):
    """
    Test options data fetching for a single ticker

    Args:
        ticker: Stock symbol to test
    """
    print(f"\n{'='*60}")
    print(f"Testing Options Data for {ticker}")
    print(f"{'='*60}")

    try:
        # Step 1: Fetch options chain
        print(f"\n1. Fetching options chain...")
        t = Ticker(ticker, session=_global_curl_session)
        options_chain = t.option_chain

        print(f"   Type: {type(options_chain)}")

        # Step 2: Check if data was returned
        if options_chain is None:
            print(f"   ❌ No options data returned (None)")
            return False

        if isinstance(options_chain, str):
            print(f"   ❌ Error returned: {options_chain}")
            return False

        # yahooquery returns DataFrame with MultiIndex
        if isinstance(options_chain, pd.DataFrame):
            print(f"   ✅ Options data received (DataFrame)")
            print(f"   Shape: {options_chain.shape}")
            print(f"   Columns: {list(options_chain.columns)}")

            # Step 3: Examine structure
            print(f"\n2. Examining DataFrame structure...")

            # Get unique expiration dates from MultiIndex
            if isinstance(options_chain.index, pd.MultiIndex):
                expiration_dates = options_chain.index.get_level_values('expiration').unique()
                option_types = options_chain.index.get_level_values('optionType').unique()

                print(f"   Index levels: {options_chain.index.names}")
                print(f"   Expiration dates: {len(expiration_dates)}")
                print(f"   Option types: {list(option_types)}")
                print(f"   Total contracts: {len(options_chain)}")

                # Show first contract
                if len(options_chain) > 0:
                    print(f"\n   Sample contract data:")
                    first_contract = options_chain.iloc[0]
                    first_idx = options_chain.index[0]

                    print(f"   - Index: {first_idx}")
                    print(f"   - Strike: {first_contract.get('strike', 'N/A')}")
                    print(f"   - Open Interest: {first_contract.get('openInterest', 'N/A')}")
                    print(f"   - Volume: {first_contract.get('volume', 'N/A')}")
                    print(f"   - Last Price: {first_contract.get('lastPrice', 'N/A')}")

                contract_count = len(options_chain)
            else:
                print(f"   ⚠️ Unexpected index type: {type(options_chain.index)}")
                contract_count = 0

        elif isinstance(options_chain, dict):
            # Legacy format support
            print(f"   ✅ Options data received (Dict)")
            print(f"   Keys: {list(options_chain.keys())[:5]}... ({len(options_chain)} total)")

            # Step 3: Examine structure
            print(f"\n2. Examining dict structure...")

            contract_count = 0
            expiration_dates = []

            for exp_date, contracts in options_chain.items():
                expiration_dates.append(exp_date)
                if isinstance(contracts, pd.DataFrame):
                    contract_count += len(contracts)

                    if len(expiration_dates) == 1:  # Show first expiration details
                        print(f"\n   First expiration: {exp_date}")
                        print(f"   Contracts: {len(contracts)}")
                        print(f"   Columns: {list(contracts.columns)}")

                        # Show sample data
                        if len(contracts) > 0:
                            print(f"\n   Sample contract data:")
                            sample = contracts.iloc[0]
                            print(f"   - Strike: {sample.get('strike', 'N/A')}")
                            print(f"   - Type: {sample.get('contractSymbol', 'N/A')}")
                            print(f"   - Open Interest: {sample.get('openInterest', 'N/A')}")
                            print(f"   - Volume: {sample.get('volume', 'N/A')}")
                            print(f"   - Last Price: {sample.get('lastPrice', 'N/A')}")

            print(f"\n   ✅ Total expiration dates: {len(expiration_dates)}")
            print(f"\n   ✅ Total contracts: {contract_count}")
        else:
            print(f"   ❌ Unexpected type: {type(options_chain)}")
            return False

        # Step 4: Test caching
        print(f"\n3. Testing options data caching...")

        date = datetime.now().strftime('%Y-%m-%d')
        cache_file = options_cache_dir / f"{ticker}_{date}.json"

        oi_data = {}
        cached_count = 0

        # Helper functions for safe type conversion
        def safe_int(val, default=0):
            try:
                return int(val) if pd.notna(val) else default
            except (ValueError, TypeError):
                return default

        def safe_float(val, default=0.0):
            try:
                return float(val) if pd.notna(val) else default
            except (ValueError, TypeError):
                return default

        # Parse DataFrame format (current yahooquery)
        if isinstance(options_chain, pd.DataFrame):
            for idx, contract in options_chain.iterrows():
                # Extract MultiIndex levels: (symbol, expiration, optionType)
                if isinstance(idx, tuple) and len(idx) >= 3:
                    symbol, exp_date, option_type = idx[0], idx[1], idx[2]
                else:
                    continue

                strike = contract.get('strike')
                contract_type = 'CALL' if option_type == 'calls' else 'PUT'

                key = f"{strike}_{contract_type[0]}_{exp_date}"

                oi_data[key] = {
                    'strike': safe_float(strike),
                    'type': contract_type,
                    'expiration': str(exp_date),
                    'openInterest': safe_int(contract.get('openInterest', 0)),
                    'volume': safe_int(contract.get('volume', 0)),
                    'lastPrice': safe_float(contract.get('lastPrice', 0)),
                    'impliedVolatility': safe_float(contract.get('impliedVolatility', 0))
                }
                cached_count += 1

        # Parse dict format (legacy support)
        elif isinstance(options_chain, dict):
            for exp_date, contracts in options_chain.items():
                if isinstance(contracts, pd.DataFrame):
                    for _, contract in contracts.iterrows():
                        strike = contract.get('strike')
                        contract_symbol = contract.get('contractSymbol', '')

                        # Determine if CALL or PUT
                        is_call = 'C' in contract_symbol[-9:] if len(contract_symbol) >= 9 else False

                        key = f"{strike}_{('C' if is_call else 'P')}_{exp_date}"

                        oi_data[key] = {
                            'strike': safe_float(strike),
                            'type': 'CALL' if is_call else 'PUT',
                            'expiration': str(exp_date),
                            'openInterest': safe_int(contract.get('openInterest', 0)),
                            'volume': safe_int(contract.get('volume', 0)),
                            'lastPrice': safe_float(contract.get('lastPrice', 0)),
                            'impliedVolatility': safe_float(contract.get('impliedVolatility', 0))
                        }
                        cached_count += 1

        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump({
                'ticker': ticker,
                'date': date,
                'data': oi_data
            }, f, indent=2)

        print(f"   ✅ Cached {cached_count} contracts to: {cache_file}")
        print(f"   ✅ Cache file size: {cache_file.stat().st_size / 1024:.2f} KB")

        # Step 5: Test cache loading
        print(f"\n4. Testing cache loading...")

        with open(cache_file, 'r') as f:
            loaded_data = json.load(f)

        print(f"   ✅ Loaded {len(loaded_data['data'])} contracts from cache")

        return True

    except Exception as e:
        print(f"\n❌ Error testing {ticker}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_tickers(tickers):
    """Test options fetching for multiple tickers"""
    print(f"\n{'#'*60}")
    print(f"# Testing Options Data Fetching for {len(tickers)} tickers")
    print(f"{'#'*60}")

    results = {}

    for ticker in tickers:
        success = test_single_ticker_options(ticker)
        results[ticker] = success

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")

    successful = sum(results.values())
    failed = len(results) - successful

    print(f"\n✅ Successful: {successful}/{len(tickers)}")
    print(f"❌ Failed: {failed}/{len(tickers)}")

    if failed > 0:
        print(f"\nFailed tickers:")
        for ticker, success in results.items():
            if not success:
                print(f"  - {ticker}")

    # Check cache directory
    print(f"\n📁 Cache directory: {options_cache_dir}")
    cache_files = list(options_cache_dir.glob("*.json"))
    print(f"📁 Cached files: {len(cache_files)}")

    return results

if __name__ == "__main__":
    # Test with a few common tickers
    test_tickers = [
        "AAPL",  # Apple - very liquid
        "MSFT",  # Microsoft - very liquid
        "SPY",   # S&P 500 ETF - extremely liquid
        "TSLA",  # Tesla - high options activity
        "NVDA",  # NVIDIA - high options activity
    ]

    print("Options Data Fetching Test Script")
    print("=" * 60)
    print(f"Cache directory: {options_cache_dir}")
    print(f"Testing {len(test_tickers)} tickers...")

    results = test_multiple_tickers(test_tickers)

    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)