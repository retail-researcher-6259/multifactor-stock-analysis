# V24: Minor fixes: historic dates
# V23: Change the Volatility_penalty factor to the Stability factor
# V22: Reorganize the factors
# V21: Updated the Value factor
# V20: Convert binary functions to continuous functions
# V19: Add "Growth" factor
# V18: Applied V09 enhanced fundamental metrics to main screener
# V17: Added company name and country columns to output
# V16: Fixed yahooquery datetime comparison issues
# V15: Switch to yahooquery only (no Stooq dependency)
# V14: Fixed ROICAdj, deleted EPS_G
# V13: Add growth (revenue, EPS, ROIC)
# V12: Update technical, momentum, size, volatility indicators
# V11: Update dividend factors to the get_carry_score function

# -- curl_cffi wrapper so yahooquery never hits Yahoo directly ------------
from curl_cffi.requests import Session as CurlSession
import yahooquery.utils as yq_utils
import types
import pandas_market_calendars as mcal


class CurlCffiSessionWrapper(CurlSession):
    def mount(self, *_):  # dummy for compatibility
        pass


_global_curl_session = CurlCffiSessionWrapper(
    impersonate="chrome110",
    headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "en-US,en;q=0.9",
    },
    timeout=30
)

# Expose Session to yahooquery
curl_module = types.ModuleType("curl_requests")
curl_module.Session = CurlCffiSessionWrapper
yq_utils.requests = curl_module
yq_utils._COOKIE = {"B": "dummy"}
yq_utils._CRUMB = "dummy"
yq_utils.get_crumb = lambda *_: "dummy"

import pandas as pd
import numpy as np
import ta
import math
import time
import random
import requests
from datetime import datetime, timedelta
from yahooquery import Ticker
from pathlib import Path
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore
import json
import warnings
from io import StringIO
import sys
import os

pd.options.mode.chained_assignment = None  # disable copy-view warning

# --- Configurable Parameters ---
FRED_API_KEY = "a6472bcc951dc72f091984a09a36fc9e"
FRED_SERIES_ID = "GDP"  # U.S. Real GDP, quarterly

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd().parent.parent.parent
TICKER_FILE = PROJECT_ROOT / "config" / "Buyable_stocks_0901.txt"

INSIDER_LOOKBACK_DAYS = 90
TOP_N = 20
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# Scoring Weights (keeping proven optimal configuration)
# Total score: 120

REGIME_WEIGHTS = {
    "Steady_Growth": {
        "momentum": 130.7,
        "stability": -30.8,
        "options_flow": -18.6,
        "financial_health": 15.3,
        "technical": 14.9,
        "value": -10.6,
        "carry": -7.0,
        "growth": 3.5,
        "quality": 2.6,
        "size": 2.3,
        "liquidity": -2.3,
        "insider": 0.0,
    },
    "Strong_Bull": {
        "momentum": 138.9,
        "stability": -60.0,
        "financial_health": 40.1,
        "carry": -33.4,
        "technical": 32.3,
        "value": -23.2,
        "options_flow": -19.6,
        "growth": 13.4,
        "quality": 8.2,
        "size": 3.3,
        "insider": 0.0,
        "liquidity": 0.0,
    },
    "Crisis_Bear": {
        "momentum": 118.0,
        "stability": -50.4,
        "financial_health": 34.5,
        "technical": 28.8,
        "carry": -21.2,
        "options_flow": -18.6,
        "value": -13.3,
        "growth": 13.3,
        "size": 8.0,
        "quality": 3.6,
        "liquidity": -2.7,
        "insider": 0.0,
    }
}

# Regime-specific thresholds - Tailored to economic conditions of each regime
#
# RATIONALE:
# - Different market regimes reward different stock characteristics
# - "Strong_Bull": Growth and momentum dominate, value metrics less predictive
#   → Relaxed thresholds to include quality growth stocks with higher valuations
# - "Steady_Growth": Balanced fundamentals matter, moderate valuations preferred
#   → Standard thresholds balancing value, quality, and growth
# - "Crisis_Bear": Flight to quality, deep value and strong balance sheets critical
#   → Strict thresholds focusing on cash flow, low debt, and margin of safety
#
# This ensures factor scores align with what actually drives returns in each regime,
# improving correlation stability and optimization effectiveness.
#
REGIME_THRESHOLDS = {
    "Strong_Bull": {
        # Value thresholds - Most lenient (bull markets tolerate high valuations)
        "pe": 30,              # Bull markets: growth stocks command premium multiples
        "pb": 4.0,             # Higher book value multiples acceptable
        "ev_to_ebitda": 18,    # Enterprise value multiples expand
        "fcf_yield": 0.02,     # Lower cash yield threshold (2%)
        "ev_to_sales": 3.5,    # Sales multiples can be higher for growth

        # Quality thresholds - Slightly relaxed (momentum matters more than fundamentals)
        "roe": 0.10,           # 10% ROE acceptable in bull markets
        "roic": 0.08,          # 8% ROIC threshold
        "gross_margin": 0.20,  # 20% gross margin
        "operating_margin": 0.08,  # 8% operating margin
        "fcf_margin": 0.06,    # 6% FCF margin
        "asset_turnover": None,

        # Financial health thresholds - Relaxed (credit is cheap, risk tolerance high)
        "de_ratio": 80,        # Higher debt tolerance (80% D/E)
        "current_ratio": 1.2,  # Lower liquidity requirement
        "interest_coverage": 2.5,  # Lower coverage requirement
        "cash_debt_ratio": 0.2,    # Lower cash/debt requirement

        # Growth thresholds - Most important in bull markets
        "rev_growth": 0.08,    # Higher growth expectations (8%+)
        "growth_est": 0.15,    # Strong forward growth (15%+)
        "peg_ratio": 2.0,      # Higher PEG tolerance (growth at any price)

        # Other thresholds
        "price": 2,
        "div_yield_min": 0.01,     # Lower dividend requirement
        "payout_ratio_max": 0.80,  # Higher payout tolerance
    },

    "Steady_Growth": {
        # Value thresholds - Moderate (balanced market, some premium for quality growth)
        "pe": 22,              # Moderate P/E threshold
        "pb": 3.0,             # Moderate P/B
        "ev_to_ebitda": 14,    # Moderate EV/EBITDA
        "fcf_yield": 0.025,    # 2.5% FCF yield
        "ev_to_sales": 2.5,    # Moderate sales multiple

        # Quality thresholds - Balanced
        "roe": 0.12,           # 12% ROE - standard quality bar
        "roic": 0.10,          # 10% ROIC
        "gross_margin": 0.25,  # 25% gross margin
        "operating_margin": 0.10,  # 10% operating margin
        "fcf_margin": 0.08,    # 8% FCF margin
        "asset_turnover": None,

        # Financial health thresholds - Moderate
        "de_ratio": 60,        # 60% D/E ratio
        "current_ratio": 1.5,  # Standard liquidity
        "interest_coverage": 3.0,  # 3x interest coverage
        "cash_debt_ratio": 0.3,    # 30% cash/debt

        # Growth thresholds - Moderate
        "rev_growth": 0.06,    # 6% revenue growth
        "growth_est": 0.12,    # 12% forward growth
        "peg_ratio": 1.8,      # Moderate PEG

        # Other thresholds
        "price": 2,
        "div_yield_min": 0.015,
        "payout_ratio_max": 0.70,
    },

    "Crisis_Bear": {
        # Value thresholds - Strictest (flight to quality, deep value focus)
        "pe": 15,              # Low P/E required (bargain hunting)
        "pb": 2.0,             # Conservative P/B
        "ev_to_ebitda": 10,    # Low enterprise value multiples
        "fcf_yield": 0.04,     # High cash yield requirement (4%+)
        "ev_to_sales": 1.5,    # Low sales multiples

        # Quality thresholds - Strictest (quality and safety paramount)
        "roe": 0.15,           # 15% ROE - only high-quality businesses
        "roic": 0.12,          # 12% ROIC - efficient capital allocation critical
        "gross_margin": 0.30,  # 30% gross margin - pricing power essential
        "operating_margin": 0.12,  # 12% operating margin
        "fcf_margin": 0.10,    # 10% FCF margin - strong cash generation
        "asset_turnover": None,

        # Financial health thresholds - Strictest (balance sheet strength critical)
        "de_ratio": 40,        # Low debt tolerance (40% D/E max)
        "current_ratio": 2.0,  # High liquidity requirement
        "interest_coverage": 5.0,  # Strong coverage (5x)
        "cash_debt_ratio": 0.5,    # High cash/debt ratio (50%)

        # Growth thresholds - Less important, but stability valued
        "rev_growth": 0.03,    # Lower growth expectations (3%+, stability key)
        "growth_est": 0.08,    # Modest forward growth (8%+)
        "peg_ratio": 1.2,      # Low PEG (don't overpay for growth)

        # Other thresholds
        "price": 2,
        "div_yield_min": 0.02,     # Higher dividend requirement (income focus)
        "payout_ratio_max": 0.60,  # Lower payout ratio (prefer retained earnings)
    }
}

# Legacy single threshold set - kept for backwards compatibility
# New code should use get_regime_thresholds() instead
FUND_THRESHOLDS = {
    # Pure value thresholds
    "pe": 20,
    "pb": 2.5,
    "ev_to_ebitda": 12,
    "fcf_yield": 0.05,
    "ev_to_sales": 2.0,

    # Quality thresholds
    "roe": 0.12,
    "roic": 0.10,  # Add ROIC threshold
    "gross_margin": 0.35,
    "operating_margin": 0.10,
    "fcf_margin": 0.08,
    "asset_turnover": 0.5,

    # Financial health thresholds
    "de_ratio": 100,
    "current_ratio": 1.5,
    "interest_coverage": 3.0,
    "cash_debt_ratio": 0.3,

    # Growth thresholds
    "rev_growth": 0.05,
    "growth_est": 0.2,
    "peg_ratio": 1.5,

    # Other thresholds
    "price": 2,
    "div_yield_min": 0.015,
    "payout_ratio_max": 0.70,
}

# Replace VALUE_METRIC_WEIGHTS with pure value metrics
VALUE_METRIC_WEIGHTS = {
    'earnings_yield': 0.25,    # Most direct value measure
    'fcf_yield': 0.25,         # Cash generation vs price
    'ev_to_ebitda': 0.20,      # Enterprise value metric
    'pb': 0.15,                # Book value
    'ev_to_sales': 0.15       # Sales multiple
}

# ADD new metric weight dictionaries
QUALITY_METRIC_WEIGHTS = {
    'roe': 0.20,               # Profitability
    'roic': 0.20,              # Capital efficiency
    'gross_margin': 0.15,      # Pricing power
    'fcf_margin': 0.15,        # Cash conversion
    'asset_turnover': 0.10,    # Efficiency
    'operating_margin': 0.10,  # Operational efficiency
    'revenue_stability': 0.10  # Revenue consistency
}

FINANCIAL_HEALTH_WEIGHTS = {
    'cash_debt_ratio': 0.25,
    'current_ratio': 0.25,
    'debt_to_equity': 0.25,
    'interest_coverage': 0.25
}

ADX_PERIOD = 14

_info_cache = {}
# Global cache to remember which tickers don't have insider data
_insider_data_cache = {}
_insider_no_data_tickers = set()

# Global cache for carry scores to avoid repeated API calls
_carry_data_cache = {}
_carry_no_data_tickers = set()

# Load Marketstack API configuration
def load_marketstack_config():
    """Load Marketstack API key from config file"""
    config_file = "./config/marketstack_config.json"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
            api_key = config.get('api_key', '')
            if api_key and 'xxxxx' not in api_key and api_key != '':
                return api_key
    return None

# Load the API key once at module level
MARKETSTACK_API_KEY = load_marketstack_config()
MARKETSTACK_BASE_URL = "https://api.marketstack.com/v2"


def set_ticker_file(file_path):
    """
    Dynamically set the ticker file path

    Args:
        file_path: Path object or string to the ticker file
    """
    global TICKER_FILE
    if isinstance(file_path, str):
        TICKER_FILE = Path(file_path)
    else:
        TICKER_FILE = file_path

    if not TICKER_FILE.exists():
        raise FileNotFoundError(f"Ticker file not found: {TICKER_FILE}")

    print(f" Ticker file updated to: {TICKER_FILE}")
    return TICKER_FILE


def get_ticker_count():
    """Get the count of tickers in the current file"""
    try:
        with open(TICKER_FILE, 'r') as f:
            tickers = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            return len(tickers)
    except:
        return 0


def load_regime_periods():
    """Load regime periods from CSV file"""
    regime_file = PROJECT_ROOT / "output" / "Regime_Detection_Results" / "regime_periods.csv"

    if not regime_file.exists():
        raise FileNotFoundError(f"Regime periods file not found at {regime_file}")

    regime_df = pd.read_csv(regime_file)
    regime_df['start_date'] = pd.to_datetime(regime_df['start_date'])
    regime_df['end_date'] = pd.to_datetime(regime_df['end_date'])

    return regime_df


def is_trading_day(date, exchange='NYSE'):
    """
    Check if a given date is a trading day

    Args:
        date: pd.Timestamp or datetime object
        exchange: Stock exchange calendar to use (default: NYSE)

    Returns:
        bool: True if trading day, False otherwise
    """
    # Convert to pandas Timestamp if needed
    if not isinstance(date, pd.Timestamp):
        date = pd.Timestamp(date)

    # Ensure timezone-naive
    if date.tz is not None:
        date = date.tz_localize(None)

    # Get the market calendar
    try:
        market = mcal.get_calendar(exchange)

        # Check if the date is a trading day
        # Get valid trading days for a range around the date
        start = date - pd.Timedelta(days=5)
        end = date + pd.Timedelta(days=5)

        valid_days = market.valid_days(start_date=start, end_date=end)

        # Convert to timezone-naive for comparison
        valid_days = pd.DatetimeIndex([d.tz_localize(None) if d.tz else d for d in valid_days])

        # Check if our date is in the valid trading days
        return date.normalize() in valid_days.normalize()

    except Exception as e:
        print(f"Warning: Could not check trading calendar: {e}")
        # Fallback to simple weekend check
        return date.weekday() < 5  # Monday=0, Friday=4


def get_last_trading_day(date, max_lookback=10):
    """
    Get the most recent trading day on or before the given date

    Args:
        date: Target date
        max_lookback: Maximum days to look back

    Returns:
        pd.Timestamp: Last trading day
    """
    if not isinstance(date, pd.Timestamp):
        date = pd.Timestamp(date)

    # If it's already a trading day, return it
    if is_trading_day(date):
        return date

    # Look back up to max_lookback days
    for i in range(1, max_lookback + 1):
        check_date = date - pd.Timedelta(days=i)
        if is_trading_day(check_date):
            print(
                f" {date.strftime('%Y-%m-%d')} is not a trading day, using {check_date.strftime('%Y-%m-%d')} instead")
            return check_date

    # If no trading day found, return original date
    print(f" Could not find trading day within {max_lookback} days of {date.strftime('%Y-%m-%d')}")
    return date


def get_regime_for_date(target_date, regime_df):
    """
    Determine which regime a specific date belongs to

    Args:
        target_date: datetime object or string
        regime_df: DataFrame with regime periods

    Returns:
        regime_name (str) or None if date not in any regime
    """
    # Convert to pandas Timestamp for consistent comparison
    if isinstance(target_date, str):
        target_date = pd.Timestamp(target_date)
    elif isinstance(target_date, datetime):
        target_date = pd.Timestamp(target_date)
    elif hasattr(target_date, 'date'):  # If it's a date object
        target_date = pd.Timestamp(target_date)

    # Ensure target_date is timezone-naive
    if target_date.tz is not None:
        target_date = target_date.tz_localize(None)

    # Find regime that contains this date
    for _, row in regime_df.iterrows():
        # Ensure regime dates are also timezone-naive Timestamps
        start = pd.Timestamp(row['start_date'])
        end = pd.Timestamp(row['end_date'])

        if start.tz is not None:
            start = start.tz_localize(None)
        if end.tz is not None:
            end = end.tz_localize(None)

        if start <= target_date <= end:
            return row['regime_name']

    return None


def fetch_historical_data_for_date(ticker, target_date, lookback_days=365 * 2):
    """
    Fetch historical data up to a specific date with ADEQUATE lookback

    CRITICAL: We need at least 252 trading days BEFORE the target date
    for proper momentum calculation. So fetch 2 years of data.

    Args:
        ticker: Stock ticker symbol
        target_date: Target date for analysis
        lookback_days: Days to look back (DEFAULT INCREASED to 730 days = 2 years)

    Returns:
        DataFrame with historical price data
    """
    # Ensure we have enough historical data for momentum calculation
    # 365*2 calendar days ≈ 504 trading days, which is plenty for 252-day momentum

    if not isinstance(target_date, pd.Timestamp):
        target_date = pd.Timestamp(target_date)

    # Ensure timezone-naive
    if target_date.tz is not None:
        target_date = target_date.tz_localize(None)

    # Calculate date range
    end_date = target_date
    start_date = target_date - pd.Timedelta(days=lookback_days)

    try:
        # Use yahooquery to fetch historical data
        from yahooquery import Ticker
        t = Ticker(ticker, session=_global_curl_session)

        # Get historical data as strings to avoid timezone issues
        hist = t.history(
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval='1d'
        )

        # Check if we got valid data
        if hist is None or (isinstance(hist, pd.DataFrame) and hist.empty):
            print(f"   No data returned for {ticker}")
            return pd.DataFrame()

        # Handle MultiIndex (ticker, date) if present
        if isinstance(hist.index, pd.MultiIndex):
            # Extract data for this specific ticker
            if ticker in hist.index.get_level_values(0):
                hist = hist.xs(ticker, level=0)
            else:
                # Try to drop the first level if it exists
                hist = hist.reset_index(level=0, drop=True)

        # Ensure we have a DataFrame with the expected columns
        # yahooquery returns lowercase columns, we need to capitalize them
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'adjclose': 'Close',  # Use adjusted close as Close
            'volume': 'Volume'
        }

        # Create a clean DataFrame with proper column names
        clean_data = pd.DataFrame(index=hist.index)

        for old_col, new_col in column_mapping.items():
            if old_col in hist.columns:
                clean_data[new_col] = hist[old_col]
            # Handle case where columns might already be capitalized
            elif new_col in hist.columns:
                clean_data[new_col] = hist[new_col]

        # Special handling for Close - prefer adjclose if available
        if 'adjclose' in hist.columns:
            clean_data['Close'] = hist['adjclose']
        elif 'close' in hist.columns and 'Close' not in clean_data.columns:
            clean_data['Close'] = hist['close']

        # Ensure we have all required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in clean_data.columns]

        if missing_columns:
            print(f"   Missing columns for {ticker}: {missing_columns}")
            # If Close is missing, we can't proceed
            if 'Close' in missing_columns:
                print(f"   Critical: No Close price data for {ticker}")
                return pd.DataFrame()

            # Fill missing columns with reasonable defaults
            for col in missing_columns:
                if col == 'Volume':
                    clean_data[col] = 0  # Default volume to 0
                elif col in ['Open', 'High', 'Low']:
                    # Use Close price as fallback for other price columns
                    clean_data[col] = clean_data['Close']

        # Ensure index is timezone-naive
        if hasattr(clean_data.index, 'tz'):
            if clean_data.index.tz is not None:
                clean_data.index = clean_data.index.tz_localize(None)

        # Sort by date and remove any NaN rows
        clean_data = clean_data.sort_index()
        clean_data = clean_data.dropna(subset=['Close'])  # At minimum, we need Close prices

        # Validate we have sufficient data
        if len(clean_data) < 50:  # Need at least 50 days for technical indicators
            print(f"   Insufficient data for {ticker}: only {len(clean_data)} days")
            return pd.DataFrame()

        print(f"   Fetched {len(clean_data)} days of historical data for {ticker}")
        return clean_data

    except Exception as e:
        print(f"   Error fetching historical data for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def run_historical_scoring(target_date, progress_callback=None):
    """
    Run scoring for a specific historical date

    Args:
        target_date: Date to run scoring for (datetime or string)
        progress_callback: Optional callback for progress updates

    Returns:
        dict with results
    """
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)

    # Load regime periods
    regime_df = load_regime_periods()

    # Get regime for this date
    regime = get_regime_for_date(target_date, regime_df)

    if regime is None:
        raise ValueError(f"No regime found for date {target_date.strftime('%Y-%m-%d')}")

    print(f" Running historical scoring for {target_date.strftime('%Y-%m-%d')}")
    print(f" Detected regime: {regime}")

    # Clean regime name for file paths
    regime_clean = regime.replace("/", "_").replace(" ", "_")

    # Get weights for this regime
    global WEIGHTS
    WEIGHTS = get_regime_weights(regime_clean)

    # The rest follows similar logic to main() but with historical data
    # ... (similar processing logic as main function but using historical data)

    # Create output filename with date
    output_dir = get_output_directory(regime_clean)
    # date_str = target_date.strftime("%m%d")
    date_str = target_date.strftime("%Y%m%d")  # Changed from "%m%d" to "%Y%m%d"
    fname = f"top_ranked_stocks_{regime_clean}_{date_str}.csv"
    output_path = output_dir / fname

    return {
        'success': True,
        'output_path': str(output_path),
        'regime': regime,
        'date': target_date.strftime('%Y-%m-%d'),
        'weights': WEIGHTS
    }


def run_historical_batch(start_date=None, end_date=None, interval_days=1, progress_callback=None):
    """
    Run historical scoring for multiple dates using the complete run_historical_specific_date function
    Auto-detects regime for each date and processes with full functionality
    """
    # Convert dates to pandas Timestamps for consistency
    if end_date is None:
        end_date = pd.Timestamp.now()
    elif isinstance(end_date, str):
        end_date = pd.Timestamp(end_date)
    elif isinstance(end_date, datetime):
        end_date = pd.Timestamp(end_date)
    elif hasattr(end_date, 'date'):
        end_date = pd.Timestamp(end_date)

    if start_date is None:
        start_date = end_date - pd.Timedelta(days=365)
    elif isinstance(start_date, str):
        start_date = pd.Timestamp(start_date)
    elif isinstance(start_date, datetime):
        start_date = pd.Timestamp(start_date)
    elif hasattr(start_date, 'date'):
        start_date = pd.Timestamp(start_date)

    # Ensure dates are timezone-naive
    if start_date.tz is not None:
        start_date = start_date.tz_localize(None)
    if end_date.tz is not None:
        end_date = end_date.tz_localize(None)

    # Load regime periods once
    regime_df = load_regime_periods()

    # First pass: identify all trading days to process
    print(" Identifying trading days...")
    if progress_callback:
        progress_callback('status', "Identifying trading days in date range...")
        progress_callback('progress', 0)

    dates_to_process = []
    skipped_dates = []
    current_date = start_date

    while current_date <= end_date:
        if is_trading_day(current_date):
            dates_to_process.append(current_date)
        else:
            skipped_dates.append(current_date)
            print(f" Skipping {current_date.strftime('%Y-%m-%d')} (Market Closed)")
        current_date += pd.Timedelta(days=interval_days)

    total_trading_days = len(dates_to_process)
    total_days_in_range = len(dates_to_process) + len(skipped_dates)

    print(f" Found {total_trading_days} trading days out of {total_days_in_range} total days")
    print(f" Processing from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    if progress_callback:
        progress_callback('status', f"Processing {total_trading_days} trading days...")
        progress_callback('progress', 5)

    results = []
    successful_dates = 0
    skipped_existing = 0
    error_dates = 0
    no_regime_dates = 0

    # Process each trading day
    for i, date in enumerate(dates_to_process):
        # Calculate precise progress
        base_progress = 5
        processing_progress = int((i / total_trading_days) * 95) if total_trading_days > 0 else 0
        current_progress = base_progress + processing_progress

        if progress_callback:
            progress_callback('progress', current_progress)
            status_msg = (f"Processing {date.strftime('%Y-%m-%d')} "
                          f"(Day {i + 1}/{total_trading_days}) - "
                          f"{current_progress}% complete")
            progress_callback('status', status_msg)

        print(f"\n[{i + 1}/{total_trading_days}] Processing {date.strftime('%Y-%m-%d')}...")

        try:
            # Get regime for this date from regime_periods.csv
            regime = get_regime_for_date(date, regime_df)

            if regime is None:
                print(f"   No regime found for {date.strftime('%Y-%m-%d')}, skipping...")
                no_regime_dates += 1
                results.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'status': 'no_regime',
                    'message': 'No regime period found'
                })
                continue

            # Clean regime name for file paths
            regime_clean = regime.replace("/", "_").replace(" ", "_")

            # Check if file already exists
            output_dir = get_output_directory(regime_clean)
            # date_str = date.strftime("%m%d")
            date_str = date.strftime("%Y%m%d")  # Changed from "%m%d" to "%Y%m%d"
            # Note: Using auto-detected regime, not manual
            fname = f"top_ranked_stocks_{regime_clean}_{date_str}.csv"
            output_path = output_dir / fname

            if output_path.exists():
                print(f"   File already exists, skipping...")
                skipped_existing += 1
                results.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'regime': regime,
                    'status': 'exists',
                    'path': str(output_path),
                    'message': 'File already exists'
                })
                continue

            # Update status for actual processing
            if progress_callback:
                progress_callback('status',
                                  f"Fetching data for {date.strftime('%Y-%m-%d')} ({regime} regime)...")

            print(f"   Auto-detected regime: {regime}")

            # Create a nested progress callback for the inner function
            def nested_progress_callback(callback_type, value):
                # Don't override the main progress bar, but pass through status updates
                if callback_type == 'status' and progress_callback:
                    progress_callback('status', f"{date.strftime('%Y-%m-%d')}: {value}")

            # Use the complete run_historical_specific_date function
            # Note: We're passing the auto-detected regime, not a manual one
            result = run_historical_specific_date(
                target_date=date,
                regime=regime,  # Auto-detected regime
                progress_callback=nested_progress_callback
            )

            # Process the result
            if result.get('success'):
                successful_dates += 1
                print(f"   Successfully processed - {result.get('total_stocks', 0)} stocks ranked")
                results.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'regime': regime,
                    'status': 'success',
                    'path': result.get('output_path'),
                    'total_stocks': result.get('total_stocks', 0),
                    'message': f"Processed {result.get('total_stocks', 0)} stocks"
                })
            else:
                error_dates += 1
                error_msg = result.get('error', 'Unknown error')
                print(f"   Failed: {error_msg}")
                results.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'regime': regime,
                    'status': 'error',
                    'error': error_msg,
                    'message': f"Processing failed: {error_msg}"
                })

        except Exception as e:
            error_dates += 1
            print(f"   Error: {e}")
            results.append({
                'date': date.strftime('%Y-%m-%d'),
                'status': 'error',
                'error': str(e),
                'message': f"Exception: {str(e)}"
            })

    # Final summary
    if progress_callback:
        progress_callback('progress', 100)
        summary_msg = (f"Batch processing complete! "
                       f"Success: {successful_dates}, "
                       f"Existing: {skipped_existing}, "
                       f"No regime: {no_regime_dates}, "
                       f"Errors: {error_dates}")
        progress_callback('status', summary_msg)

    print(f"\n{'=' * 60}")
    print(f" BATCH PROCESSING SUMMARY")
    print(f"{'=' * 60}")
    print(f" Successfully processed: {successful_dates} dates")
    print(f" Skipped (existing): {skipped_existing} dates")
    print(f" No regime found: {no_regime_dates} dates")
    print(f" Errors: {error_dates} dates")
    print(f" Weekend/holidays skipped: {len(skipped_dates)} dates")
    print(f"{'=' * 60}")

    return results


def run_historical_scoring_for_date(target_date, regime_df):
    """
    DEPRECATED: This is a placeholder function.
    Use run_historical_specific_date() instead for full functionality.
    """
    # Get regime for this date
    regime = get_regime_for_date(target_date, regime_df)
    if regime is None:
        return {
            'date': target_date.strftime('%Y-%m-%d'),
            'status': 'no_regime',
            'message': 'No regime period found'
        }

    # Call the complete function
    return run_historical_specific_date(target_date, regime)


def run_historical_specific_date(target_date, regime, progress_callback=None):
    """
    Run scoring for a specific historical date with manually selected regime
    COMPLETE FIXED VERSION with proper industry adjustment

    Args:
        target_date: Date to run scoring for (datetime or pandas Timestamp)
        regime: Manually selected regime name (e.g., "Steady_Growth")
        progress_callback: Optional callback for progress updates

    Returns:
        dict with results
    """
    # Convert to pandas Timestamp for consistency
    if isinstance(target_date, str):
        target_date = pd.Timestamp(target_date)
    elif isinstance(target_date, datetime):
        target_date = pd.Timestamp(target_date)
    elif hasattr(target_date, 'date'):  # If it's a date object
        target_date = pd.Timestamp(target_date)

    # Ensure timezone-naive
    if target_date.tz is not None:
        target_date = target_date.tz_localize(None)

    print(f" Running historical scoring for {target_date.strftime('%Y-%m-%d')}")
    print(f" Using manually selected regime: {regime}")
    print(f" This mode is for comparing data sources (yfinance vs Marketstack)")

    # Clean regime name for file paths
    regime_clean = regime.replace("/", "_").replace(" ", "_")

    # Get weights for the manually selected regime
    global WEIGHTS
    WEIGHTS = get_regime_weights(regime_clean)

    # Set up progress tracking
    if progress_callback:
        progress_callback('status', f"Loading tickers for {regime} regime...")
        progress_callback('progress', 5)

    # Load tickers
    try:
        with open(TICKER_FILE, "r") as f:
            tickers = [ln.strip().replace("$", "") for ln in f if ln.strip()]
    except FileNotFoundError:
        error_msg = f"Ticker file '{TICKER_FILE}' not found!"
        if progress_callback:
            progress_callback('error', error_msg)
        raise FileNotFoundError(error_msg)

    print(f" Processing {len(tickers)} tickers for {target_date.strftime('%Y-%m-%d')}")

    if progress_callback:
        progress_callback('status', f"Fetching historical data for {target_date.strftime('%Y-%m-%d')}...")
        progress_callback('progress', 10)

    # Pre-scan for liquidity
    avg_vol = {}
    for tk in tickers:
        try:
            avg_vol[tk] = fast_info(tk).get("averageVolume10days", 0)
        except Exception:
            avg_vol[tk] = 0
    vol_min, vol_max = min(avg_vol.values()), max(avg_vol.values())

    # Main data harvest
    raw = []
    pm_cache, em_cache, mcaps = [], [], []
    value_metrics_by_ticker = {}
    processed_count = 0
    total_tickers = len(tickers)

    for i, tk in enumerate(tickers, 1):
        if progress_callback:
            progress = 10 + int((i / total_tickers) * 80)  # 10-90% for processing
            progress_callback('progress', progress)
            progress_callback('status', f"Processing {tk} ({i}/{total_tickers})...")

        print(f"[{i}/{total_tickers}] Processing {tk} for {target_date.strftime('%Y-%m-%d')}...")

        try:
            # Get fundamentals (current, not historical)
            info = safe_get_info(tk)

            # Get historical price data up to target date
            hist = fetch_historical_data_for_date(tk, target_date, lookback_days=365 * 2)

            if hist.empty or len(hist) < 200:
                print(f"   Insufficient historical data for {tk}, skipping...")
                continue

            # Calculate Factor Scores using historical data with regime-specific thresholds
            value, quality, financial_health, roic, value_dict = get_fundamentals(tk, regime_clean)
            value_metrics_by_ticker[tk] = value_dict

            tech = get_technical_score(hist)
            insider = get_insider_score_simple(tk)
            growth = get_growth_score(tk, info, regime_clean)

            # Momentum calculations
            pm, em = get_momentum_score(tk, hist)
            pm_cache.append(pm)
            em_cache.append(em)

            # Stability score
            stability = get_stability_score(hist)

            # Other factors
            mcap = info.get("marketCap")
            mcaps.append(mcap)
            sector, industry = sector_industry(info)

            company_name, country = get_company_info(info)
            options_flow = get_options_flow_score(tk, info)
            carry = get_carry_score_simple(tk, info)

            # Liquidity score
            liq = 0
            if vol_max != vol_min:
                liq = round((avg_vol[tk] - vol_min) / (vol_max - vol_min), 2)

            raw.append(dict(
                Ticker=tk, Value=value, Quality=quality,
                FinancialHealth=financial_health,
                Technical=tech, Insider=insider, PriceMom=pm, EarnMom=em,
                Stability=stability, Growth=growth,
                MarketCap=mcap, OptionsFlow=options_flow,
                Liquidity=liq, Carry=carry, Sector=sector, Industry=industry,
                ROIC=roic,
                CompanyName=company_name, Country=country
            ))

            processed_count += 1
            print(f"   {tk} successfully processed")

        except Exception as e:
            print(f"   Error processing {tk}: {e}")
            continue

    if not raw:
        error_msg = f"No usable tickers for {target_date.strftime('%Y-%m-%d')}"
        print(error_msg)
        if progress_callback:
            progress_callback('error', error_msg)
        return {'success': False, 'error': error_msg}

    # Finalizing scores
    if progress_callback:
        progress_callback('progress', 90)
        progress_callback('status', "Finalizing scores and rankings...")

    print(f" Processed {processed_count} tickers successfully")

    # Convert to DataFrame
    df_raw = pd.DataFrame(raw)

    # Use raw growth score directly (already uses absolute thresholds)
    if 'Growth' in df_raw.columns:
        df_raw['GrowthScore_Norm'] = df_raw['Growth']
    else:
        df_raw['GrowthScore_Norm'] = 0

    # Use raw value score directly (already uses absolute thresholds)
    # No industry-relative normalization needed
    df_raw['ValueScore'] = df_raw['Value']

    # Calculate final scores
    results = []

    for k, rec in df_raw.iterrows():
        # Use absolute momentum scoring (no z-score normalization)
        pm_raw = rec['PriceMom']
        em_raw = rec['EarnMom']

        pm_score = absolute_momentum_score(pm_raw)
        em_score = absolute_earnings_momentum_score(em_raw)

        momentum = round(0.6 * pm_score + 0.4 * em_score, 3)

        size = get_size_score({'marketCap': rec['MarketCap']})

        total = (
                rec['ValueScore'] * WEIGHTS['value'] +
                rec['Quality'] * WEIGHTS['quality'] +
                rec['FinancialHealth'] * WEIGHTS['financial_health'] +
                rec['Technical'] * WEIGHTS['technical'] +
                rec['Insider'] * WEIGHTS['insider'] +
                momentum * WEIGHTS['momentum'] +
                rec['Stability'] * WEIGHTS['stability'] +
                size * WEIGHTS['size'] +
                rec['OptionsFlow'] * WEIGHTS['options_flow'] +
                rec['Liquidity'] * WEIGHTS['liquidity'] +
                rec['Carry'] * WEIGHTS['carry'] +
                rec.get('GrowthScore_Norm', 0) * WEIGHTS['growth']
        )

        results.append({
            "Ticker": rec['Ticker'],
            "CompanyName": rec['CompanyName'],
            "Country": rec['Country'],
            "Score": round(total, 2),
            "Value": round(rec['ValueScore'], 3),
            "Quality": rec['Quality'],
            "FinancialHealth": round(rec['FinancialHealth'], 3),
            "Technical": rec['Technical'],
            "Insider": rec['Insider'],
            "Momentum": momentum,
            "Stability": rec['Stability'],
            "Size": size,
            "OptionsFlow": rec['OptionsFlow'],
            "Liquidity": rec['Liquidity'],
            "Carry": rec['Carry'],
            "Growth": round(rec.get('GrowthScore_Norm', 0), 3),
            "Sector": rec['Sector'],
            "Industry": rec['Industry']
        })

    # Create output
    if progress_callback:
        progress_callback('progress', 95)
        progress_callback('status', "Generating output file...")

    df = pd.DataFrame(results).sort_values("Score", ascending=False)

    # Create output filename with date and manual regime indicator
    output_dir = get_output_directory(regime_clean)
    # date_str = target_date.strftime("%m%d")
    date_str = target_date.strftime("%Y%m%d")  # Changed from "%m%d" to "%Y%m%d"
    fname = f"top_ranked_stocks_{regime_clean}_{date_str}.csv"
    output_path = output_dir / fname

    # Save the file
    df.to_csv(output_path, index=False)
    print(f" Saved {fname} to {output_dir}")
    print(f" Top 10 stocks for {target_date.strftime('%Y-%m-%d')} with {regime} regime:")
    print(df.head(10)[['Ticker', 'CompanyName', 'Score']])

    if progress_callback:
        progress_callback('progress', 100)
        progress_callback('status', f"Complete! Analyzed {len(df)} stocks for {target_date.strftime('%Y-%m-%d')}")

    return {
        'success': True,
        'output_path': str(output_path),
        'results_df': df,
        'regime': regime,
        'date': target_date.strftime('%Y-%m-%d'),
        'total_stocks': len(df),
        'weights': WEIGHTS,
        'mode': 'historical_specific'
    }


def run_historical_batch_with_regime(start_date=None, end_date=None, manual_regime=None, interval_days=1,
                                     progress_callback=None):
    """
    Run historical scoring for multiple dates using a MANUALLY SELECTED regime
    Similar to run_historical_batch but uses the same regime for all dates

    Args:
        start_date: Start date for processing
        end_date: End date for processing
        manual_regime: Manually selected regime to use for ALL dates (e.g., "Steady Growth")
        interval_days: Interval between dates (default 1 for daily)
        progress_callback: Optional callback for progress updates

    Returns:
        List of results for each date processed
    """
    # Convert dates to pandas Timestamps for consistency
    if end_date is None:
        end_date = pd.Timestamp.now()
    elif isinstance(end_date, str):
        end_date = pd.Timestamp(end_date)
    elif isinstance(end_date, datetime):
        end_date = pd.Timestamp(end_date)
    elif hasattr(end_date, 'date'):
        end_date = pd.Timestamp(end_date)

    if start_date is None:
        start_date = end_date - pd.Timedelta(days=365)
    elif isinstance(start_date, str):
        start_date = pd.Timestamp(start_date)
    elif isinstance(start_date, datetime):
        start_date = pd.Timestamp(start_date)
    elif hasattr(start_date, 'date'):
        start_date = pd.Timestamp(start_date)

    # Ensure dates are timezone-naive
    if start_date.tz is not None:
        start_date = start_date.tz_localize(None)
    if end_date.tz is not None:
        end_date = end_date.tz_localize(None)

    # Validate manual regime is provided
    if manual_regime is None:
        raise ValueError("Manual regime must be provided for this mode")

    # First pass: identify all trading days to process
    print(f" Identifying trading days with manual regime: {manual_regime}")
    if progress_callback:
        progress_callback('status', f"Identifying trading days for {manual_regime} regime...")
        progress_callback('progress', 0)

    dates_to_process = []
    skipped_dates = []
    current_date = start_date

    while current_date <= end_date:
        if is_trading_day(current_date):
            dates_to_process.append(current_date)
        else:
            skipped_dates.append(current_date)
            print(f" Skipping {current_date.strftime('%Y-%m-%d')} (Market Closed)")
        current_date += pd.Timedelta(days=interval_days)

    total_trading_days = len(dates_to_process)
    total_days_in_range = len(dates_to_process) + len(skipped_dates)

    print(f" Found {total_trading_days} trading days out of {total_days_in_range} total days")
    print(f" Processing from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f" Using manual regime: {manual_regime} for all dates")

    if progress_callback:
        progress_callback('status', f"Processing {total_trading_days} trading days with {manual_regime} regime...")
        progress_callback('progress', 5)

    results = []
    successful_dates = 0
    skipped_existing = 0
    error_dates = 0

    # Process each trading day with the SAME manual regime
    for i, date in enumerate(dates_to_process):
        # Calculate precise progress
        base_progress = 5
        processing_progress = int((i / total_trading_days) * 95) if total_trading_days > 0 else 0
        current_progress = base_progress + processing_progress

        if progress_callback:
            progress_callback('progress', current_progress)
            progress_callback('status',
                              f"Processing {date.strftime('%Y-%m-%d')} ({i + 1}/{total_trading_days}) with {manual_regime}")

        print(f"\n{'=' * 60}")
        print(f" Processing date {i + 1}/{total_trading_days}: {date.strftime('%Y-%m-%d')}")
        print(f" Using manual regime: {manual_regime}")
        print(f"{'=' * 60}")

        try:
            # Create a nested progress callback for the inner function
            def nested_progress_callback(callback_type, value):
                if callback_type == 'status' and progress_callback:
                    progress_callback('status', f"{date.strftime('%Y-%m-%d')}: {value}")

            # Use run_historical_specific_date with the MANUAL regime
            result = run_historical_specific_date(
                target_date=date,
                regime=manual_regime,  # Use manual regime for ALL dates
                progress_callback=nested_progress_callback
            )

            # Process the result
            if result.get('success'):
                successful_dates += 1
                print(f"   Successfully processed - {result.get('total_stocks', 0)} stocks ranked")
                results.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'regime': manual_regime,
                    'status': 'success',
                    'path': result.get('output_path'),
                    'total_stocks': result.get('total_stocks', 0),
                    'message': f"Processed {result.get('total_stocks', 0)} stocks"
                })
            else:
                error_dates += 1
                error_msg = result.get('error', 'Unknown error')
                print(f"   Failed: {error_msg}")
                results.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'regime': manual_regime,
                    'status': 'error',
                    'error': error_msg,
                    'message': f"Processing failed: {error_msg}"
                })

        except Exception as e:
            error_dates += 1
            print(f"   Error: {e}")
            results.append({
                'date': date.strftime('%Y-%m-%d'),
                'regime': manual_regime,
                'status': 'error',
                'error': str(e),
                'message': f"Exception: {str(e)}"
            })

    # Final summary
    if progress_callback:
        progress_callback('progress', 100)
        summary_msg = (f"Batch processing complete! "
                       f"Success: {successful_dates}, "
                       f"Errors: {error_dates}")
        progress_callback('status', summary_msg)

    print(f"\n{'=' * 60}")
    print(f" BATCH PROCESSING SUMMARY (Manual Regime: {manual_regime})")
    print(f"{'=' * 60}")
    print(f" Successfully processed: {successful_dates} dates")
    print(f" Errors: {error_dates} dates")
    print(f" Weekend/holidays skipped: {len(skipped_dates)} dates")
    print(f"{'=' * 60}")

    return results

# Add a progress callback mechanism
class ProgressTracker:
    """Track progress and emit updates for UI"""

    def __init__(self, callback=None):
        self.callback = callback
        self.total_tickers = 0
        self.processed_tickers = 0

    def set_total(self, total):
        self.total_tickers = total

    def update_progress(self, ticker_name, action="processing"):
        if action == "completed":
            self.processed_tickers += 1

        if self.callback:
            # Calculate progress: 90% for data fetching, 10% for final processing
            fetch_progress = (self.processed_tickers / self.total_tickers) * 90 if self.total_tickers > 0 else 0
            total_progress = int(fetch_progress)

            self.callback('progress', total_progress)
            self.callback('status', f"Processing {ticker_name} ({self.processed_tickers}/{self.total_tickers})...")

# Global progress tracker
_progress_tracker = None

def set_progress_callback(callback):
    """Set the progress callback function"""
    global _progress_tracker
    _progress_tracker = ProgressTracker(callback)

# Add a function to get weights for a specific regime
def get_regime_weights(regime_name):
    """Get weights for a specific regime"""
    return REGIME_WEIGHTS.get(regime_name, REGIME_WEIGHTS["Steady_Growth"])

def get_regime_thresholds(regime_name):
    """Get thresholds for a specific regime"""
    return REGIME_THRESHOLDS.get(regime_name, REGIME_THRESHOLDS["Steady_Growth"])

# Add a function to set output directory based on regime
def get_output_directory(regime_name):
    """Get output directory path for a specific regime"""
    output_base = PROJECT_ROOT / "output" / "Ranked_Lists"
    regime_dir = output_base / regime_name
    regime_dir.mkdir(parents=True, exist_ok=True)
    return regime_dir

def fast_info(ticker):
    if ticker in _info_cache:
        return _info_cache[ticker]

    yq = Ticker(ticker, session=_global_curl_session)
    data = yq.get_modules([
        "summaryDetail",
        "defaultKeyStatistics",
        "financialData",
        "assetProfile",  # For company name and country
        "quoteType",  # Additional info source
    ])
    block = data.get(ticker, {})
    info = {}
    for key in ("summaryDetail", "defaultKeyStatistics",
                "financialData", "assetProfile", "quoteType"):
        info |= block.get(key, {}) or {}

    _info_cache[ticker] = info
    return info

def safe_get_info(ticker, retries=3, delay_range=(1, 3)):
    for attempt in range(retries):
        try:
            return fast_info(ticker)
        except Exception as e:
            print(f" Attempt {attempt + 1}/{retries} failed for {ticker}: {e}")
            time.sleep(random.uniform(*delay_range))
    print(f" All attempts failed for {ticker}")
    return {}

def get_company_info(info):
    """Extract company name and country from the info dict"""
    # Try multiple fields for company name
    company_name = (
            info.get("longName") or
            info.get("shortName") or
            info.get("companyName") or
            "N/A"
    )

    # Get country information
    country = info.get("country") or "N/A"

    return company_name, country

# def yahooquery_hist(ticker, years=5):
#     """
#     Fetch historical data with proper column formatting
#     Uses adjusted close prices for better accuracy
#     """
#     try:
#         # DEBUG: Starting data fetch
#         print(f"    → Fetching {years}-year history for {ticker}...")
#         yq = Ticker(ticker, session=_global_curl_session)
#
#         # Calculate date range
#         end_date = datetime.now()
#         start_date = end_date - timedelta(days=365 * years)
#         start_str = start_date.strftime('%Y-%m-%d')
#         end_str = end_date.strftime('%Y-%m-%d')
#
#         # Get historical data
#         hist = yq.history(start=start_str, end=end_str, interval='1d')
#
#         # DEBUG: Show initial fetch result
#         if hist is None:
#             print(f"    → History fetch returned None for {ticker}")
#         elif hist.empty:
#             print(f"    → History fetch returned empty DataFrame for {ticker}")
#         else:
#             print(f"    → Initial fetch successful: {len(hist)} raw records")
#
#         if hist is None or hist.empty:
#             return pd.DataFrame()
#
#         # Handle MultiIndex case
#         if isinstance(hist.index, pd.MultiIndex):
#             if ticker in hist.index.get_level_values(0):
#                 hist = hist.xs(ticker, level=0)
#             else:
#                 hist = hist.reset_index(level=0, drop=True)
#
#         # Standardize column names
#         column_mapping = {
#             'open': 'Open',
#             'high': 'High',
#             'low': 'Low',
#             'close': 'Close',
#             'adjclose': 'Close',  # Use adjusted close
#             'volume': 'Volume'
#         }
#
#         clean_data = pd.DataFrame(index=hist.index)
#
#         for old_col, new_col in column_mapping.items():
#             if old_col in hist.columns:
#                 clean_data[new_col] = hist[old_col]
#             elif new_col in hist.columns:
#                 clean_data[new_col] = hist[new_col]
#
#         # Prefer adjusted close if available
#         if 'adjclose' in hist.columns:
#             clean_data['Close'] = hist['adjclose']
#         elif 'close' in hist.columns and 'Close' not in clean_data.columns:
#             clean_data['Close'] = hist['close']
#
#         # Ensure all required columns exist
#         required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
#         for col in required_columns:
#             if col not in clean_data.columns:
#                 if col == 'Volume':
#                     clean_data[col] = 0
#                 elif col in ['Open', 'High', 'Low'] and 'Close' in clean_data.columns:
#                     clean_data[col] = clean_data['Close']
#
#         # Clean up index
#         if hasattr(clean_data.index, 'tz') and clean_data.index.tz is not None:
#             clean_data.index = clean_data.index.tz_localize(None)
#
#         # Sort and clean
#         clean_data = clean_data.sort_index()
#         clean_data = clean_data.dropna(subset=['Close'])
#
#         # # Filter by years if needed
#         # if years and len(clean_data) > 0:
#         #     cutoff_date = datetime.now() - timedelta(days=365 * years)
#         #     # Convert index to datetime for comparison
#         #     clean_data = clean_data[pd.to_datetime(clean_data.index) >= pd.Timestamp(cutoff_date)]
#
#         # Filter by years if needed
#         if years and len(clean_data) > 0:
#             try:
#                 # Calculate cutoff date
#                 cutoff_date = datetime.now() - timedelta(days=365 * years)
#
#                 # Ensure both sides are comparable Timestamps
#                 cutoff_timestamp = pd.Timestamp(cutoff_date).tz_localize(None)
#
#                 # Convert index to Timestamp if it isn't already
#                 if not isinstance(clean_data.index, pd.DatetimeIndex):
#                     clean_data.index = pd.to_datetime(clean_data.index)
#
#                 # Ensure index is timezone-naive for comparison
#                 if clean_data.index.tz is not None:
#                     clean_data.index = clean_data.index.tz_localize(None)
#
#                 # Now safe to compare
#                 clean_data = clean_data[clean_data.index >= cutoff_timestamp]
#
#             except Exception as e:
#                 print(f"     Warning: Could not filter by date range: {e}")
#                 # If filtering fails, return all data rather than failing completely
#                 pass
#
#         # At the end, before returning clean_data:
#         print(f"    → Final processed data: {len(clean_data)} clean records")
#
#         return clean_data
#
#     except Exception as e:
#         print(f"Error fetching history for {ticker}: {e}")
#         return pd.DataFrame()


def yahooquery_hist(ticker, years=5):
    """
    Maximally robust version that handles any timezone issues
    Fetch historical data with proper column formatting
    Uses adjusted close prices for better accuracy
    """
    try:
        # DEBUG: Starting data fetch
        print(f"    → Fetching {years}-year history for {ticker}...")

        # Use a direct approach with error handling
        yq = Ticker(ticker, session=_global_curl_session)

        # Calculate date range as strings to avoid any timezone issues
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        # Get historical data with minimal processing initially
        try:
            hist = yq.history(start=start_str, end=end_str, interval='1d')
            # DEBUG: Show initial fetch result
            if hist is None:
                print(f"    → History fetch returned None for {ticker}")
            elif hist.empty:
                print(f"    → History fetch returned empty DataFrame for {ticker}")
            else:
                print(f"    → Initial fetch successful: {len(hist)} raw records")
        except Exception as e:
            print(f"Error fetching history for {ticker}: {e}")
            return pd.DataFrame()

        # Create a completely new DataFrame to avoid any hidden metadata issues
        if hist is None or hist.empty:
            return pd.DataFrame()

        # Extract data, avoiding any index operations that might trigger timezone issues
        try:
            # Handle MultiIndex case
            if isinstance(hist.index, pd.MultiIndex):
                # Try to extract just this ticker's data
                if ticker in hist.index.get_level_values(0):
                    hist = hist.xs(ticker, level=0)
                else:
                    # Fall back to dropping the first level
                    hist = hist.reset_index(level=0, drop=True)

            # Create fresh DataFrame with timezone-naive index
            fresh_data = []
            for idx, row in hist.iterrows():
                # Convert index to string and then back to datetime to strip timezone
                date_str = pd.Timestamp(idx).strftime('%Y-%m-%d')
                fresh_data.append({
                    'Date': pd.Timestamp(date_str),
                    'Open': row.get('open', row.get('Open', None)),
                    'High': row.get('high', row.get('High', None)),
                    'Low': row.get('low', row.get('Low', None)),
                    'Close': row.get('adjclose', row.get('close', row.get('Close', None))),  # Prefer adjusted close
                    'Volume': row.get('volume', row.get('Volume', None))
                })

            # Create a completely new DataFrame with a clean index
            result_df = pd.DataFrame(fresh_data)
            if result_df.empty:
                return pd.DataFrame()

            # Set Date as index
            result_df.set_index('Date', inplace=True)

            # Ensure columns are properly capitalized
            result_df.columns = [col.title() for col in result_df.columns]

            # Filter by years with timezone-naive cutoff
            if years:
                cutoff = pd.Timestamp((datetime.now() - timedelta(days=365 * years)).strftime('%Y-%m-%d'))
                result_df = result_df[result_df.index >= cutoff]

            final_df = result_df.dropna().sort_index()

            # DEBUG: Final result
            print(f"    → Final processed data: {len(final_df)} clean records")

            return final_df

        except Exception as e:
            print(f"Error fetching history for {ticker}: {e}")
            return pd.DataFrame()

    except Exception as e:
        print(f"   Error fetching history for {ticker}: {e}")
        return pd.DataFrame()

# --- Helper Functions ---
def sector_industry(info):
    """Return (sector, industry) strings or 'Unknown' placeholders."""
    return (
        info.get("sector") or info.get("sectorDisp") or "Unknown",
        info.get("industry") or info.get("industryDisp") or "Unknown"
    )


def get_fundamentals(ticker, regime_name="Steady_Growth"):
    """
    Returns four scores: (value_score, quality_score, financial_health_score, roic)
    Each factor is scored on a continuous 0-1 scale using regime-specific thresholds.
    Also returns value_metrics_dict for industry adjustment.

    Args:
        ticker: Stock ticker symbol
        regime_name: Market regime name ("Strong_Bull", "Steady_Growth", "Crisis_Bear")
    """
    try:
        info = fast_info(ticker)

        # Get regime-specific thresholds
        thresholds = get_regime_thresholds(regime_name)

        # Basic metrics extraction (same as before)
        pe = info.get("trailingPE")
        pb = info.get("priceToBook")
        ev_to_ebitda = info.get("enterpriseToEbitda")
        fcf = info.get("freeCashflow")
        mcap = info.get("marketCap")
        fcf_yield = fcf / mcap if fcf and mcap else None
        de = info.get("debtToEquity")
        current_ratio = info.get("currentRatio")
        roe = info.get("returnOnEquity")
        gross_margin = info.get("grossMargins")
        ev_to_sales = info.get("enterpriseToRevenue")
        ebit = info.get("ebit")
        interest_expense = info.get("interestExpense")
        interest_coverage = ebit / interest_expense if ebit and interest_expense and interest_expense > 0 else None
        operating_margin = info.get("operatingMargins")
        cash = info.get("totalCash")
        debt = info.get("totalDebt")
        cash_debt_ratio = cash / debt if cash and debt and debt > 0 else None
        total_assets = info.get("totalAssets")
        revenue = info.get("totalRevenue")
        asset_turnover = revenue / total_assets if revenue and total_assets and total_assets > 0 else None

        # Calculate ROIC (new)
        invested_capital = info.get("totalAssets", 0) - info.get("totalCurrentLiabilities", 0)
        roic = ebit / invested_capital if ebit and invested_capital and invested_capital > 0 else None

        # Calculate earnings yield
        earnings_yield = 1 / pe if pe and pe > 0 else None

        # Helper function remains the same
        def add_score(score_list, metric_name, value, threshold, excellent_val, is_higher_better=False):
            if value is None or not np.isfinite(value):
                return None

            if is_higher_better:
                poor_val = threshold / 2
                score = max(0, min(1, (value - poor_val) / (excellent_val - poor_val)))
            else:
                poor_val = threshold * 1.5
                score = max(0, min(1, (poor_val - value) / (poor_val - excellent_val)))

            if isinstance(score_list, dict):
                score_list[metric_name] = score
            else:
                score_list.append(score)
            return score

        # 1. PURE VALUE METRICS (simplified) - using regime-specific thresholds
        value_scores_dict = {}

        if earnings_yield is not None:
            add_score(value_scores_dict, 'earnings_yield', earnings_yield, 0.05, 0.10, True)

        if fcf_yield is not None and fcf_yield > 0:
            add_score(value_scores_dict, 'fcf_yield', fcf_yield, thresholds['fcf_yield'], 0.10, True)

        if ev_to_ebitda is not None and ev_to_ebitda > 0:
            add_score(value_scores_dict, 'ev_to_ebitda', ev_to_ebitda, thresholds['ev_to_ebitda'], 6, False)

        if pb is not None and pb > 0:
            add_score(value_scores_dict, 'pb', pb, thresholds['pb'], 1, False)

        if ev_to_sales is not None and ev_to_sales > 0:
            add_score(value_scores_dict, 'ev_to_sales', ev_to_sales, thresholds['ev_to_sales'], 1, False)

        # 2. QUALITY METRICS (expanded) - using regime-specific thresholds
        quality_scores_dict = {}

        if roe is not None:
            add_score(quality_scores_dict, 'roe', roe, thresholds['roe'], 0.25, True)

        if roic is not None:
            add_score(quality_scores_dict, 'roic', roic, thresholds['roic'], 0.20, True)

        if gross_margin is not None:
            add_score(quality_scores_dict, 'gross_margin', gross_margin, thresholds['gross_margin'], 0.6, True)

        if operating_margin is not None:
            add_score(quality_scores_dict, 'operating_margin', operating_margin, thresholds['operating_margin'],
                      0.2, True)

        # FCF margin
        fcf_margin = fcf / revenue if fcf and revenue and revenue > 0 else None
        if fcf_margin is not None:
            add_score(quality_scores_dict, 'fcf_margin', fcf_margin, thresholds['fcf_margin'], 0.15, True)

        if asset_turnover is not None and asset_turnover > 0:
            add_score(quality_scores_dict, 'asset_turnover', asset_turnover, thresholds['asset_turnover'], 1.0,
                      True)

        # Revenue stability (new) - would need historical data
        # For now, use gross margin stability as proxy
        quality_scores_dict['revenue_stability'] = quality_scores_dict.get('gross_margin', 0) * 0.8

        # 3. FINANCIAL HEALTH METRICS (new separate factor) - using regime-specific thresholds
        financial_health_scores = []

        if cash_debt_ratio is not None and cash_debt_ratio > 0:
            add_score(financial_health_scores, None, cash_debt_ratio, thresholds['cash_debt_ratio'], 0.6, True)

        if current_ratio is not None and current_ratio > 0:
            add_score(financial_health_scores, None, current_ratio, thresholds['current_ratio'], 2.5, True)

        if de is not None and de >= 0:
            add_score(financial_health_scores, None, de, thresholds['de_ratio'], 50, False)

        if interest_coverage is not None and interest_coverage > 0:
            add_score(financial_health_scores, None, interest_coverage, thresholds['interest_coverage'], 10, True)

        # Calculate weighted scores
        # Value score
        value_score = 0
        total_weight = 0
        for metric, score in value_scores_dict.items():
            weight = VALUE_METRIC_WEIGHTS.get(metric, 0.2)
            value_score += score * weight
            total_weight += weight
        value_score = value_score / total_weight if total_weight > 0 else 0

        # Quality score
        quality_score = 0
        total_weight = 0
        for metric, score in quality_scores_dict.items():
            weight = QUALITY_METRIC_WEIGHTS.get(metric, 0.14)
            quality_score += score * weight
            total_weight += weight
        quality_score = quality_score / total_weight if total_weight > 0 else 0

        # Financial health score (simple average)
        financial_health_score = sum(financial_health_scores) / len(
            financial_health_scores) if financial_health_scores else 0

        return (round(value_score, 3),
                round(quality_score, 3),
                round(financial_health_score, 3),
                roic,
                value_scores_dict)

    except Exception as e:
        print(f" Error fetching fundamentals for {ticker}: {e}")
        return 0, 0, 0, None, {}


def get_growth_score(ticker, info=None, regime_name="Steady_Growth"):
    """
    Forward-looking growth factor with emphasis on growth acceleration and estimates.
    Uses regime-specific thresholds for growth expectations.

    Args:
        ticker: Stock ticker symbol
        info: Optional pre-fetched info dict
        regime_name: Market regime name ("Strong_Bull", "Steady_Growth", "Crisis_Bear")
    """
    if info is None:
        info = fast_info(ticker)

    # Get regime-specific thresholds
    thresholds = get_regime_thresholds(regime_name)

    growth_metrics = {}

    # 1. Revenue Growth (keep but reduce weight) - using regime threshold
    rev_growth = info.get("revenueGrowth", 0)
    if rev_growth is not None and not pd.isna(rev_growth):
        rev_threshold = thresholds['rev_growth']
        norm_rev_growth = min(max((rev_growth - rev_threshold) / 0.25, 0), 1)
        growth_metrics['revenue_growth'] = (norm_rev_growth, 0.8)  # Reduced weight

    # 2. PEG Ratio (moved from value - lower is better for growth) - using regime threshold
    peg = info.get("pegRatio")
    peg_threshold = thresholds['peg_ratio']
    if peg is not None and peg > 0:
        # Invert: PEG < 1 is excellent, > peg_threshold is poor
        norm_peg = max(0, min(1, (peg_threshold - peg) / (peg_threshold - 1.0)))
        growth_metrics['peg_ratio'] = (norm_peg, 0.9)

    # 3. Forward estimates and revisions (enhanced)
    try:
        t = Ticker(ticker, session=_global_curl_session)

        # Get earnings trend for multiple periods
        earnings_trend = t.earnings_trend.get(ticker, [])

        if earnings_trend:
            # Current and next quarter estimates
            current_q = next((e for e in earnings_trend if e.get('period') == '0q'), None)
            next_q = next((e for e in earnings_trend if e.get('period') == '+1q'), None)
            current_y = next((e for e in earnings_trend if e.get('period') == '0y'), None)
            next_y = next((e for e in earnings_trend if e.get('period') == '+1y'), None)

            # Earnings revision trend (very important)
            if current_q:
                est = current_q.get('earningsEstimate', {})
                current_est = est.get('avg', 0)
                est_30d_ago = est.get('avg30daysAgo', current_est)
                est_60d_ago = est.get('avg60daysAgo', est_30d_ago)
                est_90d_ago = est.get('avg90daysAgo', est_60d_ago)

                if est_90d_ago > 0:
                    # Calculate revision momentum
                    revision_3m = (current_est - est_90d_ago) / est_90d_ago
                    revision_1m = (current_est - est_30d_ago) / est_30d_ago if est_30d_ago > 0 else 0

                    # Positive and accelerating revisions are best
                    revision_score = min(max((revision_3m + revision_1m) / 0.1, -1), 1)
                    norm_revision = (revision_score + 1) / 2  # Convert to 0-1
                    growth_metrics['estimate_revision'] = (norm_revision, 1.0)  # Highest weight

            # Forward growth rates
            if current_y and next_y:
                current_eps = current_y.get('earningsEstimate', {}).get('avg', 0)
                next_eps = next_y.get('earningsEstimate', {}).get('avg', 0)

                if current_eps > 0:
                    forward_growth = (next_eps - current_eps) / current_eps
                    norm_forward = min(max((forward_growth - 0.05) / 0.25, 0), 1)
                    growth_metrics['forward_eps_growth'] = (norm_forward, 0.9)

        # Get revenue estimates if available
        revenue_trend = t.revenue_trend.get(ticker, [])
        if revenue_trend:
            current_q_rev = next((e for e in revenue_trend if e.get('period') == '0q'), None)
            if current_q_rev:
                rev_est = current_q_rev.get('revenueEstimate', {})
                growth = rev_est.get('growth', 0)
                if growth is not None:
                    norm_rev_growth_forward = min(max((growth - 0.05) / 0.25, 0), 1)
                    growth_metrics['forward_revenue_growth'] = (norm_rev_growth_forward, 0.8)

    except Exception as e:
        pass

    # 4. Price target upside (already exists, just enhance weight)
    price = info.get("regularMarketPrice")
    target = info.get("targetMeanPrice")
    if price is not None and target is not None and price > 0:
        upside = (target / price) - 1
        norm_upside = min(max((upside - 0.1) / 0.4, 0), 1)
        growth_metrics['price_target'] = (norm_upside, 0.7)

    # 5. Operating leverage (if available)
    try:
        # Simple version: check if margins are expanding
        current_margin = info.get('operatingMargins', 0)
        ttm_margin = info.get('profitMargins', 0)

        if current_margin > 0 and ttm_margin > 0:
            margin_expansion = (current_margin - ttm_margin) / ttm_margin
            norm_leverage = min(max((margin_expansion + 0.1) / 0.2, 0), 1)
            growth_metrics['operating_leverage'] = (norm_leverage, 0.6)
    except:
        pass

    # Calculate weighted average
    if growth_metrics:
        total_score = 0
        total_weight = 0

        for score, weight in growth_metrics.values():
            total_score += score * weight
            total_weight += weight

        if total_weight > 0:
            return round(total_score / total_weight, 3)

    return 0


def apply_sector_relative_growth_adjustment(df_raw):
    """
    Calculate sector-relative growth scores to account for
    different growth expectations across sectors
    """
    # Calculate raw growth scores for all tickers
    growth_scores = {}

    for i, row in df_raw.iterrows():
        ticker = row['Ticker']
        info = fast_info(ticker)  # Reuse cached info
        growth_scores[ticker] = get_growth_score(ticker, info)

    # Add growth scores to dataframe
    df_raw['GrowthScore'] = df_raw['Ticker'].map(growth_scores).fillna(0)

    # Calculate sector-relative z-scores
    df_raw['GrowthScore_Sector_Z'] = df_raw.groupby('Sector')['GrowthScore'].transform(
        lambda x: (x - x.mean()) / (x.std() if x.std() > 0 else 1)
    )

    # Convert z-scores to 0-1 range using sigmoid
    sigmoid = lambda z: 1 / (1 + np.exp(-np.clip(z, -2, 2)))
    df_raw['GrowthScore_Norm'] = sigmoid(df_raw['GrowthScore_Sector_Z'])

    return df_raw

def fetch_economic_growth_score():
    """
    Fetches recent US GDP data from FRED and classifies the growth rate
    into one of five buckets to assess the current economic environment:
    Boom (1.0), Strong (0.75), Moderate (0.5), Flat (0.25), Recession (0).
    """
    try:
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={FRED_SERIES_ID}&api_key={FRED_API_KEY}&file_type=json"
        response = requests.get(url)
        data = response.json()
        gdp_values = [float(obs['value']) for obs in data['observations'] if obs['value'] != "."]
        if len(gdp_values) < 4:
            return 0
        growth_rate = (gdp_values[-1] - gdp_values[-2]) / gdp_values[-2]

        # 5-bucket scale
        if growth_rate >= 0.04:
            return 1.0  # Boom
        elif growth_rate >= 0.025:
            return 0.75  # Strong growth
        elif growth_rate >= 0.01:
            return 0.5  # Moderate growth
        elif growth_rate >= -0.01:
            return 0.25  # Stagnation / Flat
        else:
            return 0.0  # Recession
    except:
        return 0


def get_size_score(info):
    """
    Absolute size score using log-scaled market cap with hardcoded thresholds.
    - Market cap: smaller is better (80% of score), using log scale
    - Price: mid-range is best - too low or too high is penalized (20% of score)

    Log-scale thresholds (absolute, not relative to ticker list):
    - $300M (micro-cap floor): score = 1.0 (highest)
    - $500B (mega-cap ceiling): score = 0.0 (lowest)

    Returns a score between 0-1, where higher means smaller/better size characteristics.
    """
    cap = info.get('marketCap')
    price = info.get('regularMarketPrice')

    if cap is None or cap <= 0:
        return 0

    # Market cap component using log scale (smaller is better)
    # Log10 scale: $300M (~8.48) to $500B (~11.7)
    LOG_MCAP_MIN = np.log10(300_000_000)      # $300M - micro-cap floor
    LOG_MCAP_MAX = np.log10(500_000_000_000)  # $500B - mega-cap ceiling

    log_cap = np.log10(cap)

    # Clamp to range and invert (smaller cap = higher score)
    cap_score = (LOG_MCAP_MAX - log_cap) / (LOG_MCAP_MAX - LOG_MCAP_MIN)
    cap_score = max(0, min(1, cap_score))

    # Price component (mid-range is best - too low or too high is problematic)
    price_score = 0
    if price is not None:
        # <$2 = penalized, $5-$50 = optimal, >$200 = discount
        if price < 2:
            price_score = price / 2  # 0-1 range (penalize very low-priced stocks)
        elif price <= 50:
            price_score = 1.0  # optimal price range
        else:
            price_score = max(0, 1 - (price - 50) / 150)  # gradually decrease score for higher prices

    # Weight components (more emphasis on market cap)
    final_score = 0.8 * cap_score + 0.2 * price_score

    return round(final_score, 2)


def get_options_flow_score(ticker, info):
    """
    Calculate institutional options flow score using instant metrics

    Uses current options data snapshot to detect institutional sentiment through:
    1. Put/Call Ratios (OI and Volume based)
    2. Near-the-money concentration
    3. Large position detection

    This instant approach works with historical analysis since it only needs
    current options data, not multi-day momentum tracking.

    Args:
        ticker: Stock symbol
        info: Stock info dict with current price

    Returns:
        float: Score 0-1 (0.5 = neutral, >0.5 = bullish, <0.5 = bearish)
    """
    try:
        # Get current stock price
        current_price = info.get('regularMarketPrice', info.get('currentPrice', None))
        if current_price is None or current_price <= 0:
            return 0.5  # Can't analyze without price

        # Fetch current options data
        try:
            t = Ticker(ticker, session=_global_curl_session)
            options_chain = t.option_chain

            if options_chain is None or not isinstance(options_chain, pd.DataFrame):
                return 0.5  # No options data

            if len(options_chain) < 10:  # Too few contracts to analyze
                return 0.5

        except Exception as e:
            # Silently fail for stocks without options
            return 0.5

        # Parse options data from DataFrame with MultiIndex
        calls_data = []
        puts_data = []

        for idx, contract in options_chain.iterrows():
            # Extract index: (symbol, expiration, optionType)
            if not isinstance(idx, tuple) or len(idx) < 3:
                continue

            symbol, exp_date, option_type = idx[0], idx[1], idx[2]

            # Get contract details
            strike = contract.get('strike', 0)
            oi = contract.get('openInterest', 0)
            volume = contract.get('volume', 0)
            last_price = contract.get('lastPrice', 0)

            # Handle NaN values
            if pd.isna(strike) or pd.isna(oi):
                continue
            if pd.isna(volume):
                volume = 0
            if pd.isna(last_price):
                last_price = 0

            strike = float(strike)
            oi = int(oi)
            volume = int(volume)
            last_price = float(last_price)

            # Skip if no meaningful data
            if oi <= 0 and volume <= 0:
                continue

            # Calculate moneyness (how close to current price)
            moneyness = strike / current_price

            contract_info = {
                'strike': strike,
                'oi': oi,
                'volume': volume,
                'price': last_price,
                'moneyness': moneyness,
                'notional_oi': oi * 100 * last_price,  # Contract value in $
                'notional_vol': volume * 100 * last_price
            }

            # Separate calls and puts
            if option_type == 'calls':
                calls_data.append(contract_info)
            else:  # puts
                puts_data.append(contract_info)

        # Need sufficient data to analyze
        if len(calls_data) < 5 or len(puts_data) < 5:
            return 0.5

        # ===== METRIC 1: Put/Call Ratios =====

        # OI-based P/C ratio
        total_call_oi = sum(c['oi'] for c in calls_data)
        total_put_oi = sum(p['oi'] for p in puts_data)

        pc_ratio_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0

        # Volume-based P/C ratio (more responsive to recent activity)
        total_call_vol = sum(c['volume'] for c in calls_data)
        total_put_vol = sum(p['volume'] for p in puts_data)

        pc_ratio_vol = total_put_vol / total_call_vol if total_call_vol > 0 else 1.0

        # ===== METRIC 2: Near-the-money concentration =====
        # Focus on strikes within ±10% of current price (where institutions trade)

        ntm_calls = [c for c in calls_data if 0.90 <= c['moneyness'] <= 1.10]
        ntm_puts = [p for p in puts_data if 0.90 <= p['moneyness'] <= 1.10]

        ntm_call_oi = sum(c['oi'] for c in ntm_calls)
        ntm_put_oi = sum(p['oi'] for p in ntm_puts)

        ntm_pc_ratio = ntm_put_oi / ntm_call_oi if ntm_call_oi > 0 else pc_ratio_oi

        # ===== METRIC 3: Large position detection =====
        # Identify unusually large positions (potential institutional activity)

        # Calculate average and std for OI
        all_oi = [c['oi'] for c in calls_data] + [p['oi'] for p in puts_data]
        avg_oi = np.mean(all_oi) if all_oi else 0
        std_oi = np.std(all_oi) if len(all_oi) > 1 else 0

        # Flag "large" contracts (>2 std above mean)
        threshold = avg_oi + 2 * std_oi if std_oi > 0 else avg_oi * 2

        large_calls = [c for c in ntm_calls if c['oi'] > threshold]
        large_puts = [p for p in ntm_puts if p['oi'] > threshold]

        large_call_oi = sum(c['oi'] for c in large_calls)
        large_put_oi = sum(p['oi'] for p in large_puts)

        # ===== COMBINE METRICS INTO SCORE =====

        signals = []

        # Signal 1: Overall P/C ratio (weight: 30%)
        # Low P/C (<0.7) = Bullish, High P/C (>1.3) = Bearish
        pc_signal = 1.0 - min(max(pc_ratio_oi - 0.7, 0) / 0.6, 1.0)
        signals.append(('pc_oi', pc_signal, 0.30))

        # Signal 2: Volume P/C ratio (weight: 20%) - recent activity
        pc_vol_signal = 1.0 - min(max(pc_ratio_vol - 0.7, 0) / 0.6, 1.0)
        signals.append(('pc_vol', pc_vol_signal, 0.20))

        # Signal 3: Near-the-money P/C (weight: 30%) - institutional positioning
        ntm_signal = 1.0 - min(max(ntm_pc_ratio - 0.7, 0) / 0.6, 1.0)
        signals.append(('ntm', ntm_signal, 0.30))

        # Signal 4: Large position imbalance (weight: 20%)
        if large_call_oi + large_put_oi > 0:
            large_imbalance = large_call_oi / (large_call_oi + large_put_oi)
            signals.append(('large', large_imbalance, 0.20))
        else:
            signals.append(('large', 0.5, 0.20))  # Neutral if no large positions

        # Weighted average of all signals
        score = sum(signal * weight for _, signal, weight in signals)

        # Ensure score is in valid range
        score = max(0.0, min(1.0, score))

        return round(score, 3)

    except Exception as e:
        # Silently return neutral for any errors (many stocks don't have options)
        return 0.5


def get_liquidity_score(info):
    """
    Multi-dimensional liquidity score considering trading volume,
    spread, market depth, and institutional ownership.
    """
    try:
        # 1. Volume-based liquidity
        avg_vol = info.get('averageVolume10days', 0)
        avg_vol_score = min(avg_vol / 10_000_000, 1.0)

        # 2. Turnover ratio (volume/shares outstanding)
        shares_out = info.get('sharesOutstanding', 1)
        if shares_out > 0:
            turnover = avg_vol / shares_out
            turnover_score = min(turnover * 100, 1.0)  # 1% daily turnover is good
        else:
            turnover_score = 0

        # 3. Market cap component - larger is more liquid
        mcap = info.get('marketCap', 0)
        mcap_score = min(mcap / 10_000_000_000, 1.0)  # $10B+ gets full score

        # 4. Price component - extremely low priced stocks are illiquid
        price = info.get('regularMarketPrice', 0)
        if price < 1:
            price_score = price  # Linear 0-1
        elif price < 5:
            price_score = 0.8 + (price - 1) / 20  # 0.8-1.0 range
        else:
            price_score = 1.0

        # 5. Institutional ownership - higher means more liquid
        inst_own = info.get('institutionsPercentHeld', 0)
        inst_score = min(inst_own / 0.7, 1.0)  # 70%+ gets full score

        # Calculate weighted score
        weights = [0.30, 0.25, 0.20, 0.15, 0.10]  # Prioritize actual trading metrics
        liquidity_score = (
                weights[0] * avg_vol_score +
                weights[1] * turnover_score +
                weights[2] * mcap_score +
                weights[3] * price_score +
                weights[4] * inst_score
        )

        return round(liquidity_score, 2)
    except:
        return 0


def get_carry_score_simple(ticker, info):
    """Continuous dividend 'carry' score ∈ [0,1]."""
    # Check cache first
    if ticker in _carry_data_cache:
        return _carry_data_cache[ticker]

    # Check if we already know this ticker has no dividend data
    if ticker in _carry_no_data_tickers:
        _carry_data_cache[ticker] = 0
        return 0

    try:
        # Get dividend yield
        div_yield = info.get("dividendYield", 0)

        # If no dividend, score is 0
        if div_yield == 0 or div_yield is None:
            _carry_data_cache[ticker] = 0
            return 0

        # Base score is proportional to yield (0-10% → 0.0-1.0)
        # Even tiny dividends get a non-zero score to avoid "constant values" issue
        base_score = min(div_yield / 0.10, 1.0)

        # Add quality bonuses
        quality_bonus = 0

        # 1. Payout Ratio Bonus - lower is better
        pr = info.get("payoutRatio")
        if pr is not None:
            if pr <= 0.3:  # Excellent: <30%
                quality_bonus += 0.20
            elif pr <= 0.5:  # Good: 30-50%
                quality_bonus += 0.15
            elif pr <= 0.7:  # Acceptable: 50-70%
                quality_bonus += 0.05

        # 2. Dividend History
        try:
            h = Ticker(ticker, session=_global_curl_session).history(
                period="10y", interval="1d")

            div_series = pd.Series(dtype=float)

            if h is not None and not h.empty:
                if isinstance(h.index, pd.MultiIndex):
                    h = h.reset_index(level=0, drop=True)

                # Check for both capitalization variants
                div_hist = h.get("dividends", h.get("Dividends", pd.Series(dtype=float)))

                if not div_hist.empty:
                    div_series = div_hist.dropna()

            # Process dividend history if available
            if not div_series.empty:
                # Handle timezone
                if hasattr(div_series.index, 'tz') and div_series.index.tz is not None:
                    div_series.index = div_series.index.tz_localize(None)

                # Group by year
                div_series.index = pd.to_datetime(div_series.index, errors="coerce")
                div_series = div_series[div_series.index.notna()]

                if not div_series.empty:
                    div_series.index = pd.DatetimeIndex(div_series.index)
                    by_year = div_series.groupby(div_series.index.year).sum()
                    by_year = by_year[by_year > 0]

                    # 2a. Dividend Growth Bonus
                    if len(by_year) >= 2:
                        try:
                            years = sorted(by_year.index)
                            first_yr = years[0]
                            last_yr = years[-1]
                            first_div = by_year.loc[first_yr]
                            last_div = by_year.loc[last_yr]
                            span = last_yr - first_yr

                            if span >= 1 and first_div > 0:
                                cagr = (last_div / first_div) ** (1 / max(1, span)) - 1

                                # Growth CAGR bonus
                                if cagr >= 0.10:  # 10%+ CAGR
                                    quality_bonus += 0.20
                                elif cagr >= 0.05:  # 5-10% CAGR
                                    quality_bonus += 0.15
                                elif cagr >= 0.02:  # 2-5% CAGR
                                    quality_bonus += 0.10
                                elif cagr >= 0:  # 0-2% CAGR
                                    quality_bonus += 0.05

                        except (IndexError, ValueError, KeyError):
                            pass

                    # 2b. Dividend Consistency Bonus
                    if not by_year.empty:
                        # Count consecutive years
                        streak = 0
                        for amt in reversed(by_year.values):
                            if amt > 0:
                                streak += 1
                            else:
                                break

                        # Streak bonus
                        if streak >= 10:  # 10+ years
                            quality_bonus += 0.20
                        elif streak >= 5:  # 5-9 years
                            quality_bonus += 0.15
                        elif streak >= 3:  # 3-4 years
                            quality_bonus += 0.10
                        elif streak >= 1:  # 1-2 years
                            quality_bonus += 0.05
        except Exception:
            pass  # Silently continue if dividend history processing fails

        # 3. Final score calculation
        # Base from yield (up to 1.0) + quality bonus (up to 0.6), max total 1.0
        final_score = round(min(base_score + quality_bonus, 1.0), 2)

        # Cache and return
        _carry_data_cache[ticker] = final_score
        return final_score

    except Exception as e:
        # Fallback if any errors
        _carry_data_cache[ticker] = 0
        return 0


def get_obv_slope(df):
    try:
        obv = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume().dropna()
        x = np.arange(len(obv)).reshape(-1, 1)
        y = obv.values.reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        return round(model.coef_[0][0], 6)
    except:
        return 0


def get_adl_slope(df):
    try:
        adl = ta.volume.AccDistIndexIndicator(high=df['High'], low=df['Low'], close=df['Close'],
                                              volume=df['Volume']).acc_dist_index().dropna()
        x = np.arange(len(adl)).reshape(-1, 1)
        y = adl.values.reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        return round(model.coef_[0][0], 6)
    except:
        return 0


# --- helper #1 : 52-week-high "break-out" score ------------------
def _breakout_score(df):
    """
    • 0   if the close is ≤ 70 % of its 52-week high
    • 1   if the close is ≥ 95 % of its 52-week high
    • linear in-between
    """
    close = df["Close"].iloc[-1]
    hi_52w = df["Close"].max()  # full 1-year window already loaded
    pct_of_hi = close / hi_52w if hi_52w else 0
    score = np.clip((pct_of_hi - 0.70) / 0.25, 0, 1)  # maps 0.70→0; 0.95→1
    return round(float(score), 4)


# --- helper #2 : modified ADX trend strength ---------------------
def _trend_score(df):
    """
    Combines           • 6-month linear-regression slope (direction & steepness)
                       • current ADX value
                       • sign from (+DI  –  –DI)
    Then tanh-normalises to (-1, 1)  and rescales to (0, 1)
    """
    # 1) rolling six-month window (~126 trading days)
    lookback = 126
    sub = df[-lookback:]
    y = sub["Close"].values.reshape(-1, 1)
    x = np.arange(len(sub)).reshape(-1, 1)
    slope = LinearRegression().fit(x, y).coef_[0][0]  # $/day
    slope_pct = slope / sub["Close"].iloc[-1]  # unitless

    # 2) ADX & DI
    adx_ind = ta.trend.ADXIndicator(sub['High'], sub['Low'], sub['Close'], window=14)
    adx = adx_ind.adx().iloc[-1] / 100  # 0-1
    pos_di = adx_ind.adx_pos().iloc[-1]
    neg_di = adx_ind.adx_neg().iloc[-1]
    sign = 1 if pos_di > neg_di else -1

    raw = sign * adx * slope_pct * 252  # annualise slope
    score = (np.tanh(raw) + 1) / 2  # maps (-∞,∞) → (0,1)
    return round(float(score), 4)


# --- helper #3 : moving-average "golden cross" -------------------
def _ma_score(df):
    """
    Continuous golden cross indicator based on percentage difference
    between 50-day and 200-day moving averages
    """
    sma_50 = df["Close"].rolling(50).mean().iloc[-1]
    sma_200 = df["Close"].rolling(200).mean().iloc[-1]

    # Calculate percentage difference
    if sma_200 == 0:  # Safety check
        return 0.5

    pct_diff = (sma_50 / sma_200) - 1

    # Convert to 0-1 scale using sigmoid-like transformation
    # Typical range: -10% to +10% difference maps to 0-1
    # 0% difference (equal SMAs) maps to 0.5
    score = 1 / (1 + np.exp(-20 * pct_diff))  # 20 controls steepness

    # Add this line to fix the error:
    return score


# --- master wrapper -----------------------------------------------------------
def get_technical_score(df):
    """
    Three components:
        • Break-out proximity to 52-w high (35%)
        • Modified ADX trend score (40%)
        • 50d vs 200d moving-average crossover (25%)
    Final output ∈ [0, 1].
    """
    # Check if we have enough data for all technical indicators
    if len(df) < 200:
        return 0  # Not enough data for 200-day MA, return 0

    bko = _breakout_score(df) or 0
    trend = _trend_score(df) or 0
    ma = _ma_score(df) or 0.5  # Default to neutral if None

    score = 0.35 * bko + 0.40 * trend + 0.25 * ma
    return round(score, 4)


def _calculate_score_from_marketstack(df):
    """
    Calculate insider score from Marketstack insider data
    Internal helper function - not called directly
    """
    if df.empty:
        return 0

    try:
        # Expected columns from Marketstack insider endpoint
        total_buy = 0
        total_sell = 0
        unique_insiders = set()
        recent_dates = []

        for _, row in df.iterrows():
            # Get transaction details
            trans_type = str(row.get('transaction_type', '')).lower()
            trans_code = str(row.get('transaction_code', '')).upper()
            value = float(row.get('value', row.get('amount', 0)))
            shares = float(row.get('shares', row.get('quantity', 0)))
            insider_name = row.get('insider_name', row.get('filer', 'Unknown'))
            trans_date = row.get('transaction_date', row.get('date', ''))

            # Track unique insiders
            unique_insiders.add(insider_name)

            # Track dates for recency
            if trans_date:
                recent_dates.append(pd.to_datetime(trans_date))

            # Determine if buy or sell
            buy_keywords = ['buy', 'purchase', 'acquire', 'exercise', 'grant', 'P']
            sell_keywords = ['sell', 'sale', 'dispose', 'S']

            is_buy = any(kw in trans_type for kw in buy_keywords) or trans_code == 'P'
            is_sell = any(kw in trans_type for kw in sell_keywords) or trans_code == 'S'

            # Use value if available, otherwise estimate from shares
            if value > 0:
                trans_value = value
            elif shares > 0 and 'price' in row:
                trans_value = shares * float(row.get('price', 0))
            else:
                trans_value = 0

            if is_buy:
                total_buy += abs(trans_value)
            elif is_sell:
                total_sell += abs(trans_value)

        # Calculate metrics
        total_value = total_buy + total_sell

        if total_value == 0:
            return 0

        # Buy ratio
        buy_ratio = total_buy / total_value

        # Magnitude factor
        magnitude = min(np.log1p(total_value) / np.log1p(10_000_000), 1.0)

        # Breadth factor (unique insiders)
        breadth = min(len(unique_insiders) / 5, 1.0)

        # NOTE: Recency factor removed for historical analysis compatibility
        # For historical correlation analysis, we treat all transactions equally
        # rather than applying datetime.now() based recency weighting
        recency = 1.0

        # Calculate final score
        raw_score = (buy_ratio - 0.5) * 2  # Convert to -1 to +1
        weighted_score = raw_score * (0.5 + 0.2 * magnitude + 0.2 * breadth + 0.1 * recency)

        return round(max(-1, min(1, weighted_score)), 2)

    except Exception:
        return 0


def _calculate_score_from_sec_filings(df):
    """
    Calculate insider score from SEC Form 4 filings
    Internal helper function - not called directly
    """
    if df.empty:
        return 0

    try:
        # Parse Form 4 data
        total_buy = 0
        total_sell = 0
        unique_insiders = set()

        for _, row in df.iterrows():
            # Extract from Form 4 filing data
            filing_text = str(row.get('filing_text', row.get('content', ''))).lower()
            reporting_owner = row.get('reporting_owner', row.get('insider', 'Unknown'))

            unique_insiders.add(reporting_owner)

            # Simple heuristic: look for acquisition/disposition keywords
            if 'acquisition' in filing_text or 'purchase' in filing_text:
                # Estimate value (you may need to parse more carefully)
                value = float(row.get('transaction_value', 100000))  # Default estimate
                total_buy += abs(value)
            elif 'disposition' in filing_text or 'sale' in filing_text:
                value = float(row.get('transaction_value', 100000))  # Default estimate
                total_sell += abs(value)

        # Calculate metrics
        total_value = total_buy + total_sell

        if total_value == 0:
            return 0

        # Simplified scoring for SEC filings
        buy_ratio = total_buy / total_value
        diversity = min(len(unique_insiders) / 3, 1.0)  # Lower threshold for SEC data

        # Calculate score
        raw_score = (buy_ratio - 0.5) * 2
        weighted_score = raw_score * (0.6 + 0.4 * diversity)

        return round(max(-1, min(1, weighted_score)), 2)

    except Exception:
        return 0


def get_insider_score_simple(ticker):
    """
    Enhanced insider score that tries Marketstack SEC data first, then yahooquery

    This is a complete drop-in replacement for the original function.
    It maintains backward compatibility while adding Marketstack support.
    """

    # Check if we already know this ticker has no insider data
    if ticker in _insider_no_data_tickers:
        return 0

    # Check cache first
    if ticker in _insider_data_cache:
        return _insider_data_cache[ticker]

    # ========================================================================
    # STEP 1: Try Marketstack SEC data first (new capability)
    # ========================================================================
    if MARKETSTACK_API_KEY:
        try:
            # Prepare date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=INSIDER_LOOKBACK_DAYS)

            # Prepare API parameters
            params = {
                'access_key': MARKETSTACK_API_KEY,
                'symbols': ticker,
                'date_from': start_date.strftime('%Y-%m-%d'),
                'date_to': end_date.strftime('%Y-%m-%d'),
                'limit': 100
            }

            # Try insider transactions endpoint first
            response = requests.get(f"{MARKETSTACK_BASE_URL}/insider", params=params, timeout=5)

            if response.status_code == 200:
                data = response.json()
                if 'data' in data and data['data']:
                    # Process Marketstack insider data
                    df = pd.DataFrame(data['data'])

                    # Calculate score from Marketstack data
                    score = _calculate_score_from_marketstack(df)

                    if score != 0:  # If we got valid data
                        _insider_data_cache[ticker] = score
                        print(f"   {ticker}: Got insider score from Marketstack SEC: {score}")
                        return score

            # If insider endpoint didn't work, try SEC filings endpoint
            response = requests.get(f"{MARKETSTACK_BASE_URL}/sec/filings", params=params, timeout=5)

            if response.status_code == 200:
                data = response.json()
                if 'data' in data and data['data']:
                    df = pd.DataFrame(data['data'])

                    # Filter for Form 4 (insider transactions)
                    if 'form_type' in df.columns:
                        form4_df = df[df['form_type'].str.contains('4', na=False)]

                        if not form4_df.empty:
                            score = _calculate_score_from_sec_filings(form4_df)

                            if score != 0:
                                _insider_data_cache[ticker] = score
                                print(f"   {ticker}: Got insider score from SEC filings: {score}")
                                return score

        except Exception as e:
            # Silently continue to yahooquery fallback
            pass

    # ========================================================================
    # STEP 2: Fallback to yahooquery (your original method)
    # ========================================================================
    try:
        from yahooquery import Ticker

        # Assuming _global_curl_session is defined in your script
        t = Ticker(ticker, session=_global_curl_session)
        df = t.insider_transactions

        # Handle various return types from yahooquery
        if df is None:
            _insider_no_data_tickers.add(ticker)
            _insider_data_cache[ticker] = 0
            return 0

        if isinstance(df, dict):
            if ticker not in df:
                _insider_no_data_tickers.add(ticker)
                _insider_data_cache[ticker] = 0
                return 0
            df = df[ticker]

        if df.empty:
            _insider_data_cache[ticker] = 0
            return 0

        # Filter to look-back window
        cutoff = (datetime.now() - timedelta(days=INSIDER_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        df = df[df["startDate"] >= cutoff]
        if df.empty:
            _insider_data_cache[ticker] = 0
            return 0

        # Calculate net buying/selling values (your original logic)
        df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
        txt = df["transactionText"].str.lower()

        buys = df[txt.str.contains("buy|purchase")]["value"].sum()
        sells = df[txt.str.contains("sell|sale")]["value"].sum()
        total = buys + sells
        net = buys - sells

        if total == 0:
            _insider_data_cache[ticker] = 0
            return 0

        # Calculate ratio of buying to total activity
        buy_ratio = buys / total

        # Calculate magnitude factor (larger transactions have more significance)
        # Log scale to avoid domination by very large transactions
        magnitude = math.log(total + 1) / math.log(10000000)  # 0 to 1 range
        magnitude = min(magnitude, 1.0)

        # Calculate breadth factor (more insiders = stronger signal)
        unique_insiders = df['filerName'].nunique()
        breadth = min(unique_insiders / 5, 1.0)  # Cap at 5 insiders

        # Calculate recency factor (more recent transactions = stronger signal)
        days_ago = []
        for date in df['startDate']:
            try:
                dt = pd.Timestamp(date)
                delta = (datetime.now() - dt).days
                days_ago.append(delta)
            except:
                days_ago.append(INSIDER_LOOKBACK_DAYS)

        avg_recency = 1 - (min(np.mean(days_ago), INSIDER_LOOKBACK_DAYS) / INSIDER_LOOKBACK_DAYS)

        # Combine factors into final score
        raw_score = (buy_ratio - 0.5) * 2  # -1 to +1 range
        weighted_score = raw_score * (0.5 + 0.2 * magnitude + 0.2 * breadth + 0.1 * avg_recency)

        # Scale to -1 to +1 and round
        final_score = round(max(-1, min(1, weighted_score)), 2)

        # Cache and return result
        _insider_data_cache[ticker] = final_score
        return final_score

    except Exception:
        # Silently handle any errors
        _insider_no_data_tickers.add(ticker)
        _insider_data_cache[ticker] = 0
        return 0


def _price_mom(df):
    """
    Calculate 12-month price momentum with fallback options
    Prefers 252-day lookback but can adjust if less data available
    """
    # Ideal: 252 trading days (1 year)
    if len(df) >= 252:
        ret = (df["Close"].iloc[-1] / df["Close"].iloc[-252]) - 1
        return ret

    # Fallback: Use as much data as available (minimum 180 days)
    elif len(df) >= 180:
        # Use all available data, but note this is not ideal
        print(f"   Using {len(df)} days for momentum (less than 252)")
        ret = (df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1
        return ret

    # Not enough data
    else:
        return None

# --- 1-b  earnings-momentum (EPS surprises & upward revisions) ---------------
def _earnings_mom(ticker):
    """
    • YahooQuery – earnings_trend stores 'earningsEstimate' with
      fields: avg, low, high, yearAgoEPS, growth.
    • We take current-qtr EPS estimate change over last 90 days plus
      trailing-quarter surprise.
    Returns a raw % effect; later z-scored.
    """
    try:
        t = Ticker(ticker, session=_global_curl_session)
        tr = t.earnings_trend.get(ticker, [])
        if not tr:
            return None

        # Current-quarter row has period = '0Q'
        cq = next(x for x in tr if x.get('period') == '0Q')
        rev = cq.get('earningsEstimate', {})
        rev_chg = (rev.get('avg') - rev.get('avg90daysAgo', rev.get('avg'))) / abs(rev.get('avg90daysAgo', 1))

        # Surprise% is in trailing-12M history
        hist = t.earnings.get(ticker, {}).get('earningsChart', {}).get('quarterly', [])
        surprise_pct = np.mean([q['surprisePercent'] for q in hist[-4:] if q.get('surprisePercent') is not None])

        return 0.5 * rev_chg + 0.5 * surprise_pct / 100  # ≈ weighted %
    except Exception:
        return None


def get_momentum_score(ticker, hist):
    """
    Fixed momentum score calculation for historical data
    Ensures consistent 252-day lookback for price momentum
    """
    # Price momentum - use fixed 252-day lookback
    pm = None
    if len(hist) >= 252:
        pm = (hist["Close"].iloc[-1] / hist["Close"].iloc[-252]) - 1
    elif len(hist) >= 180:
        # Fallback if we don't have full year
        # But flag this as it will cause differences
        print(f"     {ticker}: Only {len(hist)} days for momentum")
        pm = (hist["Close"].iloc[-1] / hist["Close"].iloc[0]) - 1

    # Earnings momentum
    em = _earnings_mom(ticker)

    # Convert None → np.nan for z-score calculations
    return (pm if pm is not None else np.nan,
            em if em is not None else np.nan)


def absolute_momentum_score(raw_momentum):
    """
    Convert raw price momentum (252-day return) to 0-1 score using fixed percentile thresholds.

    This is an absolute scoring function - independent of other tickers in the list.

    Thresholds based on typical market behavior:
    - Strong negative: < -20% → score ≈ 0.1-0.3
    - Weak negative: -20% to 0% → score ≈ 0.3-0.5
    - Weak positive: 0% to +30% → score ≈ 0.5-0.7
    - Strong positive: +30% to +100% → score ≈ 0.7-0.9
    - Exceptional: > +100% → score ≈ 0.9-1.0

    Args:
        raw_momentum: Float representing the 252-day return (e.g., 0.25 = 25% gain)

    Returns:
        Float between 0 and 1
    """
    if raw_momentum is None or np.isnan(raw_momentum):
        return 0.5  # Neutral score for missing data

    if raw_momentum < -0.20:  # < -20%
        # Strong negative momentum: map to 0.1-0.3
        # Cap at -50% for extreme losses
        clamped = max(raw_momentum, -0.50)
        return 0.1 + (clamped + 0.50) / 0.30 * 0.2
    elif raw_momentum < 0:  # -20% to 0%
        # Weak negative: linear interpolation from 0.3 to 0.5
        return 0.3 + (raw_momentum + 0.20) / 0.20 * 0.2
    elif raw_momentum < 0.30:  # 0% to +30%
        # Weak positive: linear interpolation from 0.5 to 0.7
        return 0.5 + raw_momentum / 0.30 * 0.2
    elif raw_momentum < 1.00:  # +30% to +100%
        # Strong positive: linear interpolation from 0.7 to 0.9
        return 0.7 + (raw_momentum - 0.30) / 0.70 * 0.2
    else:  # > +100%
        # Exceptional: map to 0.9-1.0, diminishing returns
        # Cap at +200% for extreme gains
        clamped = min(raw_momentum, 2.00)
        return 0.9 + (clamped - 1.00) / 1.00 * 0.1


def absolute_earnings_momentum_score(earnings_momentum):
    """
    Convert raw earnings momentum to 0-1 score using fixed percentile thresholds.

    This is an absolute scoring function - independent of other tickers in the list.

    Thresholds based on typical earnings growth patterns:
    - Strong decline: < -15% → score ≈ 0.1-0.3
    - Weak decline: -15% to 0% → score ≈ 0.3-0.5
    - Weak growth: 0% to +20% → score ≈ 0.5-0.7
    - Strong growth: +20% to +50% → score ≈ 0.7-0.9
    - Exceptional: > +50% → score ≈ 0.9-1.0

    Args:
        earnings_momentum: Float representing earnings growth rate

    Returns:
        Float between 0 and 1
    """
    if earnings_momentum is None or np.isnan(earnings_momentum):
        return 0.5  # Neutral score for missing data

    if earnings_momentum < -0.15:  # < -15%
        # Strong decline: map to 0.1-0.3
        # Cap at -40% for extreme declines
        clamped = max(earnings_momentum, -0.40)
        return 0.1 + (clamped + 0.40) / 0.25 * 0.2
    elif earnings_momentum < 0:  # -15% to 0%
        # Weak decline: linear interpolation from 0.3 to 0.5
        return 0.3 + (earnings_momentum + 0.15) / 0.15 * 0.2
    elif earnings_momentum < 0.20:  # 0% to +20%
        # Weak growth: linear interpolation from 0.5 to 0.7
        return 0.5 + earnings_momentum / 0.20 * 0.2
    elif earnings_momentum < 0.50:  # +20% to +50%
        # Strong growth: linear interpolation from 0.7 to 0.9
        return 0.7 + (earnings_momentum - 0.20) / 0.30 * 0.2
    else:  # > +50%
        # Exceptional: map to 0.9-1.0, diminishing returns
        # Cap at +100% for extreme growth
        clamped = min(earnings_momentum, 1.00)
        return 0.9 + (clamped - 0.50) / 0.50 * 0.1

def get_30d_std(df):
    """
    Returns the standard deviation of daily returns over the past 30 trading days.
    Used to penalize excessively volatile stocks.
    """
    try:
        return df['Close'].pct_change().dropna()[-30:].std()
    except:
        return None


def _cvar95(r):
    """
    Parametric ES at 95 % assuming normality (fast & no exotic libs).
    """
    mu, sigma = np.mean(r), np.std(r, ddof=1)
    var95 = mu + sigma * 1.65  # 5th-percentile
    cvar = mu - sigma * 2.06  # expected shortfall (ES) ≈ μ – 2.06σ
    return abs(min(cvar, 0))


def _cdar(df):
    """
    Conditional drawdown at risk (CDaR) over 1 year (last 252 bars).
    """
    series = df['Close'].iloc[-252:]
    dd = 1 - series / series.cummax()
    # drawdowns' 95-th percentile → average of worst drawdowns
    cutoff = np.quantile(dd, 0.95)
    return dd[dd >= cutoff].mean()

def calculate_recovery_score(df):
    """
    Calculate how quickly a stock recovers from drawdowns.
    Higher score means faster recovery (better).
    """
    # Need enough price history
    if len(df) < 126:  # At least 6 months
        return 0.5  # Neutral score if not enough data

    close = df['Close']
    # Calculate running maximum (high watermark)
    running_max = close.cummax()
    # Calculate drawdowns as percentage from peak
    drawdowns = 1 - close / running_max

    # Find significant drawdowns (> 5%)
    significant_dd = drawdowns[drawdowns > 0.05].index

    if len(significant_dd) == 0:
        return 0.8  # Good score if no significant drawdowns

    # For each significant drawdown, calculate days to recover
    recovery_times = []

    for dd_date in significant_dd:
        # Skip drawdowns that are too recent to have recovered
        if dd_date > df.index[-20]:
            continue

        dd_value = drawdowns.loc[dd_date]
        peak_value = running_max.loc[dd_date]

        # Find recovery date (when price returns to within 1% of peak)
        recovery_threshold = peak_value * 0.99
        future_prices = close[close.index > dd_date]

        recovered_dates = future_prices[future_prices >= recovery_threshold].index

        if len(recovered_dates) > 0:
            # Get first date when price recovered
            recovery_date = recovered_dates[0]

            # Calculate trading days to recovery
            dd_idx = df.index.get_loc(dd_date)
            recovery_idx = df.index.get_loc(recovery_date)
            days_to_recover = recovery_idx - dd_idx

            # Store tuple of (drawdown_amount, recovery_time)
            recovery_times.append((dd_value, days_to_recover))

    if not recovery_times:
        return 0.3  # Penalize if drawdowns haven't recovered

    # Calculate weighted average recovery time (weighted by drawdown size)
    total_weight = sum(dd for dd, _ in recovery_times)
    weighted_time = sum(dd * time for dd, time in recovery_times) / total_weight

    # Normalize: 10 days (good) → 1.0, 60+ days (bad) → 0.0
    recovery_score = max(0, min(1, 1 - (weighted_time - 10) / 50))

    return recovery_score

def calculate_beta(stock_returns, market_returns):
    """
    Calculate beta coefficient (market sensitivity) for a stock.
    """
    # Make sure we have matching data
    if len(stock_returns) != len(market_returns):
        return 1.0  # Default to market beta

    # Filter out NaN values
    valid_data = ~(np.isnan(stock_returns) | np.isnan(market_returns))
    stock_returns_clean = stock_returns[valid_data]
    market_returns_clean = market_returns[valid_data]

    if len(stock_returns_clean) < 30:  # Need sufficient data points
        return 1.0

    # Calculate beta using covariance formula
    covariance = np.cov(stock_returns_clean, market_returns_clean)[0, 1]
    market_variance = np.var(market_returns_clean)

    if market_variance == 0:
        return 1.0

    beta = covariance / market_variance

    return beta

def get_risk_penalty(df, market_df=None):
    """
    Multi-dimensional risk assessment with continuous scoring and market adaptation.
    Combines volatility, drawdowns, tail risk, and market sensitivity.
    """
    # Base calculations
    ret = df['Close'].pct_change().dropna()

    # 1. Traditional volatility (annualized)
    vol = np.std(ret, ddof=1) * np.sqrt(252)
    vol_score = 1 - np.tanh(vol * 2)  # Smoothly maps high vol toward 0

    # 2. Downside risk measures
    cvar = _cvar95(ret)
    cdar = _cdar(df)
    tail_score = 1 - np.tanh(max(cvar, cdar) * 4)  # More sensitive to tail events

    # 3. Recovery speed - how quickly price recovers after drawdowns
    # Calculate average recovery time from 5%+ drawdowns
    recovery_score = calculate_recovery_score(df)

    # 4. Market sensitivity (beta) if market data available
    beta_score = 1.0
    if market_df is not None:
        market_ret = market_df['Close'].pct_change().dropna()
        # Align dates
        common_dates = ret.index.intersection(market_ret.index)
        if len(common_dates) > 60:  # Need sufficient data
            stock_ret_aligned = ret.loc[common_dates]
            market_ret_aligned = market_ret.loc[common_dates]
            beta = calculate_beta(stock_ret_aligned, market_ret_aligned)
            # Lower beta is better in this model
            beta_score = 1 - np.tanh(max(0, beta - 1) * 2)

    # Weight the components based on market conditions (could be dynamic)
    # During high volatility, tail_score matters more
    # During normal times, recovery_score matters more
    weights = [0.3, 0.3, 0.2, 0.2]  # vol, tail, recovery, beta

    # Calculate weighted score and convert to penalty
    risk_score = (
            weights[0] * vol_score +
            weights[1] * tail_score +
            weights[2] * recovery_score +
            weights[3] * beta_score
    )

    # Convert to penalty (negative score)
    penalty = -(1 - risk_score)

    return round(penalty, 3)

def get_stability_score(df, market_df=None):
    """
    Stability score based on low volatility and risk characteristics.
    Higher score means more stable/less risky (better).
    Returns a score between 0-1, where 1 is most stable.
    """
    # Base calculations
    ret = df['Close'].pct_change().dropna()

    # 1. Traditional volatility (annualized)
    vol = np.std(ret, ddof=1) * np.sqrt(252)
    vol_score = 1 - np.tanh(vol * 2)  # Already maps low vol to high score

    # 2. Downside risk measures
    cvar = _cvar95(ret)
    cdar = _cdar(df)
    tail_score = 1 - np.tanh(max(cvar, cdar) * 4)  # Low tail risk = high score

    # 3. Recovery speed - how quickly price recovers after drawdowns
    recovery_score = calculate_recovery_score(df)  # Already returns high score for quick recovery

    # 4. Market sensitivity (beta) if market data available
    beta_score = 1.0
    if market_df is not None:
        market_ret = market_df['Close'].pct_change().dropna()
        # Align dates
        common_dates = ret.index.intersection(market_ret.index)
        if len(common_dates) > 60:  # Need sufficient data
            stock_ret_aligned = ret.loc[common_dates]
            market_ret_aligned = market_ret.loc[common_dates]
            beta = calculate_beta(stock_ret_aligned, market_ret_aligned)
            # Lower beta is better (more stable)
            beta_score = 1 - np.tanh(max(0, beta - 1) * 2)

    # Weight the components
    weights = [0.3, 0.3, 0.2, 0.2]  # vol, tail, recovery, beta

    # Calculate weighted stability score (already 0-1, higher is better)
    stability_score = (
            weights[0] * vol_score +
            weights[1] * tail_score +
            weights[2] * recovery_score +
            weights[3] * beta_score
    )

    return round(stability_score, 3)  # Return positive score


def main(regime="Steady_Growth", mode="daily", target_date=None, progress_callback=None):
    """
    Main scoring function with support for daily and historical modes

    Args:
        regime: Regime name for scoring weights (used in daily mode)
        mode: "daily" for current data, "historical" for past data
        target_date: Date for historical analysis (used in historical mode)
        progress_callback: Function to call with progress updates
    """

    if mode == "historical":
        if target_date is None:
            raise ValueError("target_date required for historical mode")
        return run_historical_scoring(target_date, progress_callback)
    else:
        # Original daily mode logic
        return run_daily_scoring(regime, progress_callback)

# --- Main Logic ---
def run_daily_scoring(regime="Steady_Growth", progress_callback=None):
    """
    Original main scoring function with progress tracking
    This is the ORIGINAL version from stock_Screener_MultiFactor_24.py

    Args:
        regime: Regime name for scoring weights
        progress_callback: Function to call with progress updates (type, value)
    """
    start_time = time.time()

    # Set up progress tracking
    if progress_callback:
        set_progress_callback(progress_callback)

    # Get weights for the specified regime
    global WEIGHTS
    WEIGHTS = get_regime_weights(regime)

    print(f" Running multifactor scoring for regime: {regime}")
    print(f" Using weights: {WEIGHTS}")

    if _progress_tracker:
        _progress_tracker.callback('status', f"Loading tickers for {regime} regime...")
        _progress_tracker.callback('progress', 5)

    # -------- 0.  load tickers -------------------------------------------------
    try:
        with open(TICKER_FILE, "r") as f:
            tickers = [ln.strip().replace("$", "") for ln in f if ln.strip()]
    except FileNotFoundError:
        error_msg = f" Ticker file '{TICKER_FILE}' not found!"
        print(error_msg)
        if _progress_tracker:
            _progress_tracker.callback('error', error_msg)
        return

    # Initialize progress tracker with total tickers
    if _progress_tracker:
        _progress_tracker.total_tickers = len(tickers)

    # -------- 1-a  pre-scan: avgVol for liquidity -----------------------------
    avg_vol = {}
    for tk in tickers:
        try:
            avg_vol[tk] = fast_info(tk).get("averageVolume10days", 0)
        except Exception:
            avg_vol[tk] = 0
    vol_min, vol_max = min(avg_vol.values()), max(avg_vol.values())

    # -------- 1-b. main data harvest ------------------------------------------
    if _progress_tracker:
        _progress_tracker.callback('status', "Downloading data and calculating factor scores...")
        _progress_tracker.callback('progress', 10)

    raw = []  # store everything, score later
    pm_cache, em_cache, mcaps = [], [], []
    value_metrics_by_ticker = {}
    skipped_tickers = []
    processed_tickers = []

    for i, tk in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] scraping {tk} …")

        # Update progress for each ticker
        if _progress_tracker:
            _progress_tracker.update_progress(tk, "processing")

        try:
            info = safe_get_info(tk)
            print(f"   Info fetched for {tk}: MarketCap={info.get('marketCap', 'N/A')}")

            # Skip if basic data missing
            if not info or not info.get("marketCap"):
                print(f"   SKIPPING {tk}: No market cap data")
                skipped_tickers.append((tk, "No market cap"))
                continue

            hist = yahooquery_hist(tk, years=2)

            if hist.empty or len(hist) < 200:
                if hist.empty:
                    print(f"   SKIPPING {tk}: Historical data is empty")
                else:
                    print(f"   SKIPPING {tk}: Insufficient data points ({len(hist)} < 200)")
                skipped_tickers.append((tk, f"Insufficient data ({len(hist)} days)"))
                continue

            # Calculate Factor Scores with regime-specific thresholds
            value, quality, financial_health, roic, value_dict = get_fundamentals(tk, regime)
            value_metrics_by_ticker[tk] = value_dict

            tech = get_technical_score(hist)
            insider = get_insider_score_simple(tk)

            pm, em = get_momentum_score(tk, hist)
            pm_cache.append(pm)
            em_cache.append(em)

            stability = get_stability_score(hist)

            mcap = info.get("marketCap")
            mcaps.append(mcap)
            sector, industry = sector_industry(info)

            company_name, country = get_company_info(info)
            options_flow = get_options_flow_score(tk, info)
            carry = get_carry_score_simple(tk, info)

            liq = 0
            if vol_max != vol_min:
                liq = round((avg_vol[tk] - vol_min) / (vol_max - vol_min), 2)

            raw.append(dict(
                Ticker=tk, Value=value, Quality=quality,
                FinancialHealth=financial_health,
                Technical=tech, Insider=insider, PriceMom=pm, EarnMom=em,
                Stability=stability,
                MarketCap=mcap, OptionsFlow=options_flow,
                Liquidity=liq, Carry=carry, Sector=sector, Industry=industry,
                Growth=get_growth_score(tk, info, regime),
                ROIC=roic,
                CompanyName=company_name, Country=country
            ))

            print(f"   {tk} successfully added to results")
            processed_tickers.append(tk)

            # Update progress tracker for completed ticker
            if _progress_tracker:
                _progress_tracker.update_progress(tk, "completed")

        except Exception as e:
            print(f" {tk}: {e}")
            if _progress_tracker:
                _progress_tracker.update_progress(tk, "completed")

    if not raw:
        error_msg = " No usable tickers"
        print(error_msg)
        if _progress_tracker:
            _progress_tracker.callback('error', error_msg)
        return

    # -------- Final processing phase ------------------------------
    if _progress_tracker:
        _progress_tracker.callback('progress', 90)
        _progress_tracker.callback('status', "Finalizing scores and rankings...")

    # -------- 2.  cross-sectional normalisations ------------------------------
    # Convert to DataFrame
    df_raw = pd.DataFrame(raw)

    # Use raw growth score directly (already uses absolute thresholds)
    if 'Growth' in df_raw.columns:
        df_raw['GrowthScore_Norm'] = df_raw['Growth']
    else:
        df_raw['GrowthScore_Norm'] = 0

    # Use raw value score directly (already uses absolute thresholds)
    # No industry-relative normalization needed
    df_raw['ValueScore'] = df_raw['Value']

    # -------- 3.  final scoring ticker-by-ticker ------------------------------
    results = []
    for k, rec in df_raw.iterrows():
        # Use absolute momentum scoring (no z-score normalization)
        pm_raw = rec['PriceMom']
        em_raw = rec['EarnMom']

        pm_score = absolute_momentum_score(pm_raw)
        em_score = absolute_earnings_momentum_score(em_raw)

        momentum = round(0.6 * pm_score + 0.4 * em_score, 3)

        size = get_size_score({'marketCap': rec['MarketCap']})

        total = (
                rec['ValueScore'] * WEIGHTS['value'] +
                rec['Quality'] * WEIGHTS['quality'] +
                rec['FinancialHealth'] * WEIGHTS['financial_health'] +
                rec['Technical'] * WEIGHTS['technical'] +
                rec['Insider'] * WEIGHTS['insider'] +
                momentum * WEIGHTS['momentum'] +
                rec['Stability'] * WEIGHTS['stability'] +
                size * WEIGHTS['size'] +
                rec['OptionsFlow'] * WEIGHTS['options_flow'] +
                rec['Liquidity'] * WEIGHTS['liquidity'] +
                rec['Carry'] * WEIGHTS['carry'] +
                rec.get('GrowthScore_Norm', 0) * WEIGHTS['growth']
        )

        results.append({
            "Ticker": rec['Ticker'],
            "CompanyName": rec['CompanyName'],
            "Country": rec['Country'],
            "Score": round(total, 2),
            "Value": round(rec['ValueScore'], 3),
            "Quality": rec['Quality'],
            "FinancialHealth": round(rec['FinancialHealth'], 3),
            "Technical": rec['Technical'],
            "Insider": rec['Insider'],
            "Momentum": momentum,
            "Stability": rec['Stability'],
            "Size": size,
            "OptionsFlow": rec['OptionsFlow'],
            "Liquidity": rec['Liquidity'],
            "Carry": rec['Carry'],
            "Growth": round(rec.get('GrowthScore_Norm', 0), 3),
            "Sector": rec['Sector'],
            "Industry": rec['Industry']
        })

    # -------- 4.  output ------------------------------------------------------
    if _progress_tracker:
        _progress_tracker.callback('progress', 95)
        _progress_tracker.callback('status', "Generating output file...")

    df = pd.DataFrame(results).sort_values("Score", ascending=False)

    # Get output directory for the regime
    output_dir = get_output_directory(regime)
    # today = datetime.now().strftime("%m%d")
    today = datetime.now().strftime("%Y%m%d")  # Changed from "%m%d" to "%Y%m%d"
    fname = f"top_ranked_stocks_{regime}_{today}.csv"
    output_path = output_dir / fname

    # Save the file
    df.to_csv(output_path, index=False)
    print(f" Saved {fname} to {output_dir}")
    print(f" Top {TOP_N} stocks for {regime} regime:")
    print(df.head(TOP_N))
    print(f" Finished in {round(time.time() - start_time, 1)} seconds")

    if _progress_tracker:
        _progress_tracker.callback('progress', 100)
        _progress_tracker.callback('status', f"Complete! Analyzed {len(processed_tickers)} stocks.")

    return output_path, df

# Add this function to be called from the UI
def run_scoring_for_regime(regime_name, progress_callback=None):
    """
    Wrapper function to run scoring for a specific regime
    Used by the UI thread

    Args:
        regime_name: Name of the regime (e.g., "Steady_Growth")
        progress_callback: Optional callback for progress updates

    Returns:
        dict with success status and results
    """
    try:
        # Clean regime name
        regime_clean = regime_name.replace(" ", "_").replace("/", "_")

        # Run daily scoring with the specified regime
        result = run_daily_scoring(regime_clean, progress_callback)

        return {
            'success': True,
            'regime': regime_name,
            'results': result
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

if __name__ == "__main__":
    main()