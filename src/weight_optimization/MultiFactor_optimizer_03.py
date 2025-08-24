# V02: Change volatility penalty to volatility score. Low volatility gets higher score.
# New V01: Add regime-based period selection
# V17: Modified from V14, 3yr analysis, reorganized factors (Value, Quality, Growth)
# V14: Use Marketstack data
# V12: Update the Value factor
# V11: Revise scoring functions to make them continuous
"""
Correlation Analysis Script for Multi-Factor Stock Screener
- Uses enhanced growth factor from V19
- Performs single-weight-set evaluation and correlation analysis
- Analyzes relationship between factor scores and returns
- Extended to 3-year correlation period with exponential weighting
- Improved error handling for small ticker samples
"""

from curl_cffi.requests import Session as CurlSession
import requests
import yahooquery.utils as yq_utils
from Functions import PerformanceEvaluator  # Import evaluation framework
import types
from scipy import stats
import pandas as pd
import numpy as np
import ta
import math
import time
from datetime import datetime, timedelta
from yahooquery import Ticker
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore, pearsonr, spearmanr
import random
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import os
import re


# Add this monkey-patching code BEFORE any yahooquery imports
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
yq_utils.requests = curl_module  # looks like real "requests" now

# Monkey-patch yahooquery's requests module
yq_utils._COOKIE = {"B": "dummy"}  # Bypass cookie checks
yq_utils._CRUMB = "dummy"  # Bypass crumb validation
yq_utils.get_crumb = lambda *_: "dummy"  # Disable crumb fetching

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="yahooquery")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
pd.options.mode.chained_assignment = None  # disable copy-view warning

# --- Constants and Parameters ---
TICKER_FILE = r"./Config/test_tickers_stocks.txt"
INSIDER_LOOKBACK_DAYS = 90
TOP_N = 20
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# Add Marketstack configuration
MARKETSTACK_API_KEY = "476419cceb4330259e5a1267533xxxxx"  # Replace with your actual API key
MARKETSTACK_BASE_URL = "https://api.marketstack.com/v2"

# Update correlation period to 3 years
CORRELATION_YEARS = 3  # Changed from 1 year to 3 years
# EXPONENTIAL_DECAY_HALFLIFE = 126  # Half-life in trading days (about 6 months)

# Add after the existing constants
REGIME_PERIODS_FILE = r"./Analysis/regime_periods.csv"
SELECTED_REGIME = r"Crisis/Bear"  # "Strong Bull", "Steady Growth", or "Crisis/Bear"
MIN_CRISIS_DAYS = 14  # Minimum days to consider a real crisis

# Scoring Weights - Use V19 updated weights
WEIGHTS = {
    "momentum": 74.85,      # Slight increase from 70.7
    "size": 18.7,          # Rounded up from 17.6
    "financial_health": 12.15,  # Reduced to minimize drag
    "credit": 0.0,         # Eliminated (negative correlation)
    "insider": 4.7,        # Rounded up from 4.4
    "growth": 4.7,         # Reduced from 8.8 (negative correlation)
    "quality": 0.0,        # Keep at zero
    "liquidity": 4.7,      # Add for positive correlation
    "value": 0.0,          # Keep at zero
    "technical": -5.0,     # Unchanged
    "carry": -4.7,         # Rounded from -4.4
    "stability": -10.1,    # Rounded from -10.3
}

# Update FUND_THRESHOLDS to reorganize metrics
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

VALUE_METRIC_WEIGHTS = {
    'earnings_yield': 0.25,    # Most direct value measure
    'fcf_yield': 0.25,         # Cash generation vs price
    'ev_to_ebitda': 0.20,      # Enterprise value metric
    'pb': 0.15,                # Book value
    'ev_to_sales': 0.15       # Sales multiple
}

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
period = '5y'
# dates = pd.date_range(start="2020-01-01", end=datetime.today(), freq='MS')
# Ensure all date objects are timezone-naive
dates = pd.date_range(
    start=pd.Timestamp("2020-01-01").tz_localize(None),
    end=pd.Timestamp(datetime.today()).tz_localize(None),
    freq='MS'
)

# --- Global caches ---
_info_cache = {}
_insider_data_cache = {}
_insider_no_data_tickers = set()
_carry_data_cache = {}
_carry_no_data_tickers = set()


# def load_regime_periods(regime_name):
#     """Load regime periods from CSV for a specific regime"""
#     regime_df = pd.read_csv(REGIME_PERIODS_FILE)
#     regime_periods = regime_df[regime_df['regime_name'] == regime_name].copy()
#
#     # Convert date strings to datetime
#     regime_periods['start_date'] = pd.to_datetime(regime_periods['start_date'])
#     regime_periods['end_date'] = pd.to_datetime(regime_periods['end_date'])
#
#     return regime_periods

def load_regime_periods(regime_name):
    """Load regime periods from CSV for a specific regime"""
    regime_df = pd.read_csv(REGIME_PERIODS_FILE)

    # Apply special filtering for Crisis/Bear regime
    if regime_name == "Crisis/Bear":
        print(f"📊 Filtering Crisis/Bear periods (minimum {MIN_CRISIS_DAYS} days)...")

        # Get all crisis periods first for statistics
        all_crisis = regime_df[regime_df['regime_name'] == regime_name]

        # Filter out short crisis periods that are likely noise
        regime_periods = regime_df[
            (regime_df['regime_name'] == regime_name) &
            (regime_df['days'] >= MIN_CRISIS_DAYS)
            ].copy()

        # Print filtering statistics
        print(f"   Total Crisis/Bear periods: {len(all_crisis)}")
        print(f"   Periods >= {MIN_CRISIS_DAYS} days: {len(regime_periods)}")
        print(f"   Filtered out: {len(all_crisis) - len(regime_periods)} short periods")
        print(f"   Total days before filtering: {all_crisis['days'].sum()}")
        print(f"   Total days after filtering: {regime_periods['days'].sum()}")

        # Show what major periods we're keeping
        major_periods = regime_periods[regime_periods['days'] >= 30].sort_values('days', ascending=False)
        if len(major_periods) > 0:
            print(f"\n   Major crisis periods included:")
            for _, period in major_periods.head(5).iterrows():
                print(f"   - {period['start_date']} to {period['end_date']}: {period['days']} days")
    else:
        # Standard filtering for other regimes
        regime_periods = regime_df[regime_df['regime_name'] == regime_name].copy()

    print(f"\n📅 Using {len(regime_periods)} {regime_name} periods")
    print(f"📅 Total days in analysis: {regime_periods['days'].sum()}")

    # Convert date strings to datetime
    regime_periods['start_date'] = pd.to_datetime(regime_periods['start_date'])
    regime_periods['end_date'] = pd.to_datetime(regime_periods['end_date'])

    return regime_periods

# Add configuration loading function
def load_marketstack_config():
    """Load Marketstack API configuration"""
    config_file = r"./Config/marketstack_config.json"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
            return config.get('api_key', MARKETSTACK_API_KEY)
    return MARKETSTACK_API_KEY


def sanitize_filename(filename):
    """
    Sanitize a string to be safe for use as a filename.
    Replaces problematic characters with underscores.
    """
    # Characters that are problematic in filenames
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']

    safe_name = filename
    for char in invalid_chars:
        safe_name = safe_name.replace(char, '_')

    # Also replace spaces with underscores for consistency
    safe_name = safe_name.replace(' ', '_')

    return safe_name

def fetch_marketstack_data(symbols, date_from, date_to, api_key):
    """
    Fetch EOD data from Marketstack API for multiple symbols
    Returns a DataFrame with adjusted close prices
    """
    print(f"Fetching data from Marketstack API for {len(symbols)} symbols...")
    print(f"Date range: {date_from} to {date_to}")

    # Collect all dataframes in a list first
    all_symbol_dfs = []

    # Fetch each symbol individually
    for i, symbol in enumerate(symbols):
        print(f"Fetching {symbol} ({i + 1}/{len(symbols)})", end='')

        params = {
            'access_key': api_key,
            'symbols': symbol,
            'date_from': date_from,
            'date_to': date_to,
            'limit': 1000,
            'sort': 'ASC'
        }

        endpoint = f"{MARKETSTACK_BASE_URL}/eod"
        symbol_data = []
        offset = 0

        while True:
            params['offset'] = offset

            try:
                response = requests.get(endpoint, params=params)
                response.raise_for_status()

                data = response.json()

                if 'error' in data:
                    print(f" ❌ API Error: {data['error']['message']}")
                    break

                if 'data' not in data or not data['data']:
                    break

                symbol_data.extend(data['data'])

                # Check pagination
                pagination = data.get('pagination', {})
                total = pagination.get('total', 0)
                count = pagination.get('count', 0)

                if offset + count >= total:
                    break

                offset += count
                time.sleep(0.1)  # Rate limiting

            except requests.exceptions.RequestException as e:
                print(f" ❌ Request Error: {e}")
                break

        # Process symbol data
        if symbol_data:
            df = pd.DataFrame(symbol_data)
            df['date'] = pd.to_datetime(df['date'])

            # Remove timezone info
            if df['date'].dt.tz is not None:
                df['date'] = df['date'].dt.tz_localize(None)

            df = df.set_index('date')

            # Create OHLCV DataFrame with prefixed column names
            price_df = pd.DataFrame({
                f'{symbol}_Open': df['open'],
                f'{symbol}_High': df['high'],
                f'{symbol}_Low': df['low'],
                f'{symbol}_Close': df['adj_close'],  # Use adjusted close
                f'{symbol}_Volume': df['volume']
            })

            price_df = price_df.sort_index()

            # Add to list instead of concatenating immediately
            all_symbol_dfs.append(price_df)

            print(f" ✓ ({len(df)} records)")
        else:
            print(f" ✗ (no data)")

    # Concatenate all dataframes at once - this is much more efficient
    if all_symbol_dfs:
        all_price_data = pd.concat(all_symbol_dfs, axis=1, join='outer')

        # Sort and clean
        all_price_data = all_price_data.sort_index()

        # Forward fill then backward fill
        all_price_data = all_price_data.ffill().bfill()
    else:
        all_price_data = pd.DataFrame()

    print(f"✅ Successfully fetched data for {len(all_symbol_dfs)} symbols")
    if not all_price_data.empty:
        print(f"Total date range: {all_price_data.index[0]} to {all_price_data.index[-1]}")

    return all_price_data

def marketstack_hist(ticker, years=5, api_key=None):
    """
    Replacement for yahooquery_hist_fixed using Marketstack API
    Returns OHLCV data for a single ticker
    """
    if api_key is None:
        api_key = load_marketstack_config()

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)

    # Fetch data
    all_data = fetch_marketstack_data(
        [ticker],
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d'),
        api_key
    )

    if all_data.empty:
        return pd.DataFrame()

    # Extract single ticker data
    ticker_data = pd.DataFrame()
    for base_col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        col_name = f"{ticker}_{base_col}"
        if col_name in all_data.columns:
            ticker_data[base_col] = all_data[col_name]

    return ticker_data


# def calculate_exponential_weights(n_periods, halflife):
#     """
#     Calculate exponential weights for time series data
#     More recent data gets higher weight
#
#     Args:
#         n_periods: Number of time periods
#         halflife: Half-life in periods for exponential decay
#
#     Returns:
#         Array of weights (most recent has highest weight)
#     """
#     # Create decay factor
#     decay_factor = np.log(2) / halflife
#
#     # Calculate weights (reversed so most recent is last)
#     periods = np.arange(n_periods)
#     weights = np.exp(-decay_factor * (n_periods - 1 - periods))
#
#     # Normalize weights to sum to n_periods (to maintain scale)
#     weights = weights * n_periods / weights.sum()
#
#     return weights


# Instead of simple exponential weights based on continuous time
# Weight based on recency of each regime period
# def calculate_regime_aware_weights(dates, regime_periods):
#     """Calculate weights that account for non-contiguous regime periods"""
#     weights = np.ones(len(dates))
#
#     # Get the most recent regime period end date
#     most_recent_date = regime_periods['end_date'].max()
#
#     for i, date in enumerate(dates):
#         # Find which regime period this date belongs to
#         period_mask = (regime_periods['start_date'] <= date) & (regime_periods['end_date'] >= date)
#         if period_mask.any():
#             period_end = regime_periods.loc[period_mask, 'end_date'].iloc[0]
#             # Calculate days from period end to most recent period
#             days_ago = (most_recent_date - period_end).days
#             # Apply exponential decay based on regime period recency
#             weights[i] = np.exp(-days_ago / (EXPONENTIAL_DECAY_HALFLIFE * 2))
#
#     # Normalize weights
#     weights = weights * len(weights) / weights.sum()
#     return weights

def get_next_month_return_weighted(ticker, date, hist_cache, exponential_weights=True):
    """
    Compute 1‑month forward return with optional exponential weighting
    """
    hist = hist_cache.get(ticker, pd.DataFrame())
    if hist.empty:
        return None

    # Ensure both the date and hist index are timezone-naive
    date_naive = date
    if hasattr(date, 'tz') and date.tz is not None:
        date_naive = date.tz_localize(None)

    try:
        # Find the exact date or closest date in the index
        closest_idx = hist.index.get_indexer([date_naive], method='nearest')[0]
        if closest_idx < 0 or closest_idx >= len(hist):
            return None

        start_date = hist.index[closest_idx]

        # Look ahead 21 trading days
        if closest_idx + 21 >= len(hist):
            return None

        start_price = hist['Close'].iloc[closest_idx]
        end_price = hist['Close'].iloc[closest_idx + 21]

        # Calculate return
        simple_return = (end_price - start_price) / start_price

        # If using exponential weighting, apply more weight to recent returns
        if exponential_weights and closest_idx > 0:
            # Get returns leading up to this date for context
            lookback_window = min(126, closest_idx)  # 6 months or available data
            recent_returns = hist['Close'].pct_change().iloc[closest_idx - lookback_window:closest_idx]

            if len(recent_returns) > 20:
                # Calculate exponential weights
                regime_periods = load_regime_periods(SELECTED_REGIME)
                weights = calculate_regime_aware_weights(dates, regime_periods)

                # Calculate weighted average recent volatility
                weighted_vol = np.sqrt(np.average(recent_returns ** 2, weights=weights))

                # Adjust return by recent volatility context
                # This gives more weight to returns during stable periods
                vol_adjustment = 1.0 / (1.0 + weighted_vol * 10)  # Scaling factor

                return simple_return * vol_adjustment

        return simple_return

    except Exception as e:
        print(f"⚠️ Error calculating return for {ticker}: {e}")
        return None

def get_next_month_return(ticker, date, hist_cache):
    """
    Compute 1‑month forward return - simplified version
    """
    hist = hist_cache.get(ticker, pd.DataFrame())
    if hist.empty:
        return None

    # Ensure both the date and hist index are timezone-naive
    date_naive = date
    if hasattr(date, 'tz') and date.tz is not None:
        date_naive = date.tz_localize(None)

    try:
        # Find the exact date or closest date in the index
        closest_idx = hist.index.get_indexer([date_naive], method='nearest')[0]
        if closest_idx < 0 or closest_idx >= len(hist):
            return None

        # Look ahead 21 trading days
        if closest_idx + 21 >= len(hist):
            return None

        start_price = hist['Close'].iloc[closest_idx]
        end_price = hist['Close'].iloc[closest_idx + 21]

        # Calculate simple return
        return (end_price - start_price) / start_price

    except Exception as e:
        print(f"⚠️ Error calculating return for {ticker}: {e}")
        return None

# --- Data Loading Functions ---
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

def yahooquery_hist_fixed(ticker, years=5):
    """Maximally robust version that handles any timezone issues"""
    try:
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
        except Exception as e:
            print(f"Initial history fetch failed for {ticker}: {e}")
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
                    'Close': row.get('close', row.get('Close', None)),
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

            return result_df.dropna().sort_index()

        except Exception as e:
            print(f"Data processing failed for {ticker}: {e}")
            return pd.DataFrame()

    except Exception as e:
        print(f"⚠️ Error fetching history for {ticker}: {e}")
        return pd.DataFrame()

def safe_get_info(ticker, retries=3, delay_range=(1, 3)):
    for attempt in range(retries):
        try:
            return fast_info(ticker)
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1}/{retries} failed for {ticker}: {e}")
            time.sleep(random.uniform(*delay_range))
    print(f"❌ All attempts failed for {ticker}")
    return {}


def sector_industry(info):
    """Return (sector, industry) strings or 'Unknown' placeholders."""
    return (
        info.get("sector") or info.get("sectorDisp") or "Unknown",
        info.get("industry") or info.get("industryDisp") or "Unknown"
    )


# --- Helper Functions ---
def get_next_month_return(ticker, date, hist_cache):
    """
    Compute 1‑month forward return using cached daily prices.
    Handles timezone issues robustly.
    """
    hist = hist_cache.get(ticker, pd.DataFrame())
    if hist.empty:
        return None

    # Ensure both the date and hist index are timezone-naive
    date_naive = date
    if hasattr(date, 'tz') and date.tz is not None:
        date_naive = date.tz_localize(None)

    try:
        # Find the exact date or closest date in the index
        closest_idx = hist.index.get_indexer([date_naive], method='nearest')[0]
        if closest_idx < 0 or closest_idx >= len(hist):
            return None

        start_date = hist.index[closest_idx]

        # Look ahead 21 trading days (if we have enough data)
        if closest_idx + 21 >= len(hist):
            return None

        start_price = hist['Close'].iloc[closest_idx]
        end_price = hist['Close'].iloc[closest_idx + 21]  # 21st trading day

        return (end_price - start_price) / start_price
    except Exception as e:
        print(f"⚠️ Error calculating return for {ticker}: {e}")
        return None


def get_fundamentals(ticker):
    """
    Returns four scores: (value_score, quality_score, financial_health_score, roic)
    Each factor is scored on a continuous 0-1 scale.
    Also returns value_metrics_dict for industry adjustment.
    """
    try:
        info = fast_info(ticker)

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

        # 1. PURE VALUE METRICS (simplified)
        value_scores_dict = {}

        if earnings_yield is not None:
            add_score(value_scores_dict, 'earnings_yield', earnings_yield, 0.05, 0.10, True)

        if fcf_yield is not None and fcf_yield > 0:
            add_score(value_scores_dict, 'fcf_yield', fcf_yield, FUND_THRESHOLDS['fcf_yield'], 0.10, True)

        if ev_to_ebitda is not None and ev_to_ebitda > 0:
            add_score(value_scores_dict, 'ev_to_ebitda', ev_to_ebitda, FUND_THRESHOLDS['ev_to_ebitda'], 6, False)

        if pb is not None and pb > 0:
            add_score(value_scores_dict, 'pb', pb, FUND_THRESHOLDS['pb'], 1, False)

        if ev_to_sales is not None and ev_to_sales > 0:
            add_score(value_scores_dict, 'ev_to_sales', ev_to_sales, FUND_THRESHOLDS['ev_to_sales'], 1, False)

        # 2. QUALITY METRICS (expanded)
        quality_scores_dict = {}

        if roe is not None:
            add_score(quality_scores_dict, 'roe', roe, FUND_THRESHOLDS['roe'], 0.25, True)

        if roic is not None:
            add_score(quality_scores_dict, 'roic', roic, FUND_THRESHOLDS['roic'], 0.20, True)

        if gross_margin is not None:
            add_score(quality_scores_dict, 'gross_margin', gross_margin, FUND_THRESHOLDS['gross_margin'], 0.6, True)

        if operating_margin is not None:
            add_score(quality_scores_dict, 'operating_margin', operating_margin, FUND_THRESHOLDS['operating_margin'],
                      0.2, True)

        # FCF margin
        fcf_margin = fcf / revenue if fcf and revenue and revenue > 0 else None
        if fcf_margin is not None:
            add_score(quality_scores_dict, 'fcf_margin', fcf_margin, FUND_THRESHOLDS['fcf_margin'], 0.15, True)

        if asset_turnover is not None and asset_turnover > 0:
            add_score(quality_scores_dict, 'asset_turnover', asset_turnover, FUND_THRESHOLDS['asset_turnover'], 1.0,
                      True)

        # Revenue stability (new) - would need historical data
        # For now, use gross margin stability as proxy
        quality_scores_dict['revenue_stability'] = quality_scores_dict.get('gross_margin', 0) * 0.8

        # 3. FINANCIAL HEALTH METRICS (new separate factor)
        financial_health_scores = []

        if cash_debt_ratio is not None and cash_debt_ratio > 0:
            add_score(financial_health_scores, None, cash_debt_ratio, FUND_THRESHOLDS['cash_debt_ratio'], 0.6, True)

        if current_ratio is not None and current_ratio > 0:
            add_score(financial_health_scores, None, current_ratio, FUND_THRESHOLDS['current_ratio'], 2.5, True)

        if de is not None and de >= 0:
            add_score(financial_health_scores, None, de, FUND_THRESHOLDS['de_ratio'], 50, False)

        if interest_coverage is not None and interest_coverage > 0:
            add_score(financial_health_scores, None, interest_coverage, FUND_THRESHOLDS['interest_coverage'], 10, True)

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
        print(f"⚠️ Error fetching fundamentals for {ticker}: {e}")
        return 0, 0, 0, None, {}


def get_growth_score(ticker, info=None):
    """
    Forward-looking growth factor with emphasis on growth acceleration and estimates.
    """
    if info is None:
        info = fast_info(ticker)

    growth_metrics = {}

    # 1. Revenue Growth (keep but reduce weight)
    rev_growth = info.get("revenueGrowth", 0)
    if rev_growth is not None and not pd.isna(rev_growth):
        norm_rev_growth = min(max((rev_growth - 0.05) / 0.25, 0), 1)
        growth_metrics['revenue_growth'] = (norm_rev_growth, 0.8)  # Reduced weight

    # 2. PEG Ratio (moved from value - lower is better for growth)
    peg = info.get("pegRatio")
    if peg is not None and peg > 0:
        # Invert: PEG < 1 is excellent, > 2 is poor
        norm_peg = max(0, min(1, (2 - peg) / 1.5))
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

def get_size_score(info, mcap_min, mcap_80th):
    """
    Enhanced size score with continuous scaling that considers both market cap and stock price.
    - Market cap: smaller is better (80% of score)
    - Price: mid-range is best - too low or too high is penalized (20% of score)

    Returns a score between 0-1, where higher means smaller/better size characteristics.
    """
    cap = info.get('marketCap')
    price = info.get('regularMarketPrice')

    if cap is None:
        return 0

    # Market cap component (smaller is better)
    cap_score = (mcap_80th - min(cap, mcap_80th)) / (mcap_80th - mcap_min) if mcap_80th > mcap_min else 0
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

def get_credit_score(info):
    """
    Enhanced credit scoring that evaluates solvency, debt service capacity,
    and earnings quality across multiple time frames.
    """
    try:
        # Basic financial metrics
        ebitda = info.get("ebitda", 0)
        sales = info.get("totalRevenue", 1)
        debt = info.get("totalDebt", 0)
        mcap = info.get("marketCap", 0)
        interest_expense = info.get("interestExpense", 1)
        cash = info.get("totalCash", 0)
        current_assets = info.get("totalCurrentAssets", 0)
        current_liabilities = info.get("totalCurrentLiabilities", 1)

        # Handle missing data
        if any(x in (None, 0, "None") or pd.isna(x) for x in
               (ebitda, sales, debt, mcap, interest_expense)):
            return 0

        # 1. Altman Z-Score components
        profit_ratio = ebitda / sales  # Operating efficiency
        solvency_ratio = mcap / debt  # Market-based cushion

        # 2. Debt service metrics
        interest_coverage = ebitda / interest_expense  # EBITDA/Interest

        # Normalize these metrics to 0-1 scale
        z_component = np.tanh((3.3 * profit_ratio + 0.6 * solvency_ratio) / 4)
        coverage_score = np.tanh(interest_coverage / 10)  # >10x is excellent

        # 3. Liquidity metrics - short-term debt payment ability
        quick_ratio = current_assets / current_liabilities
        cash_to_debt = cash / debt if debt > 0 else 1.0

        liquidity_score = np.tanh((quick_ratio + cash_to_debt) / 2)

        # 4. Weighted final score - more weight to coverage during economic stress
        final_score = (
                0.4 * z_component +
                0.4 * coverage_score +
                0.2 * liquidity_score
        )

        return round(max(0, min(final_score, 1)), 2)

    except Exception as e:
        print(f"⚠️ Credit score error for {info.get('symbol', '?')}: {e}")
        return 0

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

def get_robust_carry_score(ticker, info):
    """Robust carry score with fallbacks for API inconsistency"""
    # Attempt to use the regular method first
    try:
        score = get_carry_score_simple(ticker, info)
        if score != 0:  # If we got a non-zero score, use it
            return score
    except Exception as e:
        print(f"Regular carry calculation failed for {ticker}: {e}")

    # If the regular method fails or returns zero, try a simplified approach
    try:
        # Simplified dividend scoring based purely on yield
        div_yield = info.get("dividendYield", 0)
        if div_yield is None or div_yield == 0:
            return 0

        # Simple approach: any dividend > 0 gets at least 0.25
        # Higher yields get proportionally more up to 1.0
        base_score = 0.25
        # Add bonus for high yields (3% → 0, 8% → 1)
        bonus = min(max((div_yield - 0.03) / 0.05, 0), 0.75)

        return round(base_score + bonus, 2)
    except Exception:
        # Last resort fallback
        return 0

def get_insider_score_simple(ticker):
    """Continuous insider score based on magnitude and patterns"""
    # Check if we already know this ticker has no insider data
    if ticker in _insider_no_data_tickers:
        return 0

    # Check cache first
    if ticker in _insider_data_cache:
        return _insider_data_cache[ticker]

    try:
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

        # Calculate net buying/selling values
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
    """Simple 12-month momentum (no 1-month skip)"""
    if len(df) < 180:
        return None
    ret = (df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1
    return ret


def _earnings_mom(ticker):
    """EPS estimate changes + surprises"""
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


def get_momentum_score(ticker, df):
    """Returns raw price momentum and earnings-momentum"""
    pm = _price_mom(df)
    em = _earnings_mom(ticker)
    # convert None → np.nan so zscore can ignore them
    return (pm if pm is not None else np.nan,
            em if em is not None else np.nan)

def _cvar95(r):
    """Parametric ES at 95% assuming normality"""
    mu, sigma = np.mean(r), np.std(r, ddof=1)
    cvar = mu - sigma * 2.06  # expected shortfall (ES) ≈ μ – 2.06σ
    return abs(min(cvar, 0))


def _cdar(df):
    """Conditional drawdown at risk (CDaR) over 1 year"""
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

def _breakout_score(df):
    """52-week high proximity"""
    close = df["Close"].iloc[-1]
    hi_52w = df["Close"].max()  # full 1-year window already loaded
    pct_of_hi = close / hi_52w if hi_52w else 0
    score = np.clip((pct_of_hi - 0.70) / 0.25, 0, 1)  # maps 0.70→0; 0.95→1
    return round(float(score), 4)


def _trend_score(df):
    """Combined price trend strength"""
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

    return round(score, 4)


def get_technical_score(df):
    """Three-component technical score"""
    bko = _breakout_score(df)
    trend = _trend_score(df)
    ma = _ma_score(df)

    score = 0.35 * bko + 0.40 * trend + 0.25 * ma
    return round(score, 4)


# --- Correlation Analysis Functions ---
def calculate_factor_correlations(factor_data, returns):
    """
    Calculate correlations between factor scores and returns
    Returns both Pearson and Spearman correlations with proper error handling
    """
    correlations = {}

    # Convert returns Series to right shape for correlation
    returns_array = returns.values

    for factor, values in factor_data.items():
        values_array = np.array(values)
        if len(values_array) == len(returns_array):
            try:
                # Check if the array is constant (all values the same)
                unique_values = np.unique(values_array)
                if len(unique_values) <= 1:
                    print(f"Factor '{factor}' has constant values: {unique_values}")
                    # Instead of skipping, assign a zero correlation
                    correlations[factor] = {
                        'pearson': 0.0,
                        'pearson_p': 1.0,
                        'spearman': 0.0,
                        'spearman_p': 1.0,
                        'is_significant': False,
                        'error': 'Constant factor values'
                    }
                    continue

                # Calculate Pearson correlation
                pearson_corr, p_value = pearsonr(values_array, returns_array)

                # Calculate Spearman rank correlation
                spearman_corr, sp_p_value = spearmanr(values_array, returns_array)

                correlations[factor] = {
                    'pearson': pearson_corr,
                    'pearson_p': p_value,
                    'spearman': spearman_corr,
                    'spearman_p': sp_p_value,
                    'is_significant': p_value < 0.05 or sp_p_value < 0.05
                }
            except Exception as e:
                # Handle any correlation calculation errors
                correlations[factor] = {
                    'pearson': None,
                    'pearson_p': None,
                    'spearman': None,
                    'spearman_p': None,
                    'is_significant': False,
                    'error': str(e)
                }

    return correlations


def plot_correlation_heatmap(factor_data, returns, output_file='factor_correlation_heatmap.png', results_dir='./Results'):
    """Create a correlation heatmap of factors and returns"""
    # Update the output path
    output_path = os.path.join(results_dir, output_file)

    # Create DataFrame from factor data
    df = pd.DataFrame(factor_data)

    # Add returns to the DataFrame
    df['Returns'] = returns.values

    # Calculate correlation matrix
    corr_matrix = df.corr(method='spearman')

    # Create the heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                vmin=-1, vmax=1, center=0, square=True, linewidths=.5)

    plt.title('Factor Correlation Heatmap (Spearman)', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Correlation heatmap saved to {output_path}")
    return corr_matrix


def weight_optimization_analysis(factor_data, returns, weight_set):
    """
    Analyze how the weight set performs - simplified version
    that doesn't rely on the arch package
    """
    # Create factor score DataFrame
    df = pd.DataFrame(factor_data)

    # Calculate weighted scores based on provided weights
    weights = []
    for col in df.columns:
        weights.append(weight_set.get(col, 0))

    # Calculate weighted score for each ticker
    df['weighted_score'] = 0
    for i, col in enumerate(df.columns):
        if col in weight_set:
            df['weighted_score'] += df[col] * weight_set[col]

    # Calculate correlation between weighted score and returns
    try:
        corr, p_value = pearsonr(df['weighted_score'].values, returns.values)
    except:
        corr, p_value = None, None

    result = {
        'mean_correlation': corr,
        'confidence_interval': [corr - 0.1, corr + 0.1] if corr is not None else [None, None],
        'weights': weight_set,
        'p_value': p_value
    }

    return result


def save_analysis_results(results_text, filename="analysis_results.txt", results_dir="./Results"):
    """
    Save the analysis results to a text file.

    Args:
        results_text (str): The text to save
        filename (str): The output filename
        results_dir (str): The output directory
    """
    try:
        # Create full path
        filepath = os.path.join(results_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(results_text)
        print(f"✅ Analysis results saved to {filepath}")
    except Exception as e:
        print(f"❌ Failed to save analysis results: {e}")

# --- Main Logic ---
def main():
    start_time = time.time()

    # Create Results directory if it doesn't exist
    results_dir = "./Results"
    os.makedirs(results_dir, exist_ok=True)

    # Load Marketstack API key
    api_key = load_marketstack_config()
    if api_key == "YOUR_API_KEY_HERE":
        print("⚠️ Please set your Marketstack API key in marketstack_config.json")
        return

    # Clear caches to avoid stale data
    _info_cache.clear()
    _insider_data_cache.clear()
    _insider_no_data_tickers.clear()
    _carry_data_cache.clear()
    _carry_no_data_tickers.clear()

    # -------- 0.  load tickers -------------------------------------------------
    try:
        with open(TICKER_FILE, "r") as f:
            tickers = [ln.strip().replace("$", "") for ln in f if ln.strip()]
    except FileNotFoundError:
        print(f"❌ Ticker file '{TICKER_FILE}' not found!")
        return

    if len(tickers) < 5:
        print(f"⚠️ Warning: Only {len(tickers)} tickers found. Correlation analysis works best with 10+ tickers.")

    print(f"✅ Loaded {len(tickers)} tickers from file.")

    # -------- 1. Fetch price data using Marketstack ---------------------------
    print("📦 Downloading data from Marketstack...")

    # Calculate date range (5 years for fundamentals, extra for correlation analysis)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 5)  # 5 years of data

    # Fetch all ticker data at once
    all_marketstack_data = fetch_marketstack_data(
        tickers,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d'),
        api_key
    )

    # Process into individual ticker DataFrames
    history_cache = {}
    failed_tickers = []

    for ticker in tickers:
        ticker_data = pd.DataFrame()
        has_data = False

        for base_col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            col_name = f"{ticker}_{base_col}"
            if col_name in all_marketstack_data.columns:
                ticker_data[base_col] = all_marketstack_data[col_name]
                has_data = True

        if has_data and not ticker_data.empty:
            ticker_data = ticker_data.dropna()
            if len(ticker_data) >= 200:  # Minimum data requirement
                history_cache[ticker] = ticker_data
            else:
                failed_tickers.append(ticker)
                print(f"⚠️ Insufficient data for {ticker}: only {len(ticker_data)} days")
        else:
            failed_tickers.append(ticker)
            print(f"⚠️ No data found for {ticker}")

    # Remove failed tickers
    tickers = [t for t in tickers if t not in failed_tickers]

    if not tickers:
        print("❌ No valid tickers with sufficient data")
        return

    print(f"✅ Successfully loaded price data for {len(tickers)} tickers")

    # -------- 1-a. pre-scan: avgVol for liquidity -----------------------------
    avg_vol = {}
    for tk in tickers:
        try:
            avg_vol[tk] = fast_info(tk).get("averageVolume10days", 0)
        except Exception:
            avg_vol[tk] = 0
    vol_min, vol_max = min(avg_vol.values()), max(avg_vol.values())

    # -------- 1-b. main data harvest ------------------------------------------
    print("📦 Downloading data and calculating factor scores...")

    fundamentals_cache = {}
    history_cache = {}
    raw = []  # store everything, score later
    pm_cache, em_cache, mcaps = [], [], []
    value_metrics_by_ticker = {}

    for i, tk in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] Processing {tk} …")
        try:
            # Get fundamentals first
            info = safe_get_info(tk)
            fundamentals_cache[tk] = info

            # Get price history
            hist = yahooquery_hist_fixed(tk, years=5)

            if hist.empty or len(hist) < 200:
                print(f"⚠️ Insufficient data for {tk}, skipping...")
                continue

            history_cache[tk] = hist

            # --- Calculate Factor Scores with new structure ---
            value, quality, financial_health, roic, value_dict = get_fundamentals(tk)
            value_metrics_by_ticker[tk] = value_dict

            tech = get_technical_score(hist)
            insider = get_insider_score_simple(tk)

            # Enhanced growth score
            growth = get_growth_score(tk, info)

            # ─ momentum raw numbers
            pm, em = get_momentum_score(tk, hist)
            pm_cache.append(pm)
            em_cache.append(em)

            # ─ downside-risk penalty
            market_hist = history_cache.get('SPY', None)  # Replace 'SPY' with your market index ticker
            # penalty = get_risk_penalty(hist, market_hist)
            stability = get_stability_score(hist, market_hist)

            # ─ misc factors
            mcap = info.get("marketCap")
            mcaps.append(mcap)
            sector, industry = sector_industry(info)

            # ─ get company name and country
            company_name, country = get_company_info(info)

            credit = get_credit_score(info)
            carry = get_carry_score_simple(tk, info)

            # liquidity (0-1)
            liq = 0
            if vol_max != vol_min:
                liq = round((avg_vol[tk] - vol_min) / (vol_max - vol_min), 2)

            raw.append(dict(
                Ticker=tk,
                Value=value,
                Quality=quality,
                FinancialHealth=financial_health,  # NEW
                Technical=tech,
                Insider=insider,
                PriceMom=pm,
                EarnMom=em,
                Stability=stability,  # Changed from Penalty
                MarketCap=mcap,
                Credit=credit,
                Liquidity=liq,
                Carry=carry,
                Sector=sector,
                Industry=industry,
                Growth=growth,  # Now using enhanced growth score
                ROIC=roic,  # Store ROIC for potential use
                CompanyName=company_name,
                Country=country
            ))

            # Add a small delay to be respectful to the API
            time.sleep(random.uniform(0.1, 0.5))

        except Exception as e:
            print(f"⚠️ Error processing {tk}: {e}")
            continue

    if not raw:
        print("⚠️ No usable tickers")
        return

    # -------- 2.  cross-sectional normalisations ------------------------------
    # 2-a momentum z-scores → logistic 0-1
    pm_arr = np.asarray(pm_cache)
    em_arr = np.asarray(em_cache)

    def safe_z(a):
        # Convert array to float, explicitly handling non-numeric values by converting to NaN
        a_float = np.array([np.nan if x is None or not isinstance(x, (int, float)) else x for x in a], dtype=np.float64)
        finite = np.isfinite(a_float)
        if finite.sum() <= 2:  # 0 or 1 valid numbers
            return np.zeros_like(a_float, dtype=float)
        return zscore(a_float, nan_policy='omit')

    pm_z = safe_z(pm_arr)
    em_z = safe_z(em_arr)
    squash = lambda z: 1 / (1 + np.exp(-np.clip(z, -2, 2)))

    # 2-b size-percentile breakpoints
    mcaps_clean = [m for m in mcaps if m]
    mc_min = np.nanmin(mcaps_clean)
    mc_80th = np.nanpercentile(mcaps_clean, 80)

    # --- 2‑c  sector‑z scores for growth & ROIC -----------------------
    df_raw = pd.DataFrame(raw)

    # Calculate growth scores with sector adjustment
    df_raw = apply_sector_relative_growth_adjustment(df_raw)

    # Create columns for each value metric in your DataFrame
    value_metric_names = list(VALUE_METRIC_WEIGHTS.keys())
    for metric in value_metric_names:
        df_raw[f'value_{metric}'] = df_raw['Ticker'].map(
            lambda t: value_metrics_by_ticker.get(t, {}).get(metric, np.nan)
        )

    # Calculate industry-relative z-scores for each value metric
    for metric in value_metric_names:
        col_name = f'value_{metric}'
        df_raw[f'{col_name}_industry_z'] = df_raw.groupby('Industry')[col_name].transform(
            lambda x: (x - x.mean()) / (x.std() if x.std() > 0 else 1)
        )
        # Convert z-scores to 0-1 range
        df_raw[f'{col_name}_norm'] = df_raw[f'{col_name}_industry_z'].apply(
            lambda z: 1 / (1 + np.exp(-np.clip(z, -2, 2))) if pd.notna(z) else np.nan
        )

    # 3. Calculate the final industry-adjusted value score
    industry_adjusted_value_scores = []
    for _, row in df_raw.iterrows():
        weighted_score = 0
        total_weight = 0
        for metric in value_metric_names:
            norm_col = f'value_{metric}_norm'
            if pd.notna(row[norm_col]):
                weight = VALUE_METRIC_WEIGHTS[metric]
                weighted_score += row[norm_col] * weight
                total_weight += weight

        if total_weight > 0:
            industry_adjusted_value_scores.append(weighted_score / total_weight)
        else:
            industry_adjusted_value_scores.append(row['Value'])  # Fall back to original value

    df_raw['ValueIndustryAdj'] = industry_adjusted_value_scores

    # -------- 3.  Calculate correlations with 3-year window -------------------
    # print("💻 Calculating scores and collecting 3-year return data with exponential weighting...")
    print("💻 Calculating scores...")

    # Load regime periods and generate dates within those periods
    regime_periods = load_regime_periods(SELECTED_REGIME)

    # Generate dates only within regime periods
    all_regime_dates = []
    for _, period in regime_periods.iterrows():
        period_dates = pd.date_range(
            start=period['start_date'],
            end=period['end_date'],
            freq='MS'  # Month start
        )
        all_regime_dates.extend(period_dates)

    # Remove duplicates and sort
    dates = pd.DatetimeIndex(sorted(set(all_regime_dates)))
    dates = dates.tz_localize(None)  # Ensure timezone-naive

    print(f"📊 Using {len(dates)} monthly data points from {len(regime_periods)} {SELECTED_REGIME} regime periods")
    print(f"📅 Total regime days: {regime_periods['days'].sum()}")

    # Initialize data structures for correlation analysis
    factor_scores = {
        'value': [],
        'quality': [],
        'financial_health': [],  # NEW
        'technical': [],
        'insider': [],
        'momentum': [],
        'stability': [],  # Changed from volatility_penalty
        'size': [],
        'credit': [],
        'liquidity': [],
        'carry': [],
        'growth': []
    }

    total_scores = []
    ticker_returns = []
    ticker_list = []

    # Process each ticker
    for k, rec in df_raw.iterrows():
        ticker = rec['Ticker']

        # [Keep all the existing score calculation code]
        # Calculate momentum from raw components
        pm_score = squash(pm_z[k]) if not np.isnan(pm_z[k]) else 0
        em_score = squash(em_z[k]) if not np.isnan(em_z[k]) else 0
        momentum = round(0.6 * pm_score + 0.4 * em_score, 3)

        # Size score
        size = get_size_score({'marketCap': rec['MarketCap']}, mc_min, mc_80th)

        # Store all factor scores including new ones
        factor_scores['value'].append(rec['Value'])
        factor_scores['quality'].append(rec['Quality'])
        factor_scores['financial_health'].append(rec['FinancialHealth'])  # NEW
        factor_scores['technical'].append(rec['Technical'])
        factor_scores['insider'].append(rec['Insider'])
        factor_scores['momentum'].append(momentum)
        # factor_scores['volatility_penalty'].append(rec['Penalty'])
        factor_scores['stability'].append(rec['Stability'])  # Changed from volatility_penalty and Penalty
        factor_scores['size'].append(size)
        factor_scores['credit'].append(rec['Credit'])
        factor_scores['liquidity'].append(rec['Liquidity'])
        factor_scores['carry'].append(rec['Carry'])
        factor_scores['growth'].append(rec['Growth'])  # Using enhanced growth

        # Calculate total score with new weights
        total = (
                rec['Value'] * WEIGHTS['value'] +  # Industry adjusted version if using
                rec['Quality'] * WEIGHTS['quality'] +
                rec['FinancialHealth'] * WEIGHTS['financial_health'] +  # NEW
                rec['Technical'] * WEIGHTS['technical'] +
                rec['Insider'] * WEIGHTS['insider'] +
                momentum * WEIGHTS['momentum'] +
                rec['Stability'] * WEIGHTS['stability'] +  # Changed from Penalty * volatility_penalty
                size * WEIGHTS['size'] +
                rec['Credit'] * WEIGHTS['credit'] +
                rec['Liquidity'] * WEIGHTS['liquidity'] +
                rec['Carry'] * WEIGHTS['carry'] +
                rec['Growth'] * WEIGHTS['growth']  # Using enhanced growth
        )

        total_scores.append(total)
        ticker_list.append(ticker)

        # Get returns with exponential weighting
        monthly_returns = []
        return_dates = []
        # regime_weights = calculate_regime_aware_weights(dates, regime_periods)

        for i, date in enumerate(dates):
            if hasattr(date, 'tz') and date.tz is not None:
                date = date.tz_localize(None)

            hist = history_cache.get(ticker, pd.DataFrame())
            if hist.empty:
                continue

            # Check if this date is actually within a regime period
            in_regime = any((regime_periods['start_date'] <= date) & (regime_periods['end_date'] >= date))

            if in_regime:
                fwd = get_next_month_return(ticker, date, history_cache)
                if fwd is not None:
                    monthly_returns.append(fwd)
                    return_dates.append(date)

        # Calculate SIMPLE average return
        if monthly_returns:
            avg_return = np.mean(monthly_returns)  # Simple average instead of weighted
            ticker_returns.append(avg_return)
        else:
            ticker_returns.append(np.nan)

        # # Calculate weighted average return using regime-aware weights
        # if monthly_returns:
        #     # Get weights only for dates with returns
        #     valid_weights = [regime_weights[i] for i, d in enumerate(dates) if d in return_dates]
        #     avg_return = np.average(monthly_returns, weights=valid_weights)
        #     ticker_returns.append(avg_return)
        # else:
        #     ticker_returns.append(np.nan)

    # Create analysis DataFrame
    analysis_df = pd.DataFrame({
        'Ticker': ticker_list,
        'TotalScore': total_scores,
        'Return': ticker_returns
    })

    # Add factor scores to the DataFrame
    for factor, scores in factor_scores.items():
        analysis_df[factor] = scores

    # Remove any rows with NaN returns
    analysis_df = analysis_df.dropna(subset=['Return'])

    # -------- 4.  Enhanced Correlation Analysis --------------------------------
    total_regime_days = regime_periods['days'].sum()
    print(f"📊 Analyzing factor correlations using {total_regime_days} days of {SELECTED_REGIME} regime data...")

    # Get factor scores and returns for correlation analysis
    factor_data_for_corr = {factor: analysis_df[factor].values for factor in factor_scores.keys()}
    returns_for_corr = analysis_df['Return']

    # Calculate correlations
    correlations = calculate_factor_correlations(factor_data_for_corr, returns_for_corr)

    # Create correlation heatmap
    # correlation_matrix = plot_correlation_heatmap(factor_data_for_corr, returns_for_corr)
    correlation_matrix = plot_correlation_heatmap(factor_data_for_corr, returns_for_corr, results_dir=results_dir)

    # Run weight optimization analysis
    optimization_results = weight_optimization_analysis(factor_data_for_corr, returns_for_corr, WEIGHTS)

    # -------- 5.  Enhanced Output with 3-year analysis note -------------------
    # Create a string buffer to store all output
    from io import StringIO
    output_buffer = StringIO()

    def output(text):
        if isinstance(text, pd.DataFrame):
            # Convert DataFrame to string before writing
            print(text)
            output_buffer.write(text.to_string() + "\n")
        else:
            print(text)
            output_buffer.write(str(text) + "\n")

    output("\n" + "=" * 60)
    output(f"📊 FACTOR CORRELATION ANALYSIS RESULTS - {SELECTED_REGIME} REGIME")
    output(f"📅 Using {len(regime_periods)} historical {SELECTED_REGIME} periods")
    output(f"📅 Total days analyzed: {regime_periods['days'].sum()}")
    # output(f"💫 Recent regime periods weighted more heavily (half-life: {EXPONENTIAL_DECAY_HALFLIFE} days)")
    output("=" * 60)

    # Display correlations table
    output("\nFactor Correlations with Returns:")
    correlation_table = []
    for factor, corr_data in correlations.items():
        correlation_table.append({
            'Factor': factor,
            'Pearson': corr_data['pearson'] if corr_data['pearson'] is not None else 'N/A',
            'p-value': corr_data['pearson_p'] if corr_data['pearson_p'] is not None else 'N/A',
            'Spearman': corr_data['spearman'] if corr_data['spearman'] is not None else 'N/A',
            'Significant': "Yes" if corr_data.get('is_significant', False) else "No",
            'Error': corr_data.get('error', '')
        })

    # Create DataFrame but handle floating point formatting specially
    corr_df = pd.DataFrame(correlation_table)

    # Sort, but handle the case where all correlations might be None
    try:
        corr_df = corr_df.sort_values('Spearman', ascending=False)
    except:
        pass  # Skip sorting if it fails

    # Display with custom formatting to handle None values
    pd.set_option('display.max_columns', None)
    # print(corr_df.to_string(index=False))
    output(corr_df.to_string(index=False))

    # Handle potential errors in weight optimization results display
    # print("\nWeight Set Performance Analysis:")
    output("\nWeight Set Performance Analysis:")
    if optimization_results['mean_correlation'] is not None:
        # print(f"Mean Correlation: {optimization_results['mean_correlation']:.4f}")
        # print(f"95% Confidence Interval: [{optimization_results['confidence_interval'][0]:.4f}, {optimization_results['confidence_interval'][1]:.4f}]")
        output(f"Mean Correlation: {optimization_results['mean_correlation']:.4f}")
        output(
            f"95% Confidence Interval: [{optimization_results['confidence_interval'][0]:.4f}, {optimization_results['confidence_interval'][1]:.4f}]")
    else:
        # print("Mean Correlation: Not available (insufficient data)")
        # print("95% Confidence Interval: Not available")
        output("Mean Correlation: Not available (insufficient data)")
        output("95% Confidence Interval: Not available")

    # Save correlation results to CSV
    try:
        # corr_df.to_csv('factor_correlations.csv', index=False)
        # output("✅ Saved correlation results to factor_correlations.csv")

        corr_df.to_csv(os.path.join(results_dir, 'factor_correlations.csv'), index=False)
        output(f"✅ Saved correlation results to {os.path.join(results_dir, 'factor_correlations.csv')}")
    except Exception as e:
        # print(f"❌ Failed to save CSV: {e}")
        output(f"❌ Failed to save CSV: {e}")

    # Rank factors by correlation
    # print("\nFactors Ranked by Correlation with Returns:")
    output("\nFactors Ranked by Correlation with Returns:")
    for i, row in enumerate(corr_df.itertuples(), 1):
        sig_star = "*" if row.Significant == "Yes" else ""

        # Check if Spearman is a string or numeric value
        if isinstance(row.Spearman, str):
            # If it's already a string like 'N/A', use it as is
            spearman_str = row.Spearman
        else:
            # If it's a number, format it with 3 decimal places
            spearman_str = f"{row.Spearman:.3f}"

        # print(f"{i}. {row.Factor:<20} {spearman_str}{sig_star}")
        output(f"{i}. {row.Factor:<20} {spearman_str}{sig_star}")

    # Weight effectiveness analysis
    # print("\nWeight Effectiveness Analysis:")
    output("\nWeight Effectiveness Analysis:")
    weight_effectiveness = {}
    for factor in factor_scores.keys():
        # Get the correlation value
        corr_data = correlations[factor]
        spearman_corr = corr_data['spearman']

        # Calculate effectiveness only if correlation is numeric
        current_weight = WEIGHTS.get(factor, 0)
        if spearman_corr is not None:
            effectiveness = spearman_corr * current_weight
        else:
            effectiveness = None

        weight_effectiveness[factor] = effectiveness

    effectiveness_df = pd.DataFrame([
        {'Factor': factor,
         'Weight': WEIGHTS.get(factor, 0),
         'Correlation': correlations[factor]['spearman'],
         'Effectiveness': weight_effectiveness[factor]}
        for factor in factor_scores.keys()
    ])
    # Try to sort if possible
    try:
        effectiveness_df = effectiveness_df.sort_values('Effectiveness', ascending=False)
    except:
        pass  # Skip sorting if it fails

    # Custom display that handles both numeric and string values
    # print("Factor               Weight  Correlation  Effectiveness")
    # print("-" * 55)
    output("Factor               Weight  Correlation  Effectiveness")
    output("-" * 55)
    for _, row in effectiveness_df.iterrows():
        # Format correlation
        if isinstance(row['Correlation'], (float, int)):
            corr_str = f"{row['Correlation']:.3f}"
        else:
            corr_str = str(row['Correlation'])

        # Format effectiveness
        if isinstance(row['Effectiveness'], (float, int)):
            effect_str = f"{row['Effectiveness']:.3f}"
        else:
            effect_str = str(row['Effectiveness'])

        # print(f"{row['Factor']:<20} {row['Weight']:6.1f}  {corr_str:12}  {effect_str}")
        output(f"{row['Factor']:<20} {row['Weight']:6.1f}  {corr_str:12}  {effect_str}")

    # Try to save to CSV if possible
    try:
        # effectiveness_df.to_csv('weight_effectiveness.csv', index=False)
        effectiveness_df.to_csv(os.path.join(results_dir, 'weight_effectiveness.csv'), index=False)
    except Exception as e:
        # print(f"❌ Failed to save weight effectiveness CSV: {e}")
        output(f"❌ Failed to save weight effectiveness CSV: {e}")

    # Calculate suggested weight adjustments
    # Calculate suggested weight adjustments
    # print("\nSuggested Weight Adjustments:")
    output("\nSuggested Weight Adjustments:")
    total_weight = sum(WEIGHTS.values())

    for factor in factor_scores.keys():
        current_weight = WEIGHTS.get(factor, 0)
        correlation = correlations[factor]['spearman']

        # Check if correlation is None or a string
        if correlation is None or isinstance(correlation, str):
            suggestion = "Unknown (insufficient data)"
            corr_str = "N/A"
        else:
            # More nuanced suggestion logic that considers current weight
            if current_weight == 0:
                if correlation > 0.2:
                    suggestion = "Add weight (strong positive)"
                elif correlation > 0.05:
                    suggestion = "Consider adding small weight"
                else:
                    suggestion = "Keep at zero"
            elif current_weight >= 40:
                suggestion = "Near maximum weight"
            else:
                # For factors with non-zero, non-maximum weights
                if correlation > 0.3:
                    suggestion = "Increase substantially"
                elif correlation > 0.1:
                    suggestion = "Increase moderately"
                elif correlation > -0.1:
                    suggestion = "Maintain current weight"
                elif correlation > -0.2:
                    suggestion = "Reduce moderately"
                else:
                    suggestion = "Reduce substantially"

            corr_str = f"{correlation:.3f}"

        # print(f"{factor:<20} Current: {current_weight:2d}  Correlation: {corr_str}  Suggestion: {suggestion}")
        # output(f"{factor:<20} Current: {current_weight:2d}  Correlation: {corr_str}  Suggestion: {suggestion}")
        output(f"{factor:<20} Current: {current_weight:5.1f}  Correlation: {corr_str}  Suggestion: {suggestion}")

    # new:
    # --- Enhanced Results Analysis Section ---
    # print("\n" + "=" * 60)
    # print("📊 COMPREHENSIVE PERFORMANCE ANALYSIS WITH CONTINUOUS SCORING")
    # print("=" * 60)
    output("\n" + "=" * 60)
    output("📊 COMPREHENSIVE PERFORMANCE ANALYSIS WITH CONTINUOUS SCORING")
    output("=" * 60)

    # Initialize the evaluator
    evaluator = PerformanceEvaluator()

    # Prepare data for decile analysis
    df = pd.DataFrame({'Score': total_scores, 'Return': ticker_returns})
    df = df.dropna(subset=['Return'])  # Remove rows with NaN returns

    # Create deciles
    df["Decile"] = pd.qcut(df.Score, 10, labels=False, duplicates="drop")

    # 1. TOP DECILE BASIC METRICS
    # print("\n🎯 TOP DECILE PERFORMANCE")
    # print("-" * 40)
    output("\n🎯 TOP DECILE PERFORMANCE")
    output("-" * 40)
    top_decile_returns = df[df.Decile == 9]['Return']
    basic_metrics = {
        'avg_return': top_decile_returns.mean(),
        'std': top_decile_returns.std(),
        'sharpe': top_decile_returns.mean() / top_decile_returns.std() if top_decile_returns.std() > 0 else 0,
        'hit_rate': (top_decile_returns > 0).mean(),
        'best_month': top_decile_returns.max(),
        'worst_month': top_decile_returns.min()
    }

    for metric, value in basic_metrics.items():
        if metric in ['avg_return', 'best_month', 'worst_month']:
            # print(f"{metric.replace('_', ' ').title()}: {value:.4%}")
            output(f"{metric.replace('_', ' ').title()}: {value:.4%}")
        elif metric in ['std']:
            # print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
            output(f"{metric.replace('_', ' ').title()}: {value:.4f}")
        elif metric in ['sharpe']:
            # print(f"{metric.replace('_', ' ').title()}: {value:.3f}")
            output(f"{metric.replace('_', ' ').title()}: {value:.3f}")
        else:
            # print(f"{metric.replace('_', ' ').title()}: {value:.2%}")
            output(f"{metric.replace('_', ' ').title()}: {value:.2%}")

    # 2. DECILE PERFORMANCE BREAKDOWN
    output("\n📊 DECILE PERFORMANCE BREAKDOWN")
    output("-" * 40)
    decile_analysis = df.groupby('Decile').agg({
        'Return': ['mean', 'std', 'count']
    }).round(4)
    decile_analysis.columns = ['Mean Return', 'Std Dev', 'Count']
    decile_analysis['Sharpe'] = (decile_analysis['Mean Return'] /
                                 decile_analysis['Std Dev']).round(3)
    decile_analysis['Mean Return'] = decile_analysis['Mean Return'].map(lambda x: f"{x:.4%}")
    decile_analysis['Std Dev'] = decile_analysis['Std Dev'].map(lambda x: f"{x:.4f}")

    # print(decile_analysis)
    output(decile_analysis)

    # 3. STATISTICAL TESTS
    output("\n🧪 STATISTICAL SIGNIFICANCE TESTS")
    output("-" * 40)

    # T-test comparing top decile vs rest
    bottom_nine = df[df.Decile < 9]['Return']
    t_stat, p_val = stats.ttest_ind(top_decile_returns, bottom_nine, equal_var=False)

    output(f"Top Decile vs Others t-test:")
    output(f"  t-statistic: {t_stat:.3f}")
    output(f"  p-value: {p_val:.4f}")
    output(f"  Significant at 5%: {'Yes' if p_val < 0.05 else 'No'}")

    # 4. MONOTONICITY CHECK (VERY IMPORTANT!)
    # print("\n📈 MONOTONICITY CHECK")
    # print("-" * 40)
    output("\n📈 MONOTONICITY CHECK")
    output("-" * 40)
    # Get actual mean returns by decile (not as string)
    monotone_check = df.groupby('Decile')['Return'].mean()
    is_monotonic = monotone_check.is_monotonic_increasing
    spearman_r, spearman_p = stats.spearmanr(monotone_check.index, monotone_check.values)

    # print(f"Decile-Return Spearman Correlation: {spearman_r:.3f} (p-value: {spearman_p:.4f})")
    # print(f"Returns Monotonically Increasing with Decile: {'Yes' if is_monotonic else 'No'}")
    # print(f"Return Spread (D10-D1): {monotone_check.iloc[-1] - monotone_check.iloc[0]:.4%}")
    output(f"Decile-Return Spearman Correlation: {spearman_r:.3f} (p-value: {spearman_p:.4f})")
    output(f"Returns Monotonically Increasing with Decile: {'Yes' if is_monotonic else 'No'}")
    output(f"Return Spread (D10-D1): {monotone_check.iloc[-1] - monotone_check.iloc[0]:.4%}")

    output("\n" + "=" * 60)
    output("📋 CONTINUOUS SCORING MODEL SCORECARD")
    output("=" * 60)
    output(f"Overall Correlation: {optimization_results['mean_correlation']:.4f}")
    output(f"Top Decile Mean Return: {top_decile_returns.mean():.4%}")
    output(f"Top Decile Sharpe Ratio: {basic_metrics['sharpe']:.3f}")
    output(f"Hit Rate: {basic_metrics['hit_rate']:.2%}")
    output(f"Monotonicity (Spearman): {spearman_r:.3f}")
    output(f"Statistical Significance: {'Yes' if p_val < 0.05 else 'No'}")
    output(f"Total Observations: {len(df):,}")

    # 6. SIMPLE PORTFOLIO SIMULATION
    # print("\n📉 PORTFOLIO BACKTEST SIMULATION")
    # print("-" * 40)
    output("\n📉 PORTFOLIO BACKTEST SIMULATION")
    output("-" * 40)

    # Group by date and get top N stocks for each period
    def run_portfolio_backtest(n_stocks=20):
        # Reshape data to wide format with dates
        grouped_returns = {}
        for ticker in ticker_list:
            ticker_rets = []
            for date in dates:
                hist = history_cache.get(ticker, pd.DataFrame())
                if hist.empty:
                    continue

                # Ensure date is timezone-naive for comparison
                date_naive = date
                if hasattr(date, 'tz') and date.tz is not None:
                    date_naive = date.tz_localize(None)

                fwd = get_next_month_return(ticker, date_naive, history_cache)
                if fwd is not None:
                    ticker_rets.append((date_naive, fwd))

            if ticker_rets:
                grouped_returns[ticker] = dict(ticker_rets)

        # Create portfolio for each period
        portfolio_returns = []
        for date in dates[:-1]:  # Skip last date as we need forward returns
            # Calculate scores for this date
            date_scores = {}
            for ticker, returns_dict in grouped_returns.items():
                if date in returns_dict and ticker in analysis_df['Ticker'].values:
                    # Get score from analysis_df
                    ticker_row = analysis_df[analysis_df['Ticker'] == ticker]
                    if not ticker_row.empty:
                        date_scores[ticker] = (ticker_row['TotalScore'].values[0], returns_dict[date])

            # Select top N stocks
            if date_scores:
                sorted_scores = sorted(date_scores.items(), key=lambda x: x[1][0], reverse=True)
                top_n = sorted_scores[:n_stocks]
                if top_n:
                    # Equal-weighted portfolio return
                    port_return = sum(ret for _, (_, ret) in top_n) / len(top_n)
                    portfolio_returns.append(port_return)

        return portfolio_returns

    # Run backtest with top 20 stocks
    portfolio_returns = run_portfolio_backtest(20)

    if portfolio_returns:
        # Calculate portfolio statistics
        port_mean = np.mean(portfolio_returns)
        port_std = np.std(portfolio_returns)
        port_sharpe = port_mean / port_std if port_std > 0 else 0
        port_hit_rate = np.mean([r > 0 for r in portfolio_returns])

        output(f"Top 20 Portfolio Mean Monthly Return: {port_mean:.4%}")
        output(f"Portfolio Standard Deviation: {port_std:.4%}")
        output(f"Portfolio Sharpe Ratio: {port_sharpe:.3f}")
        output(f"Win Rate: {port_hit_rate:.2%}")
        output(f"Number of Periods: {len(portfolio_returns)}")
    else:
        # print("Insufficient data for portfolio backtesting")
        output("Insufficient data for portfolio backtesting")

    # 7. FACTOR EFFECTIVENESS ANALYSIS
    # print("\n🔍 FACTOR CONTRIBUTION ANALYSIS")
    # print("-" * 40)
    output("\n🔍 FACTOR CONTRIBUTION ANALYSIS")
    output("-" * 40)

    # Calculate single-factor portfolios
    factor_portfolios = {}
    for factor in factor_scores.keys():
        # Create DataFrame with factor score and returns
        factor_df = pd.DataFrame({
            'Score': analysis_df[factor],
            'Return': analysis_df['Return']
        }).dropna()

        # Create deciles
        if len(factor_df) > 10:  # Need at least 10 rows for 10 deciles
            factor_df["Decile"] = pd.qcut(factor_df.Score, 10, labels=False, duplicates="drop")

            # Calculate top decile return
            top_returns = factor_df[factor_df.Decile == 9]['Return']
            if not top_returns.empty:
                factor_portfolios[factor] = {
                    'mean': top_returns.mean(),
                    'sharpe': top_returns.mean() / top_returns.std() if top_returns.std() > 0 else 0,
                    'weight': WEIGHTS.get(factor, 0)
                }

    # Display factor effectiveness
    if factor_portfolios:
        # print("\nSingle-Factor Portfolio Performance:")
        output("\nSingle-Factor Portfolio Performance:")
        for factor, metrics in sorted(factor_portfolios.items(), key=lambda x: x[1]['mean'], reverse=True):
            # print(
            #     f"  {factor:<20} Mean: {metrics['mean']:.4%}  Sharpe: {metrics['sharpe']:.3f}  Weight: {metrics['weight']}")
            output(
                f"  {factor:<20} Mean: {metrics['mean']:.4%}  Sharpe: {metrics['sharpe']:.3f}  Weight: {metrics['weight']}")

        # Improved Weight Alignment Analysis with normalization to 100
        suggested_weights = {}
        raw_total = 0

        # First pass: calculate all suggested weights including zeros
        for factor in factor_scores.keys():
            current = WEIGHTS.get(factor, 0)

            # Get factor performance if available
            if factor in factor_portfolios:
                metrics = factor_portfolios[factor]
                # Calculate suggested weight based on factor performance
                suggested = current * (1 + metrics['mean'] * 5)  # Simple adjustment formula
            else:
                # If no performance data, keep current weight
                suggested = current

            suggested_weights[factor] = suggested
            raw_total += suggested

        # Second pass: normalize to sum=100
        normalized_weights = {}
        for factor, suggested in suggested_weights.items():
            normalized_weights[factor] = round(suggested * 100 / raw_total, 1)

        # Ensure exact sum of 100 by adjusting the largest weight
        sum_normalized = sum(normalized_weights.values())
        if sum_normalized != 100:
            # Find the factor with the largest absolute weight
            largest_factor = max(normalized_weights.keys(), key=lambda k: abs(normalized_weights[k]))
            normalized_weights[largest_factor] += (100 - sum_normalized)

        # Display both raw and normalized weights
        output("\nWeight Alignment Analysis:")
        output(f"Total Raw Weight: {raw_total:.1f}, Normalized to Sum=100")
        output("\nFactor                Current  Suggested  Normalized")
        output("-" * 50)

        for factor, current in sorted(WEIGHTS.items(), key=lambda x: abs(normalized_weights[x[0]]), reverse=True):
            suggested = suggested_weights.get(factor, 0)
            normalized = normalized_weights.get(factor, 0)
            output(f"{factor:<20} {current:8.1f} {suggested:10.1f} {normalized:10.1f}")

        # Create a dictionary of the normalized weights for potential usage
        new_weights = {k: float(v) for k, v in normalized_weights.items()}
        output("\nNew Normalized Weights Python Dict:")
        output("WEIGHTS = {")
        for factor, weight in sorted(new_weights.items(), key=lambda x: abs(x[1]), reverse=True):
            output(f"    \"{factor}\": {weight},")
        output("}")

    output(f"\n✅ Enhanced correlation analysis completed in {round(time.time() - start_time, 1)} seconds")
    # output(f"📊 Analysis used {CORRELATION_YEARS} years of data with exponential weighting")
    # output(f"💫 Recent data weighted {np.exp(np.log(2) * CORRELATION_YEARS):.1f}x more than oldest data")

    output(f"📊 Analysis used {len(regime_periods)} {SELECTED_REGIME} regime periods")
    # output(f"💫 Recent regime periods weighted more heavily with {EXPONENTIAL_DECAY_HALFLIFE}-day half-life")

    # Save enhanced results
    # save_analysis_results(output_buffer.getvalue(),
    #                       f"factor_analysis_results_{SELECTED_REGIME.replace(' ', '_').lower()}.txt",
    #                       results_dir=results_dir)

    # Sanitize the regime name for use in filename
    save_analysis_results(output_buffer.getvalue(),
                          f"factor_analysis_results_{sanitize_filename(SELECTED_REGIME).lower()}.txt",
                          results_dir=results_dir)

    # return {
    #     'correlations': correlations,
    #     'correlation_matrix': correlation_matrix,
    #     'weight_effectiveness': effectiveness_df,
    #     'analysis_df': analysis_df,
    #     'optimization_results': optimization_results,
    #     'methodology': {
    #         'correlation_period_years': CORRELATION_YEARS,
    #         'exponential_halflife_days': EXPONENTIAL_DECAY_HALFLIFE,
    #         'data_source': 'Marketstack API'
    #     }
    # }
    return {
        'correlations': correlations,
        'correlation_matrix': correlation_matrix,
        'weight_effectiveness': effectiveness_df,
        'analysis_df': analysis_df,
        'optimization_results': optimization_results,
        'methodology': {
            'correlation_period_years': CORRELATION_YEARS,
            'data_source': 'Marketstack API'
        }
    }


if __name__ == "__main__":
    # Create a config file if it doesn't exist
    if not os.path.exists("./Config/marketstack_config.json"):
        config = {
            "api_key": "476419cceb4330259e5a1267533xxxxx",
            "description": "Replace YOUR_API_KEY_HERE with your actual Marketstack API key"
        }
        with open("./Config/marketstack_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        print("📝 Created marketstack_config.json - please add your API key")

    main()