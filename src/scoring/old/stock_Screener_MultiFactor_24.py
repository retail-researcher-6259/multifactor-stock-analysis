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

pd.options.mode.chained_assignment = None  # disable copy-view warning

# --- Configurable Parameters ---
FRED_API_KEY = "a6472bcc951dc72f091984a09a36fc9e"
FRED_SERIES_ID = "GDP"  # U.S. Real GDP, quarterly

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd().parent.parent.parent
# PROJECT_ROOT = Path(__file__).parent.parent  # Go up two levels from src/scoring/
TICKER_FILE = PROJECT_ROOT / "config" / "Buyable_stocks_test.txt"

INSIDER_LOOKBACK_DAYS = 90
TOP_N = 20
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# Scoring Weights (keeping proven optimal configuration)
# Total score: 120

REGIME_WEIGHTS = {
        "Steady_Growth": {
        "credit": 48.6,
        "quality": 18.1,
        "momentum": 18.1,
        "financial_health": 14.4,
        "growth": 10.9,
        "value": -7.3,
        "technical": 3.8,
        "liquidity": 3.6,
        "carry": -3.6,
        "stability": -3.0,
        "size": -2.2,
        "insider": -1.4,
    },
    "Strong_Bull": {
        # Placeholder weights - to be optimized later
        "credit": 40.0,
        "quality": 20.0,
        "momentum": 25.0,
        "financial_health": 15.0,
        "growth": 15.0,
        "value": -5.0,
        "technical": 8.0,
        "liquidity": 5.0,
        "carry": -2.0,
        "stability": -5.0,
        "size": 0.0,
        "insider": 2.0,
    },
    "Crisis_Bear": {
        "momentum": 72.0,      # Slight increase from 70.7
        "size": 20.0,          # Rounded up from 17.6
        "financial_health": 13.0,  # Reduced to minimize drag
        "credit": 0.0,         # Eliminated (negative correlation)
        "insider": 5.0,        # Rounded up from 4.4
        "growth": 5.0,         # Reduced from 8.8 (negative correlation)
        "quality": 0.0,        # Keep at zero
        "liquidity": 5.0,      # Add for positive correlation
        "value": 0.0,          # Keep at zero
        "technical": -5.0,     # Unchanged
        "carry": -5.0,         # Rounded from -4.4
        "stability": -10.0,    # Rounded from -10.3
    }
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
            print(f"⚠️ Attempt {attempt + 1}/{retries} failed for {ticker}: {e}")
            time.sleep(random.uniform(*delay_range))
    print(f"❌ All attempts failed for {ticker}")
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


def yahooquery_hist(ticker, years=5):
    """Maximally robust version that handles any timezone issues"""
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

            final_df = result_df.dropna().sort_index()

            # DEBUG: Final result
            print(f"    → Final processed data: {len(final_df)} clean records")

            return final_df

        except Exception as e:
            print(f"Data processing failed for {ticker}: {e}")
            return pd.DataFrame()

    except Exception as e:
        print(f"⚠️ Error fetching history for {ticker}: {e}")
        return pd.DataFrame()

# --- Helper Functions ---
def sector_industry(info):
    """Return (sector, industry) strings or 'Unknown' placeholders."""
    return (
        info.get("sector") or info.get("sectorDisp") or "Unknown",
        info.get("industry") or info.get("industryDisp") or "Unknown"
    )


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


# --- 1-a  price-momentum (12-mo total return, skip last 21 trading days) ------
def _price_mom(df):
    """
    Classic US-equity '12-1' momentum:
       r = (P[t-21] / P[t-252]) – 1
    Then z-score across the universe later & squash to 0-1 with logistic.
    Here we only compute the raw return; scaling happens in main().

    Fix:
    Simple 12-month momentum (no 1-month skip):
        r = (P[t] / P[t-252]) – 1
    """
    if len(df) < 180:
        return None
    ret = (df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1
    return ret


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


def get_momentum_score(ticker, df):
    """
    Returns raw 12-1 price momentum and raw earnings-momentum (%),
    or np.nan when not available.  No side-effects.
    """
    pm = _price_mom(df)
    em = _earnings_mom(ticker)
    # convert None → np.nan so zscore can ignore them
    return (pm if pm is not None else np.nan,
            em if em is not None else np.nan)

# def get_momentum_score(ticker, df):
#     """
#     Enhanced momentum score that evaluates magnitude, consistency,
#     and acceleration across multiple timeframes.
#     """
#     if len(df) < 252:  # Need at least 1 year of data
#         return 0.5, None  # Return a neutral score instead of np.nan
#
#     # Price momentum components
#     try:
#         # 1. Multiple timeframe momentum magnitude
#         periods = [21, 63, 126, 252]  # 1, 3, 6, 12 months
#         weights = [0.1, 0.3, 0.3, 0.3]  # More weight to medium-term
#
#         momentum_scores = []
#         for period in periods:
#             if len(df) >= period:
#                 ret = df['Close'].iloc[-1] / df['Close'].iloc[-period] - 1
#                 # Transform to 0-1 scale (-20% to +40% → 0 to 1)
#                 norm_ret = (ret + 0.2) / 0.6
#                 momentum_scores.append(np.clip(norm_ret, 0, 1))
#
#         # 2. Momentum consistency
#         daily_returns = df['Close'].pct_change().dropna()
#         # Ratio of up days to total days in last 3 months
#         if len(daily_returns) >= 63:
#             up_ratio = np.sum(daily_returns.iloc[-63:] > 0) / 63
#             consistency_score = up_ratio  # Already 0-1
#         else:
#             consistency_score = 0.5  # Neutral
#
#         # 3. Momentum acceleration
#         if len(df) >= 252:
#             ret_3m = df['Close'].iloc[-1] / df['Close'].iloc[-63] - 1
#             ret_6m = df['Close'].iloc[-63] / df['Close'].iloc[-126] - 1
#             acceleration = ret_3m - ret_6m
#             # Transform to 0-1 scale (-15% to +15% → 0 to 1)
#             accel_score = np.clip((acceleration + 0.15) / 0.3, 0, 1)
#         else:
#             accel_score = 0.5  # Neutral
#
#         # Weighted price momentum
#         weighted_magnitude = sum(w * s for w, s in zip(weights, momentum_scores)) if momentum_scores else 0.5
#         pm_score = 0.6 * weighted_magnitude + 0.2 * consistency_score + 0.2 * accel_score
#
#         # Get earnings momentum
#         em = _earnings_mom(ticker)
#
#         # return pm_score, em
#         # Ensure we return a float for pm_score and a float/np.nan for em
#         return float(pm_score), float(em) if em is not None else np.nan
#
#     except Exception as e:
#         print(f"⚠️ Momentum calculation error for {ticker}: {e}")
#         return 0.5, None  # Return neutral score on error


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

# --- Main Logic ---
def main(regime="Steady_Growth", progress_callback=None):
    """
    Main scoring function with progress tracking

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

    print(f"🎯 Running multifactor scoring for regime: {regime}")
    print(f"📊 Using weights: {WEIGHTS}")

    if _progress_tracker:
        _progress_tracker.callback('status', f"Loading tickers for {regime} regime...")
        _progress_tracker.callback('progress', 5)

    # -------- 0.  load tickers -------------------------------------------------
    try:
        with open(TICKER_FILE, "r") as f:
            tickers = [ln.strip().replace("$", "") for ln in f if ln.strip()]
    except FileNotFoundError:
        error_msg = f"⛔ Ticker file '{TICKER_FILE}' not found!"
        print(error_msg)
        if _progress_tracker:
            _progress_tracker.callback('error', error_msg)
        return

    if len(tickers) < 5:
        warning_msg = f"⚠️ Warning: Only {len(tickers)} tickers found"
        print(warning_msg)
        if _progress_tracker:
            _progress_tracker.callback('status', warning_msg)

    print(f"✅ Loaded {len(tickers)} tickers from file.")

    if _progress_tracker:
        _progress_tracker.set_total(len(tickers))
        _progress_tracker.callback('status', f"Analyzing {len(tickers)} tickers...")
        _progress_tracker.callback('progress', 10)

    # -------- 1-a. pre-scan: avgVol for liquidity -----------------------------
    avg_vol = {}
    if _progress_tracker:
        _progress_tracker.callback('status', "Pre-scanning for liquidity data...")

    for tk in tickers:
        try:
            avg_vol[tk] = fast_info(tk).get("averageVolume10days", 0)
        except Exception:
            avg_vol[tk] = 0
    vol_min, vol_max = min(avg_vol.values()), max(avg_vol.values())

    if _progress_tracker:
        _progress_tracker.callback('progress', 15)

    # -------- 1-b. main data harvest ------------------------------------------
    raw = []
    pm_cache, em_cache, mcaps = [], [], []
    value_metrics_by_ticker = {}

    processed_tickers = []
    skipped_tickers = []

    for i, tk in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] scraping {tk} …")

        # Update progress tracker
        if _progress_tracker:
            _progress_tracker.update_progress(tk, "processing")

        try:
            info = safe_get_info(tk)
            print(f"  ✓ Info fetched for {tk}: MarketCap={info.get('marketCap', 'N/A')}")

            hist = yahooquery_hist(tk, years=1)

            if hist.empty or len(hist) < 150:
                if hist.empty:
                    print(f"  ⚠️ SKIPPING {tk}: Historical data is empty")
                else:
                    print(f"  ⚠️ SKIPPING {tk}: Insufficient data points ({len(hist)} < 150)")
                skipped_tickers.append((tk, f"Empty" if hist.empty else f"Only {len(hist)} points"))

                # Still update progress for skipped tickers
                if _progress_tracker:
                    _progress_tracker.update_progress(tk, "completed")
                continue

            # Get all the scores
            value, quality, financial_health, roic, value_dict = get_fundamentals(tk)
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
            credit = get_credit_score(info)
            carry = get_carry_score_simple(tk, info)

            liq = 0
            if vol_max != vol_min:
                liq = round((avg_vol[tk] - vol_min) / (vol_max - vol_min), 2)

            raw.append(dict(
                Ticker=tk, Value=value, Quality=quality,
                FinancialHealth=financial_health,
                Technical=tech, Insider=insider, PriceMom=pm, EarnMom=em,
                Stability=stability,
                MarketCap=mcap, Credit=credit,
                Liquidity=liq, Carry=carry, Sector=sector, Industry=industry,
                Growth=get_growth_score(tk, info),
                ROIC=roic,
                CompanyName=company_name, Country=country
            ))

            print(f"  ✅ {tk} successfully added to results")
            processed_tickers.append(tk)

            # Update progress tracker for completed ticker
            if _progress_tracker:
                _progress_tracker.update_progress(tk, "completed")

        except Exception as e:
            print(f"⚠️ {tk}: {e}")
            # Update progress even for failed tickers
            if _progress_tracker:
                _progress_tracker.update_progress(tk, "completed")

    # Debug summary
    print("\n" + "=" * 60)
    print(f"PROCESSING SUMMARY:")
    print(f"Total tickers: {len(tickers)}")
    print(f"Successfully processed: {len(processed_tickers)} - {processed_tickers}")
    print(f"Skipped: {len(skipped_tickers)}")
    for tk, reason in skipped_tickers:
        print(f"  - {tk}: {reason}")
    print("=" * 60 + "\n")

    if not raw:
        error_msg = "⚠️ No usable tickers"
        print(error_msg)
        if _progress_tracker:
            _progress_tracker.callback('error', error_msg)
        return

    # -------- Final processing phase (10% of progress) ----------------------
    if _progress_tracker:
        _progress_tracker.callback('progress', 90)
        _progress_tracker.callback('status', "Finalizing scores and rankings...")

    # -------- 2.  cross-sectional normalisations ------------------------------
    # 2-a momentum z-scores → logistic 0-1
    pm_arr = np.asarray(pm_cache)
    em_arr = np.asarray(em_cache)

    def safe_z(a):
        """
        Calculate z-scores with proper handling of NaN values and sparse data.
        Returns z-scores of input array with safety checks.
        """
        # Convert array to float, explicitly handling non-numeric values
        a_float = np.array([np.nan if x is None or not isinstance(x, (int, float)) else x for x in a], dtype=np.float64)
        finite = np.isfinite(a_float)
        if finite.sum() <= 2:  # 0 or 1 valid numbers
            return np.zeros_like(a_float, dtype=float)
        return zscore(a_float, nan_policy='omit')

    pm_z = safe_z(pm_arr)
    em_z = safe_z(em_arr)
    squash = lambda z: 1 / (1 + np.exp(-np.clip(z, -2, 2)))

    # Create DataFrame and apply growth adjustment
    df_raw = pd.DataFrame(raw)
    df_raw = apply_sector_relative_growth_adjustment(df_raw)

    # 2-b size-percentile breakpoints
    mcaps_clean = [m for m in mcaps if m]
    mc_min = np.nanmin(mcaps_clean)
    mc_80th = np.nanpercentile(mcaps_clean, 80)

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


    # -------- 3.  final scoring ticker-by-ticker ------------------------------
    results = []
    for k, rec in df_raw.iterrows():  # iterate over DataFrame rows
        # For price momentum, apply z-score normalization and sigmoid
        pm_val = pm_z[k] if k < len(pm_z) and not np.isnan(pm_z[k]) else 0
        pm_score = squash(pm_val)  # Apply sigmoid to normalize to 0-1

        # For earnings momentum, apply z-score normalization and sigmoid
        em_val = em_z[k] if k < len(em_z) and not np.isnan(em_z[k]) else 0
        em_score = squash(em_val)  # Apply sigmoid to normalize to 0-1

        # Combine with same weights as optimizer (60% price, 40% earnings)
        momentum = round(0.6 * pm_score + 0.4 * em_score, 3)

        size = get_size_score({'marketCap': rec['MarketCap']},
                              mc_min, mc_80th)

        total = (
                rec['ValueIndustryAdj'] * WEIGHTS['value'] +
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
                rec['GrowthScore_Norm'] * WEIGHTS['growth']  # Using enhanced growth
        )

        # UPDATE THE RESULTS DICTIONARY:
        results.append({
            "Ticker": rec['Ticker'],
            "CompanyName": rec['CompanyName'],
            "Country": rec['Country'],
            "Score": round(total, 2),
            "Value": round(rec['ValueIndustryAdj'], 3),
            "Quality": rec['Quality'],
            "FinancialHealth": round(rec['FinancialHealth'], 3),  # NEW
            "Technical": rec['Technical'],
            "Insider": rec['Insider'],
            "Momentum": momentum,
            "Stability": rec['Stability'],  # Changed from "VolPenalty": rec['Penalty']
            "Size": size,
            "Credit": rec['Credit'],
            "Liquidity": rec['Liquidity'],
            "Carry": rec['Carry'],
            "Growth": round(rec.get('GrowthScore_Norm', 0), 3),  # Enhanced growth
            # "ROICAdj": round(rec.ROICAdj, 3),  # REMOVE THIS
            "Sector": rec.Sector,
            "Industry": rec.Industry
        })

    # -------- 4.  output ------------------------------------------------------
    if _progress_tracker:
        _progress_tracker.callback('progress', 95)
        _progress_tracker.callback('status', "Generating output file...")

    df = pd.DataFrame(results).sort_values("Score", ascending=False)

    # Get output directory for the regime
    output_dir = get_output_directory(regime)
    today = datetime.now().strftime("%m%d")  # e.g. 0503
    fname = f"top_ranked_stocks_{regime}_{today}.csv"
    output_path = output_dir / fname

    # Save the file
    df.to_csv(output_path, index=False)
    print(f"📁 Saved {fname} to {output_dir}")
    print(f"📈 Top {TOP_N} stocks for {regime} regime:")
    print(df.head(TOP_N))
    print(f"✅ Finished in {round(time.time() - start_time, 1)} seconds")

    if _progress_tracker:
        _progress_tracker.callback('progress', 100)
        _progress_tracker.callback('status', f"Complete! Analyzed {len(processed_tickers)} stocks.")

    return output_path, df

# Add this function to be called from the UI
def run_scoring_for_regime(regime_name, progress_callback=None):
    """Entry point for UI to run scoring for a specific regime with progress tracking"""
    try:
        output_path, df = main(regime_name, progress_callback)
        return {
            'success': True,
            'output_path': str(output_path),
            'results_df': df,
            'regime': regime_name,
            'total_stocks': len(df),
            'weights': get_regime_weights(regime_name)
        }
    except Exception as e:
        if progress_callback:
            progress_callback('error', str(e))
        return {
            'success': False,
            'error': str(e),
            'regime': regime_name
        }

if __name__ == "__main__":
    main()