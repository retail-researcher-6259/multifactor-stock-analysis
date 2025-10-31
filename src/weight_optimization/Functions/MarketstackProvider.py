# Modified sections for Marketstack integration
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random

# Add this at the top of your script after the imports
MARKETSTACK_API_KEY = "aaaaa"  # Replace xxx with your actual last 3 digits
MARKETSTACK_BASE_URL = "http://api.marketstack.com/v2/"


class MarketstackProvider:
    """
    Marketstack API provider for historical stock data
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = MARKETSTACK_BASE_URL
        self.session = requests.Session()

    def get_historical_data(self, symbol: str, years: int = 10, limit: int = 1000) -> pd.DataFrame:
        """
        Fetch historical data from Marketstack

        Args:
            symbol: Stock ticker symbol
            years: Number of years of historical data
            limit: Maximum number of data points (Marketstack limit is 1000 for free plan)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * years)

            # Marketstack parameters
            params = {
                'access_key': self.api_key,
                'symbols': symbol,
                'date_from': start_date.strftime('%Y-%m-%d'),
                'date_to': end_date.strftime('%Y-%m-%d'),
                'limit': limit,
                'sort': 'ASC'  # Oldest first
            }

            # Make API request
            response = self.session.get(f"{self.base_url}eod", params=params)

            if response.status_code != 200:
                print(f"❌ Marketstack API error for {symbol}: {response.status_code}")
                return pd.DataFrame()

            data = response.json()

            # Check if we have data
            if 'data' not in data or not data['data']:
                print(f"⚠️ No data returned for {symbol}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(data['data'])

            # Ensure we have the required columns
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                print(f"⚠️ Missing required columns for {symbol}")
                return pd.DataFrame()

            # Process the data to match yahooquery format
            df['date'] = pd.to_datetime(df['date'])

            # IMPORTANT: Convert to timezone-naive to match the rest of the script
            if df['date'].dt.tz is not None:
                df['date'] = df['date'].dt.tz_localize(None)

            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)  # Ensure chronological order

            # Rename columns to match Yahoo format (capitalized)
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })

            # Convert to numeric
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove any rows with NaN values
            df = df.dropna()

            print(f"✅ Downloaded {len(df)} days of data for {symbol}")
            return df

        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def get_bulk_historical_data(self, symbols: list, years: int = 5) -> dict:
        """
        Fetch historical data for multiple symbols with rate limiting

        Args:
            symbols: List of ticker symbols
            years: Number of years of historical data

        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        all_data = {}

        for i, symbol in enumerate(symbols):
            print(f"📊 Fetching data for {symbol} ({i + 1}/{len(symbols)})")

            df = self.get_historical_data(symbol, years)
            if not df.empty:
                all_data[symbol] = df

            # Rate limiting - Marketstack allows 1000 requests/month on free plan
            # Be conservative with 2-3 second delays
            if i < len(symbols) - 1:
                time.sleep(random.uniform(2, 3))

        return all_data


# Replace the yahooquery_hist_fixed function with this:
def marketstack_hist_fixed(ticker, years=5):
    """
    Fetch historical data using Marketstack API
    Replacement for yahooquery_hist_fixed function
    """
    try:
        # Initialize Marketstack provider
        provider = MarketstackProvider(MARKETSTACK_API_KEY)

        # Get historical data
        df = provider.get_historical_data(ticker, years)

        if df.empty:
            print(f"⚠️ No data available for {ticker}")
            return pd.DataFrame()

        # Ensure we have enough data
        if len(df) < 100:  # Minimum threshold
            print(f"⚠️ Insufficient data for {ticker}: only {len(df)} days")
            return pd.DataFrame()

        return df

    except Exception as e:
        print(f"⚠️ Error fetching history for {ticker}: {e}")
        return pd.DataFrame()


# Modified main function section - replace the history fetching part:
def main_modified():
    """
    Modified main function with Marketstack integration
    Replace the relevant sections in your main() function with this code
    """

    # ... (keep your existing code until the data fetching section)

    # -------- MODIFIED SECTION: Replace the history fetching loop --------
    print("📦 Downloading data from Marketstack and calculating factor scores...")

    # Initialize Marketstack provider
    marketstack_provider = MarketstackProvider(MARKETSTACK_API_KEY)

    fundamentals_cache = {}
    history_cache = {}
    raw = []
    pm_cache, em_cache, mcaps = [], [], []
    value_metrics_by_ticker = {}

    for i, tk in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] Processing {tk} …")
        try:
            # Get fundamentals from Yahoo (still needed for fundamental analysis)
            info = safe_get_info(tk)
            fundamentals_cache[tk] = info

            # Get price history from Marketstack (NEW!)
            hist = marketstack_hist_fixed(tk, years=5)

            if hist.empty or len(hist) < 200:
                print(f"⚠️ Insufficient data for {tk}, skipping...")
                continue

            history_cache[tk] = hist

            # --- Rest of your factor calculations remain the same ---
            value, quality, value_dict = get_fundamentals(tk)
            value_metrics_by_ticker[tk] = value_dict

            tech = get_technical_score(hist)
            insider = get_insider_score_simple(tk)

            rev_g, roic = get_growth_roic(tk)

            # ─ momentum raw numbers
            pm, em = get_momentum_score(tk, hist)
            pm_cache.append(pm)
            em_cache.append(em)

            # ─ downside-risk penalty
            market_hist = history_cache.get('SPY', None)
            penalty = get_risk_penalty(hist, market_hist)

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
                Ticker=tk, Value=value, Quality=quality, Technical=tech,
                Insider=insider, PriceMom=pm, EarnMom=em, Penalty=penalty,
                MarketCap=mcap, Credit=credit, Liquidity=liq, Carry=carry,
                Sector=sector, Industry=industry, RevG=rev_g, ROIC=roic,
                CompanyName=company_name, Country=country
            ))

            # Increased delay to respect API limits
            time.sleep(random.uniform(2, 4))

        except Exception as e:
            print(f"⚠️ Error processing {tk}: {e}")
            continue

    # ... (rest of your analysis code remains the same)


# Configuration update for your script
def update_config():
    """
    Add Marketstack configuration to your script
    """
    print("📝 Configuration Updates:")
    print("1. Add your Marketstack API key at the top of the script")
    print("2. Replace 'yahooquery_hist_fixed' calls with 'marketstack_hist_fixed'")
    print("3. The script will now use:")
    print("   - Marketstack: Historical price data (OHLCV)")
    print("   - Yahoo Finance: Fundamental data (P/E, financials, etc.)")
    print("4. Expect slower execution due to API rate limits")


# Error handling and diagnostics
def test_marketstack_connection():
    """
    Test your Marketstack API connection
    """
    try:
        provider = MarketstackProvider(MARKETSTACK_API_KEY)
        test_data = provider.get_historical_data("AAPL", years=1)

        if not test_data.empty:
            print("✅ Marketstack API connection successful!")
            print(f"✅ Retrieved {len(test_data)} days of AAPL data")
            print(f"✅ Date range: {test_data.index[0].date()} to {test_data.index[-1].date()}")
            return True
        else:
            print("❌ Marketstack API connection failed - no data returned")
            return False

    except Exception as e:
        print(f"❌ Marketstack API connection failed: {e}")
        return False


# Usage instructions
if __name__ == "__main__":
    print("🔧 MARKETSTACK INTEGRATION SETUP")
    print("=" * 50)

    print("\n1. Testing API connection...")
    if test_marketstack_connection():
        print("\n2. Ready to integrate into your main script!")
        print("\nNext steps:")
        print("- Replace 'yahooquery_hist_fixed' with 'marketstack_hist_fixed' in your main script")
        print("- Update your API key in the MARKETSTACK_API_KEY variable")
        print("- Run your analysis as normal")

        print("\n⚠️ Important Notes:")
        print("- Your free plan allows 1000 API calls per month")
        print("- With rate limiting, expect ~2-3 seconds per ticker")
        print("- Historical data limited to ~3 years with free plan")
        print("- Fundamental analysis still uses Yahoo Finance")
    else:
        print("\n❌ Please check your API key and try again")