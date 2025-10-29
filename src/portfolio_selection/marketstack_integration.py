"""
Marketstack Integration Module for Portfolio Systems
Provides common data fetching functionality for both dynamic_portfolio_selector and Backtest_advanced
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import List, Dict, Tuple, Optional


class MarketstackDataFetcher:
    """
    Unified Marketstack API data fetcher for portfolio systems
    """

    def __init__(self, api_key: str):
        """
        Initialize with Marketstack API key

        Args:
            api_key: Your Marketstack API key
        """
        self.api_key = api_key
        self.base_url = "https://api.marketstack.com/v2"

    def fetch_prices(self,
                     symbols: List[str],
                     start_date: str,
                     end_date: str,
                     verbose: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Fetch EOD price data from Marketstack API

        Args:
            symbols: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            verbose: Whether to print progress

        Returns:
            Tuple of (prices DataFrame, list of failed tickers)
        """
        if verbose:
            print(f"Fetching data from Marketstack API for {len(symbols)} symbols...")

        all_price_data = pd.DataFrame()
        failed_tickers = []

        # Fetch each symbol individually
        for i, symbol in enumerate(symbols):
            if verbose and (i + 1) % 5 == 0:
                print(f"Progress: {i + 1}/{len(symbols)} symbols fetched...")

            try:
                params = {
                    'access_key': self.api_key,
                    'symbols': symbol,
                    'date_from': start_date,
                    'date_to': end_date,
                    'limit': 1000,
                    'sort': 'ASC'
                }

                endpoint = f"{self.base_url}/eod"
                symbol_data = []
                offset = 0

                while True:
                    params['offset'] = offset

                    try:
                        response = requests.get(endpoint, params=params, timeout=30)
                        response.raise_for_status()

                        data = response.json()

                        if 'error' in data:
                            if verbose:
                                print(f"   API Error for {symbol}: {data['error'].get('message', 'Unknown error')}")
                            failed_tickers.append(symbol)
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
                        if verbose:
                            print(f"   API Request Error for {symbol}: {e}")
                        failed_tickers.append(symbol)
                        break

                # Process symbol data
                if symbol_data:
                    df = pd.DataFrame(symbol_data)
                    df['date'] = pd.to_datetime(df['date'])

                    # Remove timezone info
                    if df['date'].dt.tz is not None:
                        df['date'] = df['date'].dt.tz_localize(None)

                    # Handle both 'adj_close' and 'close' columns
                    price_col = 'adj_close' if 'adj_close' in df.columns else 'close'
                    df = df.set_index('date')[price_col]
                    df.name = symbol

                    # Add to main dataframe
                    if all_price_data.empty:
                        all_price_data = pd.DataFrame(df)
                    else:
                        all_price_data = all_price_data.join(df, how='outer')
                else:
                    if symbol not in failed_tickers:
                        failed_tickers.append(symbol)

            except Exception as e:
                if verbose:
                    print(f"   Error fetching {symbol}: {e}")
                if symbol not in failed_tickers:
                    failed_tickers.append(symbol)

        # Sort and clean
        if not all_price_data.empty:
            all_price_data = all_price_data.sort_index()

            # Forward fill then backward fill
            all_price_data = all_price_data.ffill().bfill()

            # Check for remaining NaNs
            nan_pct = all_price_data.isna().mean().mean() * 100
            if nan_pct > 0 and verbose:
                print(f" Data contains {nan_pct:.2f}% NaN values after fill")

        if verbose:
            if len(all_price_data) > 0:
                print(f" Successfully fetched data for {len(all_price_data.columns)} symbols")
                print(f"Date range: {all_price_data.index[0]} to {all_price_data.index[-1]}")
                print(f"Total data points: {len(all_price_data)}")
            else:
                print(" No data was fetched")

        return all_price_data, list(set(failed_tickers))

    def fetch_prices_with_lookback(self,
                                   symbols: List[str],
                                   start_date: str,
                                   end_date: Optional[str] = None,
                                   lookback_days: int = 252,
                                   verbose: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Fetch price data with additional lookback period

        Args:
            symbols: List of ticker symbols
            start_date: Start date for actual data
            end_date: End date (None for today)
            lookback_days: Additional days of historical data
            verbose: Whether to print progress

        Returns:
            Tuple of (prices DataFrame, list of failed tickers)
        """
        # Calculate extended start date
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        lookback_start_dt = start_dt - timedelta(days=int(lookback_days * 1.5))
        lookback_start = lookback_start_dt.strftime("%Y-%m-%d")

        # Determine end date
        if end_date is None:
            end = datetime.now().strftime("%Y-%m-%d")
        else:
            end = end_date

        if verbose:
            print(f"Fetching price data from {lookback_start} to {end}")
            print(f"This includes {lookback_days} days of lookback data")

        return self.fetch_prices(symbols, lookback_start, end, verbose)

    def validate_data_availability(self,
                                   tickers: List[str],
                                   start_date: str,
                                   lookback_days: int,
                                   verbose: bool = True) -> Dict[str, bool]:
        """
        Check if tickers have sufficient historical data

        Args:
            tickers: List of ticker symbols
            start_date: Start date for backtesting
            lookback_days: Required lookback period
            verbose: Whether to print progress

        Returns:
            Dictionary mapping ticker to availability status
        """
        start_dt = pd.to_datetime(start_date)
        required_start = start_dt - pd.DateOffset(days=lookback_days + 30)

        if verbose:
            print(f"\n Validating data availability for {len(tickers)} tickers...")
            print(f"   Required data from: {required_start.strftime('%Y-%m-%d')} to {start_date}")

        availability = {}
        insufficient_tickers = []

        # Test in batches to be more efficient
        batch_size = 20
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]

            if verbose and i > 0:
                print(f"   Checking batch {i // batch_size + 1}/{(len(tickers) + batch_size - 1) // batch_size}...")

            try:
                data, failed = self.fetch_prices(
                    batch,
                    required_start.strftime('%Y-%m-%d'),
                    start_dt.strftime('%Y-%m-%d'),
                    verbose=False
                )

                for ticker in batch:
                    if ticker in failed:
                        availability[ticker] = False
                        insufficient_tickers.append(ticker)
                    elif ticker in data.columns:
                        has_sufficient_data = len(data[ticker].dropna()) >= lookback_days * 0.8
                        availability[ticker] = has_sufficient_data
                        if not has_sufficient_data:
                            insufficient_tickers.append(ticker)
                    else:
                        availability[ticker] = False
                        insufficient_tickers.append(ticker)

            except Exception as e:
                if verbose:
                    print(f"    Error checking batch: {e}")
                for ticker in batch:
                    availability[ticker] = False
                    insufficient_tickers.append(ticker)

        if insufficient_tickers and verbose:
            print(
                f"    {len(insufficient_tickers)} tickers have insufficient data: {', '.join(insufficient_tickers[:5])}")
            if len(insufficient_tickers) > 5:
                print(f"      ... and {len(insufficient_tickers) - 5} more")

        return availability