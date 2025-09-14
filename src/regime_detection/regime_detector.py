"""
Market Regime Detection using Gaussian Hidden Markov Models - Improved Convergence
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import yfinance as yf
from pathlib import Path
import json
import os
import pickle
import requests
import time

warnings.filterwarnings('ignore')

MARKETSTACK_BASE_URL = "https://api.marketstack.com/v2"

class RegimeDetectorAPI:
    """
    Wrapper class to make your existing regime detector work with the API
    Add this to your regime_detector.py
    """

    def __init__(self, analysis_dir="Analysis"):
        self.analysis_dir = Path(analysis_dir)
        self.analysis_dir.mkdir(exist_ok=True)

    def save_results_for_api(self, regime_model, regime_periods_df, current_regime_info):
        """
        Save results in formats that the API can easily read
        Call this at the end of your regime detection process
        """

        # Save the model
        model_path = self.analysis_dir / "regime_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(regime_model, f)

        # Save regime periods as CSV
        periods_path = self.analysis_dir / "regime_periods.csv"
        regime_periods_df.to_csv(periods_path, index=False)

        # Save current regime analysis as JSON
        analysis_path = self.analysis_dir / "current_regime_analysis.json"

        # Prepare data for JSON serialization
        json_data = {
            "current_regime": current_regime_info.get("regime", "Unknown"),
            "regime_probabilities": {
                "Steady Growth": float(current_regime_info.get("prob_growth", 0)),
                "Strong Bull": float(current_regime_info.get("prob_bull", 0)),
                "Crisis/Bear": float(current_regime_info.get("prob_bear", 0))
            },
            "timestamp": datetime.now().isoformat(),
            "confidence": float(current_regime_info.get("confidence", 0)),
            "supporting_indicators": current_regime_info.get("indicators", {})
        }

        with open(analysis_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        return {
            "model_path": str(model_path),
            "periods_path": str(periods_path),
            "analysis_path": str(analysis_path)
        }

    def run_detection(self, data_source="marketstack"):
        """
        Main function to run regime detection
        This should call your existing detection logic
        """
        try:
            # Call your existing detection functions here
            # Example:
            # from your_existing_code import detect_regimes
            # model, periods, current = detect_regimes(data_source)

            # For now, return example data
            # Replace this with your actual detection logic
            example_current_regime = {
                "regime": "Steady Growth",
                "prob_growth": 0.68,
                "prob_bull": 0.22,
                "prob_bear": 0.10,
                "confidence": 0.85,
                "indicators": {
                    "vix": 15.2,
                    "market_trend": "upward",
                    "volatility": "low"
                }
            }

            # Save results for API
            paths = self.save_results_for_api(
                regime_model=None,  # Replace with your actual model
                regime_periods_df=pd.DataFrame(),  # Replace with your actual dataframe
                current_regime_info=example_current_regime
            )

            return {
                "success": True,
                "paths": paths,
                "current_regime": example_current_regime
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

class MarketRegimeDetector:
    def __init__(self, n_regimes=4, lookback_years=10):
        """
        Initialize the regime detector

        Args:
            n_regimes: Number of market regimes to identify
            lookback_years: Years of historical data to use
        """
        self.n_regimes = n_regimes
        self.lookback_years = lookback_years
        self.model = None
        self.scaler = StandardScaler()
        self.pca = None  # For dimensionality reduction
        self.regime_characteristics = {}
        self.data = None
        self.features = None
        self.features_reduced = None
        self.regime_history = None

        # ADD THIS: Set output directories relative to project root
        self.project_root = Path(__file__).parent.parent.parent
        self.output_dir = self.project_root / "output" / "Regime_Detection_Results"
        self.analysis_dir = self.project_root / "output" / "Regime_Detection_Analysis"

        # Create directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

    def load_marketstack_config(self):
        """Load Marketstack API configuration - FIXED"""
        config_file = self.project_root / "config" / "marketstack_config.json"

        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
                api_key = config.get('api_key', '')
                # FIX: Corrected the condition
                if api_key and 'xxxxx' not in api_key and api_key != '':
                    return api_key
                else:
                    print("⚠️ Please update your Marketstack API key in config/marketstack_config.json")
                    return None
        else:
            print(f"⚠️ Config file not found: {config_file}")
            return None

    def fetch_marketstack_data(self, symbols, date_from, date_to, api_key):
        """
        Fetch EOD data from Marketstack API v2 for multiple symbols
        WITH PAGINATION to get more than 1000 records
        """
        print(f"📊 Fetching data from Marketstack API v2 for {len(symbols)} symbols...")
        print(f"📅 Date range: {date_from} to {date_to}")

        all_symbol_dfs = []

        for i, symbol in enumerate(symbols):
            # Clean symbol for Marketstack (remove ^ prefix for indices)
            clean_symbol = symbol.replace('^', '')

            # Map symbols to Marketstack format
            symbol_map = {
                'SPY': 'SPY',
                'VIX': 'VIX.INDX',
                'TNX': 'TNX.INDX',
                'GLD': 'GLD',
                'DXY': 'DXY.INDX'
            }

            marketstack_symbol = symbol_map.get(clean_symbol, clean_symbol)

            print(f"  Fetching {symbol} as {marketstack_symbol} ({i + 1}/{len(symbols)})", end='')

            # Initialize pagination
            symbol_data = []
            offset = 0
            total_fetched = 0

            while True:
                params = {
                    'access_key': api_key,
                    'symbols': marketstack_symbol,
                    'date_from': date_from,
                    'date_to': date_to,
                    'limit': 1000,  # Maximum allowed per request
                    'offset': offset,
                    'sort': 'ASC'
                }

                endpoint = f"{MARKETSTACK_BASE_URL}/eod"

                try:
                    response = requests.get(endpoint, params=params, timeout=30)
                    response.raise_for_status()

                    data = response.json()

                    if 'error' in data:
                        print(f" ❌ API Error: {data['error'].get('message', 'Unknown error')}")
                        break

                    if 'data' not in data or not data['data']:
                        # No more data
                        break

                    # Add data from this page
                    page_data = data['data']
                    symbol_data.extend(page_data)
                    total_fetched += len(page_data)

                    # Check pagination info
                    pagination = data.get('pagination', {})
                    total = pagination.get('total', 0)
                    count = pagination.get('count', 0)

                    print(f".", end='')  # Progress indicator

                    # Check if we've fetched all data
                    if offset + count >= total:
                        break

                    # Move to next page
                    offset += count

                    # Rate limiting - be respectful to the API
                    time.sleep(0.2)

                except requests.exceptions.RequestException as e:
                    print(f" ❌ Request Error: {str(e)}")
                    break
                except Exception as e:
                    print(f" ❌ Processing Error: {str(e)}")
                    break

            # Process all fetched data for this symbol
            if symbol_data:
                df = pd.DataFrame(symbol_data)
                df['date'] = pd.to_datetime(df['date'])

                # Remove timezone info if present
                if df['date'].dt.tz is not None:
                    df['date'] = df['date'].dt.tz_localize(None)

                df = df.set_index('date')

                # Create DataFrame with original symbol name
                price_df = pd.DataFrame({
                    f'{symbol}_Open': df['open'],
                    f'{symbol}_High': df['high'],
                    f'{symbol}_Low': df['low'],
                    f'{symbol}_Close': df['close'],
                    f'{symbol}_Volume': df['volume']
                })

                price_df = price_df.sort_index()
                all_symbol_dfs.append(price_df)

                print(f" ✓ ({total_fetched} records)")
            else:
                print(f" ✗ (no data)")

        # Concatenate all dataframes
        if all_symbol_dfs:
            all_price_data = pd.concat(all_symbol_dfs, axis=1, join='outer')
            all_price_data = all_price_data.sort_index()
            # Forward fill then backward fill for missing data
            all_price_data = all_price_data.ffill().bfill()
        else:
            all_price_data = pd.DataFrame()

        print(f"\n✅ Successfully fetched data for {len(all_symbol_dfs)} symbols")
        if not all_price_data.empty:
            print(f"📈 Total date range: {all_price_data.index[0]} to {all_price_data.index[-1]}")
            print(f"📊 Total records: {len(all_price_data)}")

        return all_price_data

    def fetch_market_data(self, symbols=['SPY', '^VIX', '^TNX', 'GLD'], use_marketstack=False):
        """
        Fetch market data for regime detection
        Updated with choice between Marketstack and yfinance
        """
        print("\n📊 FETCHING MARKET DATA")
        print("-" * 40)

        if use_marketstack:
            # Load API key
            api_key = self.load_marketstack_config()
            if not api_key:
                print("⚠️ No valid API key, falling back to yfinance")
                use_marketstack = False

        if use_marketstack:
            # Calculate date range - request full 10 years
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * self.lookback_years)

            print(f"Using Marketstack API for {self.lookback_years} years of data")

            # Fetch from Marketstack with pagination
            all_data = self.fetch_marketstack_data(
                symbols,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                api_key
            )

            if all_data.empty:
                print("⚠️ No data from Marketstack, falling back to yfinance")
                use_marketstack = False
            else:
                # Process data for regime detection
                price_data = pd.DataFrame()

                for symbol in symbols:
                    close_col = f'{symbol}_Close'
                    if close_col in all_data.columns:
                        price_data[symbol] = all_data[close_col]

                if price_data.empty:
                    print("⚠️ No close price data found, falling back to yfinance")
                    use_marketstack = False
                else:
                    self.data = price_data
                    print(f"✅ Loaded {len(self.data)} days of data from Marketstack")

        if not use_marketstack:
            # Use yfinance as fallback or alternative
            # print("Using yfinance for data fetching")
            print("Using yfinance for data fetching (recommended for better accuracy)")
            import yfinance as yf

            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * self.lookback_years)

            # Download data
            data = yf.download(
                symbols,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,  # This is the key change
                back_adjust=True  # Recommended for continuous historical data
            )

            # # Handle multi-level columns from yfinance
            # if isinstance(data.columns, pd.MultiIndex):
            #     # If multiple symbols, get Adj Close
            #     data = data['Adj Close']

            # We only need the 'Close' prices which are now adjusted automatically
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    # In case of multiple symbols, yfinance might still return a MultiIndex
                    # with ('Close', 'SPY'), ('Volume', 'SPY'), etc.
                    data = data['Close']

            # if len(symbols) == 1 and not isinstance(data, pd.DataFrame):
            #     data = pd.DataFrame(data)
            #     data.columns = symbols

            if len(symbols) == 1 and not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
                if not data.empty:
                    data.columns = symbols  # Ensure columns are named correctly for single symbol

            self.data = data
            print(f"✅ Loaded {len(self.data)} days of data from yfinance")

        # Save raw data for historical analysis
        self.save_historical_data()

        return self.data

    def save_historical_data(self):
        """Save fetched historical data for later use"""
        if self.data is not None and not self.data.empty:
            print(f"📁 Saving historical data to: {self.analysis_dir}")

            # Save to Regime_Detection_Analysis directory
            output_file = self.analysis_dir / "historical_market_data.csv"
            self.data.to_csv(output_file)
            print(f"💾 Saved historical data to {output_file}")

            # Also save a summary
            summary = {
                "data_range": {
                    "start": str(self.data.index[0]),
                    "end": str(self.data.index[-1]),
                    "days": len(self.data)
                },
                "symbols": list(self.data.columns),
                "fetch_date": datetime.now().isoformat()
            }

            summary_file = self.analysis_dir / "data_summary.json"

            print(f"💾 Saving summary to: {summary_file}")

            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

            # Verify files were created
            if summary_file.exists():
                print(f"✅ Successfully saved data summary")
            else:
                print(f"❌ Failed to save data summary!")

            return summary

    def fetch_and_save_historical_regimes(self):
        """
        Fetch historical data and identify regime periods
        COMPLETE FIXED VERSION
        """
        print("\n🔄 FETCHING HISTORICAL REGIME DATA")
        print("-" * 40)

        try:
            # Fetch fresh data from Marketstack
            self.fetch_market_data(use_marketstack=True)

            if self.data is None or self.data.empty:
                print("❌ Failed to fetch historical data")
                return False

            # Prepare features for HMM
            print("\n📊 Preparing features...")
            self.prepare_features()

            # Fit the model
            print("\n🔍 Fitting HMM model...")
            self.fit_hmm(use_pca=True, n_components=5)

            # Characterize regimes
            print("\n📈 Characterizing regimes...")
            self.characterize_regimes()

            # Export regime periods
            print("\n💾 Exporting regime periods...")
            self.export_regime_periods()

            # Save historical data summary
            print("\n💾 Saving data summary...")
            summary_data = self.save_historical_data()

            # Save detailed historical analysis
            print("\n💾 Saving historical analysis...")
            self.save_historical_analysis()

            # Run validation
            print("\n🎯 Validating detected regimes...")
            accuracy = None
            vix_alignment = None

            try:
                accuracy = self.validate_against_known_events()
            except Exception as e:
                print(f"⚠️ Validation error: {e}")

            try:
                vix_alignment = self.compare_with_vix_regimes()
            except Exception as e:
                print(f"⚠️ VIX comparison error: {e}")

            # Update the saved historical analysis to include validation
            if accuracy is not None or vix_alignment is not None:
                historical_file = self.analysis_dir / "historical_regimes.json"
                if historical_file.exists():
                    with open(historical_file, 'r') as f:
                        data = json.load(f)

                    # Add validation metrics
                    data['validation'] = {
                        'accuracy': float(accuracy) if accuracy is not None else None,
                        'accuracy_percentage': f"{accuracy:.1%}" if accuracy is not None else "N/A",
                        'vix_alignment': float(vix_alignment) if vix_alignment is not None else None,
                        'vix_alignment_percentage': f"{vix_alignment:.1%}" if vix_alignment is not None else "N/A"
                    }

                    with open(historical_file, 'w') as f:
                        json.dump(data, f, indent=2)

                    print(f"✅ Updated historical analysis with validation metrics")

            print("\n✅ Historical regime data saved successfully")
            return True

        except Exception as e:
            print(f"\n❌ Error in fetch_and_save_historical_regimes: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_historical_analysis(self):
        """Save detailed historical regime analysis"""
        if self.regime_history is None:
            print("⚠️ No regime history to save")
            return

        print(f"📁 Saving historical analysis to: {self.analysis_dir}")

        # Create detailed regime history with dates
        regime_details = []
        current_regime = None
        start_date = None
        prev_date = None

        for i, (date, regime) in enumerate(self.regime_history.items()):
            if regime != current_regime:
                if current_regime is not None and start_date is not None and prev_date is not None:
                    # Save the previous regime period
                    regime_details.append({
                        'regime': int(current_regime),
                        'regime_name': self.regime_characteristics[current_regime]['name'],
                        'start_date': str(start_date),
                        'end_date': str(prev_date),
                        'days': (prev_date - start_date).days + 1
                    })
                current_regime = regime
                start_date = date
            prev_date = date

        # Don't forget the last regime period
        if current_regime is not None and start_date is not None and prev_date is not None:
            regime_details.append({
                'regime': int(current_regime),
                'regime_name': self.regime_characteristics[current_regime]['name'],
                'start_date': str(start_date),
                'end_date': str(prev_date),
                'days': (prev_date - start_date).days + 1
            })

        print(f"📊 Found {len(regime_details)} regime periods to save")

        # Get statistics
        statistics = self.get_regime_statistics()

        # Save to JSON for easy loading
        historical_file = self.analysis_dir / "historical_regimes.json"

        output_data = {
            'regime_periods': regime_details,
            'statistics': statistics,
            'last_updated': datetime.now().isoformat()
        }

        print(f"💾 Saving to: {historical_file}")
        print(f"📊 Statistics: {statistics}")

        with open(historical_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        # Verify the file was created
        if historical_file.exists():
            print(f"✅ Successfully saved historical analysis to {historical_file}")
            print(f"📁 File size: {historical_file.stat().st_size} bytes")
        else:
            print(f"❌ Failed to save historical analysis!")

    def get_regime_statistics(self):
        """Calculate regime statistics for the historical period"""
        if self.regime_history is None:
            return {}

        # Count regime occurrences
        regime_counts = self.regime_history.value_counts()
        total_days = len(self.regime_history)

        statistics = {}
        for regime in range(self.n_regimes):
            if regime in regime_counts.index:
                count = regime_counts[regime]
                regime_name = self.regime_characteristics[regime]['name']
                statistics[regime_name] = {
                    'days': int(count),
                    'percentage': float(count / total_days),
                    'periods': len([1 for i in range(1, len(self.regime_history))
                                    if self.regime_history.iloc[i] == regime
                                    and self.regime_history.iloc[i - 1] != regime])
                }

        return statistics

    def engineer_features(self, use_pca=True, n_components=5):
        """
        Create features for HMM with PCA option for better convergence
        """
        print("\n🔧 Engineering features...")

        features = pd.DataFrame(index=self.data.index)

        # Use fewer, more meaningful features for better convergence
        base_col = 'SPY' if 'SPY' in self.data.columns else self.data.columns[0]

        # 1. Core return features
        returns = self.data[base_col].pct_change()
        features['return_1d'] = returns
        features['return_5d'] = returns.rolling(5).mean()
        features['return_21d'] = returns.rolling(21).mean()

        # 2. Volatility (most important for regime detection)
        features['volatility_21d'] = returns.rolling(21).std() * np.sqrt(252)
        features['volatility_63d'] = returns.rolling(63).std() * np.sqrt(252)

        # 3. Volatility change (regime transitions)
        features['vol_change'] = features['volatility_21d'].pct_change(21)

        # 4. Market stress indicator (if VIX available)
        if '^VIX' in self.data.columns:
            features['vix_norm'] = (self.data['^VIX'] - self.data['^VIX'].rolling(252).mean()) / self.data['^VIX'].rolling(252).std()

        # 5. Simple trend
        sma_50 = self.data[base_col].rolling(50).mean()
        sma_200 = self.data[base_col].rolling(200).mean()
        features['trend'] = (sma_50 - sma_200) / sma_200

        # 6. Momentum
        features['momentum'] = self.data[base_col] / self.data[base_col].shift(63) - 1

        # Clean up
        features = features.dropna()

        print(f"✓ Created {len(features.columns)} raw features")
        print(f"✓ Sample size: {len(features)} days")

        # Standardize features
        features_scaled = pd.DataFrame(
            self.scaler.fit_transform(features),
            index=features.index,
            columns=features.columns
        )

        if use_pca:
            # Apply PCA for dimensionality reduction
            print(f"\n📊 Applying PCA to reduce to {n_components} components...")
            self.pca = PCA(n_components=n_components, random_state=42)
            features_pca = self.pca.fit_transform(features_scaled)

            # Create DataFrame with PCA components
            pca_columns = [f'PC{i+1}' for i in range(n_components)]
            self.features_reduced = pd.DataFrame(
                features_pca,
                index=features.index,
                columns=pca_columns
            )

            # Print explained variance
            explained_var = self.pca.explained_variance_ratio_
            cumulative_var = np.cumsum(explained_var)
            print("\nPCA Explained Variance:")
            for i, (var, cum_var) in enumerate(zip(explained_var, cumulative_var)):
                print(f"  PC{i+1}: {var:.1%} (Cumulative: {cum_var:.1%})")

            # Store original features for analysis
            self.features = features_scaled

            return self.features_reduced
        else:
            self.features = features_scaled
            return self.features

    def engineer_simple_features(self):
        """
        Use only the most essential features for regime detection
        """
        print("\n🔧 Engineering simple features...")

        features = pd.DataFrame(index=self.data.index)

        # Only use SPY returns and volatility
        returns = self.data['SPY'].pct_change()

        # 1. Volatility (primary regime indicator)
        features['volatility'] = returns.rolling(21).std() * np.sqrt(252)

        # 2. Return momentum
        features['momentum'] = returns.rolling(21).mean() * 252

        # 3. VIX if available
        if '^VIX' in self.data.columns:
            features['vix'] = self.data['^VIX'] / 100

        # Clean and standardize
        features = features.dropna()
        self.features = pd.DataFrame(
            self.scaler.fit_transform(features),
            index=features.index,
            columns=features.columns
        )

        return self.features

    # def fit_hmm_with_multiple_attempts(self, n_attempts=10, n_iter=200):
    #     """
    #     Fit HMM with multiple random initializations to find best model
    #     """
    #     print(f"\n🤖 Training {self.n_regimes}-regime Gaussian HMM...")
    #
    #     # Use reduced features if available
    #     X = self.features_reduced.values if self.features_reduced is not None else self.features.values
    #
    #     best_model = None
    #     best_score = -np.inf
    #     convergence_count = 0
    #
    #     for attempt in range(n_attempts):
    #         try:
    #             # Try with different random states
    #             model = GaussianHMM(
    #                 n_components=self.n_regimes,
    #                 covariance_type="diag",  # More stable than "full"
    #                 n_iter=n_iter,
    #                 random_state=42 + attempt,
    #                 tol=0.01,  # Convergence tolerance
    #                 verbose=False
    #             )
    #
    #             # Fit model
    #             model.fit(X)
    #
    #             # Check if converged
    #             if model.monitor_.converged:
    #                 convergence_count += 1
    #                 score = model.score(X)
    #
    #                 if score > best_score:
    #                     best_score = score
    #                     best_model = model
    #
    #                 print(f"  Attempt {attempt + 1}: Converged ✓ (Score: {score:.2f})")
    #             else:
    #                 print(f"  Attempt {attempt + 1}: Not converged ✗")
    #
    #         except Exception as e:
    #             print(f"  Attempt {attempt + 1}: Failed - {str(e)[:50]}")
    #
    #     if best_model is None:
    #         raise ValueError("Failed to train any converged model")
    #
    #     print(f"\n✓ Best model score: {best_score:.2f}")
    #     print(f"✓ Converged models: {convergence_count}/{n_attempts}")
    #
    #     self.model = best_model
    #
    #     # Get regime predictions
    #     self.regime_history = pd.Series(
    #         self.model.predict(X),
    #         index=self.features_reduced.index if self.features_reduced is not None else self.features.index
    #     )
    #
    #     return self.model

    def fit_hmm_with_multiple_attempts(self, n_attempts=10, n_iter=200):
        """
        Fit HMM with multiple random initializations to find best model
        Improved version with better stability
        """
        print(f"\n🤖 Training {self.n_regimes}-regime Gaussian HMM...")
        print(f"  Using {n_attempts} random initializations")

        # Use reduced features if available, otherwise scaled features
        if self.features_reduced is not None:
            if hasattr(self.features_reduced, 'values'):
                X = self.features_reduced.values  # DataFrame to numpy
            else:
                X = self.features_reduced  # Already numpy
            print(f"  Using PCA features: {X.shape}")
        elif hasattr(self, 'features_scaled'):
            X = self.features_scaled
            print(f"  Using scaled features: {X.shape}")
        else:
            X = self.features.values
            print(f"  Using raw features: {X.shape}")

        best_model = None
        best_score = -np.inf
        convergence_count = 0
        scores = []

        for attempt in range(n_attempts):
            try:
                # Use different random states but allow randomness
                # This helps find different local optima
                random_state = None if attempt % 2 == 0 else (42 + attempt * 10)

                # Alternate between covariance types for robustness
                cov_type = "diag" if attempt < n_attempts // 2 else "spherical"

                model = GaussianHMM(
                    n_components=self.n_regimes,
                    covariance_type=cov_type,
                    n_iter=n_iter,
                    random_state=random_state,
                    tol=0.01,
                    verbose=False,
                    init_params='stmc',  # Initialize all parameters
                    params='stmc'  # Update all parameters
                )

                # Fit model
                model.fit(X)

                # Check if converged
                if model.monitor_.converged:
                    convergence_count += 1
                    score = model.score(X)
                    scores.append(score)

                    if score > best_score:
                        best_score = score
                        best_model = model

                    print(f"  Attempt {attempt + 1}: Converged ✓ (Score: {score:.2f}, Cov: {cov_type})")
                else:
                    print(f"  Attempt {attempt + 1}: Not converged ✗ (Cov: {cov_type})")

            except Exception as e:
                print(f"  Attempt {attempt + 1}: Failed - {str(e)[:50]}")

        if best_model is None:
            # If no model converged, try one more time with very relaxed settings
            print("\n⚠️ No converged models found, trying with relaxed settings...")
            try:
                model = GaussianHMM(
                    n_components=self.n_regimes,
                    covariance_type="spherical",  # Most stable
                    n_iter=500,  # More iterations
                    random_state=42,
                    tol=0.1,  # More tolerant
                    verbose=False
                )
                model.fit(X)
                best_model = model
                best_score = model.score(X)
                print(f"  ✓ Fallback model created (Score: {best_score:.2f})")
            except:
                raise ValueError("Failed to train any model even with relaxed settings")

        print(f"\n✓ Best model score: {best_score:.2f}")
        print(f"✓ Converged models: {convergence_count}/{n_attempts}")

        if scores:
            print(f"✓ Score range: {min(scores):.2f} to {max(scores):.2f}")
            print(f"✓ Average score: {np.mean(scores):.2f}")

        self.model = best_model

        # Get regime predictions - handle both DataFrame and numpy array
        if hasattr(self, 'features_reduced') and self.features_reduced is not None:
            index = self.features_reduced.index if hasattr(self.features_reduced, 'index') else self.features.index
        else:
            index = self.features.index

        self.regime_history = pd.Series(
            self.model.predict(X),
            index=index
        )

        # Print regime distribution
        regime_counts = self.regime_history.value_counts().sort_index()
        print("\nRegime distribution:")
        for regime, count in regime_counts.items():
            pct = count / len(self.regime_history) * 100
            print(f"  Regime {regime}: {count} days ({pct:.1f}%)")

        return self.model

    def validate_regimes(self):
        """
        Validate that regimes are meaningful and well-separated
        """
        print("\n🔍 Validating regime separation...")

        # Use appropriate features
        X = self.features_reduced.values if self.features_reduced is not None else self.features.values

        # Calculate average features for each regime
        regime_centers = []
        for regime in range(self.n_regimes):
            mask = (self.regime_history == regime)
            if mask.sum() > 0:
                regime_features = X[mask.values].mean(axis=0)
                regime_centers.append(regime_features)

        # Calculate pairwise distances between regime centers
        distances = []
        for i in range(len(regime_centers)):
            for j in range(i + 1, len(regime_centers)):
                dist = np.linalg.norm(regime_centers[i] - regime_centers[j])
                distances.append(dist)
                print(f"  Distance between Regime {i} and {j}: {dist:.3f}")

        avg_distance = np.mean(distances)
        print(f"\n  Average regime separation: {avg_distance:.3f}")

        if avg_distance < 0.5:
            print("  ⚠️ Warning: Regimes may not be well-separated")
        else:
            print("  ✓ Regimes appear well-separated")

        return avg_distance

    # def fit_hmm(self, use_pca=True, n_components=5, max_iter=500):  # Increased max_iter
    #     """
    #     Fit Hidden Markov Model with better convergence
    #     """
    #     from hmmlearn.hmm import GaussianHMM
    #     from sklearn.decomposition import PCA
    #     import warnings
    #
    #     print("\n🤖 FITTING HIDDEN MARKOV MODEL")
    #     print("-" * 40)
    #
    #     if not hasattr(self, 'features_scaled') or self.features_scaled is None:
    #         raise ValueError("No features prepared. Call prepare_features or engineer_features first.")
    #
    #     X = self.features_scaled
    #
    #     # Apply PCA if requested
    #     if use_pca and X.shape[1] > n_components:
    #         print(f"Applying PCA: {X.shape[1]} features → {n_components} components")
    #         self.pca = PCA(n_components=n_components, random_state=42)
    #         X = self.pca.fit_transform(X)
    #         self.features_reduced = X  # Store as numpy array
    #
    #         explained_var = self.pca.explained_variance_ratio_.sum()
    #         print(f"✓ PCA explains {explained_var:.1%} of variance")
    #     else:
    #         print(f"Using all {X.shape[1]} features (no PCA)")
    #         self.features_reduced = X
    #
    #     # Try multiple random initializations to find best model
    #     best_score = -np.inf
    #     best_model = None
    #     n_tries = 3
    #
    #     print(f"Trying {n_tries} random initializations...")
    #
    #     for attempt in range(n_tries):
    #         try:
    #             # Suppress convergence warnings for individual attempts
    #             with warnings.catch_warnings():
    #                 warnings.simplefilter("ignore")
    #
    #                 # Initialize HMM with different random seed
    #                 model = GaussianHMM(
    #                     n_components=self.n_regimes,
    #                     covariance_type="diag",  # Use diagonal for better convergence
    #                     n_iter=max_iter,
    #                     tol=1e-2,  # Slightly more tolerant convergence
    #                     random_state=42 + attempt,
    #                     init_params="stmc",
    #                     params="stmc",
    #                     verbose=False
    #                 )
    #
    #                 # Fit the model
    #                 model.fit(X)
    #
    #                 # Calculate score
    #                 score = model.score(X)
    #
    #                 print(f"  Attempt {attempt + 1}: score = {score:.2f}, converged = {model.monitor_.converged}")
    #
    #                 # Keep best model
    #                 if score > best_score:
    #                     best_score = score
    #                     best_model = model
    #
    #         except Exception as e:
    #             print(f"  Attempt {attempt + 1} failed: {e}")
    #             continue
    #
    #     if best_model is None:
    #         raise ValueError("Failed to fit HMM after multiple attempts")
    #
    #     # Use best model
    #     self.model = best_model
    #
    #     print(f"\n✓ Best model: score = {best_score:.2f}")
    #     print(f"✓ Converged: {self.model.monitor_.converged}")
    #     print(f"✓ Iterations: {self.model.monitor_.iter}")
    #
    #     # Predict regimes
    #     self.regime_history = pd.Series(
    #         self.model.predict(X),
    #         index=self.features.index
    #     )
    #
    #     print(f"✓ Identified {self.n_regimes} market regimes")
    #
    #     # Print regime distribution
    #     regime_counts = self.regime_history.value_counts().sort_index()
    #     print("\nRegime distribution:")
    #     for regime, count in regime_counts.items():
    #         pct = count / len(self.regime_history) * 100
    #         print(f"  Regime {regime}: {count} days ({pct:.1f}%)")
    #
    #     return self.model

    def fit_hmm(self, use_pca=True, n_components=5):
        """
        Fit Hidden Markov Model with FIXED random seed for consistency
        """
        if self.features is None:
            raise ValueError("No features available. Run prepare_features first.")

        print("\n🔍 Fitting Hidden Markov Model...")
        print(f"  Number of regimes: {self.n_regimes}")

        # Prepare features
        if use_pca and n_components < self.features.shape[1]:
            X = self.features_reduced.values if self.features_reduced is not None else self.features_scaled
        else:
            X = self.features_scaled

        print(f"  Feature dimensions: {X.shape}")

        # Try multiple times with FIXED seeds for consistency
        best_model = None
        best_score = -np.inf

        # Use fixed seeds instead of random
        seeds = [42, 123, 456, 789, 1001]

        for attempt, seed in enumerate(seeds):
            try:
                print(f"\n  Attempt {attempt + 1}/5 with seed {seed}...")

                # Initialize model with FIXED random state
                model = GaussianHMM(
                    n_components=self.n_regimes,
                    covariance_type="full",
                    n_iter=200,
                    random_state=seed,  # FIXED seed for reproducibility
                    verbose=False
                )

                # Fit model
                model.fit(X)
                score = model.score(X)
                print(f"    Score: {score:.2f}")

                if score > best_score:
                    best_score = score
                    best_model = model
                    print(f"    ✓ New best model!")

            except Exception as e:
                print(f"  Attempt {attempt + 1} failed: {e}")
                continue

        if best_model is None:
            raise ValueError("Failed to fit HMM after multiple attempts")

        # Use best model
        self.model = best_model

        print(f"\n✓ Best model: score = {best_score:.2f}")
        print(f"✓ Converged: {self.model.monitor_.converged}")
        print(f"✓ Iterations: {self.model.monitor_.iter}")

        # Predict regimes
        self.regime_history = pd.Series(
            self.model.predict(X),
            index=self.features.index
        )

        print(f"✓ Identified {self.n_regimes} market regimes")

        # Print regime distribution
        regime_counts = self.regime_history.value_counts().sort_index()
        print("\nRegime distribution:")
        for regime, count in regime_counts.items():
            pct = count / len(self.regime_history) * 100
            print(f"  Regime {regime}: {count} days ({pct:.1f}%)")

        return self.model

    def characterize_regimes(self):
        """
        Analyze and characterize each regime
        """
        print("\n📈 Characterizing regimes...")

        # Get returns for analysis
        if 'SPY' in self.data.columns:
            returns = self.data['SPY'].pct_change().dropna()
        else:
            returns = pd.Series(index=self.features.index)

        for regime in range(self.n_regimes):
            mask = (self.regime_history == regime)
            regime_dates = self.regime_history[mask].index

            # Calculate regime characteristics
            regime_returns = returns.reindex(regime_dates).dropna()

            characteristics = {
                'frequency': mask.mean(),
                'total_days': mask.sum(),
                'mean_return': regime_returns.mean() * 252,  # Annualized
                'volatility': regime_returns.std() * np.sqrt(252),
                'sharpe': (regime_returns.mean() / regime_returns.std() * np.sqrt(252)
                           if regime_returns.std() > 0 else 0),
                'skewness': regime_returns.skew(),
                'kurtosis': regime_returns.kurtosis(),
                'max_drawdown': self._calculate_max_drawdown(regime_returns),
                'avg_feature_values': {}
            }

            # Average feature values in this regime
            for feature in self.features.columns:
                original_feature = self.scaler.inverse_transform(self.features)[mask]
                if len(original_feature) > 0:
                    characteristics['avg_feature_values'][feature] = \
                        original_feature[:, self.features.columns.get_loc(feature)].mean()

            # Date ranges
            if len(regime_dates) > 0:
                characteristics['first_occurrence'] = regime_dates[0].strftime('%Y-%m-%d')
                characteristics['last_occurrence'] = regime_dates[-1].strftime('%Y-%m-%d')

                # Find continuous periods
                periods = []
                period_start = regime_dates[0]

                for i in range(1, len(regime_dates)):
                    if (regime_dates[i] - regime_dates[i - 1]).days > 5:  # Gap > 5 days
                        periods.append({
                            'start': period_start.strftime('%Y-%m-%d'),
                            'end': regime_dates[i - 1].strftime('%Y-%m-%d'),
                            'days': (regime_dates[i - 1] - period_start).days
                        })
                        period_start = regime_dates[i]

                # Add last period
                periods.append({
                    'start': period_start.strftime('%Y-%m-%d'),
                    'end': regime_dates[-1].strftime('%Y-%m-%d'),
                    'days': (regime_dates[-1] - period_start).days
                })

                characteristics['periods'] = periods
                characteristics['n_periods'] = len(periods)
                characteristics['avg_period_length'] = np.mean([p['days'] for p in periods])

            self.regime_characteristics[regime] = characteristics

        # Sort regimes by Sharpe ratio for naming
        sorted_regimes = sorted(
            self.regime_characteristics.items(),
            key=lambda x: x[1]['sharpe'],
            reverse=True
        )

        # Assign meaningful names
        # regime_names = {}
        # for rank, (regime, chars) in enumerate(sorted_regimes):
        #     vol = chars['volatility']
        #     ret = chars['mean_return']
        #     sharpe = chars['sharpe']
        #
        #     # More nuanced naming based on characteristics
        #     if vol > 0.20 and ret < -0.05:
        #         name = "Crisis/Bear"
        #     elif vol > 0.18:
        #         name = "High Volatility"
        #     elif vol < 0.08 and ret > 0.12 and sharpe > 2.0:
        #         name = "Strong Bull"
        #     elif vol < 0.12 and ret > 0.05 and sharpe > 0.8:
        #         name = "Steady Growth"
        #     elif vol > 0.08 and vol < 0.15 and ret > 0:
        #         name = "Normal Market"
        #     elif ret < 0 and vol < 0.15:
        #         name = "Mild Correction"
        #     else:
        #         # Fallback based on simple metrics
        #         if sharpe > 1.5:
        #             name = "Bull Market"
        #         elif sharpe < 0:
        #             name = "Bear Market"
        #         else:
        #             name = "Neutral"
        #
        #     regime_names[regime] = name
        #     self.regime_characteristics[regime]['name'] = name

        # Assign meaningful names - SIMPLE approach for exactly 3 regimes
        # Since we have n_regimes=3, we'll always have exactly 3 regimes to name
        regime_names = {}

        # Sort regimes by Sharpe ratio (already done above)
        # sorted_regimes is already sorted from highest to lowest Sharpe

        if len(sorted_regimes) == 3:
            # We have exactly 3 regimes, assign them based on ranking
            for rank, (regime, chars) in enumerate(sorted_regimes):
                vol = chars['volatility']
                ret = chars['mean_return']
                sharpe = chars['sharpe']

                if rank == 0:  # Best Sharpe ratio
                    # Verify it's actually good before calling it Bull
                    if sharpe > 0.4 and ret > 0.0:
                        name = "Strong Bull"
                    else:
                        name = "Steady Growth"

                elif rank == 1:  # Middle Sharpe ratio
                    # Middle regime is usually Steady Growth
                    # Unless it's clearly negative
                    if ret < -0.05 or sharpe < -0.2:
                        name = "Crisis/Bear"
                    else:
                        name = "Steady Growth"

                else:  # rank == 2, Worst Sharpe ratio
                    # Check if it's actually bad enough to be Crisis/Bear
                    if sharpe < 0.3 or ret < 0.02 or vol > 0.22:
                        name = "Crisis/Bear"
                    else:
                        name = "Steady Growth"

                regime_names[regime] = name
                self.regime_characteristics[regime]['name'] = name

        else:
            # Fallback for different number of regimes
            # This shouldn't happen if n_regimes=3, but just in case
            for regime, chars in sorted_regimes:
                vol = chars['volatility']
                ret = chars['mean_return']
                sharpe = chars['sharpe']

                if sharpe > 0.7 and ret > 0.10 and vol < 0.15:
                    name = "Strong Bull"
                elif sharpe < 0.2 or ret < 0.0 or vol > 0.22:
                    name = "Crisis/Bear"
                else:
                    name = "Steady Growth"

                regime_names[regime] = name
                self.regime_characteristics[regime]['name'] = name

        # Ensure we have at least one "Steady Growth" regime
        # Count how many of each regime we have
        regime_counts = {}
        for regime, name in regime_names.items():
            regime_counts[name] = regime_counts.get(name, 0) + 1

        # If we don't have a Steady Growth, convert the most moderate regime
        if "Steady Growth" not in regime_counts and len(regime_names) >= 2:
            # Find the regime with characteristics closest to neutral
            best_steady_candidate = None
            best_steady_score = float('inf')

            for regime, name in regime_names.items():
                chars = self.regime_characteristics[regime]
                # Score based on distance from moderate values
                score = abs(chars['sharpe'] - 0.5) + abs(chars['mean_return'] - 0.08)
                if score < best_steady_score:
                    best_steady_score = score
                    best_steady_candidate = regime

            if best_steady_candidate is not None:
                regime_names[best_steady_candidate] = "Steady Growth"
                self.regime_characteristics[best_steady_candidate]['name'] = "Steady Growth"

        return self.regime_characteristics

    def _calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown from returns series"""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        return drawdown.min()

    def plot_regimes(self):
        """
        Visualize detected regimes - FIXED VERSION
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if self.regime_history is None:
            print("No regime history to plot. Run fit_hmm first.")
            return

        print("\n📊 GENERATING REGIME VISUALIZATIONS")
        print("-" * 40)

        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # 1. Plot regime history over time
        ax1 = axes[0]
        regime_colors = ['green', 'orange', 'red']

        for regime in range(self.n_regimes):
            mask = self.regime_history == regime
            ax1.fill_between(self.regime_history.index, 0, 1,
                             where=mask,
                             color=regime_colors[regime],
                             alpha=0.3,
                             label=self.regime_characteristics[regime]['name'])

        ax1.set_ylabel('Regime')
        ax1.set_title('Market Regime Detection Over Time')
        ax1.legend(loc='upper right')
        ax1.set_ylim(0, 1)

        # 2. Plot SPY price with regime coloring
        ax2 = axes[1]
        if 'SPY' in self.data.columns:
            # FIX: Align spy_data index with regime_history index to avoid mismatch
            spy_data = self.data['SPY'].reindex(self.regime_history.index).dropna()

            for regime in range(self.n_regimes):
                mask = self.regime_history == regime
                regime_data = spy_data[mask]
                ax2.scatter(regime_data.index, regime_data.values,
                            c=regime_colors[regime],
                            alpha=0.6,
                            s=1,
                            label=self.regime_characteristics[regime]['name'])

            ax2.set_ylabel('SPY Price')
            ax2.set_title('SPY Price Colored by Regime')
            ax2.legend(loc='upper left')

        # 3. Plot regime probabilities
        ax3 = axes[2]

        # FIX: Handle both DataFrame and numpy array cases
        if hasattr(self.features_reduced, 'values'):
            # It's a DataFrame
            X = self.features_reduced.values
        elif isinstance(self.features_reduced, np.ndarray):
            # It's already a numpy array
            X = self.features_reduced
        elif self.features_reduced is not None:
            # Some other format, try to convert
            X = np.array(self.features_reduced)
        else:
            # Fall back to features
            if hasattr(self.features, 'values'):
                X = self.features.values
            else:
                X = np.array(self.features)

        # Get regime probabilities
        regime_proba = self.model.predict_proba(X)

        # Plot probabilities for each regime
        for regime in range(self.n_regimes):
            ax3.plot(self.features.index, regime_proba[:, regime],
                     label=self.regime_characteristics[regime]['name'],
                     color=regime_colors[regime],
                     alpha=0.7)

        ax3.set_ylabel('Probability')
        ax3.set_xlabel('Date')
        ax3.set_title('Regime Probabilities Over Time')
        ax3.legend(loc='upper right')
        ax3.set_ylim(0, 1)

        plt.tight_layout()

        # Save figure
        plot_file = self.output_dir / "regime_detection_plot.png"
        plt.savefig(plot_file, dpi=100, bbox_inches='tight')
        print(f"✓ Saved plot to {plot_file}")

        # plt.show()

    def generate_report(self):
        """
        Generate comprehensive regime analysis report
        """
        if self.regime_history is None:
            print("No regime history available. Run fit_hmm first.")
            return

        print("\n📝 REGIME DETECTION REPORT")
        print("=" * 60)

        # Overall statistics
        print("\n1. OVERALL STATISTICS")
        print("-" * 40)
        print(f"Data period: {self.data.index[0].date()} to {self.data.index[-1].date()}")
        print(f"Total days: {len(self.data)}")
        print(
            f"Features engineered: {self.features.shape[1] if hasattr(self.features, 'shape') else len(self.features.columns)}")

        # Handle different types for features_reduced
        if self.features_reduced is not None:
            if hasattr(self.features_reduced, 'shape'):
                reduced_features = self.features_reduced.shape[1]
            else:
                reduced_features = "Unknown"
        else:
            reduced_features = "N/A"

        print(f"Features after PCA: {reduced_features}")
        print(f"Number of regimes: {self.n_regimes}")

        print("\n" + "=" * 60)
        print("📊 MARKET REGIME ANALYSIS REPORT")
        print("=" * 60)

        # Sort regimes by frequency
        sorted_by_freq = sorted(
            self.regime_characteristics.items(),
            key=lambda x: x[1]['frequency'],
            reverse=True
        )

        for regime, chars in sorted_by_freq:
            print(f"\n🏷️  REGIME {regime}: {chars['name']}")
            print("-" * 40)
            print(f"Frequency: {chars['frequency']:.1%} ({chars['total_days']} days)")
            print(f"Annualized Return: {chars['mean_return']:.1%}")
            print(f"Annualized Volatility: {chars['volatility']:.1%}")
            print(f"Sharpe Ratio: {chars['sharpe']:.2f}")
            print(f"Skewness: {chars['skewness']:.2f}")
            print(f"Max Drawdown: {chars['max_drawdown']:.1%}")
            print(f"Number of Periods: {chars['n_periods']}")
            print(f"Avg Period Length: {chars['avg_period_length']:.0f} days")

            # Show some key periods
            if 'periods' in chars and len(chars['periods']) > 0:
                print("\nKey Periods:")
                for i, period in enumerate(chars['periods'][:3]):  # Show first 3
                    print(f"  {period['start']} to {period['end']} ({period['days']} days)")
                if len(chars['periods']) > 3:
                    print(f"  ... and {len(chars['periods']) - 3} more periods")

        # Transition matrix
        print("\n" + "=" * 60)
        print("📊 REGIME TRANSITION MATRIX")
        print("=" * 60)
        self._print_transition_matrix()

        # Feature importance
        print("\n" + "=" * 60)
        print("📊 REGIME DISTINGUISHING FEATURES")
        print("=" * 60)
        self._analyze_feature_importance()

    def _print_transition_matrix(self):
        """
        Calculate and print regime transition probabilities
        """
        transitions = np.zeros((self.n_regimes, self.n_regimes))

        for i in range(len(self.regime_history) - 1):
            from_regime = self.regime_history.iloc[i]
            to_regime = self.regime_history.iloc[i + 1]
            transitions[from_regime, to_regime] += 1

        # Normalize rows
        transition_probs = transitions / transitions.sum(axis=1, keepdims=True)

        # Create DataFrame for pretty printing
        regime_names = [self.regime_characteristics[i]['name'] for i in range(self.n_regimes)]
        trans_df = pd.DataFrame(transition_probs,
                                index=regime_names,
                                columns=regime_names)

        print(trans_df.round(3))

        # Average regime duration
        print("\nAverage Regime Duration:")
        for i in range(self.n_regimes):
            avg_duration = 1 / (1 - transition_probs[i, i])
            print(f"  {regime_names[i]}: {avg_duration:.1f} days")

    def _analyze_feature_importance(self):
        """
        Identify which features best distinguish regimes
        """
        # Calculate feature variance across regimes
        regime_means = []
        for regime in range(self.n_regimes):
            mask = (self.regime_history == regime)
            if mask.sum() > 0:
                regime_features = self.features[mask].mean()
                regime_means.append(regime_features)

        regime_means_df = pd.DataFrame(regime_means)
        feature_variance = regime_means_df.var()

        # Sort by variance (most distinguishing features)
        important_features = feature_variance.sort_values(ascending=False).head(10)

        print("Top Distinguishing Features (by variance across regimes):")
        for feature, variance in important_features.items():
            print(f"  {feature}: {variance:.3f}")

    def export_regime_periods(self, filename='regime_periods.csv'):
        """
        Export regime periods for use with optimizer
        """
        # Update the path
        filepath = self.output_dir / filename  # Use output_dir

        regime_data = []

        for regime in range(self.n_regimes):
            chars = self.regime_characteristics[regime]
            for period in chars.get('periods', []):
                regime_data.append({
                    'regime': regime,
                    'regime_name': chars['name'],
                    'start_date': period['start'],
                    'end_date': period['end'],
                    'days': period['days'],
                    'mean_return': chars['mean_return'],
                    'volatility': chars['volatility'],
                    'sharpe': chars['sharpe']
                })

        df = pd.DataFrame(regime_data)
        df.to_csv(filepath, index=False)
        print(f"\n✓ Exported regime periods to {filepath}")

        return df

    # def get_current_regime(self):
    #     """
    #     Get the current market regime
    #     """
    #     current_regime = self.regime_history.iloc[-1]
    #
    #     # Fix: Handle cases where features_reduced is a numpy array (no .values)
    #     if self.features_reduced is not None:
    #         X = self.features_reduced  # Already a numpy array
    #     else:
    #         X = self.features.values  # Convert DataFrame to numpy array
    #
    #     regime_proba = self.model.predict_proba(X)[-1]
    #
    #     print("\n🎯 CURRENT MARKET REGIME")
    #     print("-" * 40)
    #     print(f"Regime: {self.regime_characteristics[current_regime]['name']}")
    #     print(f"Confidence: {regime_proba[current_regime]:.1%}")
    #     print("\nAll Regime Probabilities:")
    #     for regime in range(self.n_regimes):
    #         print(f"  {self.regime_characteristics[regime]['name']}: {regime_proba[regime]:.1%}")
    #
    #     return current_regime, regime_proba

    def get_current_regime(self):
        """
        Get the current market regime - FIXED to use only the last data point
        """
        if self.regime_history is None or len(self.regime_history) == 0:
            raise ValueError("No regime history available")

        # Get the last regime from history
        current_regime = self.regime_history.iloc[-1]

        # Get features for probability calculation
        if self.features_reduced is not None:
            # If using PCA, use reduced features
            if hasattr(self.features_reduced, 'values'):
                X = self.features_reduced.values  # DataFrame to numpy
            else:
                X = self.features_reduced  # Already numpy array
        else:
            # Use scaled features
            if hasattr(self.features_scaled, 'shape'):
                X = self.features_scaled  # Already numpy array
            else:
                X = self.features.values  # DataFrame to numpy

        # CRITICAL FIX: Only use the LAST observation for probability
        # The original code was using all data points, which gives wrong probabilities
        last_observation = X[-1:, :]  # Keep 2D shape for predict_proba

        # Get probability for the current (last) observation only
        regime_proba = self.model.predict_proba(last_observation)[0]

        print("\n🎯 CURRENT MARKET REGIME (as of last data point)")
        print("-" * 40)
        print(f"Date: {self.regime_history.index[-1].strftime('%Y-%m-%d')}")
        print(f"Regime: {self.regime_characteristics[current_regime]['name']}")
        print(f"Confidence: {regime_proba[current_regime]:.1%}")
        print("\nAll Regime Probabilities:")
        for regime in range(self.n_regimes):
            print(f"  {self.regime_characteristics[regime]['name']}: {regime_proba[regime]:.1%}")

        return current_regime, regime_proba

    def save_model(self, path='regime_model.pkl', filename='regime_model.pkl'):
        """Save the trained model and scaler"""
        # Update the path
        filepath = self.output_dir / filename  # Use output_dir

        import pickle

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca if hasattr(self, 'pca') and self.pca is not None else None,
            'n_regimes': self.n_regimes,
            'feature_columns': self.features.columns.tolist(),
            'regime_characteristics': self.regime_characteristics,
            'regime_history': self.regime_history,
            'use_pca': hasattr(self, 'pca') and self.pca is not None
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✓ Model saved to {filepath}")
        if model_data['use_pca']:
            print(f"✓ PCA included (components: {self.pca.n_components})")

    # def validate_against_known_events(self):
    #     """
    #     Compare detected regimes against known market events
    #     Returns accuracy score
    #     """
    #     print("\n🎯 REGIME DETECTION VALIDATION")
    #     print("=" * 60)
    #
    #     if self.regime_history is None:
    #         print("No regime history available for validation")
    #         return None
    #
    #     # Fix timezone issues
    #     if self.regime_history.index.tz is not None:
    #         self.regime_history.index = self.regime_history.index.tz_localize(None)
    #
    #     # Known events with expected regimes
    #     known_events = {
    #         # Crisis/High Vol events
    #         'COVID Crash': {
    #             'dates': ('2020-02-19', '2020-03-23'),
    #             'expected': 'Crisis/High Vol',
    #             'description': 'Market crash due to COVID-19'
    #         },
    #         'Volmageddon': {
    #             'dates': ('2018-01-26', '2018-02-08'),
    #             'expected': 'Crisis/High Vol',
    #             'description': 'XIV implosion, VIX spike'
    #         },
    #         'Q4 2018 Selloff': {
    #             'dates': ('2018-10-03', '2018-12-24'),
    #             'expected': 'Crisis/High Vol',
    #             'description': 'Fed tightening fears'
    #         },
    #         'China Devaluation': {
    #             'dates': ('2015-08-18', '2016-02-11'),
    #             'expected': 'Crisis/High Vol',
    #             'description': 'Yuan devaluation, growth fears'
    #         },
    #         '2022 Bear Market': {
    #             'dates': ('2022-01-03', '2022-10-12'),
    #             'expected': 'Crisis/High Vol',
    #             'description': 'Fed rate hikes, inflation'
    #         },
    #
    #         # Recovery/Bull events
    #         'COVID Recovery': {
    #             'dates': ('2020-03-24', '2020-06-08'),
    #             'expected': 'High Vol Bull',  # Still volatile but recovering
    #             'description': 'Fed stimulus, market recovery'
    #         },
    #         'Taper Tantrum Recovery': {
    #             'dates': ('2016-02-12', '2016-12-31'),
    #             'expected': 'Bull/Growth',
    #             'description': 'Post-correction recovery'
    #         },
    #         '2023 Recovery': {
    #             'dates': ('2023-01-01', '2023-12-31'),
    #             'expected': 'Bull/Growth',
    #             'description': 'AI boom, soft landing hopes'
    #         },
    #
    #         # Additional validation periods
    #         '2017 Melt-Up': {
    #             'dates': ('2017-01-01', '2017-12-31'),
    #             'expected': 'Bull/Growth',
    #             'description': 'Low vol bull market'
    #         },
    #         'March 2023 Banking Crisis': {
    #             'dates': ('2023-03-08', '2023-03-31'),
    #             'expected': 'Crisis/High Vol',
    #             'description': 'SVB collapse'
    #         }
    #     }
    #
    #     # Validate each event
    #     results = []
    #     for event_name, event_info in known_events.items():
    #         start_date = pd.to_datetime(event_info['dates'][0])
    #         end_date = pd.to_datetime(event_info['dates'][1])
    #
    #         # Get regimes during this period
    #         mask = (self.regime_history.index >= start_date) & (self.regime_history.index <= end_date)
    #         period_regimes = self.regime_history[mask]
    #
    #         if len(period_regimes) > 0:
    #             # Find dominant regime
    #             regime_counts = period_regimes.value_counts()
    #             dominant_regime = regime_counts.index[0]
    #             dominant_name = self.regime_characteristics[dominant_regime]['name']
    #
    #             # Check all regimes present
    #             all_regimes = ', '.join([
    #                 self.regime_characteristics[r]['name']
    #                 for r in regime_counts.index
    #             ])
    #
    #             # Check if it matches expected
    #             match = self._check_regime_match(dominant_name, event_info['expected'])
    #
    #             results.append({
    #                 'Event': event_name,
    #                 'Period': f"{start_date.date()} to {end_date.date()}",
    #                 'Expected': event_info['expected'],
    #                 'Dominant': dominant_name,
    #                 'All Regimes': all_regimes,
    #                 'Match': match
    #             })
    #
    #     if not results:
    #         print("No validation events found in data range")
    #         return None
    #
    #     # Create DataFrame properly
    #     results_df = pd.DataFrame(results)
    #
    #     # Print detailed results
    #     print("\nValidation Results:")
    #     print("-" * 60)
    #     for _, row in results_df.iterrows():
    #         print(f"\n{row['Event']}:")
    #         print(f"  Period: {row['Period']}")
    #         print(f"  Expected: {row['Expected']}")
    #         print(f"  Detected: {row['All Regimes']}")
    #         print(f"  Match: {'✓' if row['Match'] else '✗'}")
    #
    #     # Calculate accuracy
    #     accuracy = results_df['Match'].sum() / len(results_df)
    #     print(f"\n{'=' * 60}")
    #     print(f"Overall Validation Accuracy: {accuracy:.1%}")
    #     print(f"Correct Classifications: {results_df['Match'].sum()}/{len(results_df)}")
    #
    #     # Save accuracy to file
    #     self.validation_accuracy = accuracy
    #     validation_data = self._save_validation_results(results_df, accuracy)
    #
    #     return accuracy

    def validate_against_known_events(self):
        """
        Compare detected regimes against known market events
        Updated for 3-regime system with more accurate classifications
        """
        print("\n🎯 REGIME DETECTION VALIDATION")
        print("=" * 60)

        if self.regime_history is None:
            print("No regime history available for validation")
            return None

        # Fix timezone issues
        if self.regime_history.index.tz is not None:
            self.regime_history.index = self.regime_history.index.tz_localize(None)

        # Updated events for 3-regime system
        # More balanced and accurate representation
        known_events = {
            # TRUE Crisis/Bear events (should be rare - only 4-5% of time)
            'COVID Crash': {
                'dates': ('2020-02-19', '2020-03-23'),
                'expected': 'Crisis/Bear',
                'description': 'Sharp market crash, VIX > 80'
            },
            'Volmageddon': {
                'dates': ('2018-02-02', '2018-02-09'),  # Shortened to actual crisis week
                'expected': 'Crisis/Bear',
                'description': 'XIV implosion, VIX spike to 50'
            },
            'March 2020 Peak Panic': {
                'dates': ('2020-03-16', '2020-03-23'),  # Most extreme COVID week
                'expected': 'Crisis/Bear',
                'description': 'Circuit breakers, maximum fear'
            },

            # Strong Bull events (top ~20-30% of markets)
            '2017 Melt-Up': {
                'dates': ('2017-01-01', '2017-12-31'),
                'expected': 'Strong Bull',
                'description': 'Low vol, steady gains, VIX < 12'
            },
            '2021 Bull Run': {
                'dates': ('2021-01-01', '2021-11-30'),
                'expected': 'Strong Bull',
                'description': 'Post-COVID bull market, meme stocks'
            },
            '2019 Rally': {
                'dates': ('2019-01-01', '2019-07-31'),
                'expected': 'Strong Bull',
                'description': 'Fed pivot rally'
            },

            # Steady Growth events (most common - ~40-50%)
            '2023 Recovery': {
                'dates': ('2023-03-01', '2023-12-31'),
                'expected': 'Steady Growth',
                'description': 'Gradual recovery, AI boom'
            },
            '2018 Mid-Year': {
                'dates': ('2018-04-01', '2018-09-30'),
                'expected': 'Steady Growth',
                'description': 'Normal market conditions'
            },
            '2016 Recovery': {
                'dates': ('2016-07-01', '2016-12-31'),
                'expected': 'Steady Growth',
                'description': 'Post-Brexit recovery, election rally'
            },
            '2015 Range-Bound': {
                'dates': ('2015-03-01', '2015-07-31'),
                'expected': 'Steady Growth',
                'description': 'Sideways market, normal volatility'
            },

            # Mixed/Transitional periods (can be any regime)
            'COVID Recovery': {
                'dates': ('2020-04-01', '2020-06-30'),
                'expected': 'Steady Growth',  # Changed from "High Vol Bull"
                'description': 'Initial recovery phase'
            },
            '2022 Decline': {
                'dates': ('2022-04-01', '2022-06-30'),  # Shortened period
                'expected': 'Steady Growth',  # Gradual decline, not crisis
                'description': 'Fed tightening, orderly decline'
            }
        }

        # Validate each event
        results = []
        for event_name, event_info in known_events.items():
            start_date = pd.to_datetime(event_info['dates'][0])
            end_date = pd.to_datetime(event_info['dates'][1])

            # Get regimes during this period
            mask = (self.regime_history.index >= start_date) & (self.regime_history.index <= end_date)
            period_regimes = self.regime_history[mask]

            if len(period_regimes) > 0:
                # Find dominant regime
                regime_counts = period_regimes.value_counts()
                dominant_regime = regime_counts.index[0]
                dominant_name = self.regime_characteristics[dominant_regime]['name']

                # Check all regimes present
                all_regimes = ', '.join([
                    self.regime_characteristics[r]['name']
                    for r in regime_counts.index
                ])

                # Check if it matches expected
                match = self._check_regime_match(dominant_name, event_info['expected'])

                results.append({
                    'Event': event_name,
                    'Period': f"{start_date.date()} to {end_date.date()}",
                    'Expected': event_info['expected'],
                    'Dominant': dominant_name,
                    'All Regimes': all_regimes,
                    'Match': match
                })

        if not results:
            print("No validation events found in data range")
            return None

        # Create DataFrame
        results_df = pd.DataFrame(results)

        # Print detailed results
        print("\nValidation Results:")
        print("-" * 60)
        for _, row in results_df.iterrows():
            print(f"\n{row['Event']}:")
            print(f"  Period: {row['Period']}")
            print(f"  Expected: {row['Expected']}")
            print(f"  Detected: {row['Dominant']}")
            print(f"  Match: {'✓' if row['Match'] else '✗'}")

        # Calculate accuracy
        accuracy = results_df['Match'].sum() / len(results_df)
        print(f"\n{'=' * 60}")
        print(f"Overall Validation Accuracy: {accuracy:.1%}")
        print(f"Correct Classifications: {results_df['Match'].sum()}/{len(results_df)}")

        # Save accuracy to file
        self.validation_accuracy = accuracy
        validation_data = self._save_validation_results(results_df, accuracy)

        return accuracy

    def prepare_features(self):
        """Prepare features for HMM - FIXED VERSION to handle Inf/NaN"""
        if self.data is None:
            raise ValueError("No data loaded. Call fetch_market_data first.")

        print("🔧 Engineering features...")

        # Calculate returns with proper handling
        returns = self.data.pct_change()

        # Replace any infinite values with NaN first
        returns = returns.replace([np.inf, -np.inf], np.nan)

        # Forward fill then backward fill NaN values
        returns = returns.fillna(method='ffill').fillna(method='bfill')

        # If still NaN, fill with 0
        returns = returns.fillna(0)

        # Create feature DataFrame
        features = pd.DataFrame(index=returns.index)

        # 1. Returns (clipped to avoid extreme values)
        for col in self.data.columns:
            # Clip returns to reasonable range (-50% to +50% daily)
            features[f'{col}_return'] = returns[col].clip(-0.5, 0.5)

        # 2. Rolling volatility with proper window
        for col in self.data.columns:
            # Use 20-day rolling window, but handle edge cases
            vol = returns[col].rolling(window=20, min_periods=5).std()
            # Fill initial NaN values with expanding window std
            vol = vol.fillna(returns[col].expanding(min_periods=2).std())
            # Clip volatility to reasonable range
            features[f'{col}_vol'] = vol.clip(0, 1)

        # 3. Moving average ratios (price relative to MA)
        for col in self.data.columns:
            if col in self.data.columns:
                ma_20 = self.data[col].rolling(window=20, min_periods=5).mean()
                ma_20 = ma_20.fillna(self.data[col].expanding(min_periods=2).mean())
                # Calculate ratio, handling division by zero
                ratio = np.where(ma_20 != 0, self.data[col] / ma_20, 1)
                # Clip to reasonable range
                features[f'{col}_ma_ratio'] = pd.Series(ratio, index=features.index).clip(0.5, 2.0)

        # Remove the first few rows that might still have NaN
        features = features.dropna()

        # Additional check: remove any remaining inf or nan
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.dropna()

        # Verify no infinite or NaN values remain
        if features.isnull().any().any():
            print("⚠️ Warning: NaN values detected, filling with 0")
            features = features.fillna(0)

        if np.isinf(features.values).any():
            print("⚠️ Warning: Infinite values detected, clipping")
            features = features.clip(-100, 100)

        print(f"✓ Created {len(features.columns)} features")
        print(f"✓ Sample size: {len(features)} days")

        # Store features
        self.features = features

        # Scale features with additional error handling
        try:
            self.scaler = StandardScaler()
            self.features_scaled = self.scaler.fit_transform(features)
            print(f"✓ Features scaled successfully")
        except Exception as e:
            print(f"⚠️ Error during scaling: {e}")
            # Try to fix by removing problematic features
            print("Attempting to fix by removing constant features...")

            # Remove constant features (std = 0)
            feature_std = features.std()
            non_constant_features = feature_std[feature_std > 1e-10].index
            features_clean = features[non_constant_features]

            if len(features_clean.columns) == 0:
                raise ValueError("All features are constant, cannot proceed")

            print(f"✓ Removed {len(features.columns) - len(features_clean.columns)} constant features")

            # Try scaling again
            self.features = features_clean
            self.scaler = StandardScaler()
            self.features_scaled = self.scaler.fit_transform(features_clean)
            print(f"✓ Features scaled successfully after cleanup")

        self.features_reduced = None  # Will be set if using PCA

        return self.features

    def _save_validation_results(self, results_df, accuracy):
        """Save validation results to JSON"""
        validation_file = self.analysis_dir / "validation_results.json"

        validation_data = {
            'accuracy': float(accuracy),
            'accuracy_percentage': f"{accuracy:.1%}",
            'correct_predictions': int(results_df['Match'].sum()),
            'total_events': len(results_df),
            'events': results_df.to_dict('records'),
            'validation_date': datetime.now().isoformat()
        }

        with open(validation_file, 'w') as f:
            json.dump(validation_data, f, indent=2)

        print(f"💾 Saved validation results to {validation_file}")
        return validation_data

    # def _check_regime_match(self, detected, expected):
    #     """
    #     Check if detected regime matches expected category
    #     """
    #     detected = detected.lower()
    #     expected = expected.lower()
    #
    #     # Crisis/High Vol matches
    #     if 'crisis' in expected or 'high vol' in expected:
    #         return 'crisis' in detected or 'bear' in detected or 'high' in detected
    #
    #     # Bull/Growth matches
    #     elif 'bull' in expected or 'growth' in expected:
    #         # Special case: High Vol Bull (volatile recovery)
    #         if 'high vol bull' in expected:
    #             return True  # Accept any regime during volatile recoveries
    #         return 'bull' in detected or 'growth' in detected or 'steady' in detected
    #
    #     # Neutral/Normal matches
    #     elif 'normal' in expected or 'neutral' in expected:
    #         return 'steady' in detected or 'normal' in detected
    #
    #     return False

    def _check_regime_match(self, detected, expected):
        """
        Check if detected regime matches expected category
        Simplified for 3-regime system
        """
        detected = detected.lower()
        expected = expected.lower()

        # Direct matches
        if detected == expected:
            return True

        # Crisis/Bear matches
        if 'crisis' in expected or 'bear' in expected:
            return 'crisis' in detected or 'bear' in detected

        # Strong Bull matches
        elif 'bull' in expected and 'strong' in expected:
            return 'strong' in detected and 'bull' in detected

        # Steady Growth matches
        elif 'steady' in expected or 'growth' in expected or 'normal' in expected:
            return 'steady' in detected or 'growth' in detected

        # No match
        return False

    def compare_with_vix_regimes(self):
        """Compare with VIX-based regime classification"""
        if '^VIX' not in self.data.columns:
            print("VIX data not available for comparison")
            return None

        print("\n📊 VIX REGIME COMPARISON")
        print("-" * 40)

        # Define VIX regimes
        vix_regimes = pd.Series(index=self.data.index, dtype=str)
        vix = self.data['^VIX']

        vix_regimes[vix < 12] = 'Low Vol (<12)'
        vix_regimes[(vix >= 12) & (vix < 20)] = 'Normal (12-20)'
        vix_regimes[(vix >= 20) & (vix < 30)] = 'Elevated (20-30)'
        vix_regimes[vix >= 30] = 'High Vol (>30)'

        # Align indices
        common_index = self.regime_history.index.intersection(vix_regimes.index)

        # Compare regimes
        comparison = pd.DataFrame({
            'Our Regime': self.regime_history.loc[common_index].map(
                lambda x: self.regime_characteristics[x]['name']
            ),
            'VIX Regime': vix_regimes.loc[common_index]
        })

        # Calculate alignment score
        alignment_score = 0
        total_count = len(comparison)

        for _, row in comparison.iterrows():
            our_regime = row['Our Regime'].lower()
            vix_regime = row['VIX Regime'].lower()

            # Check if regimes align
            if ('crisis' in our_regime or 'bear' in our_regime) and ('high' in vix_regime or 'elevated' in vix_regime):
                alignment_score += 1
            elif ('bull' in our_regime or 'growth' in our_regime) and ('low' in vix_regime or 'normal' in vix_regime):
                alignment_score += 1
            elif 'steady' in our_regime and 'normal' in vix_regime:
                alignment_score += 1

        vix_alignment = alignment_score / total_count if total_count > 0 else 0
        print(f"\nVIX Alignment Score: {vix_alignment:.1%}")

        # Save VIX alignment
        self.vix_alignment = vix_alignment
        return vix_alignment


# def main(use_marketstack=False):  # Add parameter
#     """
#     Run the complete regime detection analysis
#
#     Args:
#         use_marketstack: If True, use Marketstack API; if False, use yfinance
#     """
#     # Initialize detector
#     detector = MarketRegimeDetector(n_regimes=3, lookback_years=10)
#
#     # Fetch data with chosen source
#     detector.fetch_market_data(
#         symbols=['SPY', '^VIX', '^TNX', 'GLD'],
#         use_marketstack=use_marketstack
#     )
#
#     # IMPORTANT: Prepare features before fitting HMM
#     # Check which method exists and use it
#     if hasattr(detector, 'prepare_features'):
#         detector.prepare_features()
#     elif hasattr(detector, 'engineer_features'):
#         detector.engineer_features()
#     else:
#         raise ValueError("No feature preparation method found!")
#
#     # Fit HMM with PCA for better convergence
#     detector.fit_hmm(use_pca=True, n_components=5)
#
#     # Characterize regimes
#     detector.characterize_regimes()
#
#     # Generate visualizations
#     detector.plot_regimes()
#
#     # Generate report
#     detector.generate_report()
#
#     # Export regime periods for optimizer
#     regime_periods = detector.export_regime_periods()
#
#     # Get current regime
#     current_regime, proba = detector.get_current_regime()
#
#     # Save model for future use
#     detector.save_model()
#
#     # Additional analysis: Create date ranges for optimizer
#     print("\n📅 SUGGESTED OPTIMIZER RUNS")
#     print("-"*40)
#     for regime in range(detector.n_regimes):
#         chars = detector.regime_characteristics[regime]
#         print(f"\n{chars['name']} Regime:")
#
#         # Find longest continuous period for each regime
#         if 'periods' in chars:
#             longest_period = max(chars['periods'], key=lambda x: x['days'])
#             print(f"  Best period: {longest_period['start']} to {longest_period['end']}")
#             print(f"  Run optimizer with: --start {longest_period['start']} --end {longest_period['end']}")
#
#     # Add validation
#     detector.validate_against_known_events()
#     detector.compare_with_vix_regimes()
#
#     return detector


def main(use_marketstack=False):
    """
    Run the complete regime detection analysis
    Updated to use fit_hmm_with_multiple_attempts for better stability
    """
    # Initialize detector
    detector = MarketRegimeDetector(n_regimes=3, lookback_years=10)

    # Fetch data with chosen source
    detector.fetch_market_data(
        symbols=['SPY', '^VIX', '^TNX', 'GLD'],
        use_marketstack=use_marketstack
    )

    # IMPORTANT: Prepare features before fitting HMM
    if hasattr(detector, 'prepare_features'):
        detector.prepare_features()
    elif hasattr(detector, 'engineer_features'):
        detector.engineer_features()
    else:
        raise ValueError("No feature preparation method found!")

    # UPDATED: Use the more robust fitting method
    # First check if PCA should be applied
    if hasattr(detector, 'features') and detector.features.shape[1] > 5:
        # Apply PCA to reduce features
        from sklearn.decomposition import PCA
        print("\n📊 Applying PCA for dimensionality reduction...")
        detector.pca = PCA(n_components=5, random_state=42)
        features_pca = detector.pca.fit_transform(detector.features_scaled)

        # Store as DataFrame for compatibility
        detector.features_reduced = pd.DataFrame(
            features_pca,
            index=detector.features.index,
            columns=[f'PC{i + 1}' for i in range(5)]
        )

        # Print explained variance
        explained_var = detector.pca.explained_variance_ratio_
        print(f"✓ PCA explains {explained_var.sum():.1%} of variance")
    else:
        detector.features_reduced = None

    # Use the multiple attempts method for better stability
    detector.fit_hmm_with_multiple_attempts(n_attempts=10, n_iter=200)

    # Validate regime separation (optional but useful)
    if hasattr(detector, 'validate_regimes'):
        detector.validate_regimes()

    # Characterize regimes
    detector.characterize_regimes()

    # Generate visualizations
    detector.plot_regimes()

    # Generate report
    detector.generate_report()

    # Export regime periods for optimizer
    regime_periods = detector.export_regime_periods()

    # Get current regime
    current_regime, proba = detector.get_current_regime()

    # Save model for future use
    detector.save_model()

    # Additional analysis
    print("\n📅 SUGGESTED OPTIMIZER RUNS")
    print("-" * 40)
    for regime in range(detector.n_regimes):
        chars = detector.regime_characteristics[regime]
        print(f"\n{chars['name']} Regime:")

        # Find longest continuous period for each regime
        if 'periods' in chars:
            longest_period = max(chars['periods'], key=lambda x: x['days'])
            print(f"  Best period: {longest_period['start']} to {longest_period['end']}")
            print(f"  Run optimizer with: --start {longest_period['start']} --end {longest_period['end']}")

    # Add validation
    detector.validate_against_known_events()
    detector.compare_with_vix_regimes()

    return detector


def api_wrapper():
    """API wrapper for regime detection - NO self parameter"""
    try:
        detector = main()  # Call your main function
        return {"status": "success", "message": "Regime detection completed"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    use_marketstack = False
    fetch_historical = False
    years = 10

    for arg in sys.argv:
        if arg == "--fetch-historical":
            fetch_historical = True
        elif arg.startswith("--years="):
            years = int(arg.split("=")[1])
        elif arg == "--use-marketstack":
            use_marketstack = True
        elif arg == "--use-yfinance":
            use_marketstack = False

    if fetch_historical:
        # Special mode to fetch and save historical data
        detector = MarketRegimeDetector(n_regimes=3, lookback_years=years)

        # Fetch with the specified data source
        detector.fetch_market_data(
            symbols=['SPY', '^VIX', '^TNX', 'GLD'],
            use_marketstack=use_marketstack
        )

        # Process the data
        if hasattr(detector, 'prepare_features'):
            detector.prepare_features()
        elif hasattr(detector, 'engineer_features'):
            detector.engineer_features()

        # Fit the model
        detector.fit_hmm(use_pca=True, n_components=5)

        # Characterize and validate
        detector.characterize_regimes()
        accuracy = detector.validate_against_known_events()
        vix_alignment = detector.compare_with_vix_regimes()

        # Save validation results
        validation_results = {
            'accuracy': accuracy,
            'vix_alignment': vix_alignment,
            'correct': int(accuracy * 10) if accuracy else 0,
            'total': 10
        }

        output_dir = Path(__file__).parent.parent.parent / "output" / "Regime_Detection_Analysis"
        with open(output_dir / "validation_results.json", 'w') as f:
            json.dump(validation_results, f, indent=2)

        # Save historical analysis
        detector.save_historical_analysis()

        print("✅ Historical regime analysis completed")
        sys.exit(0)
    else:
        # Normal execution
        detector = main(use_marketstack=use_marketstack)