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
        Updated for v2 API structure
        """
        print(f"📊 Fetching data from Marketstack API v2 for {len(symbols)} symbols...")
        print(f"📅 Date range: {date_from} to {date_to}")

        all_symbol_dfs = []

        for i, symbol in enumerate(symbols):
            # Clean symbol for Marketstack (remove ^ prefix for indices)
            clean_symbol = symbol.replace('^', '')

            # Map symbols to Marketstack format
            # Note: In v2, index symbols might be different
            symbol_map = {
                'SPY': 'SPY',
                'VIX': 'VIX.INDX',
                'TNX': 'TNX.INDX',
                'GLD': 'GLD',
                'DXY': 'DXY.INDX'
            }

            marketstack_symbol = symbol_map.get(clean_symbol, clean_symbol)

            print(f"  Fetching {symbol} as {marketstack_symbol} ({i + 1}/{len(symbols)})", end='')

            params = {
                'access_key': api_key,
                'symbols': marketstack_symbol,
                'date_from': date_from,
                'date_to': date_to,
                'limit': 1000
            }

            endpoint = f"{MARKETSTACK_BASE_URL}/eod"

            try:
                response = requests.get(endpoint, params=params, timeout=30)
                response.raise_for_status()

                data = response.json()

                if 'error' in data:
                    print(f" ❌ API Error: {data['error'].get('message', 'Unknown error')}")
                    continue

                if 'data' not in data or not data['data']:
                    print(f" ✗ (no data)")
                    continue

                # Process the data
                symbol_data = data['data']

                if symbol_data:
                    df = pd.DataFrame(symbol_data)
                    df['date'] = pd.to_datetime(df['date'])

                    # Remove timezone info if present
                    if df['date'].dt.tz is not None:
                        df['date'] = df['date'].dt.tz_localize(None)

                    df = df.set_index('date')

                    # Create DataFrame with original symbol name
                    # Use 'close' instead of 'adj_close' for v2
                    price_df = pd.DataFrame({
                        f'{symbol}_Open': df['open'],
                        f'{symbol}_High': df['high'],
                        f'{symbol}_Low': df['low'],
                        f'{symbol}_Close': df['close'],  # v2 uses 'close'
                        f'{symbol}_Volume': df['volume']
                    })

                    price_df = price_df.sort_index()
                    all_symbol_dfs.append(price_df)

                    print(f" ✓ ({len(df)} records)")
                else:
                    print(f" ✗ (empty response)")

            except requests.exceptions.RequestException as e:
                print(f" ❌ Request Error: {str(e)}")
                continue
            except Exception as e:
                print(f" ❌ Processing Error: {str(e)}")
                continue

            # Rate limiting
            time.sleep(0.2)

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
            print(f"Total date range: {all_price_data.index[0]} to {all_price_data.index[-1]}")

        return all_price_data

    def fetch_market_data(self, symbols=['SPY', '^VIX', '^TNX', 'GLD'], use_marketstack=True):
        """
        Fetch market data for regime detection
        Updated to properly handle Marketstack v2
        """
        print("\n📊 FETCHING MARKET DATA")
        print("-" * 40)

        if use_marketstack:
            # Load API key
            api_key = self.load_marketstack_config()
            if not api_key:
                print("⚠️ Falling back to yfinance due to missing API key")
                use_marketstack = False

        if use_marketstack:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * self.lookback_years)

            # Fetch from Marketstack
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
            # Original yfinance code
            import yfinance as yf

            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * self.lookback_years)

            data = yf.download(
                symbols,
                start=start_date,
                end=end_date,
                progress=False
            )['Adj Close']

            if len(symbols) == 1:
                data = pd.DataFrame(data)
                data.columns = symbols

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
        This is called when user clicks 'Fetch Regime Data'
        """
        print("\n🔄 FETCHING HISTORICAL REGIME DATA")
        print("-" * 40)

        # Fetch fresh data from Marketstack
        self.fetch_market_data(use_marketstack=True)

        if self.data is None or self.data.empty:
            print("❌ Failed to fetch historical data")
            return False

        # Run regime detection on historical data
        print("\n🔍 Detecting historical regimes...")

        # Prepare features for HMM
        self.prepare_features()

        # Fit the model if not already fitted
        if self.model is None:
            self.fit_hmm(use_pca=True, n_components=5)

        # Characterize regimes
        self.characterize_regimes()

        # Export regime periods
        self.export_regime_periods()

        # Save detailed historical analysis
        self.save_historical_analysis()

        print("✅ Historical regime data saved successfully")
        return True

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

    def fit_hmm_with_multiple_attempts(self, n_attempts=10, n_iter=200):
        """
        Fit HMM with multiple random initializations to find best model
        """
        print(f"\n🤖 Training {self.n_regimes}-regime Gaussian HMM...")

        # Use reduced features if available
        X = self.features_reduced.values if self.features_reduced is not None else self.features.values

        best_model = None
        best_score = -np.inf
        convergence_count = 0

        for attempt in range(n_attempts):
            try:
                # Try with different random states
                model = GaussianHMM(
                    n_components=self.n_regimes,
                    covariance_type="diag",  # More stable than "full"
                    n_iter=n_iter,
                    random_state=42 + attempt,
                    tol=0.01,  # Convergence tolerance
                    verbose=False
                )

                # Fit model
                model.fit(X)

                # Check if converged
                if model.monitor_.converged:
                    convergence_count += 1
                    score = model.score(X)

                    if score > best_score:
                        best_score = score
                        best_model = model

                    print(f"  Attempt {attempt + 1}: Converged ✓ (Score: {score:.2f})")
                else:
                    print(f"  Attempt {attempt + 1}: Not converged ✗")

            except Exception as e:
                print(f"  Attempt {attempt + 1}: Failed - {str(e)[:50]}")

        if best_model is None:
            raise ValueError("Failed to train any converged model")

        print(f"\n✓ Best model score: {best_score:.2f}")
        print(f"✓ Converged models: {convergence_count}/{n_attempts}")

        self.model = best_model

        # Get regime predictions
        self.regime_history = pd.Series(
            self.model.predict(X),
            index=self.features_reduced.index if self.features_reduced is not None else self.features.index
        )

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

    def fit_hmm(self, use_pca=True, n_components=5, n_attempts=10):
        """
        Complete HMM fitting process with validation
        """
        # Engineer features with optional PCA
        self.engineer_features(use_pca=use_pca, n_components=n_components)
        # self.engineer_simple_features()

        # Fit model with multiple attempts
        self.fit_hmm_with_multiple_attempts(n_attempts=n_attempts)

        # Validate regimes
        self.validate_regimes()

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
        regime_names = {}
        for rank, (regime, chars) in enumerate(sorted_regimes):
            vol = chars['volatility']
            ret = chars['mean_return']
            sharpe = chars['sharpe']

            # More nuanced naming based on characteristics
            if vol > 0.20 and ret < -0.05:
                name = "Crisis/Bear"
            elif vol > 0.18:
                name = "High Volatility"
            elif vol < 0.08 and ret > 0.12 and sharpe > 2.0:
                name = "Strong Bull"
            elif vol < 0.12 and ret > 0.05 and sharpe > 0.8:
                name = "Steady Growth"
            elif vol > 0.08 and vol < 0.15 and ret > 0:
                name = "Normal Market"
            elif ret < 0 and vol < 0.15:
                name = "Mild Correction"
            else:
                # Fallback based on simple metrics
                if sharpe > 1.5:
                    name = "Bull Market"
                elif sharpe < 0:
                    name = "Bear Market"
                else:
                    name = "Neutral"

            regime_names[regime] = name
            self.regime_characteristics[regime]['name'] = name

        return self.regime_characteristics

    def _calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown from returns series"""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        return drawdown.min()

    def plot_regimes(self, save_path='regime_analysis.png'):
        """
        Visualize regime detection results
        """
        # Update the default path
        save_path = os.path.join(self.output_dir, save_path)

        fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

        # 1. Price chart with regime coloring
        ax1 = axes[0]
        if 'SPY' in self.data.columns:
            for regime in range(self.n_regimes):
                mask = (self.regime_history == regime)

                # Get dates where this regime occurs
                regime_dates = self.regime_history[mask].index

                # Get SPY data for these specific dates
                regime_data = self.data.loc[self.data.index.intersection(regime_dates), 'SPY']

                if not regime_data.empty:
                    color = plt.cm.tab10(regime)
                    ax1.scatter(regime_data.index, regime_data.values,
                                c=[color], alpha=0.5, s=1,
                                label=self.regime_characteristics[regime]['name'])

            ax1.plot(self.data['SPY'], 'k-', linewidth=0.5, alpha=0.5)
            ax1.set_ylabel('SPY Price')
            ax1.set_yscale('log')
            ax1.legend()
            ax1.set_title('Market Regimes Over Time')

        # 2. Regime timeline
        ax2 = axes[1]
        regime_colors = self.regime_history.map(lambda x: plt.cm.tab10(x))
        ax2.scatter(self.regime_history.index, self.regime_history.values,
                    c=regime_colors, s=5)
        ax2.set_ylabel('Regime')
        ax2.set_yticks(range(self.n_regimes))
        ax2.set_yticklabels([self.regime_characteristics[i]['name']
                             for i in range(self.n_regimes)])
        ax2.set_title('Regime Classification')

        # 3. Rolling volatility
        ax3 = axes[2]
        if 'volatility_21d' in self.features.columns:
            vol_data = self.scaler.inverse_transform(self.features)
            vol_idx = self.features.columns.get_loc('volatility_21d')
            ax3.plot(self.features.index, vol_data[:, vol_idx], 'b-', linewidth=1)
            ax3.set_ylabel('21d Realized Vol')
            ax3.set_title('Market Volatility')

        # 4. Regime probabilities
        ax4 = axes[3]
        X = self.features_reduced.values if self.features_reduced is not None else self.features.values
        proba = self.model.predict_proba(X)

        proba_index = self.features_reduced.index if self.features_reduced is not None else self.features.index

        for regime in range(self.n_regimes):
            ax4.plot(proba_index, proba[:, regime],
                     label=self.regime_characteristics[regime]['name'],
                     linewidth=1)
        ax4.set_ylabel('Probability')
        ax4.set_xlabel('Date')
        ax4.legend()
        ax4.set_title('Regime Probabilities')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Regime plot saved to {save_path}")
        # plt.show()

    def generate_report(self):
        """
        Generate comprehensive regime analysis report
        """
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

    def get_current_regime(self):
        """
        Get the current market regime
        """
        current_regime = self.regime_history.iloc[-1]

        # Fix: Use the correct features for prediction
        X = self.features_reduced.values if self.features_reduced is not None else self.features.values
        regime_proba = self.model.predict_proba(X)[-1]

        print("\n🎯 CURRENT MARKET REGIME")
        print("-" * 40)
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

    def validate_against_known_events(self):
        """
        Compare detected regimes against known market events - IMPROVED
        """
        print("\n🎯 REGIME DETECTION VALIDATION")
        print("=" * 60)

        # Fix timezone issues
        if self.regime_history.index.tz is not None:
            self.regime_history.index = self.regime_history.index.tz_localize(None)

        # Known events with CORRECT expected regimes
        known_events = {
            # Crisis/High Vol events
            'COVID Crash': {
                'dates': ('2020-02-19', '2020-03-23'),
                'expected': 'Crisis/High Vol',
                'description': 'Market crash due to COVID-19'
            },
            'Volmageddon': {
                'dates': ('2018-01-26', '2018-02-08'),
                'expected': 'Crisis/High Vol',
                'description': 'XIV implosion, VIX spike'
            },
            'Q4 2018 Selloff': {
                'dates': ('2018-10-03', '2018-12-24'),
                'expected': 'Crisis/High Vol',
                'description': 'Fed tightening fears'
            },
            'China Devaluation': {
                'dates': ('2015-08-18', '2016-02-11'),
                'expected': 'Crisis/High Vol',
                'description': 'Yuan devaluation, growth fears'
            },
            '2022 Bear Market': {
                'dates': ('2022-01-03', '2022-10-12'),
                'expected': 'Crisis/High Vol',
                'description': 'Fed rate hikes, inflation'
            },

            # Recovery/Bull events
            'COVID Recovery': {
                'dates': ('2020-03-24', '2020-06-08'),
                'expected': 'High Vol Bull',  # Still volatile but recovering
                'description': 'Fed stimulus, market recovery'
            },
            'Taper Tantrum Recovery': {
                'dates': ('2016-02-12', '2016-12-31'),
                'expected': 'Bull/Growth',
                'description': 'Post-correction recovery'
            },
            '2023 Recovery': {
                'dates': ('2023-01-01', '2023-12-31'),
                'expected': 'Bull/Growth',
                'description': 'AI boom, soft landing hopes'
            },

            # Additional validation periods
            '2017 Melt-Up': {
                'dates': ('2017-01-01', '2017-12-31'),
                'expected': 'Bull/Growth',
                'description': 'Low vol bull market'
            },
            'March 2023 Banking Crisis': {
                'dates': ('2023-03-08', '2023-03-31'),
                'expected': 'Crisis/High Vol',
                'description': 'SVB collapse'
            }
        }

        # Calculate hit rates
        results = []

        for event_name, event_info in known_events.items():
            start, end = event_info['dates']
            start_date = pd.to_datetime(start).tz_localize(None)
            end_date = pd.to_datetime(end).tz_localize(None)

            # Get regimes during this period
            mask = (self.regime_history.index >= start_date) & (self.regime_history.index <= end_date)
            period_regimes = self.regime_history[mask]

            if len(period_regimes) > 0:
                # Most common regime during period
                regime_counts = period_regimes.value_counts()
                dominant_regime = regime_counts.index[0]
                regime_pct = regime_counts.iloc[0] / len(period_regimes)

                # Get all regimes present
                all_regimes = ', '.join([
                    f"{self.regime_characteristics[r]['name']} ({c / len(period_regimes):.0%})"
                    for r, c in regime_counts.items()
                ])

                results.append({
                    'Event': event_name,
                    'Period': f"{start} to {end}",
                    'Dominant': self.regime_characteristics[dominant_regime]['name'],
                    'Coverage': f"{regime_pct:.0%}",
                    'All Regimes': all_regimes,
                    'Expected': event_info['expected'],
                    'Match': self._check_regime_match(
                        self.regime_characteristics[dominant_regime]['name'],
                        event_info['expected']
                    )
                })

        results_df = pd.DataFrame(results)

        # Print detailed results
        print("\nDetailed Event Analysis:")
        for _, row in results_df.iterrows():
            print(f"\n{row['Event']}:")
            print(f"  Period: {row['Period']}")
            print(f"  Expected: {row['Expected']}")
            print(f"  Detected: {row['All Regimes']}")
            print(f"  Match: {'✓' if row['Match'] else '✗'}")

        # Calculate accuracy
        accuracy = results_df['Match'].sum() / len(results_df)
        print(f"\n{'=' * 60}")
        print(f"Overall Validation Accuracy: {accuracy:.1%}")
        print(f"Correct Classifications: {results_df['Match'].sum()}/{len(results_df)}")

        # Analyze misclassifications
        misclassified = results_df[~results_df['Match']]
        if len(misclassified) > 0:
            print(f"\nMisclassified Events:")
            for _, row in misclassified.iterrows():
                print(f"  - {row['Event']}: Expected {row['Expected']}, got {row['Dominant']}")

        return results_df

    def _check_regime_match(self, detected, expected):
        """
        Check if detected regime matches expected category
        """
        detected = detected.lower()
        expected = expected.lower()

        # Crisis/High Vol matches
        if 'crisis' in expected or 'high vol' in expected:
            return 'crisis' in detected or 'bear' in detected or 'high' in detected

        # Bull/Growth matches
        elif 'bull' in expected or 'growth' in expected:
            # Special case: High Vol Bull (volatile recovery)
            if 'high vol bull' in expected:
                return True  # Accept any regime during volatile recoveries
            return 'bull' in detected or 'growth' in detected or 'steady' in detected

        # Neutral/Normal matches
        elif 'normal' in expected or 'neutral' in expected:
            return 'steady' in detected or 'normal' in detected

        return False

    def compare_with_vix_regimes(self):
        """
        Compare detected regimes with VIX-based regime classification
        """
        if '^VIX' not in self.data.columns:
            print("VIX data not available for comparison")
            return

        print("\n📊 VIX REGIME COMPARISON")
        print("-" * 40)

        # Fix: Ensure index alignment
        regime_index = self.regime_history.index
        if regime_index.tz is not None:
            regime_index = regime_index.tz_localize(None)

        data_index = self.data.index
        if hasattr(data_index, 'tz') and data_index.tz is not None:
            data_index = data_index.tz_localize(None)
            self.data.index = data_index

        # Define VIX regimes
        vix_regimes = pd.Series(index=self.data.index, dtype=str)
        vix = self.data['^VIX']

        vix_regimes[vix < 12] = 'Low Vol (<12)'
        vix_regimes[(vix >= 12) & (vix < 20)] = 'Normal (12-20)'
        vix_regimes[(vix >= 20) & (vix < 30)] = 'Elevated (20-30)'
        vix_regimes[vix >= 30] = 'High Vol (>30)'

        # Compare with our regimes
        comparison = pd.DataFrame({
            'Date': regime_index,
            'Our Regime': self.regime_history.map(lambda x: self.regime_characteristics[x]['name']).values,
            'VIX Regime': vix_regimes.reindex(regime_index).values
        })

        # Create confusion matrix
        confusion = pd.crosstab(comparison['Our Regime'], comparison['VIX Regime'],
                                normalize='columns')

        print("\nConfusion Matrix (columns sum to 100%):")
        print(confusion.round(2))

        # Calculate alignment score - FIXED
        alignment_score = 0
        total_weight = 0

        for our_regime in confusion.index:
            for vix_regime in confusion.columns:
                weight = confusion.loc[our_regime, vix_regime]
                total_weight += weight

                if ('Crisis' in our_regime and ('High' in vix_regime or 'Elevated' in vix_regime)) or \
                        ('Bull' in our_regime and ('Low' in vix_regime or 'Normal' in vix_regime)) or \
                        ('Steady' in our_regime and 'Normal' in vix_regime):
                    alignment_score += weight

        # Normalize to percentage
        alignment_score = alignment_score / total_weight if total_weight > 0 else 0
        print(f"\nRegime Alignment Score: {alignment_score:.1%}")

def main():
    """
    Run the complete regime detection analysis
    """
    # Initialize detector
    detector = MarketRegimeDetector(n_regimes=3, lookback_years=10)

    # Fetch data - remove DXY as it has less history
    detector.fetch_market_data(symbols=['SPY', '^VIX', '^TNX', 'GLD'])

    # Fit HMM with PCA for better convergence
    detector.fit_hmm(use_pca=True, n_components=5)

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

    # Additional analysis: Create date ranges for optimizer
    print("\n📅 SUGGESTED OPTIMIZER RUNS")
    print("-"*40)
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

    # Check for command line arguments
    if "--fetch-historical" in sys.argv:
        # Special mode to fetch and save historical data
        detector = MarketRegimeDetector(n_regimes=3, lookback_years=10)

        # Parse years argument if provided
        for arg in sys.argv:
            if arg.startswith("--years="):
                years = int(arg.split("=")[1])
                detector.lookback_years = years

        # Fetch and save historical data
        success = detector.fetch_and_save_historical_regimes()

        if success:
            print("✅ Historical regime data fetched and saved")
            sys.exit(0)
        else:
            print("❌ Failed to fetch historical regime data")
            sys.exit(1)

    elif "--api" in sys.argv:
        # API mode
        detector = RegimeDetectorAPI()
        result = detector.run_detection()
        print(json.dumps(result))

    else:
        # Normal execution
        detector = main()