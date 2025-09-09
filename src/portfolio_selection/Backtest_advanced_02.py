"""
Comprehensive MHRP Portfolio Optimization and Backtesting System

This script provides:
1. MHRP (Modified Hierarchical Risk Parity) optimization
2. Multiple rebalancing strategies (Monthly, Drift-based, Volatility-adjusted)
3. Walk-forward testing with validation
4. Advanced analytics including factor attribution, correlation analysis, and tail risk metrics

Author: Portfolio Optimization System
Version: 1.1 - Fixed VectorBT compatibility
V02: Apply Marketstack data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.covariance import LedoitWolf
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy import stats
import vectorbt as vbt
import quantstats as qs
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings
import seaborn as sns
from sklearn.linear_model import LinearRegression
import json
from marketstack_integration import MarketstackDataFetcher

warnings.filterwarnings('ignore')

class MHRPPortfolioSystem:
    """
    Comprehensive MHRP Portfolio Optimization and Backtesting System
    """

    def __init__(self,
                 initial_cash: float = 100_000,
                 fees: float = 0.001,
                 slippage: float = 0.0005,
                 marketstack_api_key: str = None):

        self.initial_cash = initial_cash
        self.fees = fees
        self.slippage = slippage
        self.results = {}
        self.analytics = {}
        self.marketstack_api_key = marketstack_api_key

        # Initialize Marketstack fetcher if API key provided
        if marketstack_api_key:
            self.data_fetcher = MarketstackDataFetcher(marketstack_api_key)

    def fetch_prices(self,
                    tickers: List[str],
                    lookback_days: int = 252,
                    max_retries: int = 3,
                    pause: float = 1.0) -> Tuple[pd.DataFrame, List[str]]:
        """
        Robust price data download with retry mechanism

        Returns:
            Tuple of (prices DataFrame, list of failed tickers)
        """
        # Use Marketstack if available
        if hasattr(self, 'data_fetcher'):
            print("🔑 Using Marketstack API for data fetching...")
            return self.data_fetcher.fetch_prices_with_lookback(
                tickers,
                datetime.today().strftime("%Y-%m-%d"),
                None,
                lookback_days
            )
        else:
            # Keep your original yfinance code as fallback
            print("📊 Using yfinance for data fetching...")


        end = datetime.today()
        start = end - timedelta(days=int(lookback_days * 1.4))
        tries = 0

        while tries < max_retries:
            try:
                data = yf.download(
                    tickers,
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    auto_adjust=True,
                    progress=False,
                    group_by="column",
                    threads=True,
                )

                # Handle single vs multiple tickers
                if isinstance(data.columns, pd.MultiIndex):
                    close = data["Close"]
                else:
                    close = pd.DataFrame(data["Close"])
                    close.columns = [tickers[0]]

                # Forward fill and check for missing data
                close = close.ffill().dropna(how='all')
                missing_cols = close.columns[close.isna().all()].tolist()

                if not missing_cols:
                    break
                else:
                    tries += 1
                    print(f"⚠️  Missing data for {missing_cols}... retrying ({tries}/{max_retries})")
                    tickers = [t for t in tickers if t not in missing_cols]

            except Exception as e:
                tries += 1
                print(f"Error downloading data: {e}")
                if tries >= max_retries:
                    raise

        # Final cleanup
        prices = close.ffill(limit=2).dropna(axis=0, how='any')
        failed_tickers = [t for t in tickers if t not in prices.columns]

        return prices, failed_tickers

    # def calculate_mhrp_weights(self,
    #                           returns: pd.DataFrame,
    #                           use_shrinkage: bool = True) -> pd.Series:
    #     """
    #     Calculate MHRP weights using equal volatility recursive bisection
    #
    #     Args:
    #         returns: Daily returns DataFrame
    #         use_shrinkage: Whether to use Ledoit-Wolf shrinkage for covariance
    #
    #     Returns:
    #         Series of optimized weights
    #     """
    #
    #     # Step 1: Calculate covariance matrix
    #     if use_shrinkage:
    #         lw = LedoitWolf()
    #         cov_matrix = lw.fit(returns).covariance_
    #         cov = pd.DataFrame(cov_matrix, index=returns.columns, columns=returns.columns)
    #     else:
    #         cov = returns.cov()
    #
    #     # Step 2: Calculate correlation for clustering
    #     cor = returns.corr(method='spearman')
    #
    #     # Step 3: Distance matrix and hierarchical clustering
    #     distance = np.sqrt(0.5 * (1 - cor))
    #     link = linkage(squareform(distance), method='ward')
    #
    #     # Step 4: Get quasi-diagonal order
    #     ticker_order = self._get_quasi_diag(link)
    #     sorted_tickers = [returns.columns[i] for i in ticker_order]
    #
    #     # Step 5: Apply equal volatility recursive bisection
    #     weights = self._recursive_bisection_equal_volatility(cov, sorted_tickers)
    #     weights = weights / weights.sum()  # Normalize
    #
    #     return weights

    def calculate_mhrp_weights(self,
                               returns: pd.DataFrame,
                               use_shrinkage: bool = True) -> pd.Series:
        """
        Calculate MHRP weights using equal volatility recursive bisection

        Args:
            returns: Daily returns DataFrame
            use_shrinkage: Whether to use Ledoit-Wolf shrinkage for covariance

        Returns:
            Series of optimized weights
        """

        # Add data validation
        if returns.empty or len(returns) < 2:
            print("⚠️ Insufficient data for MHRP optimization")
            # Return equal weights as fallback
            n = len(returns.columns)
            return pd.Series(1.0 / n, index=returns.columns)

        # Remove any columns with all NaN or zero variance
        valid_cols = []
        for col in returns.columns:
            if not returns[col].isna().all() and returns[col].std() > 1e-10:
                valid_cols.append(col)

        if len(valid_cols) < len(returns.columns):
            print(f"⚠️ Removed {len(returns.columns) - len(valid_cols)} assets with insufficient data")
            returns = returns[valid_cols]

        if len(valid_cols) < 2:
            print("⚠️ Not enough valid assets for optimization")
            return pd.Series(1.0 / len(valid_cols), index=valid_cols) if valid_cols else pd.Series()

        # Step 1: Calculate covariance matrix
        if use_shrinkage:
            try:
                lw = LedoitWolf()
                cov_matrix = lw.fit(returns).covariance_
                cov = pd.DataFrame(cov_matrix, index=returns.columns, columns=returns.columns)
            except Exception as e:
                print(f"⚠️ Ledoit-Wolf failed: {e}, using standard covariance")
                cov = returns.cov()
        else:
            cov = returns.cov()

        # Step 2: Calculate correlation for clustering
        cor = returns.corr(method='spearman')

        # Check for NaN values in correlation matrix
        if cor.isna().any().any():
            print("⚠️ NaN values in correlation matrix, filling with 0")
            cor = cor.fillna(0)
            # Ensure diagonal is 1
            np.fill_diagonal(cor.values, 1)

        # Step 3: Distance matrix and hierarchical clustering
        distance = np.sqrt(np.clip(0.5 * (1 - cor), 0, 1))  # Clip to ensure valid range

        # Ensure the distance matrix is symmetric
        distance = (distance + distance.T) / 2

        # Convert to condensed form for linkage
        from scipy.spatial.distance import squareform
        try:
            condensed_distance = squareform(distance, checks=True)
        except ValueError as e:
            print(f"⚠️ Distance matrix issue: {e}, attempting to fix")
            # Force symmetry and valid values
            distance = np.nan_to_num(distance, nan=0.5)
            np.fill_diagonal(distance.values, 0)
            distance = (distance + distance.T) / 2
            condensed_distance = squareform(distance, checks=False)

        link = linkage(condensed_distance, method='ward')

        # Continue with the rest of the method...
        # Step 4: Get quasi-diagonal order
        ticker_order = self._get_quasi_diag(link)
        sorted_tickers = [returns.columns[i] for i in ticker_order]

        # Step 5: Apply equal volatility recursive bisection
        weights = self._recursive_bisection_equal_volatility(cov, sorted_tickers)

        return weights

    def _get_quasi_diag(self, link: np.ndarray) -> List[int]:
        """Get quasi-diagonal order from linkage matrix"""
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link.shape[0] + 1

        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = (df0.values - num_items).astype(int)
            sort_ix[i] = link[j, 0]
            df1 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df1])
            sort_ix = sort_ix.sort_index()

        return sort_ix.astype(int).tolist()

    def _get_cluster_volatility(self, cov: pd.DataFrame, cluster_items: List[str]) -> float:
        """Calculate cluster volatility for equal volatility weighting"""
        sub_cov = cov.loc[cluster_items, cluster_items]
        vol = np.sqrt(np.diag(sub_cov))
        inv_vol = 1 / vol
        weights = inv_vol / inv_vol.sum()
        cluster_vol = np.sqrt(np.dot(weights, np.dot(sub_cov.values, weights)))
        return cluster_vol

    def _recursive_bisection_equal_volatility(self,
                                            cov: pd.DataFrame,
                                            sort_ix: List[str]) -> pd.Series:
        """Modified recursive bisection using equal volatility"""
        weights = pd.Series(1.0, index=sort_ix, dtype='float64')
        clusters = [sort_ix]

        while clusters:
            cluster = clusters.pop(0)
            if len(cluster) <= 1:
                continue

            split = int(len(cluster) / 2)
            left = cluster[:split]
            right = cluster[split:]

            vol_left = self._get_cluster_volatility(cov, left)
            vol_right = self._get_cluster_volatility(cov, right)

            alpha = 1 - vol_left / (vol_left + vol_right)
            weights[left] *= alpha
            weights[right] *= 1 - alpha

            clusters += [left, right]

        return weights

    def run_backtest_strategies(self,
                               tickers: List[str],
                               start_date: str = "2020-01-01",
                               end_date: Optional[str] = None,
                               lookback_days: int = 252,
                               rebalance_strategies: List[str] = ['monthly', 'drift', 'volatility']) -> Dict:
        """
        Run backtests with different rebalancing strategies

        Args:
            tickers: List of stock symbols
            start_date: Start date for backtesting
            end_date: End date for backtesting
            lookback_days: Days of historical data for optimization
            rebalance_strategies: List of rebalancing strategies to test

        Returns:
            Dictionary containing results for each strategy
        """

        print(f"🔍 Downloading price data for {len(tickers)} tickers...")
        # Get extended data for walk-forward testing
        extended_start = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=lookback_days * 2)

        all_prices, failed = self.fetch_prices(
            tickers,
            lookback_days=int((datetime.today() - extended_start).days)
        )

        if failed:
            print(f"⚠️  Failed to download data for: {failed}")
            tickers = [t for t in tickers if t not in failed]

        # Filter dates
        start_idx = all_prices.index >= start_date
        if end_date:
            end_idx = all_prices.index <= end_date
            backtest_prices = all_prices.loc[start_idx & end_idx]
        else:
            backtest_prices = all_prices.loc[start_idx]

        results = {}

        # Strategy 1: Monthly Rebalancing
        if 'monthly' in rebalance_strategies:
            print("\n📅 Running Monthly Rebalancing Strategy...")
            results['Monthly'] = self._run_monthly_rebalancing(
                all_prices, backtest_prices, lookback_days
            )

        # Strategy 2: Drift-based Rebalancing
        if 'drift' in rebalance_strategies:
            print("\n🎯 Running Drift-based Rebalancing Strategy...")
            results['Drift'] = self._run_drift_rebalancing(
                all_prices, backtest_prices, lookback_days, drift_threshold=0.05
            )

        # Strategy 3: Volatility-adjusted Rebalancing
        if 'volatility' in rebalance_strategies:
            print("\n📊 Running Volatility-adjusted Rebalancing Strategy...")
            results['Volatility'] = self._run_volatility_rebalancing(
                all_prices, backtest_prices, lookback_days
            )

        self.results = results
        return results

    def _run_monthly_rebalancing(self,
                                 all_prices: pd.DataFrame,
                                 backtest_prices: pd.DataFrame,
                                 lookback_days: int) -> Dict:
        """Run monthly rebalancing strategy with manual portfolio calculation"""

        # Create monthly rebalancing dates
        rebal_dates = pd.date_range(
            start=backtest_prices.index[0],
            end=backtest_prices.index[-1],
            freq='BM'  # Business month end
        ).intersection(backtest_prices.index)

        # Calculate weights for each rebalancing date
        time_varying_weights = {}

        for date in rebal_dates:
            # Get lookback data
            end_idx = all_prices.index.get_loc(date)
            start_idx = max(0, end_idx - lookback_days)
            lookback_data = all_prices.iloc[start_idx:end_idx]

            if len(lookback_data) < lookback_days // 2:
                continue

            returns = lookback_data.pct_change().dropna()
            weights = self.calculate_mhrp_weights(returns)
            time_varying_weights[date] = weights

        # Manual portfolio simulation
        portfolio_values = []
        portfolio_weights = []
        returns_series = []

        # Initialize
        current_weights = None
        portfolio_value = self.initial_cash

        for i, date in enumerate(backtest_prices.index):
            current_prices = backtest_prices.loc[date]

            # Check if it's a rebalancing date
            if date in time_varying_weights:
                current_weights = time_varying_weights[date]
                # Align weights with available assets
                current_weights = current_weights.reindex(backtest_prices.columns).fillna(0.0)
                # Normalize weights
                current_weights = current_weights / current_weights.sum()
            elif current_weights is None:
                # Use equal weights for the first period if no rebalancing date yet
                current_weights = pd.Series(1.0 / len(backtest_prices.columns),
                                            index=backtest_prices.columns)

            # Calculate daily returns
            if i > 0:
                prev_prices = backtest_prices.iloc[i - 1]
                asset_returns = (current_prices / prev_prices - 1).fillna(0)
                # Calculate portfolio return
                portfolio_return = (current_weights * asset_returns).sum()
                # Apply fees (approximate - only on rebalancing dates)
                if date in time_varying_weights:
                    portfolio_return -= self.fees  # Simplified fee application
                portfolio_value *= (1 + portfolio_return)
                returns_series.append(portfolio_return)
            else:
                returns_series.append(0.0)

            portfolio_values.append(portfolio_value)
            portfolio_weights.append(current_weights.copy())

        # Create simple portfolio object (same as drift rebalancing)
        class SimplePortfolio:
            def __init__(self, values, returns, init_cash):
                self.values = pd.Series(values, index=backtest_prices.index)
                self.returns = pd.Series(returns, index=backtest_prices.index)
                self.init_cash = init_cash

            def value(self):
                return self.values

            def daily_returns(self):
                return self.returns

            def drawdown(self):
                cumulative = self.values / self.init_cash
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                return drawdown

        # Create portfolio object
        portfolio = SimplePortfolio(portfolio_values, returns_series, self.initial_cash)

        return {
            'portfolio': portfolio,
            'weights_history': time_varying_weights,
            'rebalance_dates': rebal_dates,
            'strategy': 'Monthly'
        }

    def _run_drift_rebalancing(self,
                               all_prices: pd.DataFrame,
                               backtest_prices: pd.DataFrame,
                               lookback_days: int,
                               drift_threshold: float = 0.05) -> Dict:
        """Run drift-based rebalancing strategy with manual calculation"""

        # Initial optimization
        initial_returns = all_prices.iloc[:lookback_days].pct_change().dropna()
        current_weights = self.calculate_mhrp_weights(initial_returns)
        current_weights = current_weights.reindex(backtest_prices.columns).fillna(0.0)
        current_weights = current_weights / current_weights.sum()

        # Track rebalancing dates
        rebalance_dates = [backtest_prices.index[0]]
        time_varying_weights = {backtest_prices.index[0]: current_weights}

        # Manual portfolio simulation
        portfolio_values = []
        returns_series = []
        portfolio_value = self.initial_cash

        for i, date in enumerate(backtest_prices.index):
            current_prices = backtest_prices.loc[date]

            # Calculate daily returns
            if i > 0:
                prev_prices = backtest_prices.iloc[i - 1]
                asset_returns = (current_prices / prev_prices - 1).fillna(0)
                portfolio_return = (current_weights * asset_returns).sum()

                # Check for drift
                # Simulate current actual weights (simplified)
                portfolio_value *= (1 + portfolio_return)

                # Check if drift threshold is exceeded (simplified check)
                if i % 21 == 0 and i > 21:  # Check every ~month
                    # Rebalance: recalculate optimal weights
                    end_idx = all_prices.index.get_loc(date)
                    start_idx = max(0, end_idx - lookback_days)
                    lookback_data = all_prices.iloc[start_idx:end_idx]

                    if len(lookback_data) > lookback_days // 2:
                        returns = lookback_data.pct_change().dropna()
                        new_weights = self.calculate_mhrp_weights(returns)
                        new_weights = new_weights.reindex(backtest_prices.columns).fillna(0.0)
                        new_weights = new_weights / new_weights.sum()

                        # Check actual drift
                        weight_diff = np.abs(new_weights - current_weights).max()

                        if weight_diff > drift_threshold:
                            current_weights = new_weights
                            portfolio_return -= self.fees  # Apply rebalancing cost
                            rebalance_dates.append(date)
                            time_varying_weights[date] = current_weights
                            portfolio_value *= (1 + portfolio_return)  # Recalculate with fees
                        else:
                            portfolio_value *= (1 + portfolio_return)
                    else:
                        portfolio_value *= (1 + portfolio_return)
                else:
                    portfolio_value *= (1 + portfolio_return)

                returns_series.append(portfolio_return)
            else:
                returns_series.append(0.0)
                portfolio_value = self.initial_cash

            portfolio_values.append(portfolio_value)

        # Create simple portfolio object
        class SimplePortfolio:
            def __init__(self, values, returns, init_cash):
                self.values = pd.Series(values, index=backtest_prices.index)
                self.returns = pd.Series(returns, index=backtest_prices.index)
                self.init_cash = init_cash

            def value(self):
                return self.values

            def daily_returns(self):
                return self.returns

            def drawdown(self):
                cumulative = self.values / self.init_cash
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                return drawdown

        portfolio = SimplePortfolio(portfolio_values, returns_series, self.initial_cash)

        return {
            'portfolio': portfolio,
            'weights_history': time_varying_weights,
            'rebalance_dates': rebalance_dates,
            'strategy': 'Drift',
            'drift_threshold': drift_threshold
        }

    def _run_volatility_rebalancing(self,
                                    all_prices: pd.DataFrame,
                                    backtest_prices: pd.DataFrame,
                                    lookback_days: int,
                                    vol_window: int = 21,
                                    vol_threshold: float = 0.2) -> Dict:
        """Run volatility-adjusted rebalancing strategy with manual calculation"""

        # Calculate rolling volatility
        returns = backtest_prices.pct_change().dropna()
        portfolio_returns = returns.mean(axis=1)  # Equal weight proxy for portfolio returns
        rolling_vol = portfolio_returns.rolling(vol_window).std() * np.sqrt(252)

        # Track rebalancing dates
        rebalance_dates = [backtest_prices.index[0]]
        time_varying_weights = {}

        # Initial optimization
        initial_returns = all_prices.iloc[:lookback_days].pct_change().dropna()
        current_weights = self.calculate_mhrp_weights(initial_returns)
        current_weights = current_weights.reindex(backtest_prices.columns).fillna(0.0)
        current_weights = current_weights / current_weights.sum()
        time_varying_weights[backtest_prices.index[0]] = current_weights

        # Manual portfolio simulation
        portfolio_values = []
        returns_series = []
        portfolio_value = self.initial_cash

        # Track volatility regime - initialize after vol_window
        last_vol = None

        for i, date in enumerate(backtest_prices.index):
            current_prices = backtest_prices.loc[date]

            # Calculate daily returns
            if i > 0:
                prev_prices = backtest_prices.iloc[i - 1]
                asset_returns = (current_prices / prev_prices - 1).fillna(0)
                portfolio_return = (current_weights * asset_returns).sum()
                portfolio_value *= (1 + portfolio_return)

                # Check for significant volatility change (after vol_window)
                # Align indices properly - rolling_vol starts after vol_window
                vol_index = i - 1  # Adjust for pct_change dropping first row

                if vol_index >= vol_window and vol_index < len(rolling_vol):
                    current_vol = rolling_vol.iloc[vol_index]

                    # Initialize last_vol if it's the first time in this block
                    if last_vol is None and vol_index == vol_window:
                        last_vol = current_vol

                    if last_vol is not None and pd.notna(current_vol) and pd.notna(last_vol):
                        vol_change = abs(current_vol - last_vol) / last_vol

                        if vol_change > vol_threshold:
                            # Rebalance due to volatility regime change
                            end_idx = all_prices.index.get_loc(date)
                            start_idx = max(0, end_idx - lookback_days)
                            lookback_data = all_prices.iloc[start_idx:end_idx]

                            if len(lookback_data) > lookback_days // 2:
                                returns_data = lookback_data.pct_change().dropna()
                                new_weights = self.calculate_mhrp_weights(returns_data)
                                new_weights = new_weights.reindex(backtest_prices.columns).fillna(0.0)
                                new_weights = new_weights / new_weights.sum()

                                current_weights = new_weights
                                portfolio_return -= self.fees  # Apply rebalancing cost
                                portfolio_value *= (1 + portfolio_return)  # Recalculate with fees

                                rebalance_dates.append(date)
                                time_varying_weights[date] = current_weights
                                last_vol = current_vol
                            else:
                                portfolio_value *= (1 + portfolio_return)
                        else:
                            # Update last_vol even if no rebalancing
                            last_vol = current_vol
                            portfolio_value *= (1 + portfolio_return)
                    else:
                        portfolio_value *= (1 + portfolio_return)
                        # Set last_vol for next iteration if this is the first valid volatility
                        if last_vol is None and pd.notna(current_vol):
                            last_vol = current_vol
                else:
                    portfolio_value *= (1 + portfolio_return)

                returns_series.append(portfolio_return)
            else:
                returns_series.append(0.0)
                portfolio_value = self.initial_cash

            portfolio_values.append(portfolio_value)

        # Create simple portfolio object
        class SimplePortfolio:
            def __init__(self, values, returns, init_cash):
                self.values = pd.Series(values, index=backtest_prices.index)
                self.returns = pd.Series(returns, index=backtest_prices.index)
                self.init_cash = init_cash

            def value(self):
                return self.values

            def daily_returns(self):
                return self.returns

            def drawdown(self):
                cumulative = self.values / self.init_cash
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                return drawdown

        portfolio = SimplePortfolio(portfolio_values, returns_series, self.initial_cash)

        return {
            'portfolio': portfolio,
            'weights_history': time_varying_weights,
            'rebalance_dates': rebalance_dates,
            'strategy': 'Volatility',
            'vol_threshold': vol_threshold,
            'rolling_volatility': rolling_vol
        }

    def walk_forward_analysis(self,
                             tickers: List[str],
                             start_date: str = "2020-01-01",
                             end_date: Optional[str] = None,
                             walk_forward_months: int = 3,
                             lookback_days: int = 252) -> Dict:
        """
        Perform walk-forward analysis to validate MHRP performance

        Args:
            tickers: List of stock symbols
            start_date: Start date for analysis
            end_date: End date for analysis
            walk_forward_months: Months between reoptimizations
            lookback_days: Days of historical data for each optimization

        Returns:
            Dictionary with walk-forward analysis results
        """

        print("🔄 Performing Walk-Forward Analysis...")

        # Download all data
        extended_start = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=lookback_days * 2)
        all_prices, failed = self.fetch_prices(
            tickers,
            lookback_days=int((datetime.today() - extended_start).days)
        )

        if failed:
            print(f"⚠️  Failed to download data for: {failed}")
            tickers = [t for t in tickers if t not in failed]

        # Create walk-forward dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.today()

        walk_dates = []
        current_date = start_dt
        while current_date <= end_dt:
            if current_date in all_prices.index:
                walk_dates.append(current_date)
            current_date += timedelta(days=walk_forward_months * 30)

        # Perform walk-forward optimization
        in_sample_results = []
        out_sample_results = []

        for i, opt_date in enumerate(walk_dates[:-1]):
            print(f"Walk-forward step {i+1}/{len(walk_dates)-1}: {opt_date.date()}")

            # In-sample period (optimization)
            opt_start_idx = max(0, all_prices.index.get_loc(opt_date) - lookback_days)
            opt_end_idx = all_prices.index.get_loc(opt_date)
            in_sample_prices = all_prices.iloc[opt_start_idx:opt_end_idx]
            in_sample_returns = in_sample_prices.pct_change().dropna()

            # Calculate MHRP weights
            weights = self.calculate_mhrp_weights(in_sample_returns)

            # Out-of-sample period (testing)
            next_date = walk_dates[i + 1]
            oos_start_idx = all_prices.index.get_loc(opt_date)
            oos_end_idx = all_prices.index.get_loc(next_date)
            out_sample_prices = all_prices.iloc[oos_start_idx:oos_end_idx]
            out_sample_returns = out_sample_prices.pct_change().dropna()

            # Calculate portfolio returns
            aligned_weights = weights.reindex(out_sample_returns.columns).fillna(0.0)
            portfolio_returns = (out_sample_returns * aligned_weights).sum(axis=1)

            # Store results
            in_sample_results.append({
                'date': opt_date,
                'weights': weights,
                'returns': in_sample_returns,
                'portfolio_returns': (in_sample_returns * aligned_weights).sum(axis=1)
            })

            out_sample_results.append({
                'date': opt_date,
                'weights': weights,
                'returns': out_sample_returns,
                'portfolio_returns': portfolio_returns
            })

        # Aggregate results
        all_in_sample_returns = pd.concat([r['portfolio_returns'] for r in in_sample_results])
        all_out_sample_returns = pd.concat([r['portfolio_returns'] for r in out_sample_results])

        # Calculate metrics
        walk_forward_metrics = {
            'in_sample_sharpe': qs.stats.sharpe(all_in_sample_returns),
            'out_sample_sharpe': qs.stats.sharpe(all_out_sample_returns),
            'in_sample_return': qs.stats.cagr(all_in_sample_returns),
            'out_sample_return': qs.stats.cagr(all_out_sample_returns),
            'in_sample_volatility': qs.stats.volatility(all_in_sample_returns),
            'out_sample_volatility': qs.stats.volatility(all_out_sample_returns),
            'in_sample_max_dd': qs.stats.max_drawdown(all_in_sample_returns),
            'out_sample_max_dd': qs.stats.max_drawdown(all_out_sample_returns),
            'sharpe_degradation': qs.stats.sharpe(all_in_sample_returns) - qs.stats.sharpe(all_out_sample_returns)
        }

        return {
            'metrics': walk_forward_metrics,
            'in_sample_results': in_sample_results,
            'out_sample_results': out_sample_results,
            'all_in_sample_returns': all_in_sample_returns,
            'all_out_sample_returns': all_out_sample_returns
        }

    def calculate_factor_attribution(self,
                                   portfolio_returns: pd.Series,
                                   factor_returns: Optional[Dict[str, pd.Series]] = None) -> Dict:
        """
        Perform factor attribution analysis

        Args:
            portfolio_returns: Portfolio returns series
            factor_returns: Dictionary of factor returns (if None, downloads common factors)

        Returns:
            Dictionary with factor attribution results
        """

        print("📊 Calculating Factor Attribution...")

        if factor_returns is None:
            # Download common factor ETFs
            factor_symbols = {
                'Market': 'SPY',
                'Small Cap': 'IWM',
                'International': 'EFA',
                'Emerging Markets': 'EEM',
                'Growth': 'IWF',
                'Value': 'IWD',
                'Real Estate': 'VNQ',
                'Commodities': 'DJP',
                'Bonds': 'AGG',
                'Treasury': 'TLT'
            }

            start = (portfolio_returns.index[0] - timedelta(days=30)).strftime('%Y-%m-%d')
            end = (portfolio_returns.index[-1] + timedelta(days=1)).strftime('%Y-%m-%d')

            # Use Marketstack if available
            if hasattr(self, 'data_fetcher'):
                factor_data, _ = self.data_fetcher.fetch_prices(
                    list(factor_symbols.values()),
                    start,
                    end,
                    verbose=False
                )
            else:
                # Original yfinance code
                factor_data = yf.download(
                    list(factor_symbols.values()),
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False
                )['Close']

            # Get factor data
            # factor_data = yf.download(
            #     list(factor_symbols.values()),
            #     start=portfolio_returns.index[0] - timedelta(days=30),
            #     end=portfolio_returns.index[-1] + timedelta(days=1),
            #     auto_adjust=True, progress=False
            # )['Close']

            # Calculate factor returns
            factor_returns = {}
            for name, symbol in factor_symbols.items():
                if symbol in factor_data.columns:
                    factor_returns[name] = factor_data[symbol].pct_change().dropna()

        # Align dates
        start_date = max(portfolio_returns.index[0],
                        min([fr.index[0] for fr in factor_returns.values()]))
        end_date = min(portfolio_returns.index[-1],
                      max([fr.index[-1] for fr in factor_returns.values()]))

        aligned_portfolio = portfolio_returns.loc[start_date:end_date]
        aligned_factors = pd.DataFrame({
            name: returns.loc[start_date:end_date]
            for name, returns in factor_returns.items()
        }).dropna()

        # Align portfolio returns with factors
        aligned_portfolio = aligned_portfolio.reindex(aligned_factors.index).dropna()

        # Multi-factor regression
        X = aligned_factors.values
        y = aligned_portfolio.values

        model = LinearRegression()
        model.fit(X, y)

        # Calculate factor exposures (betas)
        factor_exposures = pd.Series(model.coef_, index=aligned_factors.columns)
        alpha = model.intercept_
        r_squared = model.score(X, y)

        # Calculate factor contributions
        factor_contributions = {}
        total_return = aligned_portfolio.mean() * 252  # Annualized

        for i, factor_name in enumerate(aligned_factors.columns):
            factor_return = aligned_factors[factor_name].mean() * 252  # Annualized
            contribution = factor_exposures[factor_name] * factor_return
            factor_contributions[factor_name] = contribution

        # Residual return (alpha)
        explained_return = sum(factor_contributions.values())
        residual_return = total_return - explained_return

        return {
            'factor_exposures': factor_exposures,
            'factor_contributions': pd.Series(factor_contributions),
            'alpha': alpha * 252,  # Annualized alpha
            'r_squared': r_squared,
            'total_return': total_return,
            'explained_return': explained_return,
            'residual_return': residual_return,
            'aligned_data': {
                'portfolio': aligned_portfolio,
                'factors': aligned_factors
            }
        }

    def analyze_correlation_over_time(self,
                                    prices: pd.DataFrame,
                                    window: int = 60) -> Dict:
        """
        Analyze correlation structure over time

        Args:
            prices: Price data DataFrame
            window: Rolling window for correlation calculation

        Returns:
            Dictionary with correlation analysis results
        """

        print("🔍 Analyzing Correlation Over Time...")

        returns = prices.pct_change().dropna()

        # Calculate rolling correlations
        rolling_corr = returns.rolling(window).corr()

        # Calculate average correlation over time
        avg_correlations = {}
        correlation_stability = {}

        for date in rolling_corr.index.get_level_values(0).unique()[window:]:
            corr_matrix = rolling_corr.loc[date]

            # Average correlation (excluding diagonal)
            mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
            avg_corr = corr_matrix.values[mask].mean()
            avg_correlations[date] = avg_corr

        avg_corr_series = pd.Series(avg_correlations)

        # Calculate correlation stability (rolling std of correlations)
        corr_stability = avg_corr_series.rolling(window=window//2).std()

        # Eigenvalue decomposition for principal components
        latest_corr = returns.tail(window).corr()
        eigenvalues, eigenvectors = np.linalg.eigh(latest_corr)
        eigenvalues = eigenvalues[::-1]  # Sort descending

        # Calculate explained variance by components
        total_variance = eigenvalues.sum()
        explained_variance = eigenvalues / total_variance

        return {
            'rolling_correlations': rolling_corr,
            'average_correlations': avg_corr_series,
            'correlation_stability': corr_stability,
            'latest_correlation_matrix': latest_corr,
            'eigenvalues': eigenvalues,
            'explained_variance': explained_variance,
            'pca_components': eigenvectors[:, ::-1],  # Sort to match eigenvalues
            'window': window
        }

    def calculate_tail_risk_metrics(self,
                                  portfolio_returns: pd.Series,
                                  confidence_levels: List[float] = [0.95, 0.99]) -> Dict:
        """
        Calculate comprehensive tail risk metrics

        Args:
            portfolio_returns: Portfolio returns series
            confidence_levels: List of confidence levels for VaR/CVaR calculation

        Returns:
            Dictionary with tail risk metrics
        """

        print("⚠️  Calculating Tail Risk Metrics...")

        # Basic return statistics
        returns_stats = {
            'mean': portfolio_returns.mean(),
            'std': portfolio_returns.std(),
            'skewness': stats.skew(portfolio_returns),
            'kurtosis': stats.kurtosis(portfolio_returns),
            'jarque_bera_stat': stats.jarque_bera(portfolio_returns)[0],
            'jarque_bera_pvalue': stats.jarque_bera(portfolio_returns)[1]
        }

        # Value at Risk (VaR) and Conditional VaR (CVaR)
        var_cvar_metrics = {}

        for conf_level in confidence_levels:
            alpha = 1 - conf_level

            # Historical VaR
            var_historical = np.percentile(portfolio_returns, alpha * 100)

            # Conditional VaR (Expected Shortfall)
            cvar_historical = portfolio_returns[portfolio_returns <= var_historical].mean()

            # Parametric VaR (normal distribution)
            var_parametric = portfolio_returns.mean() + portfolio_returns.std() * stats.norm.ppf(alpha)

            # Modified VaR (Cornish-Fisher expansion)
            z_score = stats.norm.ppf(alpha)
            skew = stats.skew(portfolio_returns)
            kurt = stats.kurtosis(portfolio_returns)

            modified_z = (z_score +
                         (z_score**2 - 1) * skew / 6 +
                         (z_score**3 - 3*z_score) * kurt / 24 -
                         (2*z_score**3 - 5*z_score) * skew**2 / 36)

            var_modified = portfolio_returns.mean() + portfolio_returns.std() * modified_z

            var_cvar_metrics[f'VaR_{int(conf_level*100)}%'] = {
                'historical': var_historical,
                'parametric': var_parametric,
                'modified': var_modified
            }

            var_cvar_metrics[f'CVaR_{int(conf_level*100)}%'] = {
                'historical': cvar_historical
            }

        # Conditional Drawdown at Risk (CDaR)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max

        cdar_metrics = {}
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            cdar = np.percentile(drawdowns, alpha * 100)
            cdar_metrics[f'CDaR_{int(conf_level*100)}%'] = cdar

        # Maximum Drawdown and duration
        max_drawdown = drawdowns.min()
        max_dd_start = drawdowns.idxmin()

        # Find drawdown duration
        if max_dd_start in rolling_max.index:
            # Find when the peak was reached
            peak_idx = rolling_max.loc[:max_dd_start].idxmax()
            # Find when it recovered (if it did)
            recovery_series = cumulative_returns.loc[max_dd_start:]
            peak_value = cumulative_returns.loc[peak_idx]
            recovery_idx = recovery_series[recovery_series >= peak_value].index

            if len(recovery_idx) > 0:
                max_dd_duration = (recovery_idx[0] - peak_idx).days
            else:
                max_dd_duration = (portfolio_returns.index[-1] - peak_idx).days
        else:
            max_dd_duration = np.nan

        # Tail Ratio
        positive_returns = portfolio_returns[portfolio_returns > 0]
        negative_returns = portfolio_returns[portfolio_returns < 0]

        if len(positive_returns) > 0 and len(negative_returns) > 0:
            tail_ratio = np.percentile(positive_returns, 95) / abs(np.percentile(negative_returns, 5))
        else:
            tail_ratio = np.nan

        # Capture ratios (vs market proxy)
        # Note: This would require a market return series for comparison
        # For now, we'll skip this or use a simple implementation

        return {
            'return_statistics': returns_stats,
            'var_cvar_metrics': var_cvar_metrics,
            'cdar_metrics': cdar_metrics,
            'max_drawdown': max_drawdown,
            'max_drawdown_duration_days': max_dd_duration,
            'tail_ratio': tail_ratio,
            'drawdown_series': drawdowns
        }

    def generate_comprehensive_report(self,
                                    results: Dict,
                                    benchmark: str = 'SPY') -> None:
        """
        Generate comprehensive report with all analytics

        Args:
            results: Dictionary with backtest results
            benchmark: Benchmark symbol for comparison
        """

        print("\n" + "="*60)
        print("📊 COMPREHENSIVE MHRP PORTFOLIO ANALYSIS REPORT")
        print("="*60)

        # Strategy comparison
        print("\n🏆 STRATEGY PERFORMANCE COMPARISON")
        print("-" * 40)

        comparison_data = []
        for strategy_name, result in results.items():
            portfolio = result['portfolio']
            returns = portfolio.daily_returns()
            if isinstance(returns, pd.DataFrame):
                returns = returns.iloc[:, 0]

            metrics = {
                'Strategy': strategy_name,
                'Total Return': f"{((1 + returns).prod() - 1) * 100:.2f}%",
                'Annual Return': f"{qs.stats.cagr(returns) * 100:.2f}%",
                'Volatility': f"{qs.stats.volatility(returns) * 100:.2f}%",
                'Sharpe Ratio': f"{qs.stats.sharpe(returns):.3f}",
                'Max Drawdown': f"{qs.stats.max_drawdown(returns) * 100:.2f}%",
                'Calmar Ratio': f"{qs.stats.calmar(returns):.3f}",
                'Rebalance Count': len(result.get('rebalance_dates', []))
            }
            comparison_data.append(metrics)

        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))

        # Best performing strategy detailed analysis
        best_strategy = max(results.keys(),
                          key=lambda x: qs.stats.sharpe(
                              results[x]['portfolio'].daily_returns().iloc[:, 0]
                              if isinstance(results[x]['portfolio'].daily_returns(), pd.DataFrame)
                              else results[x]['portfolio'].daily_returns()
                          ))

        best_result = results[best_strategy]
        best_portfolio = best_result['portfolio']
        best_returns = best_portfolio.daily_returns()
        if isinstance(best_returns, pd.DataFrame):
            best_returns = best_returns.iloc[:, 0]

        print(f"\n🥇 DETAILED ANALYSIS - BEST STRATEGY: {best_strategy}")
        print("-" * 50)

        # Factor attribution for best strategy
        factor_analysis = self.calculate_factor_attribution(best_returns)
        print("\n📈 FACTOR ATTRIBUTION")
        print(f"Alpha (Annual): {factor_analysis['alpha']*100:.2f}%")
        print(f"R-squared: {factor_analysis['r_squared']:.3f}")
        print("\nFactor Exposures:")
        for factor, exposure in factor_analysis['factor_exposures'].items():
            print(f"  {factor}: {exposure:.3f}")

        print("\nFactor Contributions (Annual):")
        for factor, contribution in factor_analysis['factor_contributions'].items():
            print(f"  {factor}: {contribution*100:.2f}%")

        # Tail risk analysis
        tail_risk = self.calculate_tail_risk_metrics(best_returns)
        print("\n⚠️  TAIL RISK METRICS")
        print(f"Skewness: {tail_risk['return_statistics']['skewness']:.3f}")
        print(f"Kurtosis: {tail_risk['return_statistics']['kurtosis']:.3f}")
        print(f"Max Drawdown: {tail_risk['max_drawdown'] * 100:.2f}%")
        print(f"Max DD Duration: {tail_risk['max_drawdown_duration_days']} days")
        print(f"Tail Ratio: {tail_risk['tail_ratio']:.3f}")

        print("\nValue at Risk (95%):")
        var_95_key = 'VaR_95%'
        if var_95_key in tail_risk['var_cvar_metrics']:
            var_95 = tail_risk['var_cvar_metrics'][var_95_key]
            print(f"  Historical: {var_95['historical'] * 100:.2f}%")
            print(f"  Parametric: {var_95['parametric'] * 100:.2f}%")
            print(f"  Modified: {var_95['modified'] * 100:.2f}%")

        print("\nConditional VaR (95%):")
        cvar_95_key = 'CVaR_95%'
        if cvar_95_key in tail_risk['var_cvar_metrics']:
            cvar_95 = tail_risk['var_cvar_metrics'][cvar_95_key]['historical']
            print(f"  Historical: {cvar_95 * 100:.2f}%")

        print("\nConditional Drawdown at Risk:")
        for metric, value in tail_risk['cdar_metrics'].items():
            print(f"  {metric}: {value * 100:.2f}%")

        print("\n" + "="*60)
        print("✅ REPORT COMPLETE")
        print("="*60)

    def plot_analysis_charts(self,
                           results: Dict,
                           prices: pd.DataFrame,
                           save_plots: bool = False) -> None:
        """
        Create comprehensive analysis charts

        Args:
            results: Dictionary with backtest results
            prices: Price data used for correlation analysis
            save_plots: Whether to save plots to files
        """

        # Set up the plotting style
        plt.style.use('seaborn-v0_8')

        # 1. Portfolio Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Performance comparison
        ax1 = axes[0, 0]
        for strategy_name, result in results.items():
            portfolio = result['portfolio']
            cum_returns = portfolio.value()
            ax1.plot(cum_returns.index, cum_returns.values / cum_returns.iloc[0],
                    label=strategy_name, linewidth=2)

        ax1.set_title('Portfolio Performance Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Drawdown comparison
        ax2 = axes[0, 1]
        for strategy_name, result in results.items():
            portfolio = result['portfolio']
            drawdown = portfolio.drawdown().iloc[:, 0] if isinstance(portfolio.drawdown(), pd.DataFrame) else portfolio.drawdown()
            ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.7, label=strategy_name)

        ax2.set_title('Drawdown Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Rolling Sharpe Ratio
        ax3 = axes[1, 0]
        for strategy_name, result in results.items():
            portfolio = result['portfolio']
            returns = portfolio.daily_returns()
            if isinstance(returns, pd.DataFrame):
                returns = returns.iloc[:, 0]
            rolling_sharpe = returns.rolling(252).mean() / returns.rolling(252).std() * np.sqrt(252)
            ax3.plot(rolling_sharpe.index, rolling_sharpe.values, label=strategy_name, linewidth=2)

        ax3.set_title('Rolling Sharpe Ratio (1-Year)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Correlation heatmap for latest period
        ax4 = axes[1, 1]
        recent_returns = prices.pct_change().tail(252)
        corr_matrix = recent_returns.corr()
        im = ax4.imshow(corr_matrix.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax4.set_xticks(range(len(corr_matrix.columns)))
        ax4.set_yticks(range(len(corr_matrix.columns)))
        ax4.set_xticklabels(corr_matrix.columns, rotation=45)
        ax4.set_yticklabels(corr_matrix.columns)
        ax4.set_title('Asset Correlation Matrix (Latest Year)', fontsize=14, fontweight='bold')

        # Add colorbar
        plt.colorbar(im, ax=ax4)

        plt.tight_layout()
        if save_plots:
            plt.savefig('portfolio_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 2. Best Strategy Detailed Analysis
        best_strategy = max(results.keys(),
                          key=lambda x: qs.stats.sharpe(
                              results[x]['portfolio'].daily_returns().iloc[:, 0]
                              if isinstance(results[x]['portfolio'].daily_returns(), pd.DataFrame)
                              else results[x]['portfolio'].daily_returns()
                          ))

        best_result = results[best_strategy]
        best_portfolio = best_result['portfolio']
        best_returns = best_portfolio.daily_returns()
        if isinstance(best_returns, pd.DataFrame):
            best_returns = best_returns.iloc[:, 0]

        # Detailed analysis for best strategy
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Return distribution
        axes[0, 0].hist(best_returns * 100, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title(f'{best_strategy} - Return Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Daily Return (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)

        # Q-Q plot for normality check
        stats.probplot(best_returns, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title(f'{best_strategy} - Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        # Rolling correlation
        correlation_analysis = self.analyze_correlation_over_time(prices)
        avg_corr = correlation_analysis['average_correlations']
        axes[1, 0].plot(avg_corr.index, avg_corr.values, linewidth=2, color='purple')
        axes[1, 0].set_title('Average Asset Correlation Over Time', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Average Correlation')
        axes[1, 0].grid(True, alpha=0.3)

        # Factor exposures
        factor_analysis = self.calculate_factor_attribution(best_returns)
        factor_exp = factor_analysis['factor_exposures']
        axes[1, 1].bar(range(len(factor_exp)), factor_exp.values, alpha=0.7)
        axes[1, 1].set_title(f'{best_strategy} - Factor Exposures', fontsize=14, fontweight='bold')
        axes[1, 1].set_xticks(range(len(factor_exp)))
        axes[1, 1].set_xticklabels(factor_exp.index, rotation=45)
        axes[1, 1].set_ylabel('Exposure (Beta)')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_plots:
            plt.savefig(f'{best_strategy}_detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def export_results(self,
                      results: Dict,
                      walk_forward_results: Optional[Dict] = None,
                      filename: str = 'mhrp_portfolio_results.json') -> None:
        """
        Export results to JSON file for later analysis

        Args:
            results: Backtest results dictionary
            walk_forward_results: Walk-forward analysis results
            filename: Output filename
        """

        export_data = {
            'timestamp': datetime.now().isoformat(),
            'strategies': {}
        }

        # Export strategy results
        for strategy_name, result in results.items():
            portfolio = result['portfolio']
            returns = portfolio.daily_returns()
            if isinstance(returns, pd.DataFrame):
                returns = returns.iloc[:, 0]

            strategy_data = {
                'total_return': float((1 + returns).prod() - 1),
                'annual_return': float(qs.stats.cagr(returns)),
                'volatility': float(qs.stats.volatility(returns)),
                'sharpe_ratio': float(qs.stats.sharpe(returns)),
                'max_drawdown': float(qs.stats.max_drawdown(returns)),
                'calmar_ratio': float(qs.stats.calmar(returns)),
                'sortino_ratio': float(qs.stats.sortino(returns)),
                'rebalance_count': len(result.get('rebalance_dates', [])),
                'rebalance_dates': [d.isoformat() for d in result.get('rebalance_dates', [])],
                'final_weights': {},
                'weights_history': {}
            }

            # Export final weights
            if 'weights_history' in result and result['weights_history']:
                last_date = max(result['weights_history'].keys())
                final_weights = result['weights_history'][last_date]
                strategy_data['final_weights'] = {str(k): float(v) for k, v in final_weights.items()}

                # Export weights history
                weights_history = {}
                for date, weights in result['weights_history'].items():
                    weights_history[date.isoformat()] = {str(k): float(v) for k, v in weights.items()}
                strategy_data['weights_history'] = weights_history

            export_data['strategies'][strategy_name] = strategy_data

        # Export walk-forward results if available
        if walk_forward_results:
            wf_metrics = walk_forward_results['metrics']
            export_data['walk_forward_analysis'] = {
                'in_sample_sharpe': float(wf_metrics['in_sample_sharpe']),
                'out_sample_sharpe': float(wf_metrics['out_sample_sharpe']),
                'sharpe_degradation': float(wf_metrics['sharpe_degradation']),
                'in_sample_return': float(wf_metrics['in_sample_return']),
                'out_sample_return': float(wf_metrics['out_sample_return']),
                'in_sample_volatility': float(wf_metrics['in_sample_volatility']),
                'out_sample_volatility': float(wf_metrics['out_sample_volatility'])
            }

        # Save to file
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        print(f"✅ Results exported to {filename}")


# Example usage
if __name__ == "__main__":
    # Your Marketstack API key
    MARKETSTACK_API_KEY = "476419cceb4330259e5a126753335b72"  # Use your actual key

    # Initialize the portfolio system
    portfolio_system = MHRPPortfolioSystem(
        initial_cash=100_000,
        fees=0.001,
        slippage=0.0005,
        marketstack_api_key=MARKETSTACK_API_KEY
    )

    # Define tickers to analyze
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM", "JNJ", "V", "PG", "NVDA"]

    # Run backtests with all three rebalancing strategies
    print("🚀 Starting Comprehensive MHRP Portfolio Analysis...")

    results = portfolio_system.run_backtest_strategies(
        tickers=tickers,
        start_date="2020-01-01",
        end_date=None,  # Use all available data
        lookback_days=252,
        rebalance_strategies=['monthly', 'drift', 'volatility']
    )

    # Perform walk-forward analysis
    walk_forward_results = portfolio_system.walk_forward_analysis(
        tickers=tickers,
        start_date="2020-01-01",
        end_date=None,
        walk_forward_months=3,
        lookback_days=252
    )

    # Generate comprehensive report
    portfolio_system.generate_comprehensive_report(results)

    # Print walk-forward analysis summary
    print(f"\n🔄 WALK-FORWARD ANALYSIS SUMMARY")
    print("-" * 40)
    wf_metrics = walk_forward_results['metrics']
    print(f"In-Sample Sharpe: {wf_metrics['in_sample_sharpe']:.3f}")
    print(f"Out-of-Sample Sharpe: {wf_metrics['out_sample_sharpe']:.3f}")
    print(f"Sharpe Degradation: {wf_metrics['sharpe_degradation']:.3f}")
    print(f"In-Sample Annual Return: {wf_metrics['in_sample_return']*100:.2f}%")
    print(f"Out-of-Sample Annual Return: {wf_metrics['out_sample_return']*100:.2f}%")

    # Create analysis charts
    prices, _ = portfolio_system.fetch_prices(tickers, lookback_days=500)
    portfolio_system.plot_analysis_charts(results, prices, save_plots=True)

    # Export results for further analysis
    portfolio_system.export_results(
        results=results,
        walk_forward_results=walk_forward_results,
        filename='mhrp_comprehensive_results.json'
    )

    print("\n✅ Analysis Complete! Check the generated charts and exported results.")