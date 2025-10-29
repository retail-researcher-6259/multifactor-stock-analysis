"""
Dynamic Portfolio Selection System with YFinance Support
This version is optimized for yfinance data source to reduce API calls
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import itertools
import warnings

warnings.filterwarnings('ignore')

class DynamicPortfolioSelectorYFinance:
    """
    Dynamic Portfolio Selection System optimized for yfinance
    Designed to work with shorter lookback periods and cached data
    """

    def __init__(self, stocks_file: str, exclude_real_estate: bool = True):
        """
        Initialize with stocks data

        Args:
            stocks_file: Path to CSV file with stocks
            exclude_real_estate: Whether to exclude real estate stocks
        """
        self.stocks_df = pd.read_csv(stocks_file)
        self.exclude_real_estate = exclude_real_estate
        self.portfolios = {}
        self.backtest_results = {}
        self.cached_data = {}  # Cache for yfinance data

        # Detect file type
        self.file_type = self._detect_file_type()
        print(f" Detected file type: {self.file_type.upper()}")

        # Define real estate sectors/industries to exclude
        self.real_estate_sectors = [
            'Real Estate', 'REIT', 'Real Estate Investment Trust',
            'Real Estate - Development', 'Real Estate - Diversified',
            'Real Estate - General', 'Real Estate Services'
        ]

        self.real_estate_industries = [
            'Real Estate - Development', 'Real Estate - Diversified',
            'Real Estate - General', 'Real Estate Services',
            'REIT - Diversified', 'REIT - Healthcare Facilities',
            'REIT - Industrial', 'REIT - Office', 'REIT - Residential',
            'REIT - Retail', 'REIT - Hotel & Motel', 'REIT - Mortgage',
            'Housing', 'Home Construction', 'Home Builders',
            'Building Materials', 'Lumber & Wood Production'
        ]

        self._prepare_data()

    def _detect_file_type(self):
        """Detect whether file is stability analysis or other format"""
        columns = self.stocks_df.columns.tolist()

        # Check for stability analysis specific columns
        stability_columns = ['stability_adjusted_score', 'linear_r2', 'linear_slope',
                             'Stability_Adjusted_Score', 'Linear_R2', 'Linear_Slope']

        if any(col in columns for col in stability_columns):
            return 'stability'

        # Check for ranked stocks columns
        elif 'Score' in columns and 'Value' in columns and 'Quality' in columns:
            return 'ranked'

        # Default to stability if has score-like columns
        elif any('score' in col.lower() for col in columns):
            return 'stability'

        else:
            # Default to stability type for unknown
            print(" File type unclear, assuming stability analysis format")
            return 'stability'

    def _prepare_data(self):
        """Prepare and clean the data"""

        # Standardize column names for common operations
        if 'ticker' in self.stocks_df.columns:
            self.stocks_df = self.stocks_df.rename(columns={'ticker': 'Ticker'})

        # Standardize other common column names from stability analysis
        column_mappings = {
            'linear_r2': 'Linear_R2',
            'linear_slope': 'Linear_Slope',
            'stability_adjusted_score': 'Stability_Adjusted_Score',
            'avg_score': 'Avg_Score',
            'score_cv': 'Score_CV',
            'combined_rank': 'Combined_Rank',
            'recommendation': 'Recommendation',
            'appearances': 'Appearances',
            'avg_rank': 'Avg_Rank',
            'stability_score': 'Stability_Score',
            'r2_adjusted_score': 'R2_Adjusted_Score',
            'slope_adjusted_score': 'Slope_Adjusted_Score',
            'trend_consistency': 'Trend_Consistency',
            'score_std': 'Score_Std'
        }

        # Apply column mappings
        for old_name, new_name in column_mappings.items():
            if old_name in self.stocks_df.columns:
                self.stocks_df = self.stocks_df.rename(columns={old_name: new_name})

        if self.exclude_real_estate:
            initial_count = len(self.stocks_df)

            # Filter by sector if exists
            if 'Sector' in self.stocks_df.columns:
                self.stocks_df = self.stocks_df[
                    ~self.stocks_df['Sector'].isin(self.real_estate_sectors)
                ]

            # Filter by industry if exists
            if 'Industry' in self.stocks_df.columns:
                self.stocks_df = self.stocks_df[
                    ~self.stocks_df['Industry'].isin(self.real_estate_industries)
                ]

            excluded_count = initial_count - len(self.stocks_df)
            if excluded_count > 0:
                print(f" Excluded {excluded_count} real estate related stocks")

        # Ensure required columns exist (add defaults if missing)
        if 'CompanyName' not in self.stocks_df.columns:
            self.stocks_df['CompanyName'] = 'Unknown'

        if 'Sector' not in self.stocks_df.columns:
            self.stocks_df['Sector'] = 'Unknown'

        if 'Industry' not in self.stocks_df.columns:
            self.stocks_df['Industry'] = 'Unknown'

        # Sort by appropriate score
        if self.file_type == 'stability':
            if 'Avg_Score' in self.stocks_df.columns:
                self.stocks_df = self.stocks_df.sort_values('Avg_Score', ascending=False)
        elif self.file_type == 'ranked':
            if 'Score' in self.stocks_df.columns:
                self.stocks_df = self.stocks_df.sort_values('Score', ascending=False)

        print(f" Total stocks available: {len(self.stocks_df)}")

    # def fetch_batch_data(self, tickers: List[str], lookback_days: int = 252) -> pd.DataFrame:
    #     """
    #     Fetch data for multiple tickers at once using yfinance
    #     Uses caching to reduce redundant API calls
    #     """
    #     cache_key = f"{','.join(sorted(tickers))}_{lookback_days}"
    #
    #     if cache_key in self.cached_data:
    #         print(f"   Using cached data for {len(tickers)} tickers")
    #         return self.cached_data[cache_key]
    #
    #     print(f"   Fetching data for {len(tickers)} tickers (lookback: {lookback_days} days)")
    #
    #     end_date = datetime.now()
    #     start_date = end_date - timedelta(days=int(lookback_days * 1.5))
    #
    #     try:
    #         # Download all tickers at once
    #         data = yf.download(
    #             tickers,
    #             start=start_date,
    #             end=end_date,
    #             progress=False,
    #             auto_adjust=True,
    #             threads=True
    #         )
    #
    #         if data.empty:
    #             return pd.DataFrame()
    #
    #         # Handle multi-level columns
    #         if isinstance(data.columns, pd.MultiIndex):
    #             prices = data['Close']
    #         else:
    #             prices = data[['Close']].rename(columns={'Close': tickers[0]})
    #
    #         # Clean data
    #         prices = prices.ffill().dropna(how='all')
    #
    #         # Cache the data
    #         self.cached_data[cache_key] = prices
    #
    #         return prices
    #
    #     except Exception as e:
    #         print(f"   Error fetching data: {e}")
    #         return pd.DataFrame()

    def fetch_batch_data(self, tickers: List[str], lookback_days: int = 252) -> pd.DataFrame:
        """
        Fetch data for multiple tickers with error handling for delisted stocks
        """
        cache_key = f"{','.join(sorted(tickers))}_{lookback_days}"

        if cache_key in self.cached_data:
            print(f"   Using cached data for {len(tickers)} tickers")
            return self.cached_data[cache_key]

        print(f"   Fetching data for {len(tickers)} tickers (lookback: {lookback_days} days)")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(lookback_days * 1.5))

        # Try to download with error handling
        valid_tickers = []
        invalid_tickers = []

        # First try batch download
        try:
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
                threads=True,
            )

            if not data.empty:
                # Handle multi-level columns
                if isinstance(data.columns, pd.MultiIndex):
                    prices = data['Close']
                    # Remove columns with all NaN (failed tickers)
                    prices = prices.dropna(axis=1, how='all')
                else:
                    prices = data[['Close']].rename(columns={'Close': tickers[0]})

                # Clean data
                prices = prices.ffill().dropna(how='all')

                # Cache the data
                self.cached_data[cache_key] = prices

                return prices

        except Exception as e:
            print(f"   Batch download failed: {e}")

        # If batch fails, return empty DataFrame
        return pd.DataFrame()

    def create_portfolio_strategies(self, top_n: int = 50) -> Dict[str, List[str]]:
        """
        Create different portfolio strategies optimized for yfinance
        """
        strategies = {}

        # Get top stocks
        top_stocks = self.stocks_df.head(top_n)

        if self.file_type == 'ranked':
            # Factor-based strategies
            if all(col in self.stocks_df.columns for col in ['Value', 'Quality', 'Momentum']):
                # Balanced strategy
                strategies['Balanced'] = top_stocks['Ticker'].tolist()

                # Value-focused
                value_stocks = top_stocks.nlargest(20, 'Value')
                strategies['Value'] = value_stocks['Ticker'].tolist()

                # Quality-focused
                quality_stocks = top_stocks.nlargest(20, 'Quality')
                strategies['Quality'] = quality_stocks['Ticker'].tolist()

                # Momentum-focused
                if 'Momentum' in top_stocks.columns:
                    momentum_stocks = top_stocks.nlargest(20, 'Momentum')
                    strategies['Momentum'] = momentum_stocks['Ticker'].tolist()

        elif self.file_type == 'stability':
            # Stability-based strategies
            strategies['TopScored'] = top_stocks['Ticker'].tolist()

            if 'Stability_Adjusted_Score' in top_stocks.columns:
                stable_stocks = top_stocks.nlargest(20, 'Stability_Adjusted_Score')
                strategies['MostStable'] = stable_stocks['Ticker'].tolist()

            if 'Linear_Slope' in top_stocks.columns:
                trending_stocks = top_stocks.nlargest(20, 'Linear_Slope')
                strategies['Trending'] = trending_stocks['Ticker'].tolist()

        # Diversification strategies (sector-based if available)
        if 'Sector' in self.stocks_df.columns:
            sectors = top_stocks['Sector'].value_counts().head(5).index
            diversified = []
            for sector in sectors:
                sector_stocks = top_stocks[top_stocks['Sector'] == sector].head(4)
                diversified.extend(sector_stocks['Ticker'].tolist())
            if len(diversified) >= 10:
                strategies['Diversified'] = diversified[:20]

        return strategies

    def create_multiple_portfolios(self,
                                   portfolio_sizes: List[int] = [10, 15, 20],
                                   top_pools: List[int] = [50, 75, 100],
                                   max_per_sector: int = 3) -> Dict[str, List[str]]:
        """
        Create multiple portfolio combinations matching the original script's approach
        """
        print("\n Creating portfolio combinations...")

        for pool_size in top_pools:
            for portfolio_size in portfolio_sizes:
                # Skip if pool is smaller than portfolio size
                if pool_size < portfolio_size:
                    continue

                print(f"\n Creating portfolios: Pool={pool_size}, Size={portfolio_size}")

                # Get top stocks for this pool
                top_stocks = self.stocks_df.head(pool_size)

                # 1. Top Rank portfolio (always created)
                name = f"TopRank_S{portfolio_size}_P{pool_size}"
                self.portfolios[name] = top_stocks.head(portfolio_size)['Ticker'].tolist()

                # 2. Sector-balanced portfolios (if Sector column exists)
                if 'Sector' in self.stocks_df.columns:
                    for max_sector in [2, 3]:
                        name = f"SectorBal{max_sector}_S{portfolio_size}_P{pool_size}"
                        portfolio = self._create_sector_balanced(top_stocks, portfolio_size, max_sector)
                        if portfolio:
                            self.portfolios[name] = portfolio

                # 3. Industry-balanced portfolio (if Industry column exists)
                if 'Industry' in self.stocks_df.columns:
                    name = f"IndustryBal_S{portfolio_size}_P{pool_size}"
                    portfolio = self._create_industry_balanced(top_stocks, portfolio_size)
                    if portfolio:
                        self.portfolios[name] = portfolio

                # 4. File-type specific strategies
                if self.file_type == 'stability':
                    # Stability-focused
                    if 'Stability_Adjusted_Score' in top_stocks.columns:
                        name = f"Stability_S{portfolio_size}_P{pool_size}"
                        stable_stocks = top_stocks.nlargest(portfolio_size, 'Stability_Adjusted_Score')
                        self.portfolios[name] = stable_stocks['Ticker'].tolist()

                    # Positive momentum
                    if 'Linear_Slope' in top_stocks.columns:
                        name = f"PosMomentum_S{portfolio_size}_P{pool_size}"
                        positive_momentum = top_stocks[top_stocks['Linear_Slope'] > 0]
                        if len(positive_momentum) >= portfolio_size:
                            self.portfolios[name] = positive_momentum.head(portfolio_size)['Ticker'].tolist()

                    # Recommended stocks
                    if 'Recommendation' in top_stocks.columns:
                        name = f"Recommended_S{portfolio_size}_P{pool_size}"
                        recommended = top_stocks[top_stocks['Recommendation'].str.contains('BUY', na=False)]
                        if len(recommended) >= portfolio_size:
                            self.portfolios[name] = recommended.head(portfolio_size)['Ticker'].tolist()

                    # Hybrid stability (only for certain sizes)
                    if portfolio_size in [15, 20, 25]:
                        name = f"HybridStability_S{portfolio_size}_P{pool_size}"
                        # Mix of stability and sector balance
                        portfolio = self._create_hybrid_stability_sector(top_stocks, portfolio_size)
                        if portfolio:
                            self.portfolios[name] = portfolio

                elif self.file_type == 'ranked':
                    # Factor-balanced portfolio
                    if all(col in top_stocks.columns for col in ['Value', 'Quality']):
                        name = f"FactorBal_S{portfolio_size}_P{pool_size}"
                        # Create balanced factor portfolio
                        portfolio = self._create_factor_balanced(top_stocks, portfolio_size)
                        if portfolio:
                            self.portfolios[name] = portfolio

        print(f"\n Total portfolios created: {len(self.portfolios)}")
        return self.portfolios

    def _create_sector_balanced(self, stocks_df, portfolio_size, max_per_sector):
        """Helper method to create sector-balanced portfolio"""
        selected = []
        sector_counts = {}

        for _, stock in stocks_df.iterrows():
            if len(selected) >= portfolio_size:
                break
            sector = stock.get('Sector', 'Unknown')
            if sector_counts.get(sector, 0) < max_per_sector:
                selected.append(stock['Ticker'])
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

        return selected if len(selected) == portfolio_size else None

    def _create_industry_balanced(self, stocks_df, portfolio_size):
        """Helper method to create industry-balanced portfolio"""
        selected = []
        industry_counts = {}
        max_per_industry = max(2, portfolio_size // 10)

        for _, stock in stocks_df.iterrows():
            if len(selected) >= portfolio_size:
                break
            industry = stock.get('Industry', 'Unknown')
            if industry_counts.get(industry, 0) < max_per_industry:
                selected.append(stock['Ticker'])
                industry_counts[industry] = industry_counts.get(industry, 0) + 1

        return selected if len(selected) == portfolio_size else None

    def _create_hybrid_stability_sector(self, stocks_df, portfolio_size):
        """Create hybrid portfolio balancing stability and sector diversification"""
        # Implementation similar to sector_balanced but with stability weighting
        return self._create_sector_balanced(stocks_df, portfolio_size, 3)

    def _create_factor_balanced(self, stocks_df, portfolio_size):
        """Create factor-balanced portfolio for ranked files"""
        # Simple implementation - can be enhanced
        return stocks_df.head(portfolio_size)['Ticker'].tolist()

    def quick_backtest_portfolios(self,
                                  lookback_days: int = 252,
                                  rebalance_frequency: str = 'monthly') -> Dict:
        """
        Simplified backtesting optimized for yfinance with error handling for delisted tickers
        """
        results = {}

        if not self.portfolios:
            print(" No portfolios to backtest. Create portfolios first.")
            return results

        print(f"\n Running quick backtests for {len(self.portfolios)} portfolios")
        print(f"  Lookback: {lookback_days} days, Rebalance: {rebalance_frequency}")

        # Pre-fetch all unique tickers
        all_tickers = list(set(
            ticker for portfolio in self.portfolios.values()
            for ticker in portfolio
        ))

        print(f"\n Pre-fetching data for {len(all_tickers)} unique tickers...")
        all_data = self.fetch_batch_data(all_tickers, lookback_days)

        if all_data.empty:
            print(" Failed to fetch market data")
            return results

        # Track available tickers
        available_tickers = list(all_data.columns)
        missing_tickers = set(all_tickers) - set(available_tickers)
        if missing_tickers:
            print(f" {len(missing_tickers)} tickers unavailable (delisted/invalid): {missing_tickers}")

        for portfolio_name, tickers in self.portfolios.items():
            print(f"\n Backtesting {portfolio_name}...")

            # Filter to only available tickers
            valid_tickers = [t for t in tickers if t in available_tickers]

            # Skip if too few valid tickers (less than 50% of original)
            if len(valid_tickers) < max(3, len(tickers) * 0.5):
                print(f"   Skipping - only {len(valid_tickers)}/{len(tickers)} tickers available")
                continue

            # Get data for valid tickers only
            portfolio_data = all_data[valid_tickers]

            if portfolio_data.empty or len(portfolio_data) < 20:  # Need minimum data points
                print(f"   Insufficient data for {portfolio_name}")
                continue

            try:
                # Calculate simple metrics
                returns = portfolio_data.pct_change().dropna()

                # Skip if returns are all zeros or too few data points
                if returns.empty or len(returns) < 20 or returns.std().sum() == 0:
                    print(f"   Invalid returns data for {portfolio_name}")
                    continue

                # Equal weight portfolio
                portfolio_returns = returns.mean(axis=1)

                # Check for valid returns
                if portfolio_returns.std() == 0 or len(portfolio_returns) < 20:
                    print(f"   Invalid portfolio returns for {portfolio_name}")
                    continue

                # Calculate metrics with error handling
                total_return = (1 + portfolio_returns).prod() - 1
                annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
                volatility = portfolio_returns.std() * np.sqrt(252)

                # Prevent division by zero for Sharpe ratio
                if volatility > 0:
                    sharpe_ratio = annual_return / volatility
                else:
                    sharpe_ratio = 0

                # Calculate max drawdown
                cumulative = (1 + portfolio_returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = drawdown.min()

                results[portfolio_name] = {
                    'tickers': valid_tickers,  # Store the valid tickers used
                    'original_tickers': tickers,  # Store original for reference
                    'metrics': {
                        'total_return': total_return,
                        'annual_return': annual_return,
                        'volatility': volatility,
                        'sharpe_ratio': sharpe_ratio,
                        'max_drawdown': max_drawdown
                    },
                    'data_points': len(portfolio_returns),
                    'valid_ticker_count': len(valid_tickers),
                    'original_ticker_count': len(tickers)
                }

                print(f"   Sharpe: {sharpe_ratio:.3f}, Return: {annual_return:.2%}, MaxDD: {max_drawdown:.2%}")
                print(f"     Using {len(valid_tickers)}/{len(tickers)} tickers")

            except Exception as e:
                print(f"   Error backtesting {portfolio_name}: {str(e)}")
                continue

        # Find best portfolio
        if results:
            valid_results = {k: v for k, v in results.items()
                             if v['metrics']['sharpe_ratio'] != 0}  # Filter out zero Sharpe ratios

            if valid_results:
                best_portfolio = max(valid_results.items(),
                                     key=lambda x: x[1]['metrics']['sharpe_ratio'])
                print(f"\n Best Portfolio: {best_portfolio[0]}")
                print(f"   Sharpe Ratio: {best_portfolio[1]['metrics']['sharpe_ratio']:.3f}")
                print(
                    f"   Using {best_portfolio[1]['valid_ticker_count']}/{best_portfolio[1]['original_ticker_count']} tickers")

        return results

    def export_results(self, regime: str = 'Steady_Growth', output_dir: str = None):
        """
        Export results to files with correct directory structure

        Args:
            regime: Current regime name (e.g., 'Steady_Growth', 'Strong_Bull', 'Crisis_Bear')
            output_dir: Optional custom output directory
        """
        from pathlib import Path

        # Standardize regime name
        regime = regime.replace(" ", "_").replace("/", "_")

        # Create the correct output path
        if output_dir:
            output_path = Path(output_dir)
        else:
            # Use the standard directory structure
            output_path = Path('./output/Dynamic_Portfolio_Selection') / regime

        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Export portfolios
        portfolios_df = pd.DataFrame([
            {'Portfolio': name, 'Tickers': ', '.join(tickers)}
            for name, tickers in self.portfolios.items()
        ])
        portfolios_file = output_path / f'portfolios_yfinance_{timestamp}.csv'
        portfolios_df.to_csv(portfolios_file, index=False)

        # Export backtest results if available
        if self.backtest_results:
            results_data = []
            for name, result in self.backtest_results.items():
                row = {'Portfolio': name}
                row.update(result.get('metrics', {}))
                results_data.append(row)

            results_df = pd.DataFrame(results_data)
            results_file = output_path / f'backtest_results_yfinance_{timestamp}.csv'
            results_df.to_csv(results_file, index=False)

            print(f"\n Results exported to {output_path}")
            return str(results_file)

        return str(portfolios_file)

    def export_best_portfolio_json(self, backtest_results: Dict, regime: str = 'Steady_Growth') -> str:
        """
        Export the best portfolio to JSON format compatible with Portfolio Optimizer

        Args:
            backtest_results: Dictionary with backtest results
            regime: Current regime name

        Returns:
            Path to the saved JSON file
        """
        from pathlib import Path
        import json
        from datetime import datetime

        if not backtest_results:
            return None

        # Find the best portfolio across all lookback periods
        best_sharpe = -float('inf')
        best_portfolio_name = None
        best_tickers = []

        for lookback, data in backtest_results.items():
            if isinstance(data, dict) and 'metrics' in data:
                sharpe = data['metrics'].get('sharpe_ratio', 0)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_portfolio_name = data.get('portfolio_name', '')
                    best_tickers = data.get('tickers', [])

        if not best_tickers:
            print(" No valid portfolio found for JSON export")
            return None

        # Prepare JSON structure
        portfolio_data = {
            "tickers": best_tickers,
            "date_saved": datetime.now().isoformat(),
            "has_shares": False,  # Default to false
            "portfolio_name": best_portfolio_name,
            "sharpe_ratio": round(best_sharpe, 3),
            "regime": regime,
            "data_source": "yfinance",
            "selection_method": "dynamic_portfolio_selection"
        }

        # Create output directory
        regime = regime.replace(" ", "_").replace("/", "_")
        output_path = Path('./output/Dynamic_Portfolio_Selection') / regime
        output_path.mkdir(parents=True, exist_ok=True)

        # Save JSON file
        timestamp = datetime.now().strftime('%Y%m%d')
        json_file = output_path / f'best_portfolio_{timestamp}.json'

        with open(json_file, 'w') as f:
            json.dump(portfolio_data, f, indent=2)

        print(f" Best portfolio saved to JSON: {json_file}")
        return str(json_file)


# Example usage
if __name__ == "__main__":
    # Initialize selector
    selector = DynamicPortfolioSelectorYFinance(
        './ranked_lists/common_top150_tickers_0830.csv',
        exclude_real_estate=True
    )

    # Create portfolios
    portfolios = selector.create_multiple_portfolios(
        portfolio_sizes=[10, 15, 20],
        top_pools=[50, 100, 150]
    )

    # Run quick backtests
    results = selector.quick_backtest_portfolios(
        lookback_days=252,
        rebalance_frequency='monthly'
    )

    # Export results
    selector.export_results()