"""
Dynamic Portfolio Selection System - Modified for Common Top Tickers List

This modified version can work with either:
1. Original top_ranked_stocks files (with factor scores)
2. Common top tickers files (with stability metrics)

The system automatically detects the file type and adjusts available strategies accordingly.

Version: 2.0 - Common Tickers Support
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import itertools
import json
from scipy.stats import entropy
import warnings
from marketstack_integration import MarketstackDataFetcher

# Import the MHRP Portfolio System from the backtest script
from Backtest_advanced_02 import MHRPPortfolioSystem

warnings.filterwarnings('ignore')

class DynamicPortfolioSelector:
    """
    Dynamic Portfolio Selection System that creates diverse portfolios
    Now supports both ranked stocks files and common tickers files
    """

    def __init__(self, stocks_file: str, exclude_real_estate: bool = True,
                 marketstack_api_key: str = None, file_type: str = 'auto'):
        """
        Initialize with stocks data

        Args:
            stocks_file: Path to CSV file with stocks (either ranked or common tickers)
            exclude_real_estate: Whether to exclude real estate and housing-related stocks
            marketstack_api_key: API key for Marketstack
            file_type: 'ranked', 'common', or 'auto' (auto-detect)
        """
        self.stocks_df = pd.read_csv(stocks_file)
        self.exclude_real_estate = exclude_real_estate
        self.portfolios = {}
        self.backtest_results = {}
        self.marketstack_api_key = marketstack_api_key

        # Initialize Marketstack fetcher if API key provided
        if marketstack_api_key:
            self.data_fetcher = MarketstackDataFetcher(marketstack_api_key)

        # Detect file type
        if file_type == 'auto':
            self.file_type = self._detect_file_type()
        else:
            self.file_type = file_type

        print(f"📁 Detected file type: {self.file_type.upper()}")

        # Define real estate and housing related sectors/industries to exclude
        self.real_estate_sectors = [
            'Real Estate',
            'REIT',
            'Real Estate Investment Trust',
            'Real Estate - Development',
            'Real Estate - Diversified',
            'Real Estate - General',
            'Real Estate Services'
        ]

        self.real_estate_industries = [
            'Real Estate - Development',
            'Real Estate - Diversified',
            'Real Estate - General',
            'Real Estate Services',
            'REIT - Diversified',
            'REIT - Healthcare Facilities',
            'REIT - Hotel & Motel',
            'REIT - Industrial',
            'REIT - Mortgage',
            'REIT - Office',
            'REIT - Residential',
            'REIT - Retail',
            'REIT - Specialty',
            'Residential Construction',
            'Engineering & Construction',
            'Building Materials',
            'Building Products & Equipment',
            'Home Improvement Retail',
            'Homebuilding & Construction',
            'Lumber & Wood Production',
            'Residential Construction'
        ]

        # Clean and prepare data
        self._prepare_data()

    # def _detect_file_type(self) -> str:
    #     """
    #     Detect whether the file is a ranked stocks file or common tickers file
    #
    #     Returns:
    #         'ranked' or 'common'
    #     """
    #     # Check for columns unique to each file type
    #     if 'Rank_in_Ranked' in self.stocks_df.columns and 'Linear_R2' in self.stocks_df.columns:
    #         return 'common'
    #     elif 'Value' in self.stocks_df.columns and 'Quality' in self.stocks_df.columns:
    #         return 'ranked'
    #     else:
    #         # Default to ranked if unclear
    #         print("⚠️ Could not definitively determine file type, assuming 'ranked'")
    #         return 'ranked'

    def _detect_file_type(self) -> str:
        """
        Detect whether the file is a ranked stocks file or common tickers file

        Returns:
            'ranked' or 'common'
        """
        # Check for columns unique to each file type
        columns_lower = [col.lower() for col in self.stocks_df.columns]

        # Common tickers file (from stability analysis) has these columns
        if 'linear_r2' in columns_lower or 'stability_adjusted_score' in columns_lower:
            return 'common'
        # Ranked stocks file has factor columns
        elif 'value' in columns_lower or 'quality' in columns_lower:
            return 'ranked'
        else:
            # Check for other indicators
            if 'ticker' in columns_lower and 'avg_score' in columns_lower:
                return 'common'
            else:
                print("⚠️ Could not definitively determine file type, assuming 'ranked'")
                return 'ranked'

    def _is_real_estate_related(self, sector: str, industry: str) -> bool:
        """
        Check if a stock is related to real estate or housing

        Args:
            sector: Stock sector
            industry: Stock industry

        Returns:
            True if the stock is real estate/housing related
        """
        if not isinstance(sector, str) or not isinstance(industry, str):
            return False

        sector_lower = sector.lower()
        industry_lower = industry.lower()

        # Check exact matches first
        if sector in self.real_estate_sectors or industry in self.real_estate_industries:
            return True

        # Check for keyword matches
        real_estate_keywords = [
            'real estate', 'reit', 'housing', 'homebuilding', 'construction',
            'building', 'property', 'residential', 'commercial property',
            'mortgage', 'lumber', 'home improvement'
        ]

        for keyword in real_estate_keywords:
            if keyword in sector_lower or keyword in industry_lower:
                return True

        return False

    def _prepare_data(self):
        """Clean and prepare the stocks data"""
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

        # Handle different score column names
        if self.file_type == 'common':
            # For stability analysis files - try multiple possible column names
            score_columns = ['Stability_Adjusted_Score', 'stability_adjusted_score',
                             'Avg_Score', 'avg_score']
            for col in score_columns:
                if col in self.stocks_df.columns:
                    self.stocks_df['Score'] = self.stocks_df[col]
                    break

            # Check if we have the merged file with sector/industry info
            if 'Sector' not in self.stocks_df.columns:
                print("⚠️ Warning: No Sector/Industry information in stability file")
                print("   Sector-based portfolio strategies will be limited")
                # Add placeholder columns
                self.stocks_df['Sector'] = 'Unknown'
                self.stocks_df['Industry'] = 'Unknown'
                self.stocks_df['CompanyName'] = self.stocks_df.get('CompanyName', self.stocks_df['Ticker'])
                self.stocks_df['Country'] = self.stocks_df.get('Country', 'Unknown')

        # Remove rows with missing critical data
        required_columns = ['Ticker', 'Score']
        self.stocks_df = self.stocks_df.dropna(subset=required_columns)

        # Filter out real estate stocks if requested and sector info is available
        if self.exclude_real_estate and 'Sector' in self.stocks_df.columns and self.stocks_df['Sector'].notna().any():
            initial_count = len(self.stocks_df)

            # Create mask for non-real estate stocks
            mask = ~self.stocks_df.apply(
                lambda row: self._is_real_estate_related(row.get('Sector', ''), row.get('Industry', '')),
                axis=1
            )

            excluded_stocks = self.stocks_df[~mask]
            self.stocks_df = self.stocks_df[mask]

            excluded_count = initial_count - len(self.stocks_df)

            if excluded_count > 0:
                print(f"🏠 Excluded {excluded_count} real estate/housing related stocks")

        # Sort and create rank based on file type
        if self.file_type == 'common':
            # For common tickers, sort by Score (higher is better for stability adjusted score)
            if 'Combined_Rank' in self.stocks_df.columns:
                # If we have combined rank from the comparison script
                self.stocks_df = self.stocks_df.sort_values('Combined_Rank').reset_index(drop=True)
            else:
                # Otherwise sort by score
                self.stocks_df = self.stocks_df.sort_values('Score', ascending=False).reset_index(drop=True)
            self.stocks_df['Rank'] = range(1, len(self.stocks_df) + 1)
        else:
            # For ranked stocks, sort by Score (higher is better)
            self.stocks_df = self.stocks_df.sort_values('Score', ascending=False).reset_index(drop=True)
            self.stocks_df['Rank'] = range(1, len(self.stocks_df) + 1)

        print(f"\n📊 Final dataset: {len(self.stocks_df)} stocks")
        if self.file_type == 'common':
            if 'Linear_R2' in self.stocks_df.columns:
                print(f"Average Linear R²: {self.stocks_df['Linear_R2'].mean():.3f}")
            print(f"Top score: {self.stocks_df['Score'].max():.3f}")
        else:
            print(f"Top score: {self.stocks_df['Score'].max():.3f}")
            print(f"Bottom score: {self.stocks_df['Score'].min():.3f}")

        if 'Sector' in self.stocks_df.columns and self.stocks_df['Sector'].notna().any():
            unique_sectors = self.stocks_df[self.stocks_df['Sector'] != 'Unknown']['Sector'].nunique()
            print(f"Unique sectors: {unique_sectors}")

    def get_top_stocks(self, n: int = 100) -> pd.DataFrame:
        """Get top N stocks (already sorted by best metric)"""
        return self.stocks_df.head(n)

    def create_portfolio_by_top_rank(self,
                                   portfolio_size: int = 20,
                                   top_pool: int = 100) -> List[str]:
        """
        Create portfolio by selecting top-ranked stocks
        Works with both file types

        Args:
            portfolio_size: Number of stocks in the portfolio
            top_pool: Pool size to select from (top N stocks)

        Returns:
            List of selected ticker symbols
        """
        top_stocks = self.get_top_stocks(top_pool)
        selected = top_stocks.head(portfolio_size)
        return selected['Ticker'].tolist()

    def create_sector_balanced_portfolio(self,
                                       portfolio_size: int = 20,
                                       top_pool: int = 100,
                                       max_per_sector: int = 3) -> List[str]:
        """
        Create sector-balanced portfolio ensuring diversification across sectors
        Works with both file types

        Args:
            portfolio_size: Target number of stocks
            top_pool: Pool to select from
            max_per_sector: Maximum stocks per sector

        Returns:
            List of selected ticker symbols
        """
        top_stocks = self.get_top_stocks(top_pool)
        selected_tickers = []
        sector_counts = {}

        for _, stock in top_stocks.iterrows():
            sector = stock['Sector']

            # Check if we can add this stock
            if (len(selected_tickers) < portfolio_size and
                sector_counts.get(sector, 0) < max_per_sector):

                selected_tickers.append(stock['Ticker'])
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

        return selected_tickers

    def create_industry_balanced_portfolio(self,
                                         portfolio_size: int = 20,
                                         top_pool: int = 100,
                                         max_per_industry: int = 2) -> List[str]:
        """
        Create industry-balanced portfolio ensuring diversification across industries
        Works with both file types

        Args:
            portfolio_size: Target number of stocks
            top_pool: Pool to select from
            max_per_industry: Maximum stocks per industry

        Returns:
            List of selected ticker symbols
        """
        top_stocks = self.get_top_stocks(top_pool)
        selected_tickers = []
        industry_counts = {}

        for _, stock in top_stocks.iterrows():
            industry = stock['Industry']

            # Check if we can add this stock
            if (len(selected_tickers) < portfolio_size and
                industry_counts.get(industry, 0) < max_per_industry):

                selected_tickers.append(stock['Ticker'])
                industry_counts[industry] = industry_counts.get(industry, 0) + 1

        return selected_tickers

    # New methods specific to common tickers file

    def create_stability_focused_portfolio(self,
                                         portfolio_size: int = 20,
                                         top_pool: int = 100,
                                         min_r2: float = 0.5) -> List[str]:
        """
        Create portfolio focused on stocks with stable trends (high R²)
        Only works with common tickers file

        Args:
            portfolio_size: Target number of stocks
            top_pool: Pool to select from
            min_r2: Minimum R² threshold

        Returns:
            List of selected ticker symbols
        """
        if self.file_type != 'common':
            print("⚠️ Stability-focused portfolio requires common tickers file, falling back to top rank")
            return self.create_portfolio_by_top_rank(portfolio_size, top_pool)

        top_stocks = self.get_top_stocks(top_pool).copy()

        # Filter by minimum R²
        stable_stocks = top_stocks[top_stocks['Linear_R2'] >= min_r2]

        if len(stable_stocks) < portfolio_size:
            print(f"⚠️ Only {len(stable_stocks)} stocks meet R² threshold of {min_r2}")
            # Fall back to sorting by R² without threshold
            stable_stocks = top_stocks.sort_values('Linear_R2', ascending=False)

        selected = stable_stocks.head(portfolio_size)
        return selected['Ticker'].tolist()

    def create_positive_momentum_portfolio(self,
                                         portfolio_size: int = 20,
                                         top_pool: int = 100,
                                         min_slope: float = 0.0) -> List[str]:
        """
        Create portfolio focused on stocks with positive linear slope
        Only works with common tickers file

        Args:
            portfolio_size: Target number of stocks
            top_pool: Pool to select from
            min_slope: Minimum slope threshold

        Returns:
            List of selected ticker symbols
        """
        if self.file_type != 'common':
            print("⚠️ Positive momentum portfolio requires common tickers file, falling back to top rank")
            return self.create_portfolio_by_top_rank(portfolio_size, top_pool)

        top_stocks = self.get_top_stocks(top_pool).copy()

        # Filter by positive slope
        momentum_stocks = top_stocks[top_stocks['Linear_Slope'] > min_slope]

        if len(momentum_stocks) < portfolio_size:
            print(f"⚠️ Only {len(momentum_stocks)} stocks have slope > {min_slope}")
            momentum_stocks = top_stocks.sort_values('Linear_Slope', ascending=False)

        selected = momentum_stocks.head(portfolio_size)
        return selected['Ticker'].tolist()

    # def create_recommendation_based_portfolio(self,
    #                                         portfolio_size: int = 20,
    #                                         top_pool: int = 100,
    #                                         recommendations: List[str] = None) -> List[str]:
    #     """
    #     Create portfolio based on recommendation categories
    #     Only works with common tickers file
    #
    #     Args:
    #         portfolio_size: Target number of stocks
    #         top_pool: Pool to select from
    #         recommendations: List of recommendations to include
    #
    #     Returns:
    #         List of selected ticker symbols
    #     """
    #     if self.file_type != 'common':
    #         print("⚠️ Recommendation-based portfolio requires common tickers file, falling back to top rank")
    #         return self.create_portfolio_by_top_rank(portfolio_size, top_pool)
    #
    #     if recommendations is None:
    #         recommendations = ['STRONG BUY - Elite performer', 'STRONG BUY - High conviction',
    #                          'BUY - Quality growth', 'BUY - High score acceptable risk']
    #
    #     top_stocks = self.get_top_stocks(top_pool).copy()
    #
    #     # Filter by recommendations
    #     recommended_stocks = top_stocks[top_stocks['Recommendation'].isin(recommendations)]
    #
    #     if len(recommended_stocks) < portfolio_size:
    #         print(f"⚠️ Only {len(recommended_stocks)} stocks meet recommendation criteria")
    #         # Add more recommendations if needed
    #         recommended_stocks = top_stocks
    #
    #     selected = recommended_stocks.head(portfolio_size)
    #     return selected['Ticker'].tolist()

    def create_recommendation_based_portfolio(self,
                                              portfolio_size: int = 20,
                                              top_pool: int = 100,
                                              recommendations: List[str] = None) -> List[str]:
        """
        Create portfolio based on recommendation categories
        Only works with common tickers file

        Args:
            portfolio_size: Target number of stocks
            top_pool: Pool to select from
            recommendations: List of recommendations to include

        Returns:
            List of selected ticker symbols
        """
        if self.file_type != 'common':
            print("⚠️ Recommendation-based portfolio requires common tickers file, falling back to top rank")
            return self.create_portfolio_by_top_rank(portfolio_size, top_pool)

        if recommendations is None:
            # Updated to include HOLD recommendations
            recommendations = [
                'STRONG BUY - Elite performer',
                'STRONG BUY - High conviction',
                'BUY - Quality growth',
                'BUY - Stable quality',
                'BUY - High score acceptable risk',
                'SPECULATIVE BUY - Strong momentum',
                'HOLD - Stable but flat',  # Add HOLD recommendations
                'HOLD - Needs monitoring'  # Add more HOLD categories
            ]

        top_stocks = self.get_top_stocks(top_pool).copy()

        # Filter by recommendations
        recommended_stocks = top_stocks[top_stocks['Recommendation'].isin(recommendations)]

        if len(recommended_stocks) < portfolio_size:
            print(f"⚠️ Only {len(recommended_stocks)} stocks meet recommendation criteria")
            # If not enough stocks, add the next best stocks by score
            if len(recommended_stocks) < portfolio_size:
                # Get stocks not in recommended_stocks
                remaining_stocks = top_stocks[~top_stocks['Ticker'].isin(recommended_stocks['Ticker'])]
                # Add the best remaining stocks
                additional_needed = portfolio_size - len(recommended_stocks)
                additional_stocks = remaining_stocks.head(additional_needed)
                recommended_stocks = pd.concat([recommended_stocks, additional_stocks])

        selected = recommended_stocks.head(portfolio_size)
        return selected['Ticker'].tolist()

    def create_hybrid_stability_sector_portfolio(self,
                                               portfolio_size: int = 20,
                                               top_pool: int = 100,
                                               max_per_sector: int = 3,
                                               stability_weight: float = 0.6) -> List[str]:
        """
        Create portfolio balancing stability (R²) and sector diversification
        Only works with common tickers file

        Args:
            portfolio_size: Target number of stocks
            top_pool: Pool to select from
            max_per_sector: Maximum stocks per sector
            stability_weight: Weight given to stability vs combined rank

        Returns:
            List of selected ticker symbols
        """
        if self.file_type != 'common':
            print("⚠️ Hybrid stability portfolio requires common tickers file, falling back to sector balanced")
            return self.create_sector_balanced_portfolio(portfolio_size, top_pool, max_per_sector)

        top_stocks = self.get_top_stocks(top_pool).copy()

        # Create a composite score weighted between R² and combined rank
        # Normalize both to 0-1 range
        r2_norm = (top_stocks['Linear_R2'] - top_stocks['Linear_R2'].min()) / \
                  (top_stocks['Linear_R2'].max() - top_stocks['Linear_R2'].min())

        # Check if Combined_Rank exists, otherwise use regular Rank
        if 'Combined_Rank' in top_stocks.columns:
            # For combined rank, lower is better, so invert
            rank_norm = 1 - ((top_stocks['Combined_Rank'] - top_stocks['Combined_Rank'].min()) / \
                             (top_stocks['Combined_Rank'].max() - top_stocks['Combined_Rank'].min()))
        else:
            # Use regular Rank column instead
            rank_norm = 1 - ((top_stocks['Rank'] - top_stocks['Rank'].min()) / \
                             (top_stocks['Rank'].max() - top_stocks['Rank'].min()))

        top_stocks['Composite_Score'] = (stability_weight * r2_norm) + ((1 - stability_weight) * rank_norm)

        # Sort by composite score
        top_stocks = top_stocks.sort_values('Composite_Score', ascending=False)

        # Select with sector balance constraint
        selected_tickers = []
        sector_counts = {}

        for _, stock in top_stocks.iterrows():
            sector = stock['Sector']

            if (len(selected_tickers) < portfolio_size and
                sector_counts.get(sector, 0) < max_per_sector):

                selected_tickers.append(stock['Ticker'])
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

        return selected_tickers

    # Modified existing methods to work with both file types

    def create_factor_balanced_portfolio(self,
                                       portfolio_size: int = 20,
                                       top_pool: int = 100,
                                       factor_weights: Optional[Dict[str, float]] = None) -> List[str]:
        """
        Create portfolio balanced across multiple factors
        Only works with ranked stocks file

        Args:
            portfolio_size: Target number of stocks
            top_pool: Pool to select from
            factor_weights: Weights for different factors

        Returns:
            List of selected ticker symbols
        """
        if self.file_type != 'ranked':
            print("⚠️ Factor-balanced portfolio requires ranked stocks file, falling back to top rank")
            return self.create_portfolio_by_top_rank(portfolio_size, top_pool)

        if factor_weights is None:
            factor_weights = {
                'Value': 0.2,
                'Quality': 0.2,
                'Technical': 0.15,
                'Momentum': 0.15,
                'Growth': 0.15,
                'Stability': 0.15
            }

        top_stocks = self.get_top_stocks(top_pool).copy()

        # Calculate composite factor score
        top_stocks['FactorScore'] = 0
        for factor, weight in factor_weights.items():
            if factor in top_stocks.columns:
                # Normalize factor scores to 0-1 range
                factor_values = top_stocks[factor].fillna(top_stocks[factor].median())
                if factor_values.max() != factor_values.min():
                    normalized = (factor_values - factor_values.min()) / \
                               (factor_values.max() - factor_values.min())
                else:
                    normalized = pd.Series([0.5] * len(factor_values), index=factor_values.index)
                top_stocks['FactorScore'] += weight * normalized

        # Sort by composite factor score and select top stocks
        top_stocks = top_stocks.sort_values('FactorScore', ascending=False)
        selected = top_stocks.head(portfolio_size)

        return selected['Ticker'].tolist()

    def create_multiple_portfolios(self,
                                 portfolio_sizes: List[int] = [10, 15, 20, 25, 30],
                                 top_pools: List[int] = [40, 50, 75, 100]) -> Dict[str, List[str]]:
        """
        Create multiple portfolios using different strategies and parameters
        Adapts strategies based on file type

        Args:
            portfolio_sizes: Different portfolio sizes to test
            top_pools: Different pool sizes to select from

        Returns:
            Dictionary mapping portfolio names to ticker lists
        """
        portfolios = {}

        for size in portfolio_sizes:
            for pool in top_pools:
                # Only create if pool is larger than size
                if pool < size:
                    continue

                # Universal strategies (work with both file types)

                # Top-ranked portfolio
                name = f"TopRank_S{size}_P{pool}"
                portfolios[name] = self.create_portfolio_by_top_rank(size, pool)

                # Sector-balanced portfolio
                for max_sector in [2, 3]:
                    name = f"SectorBal{max_sector}_S{size}_P{pool}"
                    portfolios[name] = self.create_sector_balanced_portfolio(size, pool, max_per_sector=max_sector)

                # Industry-balanced portfolio
                name = f"IndustryBal_S{size}_P{pool}"
                portfolios[name] = self.create_industry_balanced_portfolio(size, pool)

                # File-type specific strategies
                if self.file_type == 'common':
                    # Strategies for common tickers file

                    # Stability-focused portfolio
                    name = f"Stability_S{size}_P{pool}"
                    portfolios[name] = self.create_stability_focused_portfolio(size, pool)

                    # Positive momentum portfolio
                    name = f"PosMomentum_S{size}_P{pool}"
                    portfolios[name] = self.create_positive_momentum_portfolio(size, pool)

                    # Recommendation-based portfolio
                    name = f"Recommended_S{size}_P{pool}"
                    portfolios[name] = self.create_recommendation_based_portfolio(size, pool)

                    # Hybrid stability-sector portfolio
                    if size in [15, 20, 25]:  # Only for certain sizes
                        name = f"HybridStability_S{size}_P{pool}"
                        portfolios[name] = self.create_hybrid_stability_sector_portfolio(size, pool)

                else:  # ranked file type
                    # Factor-balanced portfolio
                    name = f"FactorBal_S{size}_P{pool}"
                    portfolios[name] = self.create_factor_balanced_portfolio(size, pool)

                    # Risk-adjusted portfolio
                    name = f"RiskAdj_S{size}_P{pool}"
                    portfolios[name] = self.create_risk_adjusted_portfolio(size, pool, vol_penalty_weight=0.3)

        # Print summary of created portfolios
        print(f"Created {len(portfolios)} portfolio combinations:")

        # Count by type
        strategy_counts = {}
        for name in portfolios.keys():
            strategy = name.split('_')[0]
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        for strategy, count in strategy_counts.items():
            print(f"  {strategy}: {count} variations")

        self.portfolios = portfolios
        return portfolios

    def create_risk_adjusted_portfolio(self,
                                     portfolio_size: int = 20,
                                     top_pool: int = 100,
                                     vol_penalty_weight: float = 0.3) -> List[str]:
        """
        Create portfolio considering volatility penalty
        Only works with ranked stocks file

        Args:
            portfolio_size: Target number of stocks
            top_pool: Pool to select from
            vol_penalty_weight: Weight for volatility penalty in selection

        Returns:
            List of selected ticker symbols
        """
        if self.file_type != 'ranked':
            print("⚠️ Risk-adjusted portfolio requires ranked stocks file, falling back to stability-focused")
            return self.create_stability_focused_portfolio(portfolio_size, top_pool)

        top_stocks = self.get_top_stocks(top_pool).copy()

        # Create risk-adjusted score
        stability_values = top_stocks['Stability'].fillna(top_stocks['Stability'].median())

        if stability_values.max() != stability_values.min():
            normalized_stability = (stability_values - stability_values.min()) / \
                                 (stability_values.max() - stability_values.min())
        else:
            normalized_stability = pd.Series([0.5] * len(stability_values), index=stability_values.index)

        # Adjust score by stability
        top_stocks['RiskAdjScore'] = top_stocks['Score'] + (vol_penalty_weight * normalized_stability)

        # Sort by risk-adjusted score and select
        top_stocks = top_stocks.sort_values('RiskAdjScore', ascending=False)
        selected = top_stocks.head(portfolio_size)

        return selected['Ticker'].tolist()

    # Keep all the other methods unchanged (analyze_portfolio_diversity, backtest_all_portfolios, etc.)
    # They should work with both file types as they only use Ticker, Sector, Industry columns

    def analyze_portfolio_diversity(self, portfolio_name: str, tickers: List[str]) -> Dict:
        """
        Analyze diversity metrics for a portfolio
        Works with both file types

        Args:
            portfolio_name: Name of the portfolio
            tickers: List of ticker symbols in the portfolio

        Returns:
            Dictionary with diversity metrics
        """
        # Get portfolio data
        portfolio_data = self.stocks_df[self.stocks_df['Ticker'].isin(tickers)]

        # Sector diversity
        sector_counts = portfolio_data['Sector'].value_counts()
        sector_entropy = entropy(sector_counts.values)
        max_sector_entropy = np.log(len(sector_counts))
        sector_diversity = sector_entropy / max_sector_entropy if max_sector_entropy > 0 else 0

        # Industry diversity
        industry_counts = portfolio_data['Industry'].value_counts()
        industry_entropy = entropy(industry_counts.values)
        max_industry_entropy = np.log(len(industry_counts))
        industry_diversity = industry_entropy / max_industry_entropy if max_industry_entropy > 0 else 0

        # Average rank and score
        avg_rank = portfolio_data['Rank'].mean()
        avg_score = portfolio_data['Score'].mean()
        score_std = portfolio_data['Score'].std()

        # Add stability metrics if available (common tickers file)
        results = {
            'portfolio_name': portfolio_name,
            'num_stocks': len(tickers),
            'num_sectors': len(sector_counts),
            'num_industries': len(industry_counts),
            'sector_diversity': sector_diversity,
            'industry_diversity': industry_diversity,
            'avg_rank': avg_rank,
            'avg_score': avg_score,
            'score_std': score_std,
            'max_sector_concentration': sector_counts.max() / len(tickers),
            'max_industry_concentration': industry_counts.max() / len(tickers)
        }

        if self.file_type == 'common':
            # Check for both uppercase and lowercase versions
            if 'Linear_R2' in portfolio_data.columns:
                results['avg_linear_r2'] = portfolio_data['Linear_R2'].mean()
            elif 'linear_r2' in portfolio_data.columns:
                results['avg_linear_r2'] = portfolio_data['linear_r2'].mean()

            if 'Linear_Slope' in portfolio_data.columns:
                results['avg_linear_slope'] = portfolio_data['Linear_Slope'].mean()
            elif 'linear_slope' in portfolio_data.columns:
                results['avg_linear_slope'] = portfolio_data['linear_slope'].mean()

            if 'Stability_Adjusted_Score' in portfolio_data.columns:
                results['avg_stability_score'] = portfolio_data['Stability_Adjusted_Score'].mean()
            elif 'stability_adjusted_score' in portfolio_data.columns:
                results['avg_stability_score'] = portfolio_data['stability_adjusted_score'].mean()

        return results

    # Include the backtest and other analysis methods unchanged...
    # (backtest_all_portfolios, find_best_portfolios, create_performance_comparison,
    # export_detailed_results, print_best_portfolio_summary methods remain the same)

    def backtest_all_portfolios(self,
                              start_date: str = "2020-01-01",
                              end_date: Optional[str] = None,
                              initial_cash: float = 100_000,
                              lookback_days: int = 252) -> Dict:
        """
        Backtest all created portfolios using the MHRP system (Drift-based only)

        Args:
            start_date: Start date for backtesting
            end_date: End date for backtesting
            initial_cash: Initial portfolio value
            lookback_days: Lookback period for optimization

        Returns:
            Dictionary with backtest results for each portfolio
        """
        # Initialize MHRP system
        mhrp_system = MHRPPortfolioSystem(
            initial_cash=initial_cash,
            fees=0.001,
            slippage=0.0005,
            marketstack_api_key=self.marketstack_api_key  # Pass API key
        )

        results = {}
        total_portfolios = len(self.portfolios)

        # ADD THIS: Data availability check
        start_dt = pd.to_datetime(start_date)
        required_start = start_dt - pd.DateOffset(days=lookback_days + 30)  # Add buffer
        print(f"\n📅 Lookback configuration:")
        print(f"   - Lookback period: {lookback_days} days (~{lookback_days / 252:.1f} years)")
        print(f"   - Start date: {start_date}")
        print(f"   - Data needed from: {required_start.strftime('%Y-%m-%d')}")
        print(f"   - This requires {(start_dt - required_start).days} days of historical data")

        print(f"🚀 Starting backtests for {total_portfolios} portfolios...")

        for i, (portfolio_name, tickers) in enumerate(self.portfolios.items(), 1):
            print(f"\n📊 Backtesting {portfolio_name} ({i}/{total_portfolios})")
            print(f"Tickers: {tickers}")

            # ADD THIS: Data fetching status
            print(f"   📥 Fetching {lookback_days} days of historical data...")

            try:
                # Run backtest using only Drift-based rebalancing
                backtest_result = mhrp_system.run_backtest_strategies(
                    tickers=tickers,
                    start_date=start_date,
                    end_date=end_date,
                    lookback_days=lookback_days,
                    rebalance_strategies=['drift']  # Only use drift-based
                )

                # ADD THIS: Success message
                print(f"   ✅ Data fetched successfully")

                # Extract drift results
                drift_result = backtest_result['Drift']
                portfolio = drift_result['portfolio']

                # Get returns series
                returns = portfolio.daily_returns()
                if isinstance(returns, pd.DataFrame):
                    returns = returns.iloc[:, 0]

                # Calculate key metrics
                import quantstats as qs

                metrics = {
                    'portfolio_name': portfolio_name,
                    'tickers': tickers,
                    'total_return': (1 + returns).prod() - 1,
                    'annual_return': qs.stats.cagr(returns),
                    'volatility': qs.stats.volatility(returns),
                    'sharpe_ratio': qs.stats.sharpe(returns),
                    'max_drawdown': qs.stats.max_drawdown(returns),
                    'calmar_ratio': qs.stats.calmar(returns),
                    'sortino_ratio': qs.stats.sortino(returns),
                    'rebalance_count': len(drift_result.get('rebalance_dates', [])),
                    'final_value': portfolio.value().iloc[-1],
                    'returns_series': returns,
                    'portfolio_object': portfolio
                }

                # Add diversity analysis
                diversity_metrics = self.analyze_portfolio_diversity(portfolio_name, tickers)
                metrics.update(diversity_metrics)

                results[portfolio_name] = metrics

                print(f"✅ Completed - Sharpe: {metrics['sharpe_ratio']:.3f}, Return: {metrics['annual_return']*100:.2f}%")

            except Exception as e:
                error_msg = str(e).lower()
                if 'no data' in error_msg or 'insufficient data' in error_msg:
                    print(f"   ❌ Insufficient historical data for {portfolio_name}")
                    print(
                        f"      Need data from {required_start.strftime('%Y-%m-%d')} but some tickers may not have enough history")
                elif 'download' in error_msg or 'fetch' in error_msg:
                    print(f"   ❌ Error downloading data for {portfolio_name}")
                    print(f"      Some tickers may be delisted or have connection issues")
                else:
                    print(f"   ❌ Error backtesting {portfolio_name}: {str(e)}")

                # ADD THIS: Show which tickers might be problematic
                print(f"      Tickers in portfolio: {', '.join(tickers[:5])}" +
                      (f"... and {len(tickers) - 5} more" if len(tickers) > 5 else ""))
                continue

        self.backtest_results = results
        return results

    def find_best_portfolios(self,
                           criteria: str = 'sharpe_ratio',
                           top_n: int = 5) -> pd.DataFrame:
        """
        Find the best performing portfolios based on specified criteria

        Args:
            criteria: Performance metric to rank by
            top_n: Number of top portfolios to return

        Returns:
            DataFrame with top performing portfolios
        """
        if not self.backtest_results:
            raise ValueError("No backtest results available. Run backtest_all_portfolios() first.")

        # Convert results to DataFrame for easier analysis
        results_list = []
        for name, metrics in self.backtest_results.items():
            # Create a copy of metrics without pandas objects for clean display
            clean_metrics = {k: v for k, v in metrics.items()
                           if k not in ['returns_series', 'portfolio_object', 'tickers']}
            clean_metrics['tickers_count'] = len(metrics['tickers'])
            results_list.append(clean_metrics)

        results_df = pd.DataFrame(results_list)

        # Sort by criteria (descending for performance metrics, ascending for drawdown)
        ascending = criteria in ['max_drawdown', 'volatility']
        top_portfolios = results_df.sort_values(criteria, ascending=ascending).head(top_n)

        return top_portfolios

    def create_performance_comparison(self, save_fig: bool = True) -> None:
        """
        Create comprehensive performance comparison visualizations

        Args:
            save_fig: Whether to save the figure to file
        """
        if not self.backtest_results:
            raise ValueError("No backtest results available. Run backtest_all_portfolios() first.")

        # Set up the plotting
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Portfolio Performance Comparison (Excluding Real Estate)', fontsize=16, fontweight='bold')

        # Prepare data
        results_df = pd.DataFrame([
            {k: v for k, v in metrics.items()
             if k not in ['returns_series', 'portfolio_object', 'tickers']}
            for metrics in self.backtest_results.values()
        ])

        # 1. Sharpe Ratio comparison
        axes[0, 0].barh(range(len(results_df)), results_df['sharpe_ratio'])
        axes[0, 0].set_yticks(range(len(results_df)))
        axes[0, 0].set_yticklabels(results_df['portfolio_name'], fontsize=8)
        axes[0, 0].set_xlabel('Sharpe Ratio')
        axes[0, 0].set_title('Sharpe Ratio by Portfolio')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Annual Return vs Volatility scatter
        axes[0, 1].scatter(results_df['volatility'] * 100, results_df['annual_return'] * 100,
                          s=100, alpha=0.7)
        for i, name in enumerate(results_df['portfolio_name']):
            axes[0, 1].annotate(name.split('_')[0],
                               (results_df['volatility'].iloc[i] * 100,
                                results_df['annual_return'].iloc[i] * 100),
                               fontsize=8, ha='center')
        axes[0, 1].set_xlabel('Volatility (%)')
        axes[0, 1].set_ylabel('Annual Return (%)')
        axes[0, 1].set_title('Risk-Return Profile')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Maximum Drawdown
        axes[0, 2].barh(range(len(results_df)), results_df['max_drawdown'] * 100)
        axes[0, 2].set_yticks(range(len(results_df)))
        axes[0, 2].set_yticklabels(results_df['portfolio_name'], fontsize=8)
        axes[0, 2].set_xlabel('Max Drawdown (%)')
        axes[0, 2].set_title('Maximum Drawdown by Portfolio')
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Portfolio diversity analysis
        axes[1, 0].scatter(results_df['sector_diversity'], results_df['industry_diversity'],
                          s=100, alpha=0.7)
        axes[1, 0].set_xlabel('Sector Diversity')
        axes[1, 0].set_ylabel('Industry Diversity')
        axes[1, 0].set_title('Portfolio Diversification')
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Performance by portfolio size
        size_groups = results_df.groupby('num_stocks').agg({
            'sharpe_ratio': 'mean',
            'annual_return': 'mean',
            'max_drawdown': 'mean'
        })

        axes[1, 1].bar(size_groups.index, size_groups['sharpe_ratio'])
        axes[1, 1].set_xlabel('Portfolio Size (Number of Stocks)')
        axes[1, 1].set_ylabel('Average Sharpe Ratio')
        axes[1, 1].set_title('Performance by Portfolio Size')
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Cumulative returns for top 5 portfolios
        top_5 = results_df.nlargest(5, 'sharpe_ratio')
        for i, row in top_5.iterrows():
            portfolio_name = row['portfolio_name']
            returns = self.backtest_results[portfolio_name]['returns_series']
            cum_returns = (1 + returns).cumprod()
            axes[1, 2].plot(cum_returns.index, cum_returns.values,
                           label=portfolio_name.split('_')[0], linewidth=2)

        axes[1, 2].set_ylabel('Cumulative Return')
        axes[1, 2].set_title('Cumulative Returns - Top 5 Portfolios')
        axes[1, 2].legend(fontsize=8)
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_fig:
            plt.savefig(r'./Results/portfolio_performance_comparison_final.png', dpi=300, bbox_inches='tight')
            print("📊 Performance comparison chart saved as 'portfolio_performance_comparison_final.png'")

        # plt.show()

    def export_detailed_results(self, filename: str = r'./Results/portfolio_selection_results_no_realestate.json') -> None:
        """
        Export detailed results including portfolios and performance metrics

        Args:
            filename: Output filename for results
        """
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'real_estate_excluded': self.exclude_real_estate,
            'summary': {
                'total_portfolios_tested': len(self.backtest_results),
                'best_portfolio_by_sharpe': None,
                'best_portfolio_by_return': None,
                'best_portfolio_by_drawdown': None
            },
            'portfolios': {},
            'performance_metrics': {}
        }

        if self.backtest_results:
            # Find best portfolios
            best_sharpe = max(self.backtest_results.items(), key=lambda x: x[1]['sharpe_ratio'])
            best_return = max(self.backtest_results.items(), key=lambda x: x[1]['annual_return'])
            best_drawdown = min(self.backtest_results.items(), key=lambda x: x[1]['max_drawdown'])

            export_data['summary']['best_portfolio_by_sharpe'] = best_sharpe[0]
            export_data['summary']['best_portfolio_by_return'] = best_return[0]
            export_data['summary']['best_portfolio_by_drawdown'] = best_drawdown[0]

            # Export portfolio compositions
            for name, tickers in self.portfolios.items():
                export_data['portfolios'][name] = {
                    'tickers': tickers,
                    'composition': []
                }

                # Add stock details for each portfolio
                portfolio_stocks = self.stocks_df[self.stocks_df['Ticker'].isin(tickers)]
                for _, stock in portfolio_stocks.iterrows():
                    stock_info = {
                        'ticker': stock['Ticker'],
                        'company': stock['CompanyName'],
                        'sector': stock['Sector'],
                        'industry': stock['Industry'],
                        'rank': int(stock['Rank']),
                        'score': float(stock['Score'])
                    }
                    export_data['portfolios'][name]['composition'].append(stock_info)

            # Export performance metrics
            for name, metrics in self.backtest_results.items():
                clean_metrics = {
                    k: (float(v) if isinstance(v, (int, float)) else str(v))
                    for k, v in metrics.items()
                    if k not in ['returns_series', 'portfolio_object', 'tickers']
                }
                export_data['performance_metrics'][name] = clean_metrics

        # Save to file
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        print(f"✅ Detailed results exported to {filename}")

    def print_best_portfolio_summary(self, top_n: int = 3) -> None:
        """
        Print a summary of the best performing portfolios

        Args:
            top_n: Number of top portfolios to display
        """
        if not self.backtest_results:
            print("No backtest results available.")
            return

        print("\n" + "="*60)
        print("🏆 BEST PORTFOLIO PERFORMANCE SUMMARY (NO REAL ESTATE)")
        print("="*60)

        # Get top portfolios by different metrics
        metrics = ['sharpe_ratio', 'annual_return', 'calmar_ratio']

        for metric in metrics:
            print(f"\n📊 Top {top_n} by {metric.replace('_', ' ').title()}:")
            print("-" * 50)

            top_portfolios = self.find_best_portfolios(criteria=metric, top_n=top_n)

            for i, (_, portfolio) in enumerate(top_portfolios.iterrows(), 1):
                print(f"{i}. {portfolio['portfolio_name']}")
                print(f"   Sharpe: {portfolio['sharpe_ratio']:.3f} | "
                      f"Return: {portfolio['annual_return']*100:.2f}% | "
                      f"Volatility: {portfolio['volatility']*100:.2f}% | "
                      f"Max DD: {portfolio['max_drawdown']*100:.2f}%")
                print(f"   Sectors: {portfolio['num_sectors']} | "
                      f"Industries: {portfolio['num_industries']} | "
                      f"Size: {portfolio['num_stocks']} stocks")
                print()

        # Best overall portfolio details
        best_portfolio_name = max(self.backtest_results.items(), key=lambda x: x[1]['sharpe_ratio'])[0]
        best_metrics = self.backtest_results[best_portfolio_name]

        print("\n🥇 BEST OVERALL PORTFOLIO (by Sharpe Ratio):")
        print("-" * 50)
        print(f"Portfolio: {best_portfolio_name}")
        print(f"Sharpe Ratio: {best_metrics['sharpe_ratio']:.3f}")
        print(f"Annual Return: {best_metrics['annual_return']*100:.2f}%")
        print(f"Volatility: {best_metrics['volatility']*100:.2f}%")
        print(f"Max Drawdown: {best_metrics['max_drawdown']*100:.2f}%")
        print(f"Calmar Ratio: {best_metrics['calmar_ratio']:.3f}")
        print(f"Number of Rebalances: {best_metrics['rebalance_count']}")
        print(f"\nStocks in Portfolio:")
        for ticker in best_metrics['tickers']:
            stock_info = self.stocks_df[self.stocks_df['Ticker'] == ticker].iloc[0]
            print(f"  {ticker} ({stock_info['CompanyName']}) - Rank: {int(stock_info['Rank'])}, "
                  f"Sector: {stock_info['Sector']}")

        print("\n" + "="*60)

    def add_custom_exclusions(self,
                             exclude_sectors: List[str] = None,
                             exclude_industries: List[str] = None,
                             exclude_tickers: List[str] = None) -> None:
        """
        Add custom exclusions beyond real estate

        Args:
            exclude_sectors: List of sectors to exclude
            exclude_industries: List of industries to exclude
            exclude_tickers: List of specific tickers to exclude
        """
        if exclude_sectors:
            self.real_estate_sectors.extend(exclude_sectors)

        if exclude_industries:
            self.real_estate_industries.extend(exclude_industries)

        if exclude_tickers:
            # Remove specific tickers from dataset
            initial_count = len(self.stocks_df)
            self.stocks_df = self.stocks_df[~self.stocks_df['Ticker'].isin(exclude_tickers)]
            excluded_count = initial_count - len(self.stocks_df)

            if excluded_count > 0:
                print(f"🚫 Excluded {excluded_count} additional tickers: {', '.join(exclude_tickers)}")

        # Re-prepare data with new exclusions
        self._prepare_data()

    def validate_data_availability(self, tickers: List[str], start_date: str, lookback_days: int) -> Dict[str, bool]:
        """
        Check if tickers have sufficient historical data

        Returns:
            Dictionary mapping ticker to availability status
        """
        if hasattr(self, 'data_fetcher'):
            # Use Marketstack
            print("   🔑 Using Marketstack API for validation...")
            return self.data_fetcher.validate_data_availability(
                tickers, start_date, lookback_days
            )
        else:
            # Keep original yfinance implementation
            print("   📊 Using yfinance for validation...")

        import yfinance as yf

        start_dt = pd.to_datetime(start_date)
        required_start = start_dt - pd.DateOffset(days=lookback_days + 30)

        print(f"\n🔍 Validating data availability for {len(tickers)} tickers...")
        print(f"   Required data from: {required_start.strftime('%Y-%m-%d')} to {start_date}")

        availability = {}
        insufficient_tickers = []

        for ticker in tickers:
            try:
                data = yf.download(ticker, start=required_start, end=start_dt, progress=False)
                has_sufficient_data = len(data) >= lookback_days * 0.8  # 80% threshold
                availability[ticker] = has_sufficient_data
                if not has_sufficient_data:
                    insufficient_tickers.append(ticker)
            except:
                availability[ticker] = False
                insufficient_tickers.append(ticker)

        if insufficient_tickers:
            print(
                f"   ⚠️ {len(insufficient_tickers)} tickers have insufficient data: {', '.join(insufficient_tickers[:5])}")
            if len(insufficient_tickers) > 5:
                print(f"      ... and {len(insufficient_tickers) - 5} more")

        return availability

if __name__ == "__main__":
    # Your Marketstack API key
    MARKETSTACK_API_KEY = "476419cceb4330259e5a126753335b72"  # Use your actual key

    # Initialize the portfolio selector
    print("🚀 Initializing Dynamic Portfolio Selection System...")
    selector = DynamicPortfolioSelector(r'./ranked_lists/common_top150_tickers_0712.csv', exclude_real_estate=True)

    # Create portfolios
    print("\n🔧 Creating multiple portfolio combinations...")
    portfolios = selector.create_multiple_portfolios(
        portfolio_sizes=[5, 10, 15],
        top_pools=[60]
    )

    # Test different lookback periods
    # lookback_periods = [252, 504, 756, 1008, 1260]  # 1, 2, 3, 4, 5 years
    lookback_periods = [252, 504, 756]  # 1, 2, 3 years

    # ADD THIS: Pre-flight data check
    print("\n🔍 Pre-flight data availability check...")
    # test_ticker = selector.stocks_df.iloc[0]['Ticker']  # Get first ticker as test
    # print(f"   Testing data availability with ticker: {test_ticker}")

    # Pre-flight data check with a better ticker
    print("\n🔍 Pre-flight data availability check...")
    # Use a ticker likely to have long history
    test_tickers = ['NFLX', 'NVDA', 'TSM', 'CBOE', 'MNST']
    for test_ticker in test_tickers:
        if test_ticker in selector.stocks_df['Ticker'].values:
            print(f"   Testing data availability with ticker: {test_ticker}")
            try:
                import yfinance as yf

                test_data = yf.download(test_ticker,
                                        start='2017-01-01',  # 3+ years back from 2020
                                        end='2020-01-01',
                                        progress=False)
                if len(test_data) > 0:
                    actual_start = test_data.index[0].strftime('%Y-%m-%d')
                    print(f"   ✅ Test ticker {test_ticker} has data from {actual_start}")
                    print(f"   📊 Total days available: {len(test_data)}")
                    break
            except Exception as e:
                continue

    try:
        import yfinance as yf

        test_data = yf.download(test_ticker,
                                start='2015-01-01',  # 5+ years back from 2020
                                end='2020-01-01',
                                progress=False)
        if len(test_data) > 0:
            actual_start = test_data.index[0].strftime('%Y-%m-%d')
            print(f"   ✅ Test ticker has data from {actual_start}")
            print(f"   📊 Total days available: {len(test_data)}")
        else:
            print(f"   ⚠️ Warning: Test ticker has no data for the required period")
    except Exception as e:
        print(f"   ⚠️ Warning: Could not verify data availability: {str(e)}")
    ###

    print("\n📊 Testing different lookback periods for robustness...")
    best_results_by_lookback = {}

    for lookback in lookback_periods:
        print(f"\n{'=' * 60}")
        print(f"Testing with {lookback}-day lookback (~{lookback / 252:.1f} years)")
        print('=' * 60)

        # ADD THIS: Calculate required data start date
        required_data_start = pd.to_datetime("2020-01-01") - pd.DateOffset(days=lookback + 30)
        print(f"📅 This lookback requires data from {required_data_start.strftime('%Y-%m-%d')}")

        # Run backtests with current lookback period

        # Instead of starting from 2020-01-01, start from 2021-01-01
        # This gives more stocks time to have IPO'd and accumulate history
        backtest_results = selector.backtest_all_portfolios(
            start_date="2021-01-01",  # Changed from 2020-01-01
            end_date=None,
            initial_cash=100_000,
            lookback_days=lookback
        )

        # Find best portfolio for this lookback period
        if backtest_results:
            best_portfolio = max(backtest_results.items(),
                                 key=lambda x: x[1]['sharpe_ratio'])
            best_results_by_lookback[lookback] = {
                'portfolio_name': best_portfolio[0],
                'sharpe_ratio': best_portfolio[1]['sharpe_ratio'],
                'annual_return': best_portfolio[1]['annual_return'],
                'max_drawdown': best_portfolio[1]['max_drawdown'],
                'volatility': best_portfolio[1]['volatility']
            }

            # Save results for this specific lookback period
            print(f"\n💾 Saving results for {lookback}-day lookback...")
            selector.create_performance_comparison(save_fig=True)
            plt.savefig(f'./Results/portfolio_performance_comparison_{lookback}days.png', dpi=300, bbox_inches='tight')
            plt.close()  # Close to avoid memory issues

            selector.export_detailed_results(f'./Results/portfolio_selection_results_{lookback}days.json')

            print(f"Best portfolio: {best_portfolio[0]}")
            print(f"Sharpe Ratio: {best_portfolio[1]['sharpe_ratio']:.3f}")
            print(f"Annual Return: {best_portfolio[1]['annual_return'] * 100:.2f}%")

    # Compare results across lookback periods
    print("\n📈 LOOKBACK PERIOD COMPARISON")
    print("-" * 80)
    print(f"{'Lookback (days)':<20} {'Best Portfolio':<30} {'Sharpe':<10} {'Return':<10}")
    print("-" * 80)

    for lookback, results in best_results_by_lookback.items():
        print(f"{lookback:<20} {results['portfolio_name']:<30} "
              f"{results['sharpe_ratio']:<10.3f} "
              f"{results['annual_return'] * 100:<10.2f}%")

    # ADD THESE LINES: Create performance comparison and export results
    # Use the last lookback period's results for visualization and export
    if backtest_results:  # Check if we have results from the last lookback period
        # Print summary of best portfolios
        selector.print_best_portfolio_summary(top_n=3)

        # Create performance comparison charts
        print("\n📈 Creating performance comparison charts...")
        selector.create_performance_comparison(save_fig=True)

        # Export detailed results
        print("\n💾 Exporting detailed results...")
        today = datetime.now().strftime("%m%d")  # e.g. 0503
        selector.export_detailed_results(f'./Results/portfolio_selection_results_{today}.json')

        print("\n✅ Analysis Complete! Check the generated charts and exported results.")
        print("💡 Consider running the best performing portfolio through additional analysis")
        print("   using the walk-forward testing functionality in the MHRP system.")
        print("🏠 Real estate and housing-related stocks have been excluded from all portfolios.")