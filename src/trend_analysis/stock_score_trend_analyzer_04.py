"""
Stock Score Trend Analyzer with Market Regime Support
Analyzes historical ranking data to predict future scores and assess stability
V03: Added market regime selector for Steady_Growth and Crisis_Bear analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

class StockScoreTrendAnalyzer:
    def __init__(self, csv_directory="./ranked_lists", start_date="0611", end_date="0621", sigmoid_sensitivity=5):
        self.csv_directory = Path(csv_directory)
        self.start_date = start_date
        self.end_date = end_date
        self.score_history = {}
        self.regression_results = {}
        self.sigmoid_sensitivity = sigmoid_sensitivity

    def load_ranking_files(self):
        """Load all ranking CSV files and assign sequential indices"""
        files_data = []

        # Find all matching CSV files in the specified directory
        csv_files = sorted(self.csv_directory.glob("top_ranked_stocks_*.csv"))

        # Filter files within date range if specified
        filtered_files = []
        # for csv_file in csv_files:
        #     date_str = csv_file.stem.split('_')[-1]  # e.g., "0612"
        #     if len(date_str) == 4:
        #         # Simple string comparison works for MMDD format
        #         if self.start_date <= date_str <= self.end_date:
        #             filtered_files.append((date_str, csv_file))

        for csv_file in csv_files:
            date_str = csv_file.stem.split('_')[-1]  # e.g., "20250612"
            if len(date_str) == 8:  # Changed from 4 to 8
                # Simple string comparison works for YYYYMMDD format
                if self.start_date <= date_str <= self.end_date:
                    filtered_files.append((date_str, csv_file))

        # Sort by date string and assign sequential indices
        filtered_files.sort(key=lambda x: x[0])

        for idx, (date_str, csv_file) in enumerate(filtered_files):
            df = pd.read_csv(csv_file)
            files_data.append({
                'date': date_str,
                'sequence_index': idx,  # Simple sequential index: 0, 1, 2, ...
                'data': df
            })
            print(f"Loaded {date_str} as index {idx}")

        print(f"\nTotal files loaded: {len(files_data)}")
        return files_data

    def build_score_history(self, files_data):
        """Build historical score data for each stock"""
        for file_info in files_data:
            seq_idx = file_info['sequence_index']
            df = file_info['data']
            date = file_info['date']

            for _, row in df.iterrows():
                ticker = row['Ticker']
                score = row['Score']

                if ticker not in self.score_history:
                    self.score_history[ticker] = {
                        'indices': [],
                        'dates': [],
                        'scores': [],
                        'ranks': [],
                        'factors': {}
                    }

                self.score_history[ticker]['indices'].append(seq_idx)
                self.score_history[ticker]['dates'].append(date)
                self.score_history[ticker]['scores'].append(score)
                self.score_history[ticker]['ranks'].append(row.name + 1)  # Rank position

                # Store individual factor scores
                for factor in ['Value', 'Quality', 'FinancialHealth', 'Momentum',
                               'Technical', 'Stability', 'Growth']:
                    if factor in row:
                        if factor not in self.score_history[ticker]['factors']:
                            self.score_history[ticker]['factors'][factor] = []
                        self.score_history[ticker]['factors'][factor].append(row[factor])

    def perform_regression_analysis(self, ticker, degree=1):
        """Perform regression analysis for a specific stock"""
        if ticker not in self.score_history:
            return None

        data = self.score_history[ticker]
        if len(data['indices']) < 3:  # Need at least 3 points
            return None

        X = np.array(data['indices']).reshape(-1, 1)
        y = np.array(data['scores'])

        results = {'ticker': ticker}

        # Linear regression
        lr = LinearRegression()
        lr.fit(X, y)
        y_pred_linear = lr.predict(X)

        results['linear'] = {
            'slope': lr.coef_[0],
            'intercept': lr.intercept_,
            'r2': r2_score(y, y_pred_linear),
            'predictions': y_pred_linear,
            'model': lr
        }

        # Polynomial regression if requested
        if degree > 1 and len(data['indices']) > degree + 1:
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(X)

            pr = LinearRegression()
            pr.fit(X_poly, y)
            y_pred_poly = pr.predict(X_poly)

            results['polynomial'] = {
                'degree': degree,
                'r2': r2_score(y, y_pred_poly),
                'predictions': y_pred_poly,
                'model': pr,
                'poly_features': poly
            }

        # Calculate stability metrics
        results['stability_metrics'] = self._calculate_stability_metrics(data)

        return results

    def _calculate_stability_metrics(self, data):
        """Calculate various stability metrics"""
        scores = np.array(data['scores'])
        ranks = np.array(data['ranks'])

        metrics = {
            'score_std': np.std(scores),
            'score_cv': np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else np.inf,
            'rank_std': np.std(ranks),
            'max_score_change': np.max(np.abs(np.diff(scores))) if len(scores) > 1 else 0,
            'max_rank_change': np.max(np.abs(np.diff(ranks))) if len(ranks) > 1 else 0,
            'score_range': np.max(scores) - np.min(scores),
            'trend_consistency': self._calculate_trend_consistency(scores)
        }

        return metrics

    def _calculate_trend_consistency(self, scores):
        """Calculate how consistently the score moves in one direction"""
        if len(scores) < 2:
            return 0

        differences = np.diff(scores)
        positive_moves = np.sum(differences > 0)
        negative_moves = np.sum(differences < 0)

        # Consistency score: 1 = always same direction, 0 = perfectly alternating
        total_moves = len(differences)
        consistency = abs(positive_moves - negative_moves) / total_moves

        return consistency

    def predict_future_scores(self, ticker, periods_ahead=5):
        """Predict future scores using the regression model"""
        if ticker not in self.regression_results:
            return None

        results = self.regression_results[ticker]
        last_index = max(self.score_history[ticker]['indices'])
        future_indices = np.array([last_index + i for i in range(1, periods_ahead + 1)]).reshape(-1, 1)

        predictions = {}

        # Linear predictions
        if 'linear' in results:
            predictions['linear'] = results['linear']['model'].predict(future_indices)

        # Polynomial predictions
        if 'polynomial' in results:
            poly_features = results['polynomial']['poly_features']
            future_indices_poly = poly_features.transform(future_indices)
            predictions['polynomial'] = results['polynomial']['model'].predict(future_indices_poly)

        return predictions

    def analyze_all_stocks(self, min_appearances=5, degree=2):
        """Analyze all stocks and rank by stability/predictability"""
        files_data = self.load_ranking_files()
        self.build_score_history(files_data)

        analysis_results = []

        for ticker in self.score_history:
            if len(self.score_history[ticker]['indices']) >= min_appearances:
                regression_result = self.perform_regression_analysis(ticker, degree)
                if regression_result:
                    self.regression_results[ticker] = regression_result

                    # Create summary for ranking
                    summary = {
                        'ticker': ticker,
                        'appearances': len(self.score_history[ticker]['indices']),
                        'avg_score': np.mean(self.score_history[ticker]['scores']),
                        'linear_slope': regression_result['linear']['slope'],
                        'linear_r2': regression_result['linear']['r2'],
                        'score_std': regression_result['stability_metrics']['score_std'],
                        'score_cv': regression_result['stability_metrics']['score_cv'],
                        'trend_consistency': regression_result['stability_metrics']['trend_consistency'],
                        'avg_rank': np.mean(self.score_history[ticker]['ranks'])
                    }

                    # Add stability score (composite metric)
                    summary['stability_score'] = self._calculate_composite_stability_score(
                        regression_result
                    )

                    # Calculate R² adjusted score
                    summary['r2_adjusted_score'] = summary['avg_score'] * summary['linear_r2']

                    # After calculating r2_adjusted_score
                    sigmoid_factor = 1 / (1 + np.exp(-self.sigmoid_sensitivity * summary['linear_slope']))
                    summary['slope_adjusted_score'] = summary['r2_adjusted_score'] * (0.5 + sigmoid_factor)

                    # Add stability factor using CV (since CV = STD/Mean)
                    stability_factor = 1 / (1 + summary['score_std'])
                    summary['stability_adjusted_score'] = summary['slope_adjusted_score'] * stability_factor

                    # Round numerical values for cleaner output
                    summary['avg_score'] = round(summary['avg_score'], 2)
                    summary['linear_slope'] = round(summary['linear_slope'], 3)
                    summary['linear_r2'] = round(summary['linear_r2'], 3)  # Keep 3 for R²
                    summary['score_std'] = round(summary['score_std'], 2)
                    summary['score_cv'] = round(summary['score_cv'], 3)
                    summary['trend_consistency'] = round(summary['trend_consistency'], 2)
                    summary['avg_rank'] = round(summary['avg_rank'], 2)
                    summary['stability_score'] = round(summary['stability_score'], 2)
                    summary['r2_adjusted_score'] = round(summary['r2_adjusted_score'], 2)
                    summary['slope_adjusted_score'] = round(summary['slope_adjusted_score'], 2)
                    summary['stability_adjusted_score'] = round(summary['stability_adjusted_score'], 2)

                    analysis_results.append(summary)

        return pd.DataFrame(analysis_results)

    def _calculate_composite_stability_score(self, regression_result):
        """Calculate a composite stability score (0-100)"""
        metrics = regression_result['stability_metrics']
        linear = regression_result['linear']

        # Components of stability score
        r2_score = linear['r2'] * 30  # R² contributes 30 points max
        cv_score = max(0, 30 - metrics['score_cv'] * 100)  # Low CV is good
        consistency_score = metrics['trend_consistency'] * 20  # Consistency contributes 20 points

        # Positive slope bonus (if trending up consistently)
        slope_bonus = 0
        if linear['slope'] > 0 and linear['r2'] > 0.5:
            slope_bonus = min(20, linear['slope'] * 1000)  # Cap at 20 points

        total_score = r2_score + cv_score + consistency_score + slope_bonus
        return min(100, max(0, total_score))

    def plot_stock_trends(self, tickers, save_path=None):
        """Plot score trends for specified stocks"""
        n_stocks = len(tickers)
        fig, axes = plt.subplots(n_stocks, 1, figsize=(12, 4*n_stocks))
        if n_stocks == 1:
            axes = [axes]

        for idx, ticker in enumerate(tickers):
            if ticker not in self.score_history:
                continue

            ax = axes[idx]
            data = self.score_history[ticker]

            # Plot actual scores
            ax.scatter(data['indices'], data['scores'], label='Actual Scores', s=50)

            # Plot regression lines
            if ticker in self.regression_results:
                results = self.regression_results[ticker]
                indices_array = np.array(data['indices'])

                # Linear regression line
                if 'linear' in results:
                    ax.plot(indices_array, results['linear']['predictions'],
                            'r-', label=f"Linear (R²={results['linear']['r2']:.3f})",
                            linewidth=2)

                # Polynomial regression line
                if 'polynomial' in results:
                    ax.plot(indices_array, results['polynomial']['predictions'],
                            'g--', label=f"Poly (R²={results['polynomial']['r2']:.3f})",
                            linewidth=2)

                # Add future predictions
                future_predictions = self.predict_future_scores(ticker, periods_ahead=5)
                if future_predictions:
                    future_indices = list(range(max(data['indices'])+1, max(data['indices'])+6))

                    if 'linear' in future_predictions:
                        ax.plot(future_indices, future_predictions['linear'],
                                'r:', label='Linear Forecast', linewidth=2)

                    # Add confidence interval
                    ax.axvspan(max(data['indices']), max(future_indices),
                               alpha=0.1, color='gray', label='Forecast Period')

            # Add date labels on x-axis for reference
            ax.set_xticks(data['indices'])
            ax.set_xticklabels(data['dates'], rotation=45, ha='right')

            ax.set_title(f"{ticker} - Score Trend Analysis", fontsize=14)
            ax.set_xlabel("Trading Days")
            ax.set_ylabel("Score")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        # if save_path:
        #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()

    def export_results(self, output_path="./results/stability_analysis_results.csv",
                       ranked_file_path=None):
        """Export analysis results to CSV with optional sector/industry info"""
        df = self.analyze_all_stocks()

        # Sort by stability adjusted score
        # df = df.sort_values('stability_adjusted_score', ascending=False)

        df = df.sort_values('slope_adjusted_score', ascending=False)

        # Add recommendation AFTER sorting
        df['recommendation'] = df.apply(self._generate_recommendation, axis=1)

        # Add sector/industry information if available
        if ranked_file_path is None:
            # Try to find the most recent ranked file automatically
            try:
                # Try to detect regime name from directory path
                regime_name = Path(self.csv_directory).name

                # Try new pattern first (with regime name)
                csv_files = sorted(Path(self.csv_directory).glob(f"top_ranked_stocks_{regime_name}*.csv"))

                # Fallback to old pattern if no files found
                if not csv_files:
                    csv_files = sorted(Path(self.csv_directory).glob("top_ranked_stocks_*.csv"))

                if csv_files:
                    ranked_file_path = csv_files[-1]
                    print(f"Using ranked file for sector info: {ranked_file_path.name}")
            except Exception as e:
                print(f"Warning: Could not find ranked file: {e}")
                pass

        if ranked_file_path and Path(ranked_file_path).exists():
            try:
                # Load sector/industry data
                ranked_df = pd.read_csv(ranked_file_path)

                # Check if required columns exist
                required_cols = ['Ticker', 'CompanyName', 'Country', 'Sector', 'Industry']
                missing_cols = [col for col in required_cols if col not in ranked_df.columns]

                if missing_cols:
                    print(f"Warning: Missing columns in ranked file: {missing_cols}")
                    # Use only available columns
                    available_cols = [col for col in required_cols if col in ranked_df.columns]
                    if 'Ticker' in available_cols:  # At minimum we need Ticker
                        sector_info = ranked_df[available_cols]
                    else:
                        print("Error: No 'Ticker' column in ranked file")
                        sector_info = None
                else:
                    sector_info = ranked_df[required_cols]

                if sector_info is not None:
                    # Merge with our results (ticker is lowercase in stability results)
                    df = df.merge(
                        sector_info,
                        left_on='ticker',
                        right_on='Ticker',
                        how='left'
                    )

                    # Drop the duplicate Ticker column and reorder
                    if 'Ticker' in df.columns:
                        df = df.drop('Ticker', axis=1)

                    # Reorder columns to put company info early
                    first_cols = ['ticker']
                    optional_cols = ['CompanyName', 'Country', 'Sector', 'Industry']
                    for col in optional_cols:
                        if col in df.columns:
                            first_cols.append(col)

                    other_cols = [col for col in df.columns if col not in first_cols]
                    df = df[first_cols + other_cols]

                    # Fill missing values for any columns that exist
                    for col in optional_cols:
                        if col in df.columns:
                            df[col] = df[col].fillna('Unknown')

            except Exception as e:
                print(f"Warning: Could not merge sector info: {e}")

        df.to_csv(output_path, index=False)
        print(f"Results exported to: {output_path}")
        return df

    # def _generate_recommendation(self, row):
    #     """Generate investment recommendation based on stability adjusted score"""
    #     stability_adj = row['stability_adjusted_score']
    #     avg_score = row['avg_score']
    #     r2 = row['linear_r2']
    #     slope = row['linear_slope']
    #     score_std = row['score_std']
    #
    #     # Elite tier - Best of the best
    #     if stability_adj >= 65 and slope > 0.5 and r2 > 0.8:
    #         return "STRONG BUY - Elite performer"
    #
    #     # High conviction buys
    #     elif stability_adj >= 55 and slope > 0.3 and r2 > 0.7:
    #         return "STRONG BUY - High conviction"
    #
    #     # Quality growth
    #     elif stability_adj >= 45 and slope > 0.2 and r2 > 0.6:
    #         return "BUY - Quality growth"
    #
    #     # Stable high scorers
    #     elif stability_adj >= 40 and score_std < 3 and avg_score > 60:
    #         return "BUY - Stable quality"
    #
    #     # Good score worth some volatility
    #     elif avg_score >= 70 and stability_adj >= 35:
    #         return "BUY - High score acceptable risk"
    #
    #     # Momentum plays
    #     elif slope > 1.0 and stability_adj >= 30:
    #         return "SPECULATIVE BUY - Strong momentum"
    #
    #     # Hold positions
    #     elif stability_adj >= 35 and abs(slope) < 0.1:
    #         return "HOLD - Stable but flat"
    #
    #     elif stability_adj >= 30 and slope > 0:
    #         return "HOLD - Modest potential"
    #
    #     # Deteriorating positions
    #     elif slope < -0.5 and r2 > 0.7:
    #         return "SELL - Reliable downtrend"
    #
    #     elif slope < -0.3 and stability_adj < 30:
    #         return "REDUCE - Weakening position"
    #
    #     # High volatility warning
    #     elif score_std > 10:
    #         return "AVOID - Too volatile"
    #
    #     # Poor overall score
    #     elif stability_adj < 25:
    #         return "AVOID - Poor risk-adjusted score"
    #
    #     # Weak fundamentals
    #     elif avg_score < 30:
    #         return "AVOID - Weak fundamentals"
    #
    #     # Default
    #     else:
    #         return "WATCH - Monitor for opportunity"

    def _generate_recommendation(self, row):
        """
        Generate investment recommendation based on stability adjusted score
        Updated with more realistic thresholds for current market conditions
        """
        stability_adj = row['stability_adjusted_score']
        avg_score = row['avg_score']
        r2 = row['linear_r2']
        slope = row['linear_slope']
        score_std = row['score_std']

        # --- STRONG BUY TIER (Top performers) ---
        # Elite tier - Exceptional stocks (loosened from 65/0.5/0.8)
        if stability_adj >= 50 and slope > 0.3 and r2 > 0.7:
            return "STRONG BUY - Elite performer"

        # High conviction buys (loosened from 55/0.3/0.7)
        elif stability_adj >= 45 and slope > 0.2 and r2 > 0.6:
            return "STRONG BUY - High conviction"

        # Strong momentum with good stability (new category)
        elif stability_adj >= 40 and slope > 0.4 and r2 > 0.5:
            return "STRONG BUY - Strong momentum"

        # --- BUY TIER (Good opportunities) ---
        # Quality growth (loosened from 45/0.2/0.6)
        elif stability_adj >= 38 and slope > 0.1 and r2 > 0.5:
            return "BUY - Quality growth"

        # Stable high scorers (loosened from 40/3/60)
        elif stability_adj >= 35 and score_std < 5 and avg_score > 55:
            return "BUY - Stable quality"

        # Good score worth some volatility (loosened from 70/35)
        elif avg_score >= 65 and stability_adj >= 32:
            return "BUY - High score acceptable risk"

        # Consistent performers with positive trend (new category)
        elif stability_adj >= 33 and slope > 0.05 and r2 > 0.4:
            return "BUY - Consistent performer"

        # --- SPECULATIVE BUY TIER ---
        # Strong momentum plays (loosened from 1.0/30)
        elif slope > 0.5 and stability_adj >= 28:
            return "SPECULATIVE BUY - Strong momentum"

        # Recovery plays (new category)
        elif slope > 0.3 and stability_adj >= 25 and avg_score > 50:
            return "SPECULATIVE BUY - Recovery candidate"

        # --- HOLD TIER (Neutral positions) ---
        # Stable but flat (loosened from 35)
        elif stability_adj >= 30 and abs(slope) < 0.1:
            return "HOLD - Stable but flat"

        # Modest potential (loosened from 30)
        elif stability_adj >= 28 and slope > 0:
            return "HOLD - Modest potential"

        # Good fundamentals but uncertain trend (new category)
        elif avg_score >= 50 and stability_adj >= 25:
            return "HOLD - Good fundamentals"

        # Needs monitoring (new category for borderline cases)
        elif stability_adj >= 25 and stability_adj < 28:
            return "HOLD - Needs monitoring"

        # --- REDUCE/SELL TIER ---
        # Reliable downtrend (keep strict for safety)
        elif slope < -0.5 and r2 > 0.7:
            return "SELL - Reliable downtrend"

        # Deteriorating position (slightly loosened)
        elif slope < -0.2 and stability_adj < 25:
            return "REDUCE - Weakening position"

        # Weak momentum (new category)
        elif slope < -0.1 and avg_score < 45:
            return "REDUCE - Weak momentum"

        # --- AVOID TIER (High risk) ---
        # Too volatile (loosened from 10)
        elif score_std > 15:
            return "AVOID - Too volatile"

        # Poor risk-adjusted score (loosened from 25)
        elif stability_adj < 20:
            return "AVOID - Poor risk-adjusted score"

        # Weak fundamentals (loosened from 30)
        elif avg_score < 35:
            return "AVOID - Weak fundamentals"

        # Very poor trend (new category)
        elif slope < -0.3 and r2 > 0.5:
            return "AVOID - Strong downtrend"

        # --- DEFAULT ---
        else:
            return "WATCH - Monitor for opportunity"


# Example usage
if __name__ == "__main__":
    # ===== CONFIGURATION SECTION =====
    # Modify these variables for easy customization

    # MARKET REGIME SELECTOR
    # 0 = Steady_Growth, 1 = Crisis_Bear
    MARKET_REGIME = 0
    END_DATE = "0822"

    # Define regime-specific settings
    if MARKET_REGIME == 0:
        REGIME_NAME = "Steady_Growth"
        START_DATE = "0611"  # Steady Growth period
        # END_DATE = "0809"
    elif MARKET_REGIME == 1:
        REGIME_NAME = "Crisis_Bear"
        START_DATE = "0719"  # Crisis/Bear period
        # END_DATE = "0809"
    else:
        raise ValueError("MARKET_REGIME must be 0 (Steady_Growth) or 1 (Crisis_Bear)")

    # Directory containing the ranking CSV files (regime-specific)
    CSV_DIRECTORY = f"./ranked_lists/{REGIME_NAME}"

    # Output directory for results (regime-specific)
    OUTPUT_DIRECTORY = f"./results/{REGIME_NAME}"

    # Sigmoid sensitivity for slope adjustment (3-5 for long-term investment)
    SIGMOID_SENSITIVITY = 5

    print(f"\n{'='*60}")
    print(f"STOCK SCORE TREND ANALYZER - MARKET REGIME: {REGIME_NAME}")
    print(f"{'='*60}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Ranked Lists Directory: {CSV_DIRECTORY}")
    print(f"Output Directory: {OUTPUT_DIRECTORY}")
    print(f"{'='*60}\n")

    # ===== END CONFIGURATION =====

    # Create output directory if it doesn't exist
    Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

    # Initialize analyzer with regime-specific settings
    analyzer = StockScoreTrendAnalyzer(
        csv_directory=CSV_DIRECTORY,
        start_date=START_DATE,
        end_date=END_DATE,
        sigmoid_sensitivity=SIGMOID_SENSITIVITY
    )

    # Run analysis
    # today = datetime.now().strftime("%m%d")  # e.g. 0807
    # output_file = f"{OUTPUT_DIRECTORY}/stability_analysis_results_{today}.csv"

    today = datetime.now().strftime("%Y%m%d")  # e.g. 20250807
    output_file = f"{OUTPUT_DIRECTORY}/stability_analysis_results_{today}.csv"

    # results_df = analyzer.export_results(output_path=output_file)

    # Find the latest ranked file for sector info
    ranked_file = None
    try:
        # ranked_files = sorted(Path(CSV_DIRECTORY).glob("top_ranked_stocks_*.csv"))

        # Extract regime from directory
        ranked_files = sorted(Path(CSV_DIRECTORY).glob(f"top_ranked_stocks_{REGIME_NAME}*.csv"))

        # Fallback to old pattern if needed
        if not ranked_files:
            ranked_files = sorted(Path(CSV_DIRECTORY).glob("top_ranked_stocks_*.csv"))

        if ranked_files:
            ranked_file = ranked_files[-1]
    except:
        pass

    results_df = analyzer.export_results(
        output_path=output_file,
        ranked_file_path=ranked_file
    )

    # Display top stable stocks
    print("\nTop 10 Most Stable Stocks with Positive Trends:")
    print("-" * 80)
    top_stable = results_df[
        (results_df['linear_slope'] > 0) &
        (results_df['linear_r2'] > 0.5)
    ].head(10)

    if not top_stable.empty:
        print(top_stable[
            ['ticker', 'stability_adjusted_score', 'slope_adjusted_score',
             'avg_score', 'score_cv', 'linear_r2', 'recommendation']
        ].to_string(index=False))
    else:
        print("No stocks found matching the criteria.")

    # Plot trends for top 5 stocks
    if not top_stable.empty:
        top_tickers = top_stable['ticker'].head(5).tolist()
        if top_tickers:
            plot_path = f"{OUTPUT_DIRECTORY}/top_stable_trends.png"
            print(f"\nGenerating plot for top 5 stable stocks...")
            analyzer.plot_stock_trends(top_tickers, save_path=plot_path)
            print(f"Plot saved to: {plot_path}")

    # Identify concerning patterns
    print("\n\nStocks with Concerning Patterns (High Volatility):")
    print("-" * 80)
    volatile_stocks = results_df[results_df['score_cv'] > 0.1].head(10)
    if not volatile_stocks.empty:
        print(volatile_stocks[['ticker', 'score_cv', 'stability_score', 'recommendation']].to_string(index=False))
    else:
        print("No highly volatile stocks found.")

    # Show the impact of slope adjustment
    print("\n\nImpact of Slope Adjustment (Top 10):")
    print("-" * 80)
    if not results_df.empty:
        comparison = results_df.head(10)[
            ['ticker', 'avg_score', 'linear_r2', 'r2_adjusted_score',
             'slope_adjusted_score', 'linear_slope']
        ].copy()
        comparison['adjustment_factor'] = (
            comparison['slope_adjusted_score'] / comparison['r2_adjusted_score']
        ).round(3)
        print(comparison.to_string(index=False))

    print(f"\n{'='*60}")
    print(f"Analysis complete for {REGIME_NAME} regime!")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")