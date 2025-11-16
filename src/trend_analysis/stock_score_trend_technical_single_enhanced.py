"""
Enhanced Single Ticker Technical Analysis Script
Uses the ranked technical analysis approach for single ticker
Generates both plots and technical rankings CSV
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
import shutil
warnings.filterwarnings('ignore')

# Import the ranked analyzer class
# from stock_score_trend_technical_02 import StockScoreTrendAnalyzerTechnical
from stock_score_trend_technical_03 import StockScoreTrendAnalyzerTechnical


class EnhancedSingleTickerTechnicalAnalyzer(StockScoreTrendAnalyzerTechnical):
    """Enhanced single ticker analyzer using ranked approach"""

    def clean_single_ticker_directory(self, ticker, output_base_dir):
        """Clean old plots for a single ticker"""
        # Clean from both possible locations
        tech_plots_dir = Path(output_base_dir) / "technical_plots" / ticker
        ticker_dir = Path(output_base_dir) / ticker

        for dir_to_clean in [tech_plots_dir, ticker_dir]:
            if dir_to_clean.exists():
                print(f"\n Cleaning old plots for {ticker}...")
                try:
                    shutil.rmtree(dir_to_clean)
                    print(f"    Deleted old folder: {dir_to_clean}")
                except Exception as e:
                    print(f"    Error deleting {dir_to_clean}: {str(e)}")

    def analyze_single_ticker_with_ranking(self, ticker, output_base_dir, min_appearances=3):
        """
        Analyze a single ticker using the comprehensive ranking approach

        Parameters:
        -----------
        ticker : str
            The ticker symbol to analyze
        output_base_dir : str
            Base directory for output
        min_appearances : int
            Minimum number of appearances required for analysis
        """
        try:
            print(f"\n{'=' * 60}")
            print(f"ENHANCED TECHNICAL ANALYSIS FOR {ticker}")
            print(f"{'=' * 60}")

            # Load ranking files and build score history
            files_data = self.load_ranking_files()
            self.build_score_history(files_data)

            # Check if ticker exists in the data
            if ticker not in self.score_history:
                print(f" Error: Ticker '{ticker}' not found in the ranking data!")
                print(f"   Available tickers: {len(self.score_history)} total")
                return False, None

            # Check if ticker has enough data
            data_points = len(self.score_history[ticker]['indices'])
            if data_points < min_appearances:
                print(f" Warning: Ticker '{ticker}' has insufficient data!")
                print(f"   Found: {data_points} data points, Required: {min_appearances}")
                print(f"   Proceeding with limited analysis...")

            # Create output directories - ensure it's in technical_plots subdirectory
            tech_plots_dir = Path(output_base_dir) / "technical_plots" / ticker
            tech_plots_dir.mkdir(parents=True, exist_ok=True)

            # Clean old plots for this ticker
            self.clean_single_ticker_directory(ticker, output_base_dir)

            print(f" Output directory: {tech_plots_dir}")
            print(f" Data points available: {data_points}")

            # Perform comprehensive technical analysis
            print(f" Running technical analysis...")

            # Run the technical analysis for this single ticker
            self.technical_results[ticker] = {
                'moving_averages': self.calculate_moving_averages(ticker),
                'momentum': self.calculate_momentum_indicators(ticker),
                'bollinger_bands': self.calculate_bollinger_bands(ticker),
                'trend_strength': self.calculate_trend_strength(ticker),
                'forecasting': self.perform_statistical_forecasting(ticker)
            }

            # Perform regression analysis
            self.regression_results[ticker] = self.perform_regression_analysis(ticker)

            # Create comprehensive plots
            print(f" Creating technical plots...")
            # Pass use_subfolder=True to ensure plots go to technical_plots/ticker/
            self.create_comprehensive_plots(ticker, output_base_dir, use_subfolder=True)

            # Generate technical score and ranking
            print(f" Calculating technical score...")
            score_data = self.calculate_technical_score(ticker)

            if score_data:
                # Create single-ticker ranking CSV
                ranking_data = self.create_single_ticker_ranking(ticker, score_data)

                # Save ranking CSV
                ranking_file = tech_plots_dir / f"{ticker}_technical_ranking.csv"
                ranking_df = pd.DataFrame([ranking_data])
                ranking_df.to_csv(ranking_file, index=False)

                print(f" Technical ranking saved: {ranking_file}")

                # Display results
                self.display_single_ticker_results(ticker, score_data, ranking_data)

                return True, ranking_data
            else:
                print(f" Failed to calculate technical score for {ticker}")
                return False, None

        except Exception as e:
            print(f" Error analyzing {ticker}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, None

    def create_single_ticker_ranking(self, ticker, score_data):
        """Create ranking data for single ticker"""
        # Define indicator columns in the same order as ranked script
        indicator_columns = [
            'Trend Direction', 'Trend Quality', 'SMA Position', 'Recent Cross',
            'MACD Signal', 'MACD Histogram', 'RSI Level', 'RSI Trend',
            'ROC', '%B Position', 'BB Width', 'ADX Strength', 'DI Direction',
            'ARIMA', 'Exp Smooth', 'Prophet'
        ]

        # Start with basic info
        row_data = {
            'Ticker': ticker,
            'Score': round(score_data['total_score'], 1)
        }

        # Initialize all indicators
        for col in indicator_columns:
            row_data[col] = 0

        # Map detailed scores to columns
        details = score_data['details']
        detail_to_column = {
            'trend': 'Trend Direction',
            'trend_quality': 'Trend Quality',
            'sma_cross': 'SMA Position',
            'recent_cross': 'Recent Cross',
            'macd': 'MACD Signal',
            'macd_hist': 'MACD Histogram',
            'rsi_level': 'RSI Level',
            'rsi_trend': 'RSI Trend',
            'roc': 'ROC',
            'bb_position': '%B Position',
            'bb_width': 'BB Width',
            'adx_strength': 'ADX Strength',
            'di_direction': 'DI Direction',
            'arima': 'ARIMA',
            'exp_smooth': 'Exp Smooth',
            'prophet': 'Prophet'
        }

        # Extract scores from detail strings
        for detail_key, detail_value in details.items():
            if detail_key in detail_to_column:
                column = detail_to_column[detail_key]

                # Extract score from detail string
                if '(+1.5)' in str(detail_value):
                    row_data[column] = 1.5
                elif '(+1)' in str(detail_value):
                    row_data[column] = 1
                elif '(+0.5)' in str(detail_value):
                    row_data[column] = 0.5
                elif '(+0.3)' in str(detail_value):
                    row_data[column] = 0.3
                elif '(0)' in str(detail_value) or '(+0)' in str(detail_value):
                    row_data[column] = 0
                else:
                    # Default interpretation
                    if 'bullish' in str(detail_value).lower() or 'positive' in str(detail_value).lower():
                        row_data[column] = 1
                    elif 'neutral' in str(detail_value).lower():
                        row_data[column] = 0.5
                    else:
                        row_data[column] = 0

        # Add recommendation
        row_data['Recommendation'] = score_data['recommendation']

        return row_data

    def display_single_ticker_results(self, ticker, score_data, ranking_data):
        """Display comprehensive results for single ticker"""
        print(f"\n{'=' * 60}")
        print(f"TECHNICAL ANALYSIS RESULTS FOR {ticker}")
        print(f"{'=' * 60}")

        # Basic score info
        print(f" Technical Score: {score_data['total_score']:.1f}/{score_data['max_possible']:.1f}")
        print(f" Percentage Score: {score_data['percentage']:.1f}%")
        print(f" Recommendation: {score_data['recommendation']}")

        # Detailed breakdown
        print(f"\n DETAILED INDICATOR BREAKDOWN:")
        print(f"{'-' * 40}")

        # Group indicators by category
        trend_indicators = ['trend', 'trend_quality', 'adx_strength', 'di_direction']
        ma_indicators = ['sma_cross', 'recent_cross']
        momentum_indicators = ['macd', 'macd_hist', 'rsi_level', 'rsi_trend', 'roc']
        bb_indicators = ['bb_position', 'bb_width']
        forecast_indicators = ['arima', 'exp_smooth', 'prophet']

        categories = [
            (" Trend Analysis", trend_indicators),
            (" Moving Averages", ma_indicators),
            (" Momentum Indicators", momentum_indicators),
            (" Bollinger Bands", bb_indicators),
            (" Forecasting", forecast_indicators)
        ]

        for category_name, indicators in categories:
            category_items = [(k, v) for k, v in score_data['details'].items() if k in indicators]
            if category_items:
                print(f"\n{category_name}:")

                # Special handling for forecasting - add actual values and percentages
                if category_name == " Forecasting":
                    # Get forecast data from technical_results
                    forecast_data = self.technical_results.get(ticker, {}).get('forecasting', {})
                    current_score = self.score_history[ticker]['scores'][-1] if ticker in self.score_history else None

                    if forecast_data and current_score is not None:
                        print(f"  • Current Score      : {current_score:.2f}")
                        print(f"  • Forecast Range     : {len(forecast_data.get('arima', {}).get('forecast', []))} days")
                        print()

                    for key, value in category_items:
                        formatted_key = key.replace('_', ' ').title()

                        # Add forecast values and percentages if available
                        if forecast_data and current_score is not None:
                            if key == 'arima' and forecast_data.get('arima'):
                                final_val = forecast_data['arima']['forecast'][-1]
                                pct_change = ((final_val - current_score) / current_score) * 100
                                print(f"  • {formatted_key:20s}: {value}  →  {final_val:.2f} ({pct_change:+.1f}%)")
                            elif key == 'exp_smooth' and forecast_data.get('exp_smoothing'):
                                final_val = forecast_data['exp_smoothing']['forecast'][-1]
                                pct_change = ((final_val - current_score) / current_score) * 100
                                print(f"  • {formatted_key:20s}: {value}  →  {final_val:.2f} ({pct_change:+.1f}%)")
                            elif key == 'prophet' and forecast_data.get('prophet'):
                                final_val = forecast_data['prophet']['forecast'][-1]
                                pct_change = ((final_val - current_score) / current_score) * 100
                                lower = forecast_data['prophet']['lower_bound'][-1]
                                upper = forecast_data['prophet']['upper_bound'][-1]
                                offset = forecast_data['prophet'].get('offset', 0)
                                print(f"  • {formatted_key:20s}: {value}  →  {final_val:.2f} ({pct_change:+.1f}%)  [CI: {lower:.2f}-{upper:.2f}]")
                                if offset != 0:
                                    print(f"     Offset correction: {offset:+.2f}")
                            else:
                                print(f"  • {formatted_key:20s}: {value}")
                        else:
                            print(f"  • {formatted_key:20s}: {value}")
                else:
                    # Regular indicators - just print as before
                    for key, value in category_items:
                        formatted_key = key.replace('_', ' ').title()
                        print(f"  • {formatted_key:20s}: {value}")

        # Summary statistics
        print(f"\n RANKING SUMMARY:")
        print(f"{'-' * 40}")

        # Count positive indicators
        positive_count = sum(1 for col in ranking_data.keys()
                           if col not in ['Ticker', 'Score', 'Recommendation']
                           and ranking_data[col] > 0)
        total_indicators = len([col for col in ranking_data.keys()
                              if col not in ['Ticker', 'Score', 'Recommendation']])

        print(f"  Positive Indicators: {positive_count}/{total_indicators}")
        print(f"  Technical Score: {ranking_data['Score']:.1f}")
        print(f"  Final Recommendation: {ranking_data['Recommendation']}")

        # Signal strength
        if score_data['percentage'] >= 70:
            signal_emoji = ""
            signal_text = "VERY STRONG"
        elif score_data['percentage'] >= 55:
            signal_emoji = ""
            signal_text = "STRONG"
        elif score_data['percentage'] >= 45:
            signal_emoji = ""
            signal_text = "NEUTRAL"
        elif score_data['percentage'] >= 30:
            signal_emoji = ""
            signal_text = "WEAK"
        else:
            signal_emoji = ""
            signal_text = "VERY WEAK"

        print(f"\n{signal_emoji} Signal Strength: {signal_text}")
        print(f"{'=' * 60}")


# Main execution
if __name__ == "__main__":
    # Configuration
    TICKER = "AAPL"  # Change this to analyze different ticker
    CSV_DIRECTORY = "./ranked_lists"
    OUTPUT_DIRECTORY = "./output/Score_Trend_Analysis_Results"
    # START_DATE = "0101"  # Load all data from Jan 1st onwards (for technical plots)

    # Auto-detect earliest available file
    csv_files = sorted(Path(CSV_DIRECTORY).glob("top_ranked_stocks_*.csv"))
    if csv_files:
        START_DATE = csv_files[0].stem.split('_')[-1]  # Use earliest file date
    else:
        START_DATE = f"{datetime.now().year}0101"  # Fallback

    END_DATE = "0831"
    MIN_APPEARANCES = 3

    print(f"\n{'='*60}")
    print(f"ENHANCED SINGLE TICKER TECHNICAL ANALYSIS")
    print(f"{'='*60}")
    print(f"Ticker: {TICKER}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Ranked Lists Directory: {CSV_DIRECTORY}")
    print(f"Output Directory: {OUTPUT_DIRECTORY}")
    print(f"Minimum Appearances: {MIN_APPEARANCES}")
    print(f"{'='*60}\n")

    # Create output directory if it doesn't exist
    Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

    # Initialize analyzer
    analyzer = EnhancedSingleTickerTechnicalAnalyzer(
        csv_directory=CSV_DIRECTORY,
        start_date=START_DATE,
        end_date=END_DATE,
        sigmoid_sensitivity=5
    )

    # Run analysis
    success, ranking_data = analyzer.analyze_single_ticker_with_ranking(
        ticker=TICKER,
        output_base_dir=OUTPUT_DIRECTORY,
        min_appearances=MIN_APPEARANCES
    )

    if success:
        print(f"\n Analysis complete for {TICKER}!")
        print(f" Results saved to: {OUTPUT_DIRECTORY}/technical_plots/{TICKER}/")
    else:
        print(f"\n Analysis failed for {TICKER}")
        print(f"   Please check if the ticker exists in the ranking data")