"""
Single Ticker Technical Analysis Script
Performs comprehensive technical analysis on a single selected ticker
Saves plots in organized folder structure
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

# Import the main analyzer class from the original script
from stock_score_trend_technical_02 import StockScoreTrendAnalyzerTechnical

class SingleTickerTechnicalAnalyzer(StockScoreTrendAnalyzerTechnical):
    """Extended class for single ticker analysis"""

    def analyze_single_ticker(self, ticker, output_base_dir, min_appearances=3):
        """
        Analyze a single ticker and create all technical plots

        Parameters:
        -----------
        ticker : str
            The ticker symbol to analyze
        output_base_dir : str
            Base directory for output
        min_appearances : int
            Minimum number of appearances required for analysis
        """
        # Load ranking files and build score history
        files_data = self.load_ranking_files()
        self.build_score_history(files_data)

        # Check if ticker exists in the data
        if ticker not in self.score_history:
            print(f"\n❌ Error: Ticker '{ticker}' not found in the ranking data!")
            print(f"   The ticker might not be in the top ranked stocks for the selected date range.")
            return False

        # Check if ticker has enough data
        data_points = len(self.score_history[ticker]['indices'])
        if data_points < min_appearances:
            print(f"\n⚠️ Warning: Ticker '{ticker}' has insufficient data!")
            print(f"   Found: {data_points} data points")
            print(f"   Required: {min_appearances} data points")
            print(f"   Proceeding with limited analysis...")

        # Create output directory for this single ticker
        tech_dir = Path(output_base_dir) / "technical_plots_single" / ticker
        tech_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"ANALYZING TICKER: {ticker}")
        print(f"{'='*60}")
        print(f"Data points available: {data_points}")
        print(f"Output directory: {tech_dir}")
        print("="*60)

        try:
            # Perform regression analysis
            print(f"\n📊 Performing regression analysis...")
            self.regression_results[ticker] = self.perform_regression_analysis(ticker)

            # Perform technical analysis
            print(f"📈 Calculating technical indicators...")
            self.technical_results[ticker] = {
                'moving_averages': self.calculate_moving_averages(ticker),
                'momentum': self.calculate_momentum_indicators(ticker),
                'bollinger_bands': self.calculate_bollinger_bands(ticker),
                'trend_strength': self.calculate_trend_strength(ticker),
                'forecasting': self.perform_statistical_forecasting(ticker)
            }

            # Create plots
            print(f"🎨 Creating visualizations...")
            self.create_comprehensive_plots(ticker, tech_dir.parent)

            # Generate detailed report for this ticker
            self.generate_single_ticker_report(ticker, tech_dir)

            print(f"\n✅ Analysis complete for {ticker}!")
            print(f"📁 All files saved to: {tech_dir}")

            # Display key metrics
            self.display_key_metrics(ticker)

            return True

        except Exception as e:
            print(f"\n❌ Error analyzing {ticker}: {str(e)}")
            return False

    def generate_single_ticker_report(self, ticker, output_dir):
        """Generate a detailed report for the single ticker"""
        report_path = output_dir / f"{ticker}_analysis_report.txt"

        # with open(report_path, 'w') as f:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"TECHNICAL ANALYSIS REPORT FOR {ticker}\n")
            f.write("="*60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Date Range: {self.start_date} to {self.end_date}\n\n")

            # Data summary
            data = self.score_history[ticker]
            f.write("DATA SUMMARY:\n")
            f.write("-"*40 + "\n")
            f.write(f"Data Points: {len(data['indices'])}\n")
            f.write(f"Average Score: {np.mean(data['scores']):.2f}\n")
            f.write(f"Score Std Dev: {np.std(data['scores']):.2f}\n")
            f.write(f"Min Score: {np.min(data['scores']):.2f}\n")
            f.write(f"Max Score: {np.max(data['scores']):.2f}\n")
            f.write(f"Score Range: {np.max(data['scores']) - np.min(data['scores']):.2f}\n\n")

            # Regression results
            if ticker in self.regression_results and self.regression_results[ticker]:
                reg = self.regression_results[ticker]
                f.write("REGRESSION ANALYSIS:\n")
                f.write("-"*40 + "\n")
                if 'linear' in reg:
                    f.write(f"Linear R²: {reg['linear']['r2']:.3f}\n")
                    f.write(f"Linear Slope: {reg['linear']['slope']:.3f}\n")
                if 'polynomial' in reg:
                    f.write(f"Polynomial R²: {reg['polynomial']['r2']:.3f}\n")
                    f.write(f"Polynomial Degree: {reg['polynomial']['degree']}\n")
                f.write("\n")

            # Technical indicators
            tech = self.technical_results.get(ticker, {})

            # Moving averages
            if tech.get('moving_averages'):
                ma = tech['moving_averages']
                f.write("MOVING AVERAGES:\n")
                f.write("-"*40 + "\n")

                # Check for recent crossovers
                if 'crossover_signal' in ma:
                    recent_signals = ma['crossover_signal'][-5:]
                    bullish_count = recent_signals.count('bullish')
                    bearish_count = recent_signals.count('bearish')
                    f.write(f"Recent Bullish Crossovers (last 5): {bullish_count}\n")
                    f.write(f"Recent Bearish Crossovers (last 5): {bearish_count}\n")

                    # Last signal
                    for i in range(len(ma['crossover_signal'])-1, -1, -1):
                        if ma['crossover_signal'][i] != 'none':
                            f.write(f"Last Crossover: {ma['crossover_signal'][i]} at index {i}\n")
                            break
                f.write("\n")

            # Momentum indicators
            if tech.get('momentum'):
                mom = tech['momentum']
                f.write("MOMENTUM INDICATORS:\n")
                f.write("-"*40 + "\n")

                if 'rsi' in mom:
                    current_rsi = [x for x in mom['rsi'] if not np.isnan(x)]
                    if current_rsi:
                        latest_rsi = current_rsi[-1]
                        f.write(f"Current RSI: {latest_rsi:.2f}\n")
                        if latest_rsi > 70:
                            f.write("  Status: OVERBOUGHT\n")
                        elif latest_rsi < 30:
                            f.write("  Status: OVERSOLD\n")
                        else:
                            f.write("  Status: NEUTRAL\n")

                if 'roc' in mom:
                    current_roc = [x for x in mom['roc'] if not np.isnan(x)]
                    if current_roc:
                        latest_roc = current_roc[-1]
                        f.write(f"Current ROC: {latest_roc:.2f}%\n")
                        if latest_roc > 0:
                            f.write("  Momentum: POSITIVE\n")
                        else:
                            f.write("  Momentum: NEGATIVE\n")
                f.write("\n")

            # Trend strength
            if tech.get('trend_strength'):
                trend = tech['trend_strength']
                f.write("TREND STRENGTH:\n")
                f.write("-"*40 + "\n")

                if 'adx' in trend:
                    current_adx = [x for x in trend['adx'] if not np.isnan(x)]
                    if current_adx:
                        latest_adx = current_adx[-1]
                        f.write(f"Current ADX: {latest_adx:.2f}\n")

                        if 'trend_strength' in trend:
                            latest_strength = [x for x in trend['trend_strength'] if x != 'Unknown']
                            if latest_strength:
                                f.write(f"Trend Strength: {latest_strength[-1]}\n")
                f.write("\n")

            # Forecasting
            if tech.get('forecasting'):
                forecast = tech['forecasting']
                f.write("FORECASTING (5-day ahead):\n")
                f.write("-"*40 + "\n")

                if forecast and 'arima' in forecast and forecast['arima']:
                    arima_forecast = forecast['arima']['forecast']
                    f.write(f"ARIMA Forecast: {', '.join([f'{x:.2f}' for x in arima_forecast])}\n")
                    f.write(f"ARIMA Parameters: {forecast['arima']['params']}\n")

                if forecast and 'exp_smoothing' in forecast and forecast['exp_smoothing']:
                    exp_forecast = forecast['exp_smoothing']['forecast']
                    f.write(f"Exp Smoothing: {', '.join([f'{x:.2f}' for x in exp_forecast])}\n")

                f.write("\n")

            # Trading recommendation
            f.write("ANALYSIS SUMMARY:\n")
            f.write("-"*40 + "\n")
            recommendation = self.generate_recommendation(ticker)
            f.write(f"Recommendation: {recommendation}\n")

        print(f"📄 Report saved: {report_path}")

    def generate_recommendation(self, ticker):
        """Generate a trading recommendation based on all indicators"""
        signals = []

        tech = self.technical_results.get(ticker, {})
        reg = self.regression_results.get(ticker, {})

        # Check regression trend
        if reg and 'linear' in reg:
            if reg['linear']['slope'] > 0.5:
                signals.append("BULLISH - Strong upward trend")
            elif reg['linear']['slope'] > 0:
                signals.append("BULLISH - Upward trend")
            elif reg['linear']['slope'] < -0.5:
                signals.append("BEARISH - Strong downward trend")
            elif reg['linear']['slope'] < 0:
                signals.append("BEARISH - Downward trend")

        # Check MA crossovers
        if tech.get('moving_averages'):
            ma = tech['moving_averages']
            if 'crossover_signal' in ma:
                recent = ma['crossover_signal'][-3:]
                if 'bullish' in recent:
                    signals.append("BULLISH - Recent MA crossover")
                elif 'bearish' in recent:
                    signals.append("BEARISH - Recent MA crossover")

        # Check RSI
        if tech.get('momentum'):
            mom = tech['momentum']
            if 'rsi' in mom:
                current_rsi = [x for x in mom['rsi'] if not np.isnan(x)]
                if current_rsi:
                    latest_rsi = current_rsi[-1]
                    if latest_rsi > 70:
                        signals.append("BEARISH - RSI overbought")
                    elif latest_rsi < 30:
                        signals.append("BULLISH - RSI oversold")

        # Aggregate signals
        bullish_count = sum(1 for s in signals if "BULLISH" in s)
        bearish_count = sum(1 for s in signals if "BEARISH" in s)

        if bullish_count > bearish_count:
            return "BUY - Multiple bullish signals"
        elif bearish_count > bullish_count:
            return "SELL - Multiple bearish signals"
        else:
            return "HOLD - Mixed signals"

    def display_key_metrics(self, ticker):
        """Display key metrics in the console"""
        print(f"\n{'='*60}")
        print(f"KEY METRICS FOR {ticker}")
        print(f"{'='*60}")

        data = self.score_history[ticker]
        print(f"📊 Score Statistics:")
        print(f"   Average: {np.mean(data['scores']):.2f}")
        print(f"   Std Dev: {np.std(data['scores']):.2f}")
        print(f"   Range: {np.min(data['scores']):.2f} - {np.max(data['scores']):.2f}")

        reg = self.regression_results.get(ticker)
        if reg and 'linear' in reg:
            print(f"\n📈 Trend Analysis:")
            print(f"   Linear R²: {reg['linear']['r2']:.3f}")
            print(f"   Slope: {reg['linear']['slope']:.3f}")

            if reg['linear']['slope'] > 0:
                print(f"   Direction: ↗️ UPWARD")
            else:
                print(f"   Direction: ↘️ DOWNWARD")

        tech = self.technical_results.get(ticker, {})

        # Latest RSI
        if tech.get('momentum') and 'rsi' in tech['momentum']:
            current_rsi = [x for x in tech['momentum']['rsi'] if not np.isnan(x)]
            if current_rsi:
                print(f"\n🔄 RSI: {current_rsi[-1]:.2f}")

        # Recommendation
        print(f"\n💡 Recommendation: {self.generate_recommendation(ticker)}")
        print("="*60)

    def clean_single_ticker_directory(self, ticker, output_base_dir):
        """Clean the directory for a single ticker before analysis"""
        ticker_dir = Path(output_base_dir) / "technical_plots_single" / ticker

        if ticker_dir.exists():
            print(f"\n🧹 Cleaning old plots for {ticker}...")
            try:
                shutil.rmtree(ticker_dir)
                print(f"   ✓ Deleted old folder: {ticker_dir}")
            except Exception as e:
                print(f"   ✗ Error deleting folder: {str(e)}")


# Main execution
if __name__ == "__main__":
    # ===== CONFIGURATION SECTION =====

    # TICKER TO ANALYZE
    TICKER = "VEEV"  # Change this to any ticker you want to analyze

    # MARKET REGIME SELECTOR
    # 0 = Steady_Growth, 1 = Crisis_Bear
    MARKET_REGIME = 0
    END_DATE = "0828"

    # Define regime-specific settings
    if MARKET_REGIME == 0:
        REGIME_NAME = "Steady_Growth"
        START_DATE = "0611"  # Steady Growth period
        # END_DATE = "0807"
    elif MARKET_REGIME == 1:
        REGIME_NAME = "Crisis_Bear"
        START_DATE = "0719"  # Crisis/Bear period
        # END_DATE = "0807"
    else:
        raise ValueError("MARKET_REGIME must be 0 (Steady_Growth) or 1 (Crisis_Bear)")

    # Directory containing the ranking CSV files (regime-specific)
    CSV_DIRECTORY = f"./ranked_lists/{REGIME_NAME}"

    # Output directory for results (regime-specific)
    OUTPUT_DIRECTORY = f"./results/{REGIME_NAME}"

    # Minimum appearances required for analysis (can be lower for single ticker)
    MIN_APPEARANCES = 3

    # Clean old plots for this ticker
    CLEAN_OLD_PLOTS = True

    print(f"\n{'='*60}")
    print(f"SINGLE TICKER TECHNICAL ANALYSIS")
    print(f"{'='*60}")
    print(f"Ticker: {TICKER}")
    print(f"Market Regime: {REGIME_NAME}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Ranked Lists Directory: {CSV_DIRECTORY}")
    print(f"Output Directory: {OUTPUT_DIRECTORY}")
    print(f"{'='*60}\n")

    # ===== END CONFIGURATION =====

    # Create output directory if it doesn't exist
    Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

    # Initialize analyzer
    analyzer = SingleTickerTechnicalAnalyzer(
        csv_directory=CSV_DIRECTORY,
        start_date=START_DATE,
        end_date=END_DATE
    )

    # Clean old plots if requested
    if CLEAN_OLD_PLOTS:
        analyzer.clean_single_ticker_directory(TICKER, OUTPUT_DIRECTORY)

    # Analyze the single ticker
    success = analyzer.analyze_single_ticker(
        ticker=TICKER,
        output_base_dir=OUTPUT_DIRECTORY,
        min_appearances=MIN_APPEARANCES
    )

    if success:
        print(f"\n✨ Success! All analysis files for {TICKER} have been created.")
        print(f"📁 Location: {OUTPUT_DIRECTORY}/technical_plots_single/{TICKER}/")
    else:
        print(f"\n❌ Analysis failed for {TICKER}. Please check the error messages above.")