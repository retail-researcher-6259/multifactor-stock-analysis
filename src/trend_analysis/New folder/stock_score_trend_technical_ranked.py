"""
Stock Score Trend Analyzer with Technical Analysis Methods
Enhanced version with MA, RSI, ROC, Bollinger Bands, and Statistical Forecasting
Saves individual plots for each ticker in organized folder structure
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
warnings.filterwarnings('ignore')

class StockScoreTrendAnalyzerTechnical:
    def __init__(self, csv_directory="./ranked_lists", start_date="0611", end_date="0621", sigmoid_sensitivity=5):
        self.csv_directory = Path(csv_directory)
        self.start_date = start_date
        self.end_date = end_date
        self.score_history = {}
        self.regression_results = {}
        self.technical_results = {}
        self.sigmoid_sensitivity = sigmoid_sensitivity

    def clean_technical_plots_directory(self, output_base_dir):
        """Clean all existing plots from the technical_plots directory"""
        tech_dir = Path(output_base_dir) / "technical_plots"

        if tech_dir.exists():
            print(f"\n🧹 Cleaning old plots from: {tech_dir}")

            # Count total files to delete
            total_files = 0
            for ticker_dir in tech_dir.iterdir():
                if ticker_dir.is_dir():
                    png_files = list(ticker_dir.glob("*.png"))
                    total_files += len(png_files)

            if total_files > 0:
                print(f"   Found {total_files} existing plot(s) to delete")

                # Delete all ticker subdirectories and their contents
                import shutil
                for ticker_dir in tech_dir.iterdir():
                    if ticker_dir.is_dir():
                        try:
                            shutil.rmtree(ticker_dir)
                            print(f"   ✓ Deleted folder: {ticker_dir.name}")
                        except Exception as e:
                            print(f"   ✗ Error deleting {ticker_dir.name}: {str(e)}")

                print(f"   Cleanup complete!\n")
            else:
                print(f"   Directory is already clean\n")
        else:
            print(f"\n📁 Technical plots directory doesn't exist yet: {tech_dir}\n")

    def load_ranking_files(self):
        """Load all ranking CSV files and assign sequential indices"""
        files_data = []

        # Find all matching CSV files in the specified directory
        csv_files = sorted(self.csv_directory.glob("top_ranked_stocks_*.csv"))

        # Filter files within date range if specified
        filtered_files = []
        for csv_file in csv_files:
            date_str = csv_file.stem.split('_')[-1]  # e.g., "0612"
            if len(date_str) == 4:
                if self.start_date <= date_str <= self.end_date:
                    filtered_files.append((date_str, csv_file))

        # Sort by date string and assign sequential indices
        filtered_files.sort(key=lambda x: x[0])

        for idx, (date_str, csv_file) in enumerate(filtered_files):
            df = pd.read_csv(csv_file)
            files_data.append({
                'date': date_str,
                'sequence_index': idx,
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
                        'ranks': []
                    }

                self.score_history[ticker]['indices'].append(seq_idx)
                self.score_history[ticker]['dates'].append(date)
                self.score_history[ticker]['scores'].append(score)
                self.score_history[ticker]['ranks'].append(row.name + 1)

    def perform_regression_analysis(self, ticker, degree=2):
        """Perform regression analysis for a specific stock"""
        if ticker not in self.score_history:
            return None

        data = self.score_history[ticker]
        if len(data['indices']) < 3:
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

        return results

    def calculate_moving_averages(self, ticker):
        """Calculate various moving averages"""
        if ticker not in self.score_history:
            return None

        scores = np.array(self.score_history[ticker]['scores'])
        indices = self.score_history[ticker]['indices']

        results = {
            'scores': scores,
            'indices': indices
        }

        # Simple Moving Averages
        if len(scores) >= 5:
            results['sma_5'] = self._calculate_sma(scores, 5)
        if len(scores) >= 10:
            results['sma_10'] = self._calculate_sma(scores, 10)
        if len(scores) >= 20:
            results['sma_20'] = self._calculate_sma(scores, 20)

        # Exponential Moving Averages
        if len(scores) >= 5:
            results['ema_5'] = self._calculate_ema(scores, 5)
        if len(scores) >= 10:
            results['ema_10'] = self._calculate_ema(scores, 10)
        if len(scores) >= 20:
            results['ema_20'] = self._calculate_ema(scores, 20)

        # MACD (12, 26, 9)
        if len(scores) >= 26:
            ema_12 = self._calculate_ema(scores, 12)
            ema_26 = self._calculate_ema(scores, 26)
            macd_line = ema_12 - ema_26
            signal_line = self._calculate_ema(macd_line[~np.isnan(macd_line)], 9)
            results['macd'] = {
                'macd_line': macd_line,
                'signal_line': signal_line,
                'histogram': macd_line[-len(signal_line):] - signal_line
            }

        # Crossover signals
        if 'sma_5' in results and 'sma_20' in results:
            results['crossover_signal'] = self._detect_crossovers(
                results['sma_5'], results['sma_20']
            )

        return results

    def _calculate_sma(self, data, period):
        """Calculate Simple Moving Average"""
        sma = np.full(len(data), np.nan)
        for i in range(period - 1, len(data)):
            sma[i] = np.mean(data[i - period + 1:i + 1])
        return sma

    def _calculate_ema(self, data, period):
        """Calculate Exponential Moving Average"""
        alpha = 2 / (period + 1)
        ema = np.full(len(data), np.nan)
        ema[period - 1] = np.mean(data[:period])  # Start with SMA

        for i in range(period, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

        return ema

    def _detect_crossovers(self, fast_ma, slow_ma):
        """Detect MA crossover signals"""
        signals = []
        for i in range(1, len(fast_ma)):
            if np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]):
                signals.append('none')
            elif fast_ma[i-1] <= slow_ma[i-1] and fast_ma[i] > slow_ma[i]:
                signals.append('bullish')
            elif fast_ma[i-1] >= slow_ma[i-1] and fast_ma[i] < slow_ma[i]:
                signals.append('bearish')
            else:
                signals.append('none')
        return ['none'] + signals  # Add first element

    def calculate_momentum_indicators(self, ticker):
        """Calculate RSI and ROC"""
        if ticker not in self.score_history:
            return None

        scores = np.array(self.score_history[ticker]['scores'])

        results = {}

        # RSI (14-period)
        if len(scores) >= 15:
            results['rsi'] = self._calculate_rsi(scores, 14)

        # Rate of Change (10-period)
        if len(scores) >= 11:
            results['roc'] = self._calculate_roc(scores, 10)

        # Stochastic Oscillator
        if len(scores) >= 14:
            results['stochastic'] = self._calculate_stochastic(scores, 14)

        return results

    def _calculate_rsi(self, data, period=14):
        """Calculate Relative Strength Index"""
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.zeros(len(data))
        avg_loss = np.zeros(len(data))

        # First average
        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])

        # Subsequent averages
        for i in range(period + 1, len(data)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period

        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        rsi[:period] = np.nan  # Set initial values to NaN

        return rsi

    def _calculate_roc(self, data, period=10):
        """Calculate Rate of Change"""
        roc = np.full(len(data), np.nan)
        for i in range(period, len(data)):
            roc[i] = ((data[i] - data[i-period]) / data[i-period]) * 100
        return roc

    def _calculate_stochastic(self, data, period=14):
        """Calculate Stochastic Oscillator"""
        stoch = np.full(len(data), np.nan)
        for i in range(period - 1, len(data)):
            window = data[i - period + 1:i + 1]
            high = np.max(window)
            low = np.min(window)
            if high != low:
                stoch[i] = ((data[i] - low) / (high - low)) * 100
            else:
                stoch[i] = 50
        return stoch

    def calculate_bollinger_bands(self, ticker, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        if ticker not in self.score_history:
            return None

        scores = np.array(self.score_history[ticker]['scores'])

        if len(scores) < period:
            return None

        sma = self._calculate_sma(scores, period)

        # Calculate standard deviation
        std = np.full(len(scores), np.nan)
        for i in range(period - 1, len(scores)):
            std[i] = np.std(scores[i - period + 1:i + 1])

        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)

        # Calculate %B (position within bands)
        percent_b = np.where(
            upper_band != lower_band,
            (scores - lower_band) / (upper_band - lower_band),
            0.5
        )

        # Bandwidth
        bandwidth = (upper_band - lower_band) / sma

        return {
            'middle': sma,
            'upper': upper_band,
            'lower': lower_band,
            'percent_b': percent_b,
            'bandwidth': bandwidth
        }

    # def calculate_trend_strength(self, ticker):
    #     """Calculate trend strength indicators"""
    #     if ticker not in self.score_history:
    #         return None
    #
    #     scores = np.array(self.score_history[ticker]['scores'])
    #     indices = np.array(self.score_history[ticker]['indices'])
    #
    #     # ADX-like trend strength (simplified)
    #     if len(scores) < 14:
    #         return None
    #
    #     # Calculate directional movement
    #     dm_plus = np.zeros(len(scores) - 1)
    #     dm_minus = np.zeros(len(scores) - 1)
    #
    #     for i in range(1, len(scores)):
    #         up_move = scores[i] - scores[i-1]
    #         if up_move > 0:
    #             dm_plus[i-1] = up_move
    #         else:
    #             dm_minus[i-1] = abs(up_move)
    #
    #     # Smooth the DM values
    #     period = 14
    #     adx_plus = np.full(len(scores), np.nan)
    #     adx_minus = np.full(len(scores), np.nan)
    #
    #     if len(scores) >= period:
    #         for i in range(period, len(scores)):
    #             adx_plus[i] = np.mean(dm_plus[i-period:i])
    #             adx_minus[i] = np.mean(dm_minus[i-period:i])
    #
    #     # Calculate DX
    #     dx = np.where(
    #         (adx_plus + adx_minus) != 0,
    #         np.abs(adx_plus - adx_minus) / (adx_plus + adx_minus) * 100,
    #         0
    #     )
    #
    #     # Smooth DX to get ADX
    #     adx = self._calculate_sma(dx, period)
    #
    #     return {
    #         'adx': adx,
    #         'di_plus': adx_plus,
    #         'di_minus': adx_minus,
    #         'trend_strength': self._classify_trend_strength(adx)
    #     }

    def calculate_trend_strength(self, ticker):
        """Calculate ADX with proper +DI and -DI"""
        if ticker not in self.score_history:
            return None

        scores = np.array(self.score_history[ticker]['scores'])

        if len(scores) < 14:
            return None

        # Calculate True Range (adapted for scores)
        tr = np.zeros(len(scores) - 1)
        for i in range(1, len(scores)):
            tr[i - 1] = abs(scores[i] - scores[i - 1])

        # Calculate Directional Movement
        dm_plus = np.zeros(len(scores) - 1)
        dm_minus = np.zeros(len(scores) - 1)

        for i in range(1, len(scores)):
            up_move = scores[i] - scores[i - 1]
            if up_move > 0:
                dm_plus[i - 1] = up_move
            else:
                dm_minus[i - 1] = abs(up_move)

        # Smooth using 14-period average
        period = 14
        atr = np.full(len(scores), np.nan)
        smooth_dm_plus = np.full(len(scores), np.nan)
        smooth_dm_minus = np.full(len(scores), np.nan)

        # Initial values
        atr[period] = np.mean(tr[:period])
        smooth_dm_plus[period] = np.mean(dm_plus[:period])
        smooth_dm_minus[period] = np.mean(dm_minus[:period])

        # Smoothing
        for i in range(period + 1, len(scores)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i - 1]) / period
            smooth_dm_plus[i] = (smooth_dm_plus[i - 1] * (period - 1) + dm_plus[i - 1]) / period
            smooth_dm_minus[i] = (smooth_dm_minus[i - 1] * (period - 1) + dm_minus[i - 1]) / period

        # Calculate +DI and -DI
        di_plus = np.where(atr != 0, (smooth_dm_plus / atr) * 100, 0)
        di_minus = np.where(atr != 0, (smooth_dm_minus / atr) * 100, 0)

        # Calculate DX
        di_sum = di_plus + di_minus
        dx = np.where(di_sum != 0, np.abs(di_plus - di_minus) / di_sum * 100, 0)

        # Calculate ADX (smoothed DX)
        adx = self._calculate_sma(dx, period)

        return {
            'adx': adx,
            'di_plus': di_plus,
            'di_minus': di_minus,
            'trend_strength': self._classify_trend_strength(adx),
            'current_adx': adx[-1] if not np.isnan(adx[-1]) else None,
            'current_di_plus': di_plus[-1] if not np.isnan(di_plus[-1]) else None,
            'current_di_minus': di_minus[-1] if not np.isnan(di_minus[-1]) else None
        }

    def _classify_trend_strength(self, adx):
        """Classify trend strength based on ADX values"""
        classifications = []
        for value in adx:
            if np.isnan(value):
                classifications.append('Unknown')
            elif value < 25:
                classifications.append('Weak')
            elif value < 50:
                classifications.append('Strong')
            elif value < 75:
                classifications.append('Very Strong')
            else:
                classifications.append('Extremely Strong')
        return classifications

    def perform_statistical_forecasting(self, ticker, forecast_periods=5):
        """Perform ARIMA and Exponential Smoothing forecasting"""
        if ticker not in self.score_history:
            return None

        scores = np.array(self.score_history[ticker]['scores'])

        if len(scores) < 10:  # Need minimum data for forecasting
            return None

        results = {}

        # ARIMA
        try:
            # Auto-select best ARIMA parameters
            best_aic = np.inf
            best_params = (1, 0, 0)

            for p in range(3):
                for d in range(2):
                    for q in range(3):
                        try:
                            model = ARIMA(scores, order=(p, d, q))
                            fitted = model.fit()
                            if fitted.aic < best_aic:
                                best_aic = fitted.aic
                                best_params = (p, d, q)
                        except:
                            continue

            # Fit best model
            arima_model = ARIMA(scores, order=best_params)
            arima_fitted = arima_model.fit()
            arima_forecast = arima_fitted.forecast(steps=forecast_periods)

            results['arima'] = {
                'params': best_params,
                'aic': best_aic,
                'forecast': arima_forecast,
                'fitted_values': arima_fitted.fittedvalues
            }
        except Exception as e:
            print(f"ARIMA failed for {ticker}: {str(e)}")
            results['arima'] = None

        # Exponential Smoothing
        try:
            # Simple exponential smoothing (no trend or seasonality for short series)
            if len(scores) >= 4:
                model = ExponentialSmoothing(scores, trend=None, seasonal=None)
                fitted = model.fit()
                forecast = fitted.forecast(steps=forecast_periods)

                results['exp_smoothing'] = {
                    'forecast': forecast,
                    'fitted_values': fitted.fittedvalues
                }

                # Try Holt's method (with trend) if we have enough data
                if len(scores) >= 10:
                    model_holt = ExponentialSmoothing(scores, trend='add', seasonal=None)
                    fitted_holt = model_holt.fit()
                    forecast_holt = fitted_holt.forecast(steps=forecast_periods)

                    results['holt'] = {
                        'forecast': forecast_holt,
                        'fitted_values': fitted_holt.fittedvalues
                    }
        except Exception as e:
            print(f"Exponential Smoothing failed for {ticker}: {str(e)}")
            results['exp_smoothing'] = None

        return results

    def create_comprehensive_plots(self, ticker, output_dir):
        """Create all technical analysis plots for a ticker"""
        ticker_dir = Path(output_dir) / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)

        data = self.score_history.get(ticker)
        if not data or len(data['indices']) < 5:
            print(f"Insufficient data for {ticker}")
            return

        indices = data['indices']
        scores = data['scores']
        dates = data['dates']

        # 1. Regression Analysis Plot
        self._plot_regression_analysis(ticker, indices, scores, dates, ticker_dir)

        # 2. Moving Averages Plot
        self._plot_moving_averages(ticker, indices, scores, dates, ticker_dir)

        # 3. Momentum Indicators Plot
        self._plot_momentum_indicators(ticker, indices, scores, dates, ticker_dir)

        # 4. Bollinger Bands Plot
        self._plot_bollinger_bands(ticker, indices, scores, dates, ticker_dir)

        # 5. Trend Strength Plot
        self._plot_trend_strength(ticker, indices, scores, dates, ticker_dir)

        # 6. Statistical Forecasting Plot
        self._plot_statistical_forecasting(ticker, indices, scores, dates, ticker_dir)

        # 7. Combined Overview Plot
        self._plot_combined_overview(ticker, indices, scores, dates, ticker_dir)

        print(f"✓ Created all plots for {ticker}")

    def _plot_regression_analysis(self, ticker, indices, scores, dates, output_dir):
        """Plot regression analysis"""
        regression = self.perform_regression_analysis(ticker)
        if not regression:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        # Actual scores
        ax.scatter(indices, scores, label='Actual Scores', s=50, alpha=0.7)

        # Linear regression
        if 'linear' in regression:
            ax.plot(indices, regression['linear']['predictions'],
                   'r-', label=f"Linear (R²={regression['linear']['r2']:.3f})", linewidth=2)

        # Polynomial regression
        if 'polynomial' in regression:
            ax.plot(indices, regression['polynomial']['predictions'],
                   'g--', label=f"Polynomial (R²={regression['polynomial']['r2']:.3f})", linewidth=2)

        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Score')
        ax.set_title(f'{ticker} - Regression Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add date labels
        ax.set_xticks(indices[::max(1, len(indices)//10)])
        ax.set_xticklabels(dates[::max(1, len(dates)//10)], rotation=45)

        plt.tight_layout()
        plt.savefig(output_dir / f'{ticker}_01_regression.png', dpi=100, bbox_inches='tight')
        plt.close()

    def _plot_moving_averages(self, ticker, indices, scores, dates, output_dir):
        """Plot moving averages and crossovers"""
        ma_results = self.calculate_moving_averages(ticker)
        if not ma_results:
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Top plot: Price and MAs
        ax1.plot(indices, scores, 'b-', label='Score', linewidth=1.5, alpha=0.8)

        if 'sma_5' in ma_results:
            ax1.plot(indices, ma_results['sma_5'], 'r-', label='SMA 5', linewidth=1)
        if 'sma_20' in ma_results:
            ax1.plot(indices, ma_results['sma_20'], 'g-', label='SMA 20', linewidth=1)
        if 'ema_10' in ma_results:
            ax1.plot(indices, ma_results['ema_10'], 'm--', label='EMA 10', linewidth=1)

        # Mark crossovers
        if 'crossover_signal' in ma_results:
            for i, signal in enumerate(ma_results['crossover_signal']):
                if signal == 'bullish':
                    ax1.scatter(indices[i], scores[i], color='green', s=100, marker='^', zorder=5)
                elif signal == 'bearish':
                    ax1.scatter(indices[i], scores[i], color='red', s=100, marker='v', zorder=5)

        ax1.set_ylabel('Score')
        ax1.set_title(f'{ticker} - Moving Averages & Crossovers')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # Bottom plot: MACD
        if 'macd' in ma_results:
            macd_data = ma_results['macd']
            macd_indices = indices[-len(macd_data['signal_line']):]

            ax2.plot(macd_indices, macd_data['signal_line'], 'r-', label='Signal Line')
            ax2.plot(macd_indices, macd_data['macd_line'][-len(macd_data['signal_line']):],
                    'b-', label='MACD Line')
            ax2.bar(macd_indices, macd_data['histogram'], label='Histogram', alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.set_ylabel('MACD')
            ax2.set_title('MACD Indicator')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)

        ax2.set_xlabel('Trading Days')

        # Add date labels
        for ax in [ax1, ax2]:
            ax.set_xticks(indices[::max(1, len(indices)//10)])
            ax.set_xticklabels(dates[::max(1, len(dates)//10)], rotation=45)

        plt.tight_layout()
        plt.savefig(output_dir / f'{ticker}_02_moving_averages.png', dpi=100, bbox_inches='tight')
        plt.close()

    def _plot_momentum_indicators(self, ticker, indices, scores, dates, output_dir):
        """Plot momentum indicators"""
        momentum = self.calculate_momentum_indicators(ticker)
        if not momentum:
            return

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Score plot
        axes[0].plot(indices, scores, 'b-', linewidth=1.5)
        axes[0].set_ylabel('Score')
        axes[0].set_title(f'{ticker} - Momentum Indicators')
        axes[0].grid(True, alpha=0.3)

        # RSI
        if 'rsi' in momentum:
            axes[1].plot(indices, momentum['rsi'], 'purple', linewidth=1.5)
            axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
            axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
            axes[1].set_ylabel('RSI')
            axes[1].set_ylim([0, 100])
            axes[1].legend(loc='best')
            axes[1].grid(True, alpha=0.3)

        # ROC
        if 'roc' in momentum:
            axes[2].plot(indices, momentum['roc'], 'orange', linewidth=1.5)
            axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            axes[2].set_ylabel('ROC (%)')
            axes[2].set_xlabel('Trading Days')
            axes[2].grid(True, alpha=0.3)

        # Add date labels
        for ax in axes:
            ax.set_xticks(indices[::max(1, len(indices)//10)])
            ax.set_xticklabels(dates[::max(1, len(dates)//10)], rotation=45)

        plt.tight_layout()
        plt.savefig(output_dir / f'{ticker}_03_momentum.png', dpi=100, bbox_inches='tight')
        plt.close()

    def _plot_bollinger_bands(self, ticker, indices, scores, dates, output_dir):
        """Plot Bollinger Bands"""
        bb = self.calculate_bollinger_bands(ticker)
        if not bb:
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Bollinger Bands
        ax1.plot(indices, scores, 'b-', label='Score', linewidth=1.5)
        ax1.plot(indices, bb['upper'], 'r--', label='Upper Band', alpha=0.7)
        ax1.plot(indices, bb['middle'], 'g-', label='Middle (SMA)', alpha=0.7)
        ax1.plot(indices, bb['lower'], 'r--', label='Lower Band', alpha=0.7)
        ax1.fill_between(indices, bb['lower'], bb['upper'], alpha=0.1, color='gray')

        ax1.set_ylabel('Score')
        ax1.set_title(f'{ticker} - Bollinger Bands')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # %B and Bandwidth
        ax2.plot(indices, bb['percent_b'], 'purple', label='%B', linewidth=1.5)
        ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.set_ylabel('%B')
        ax2.set_xlabel('Trading Days')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        # Add date labels
        for ax in [ax1, ax2]:
            ax.set_xticks(indices[::max(1, len(indices)//10)])
            ax.set_xticklabels(dates[::max(1, len(dates)//10)], rotation=45)

        plt.tight_layout()
        plt.savefig(output_dir / f'{ticker}_04_bollinger_bands.png', dpi=100, bbox_inches='tight')
        plt.close()

    # def _plot_trend_strength(self, ticker, indices, scores, dates, output_dir):
    #     """Plot trend strength indicators"""
    #     trend = self.calculate_trend_strength(ticker)
    #     if not trend:
    #         return
    #
    #     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    #
    #     # Score plot
    #     ax1.plot(indices, scores, 'b-', linewidth=1.5)
    #     ax1.set_ylabel('Score')
    #     ax1.set_title(f'{ticker} - Trend Strength Analysis')
    #     ax1.grid(True, alpha=0.3)
    #
    #     # ADX
    #     if 'adx' in trend:
    #         ax2.plot(indices, trend['adx'], 'purple', label='ADX', linewidth=1.5)
    #         ax2.axhline(y=25, color='gray', linestyle='--', alpha=0.5, label='Weak/Strong')
    #         ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Strong/Very Strong')
    #         ax2.set_ylabel('ADX')
    #         ax2.set_xlabel('Trading Days')
    #         ax2.legend(loc='best')
    #         ax2.grid(True, alpha=0.3)
    #
    #     # Add date labels
    #     for ax in [ax1, ax2]:
    #         ax.set_xticks(indices[::max(1, len(indices)//10)])
    #         ax.set_xticklabels(dates[::max(1, len(dates)//10)], rotation=45)
    #
    #     plt.tight_layout()
    #     plt.savefig(output_dir / f'{ticker}_05_trend_strength.png', dpi=100, bbox_inches='tight')
    #     plt.close()

    def _plot_trend_strength(self, ticker, indices, scores, dates, output_dir):
        """Plot enhanced trend strength indicators with +DI/-DI"""
        trend = self.calculate_trend_strength(ticker)
        if not trend:
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Score plot
        ax1.plot(indices, scores, 'b-', linewidth=1.5)
        ax1.set_ylabel('Score')
        ax1.set_title(f'{ticker} - Trend Strength Analysis (ADX with +DI/-DI)')
        ax1.grid(True, alpha=0.3)

        # ADX with +DI/-DI
        if 'adx' in trend:
            # Plot all three lines
            ax2.plot(indices, trend['adx'], 'black', label=f"ADX ({trend['current_adx']:.1f})", linewidth=2)
            ax2.plot(indices, trend['di_plus'], 'green', label=f"+DI ({trend['current_di_plus']:.1f})", linewidth=1.5)
            ax2.plot(indices, trend['di_minus'], 'red', label=f"-DI ({trend['current_di_minus']:.1f})", linewidth=1.5)

            # Add threshold lines
            ax2.axhline(y=25, color='gray', linestyle='--', alpha=0.5, label='Trend Threshold')
            ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.5)

            # Fill areas to show which DI is dominant
            ax2.fill_between(indices, trend['di_plus'], trend['di_minus'],
                             where=(np.array(trend['di_plus']) > np.array(trend['di_minus'])),
                             color='green', alpha=0.1, label='Bullish Zone')
            ax2.fill_between(indices, trend['di_plus'], trend['di_minus'],
                             where=(np.array(trend['di_plus']) <= np.array(trend['di_minus'])),
                             color='red', alpha=0.1, label='Bearish Zone')

            ax2.set_ylabel('ADX / DI Values')
            ax2.set_xlabel('Trading Days')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim([0, 100])

            # Add text with current status
            if trend['current_adx'] and trend['current_di_plus'] and trend['current_di_minus']:
                status = "STRONG TREND" if trend['current_adx'] > 25 else "WEAK TREND"
                direction = "BULLISH" if trend['current_di_plus'] > trend['current_di_minus'] else "BEARISH"
                ax2.text(0.02, 0.98, f"Status: {status} - {direction}",
                         transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Add date labels
        for ax in [ax1, ax2]:
            ax.set_xticks(indices[::max(1, len(indices) // 10)])
            ax.set_xticklabels(dates[::max(1, len(dates) // 10)], rotation=45)

        plt.tight_layout()
        plt.savefig(output_dir / f'{ticker}_05_trend_strength.png', dpi=100, bbox_inches='tight')
        plt.close()

    def _plot_statistical_forecasting(self, ticker, indices, scores, dates, output_dir):
        """Plot statistical forecasting results"""
        forecast = self.perform_statistical_forecasting(ticker)
        if not forecast:
            return

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Extended indices for forecast
        last_index = indices[-1]
        forecast_indices = list(range(last_index + 1, last_index + 6))
        all_indices = list(indices) + forecast_indices

        # ARIMA plot
        if forecast and 'arima' in forecast and forecast['arima']:
            ax = axes[0]
            ax.plot(indices, scores, 'b-', label='Actual Scores', linewidth=1.5)
            ax.plot(indices, forecast['arima']['fitted_values'], 'g--',
                   label=f'ARIMA{forecast["arima"]["params"]} Fitted', alpha=0.7)
            ax.plot(forecast_indices, forecast['arima']['forecast'], 'r-',
                   label='ARIMA Forecast', linewidth=2, marker='o')

            # Shade forecast area
            ax.axvspan(indices[-1], forecast_indices[-1], alpha=0.1, color='gray')

            ax.set_ylabel('Score')
            ax.set_title(f'{ticker} - ARIMA Forecasting')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

        # Exponential Smoothing plot
        if forecast and 'exp_smoothing' in forecast and forecast['exp_smoothing']:
            ax = axes[1]
            ax.plot(indices, scores, 'b-', label='Actual Scores', linewidth=1.5)
            ax.plot(indices, forecast['exp_smoothing']['fitted_values'], 'g--',
                   label='Exp Smoothing Fitted', alpha=0.7)
            ax.plot(forecast_indices, forecast['exp_smoothing']['forecast'], 'orange',
                   label='Exp Smoothing Forecast', linewidth=2, marker='o')

            # Add Holt's method if available
            if 'holt' in forecast and forecast['holt']:
                ax.plot(forecast_indices, forecast['holt']['forecast'], 'purple',
                       label="Holt's Forecast", linewidth=2, marker='s', markersize=5)

            # Shade forecast area
            ax.axvspan(indices[-1], forecast_indices[-1], alpha=0.1, color='gray')

            ax.set_ylabel('Score')
            ax.set_xlabel('Trading Days')
            ax.set_title('Exponential Smoothing Forecasting')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

        # Add date labels (extend for forecast)
        forecast_dates = [f"F+{i}" for i in range(1, 6)]
        all_dates = list(dates) + forecast_dates

        for ax in axes:
            tick_positions = all_indices[::max(1, len(all_indices)//10)]
            tick_labels = [all_dates[i] if i < len(all_dates) else ''
                          for i in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45)

        plt.tight_layout()
        plt.savefig(output_dir / f'{ticker}_06_forecasting.png', dpi=100, bbox_inches='tight')
        plt.close()

    def _plot_combined_overview(self, ticker, indices, scores, dates, output_dir):
        """Create a combined overview plot with key indicators"""
        fig = plt.figure(figsize=(15, 12))

        # Create grid
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

        # 1. Main price and MA plot
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(indices, scores, 'b-', label='Score', linewidth=2)

        ma_results = self.calculate_moving_averages(ticker)
        if ma_results and 'sma_20' in ma_results:
            ax1.plot(indices, ma_results['sma_20'], 'g-', label='SMA 20', alpha=0.7)
        if ma_results and 'ema_10' in ma_results:
            ax1.plot(indices, ma_results['ema_10'], 'r--', label='EMA 10', alpha=0.7)

        ax1.set_ylabel('Score')
        ax1.set_title(f'{ticker} - Technical Analysis Overview')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # 2. Bollinger Bands
        ax2 = fig.add_subplot(gs[1, :])
        bb = self.calculate_bollinger_bands(ticker)
        if bb:
            ax2.plot(indices, scores, 'b-', linewidth=1)
            ax2.plot(indices, bb['upper'], 'r--', alpha=0.5)
            ax2.plot(indices, bb['lower'], 'r--', alpha=0.5)
            ax2.fill_between(indices, bb['lower'], bb['upper'], alpha=0.1, color='gray')
        ax2.set_ylabel('Score')
        ax2.set_title('Bollinger Bands')
        ax2.grid(True, alpha=0.3)

        # 3. RSI
        ax3 = fig.add_subplot(gs[2, 0])
        momentum = self.calculate_momentum_indicators(ticker)
        if momentum and 'rsi' in momentum:
            ax3.plot(indices, momentum['rsi'], 'purple', linewidth=1.5)
            ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax3.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax3.set_ylim([0, 100])
        ax3.set_ylabel('RSI')
        ax3.set_title('RSI (14)')
        ax3.grid(True, alpha=0.3)

        # 4. ROC
        ax4 = fig.add_subplot(gs[2, 1])
        if momentum and 'roc' in momentum:
            ax4.plot(indices, momentum['roc'], 'orange', linewidth=1.5)
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.set_ylabel('ROC (%)')
        ax4.set_title('Rate of Change (10)')
        ax4.grid(True, alpha=0.3)

        # 5. Trend Strength
        ax5 = fig.add_subplot(gs[3, 0])
        trend = self.calculate_trend_strength(ticker)
        if trend and 'adx' in trend:
            ax5.plot(indices, trend['adx'], 'green', linewidth=1.5)
            ax5.axhline(y=25, color='gray', linestyle='--', alpha=0.5)
            ax5.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax5.set_ylabel('ADX')
        ax5.set_xlabel('Trading Days')
        ax5.set_title('Trend Strength (ADX)')
        ax5.grid(True, alpha=0.3)

        # 6. Forecast
        ax6 = fig.add_subplot(gs[3, 1])
        forecast = self.perform_statistical_forecasting(ticker)
        if forecast and 'arima' in forecast and forecast['arima']:
            last_index = indices[-1]
            forecast_indices = list(range(last_index + 1, last_index + 6))

            ax6.plot(indices[-10:], scores[-10:], 'b-', label='Recent', linewidth=1.5)
            ax6.plot(forecast_indices, forecast['arima']['forecast'], 'r-',
                    label='ARIMA Forecast', linewidth=2, marker='o')
            ax6.axvline(x=last_index, color='gray', linestyle='--', alpha=0.5)
            ax6.legend(loc='best')
        ax6.set_ylabel('Score')
        ax6.set_xlabel('Trading Days')
        ax6.set_title('5-Day Forecast')
        ax6.grid(True, alpha=0.3)

        # Add date labels to all subplots
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            if ax in [ax1, ax2]:  # Full range for top plots
                tick_indices = indices[::max(1, len(indices)//8)]
                tick_dates = [dates[i] for i in range(0, len(dates), max(1, len(dates)//8))]
            else:  # Reduced labels for smaller plots
                tick_indices = indices[::max(1, len(indices)//5)]
                tick_dates = [dates[i] for i in range(0, len(dates), max(1, len(dates)//5))]

            ax.set_xticks(tick_indices)
            ax.set_xticklabels(tick_dates, rotation=45, fontsize=8)

        plt.suptitle(f'{ticker} - Comprehensive Technical Analysis', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / f'{ticker}_00_overview.png', dpi=100, bbox_inches='tight')
        plt.close()

    def analyze_all_stocks(self, output_base_dir, min_appearances=5, ticker_list_file=None, clean_old_plots=True):
        """Analyze all stocks and create technical plots

        Parameters:
        -----------
        output_base_dir : str
            Base directory for output
        min_appearances : int
            Minimum number of appearances required for analysis
        ticker_list_file : str, optional
            Path to CSV file containing specific tickers to analyze
            If None, analyzes all tickers in the data
        """
        files_data = self.load_ranking_files()
        self.build_score_history(files_data)

        # Clean old plots if requested
        if clean_old_plots:
            self.clean_technical_plots_directory(output_base_dir)

        # Determine which tickers to analyze
        if ticker_list_file:
            # Load tickers from specified CSV file
            print(f"\nLoading ticker list from: {ticker_list_file}")
            try:
                ticker_df = pd.read_csv(ticker_list_file)

                # Find the ticker column (handle different possible column names)
                ticker_column = None
                for col in ['Ticker', 'ticker', 'Symbol', 'symbol']:
                    if col in ticker_df.columns:
                        ticker_column = col
                        break

                if ticker_column is None:
                    raise ValueError("Could not find ticker column in the CSV file")

                tickers_to_analyze = ticker_df[ticker_column].unique().tolist()
                print(f"Found {len(tickers_to_analyze)} tickers in the list")

                # Filter to only include tickers that exist in our score history
                tickers_to_analyze = [t for t in tickers_to_analyze if t in self.score_history]
                print(f"Found {len(tickers_to_analyze)} tickers with available data")

            except Exception as e:
                print(f"Error loading ticker list: {e}")
                print("Falling back to analyzing all tickers")
                tickers_to_analyze = list(self.score_history.keys())
        else:
            # Analyze all tickers
            tickers_to_analyze = list(self.score_history.keys())
            print(f"\nAnalyzing all {len(tickers_to_analyze)} tickers")

        # Create technical_plots directory
        tech_dir = Path(output_base_dir) / "technical_plots"
        tech_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nCreating technical analysis plots in: {tech_dir}")
        print("="*60)

        analyzed_count = 0
        skipped_count = 0

        for ticker in tickers_to_analyze:
            if ticker not in self.score_history:
                print(f"Skipping {ticker} - not found in ranking data")
                skipped_count += 1
                continue

            if len(self.score_history[ticker]['indices']) >= min_appearances:
                print(f"\nAnalyzing {ticker}...")
                try:
                    # Perform regression analysis
                    self.regression_results[ticker] = self.perform_regression_analysis(ticker)

                    # Perform technical analysis
                    self.technical_results[ticker] = {
                        'moving_averages': self.calculate_moving_averages(ticker),
                        'momentum': self.calculate_momentum_indicators(ticker),
                        'bollinger_bands': self.calculate_bollinger_bands(ticker),
                        'trend_strength': self.calculate_trend_strength(ticker),
                        'forecasting': self.perform_statistical_forecasting(ticker)
                    }

                    # Create plots
                    self.create_comprehensive_plots(ticker, tech_dir)
                    analyzed_count += 1

                except Exception as e:
                    print(f"  Error analyzing {ticker}: {str(e)}")
                    skipped_count += 1
            else:
                print(f"Skipping {ticker} - insufficient data ({len(self.score_history[ticker]['indices'])} points)")
                skipped_count += 1

        print("\n" + "="*60)
        print(f"Analysis Complete!")
        print(f"Analyzed: {analyzed_count} tickers")
        print(f"Skipped: {skipped_count} tickers")
        print(f"Plots saved to: {tech_dir}")

        # Generate technical rankings
        if analyzed_count > 0:
            rankings_df = self.rank_all_tickers_by_score(tech_dir)
            print(f"\n📊 Technical rankings saved to: {tech_dir}/technical_rankings.csv")

        return analyzed_count, skipped_count

    def generate_analysis_report(self, output_dir):
        """Generate a summary report of technical indicators"""
        report_path = Path(output_dir) / "technical_analysis_report.txt"

        with open(report_path, 'w') as f:
            f.write("TECHNICAL ANALYSIS SUMMARY REPORT\n")
            f.write("="*60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Date Range: {self.start_date} to {self.end_date}\n")
            f.write(f"Total Tickers Analyzed: {len(self.technical_results)}\n\n")

            # Identify key signals
            bullish_tickers = []
            bearish_tickers = []
            strong_trend_tickers = []

            for ticker, tech_data in self.technical_results.items():
                if not tech_data:
                    continue

                # Check for bullish/bearish signals
                if tech_data.get('moving_averages'):
                    ma = tech_data['moving_averages']
                    if 'crossover_signal' in ma:
                        recent_signals = ma['crossover_signal'][-5:]
                        if 'bullish' in recent_signals:
                            bullish_tickers.append(ticker)
                        elif 'bearish' in recent_signals:
                            bearish_tickers.append(ticker)

                # Check trend strength
                if tech_data.get('trend_strength'):
                    trend = tech_data['trend_strength']
                    if 'adx' in trend:
                        recent_adx = [x for x in trend['adx'][-5:] if not np.isnan(x)]
                        if recent_adx and np.mean(recent_adx) > 40:
                            strong_trend_tickers.append(ticker)

            f.write("KEY SIGNALS:\n")
            f.write("-"*40 + "\n")
            f.write(f"Recent Bullish Crossovers: {', '.join(bullish_tickers[:10])}\n")
            f.write(f"Recent Bearish Crossovers: {', '.join(bearish_tickers[:10])}\n")
            f.write(f"Strong Trending Stocks: {', '.join(strong_trend_tickers[:10])}\n")

        print(f"\nReport saved to: {report_path}")

    def calculate_technical_score(self, ticker):
        """Calculate a comprehensive technical score for a ticker"""

        if ticker not in self.score_history:
            return None

        score_details = {}
        total_score = 0
        max_possible = 0

        # 1. REGRESSION ANALYSIS (0-2 points)
        if ticker in self.regression_results and self.regression_results[ticker]:
            reg = self.regression_results[ticker]
            max_possible += 2

            # Linear trend (0-1 point)
            if 'linear' in reg:
                if reg['linear']['slope'] > 0.1:
                    score_details['trend'] = "Bullish (+1)"
                    total_score += 1
                elif reg['linear']['slope'] < -0.1:
                    score_details['trend'] = "Bearish (0)"
                else:
                    score_details['trend'] = "Neutral (+0.5)"
                    total_score += 0.5

            # R² quality (0-1 point)
            if 'linear' in reg and reg['linear']['r2'] > 0.6:
                score_details['trend_quality'] = f"Strong R²={reg['linear']['r2']:.2f} (+1)"
                total_score += 1
            else:
                score_details['trend_quality'] = "Weak trend (0)"

        # 2. MOVING AVERAGES (0-2 points)
        tech = self.technical_results.get(ticker, {})
        if tech.get('moving_averages'):
            ma = tech['moving_averages']
            max_possible += 2

            # SMA crossover (0-1 point)
            if 'sma_5' in ma and 'sma_20' in ma:
                current_sma5 = [x for x in ma['sma_5'] if not np.isnan(x)]
                current_sma20 = [x for x in ma['sma_20'] if not np.isnan(x)]
                if current_sma5 and current_sma20:
                    if current_sma5[-1] > current_sma20[-1]:
                        score_details['sma_cross'] = "SMA5 > SMA20 (+1)"
                        total_score += 1
                    else:
                        score_details['sma_cross'] = "SMA5 < SMA20 (0)"

            # Recent crossover signal (0-1 point)
            if 'crossover_signal' in ma:
                recent = ma['crossover_signal'][-5:]
                if 'bullish' in recent:
                    score_details['recent_cross'] = "Recent bullish cross (+1)"
                    total_score += 1
                elif 'bearish' in recent:
                    score_details['recent_cross'] = "Recent bearish cross (0)"
                else:
                    score_details['recent_cross'] = "No recent cross (+0.5)"
                    total_score += 0.5

        # 3. MACD (0-2 points)
        if tech.get('moving_averages') and 'macd' in tech['moving_averages']:
            macd_data = tech['moving_averages']['macd']
            max_possible += 2

            # MACD vs Signal (0-1 point)
            if len(macd_data['macd_line']) > 0 and len(macd_data['signal_line']) > 0:
                current_macd = macd_data['macd_line'][-1]
                current_signal = macd_data['signal_line'][-1]
                if not np.isnan(current_macd) and not np.isnan(current_signal):
                    if current_macd > current_signal:
                        score_details['macd'] = "MACD > Signal (+1)"
                        total_score += 1
                    else:
                        score_details['macd'] = "MACD < Signal (0)"

            # MACD histogram trend (0-1 point)
            if len(macd_data['histogram']) >= 3:
                recent_hist = macd_data['histogram'][-3:]
                if all(h > 0 for h in recent_hist if not np.isnan(h)):
                    score_details['macd_hist'] = "Positive histogram (+1)"
                    total_score += 1
                elif all(h < 0 for h in recent_hist if not np.isnan(h)):
                    score_details['macd_hist'] = "Negative histogram (0)"
                else:
                    score_details['macd_hist'] = "Mixed histogram (+0.5)"
                    total_score += 0.5

        # 4. RSI (0-2 points)
        if tech.get('momentum') and 'rsi' in tech['momentum']:
            rsi_values = [x for x in tech['momentum']['rsi'] if not np.isnan(x)]
            if rsi_values:
                current_rsi = rsi_values[-1]
                max_possible += 2

                # RSI level (0-1 point)
                if 30 <= current_rsi <= 70:
                    score_details['rsi_level'] = f"RSI neutral {current_rsi:.1f} (+0.5)"
                    total_score += 0.5
                elif current_rsi < 30:
                    score_details['rsi_level'] = f"RSI oversold {current_rsi:.1f} (+1)"
                    total_score += 1
                else:
                    score_details['rsi_level'] = f"RSI overbought {current_rsi:.1f} (0)"

                # RSI trend (0-1 point)
                if len(rsi_values) >= 5:
                    rsi_trend = rsi_values[-1] - rsi_values[-5]
                    if rsi_trend > 5:
                        score_details['rsi_trend'] = "RSI rising (+1)"
                        total_score += 1
                    elif rsi_trend < -5:
                        score_details['rsi_trend'] = "RSI falling (0)"
                    else:
                        score_details['rsi_trend'] = "RSI stable (+0.5)"
                        total_score += 0.5

        # 5. ROC (0-1 point)
        if tech.get('momentum') and 'roc' in tech['momentum']:
            roc_values = [x for x in tech['momentum']['roc'] if not np.isnan(x)]
            if roc_values:
                current_roc = roc_values[-1]
                max_possible += 1

                if current_roc > 2:
                    score_details['roc'] = f"ROC positive {current_roc:.1f}% (+1)"
                    total_score += 1
                elif current_roc < -2:
                    score_details['roc'] = f"ROC negative {current_roc:.1f}% (0)"
                else:
                    score_details['roc'] = f"ROC neutral {current_roc:.1f}% (+0.5)"
                    total_score += 0.5

        # 6. BOLLINGER BANDS & %B (0-2 points)
        if tech.get('bollinger_bands'):
            bb = tech['bollinger_bands']
            max_possible += 2

            # %B position (0-1 point)
            if 'percent_b' in bb:
                current_b = [x for x in bb['percent_b'] if not np.isnan(x)]
                if current_b:
                    b_value = current_b[-1]
                    if 0.2 <= b_value <= 0.8:
                        score_details['bb_position'] = f"%B={b_value:.2f} neutral (+0.5)"
                        total_score += 0.5
                    elif b_value < 0.2:
                        score_details['bb_position'] = f"%B={b_value:.2f} oversold (+1)"
                        total_score += 1
                    elif b_value > 0.8:
                        score_details['bb_position'] = f"%B={b_value:.2f} overbought (0)"
                    else:
                        score_details['bb_position'] = f"%B={b_value:.2f} outside bands (+0.3)"
                        total_score += 0.3

            # Bandwidth trend (0-1 point)
            if 'bandwidth' in bb:
                bw = [x for x in bb['bandwidth'] if not np.isnan(x)]
                if len(bw) >= 5:
                    bw_trend = bw[-1] - bw[-5]
                    if bw_trend > 0:
                        score_details['bb_width'] = "Volatility expanding (+0.5)"
                        total_score += 0.5
                    else:
                        score_details['bb_width'] = "Volatility contracting (+0.5)"
                        total_score += 0.5

        # 7. ADX & TREND STRENGTH (0-2 points)
        if tech.get('trend_strength'):
            trend = tech['trend_strength']
            max_possible += 2

            # ADX strength (0-1 point)
            if 'current_adx' in trend and trend['current_adx']:
                adx_val = trend['current_adx']
                if adx_val > 25:
                    score_details['adx_strength'] = f"Strong trend ADX={adx_val:.1f} (+1)"
                    total_score += 1
                else:
                    score_details['adx_strength'] = f"Weak trend ADX={adx_val:.1f} (+0)"

            # DI direction (0-1 point)
            if 'current_di_plus' in trend and 'current_di_minus' in trend:
                if trend['current_di_plus'] and trend['current_di_minus']:
                    if trend['current_di_plus'] > trend['current_di_minus']:
                        score_details['di_direction'] = "+DI > -DI Bullish (+1)"
                        total_score += 1
                    else:
                        score_details['di_direction'] = "-DI > +DI Bearish (0)"

        # 8. FORECASTING (0-2 points)
        if tech.get('forecasting'):
            forecast = tech['forecasting']
            max_possible += 2
            scores = self.score_history[ticker]['scores']
            current_score = scores[-1]

            # ARIMA forecast (0-1 point)
            if forecast and 'arima' in forecast and forecast['arima']:
                arima_avg = np.mean(forecast['arima']['forecast'])
                if arima_avg > current_score * 1.01:
                    score_details['arima'] = f"ARIMA bullish (+1)"
                    total_score += 1
                elif arima_avg < current_score * 0.99:
                    score_details['arima'] = f"ARIMA bearish (0)"
                else:
                    score_details['arima'] = f"ARIMA neutral (+0.5)"
                    total_score += 0.5

            # Exponential smoothing (0-1 point)
            if forecast and 'exp_smoothing' in forecast and forecast['exp_smoothing']:
                exp_avg = np.mean(forecast['exp_smoothing']['forecast'])
                if exp_avg > current_score * 1.01:
                    score_details['exp_smooth'] = f"Exp smooth bullish (+1)"
                    total_score += 1
                elif exp_avg < current_score * 0.99:
                    score_details['exp_smooth'] = f"Exp smooth bearish (0)"
                else:
                    score_details['exp_smooth'] = f"Exp smooth neutral (+0.5)"
                    total_score += 0.5

        # Calculate percentage score
        percentage_score = (total_score / max_possible * 100) if max_possible > 0 else 0

        return {
            'ticker': ticker,
            'total_score': total_score,
            'max_possible': max_possible,
            'percentage': percentage_score,
            'details': score_details,
            'recommendation': self._get_score_recommendation(percentage_score)
        }

    def _get_score_recommendation(self, percentage):
        """Get recommendation based on percentage score"""
        if percentage >= 75:
            return "STRONG BUY"
        elif percentage >= 60:
            return "BUY"
        elif percentage >= 40:
            return "HOLD"
        elif percentage >= 25:
            return "SELL"
        else:
            return "STRONG SELL"

    # def rank_all_tickers_by_score(self, output_dir):
    #     """Rank all analyzed tickers by their technical scores"""
    #
    #     rankings = []
    #
    #     for ticker in self.technical_results:
    #         score_data = self.calculate_technical_score(ticker)
    #         if score_data:
    #             rankings.append({
    #                 'Ticker': ticker,
    #                 'Score': score_data['total_score'],
    #                 'Max': score_data['max_possible'],
    #                 'Percentage': score_data['percentage'],
    #                 'Recommendation': score_data['recommendation']
    #             })
    #
    #     # Sort by percentage score
    #     rankings_df = pd.DataFrame(rankings)
    #     rankings_df = rankings_df.sort_values('Percentage', ascending=False)
    #
    #     # Save to CSV
    #     output_path = Path(output_dir) / "technical_rankings.csv"
    #     rankings_df.to_csv(output_path, index=False)
    #
    #     # Display top and bottom performers
    #     print("\n" + "=" * 60)
    #     print("TECHNICAL ANALYSIS RANKINGS")
    #     print("=" * 60)
    #
    #     print("\n🏆 TOP 10 STOCKS:")
    #     print("-" * 40)
    #     print(rankings_df.head(10).to_string(index=False))
    #
    #     print("\n📉 BOTTOM 10 STOCKS:")
    #     print("-" * 40)
    #     print(rankings_df.tail(10).to_string(index=False))
    #
    #     # Summary by recommendation
    #     print("\n📊 BREAKDOWN BY RECOMMENDATION:")
    #     print("-" * 40)
    #     rec_counts = rankings_df['Recommendation'].value_counts()
    #     for rec, count in rec_counts.items():
    #         print(f"{rec:<15} {count:>3} stocks")
    #
    #     return rankings_df

    def rank_all_tickers_by_score(self, output_dir):
        """Rank all analyzed tickers by their technical scores with detailed breakdown"""

        rankings = []

        # Define the indicator columns in order
        indicator_columns = [
            'Trend Direction',
            'Trend Quality',
            'SMA Position',
            'Recent Cross',
            'MACD Signal',
            'MACD Histogram',
            'RSI Level',
            'RSI Trend',
            'ROC',
            '%B Position',
            'BB Width',
            'ADX Strength',
            'DI Direction',
            'ARIMA',
            'Exp Smooth'
        ]

        for ticker in self.technical_results:
            score_data = self.calculate_technical_score(ticker)
            if score_data:
                # Start with basic info - KEEP SCORE AS FLOAT
                row_data = {
                    'Ticker': ticker,
                    'Score': score_data['total_score']  # Keep as float, not string
                }

                # Initialize all indicators to 0
                for col in indicator_columns:
                    row_data[col] = 0

                # Map the detailed scores to columns
                details = score_data['details']

                # Map each detail to the appropriate column
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
                    'exp_smooth': 'Exp Smooth'
                }

                # Extract scores from the detail strings
                for detail_key, detail_value in details.items():
                    if detail_key in detail_to_column:
                        column = detail_to_column[detail_key]

                        # Extract the score from the detail string
                        if '(+1)' in detail_value:
                            row_data[column] = 1
                        elif '(+0.5)' in detail_value:
                            row_data[column] = 0.5
                        elif '(+0.3)' in detail_value:
                            row_data[column] = 0.3
                        elif '(0)' in detail_value or '(+0)' in detail_value:
                            row_data[column] = 0
                        else:
                            # Default to checking if it's bullish/bearish
                            if 'bullish' in detail_value.lower() or 'positive' in detail_value.lower():
                                row_data[column] = 1
                            elif 'neutral' in detail_value.lower():
                                row_data[column] = 0.5
                            else:
                                row_data[column] = 0

                # Add recommendation
                row_data['Recommendation'] = score_data['recommendation']

                rankings.append(row_data)

        # Create DataFrame with columns in specific order
        columns_order = ['Ticker', 'Score'] + indicator_columns + ['Recommendation']
        rankings_df = pd.DataFrame(rankings, columns=columns_order)

        # IMPORTANT: Convert Score to float explicitly and sort
        rankings_df['Score'] = pd.to_numeric(rankings_df['Score'], errors='coerce')

        # Sort by Score (descending) - now properly sorting as numbers
        rankings_df = rankings_df.sort_values('Score', ascending=False)

        # Reset index after sorting
        rankings_df = rankings_df.reset_index(drop=True)

        # Round the Score to 1 decimal place for display
        rankings_df['Score'] = rankings_df['Score'].round(1)

        # Save to CSV
        output_path = Path(output_dir) / "technical_rankings.csv"
        rankings_df.to_csv(output_path, index=False)

        # Verify sorting
        print(f"\n✅ Rankings saved to: {output_path}")
        print(f"   Top scorer: {rankings_df.iloc[0]['Ticker']} with score {rankings_df.iloc[0]['Score']}")
        print(f"   Bottom scorer: {rankings_df.iloc[-1]['Ticker']} with score {rankings_df.iloc[-1]['Score']}")

        # Display top and bottom performers with key indicators
        print("\n" + "=" * 80)
        print("TECHNICAL ANALYSIS RANKINGS (Detailed Breakdown)")
        print("=" * 80)

        print("\n🏆 TOP 10 STOCKS:")
        print("-" * 60)
        # For display, show only key columns
        display_cols = ['Ticker', 'Score', 'Trend Direction', 'SMA Position',
                        'MACD Signal', 'RSI Level', 'ADX Strength', 'Recommendation']
        print(rankings_df[display_cols].head(10).to_string(index=False))

        print("\n📉 BOTTOM 10 STOCKS:")
        print("-" * 60)
        print(rankings_df[display_cols].tail(10).to_string(index=False))

        # Summary statistics
        print("\n📊 INDICATOR STATISTICS:")
        print("-" * 60)
        for col in indicator_columns:
            bullish_count = (rankings_df[col] > 0.5).sum()
            neutral_count = ((rankings_df[col] > 0) & (rankings_df[col] <= 0.5)).sum()
            bearish_count = (rankings_df[col] == 0).sum()
            print(f"{col:<20} Bullish: {bullish_count:>3}  Neutral: {neutral_count:>3}  Bearish: {bearish_count:>3}")

        # Summary by recommendation
        print("\n📈 BREAKDOWN BY RECOMMENDATION:")
        print("-" * 60)
        rec_counts = rankings_df['Recommendation'].value_counts()
        for rec, count in rec_counts.items():
            print(f"{rec:<15} {count:>3} stocks")

        # Find stocks with most bullish indicators
        print("\n🔥 MOST BULLISH SIGNALS (All 1s):")
        print("-" * 60)
        for col in indicator_columns:
            perfect_stocks = rankings_df[rankings_df[col] == 1]['Ticker'].tolist()[:5]
            if perfect_stocks:
                print(f"{col:<20} {', '.join(perfect_stocks)}")

        # Double-check sorting
        if not rankings_df['Score'].is_monotonic_decreasing:
            print("\n⚠️ WARNING: Scores are not properly sorted!")
            print("First 5 scores:", rankings_df['Score'].head().tolist())

        return rankings_df


# Main execution
if __name__ == "__main__":
    # ===== CONFIGURATION SECTION =====

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

    CLEAN_OLD_PLOTS = True

    # Directory containing the ranking CSV files (regime-specific)
    CSV_DIRECTORY = f"./ranked_lists/{REGIME_NAME}"

    # Output directory for results (regime-specific)
    OUTPUT_DIRECTORY = f"./results/{REGIME_NAME}"

    # Ticker list file (optional - set to None to analyze all tickers)
    # This should be the path to your common_top150_tickers CSV file
    TICKER_LIST_FILE = f"./results/{REGIME_NAME}/common_top150_tickers_{END_DATE}.csv"
    # Set to None if you want to analyze all tickers:
    # TICKER_LIST_FILE = None

    # Minimum appearances required for analysis
    MIN_APPEARANCES = 5

    print(f"\n{'='*60}")
    print(f"TECHNICAL ANALYSIS - MARKET REGIME: {REGIME_NAME}")
    print(f"{'='*60}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Ranked Lists Directory: {CSV_DIRECTORY}")
    print(f"Output Directory: {OUTPUT_DIRECTORY}")
    if TICKER_LIST_FILE:
        print(f"Ticker List File: {TICKER_LIST_FILE}")
    else:
        print(f"Analyzing: ALL TICKERS")
    print(f"Minimum Appearances: {MIN_APPEARANCES}")
    print(f"{'='*60}\n")

    # ===== END CONFIGURATION =====

    # Create output directory if it doesn't exist
    Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

    # Initialize analyzer
    analyzer = StockScoreTrendAnalyzerTechnical(
        csv_directory=CSV_DIRECTORY,
        start_date=START_DATE,
        end_date=END_DATE
    )

    # Run analysis
    analyzed, skipped = analyzer.analyze_all_stocks(
        output_base_dir=OUTPUT_DIRECTORY,
        min_appearances=MIN_APPEARANCES,
        ticker_list_file=TICKER_LIST_FILE,
        clean_old_plots=CLEAN_OLD_PLOTS
    )

    # Generate summary report
    analyzer.generate_analysis_report(OUTPUT_DIRECTORY)

    print(f"\n{'='*60}")
    print(f"Technical analysis complete for {REGIME_NAME} regime!")
    print(f"All plots saved to: {OUTPUT_DIRECTORY}/technical_plots/")
    print(f"{'='*60}")