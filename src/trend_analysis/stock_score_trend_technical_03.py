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
from prophet import Prophet
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
            print(f"\n Cleaning old plots from: {tech_dir}")

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
                            print(f"    Deleted folder: {ticker_dir.name}")
                        except Exception as e:
                            print(f"    Error deleting {ticker_dir.name}: {str(e)}")

                print(f"   Cleanup complete!\n")
            else:
                print(f"   Directory is already clean\n")
        else:
            print(f"\n Technical plots directory doesn't exist yet: {tech_dir}\n")

    def load_ranking_files(self):
        """Load all ranking CSV files in the date range"""
        files_data = []

        # # Parse date strings
        # start_month = int(self.start_date[:2])
        # start_day = int(self.start_date[2:])
        # end_month = int(self.end_date[:2])
        # end_day = int(self.end_date[2:])
        #
        # # Get all CSV files
        # csv_files = list(self.csv_directory.glob("top_ranked_stocks_*.csv"))
        #
        # for file in csv_files:
        #     # Extract date from filename
        #     file_name = file.stem
        #     date_str = file_name.split('_')[-1]
        #
        #     if len(date_str) == 4:
        #         file_month = int(date_str[:2])
        #         file_day = int(date_str[2:])
        #
        #         # Check if file is within date range
        #         file_date = file_month * 100 + file_day
        #         start_date_num = start_month * 100 + start_day
        #         end_date_num = end_month * 100 + end_day
        #
        #         if start_date_num <= file_date <= end_date_num:
        #             files_data.append({
        #                 'file': file,
        #                 'date': date_str,
        #                 'month': file_month,
        #                 'day': file_day
        #             })

        # Parse date strings (YYYYMMDD format)
        start_year = int(self.start_date[:4])
        start_month = int(self.start_date[4:6])
        start_day = int(self.start_date[6:])
        end_year = int(self.end_date[:4])
        end_month = int(self.end_date[4:6])
        end_day = int(self.end_date[6:])

        # Get all CSV files
        csv_files = list(self.csv_directory.glob("top_ranked_stocks_*.csv"))

        for file in csv_files:
            # Extract date from filename
            file_name = file.stem
            date_str = file_name.split('_')[-1]

            if len(date_str) == 8:  # Changed from 4 to 8
                file_year = int(date_str[:4])
                file_month = int(date_str[4:6])
                file_day = int(date_str[6:])

                # Check if file is within date range
                file_date = file_year * 10000 + file_month * 100 + file_day
                start_date_num = start_year * 10000 + start_month * 100 + start_day
                end_date_num = end_year * 10000 + end_month * 100 + end_day

                if start_date_num <= file_date <= end_date_num:
                    files_data.append({
                        'file': file,
                        'date': date_str,
                        'year': file_year,
                        'month': file_month,
                        'day': file_day
                    })

        # Sort by date
        files_data.sort(key=lambda x: (x['month'], x['day']))

        print(f"Found {len(files_data)} files in date range {self.start_date} to {self.end_date}")
        return files_data

    def build_score_history(self, files_data):
        """Build score history for all tickers"""
        for idx, file_info in enumerate(files_data):
            df = pd.read_csv(file_info['file'])

            for _, row in df.iterrows():
                ticker = row['Ticker']
                score = row['Score']

                if ticker not in self.score_history:
                    self.score_history[ticker] = {
                        'indices': [],
                        'scores': [],
                        'dates': []
                    }

                self.score_history[ticker]['indices'].append(idx)
                self.score_history[ticker]['scores'].append(score)
                self.score_history[ticker]['dates'].append(file_info['date'])

        print(f"Built score history for {len(self.score_history)} tickers")

    def perform_regression_analysis(self, ticker):
        """Perform various regression analyses on ticker score history"""
        data = self.score_history.get(ticker)
        if not data or len(data['indices']) < 3:
            return None

        indices = np.array(data['indices']).reshape(-1, 1)
        scores = np.array(data['scores'])

        results = {}

        # Linear regression
        lr = LinearRegression()
        lr.fit(indices, scores)
        results['linear'] = {
            'slope': lr.coef_[0],
            'intercept': lr.intercept_,
            'r2': r2_score(scores, lr.predict(indices)),
            'predictions': lr.predict(indices)
        }

        # Polynomial regression (degree 2)
        poly_features = PolynomialFeatures(degree=2)
        X_poly = poly_features.fit_transform(indices)
        poly_reg = LinearRegression()
        poly_reg.fit(X_poly, scores)
        results['polynomial'] = {
            'r2': r2_score(scores, poly_reg.predict(X_poly)),
            'predictions': poly_reg.predict(X_poly)
        }

        return results

    def calculate_moving_averages(self, ticker):
        """Calculate moving averages for score history"""
        data = self.score_history.get(ticker)
        if not data or len(data['scores']) < 3:
            return None

        scores = pd.Series(data['scores'])

        results = {}

        # Simple Moving Averages
        if len(scores) >= 5:
            results['sma_5'] = scores.rolling(window=5).mean().tolist()
        if len(scores) >= 10:
            results['sma_10'] = scores.rolling(window=10).mean().tolist()
        if len(scores) >= 20:
            results['sma_20'] = scores.rolling(window=20).mean().tolist()

        # Exponential Moving Averages
        if len(scores) >= 5:
            results['ema_5'] = scores.ewm(span=5, adjust=False).mean().tolist()
        if len(scores) >= 10:
            results['ema_10'] = scores.ewm(span=10, adjust=False).mean().tolist()

        # Crossover signals
        if 'sma_5' in results and 'sma_10' in results:
            signals = []
            for i in range(1, len(scores)):
                if i < 10:
                    signals.append('neutral')
                else:
                    if results['sma_5'][i] > results['sma_10'][i] and results['sma_5'][i-1] <= results['sma_10'][i-1]:
                        signals.append('bullish')
                    elif results['sma_5'][i] < results['sma_10'][i] and results['sma_5'][i-1] >= results['sma_10'][i-1]:
                        signals.append('bearish')
                    else:
                        signals.append('neutral')
            results['crossover_signal'] = signals

        return results

    def calculate_momentum_indicators(self, ticker):
        """Calculate RSI, MACD, and ROC for score history"""
        data = self.score_history.get(ticker)
        if not data or len(data['scores']) < 5:
            return None

        scores = pd.Series(data['scores'])
        results = {}

        # RSI (Relative Strength Index)
        if len(scores) >= 14:
            delta = scores.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            results['rsi'] = rsi.tolist()

        # MACD
        if len(scores) >= 26:
            ema_12 = scores.ewm(span=12, adjust=False).mean()
            ema_26 = scores.ewm(span=26, adjust=False).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_histogram = macd_line - signal_line

            results['macd'] = macd_line.tolist()
            results['macd_signal'] = signal_line.tolist()
            results['macd_histogram'] = macd_histogram.tolist()

        # Rate of Change (ROC)
        if len(scores) >= 10:
            roc = scores.pct_change(periods=10) * 100
            results['roc'] = roc.tolist()

        return results

    def calculate_bollinger_bands(self, ticker):
        """Calculate Bollinger Bands for score history"""
        data = self.score_history.get(ticker)
        if not data or len(data['scores']) < 20:
            return None

        scores = pd.Series(data['scores'])

        # 20-period SMA as middle band
        middle_band = scores.rolling(window=20).mean()
        std = scores.rolling(window=20).std()

        upper_band = middle_band + (std * 2)
        lower_band = middle_band - (std * 2)

        # Calculate %B (position within bands)
        percent_b = (scores - lower_band) / (upper_band - lower_band)

        # Bandwidth
        bandwidth = (upper_band - lower_band) / middle_band

        return {
            'upper_band': upper_band.tolist(),
            'middle_band': middle_band.tolist(),
            'lower_band': lower_band.tolist(),
            'percent_b': percent_b.tolist(),
            'bandwidth': bandwidth.tolist()
        }

    def calculate_trend_strength(self, ticker):
        """Calculate ADX and Directional Indicators"""
        data = self.score_history.get(ticker)
        if not data or len(data['scores']) < 14:
            return None

        scores = pd.Series(data['scores'])
        indices = pd.Series(data['indices'])

        # Calculate directional movements
        high = scores.rolling(2).max()
        low = scores.rolling(2).min()

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        # True Range
        tr = high - low

        # Smoothed averages
        atr = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(14).mean()

        return {
            'adx': adx.tolist(),
            'plus_di': plus_di.tolist(),
            'minus_di': minus_di.tolist()
        }

    def perform_statistical_forecasting(self, ticker):
        """Perform ARIMA and Exponential Smoothing forecasts"""
        data = self.score_history.get(ticker)
        if not data or len(data['scores']) < 10:
            return None

        scores = np.array(data['scores'])
        results = {}

        # Forecast periods
        forecast_periods = min(30, max(1, len(scores) // 10))

        # ARIMA
        try:
            # Grid search for best ARIMA parameters
            best_aic = np.inf
            best_params = (1, 0, 0)

            for p in range(3):
                for d in range(2):
                    for q in range(3):
                        try:
                            model = ARIMA(scores, order=(p, d, q), trend='c')  # Add constant/drift term
                            fitted = model.fit()
                            if fitted.aic < best_aic:
                                best_aic = fitted.aic
                                best_params = (p, d, q)
                        except:
                            continue

            # Fit best model with trend
            arima_model = ARIMA(scores, order=best_params, trend='c')
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
            # Try trend-based exponential smoothing for better trend capture
            if len(scores) >= 10:
                # Holt's method (with trend) - better for trending data
                model = ExponentialSmoothing(scores, trend='add', seasonal=None, damped_trend=False)
                fitted = model.fit()
                forecast = fitted.forecast(steps=forecast_periods)

                results['exp_smoothing'] = {
                    'forecast': forecast,
                    'fitted_values': fitted.fittedvalues
                }
            elif len(scores) >= 4:
                # Simple exponential smoothing for very short series
                model = ExponentialSmoothing(scores, trend=None, seasonal=None)
                fitted = model.fit()
                forecast = fitted.forecast(steps=forecast_periods)

                results['exp_smoothing'] = {
                    'forecast': forecast,
                    'fitted_values': fitted.fittedvalues
                }
            else:
                results['exp_smoothing'] = None
        except Exception as e:
            print(f"Exponential Smoothing failed for {ticker}: {str(e)}")
            results['exp_smoothing'] = None

        # Prophet (Facebook's forecasting model)
        try:
            if len(scores) >= 10:
                # Prepare data for Prophet (requires 'ds' and 'y' columns)
                prophet_df = pd.DataFrame({
                    'ds': pd.date_range(start='2024-01-01', periods=len(scores), freq='D'),
                    'y': scores
                })

                # Initialize Prophet with minimal configuration for short time series
                prophet_model = Prophet(
                    changepoint_prior_scale=0.05,  # Lower = less flexible (prevent overfitting)
                    seasonality_prior_scale=0.1,   # Lower = less seasonal variation
                    yearly_seasonality=False,       # Disable for short series
                    weekly_seasonality=False,       # Disable for short series
                    daily_seasonality=False,        # Disable for short series
                    interval_width=0.80,            # 80% confidence intervals
                    changepoint_range=0.8           # Only fit changepoints in first 80%
                )

                # Fit the model (suppress output)
                import logging
                logging.getLogger('prophet').setLevel(logging.ERROR)
                prophet_model.fit(prophet_df)

                # Create future dataframe for forecasting
                future = prophet_model.make_future_dataframe(periods=forecast_periods, freq='D')
                forecast = prophet_model.predict(future)

                # Extract forecast values
                forecast_values = forecast['yhat'].values[-forecast_periods:]
                lower_bound = forecast['yhat_lower'].values[-forecast_periods:]
                upper_bound = forecast['yhat_upper'].values[-forecast_periods:]
                fitted_values = forecast['yhat'].values[:len(scores)]

                # Apply offset correction to align forecast with last historical value
                # This improves short-term accuracy by correcting for model bias
                last_actual = scores[-1]
                last_fitted = fitted_values[-1]
                offset = last_actual - last_fitted

                # Apply offset to all forecasts
                forecast_values_corrected = forecast_values + offset
                lower_bound_corrected = lower_bound + offset
                upper_bound_corrected = upper_bound + offset

                results['prophet'] = {
                    'forecast': forecast_values_corrected,
                    'lower_bound': lower_bound_corrected,
                    'upper_bound': upper_bound_corrected,
                    'fitted_values': fitted_values,
                    'trend': forecast['trend'].values[:len(scores)],
                    'offset': offset  # Store offset for reference
                }
        except Exception as e:
            print(f"Prophet failed for {ticker}: {str(e)}")
            results['prophet'] = None

        return results

    def create_individual_plots(self, ticker, output_dir, use_subfolder=True):
        """Create individual technical analysis plots (no subplots)"""

        # Determine the correct output directory
        if use_subfolder:
            ticker_dir = Path(output_dir) / "technical_plots" / ticker
        else:
            ticker_dir = Path(output_dir) / ticker

        ticker_dir.mkdir(parents=True, exist_ok=True)

        data = self.score_history.get(ticker)
        if not data or len(data['indices']) < 5:
            print(f"Insufficient data for {ticker}")
            return

        indices = data['indices']
        scores = data['scores']
        dates = data['dates']

        # Common plot settings
        def setup_plot(title, ylabel):
            plt.figure(figsize=(10, 4))
            plt.title(title, fontsize=12, fontweight='bold')
            plt.xlabel('Trading Days', fontsize=10)
            plt.ylabel(ylabel, fontsize=10)
            plt.grid(True, alpha=0.3)

        def add_date_labels():
            tick_indices = indices[::max(1, len(indices) // 8)]
            tick_dates = [dates[i] for i in range(0, len(dates), max(1, len(dates) // 8))]
            plt.xticks(tick_indices, tick_dates, rotation=45, fontsize=8)

        # 01 - Regression Analysis
        reg_results = self.regression_results.get(ticker)
        if reg_results:
            setup_plot(f'{ticker} - Regression Analysis', 'Score')
            plt.scatter(indices, scores, alpha=0.6, label='Actual Scores', color='blue', s=20)
            if 'linear' in reg_results:
                plt.plot(indices, reg_results['linear']['predictions'],
                         'r-', label=f"Linear (R²={reg_results['linear']['r2']:.3f})", linewidth=2)
            add_date_labels()
            plt.legend(loc='best', fontsize=9)
            plt.tight_layout()
            plt.savefig(ticker_dir / f'{ticker}_01_regression.png', dpi=100)
            plt.close()

        # 02 - Moving Averages
        ma_results = self.technical_results.get(ticker, {}).get('moving_averages')
        if ma_results:
            setup_plot(f'{ticker} - Moving Averages', 'Score')
            plt.plot(indices, scores, 'b-', label='Actual Scores', alpha=0.7, linewidth=1.5)
            if 'sma_5' in ma_results:
                plt.plot(indices, ma_results['sma_5'], 'r-', label='SMA-5', linewidth=1.5)
            if 'sma_10' in ma_results:
                plt.plot(indices, ma_results['sma_10'], 'g-', label='SMA-10', linewidth=1.5)
            add_date_labels()
            plt.legend(loc='best', fontsize=9)
            plt.tight_layout()
            plt.savefig(ticker_dir / f'{ticker}_02_moving_averages.png', dpi=100)
            plt.close()

        # 03 - Momentum Indicators (split into 3 files)
        momentum = self.technical_results.get(ticker, {}).get('momentum')
        if momentum:
            # 03a - RSI
            if 'rsi' in momentum:
                setup_plot(f'{ticker} - RSI', 'RSI')
                plt.plot(indices, momentum['rsi'], 'purple', linewidth=1.5)
                plt.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
                plt.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
                add_date_labels()
                plt.legend(loc='best', fontsize=9)
                plt.tight_layout()
                plt.savefig(ticker_dir / f'{ticker}_03_momentum_01_RSI.png', dpi=100)
                plt.close()

            # 03b - MACD
            if 'macd' in momentum:
                setup_plot(f'{ticker} - MACD', 'MACD')
                plt.plot(indices, momentum['macd'], 'b-', label='MACD', linewidth=1.5)
                plt.plot(indices, momentum['macd_signal'], 'r-', label='Signal', linewidth=1.5)
                plt.bar(indices, momentum['macd_histogram'], alpha=0.3, label='Histogram')
                add_date_labels()
                plt.legend(loc='best', fontsize=9)
                plt.tight_layout()
                plt.savefig(ticker_dir / f'{ticker}_03_momentum_02_MACD.png', dpi=100)
                plt.close()

            # 03c - ROC
            if 'roc' in momentum:
                setup_plot(f'{ticker} - Rate of Change', 'ROC (%)')
                plt.plot(indices, momentum['roc'], 'orange', linewidth=1.5)
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                add_date_labels()
                plt.tight_layout()
                plt.savefig(ticker_dir / f'{ticker}_03_momentum_03_ROC.png', dpi=100)
                plt.close()

        # 04 - Bollinger Bands (split into 2 files)
        bb = self.technical_results.get(ticker, {}).get('bollinger_bands')
        if bb:
            # 04a - Bands
            setup_plot(f'{ticker} - Bollinger Bands', 'Score')
            plt.plot(indices, scores, 'b-', label='Score', linewidth=1.5)
            plt.plot(indices, bb['upper_band'], 'r--', label='Upper Band', alpha=0.7)
            plt.plot(indices, bb['middle_band'], 'g-', label='Middle Band', alpha=0.7)
            plt.plot(indices, bb['lower_band'], 'r--', label='Lower Band', alpha=0.7)
            plt.fill_between(indices, bb['lower_band'], bb['upper_band'], alpha=0.1)
            add_date_labels()
            plt.legend(loc='best', fontsize=9)
            plt.tight_layout()
            plt.savefig(ticker_dir / f'{ticker}_04_bollinger_bands_01_bands.png', dpi=100)
            plt.close()

            # 04b - %B
            setup_plot(f'{ticker} - Bollinger %B', '%B')
            plt.plot(indices, bb['percent_b'], 'purple', linewidth=1.5)
            plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
            add_date_labels()
            plt.tight_layout()
            plt.savefig(ticker_dir / f'{ticker}_04_bollinger_bands_02_percentB.png', dpi=100)
            plt.close()

        # 05 - Trend Strength (split into 2 files)
        trend = self.technical_results.get(ticker, {}).get('trend_strength')
        if trend:
            # 05a - ADX
            if 'adx' in trend:
                setup_plot(f'{ticker} - ADX', 'ADX')
                plt.plot(indices, trend['adx'], 'green', linewidth=1.5)
                plt.axhline(y=25, color='gray', linestyle='--', alpha=0.5, label='Weak Trend')
                plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Strong Trend')
                add_date_labels()
                plt.legend(loc='best', fontsize=9)
                plt.tight_layout()
                plt.savefig(ticker_dir / f'{ticker}_05_trend_strength_01_ADX.png', dpi=100)
                plt.close()

            # 05b - Directional Indicators
            if 'plus_di' in trend and 'minus_di' in trend:
                setup_plot(f'{ticker} - Directional Indicators', 'DI')
                plt.plot(indices, trend['plus_di'], 'g-', label='+DI', linewidth=1.5)
                plt.plot(indices, trend['minus_di'], 'r-', label='-DI', linewidth=1.5)
                add_date_labels()
                plt.legend(loc='best', fontsize=9)
                plt.tight_layout()
                plt.savefig(ticker_dir / f'{ticker}_05_trend_strength_02_DI.png', dpi=100)
                plt.close()

        # 06 - Forecasting (Enhanced with last 3 months + interpretability)
        forecast = self.technical_results.get(ticker, {}).get('forecasting')
        if forecast:
            setup_plot(f'{ticker} - Statistical Forecasting (Last 3 Months + Forecast)', 'Score')

            # Only show last 90 days (~3 months) of historical data for better forecast visibility
            window_size = 90
            if len(indices) > window_size:
                start_idx = len(indices) - window_size
                plot_indices = indices[start_idx:]
                plot_scores = scores[start_idx:]
            else:
                plot_indices = indices
                plot_scores = scores

            # Get last historical score for percentage calculations
            last_score = scores[-1]

            # Historical data (last 3 months only)
            plt.plot(plot_indices, plot_scores, 'b-',
                     label=f'Historical (Last 3mo) - Current: {last_score:.2f}', linewidth=1.5)

            last_index = indices[-1]
            forecast_range = None

            # ARIMA forecast with final value and % change
            if forecast.get('arima') and forecast['arima']:
                arima_forecast = forecast['arima']['forecast']
                if not hasattr(arima_forecast, '__len__'):
                    arima_forecast = [arima_forecast]
                forecast_indices = list(range(last_index + 1, last_index + 1 + len(arima_forecast)))
                forecast_range = len(arima_forecast)

                final_arima = arima_forecast[-1]
                pct_change = ((final_arima - last_score) / last_score) * 100
                pct_str = f"{pct_change:+.1f}%"

                plt.plot(forecast_indices, arima_forecast, 'r-',
                         label=f"ARIMA{forecast['arima']['params']} → {final_arima:.2f} ({pct_str})",
                         linewidth=2, marker='o')

            # Exponential Smoothing forecast with final value and % change
            if forecast.get('exp_smoothing') and forecast['exp_smoothing']:
                exp_forecast = forecast['exp_smoothing']['forecast']
                if not hasattr(exp_forecast, '__len__'):
                    exp_forecast = [exp_forecast]
                forecast_indices = list(range(last_index + 1, last_index + 1 + len(exp_forecast)))
                if forecast_range is None:
                    forecast_range = len(exp_forecast)

                final_exp = exp_forecast[-1]
                pct_change = ((final_exp - last_score) / last_score) * 100
                pct_str = f"{pct_change:+.1f}%"

                plt.plot(forecast_indices, exp_forecast, '-',
                         color='orange',
                         label=f'Exp Smoothing → {final_exp:.2f} ({pct_str})',
                         linewidth=2, marker='s', alpha=0.7)

            # Prophet forecast with confidence interval, final value and % change
            if forecast.get('prophet') and forecast['prophet']:
                prophet_forecast = forecast['prophet']['forecast']
                if not hasattr(prophet_forecast, '__len__'):
                    prophet_forecast = [prophet_forecast]
                forecast_indices = list(range(last_index + 1, last_index + 1 + len(prophet_forecast)))
                if forecast_range is None:
                    forecast_range = len(prophet_forecast)

                final_prophet = prophet_forecast[-1]
                pct_change = ((final_prophet - last_score) / last_score) * 100
                pct_str = f"{pct_change:+.1f}%"

                plt.plot(forecast_indices, prophet_forecast, 'g-',
                         label=f'Prophet → {final_prophet:.2f} ({pct_str})',
                         linewidth=2.5, marker='D')

                # Add confidence interval
                if 'lower_bound' in forecast['prophet'] and 'upper_bound' in forecast['prophet']:
                    plt.fill_between(forecast_indices,
                                   forecast['prophet']['lower_bound'],
                                   forecast['prophet']['upper_bound'],
                                   alpha=0.2, color='green', label='Prophet 80% CI')

            plt.axvline(x=last_index, color='gray', linestyle='--', alpha=0.5)

            # Add forecast range info as text box
            if forecast_range:
                textstr = f'Forecast Range: {forecast_range} days'
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
                        fontsize=10, verticalalignment='top', bbox=props)

            add_date_labels()
            plt.legend(loc='best', fontsize=9)
            plt.tight_layout()
            plt.savefig(ticker_dir / f'{ticker}_06_forecasting.png', dpi=100)
            plt.close()

        print(f" Created individual plots for {ticker}")

    def create_comprehensive_plots(self, ticker, output_dir, use_subfolder=True):
        """Create all technical analysis plots for a ticker

        Parameters:
        -----------
        ticker : str
            The ticker symbol
        output_dir : str or Path
            Base directory for output
        use_subfolder : bool
            If True, create plots in output_dir/ticker/
            If False, create plots in output_dir/technical_plots/ticker/
        """
        # Determine the correct output directory
        if use_subfolder:
            # For regular analysis: output_dir/technical_plots/ticker/
            ticker_dir = Path(output_dir) / "technical_plots" / ticker
        else:
            # For single ticker analysis when called with specific path
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

        # 6. Forecasting Plot
        self._plot_forecasting(ticker, indices, scores, dates, ticker_dir)

        # 7. Comprehensive Overview
        self._plot_comprehensive_overview(ticker, indices, scores, dates, ticker_dir)

        # Create individual plots for UI display
        self.create_individual_plots(ticker, output_dir, use_subfolder)

        print(f" Created all plots for {ticker}")

    def _plot_regression_analysis(self, ticker, indices, scores, dates, output_dir):
        """Create regression analysis plot"""
        reg_results = self.regression_results.get(ticker)
        if not reg_results:
            return

        plt.figure(figsize=(12, 6))

        # Actual scores
        plt.scatter(indices, scores, alpha=0.6, label='Actual Scores', color='blue')

        # Linear regression
        if 'linear' in reg_results:
            plt.plot(indices, reg_results['linear']['predictions'],
                    'r-', label=f"Linear (R²={reg_results['linear']['r2']:.3f})", linewidth=2)

        # Polynomial regression
        if 'polynomial' in reg_results:
            plt.plot(indices, reg_results['polynomial']['predictions'],
                    'g-', label=f"Polynomial (R²={reg_results['polynomial']['r2']:.3f})", linewidth=2)

        plt.xlabel('Trading Days')
        plt.ylabel('Score')
        plt.title(f'{ticker} - Regression Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add date labels
        tick_indices = indices[::max(1, len(indices)//10)]
        tick_dates = [dates[i] for i in range(0, len(dates), max(1, len(dates)//10))]
        plt.xticks(tick_indices, tick_dates, rotation=45)

        plt.tight_layout()
        plt.savefig(output_dir / f'{ticker}_01_regression.png', dpi=100)
        plt.close()

    def _plot_moving_averages(self, ticker, indices, scores, dates, output_dir):
        """Create moving averages plot"""
        ma_results = self.technical_results.get(ticker, {}).get('moving_averages')
        if not ma_results:
            return

        plt.figure(figsize=(12, 6))

        # Actual scores
        plt.plot(indices, scores, 'b-', label='Actual Scores', alpha=0.7, linewidth=1.5)

        # Moving averages
        if 'sma_5' in ma_results:
            plt.plot(indices, ma_results['sma_5'], 'r-', label='SMA-5', linewidth=1.5)
        if 'sma_10' in ma_results:
            plt.plot(indices, ma_results['sma_10'], 'g-', label='SMA-10', linewidth=1.5)
        if 'sma_20' in ma_results:
            plt.plot(indices, ma_results['sma_20'], 'm-', label='SMA-20', linewidth=1.5)
        if 'ema_5' in ma_results:
            plt.plot(indices, ma_results['ema_5'], 'c--', label='EMA-5', alpha=0.7)

        # Crossover signals
        if 'crossover_signal' in ma_results:
            for i, signal in enumerate(ma_results['crossover_signal']):
                if signal == 'bullish':
                    plt.scatter(indices[i], scores[i], color='green', s=100, marker='^', zorder=5)
                elif signal == 'bearish':
                    plt.scatter(indices[i], scores[i], color='red', s=100, marker='v', zorder=5)

        plt.xlabel('Trading Days')
        plt.ylabel('Score')
        plt.title(f'{ticker} - Moving Averages & Crossovers')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add date labels
        tick_indices = indices[::max(1, len(indices)//10)]
        tick_dates = [dates[i] for i in range(0, len(dates), max(1, len(dates)//10))]
        plt.xticks(tick_indices, tick_dates, rotation=45)

        plt.tight_layout()
        plt.savefig(output_dir / f'{ticker}_02_moving_averages.png', dpi=100)
        plt.close()

    def _plot_momentum_indicators(self, ticker, indices, scores, dates, output_dir):
        """Create momentum indicators plot"""
        momentum = self.technical_results.get(ticker, {}).get('momentum')
        if not momentum:
            return

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # RSI
        if 'rsi' in momentum:
            ax = axes[0]
            ax.plot(indices, momentum['rsi'], 'purple', linewidth=1.5)
            ax.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax.set_ylabel('RSI')
            ax.set_title(f'{ticker} - RSI (Relative Strength Index)')
            ax.grid(True, alpha=0.3)

        # MACD
        if 'macd' in momentum:
            ax = axes[1]
            ax.plot(indices, momentum['macd'], 'b-', label='MACD', linewidth=1.5)
            ax.plot(indices, momentum['macd_signal'], 'r-', label='Signal', linewidth=1.5)
            ax.bar(indices, momentum['macd_histogram'], alpha=0.3, label='Histogram')
            ax.set_ylabel('MACD')
            ax.set_title('MACD')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # ROC
        if 'roc' in momentum:
            ax = axes[2]
            ax.plot(indices, momentum['roc'], 'orange', linewidth=1.5)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.set_ylabel('ROC (%)')
            ax.set_xlabel('Trading Days')
            ax.set_title('Rate of Change')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'{ticker}_03_momentum.png', dpi=100)
        plt.close()

    def _plot_bollinger_bands(self, ticker, indices, scores, dates, output_dir):
        """Create Bollinger Bands plot"""
        bb = self.technical_results.get(ticker, {}).get('bollinger_bands')
        if not bb:
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Bollinger Bands
        ax1.plot(indices, scores, 'b-', label='Score', linewidth=1.5)
        ax1.plot(indices, bb['upper_band'], 'r--', label='Upper Band', alpha=0.7)
        ax1.plot(indices, bb['middle_band'], 'g-', label='Middle Band (SMA-20)', alpha=0.7)
        ax1.plot(indices, bb['lower_band'], 'r--', label='Lower Band', alpha=0.7)
        ax1.fill_between(indices, bb['lower_band'], bb['upper_band'], alpha=0.1)
        ax1.set_ylabel('Score')
        ax1.set_title(f'{ticker} - Bollinger Bands')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # %B Indicator
        ax2.plot(indices, bb['percent_b'], 'purple', linewidth=1.5)
        ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        ax2.set_ylabel('%B')
        ax2.set_xlabel('Trading Days')
        ax2.set_title('Bollinger Bands %B')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'{ticker}_04_bollinger_bands.png', dpi=100)
        plt.close()

    def _plot_trend_strength(self, ticker, indices, scores, dates, output_dir):
        """Create trend strength plot"""
        trend = self.technical_results.get(ticker, {}).get('trend_strength')
        if not trend:
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # ADX
        if 'adx' in trend:
            ax1.plot(indices, trend['adx'], 'green', linewidth=1.5)
            ax1.axhline(y=25, color='gray', linestyle='--', alpha=0.5, label='Weak Trend')
            ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Strong Trend')
            ax1.set_ylabel('ADX')
            ax1.set_title(f'{ticker} - Average Directional Index (ADX)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Directional Indicators
        if 'plus_di' in trend and 'minus_di' in trend:
            ax2.plot(indices, trend['plus_di'], 'g-', label='+DI', linewidth=1.5)
            ax2.plot(indices, trend['minus_di'], 'r-', label='-DI', linewidth=1.5)
            ax2.set_ylabel('DI')
            ax2.set_xlabel('Trading Days')
            ax2.set_title('Directional Indicators')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'{ticker}_05_trend_strength.png', dpi=100)
        plt.close()

    def _plot_forecasting(self, ticker, indices, scores, dates, output_dir):
        """Create forecasting plot (shows last 3 months + forecast for better resolution)"""
        forecast = self.technical_results.get(ticker, {}).get('forecasting')
        if not forecast:
            return

        plt.figure(figsize=(12, 6))

        # Only show last 90 days (~3 months) of historical data for better forecast visibility
        window_size = 90
        if len(indices) > window_size:
            start_idx = len(indices) - window_size
            plot_indices = indices[start_idx:]
            plot_scores = scores[start_idx:]
        else:
            plot_indices = indices
            plot_scores = scores

        # Get last historical score for percentage calculations
        last_score = scores[-1]

        # Historical data (last 3 months only)
        plt.plot(plot_indices, plot_scores, 'b-',
                 label=f'Historical (Last 3mo) - Current: {last_score:.2f}', linewidth=1.5)

        # Forecasts
        last_index = indices[-1]
        forecast_range = None

        if forecast.get('arima') and forecast['arima']:
            arima_forecast = forecast['arima']['forecast']
            # Ensure we handle both array-like and scalar forecasts
            if not hasattr(arima_forecast, '__len__'):
                arima_forecast = [arima_forecast]
            forecast_indices = list(range(last_index + 1, last_index + 1 + len(arima_forecast)))
            forecast_range = len(arima_forecast)

            # Calculate final value and percentage change
            final_arima = arima_forecast[-1]
            pct_change = ((final_arima - last_score) / last_score) * 100
            pct_str = f"{pct_change:+.1f}%"

            plt.plot(forecast_indices, arima_forecast, 'r-',
                     label=f"ARIMA{forecast['arima']['params']} → {final_arima:.2f} ({pct_str})",
                     linewidth=2, marker='o')

        if forecast.get('exp_smoothing') and forecast['exp_smoothing']:
            exp_forecast = forecast['exp_smoothing']['forecast']
            if not hasattr(exp_forecast, '__len__'):
                exp_forecast = [exp_forecast]
            forecast_indices = list(range(last_index + 1, last_index + 1 + len(exp_forecast)))
            if forecast_range is None:
                forecast_range = len(exp_forecast)

            # Calculate final value and percentage change
            final_exp = exp_forecast[-1]
            pct_change = ((final_exp - last_score) / last_score) * 100
            pct_str = f"{pct_change:+.1f}%"

            plt.plot(forecast_indices, exp_forecast, '-',
                     color='orange',
                     label=f'Exp Smoothing → {final_exp:.2f} ({pct_str})',
                     linewidth=2, marker='s', alpha=0.7)

        if forecast.get('holt') and forecast['holt']:
            holt_forecast = forecast['holt']['forecast']
            if not hasattr(holt_forecast, '__len__'):
                holt_forecast = [holt_forecast]
            forecast_indices = list(range(last_index + 1, last_index + 1 + len(holt_forecast)))
            if forecast_range is None:
                forecast_range = len(holt_forecast)

            # Calculate final value and percentage change
            final_holt = holt_forecast[-1]
            pct_change = ((final_holt - last_score) / last_score) * 100
            pct_str = f"{pct_change:+.1f}%"

            plt.plot(forecast_indices, holt_forecast, 'm-',
                     label=f'Holt Method → {final_holt:.2f} ({pct_str})',
                     linewidth=2, marker='^', alpha=0.7)

        # Prophet forecast with confidence interval
        if forecast.get('prophet') and forecast['prophet']:
            prophet_forecast = forecast['prophet']['forecast']
            if not hasattr(prophet_forecast, '__len__'):
                prophet_forecast = [prophet_forecast]
            forecast_indices = list(range(last_index + 1, last_index + 1 + len(prophet_forecast)))
            if forecast_range is None:
                forecast_range = len(prophet_forecast)

            # Calculate final value and percentage change
            final_prophet = prophet_forecast[-1]
            pct_change = ((final_prophet - last_score) / last_score) * 100
            pct_str = f"{pct_change:+.1f}%"

            plt.plot(forecast_indices, prophet_forecast, 'g-',
                     label=f'Prophet → {final_prophet:.2f} ({pct_str})',
                     linewidth=2.5, marker='D')

            # Add confidence interval
            if 'lower_bound' in forecast['prophet'] and 'upper_bound' in forecast['prophet']:
                plt.fill_between(forecast_indices,
                               forecast['prophet']['lower_bound'],
                               forecast['prophet']['upper_bound'],
                               alpha=0.2, color='green', label='Prophet 80% CI')

        # Vertical line to separate historical and forecast
        plt.axvline(x=last_index, color='gray', linestyle='--', alpha=0.5)

        # Add forecast range info as text box
        if forecast_range:
            textstr = f'Forecast Range: {forecast_range} days'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
                    fontsize=10, verticalalignment='top', bbox=props)

        plt.xlabel('Trading Days')
        plt.ylabel('Score')
        plt.title(f'{ticker} - Statistical Forecasting (Last 3 Months + Forecast)')
        plt.legend(loc='best', fontsize=9)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'{ticker}_06_forecasting.png', dpi=100)
        plt.close()

    def _plot_comprehensive_overview(self, ticker, indices, scores, dates, output_dir):
        """Create comprehensive overview plot with all indicators"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

        # 1. Main Score with MA
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(indices, scores, 'b-', label='Score', linewidth=1.5, alpha=0.7)

        ma = self.technical_results.get(ticker, {}).get('moving_averages', {})
        if 'sma_10' in ma:
            ax1.plot(indices, ma['sma_10'], 'r-', label='SMA-10', linewidth=1.5)

        regression = self.regression_results.get(ticker, {})
        if regression and 'linear' in regression:
            ax1.plot(indices, regression['linear']['predictions'], 'g--',
                     label=f"Trend (R²={regression['linear']['r2']:.3f})", alpha=0.7)

        ax1.set_ylabel('Score')
        ax1.set_title(f'{ticker} - Score History & Trend')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # 2. RSI
        ax2 = fig.add_subplot(gs[1, 0])
        momentum = self.technical_results.get(ticker, {}).get('momentum', {})
        if 'rsi' in momentum:
            ax2.plot(indices, momentum['rsi'], 'purple', linewidth=1.5)
            ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
        ax2.set_ylabel('RSI')
        ax2.set_title('RSI')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)

        # 3. MACD - FIXED: Use 'macd_signal' instead of 'signal'
        ax3 = fig.add_subplot(gs[1, 1])
        if 'macd' in momentum and 'macd_signal' in momentum:  # Check for 'macd_signal'
            ax3.plot(indices, momentum['macd'], 'b-', label='MACD', linewidth=1.5)
            ax3.plot(indices, momentum['macd_signal'], 'r-', label='Signal', linewidth=1.5)  # Use 'macd_signal'
            if 'macd_histogram' in momentum:
                ax3.bar(indices, momentum['macd_histogram'], alpha=0.3, label='Histogram')
            ax3.legend(loc='best', fontsize=8)
        ax3.set_ylabel('MACD')
        ax3.set_title('MACD')
        ax3.grid(True, alpha=0.3)

        # 4. Bollinger Bands
        ax4 = fig.add_subplot(gs[2, 0])
        volatility = self.technical_results.get(ticker, {}).get('volatility', {})
        if 'bb_upper' in volatility:
            ax4.plot(indices, scores, 'b-', linewidth=1.5, alpha=0.7)
            ax4.plot(indices, volatility['bb_upper'], 'r--', alpha=0.5)
            ax4.plot(indices, volatility['bb_middle'], 'g-', alpha=0.5)
            ax4.plot(indices, volatility['bb_lower'], 'r--', alpha=0.5)
            ax4.fill_between(indices, volatility['bb_lower'], volatility['bb_upper'], alpha=0.1)
        ax4.set_ylabel('Score')
        ax4.set_title('Bollinger Bands')
        ax4.grid(True, alpha=0.3)

        # 5. Trend Strength
        ax5 = fig.add_subplot(gs[2, 1])
        trend = self.technical_results.get(ticker, {}).get('trend_strength', {})
        if 'adx' in trend:
            ax5.plot(indices, trend['adx'], 'green', linewidth=1.5)
            ax5.axhline(y=25, color='gray', linestyle='--', alpha=0.5)
            ax5.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax5.set_ylabel('ADX')
        ax5.set_title('Trend Strength (ADX)')
        ax5.grid(True, alpha=0.3)

        # 6. Forecast - WITH DIMENSION FIX
        ax6 = fig.add_subplot(gs[3, :])
        forecast = self.technical_results.get(ticker, {}).get('forecasting', {})
        if forecast:
            last_index = indices[-1]

            # Plot recent historical data
            ax6.plot(indices[-10:], scores[-10:], 'b-', label='Recent', linewidth=1.5)

            # ARIMA Forecast
            if 'arima' in forecast and forecast['arima']:
                arima_forecast = forecast['arima']['forecast']

                # Create forecast indices matching the actual forecast length
                if hasattr(arima_forecast, '__len__'):
                    forecast_length = len(arima_forecast)
                else:
                    arima_forecast = [arima_forecast] if not isinstance(arima_forecast,
                                                                        (list, np.ndarray)) else arima_forecast
                    forecast_length = len(arima_forecast)

                forecast_indices = list(range(last_index + 1, last_index + 1 + forecast_length))
                ax6.plot(forecast_indices, arima_forecast, 'r-',
                         label='ARIMA', linewidth=2, marker='o', alpha=0.7)

            # Prophet Forecast
            if 'prophet' in forecast and forecast['prophet']:
                prophet_forecast = forecast['prophet']['forecast']
                if hasattr(prophet_forecast, '__len__'):
                    forecast_length = len(prophet_forecast)
                else:
                    prophet_forecast = [prophet_forecast] if not isinstance(prophet_forecast,
                                                                            (list, np.ndarray)) else prophet_forecast
                    forecast_length = len(prophet_forecast)

                forecast_indices = list(range(last_index + 1, last_index + 1 + forecast_length))
                ax6.plot(forecast_indices, prophet_forecast, 'g-',
                         label='Prophet', linewidth=2, marker='D')

                # Add confidence interval
                if 'lower_bound' in forecast['prophet'] and 'upper_bound' in forecast['prophet']:
                    ax6.fill_between(forecast_indices,
                                   forecast['prophet']['lower_bound'],
                                   forecast['prophet']['upper_bound'],
                                   alpha=0.15, color='green')

            ax6.axvline(x=last_index, color='gray', linestyle='--', alpha=0.5)
            ax6.legend(loc='best', fontsize=8)

        ax6.set_ylabel('Score')
        ax6.set_xlabel('Trading Days')
        ax6.set_title('5-Day Forecast (ARIMA + Prophet)')
        ax6.grid(True, alpha=0.3)

        # Add date labels to all subplots
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            if ax in [ax1, ax6]:  # Full range for top and bottom plots
                tick_indices = indices[::max(1, len(indices) // 8)]
                tick_dates = [dates[i] for i in range(0, len(dates), max(1, len(dates) // 8))]
            else:  # Reduced labels for smaller plots
                tick_indices = indices[::max(1, len(indices) // 5)]
                tick_dates = [dates[i] for i in range(0, len(dates), max(1, len(dates) // 5))]

            ax.set_xticks(tick_indices)
            ax.set_xticklabels(tick_dates, rotation=45, fontsize=8)

        plt.suptitle(f'{ticker} - Comprehensive Technical Analysis', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / f'{ticker}_00_overview.png', dpi=100, bbox_inches='tight')
        plt.close()

    def calculate_technical_score(self, ticker):
        """Calculate a comprehensive technical score for a ticker"""

        if ticker not in self.score_history:
            return None

        score_details = {}
        total_score = 0
        max_possible = 0

        # 1. Trend Direction (Linear Regression Slope)
        if ticker in self.regression_results:
            slope = self.regression_results[ticker]['linear']['slope']
            if slope > 0.5:
                score_details['trend'] = "Strong Uptrend (+1)"
                total_score += 1
            elif slope > 0:
                score_details['trend'] = "Mild Uptrend (+0.5)"
                total_score += 0.5
            elif slope > -0.5:
                score_details['trend'] = "Neutral (0)"
                total_score += 0
            else:
                score_details['trend'] = "Downtrend (0)"
            max_possible += 1

        # 2. Trend Quality (R² Score)
        if ticker in self.regression_results:
            r2 = self.regression_results[ticker]['linear']['r2']
            if r2 > 0.7:
                score_details['trend_quality'] = f"High R²={r2:.3f} (+1)"
                total_score += 1
            elif r2 > 0.4:
                score_details['trend_quality'] = f"Medium R²={r2:.3f} (+0.5)"
                total_score += 0.5
            else:
                score_details['trend_quality'] = f"Low R²={r2:.3f} (0)"
            max_possible += 1

        # 3. Moving Average Position
        ma = self.technical_results.get(ticker, {}).get('moving_averages', {})
        if ma and 'sma_5' in ma and 'sma_10' in ma:
            current_score = self.score_history[ticker]['scores'][-1]
            sma5_current = ma['sma_5'][-1] if not pd.isna(ma['sma_5'][-1]) else 0
            sma10_current = ma['sma_10'][-1] if not pd.isna(ma['sma_10'][-1]) else 0

            if current_score > sma5_current > sma10_current:
                score_details['sma_cross'] = "Above both MAs (+1)"
                total_score += 1
            elif current_score > sma5_current or current_score > sma10_current:
                score_details['sma_cross'] = "Above one MA (+0.5)"
                total_score += 0.5
            else:
                score_details['sma_cross'] = "Below MAs (0)"
            max_possible += 1

        # 4. Recent Crossover
        if ma and 'crossover_signal' in ma:
            recent_signals = ma['crossover_signal'][-5:]
            if 'bullish' in recent_signals:
                score_details['recent_cross'] = "Recent Bullish Cross (+1)"
                total_score += 1
            elif 'bearish' in recent_signals:
                score_details['recent_cross'] = "Recent Bearish Cross (0)"
            else:
                score_details['recent_cross'] = "No recent cross (+0.3)"
                total_score += 0.3
            max_possible += 1

        # 5. MACD Signal
        momentum = self.technical_results.get(ticker, {}).get('momentum', {})
        if momentum and 'macd' in momentum and 'macd_signal' in momentum:
            macd_current = momentum['macd'][-1] if len(momentum['macd']) > 0 else 0
            signal_current = momentum['macd_signal'][-1] if len(momentum['macd_signal']) > 0 else 0

            if not pd.isna(macd_current) and not pd.isna(signal_current):
                if macd_current > signal_current:
                    score_details['macd'] = "MACD above Signal (+1)"
                    total_score += 1
                else:
                    score_details['macd'] = "MACD below Signal (0)"
                max_possible += 1

        # 6. MACD Histogram Trend
        if momentum and 'macd_histogram' in momentum:
            hist = [h for h in momentum['macd_histogram'][-5:] if not pd.isna(h)]
            if len(hist) >= 3:
                if all(hist[i] > hist[i-1] for i in range(1, len(hist))):
                    score_details['macd_hist'] = "Histogram Rising (+1)"
                    total_score += 1
                elif hist[-1] > 0:
                    score_details['macd_hist'] = "Histogram Positive (+0.5)"
                    total_score += 0.5
                else:
                    score_details['macd_hist'] = "Histogram Negative (0)"
                max_possible += 1

        # 7. RSI Level
        if momentum and 'rsi' in momentum:
            rsi_current = [r for r in momentum['rsi'][-3:] if not pd.isna(r)]
            if rsi_current:
                avg_rsi = np.mean(rsi_current)
                if 40 <= avg_rsi <= 60:
                    score_details['rsi_level'] = f"RSI Neutral ({avg_rsi:.1f}) (+0.5)"
                    total_score += 0.5
                elif avg_rsi > 60:
                    score_details['rsi_level'] = f"RSI Strong ({avg_rsi:.1f}) (+1)"
                    total_score += 1
                else:
                    score_details['rsi_level'] = f"RSI Weak ({avg_rsi:.1f}) (0)"
                max_possible += 1

        # 8. RSI Trend
        if momentum and 'rsi' in momentum:
            rsi_recent = [r for r in momentum['rsi'][-5:] if not pd.isna(r)]
            if len(rsi_recent) >= 3:
                rsi_slope = np.polyfit(range(len(rsi_recent)), rsi_recent, 1)[0]
                if rsi_slope > 2:
                    score_details['rsi_trend'] = "RSI Rising (+1)"
                    total_score += 1
                elif rsi_slope > -2:
                    score_details['rsi_trend'] = "RSI Stable (+0.5)"
                    total_score += 0.5
                else:
                    score_details['rsi_trend'] = "RSI Falling (0)"
                max_possible += 1

        # 9. Rate of Change
        if momentum and 'roc' in momentum:
            roc_current = [r for r in momentum['roc'][-3:] if not pd.isna(r)]
            if roc_current:
                avg_roc = np.mean(roc_current)
                if avg_roc > 5:
                    score_details['roc'] = f"ROC Positive ({avg_roc:.1f}%) (+1)"
                    total_score += 1
                elif avg_roc > 0:
                    score_details['roc'] = f"ROC Mildly Positive ({avg_roc:.1f}%) (+0.5)"
                    total_score += 0.5
                else:
                    score_details['roc'] = f"ROC Negative ({avg_roc:.1f}%) (0)"
                max_possible += 1

        # 10. Bollinger Bands Position
        bb = self.technical_results.get(ticker, {}).get('bollinger_bands', {})
        if bb and 'percent_b' in bb:
            pb_current = [p for p in bb['percent_b'][-3:] if not pd.isna(p)]
            if pb_current:
                avg_pb = np.mean(pb_current)
                if 0.5 <= avg_pb <= 0.8:
                    score_details['bb_position'] = f"%B Optimal ({avg_pb:.2f}) (+1)"
                    total_score += 1
                elif 0.2 <= avg_pb <= 0.9:
                    score_details['bb_position'] = f"%B Normal ({avg_pb:.2f}) (+0.5)"
                    total_score += 0.5
                else:
                    score_details['bb_position'] = f"%B Extreme ({avg_pb:.2f}) (0)"
                max_possible += 1

        # 11. Bollinger Band Width
        if bb and 'bandwidth' in bb:
            bw = [b for b in bb['bandwidth'][-5:] if not pd.isna(b)]
            if len(bw) >= 3:
                bw_trend = np.polyfit(range(len(bw)), bw, 1)[0]
                if bw_trend < 0:
                    score_details['bb_width'] = "BB Contracting (+0.5)"
                    total_score += 0.5
                else:
                    score_details['bb_width'] = "BB Expanding (+0.3)"
                    total_score += 0.3
                max_possible += 1

        # 12. ADX Strength
        trend = self.technical_results.get(ticker, {}).get('trend_strength', {})
        if trend and 'adx' in trend:
            adx_current = [a for a in trend['adx'][-3:] if not pd.isna(a)]
            if adx_current:
                avg_adx = np.mean(adx_current)
                if avg_adx > 40:
                    score_details['adx_strength'] = f"Strong Trend (ADX={avg_adx:.1f}) (+1)"
                    total_score += 1
                elif avg_adx > 25:
                    score_details['adx_strength'] = f"Moderate Trend (ADX={avg_adx:.1f}) (+0.5)"
                    total_score += 0.5
                else:
                    score_details['adx_strength'] = f"Weak Trend (ADX={avg_adx:.1f}) (0)"
                max_possible += 1

        # 13. Directional Indicators
        if trend and 'plus_di' in trend and 'minus_di' in trend:
            plus_di = [p for p in trend['plus_di'][-3:] if not pd.isna(p)]
            minus_di = [m for m in trend['minus_di'][-3:] if not pd.isna(m)]

            if plus_di and minus_di:
                if np.mean(plus_di) > np.mean(minus_di):
                    score_details['di_direction'] = "+DI > -DI (+1)"
                    total_score += 1
                else:
                    score_details['di_direction'] = "-DI > +DI (0)"
                max_possible += 1

        # 14. ARIMA Forecast Direction
        forecast = self.technical_results.get(ticker, {}).get('forecasting', {})
        if forecast and 'arima' in forecast and forecast['arima']:
            current_score = self.score_history[ticker]['scores'][-1]
            forecast_values = forecast['arima']['forecast']
            if len(forecast_values) > 0:
                if forecast_values[0] > current_score:
                    score_details['arima'] = "ARIMA Bullish (+1)"
                    total_score += 1
                else:
                    score_details['arima'] = "ARIMA Bearish (0)"
                max_possible += 1

        # 15. Exponential Smoothing Forecast
        if forecast and 'exp_smoothing' in forecast and forecast['exp_smoothing']:
            current_score = self.score_history[ticker]['scores'][-1]
            forecast_values = forecast['exp_smoothing']['forecast']
            if len(forecast_values) > 0:
                if forecast_values[0] > current_score:
                    score_details['exp_smooth'] = "Exp Smooth Bullish (+1)"
                    total_score += 1
                else:
                    score_details['exp_smooth'] = "Exp Smooth Bearish (0)"
                max_possible += 1

        # 16. Prophet Forecast
        if forecast and 'prophet' in forecast and forecast['prophet']:
            current_score = self.score_history[ticker]['scores'][-1]
            forecast_values = forecast['prophet']['forecast']
            if len(forecast_values) > 0:
                forecast_score = forecast_values[0]
                # Check if prediction is bullish
                if forecast_score > current_score:
                    # Check confidence interval width for reliability
                    if 'lower_bound' in forecast['prophet'] and 'upper_bound' in forecast['prophet']:
                        lower = forecast['prophet']['lower_bound'][0]
                        upper = forecast['prophet']['upper_bound'][0]
                        ci_width = upper - lower

                        # Narrower confidence interval = more confident prediction
                        if ci_width < (current_score * 0.1):  # CI < 10% of current score
                            score_details['prophet'] = f"Prophet Strong Bullish (+1.5)"
                            total_score += 1.5
                        else:
                            score_details['prophet'] = f"Prophet Bullish (+1)"
                            total_score += 1
                    else:
                        score_details['prophet'] = "Prophet Bullish (+1)"
                        total_score += 1
                elif forecast_score < current_score:
                    score_details['prophet'] = "Prophet Bearish (0)"
                else:
                    score_details['prophet'] = "Prophet Neutral (+0.3)"
                    total_score += 0.3
                max_possible += 1.5  # Account for potential 1.5 score

        # Calculate percentage score
        percentage = (total_score / max_possible * 100) if max_possible > 0 else 0

        # Determine recommendation
        if percentage >= 70:
            recommendation = "STRONG BUY"
        elif percentage >= 55:
            recommendation = "BUY"
        elif percentage >= 45:
            recommendation = "HOLD"
        elif percentage >= 30:
            recommendation = "SELL"
        else:
            recommendation = "STRONG SELL"

        return {
            'total_score': total_score,
            'max_possible': max_possible,
            'percentage': percentage,
            'recommendation': recommendation,
            'details': score_details
        }

    def analyze_all_stocks(self, output_base_dir, min_appearances=5, ticker_list_file=None, clean_old_plots=True):
        """Analyze all stocks and create technical plots"""

        # Clean old plots if requested
        if clean_old_plots:
            self.clean_technical_plots_directory(output_base_dir)

        # Load ranking files and build score history
        files_data = self.load_ranking_files()
        self.build_score_history(files_data)

        # Determine which tickers to analyze
        if ticker_list_file and Path(ticker_list_file).exists():
            df = pd.read_csv(ticker_list_file)
            tickers_to_analyze = df['Ticker'].tolist()
            print(f"\nLoaded {len(tickers_to_analyze)} tickers from {ticker_list_file}")
        else:
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

                    # Create plots with use_subfolder=True for regular analysis
                    self.create_comprehensive_plots(ticker, output_base_dir, use_subfolder=True)
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