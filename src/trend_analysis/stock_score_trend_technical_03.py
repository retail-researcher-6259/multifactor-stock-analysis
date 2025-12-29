"""
Stock Score Trend Analyzer with Technical Analysis Methods
Enhanced version with MA, RSI, ROC, Bollinger Bands, and Statistical Forecasting
Saves individual plots for each ticker in organized folder structure
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
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

    def calculate_acceleration_metrics(self, ticker):
        """
        Calculate velocity (1st derivative) and acceleration (2nd derivative) of score trend
        Returns dict with velocity array, acceleration array, and momentum phase
        """
        data = self.score_history.get(ticker)
        if not data or len(data['scores']) < 4:
            return None

        scores = np.array(data['scores'])
        indices = np.array(data['indices'])

        # Velocity: first differences (1st derivative)
        velocity = np.diff(scores)

        # Acceleration: second differences (2nd derivative)
        acceleration = np.diff(velocity)

        # Pad arrays to match original length for plotting
        # Velocity: pad 1 NaN at start
        velocity_padded = np.concatenate([[np.nan], velocity])
        # Acceleration: pad 2 NaNs at start
        acceleration_padded = np.concatenate([[np.nan, np.nan], acceleration])

        # Calculate recent metrics (last 3 points)
        recent_velocity = np.nanmean(velocity[-3:]) if len(velocity) >= 3 else np.nanmean(velocity)
        recent_acceleration = np.nanmean(acceleration[-3:]) if len(acceleration) >= 3 else np.nanmean(acceleration)

        # Determine momentum phase
        if recent_velocity > 0.1 and recent_acceleration > 0.05:
            phase = 'accelerating_up'
            phase_display = '▲▲ Accelerating Up'
            phase_color = 'darkgreen'
        elif recent_velocity > 0.1 and recent_acceleration < -0.05:
            phase = 'decelerating_up'
            phase_display = '▲▽ Decelerating Up (Caution)'
            phase_color = 'orange'
        elif recent_velocity < -0.1 and recent_acceleration > 0.05:
            phase = 'decelerating_down'
            phase_display = '▽▲ Decelerating Down (Watch)'
            phase_color = 'gold'
        elif recent_velocity < -0.1 and recent_acceleration < -0.05:
            phase = 'accelerating_down'
            phase_display = '▽▽ Accelerating Down'
            phase_color = 'darkred'
        else:
            phase = 'stable'
            phase_display = '● Stable'
            phase_color = 'gray'

        return {
            'velocity': velocity_padded,
            'acceleration': acceleration_padded,
            'recent_velocity': round(recent_velocity, 4),
            'recent_acceleration': round(recent_acceleration, 4),
            'phase': phase,
            'phase_display': phase_display,
            'phase_color': phase_color
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

    def perform_xgboost_reversion_analysis(self, ticker):
        """
        Perform XGBoost-based reversion prediction analysis.
        Predicts probability of reversion in the next N days.
        """
        data = self.score_history.get(ticker)
        if not data or len(data['scores']) < 30:
            return None

        scores = np.array(data['scores'])
        n_samples = len(scores)

        # Minimum samples needed for meaningful training
        if n_samples < 50:
            return None

        results = {}

        try:
            # Feature engineering for each time point
            features_list = []
            labels_list = []

            # Define reversion parameters
            reversion_threshold = 0.05  # 5% reversion
            forecast_horizon = 10  # Predict reversion within 10 days
            lookback = 20  # Features from last 20 days

            # Build training data
            for i in range(lookback, n_samples - forecast_horizon):
                window = scores[i-lookback:i+1]
                current_score = scores[i]

                # Calculate features
                features = self._calculate_xgboost_features(window, current_score)
                features_list.append(features)

                # Calculate label: Did reversion occur in next N days?
                future_scores = scores[i+1:i+1+forecast_horizon]
                max_future = np.max(future_scores)
                min_future = np.min(future_scores)

                # Determine if significant reversion occurred
                if current_score > np.mean(scores[i-lookback:i]):
                    # Score is above mean - check for downward reversion
                    pct_drop = (current_score - min_future) / current_score
                    reversion_occurred = pct_drop >= reversion_threshold
                else:
                    # Score is below mean - check for upward reversion
                    pct_rise = (max_future - current_score) / current_score if current_score > 0 else 0
                    reversion_occurred = pct_rise >= reversion_threshold

                labels_list.append(1 if reversion_occurred else 0)

            if len(features_list) < 30:
                return None

            X = np.array(features_list)
            y = np.array(labels_list)

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = []

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='logloss',
                    verbosity=0
                )
                model.fit(X_train, y_train)
                cv_scores.append(model.score(X_val, y_val))

            # Train final model on all data
            final_model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                verbosity=0
            )
            final_model.fit(X, y)

            # Predict current reversion probability
            current_window = scores[-lookback-1:]
            current_features = self._calculate_xgboost_features(current_window, scores[-1])
            current_features = np.array(current_features).reshape(1, -1)

            reversion_probability = final_model.predict_proba(current_features)[0][1]
            reversion_prediction = final_model.predict(current_features)[0]

            # Get feature importance
            feature_names = [
                'z_score_10d', 'z_score_20d', 'momentum_decay',
                'rsi_proxy', 'roc_5d', 'roc_10d', 'volatility',
                'distance_from_high', 'distance_from_low', 'trend_strength',
                'consecutive_direction', 'mean_reversion_speed'
            ]
            feature_importance = dict(zip(feature_names, final_model.feature_importances_))

            # Historical accuracy analysis
            historical_predictions = []
            for i in range(max(lookback, n_samples - 30), n_samples - forecast_horizon):
                window = scores[i-lookback:i+1]
                features = self._calculate_xgboost_features(window, scores[i])
                features = np.array(features).reshape(1, -1)
                pred_prob = final_model.predict_proba(features)[0][1]

                # Actual outcome
                future_scores = scores[i+1:i+1+forecast_horizon]
                current_score = scores[i]
                max_future = np.max(future_scores)
                min_future = np.min(future_scores)

                if current_score > np.mean(scores[max(0, i-lookback):i]):
                    pct_drop = (current_score - min_future) / current_score
                    actual = pct_drop >= reversion_threshold
                else:
                    pct_rise = (max_future - current_score) / current_score if current_score > 0 else 0
                    actual = pct_rise >= reversion_threshold

                historical_predictions.append({
                    'index': i,
                    'predicted_prob': pred_prob,
                    'actual': actual
                })

            results['xgboost'] = {
                'reversion_probability': reversion_probability,
                'reversion_prediction': reversion_prediction,
                'cv_accuracy': np.mean(cv_scores),
                'feature_importance': feature_importance,
                'historical_predictions': historical_predictions,
                'forecast_horizon': forecast_horizon,
                'reversion_threshold': reversion_threshold,
                'training_samples': len(X),
                'positive_rate': np.mean(y)  # Base rate of reversions
            }

        except Exception as e:
            print(f"XGBoost analysis failed for {ticker}: {str(e)}")
            results['xgboost'] = None

        return results

    def _calculate_xgboost_features(self, window, current_score):
        """Calculate features for XGBoost reversion prediction"""

        # Z-scores (distance from mean)
        mean_10d = np.mean(window[-10:])
        std_10d = np.std(window[-10:]) if np.std(window[-10:]) > 0 else 1
        z_score_10d = (current_score - mean_10d) / std_10d

        mean_20d = np.mean(window)
        std_20d = np.std(window) if np.std(window) > 0 else 1
        z_score_20d = (current_score - mean_20d) / std_20d

        # Momentum decay (is momentum slowing?)
        if len(window) >= 10:
            roc_recent = (window[-1] - window[-5]) / window[-5] if window[-5] != 0 else 0
            roc_prior = (window[-5] - window[-10]) / window[-10] if window[-10] != 0 else 0
            momentum_decay = abs(roc_recent) - abs(roc_prior)
        else:
            momentum_decay = 0

        # RSI-like indicator (simplified)
        gains = np.maximum(np.diff(window), 0)
        losses = np.abs(np.minimum(np.diff(window), 0))
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 1
        rsi_proxy = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss > 0 else 50

        # Rate of change
        roc_5d = (window[-1] - window[-5]) / window[-5] * 100 if len(window) >= 5 and window[-5] != 0 else 0
        roc_10d = (window[-1] - window[-10]) / window[-10] * 100 if len(window) >= 10 and window[-10] != 0 else 0

        # Volatility
        volatility = np.std(window[-10:]) if len(window) >= 10 else np.std(window)

        # Distance from recent high/low
        recent_high = np.max(window[-20:])
        recent_low = np.min(window[-20:])
        distance_from_high = (recent_high - current_score) / recent_high if recent_high != 0 else 0
        distance_from_low = (current_score - recent_low) / recent_low if recent_low != 0 else 0

        # Trend strength (linear regression R²)
        if len(window) >= 5:
            x = np.arange(len(window)).reshape(-1, 1)
            y = window
            reg = LinearRegression().fit(x, y)
            trend_strength = reg.score(x, y)
        else:
            trend_strength = 0

        # Consecutive up/down days
        diffs = np.diff(window[-10:])
        if len(diffs) > 0:
            if diffs[-1] > 0:
                consecutive = 1
                for d in reversed(diffs[:-1]):
                    if d > 0:
                        consecutive += 1
                    else:
                        break
            else:
                consecutive = -1
                for d in reversed(diffs[:-1]):
                    if d < 0:
                        consecutive -= 1
                    else:
                        break
        else:
            consecutive = 0

        # Mean reversion speed (historical)
        deviations = window - np.mean(window)
        mean_reversion_speed = -np.corrcoef(deviations[:-1], np.diff(window))[0, 1] if len(window) > 2 else 0
        if np.isnan(mean_reversion_speed):
            mean_reversion_speed = 0

        return [
            z_score_10d, z_score_20d, momentum_decay,
            rsi_proxy, roc_5d, roc_10d, volatility,
            distance_from_high, distance_from_low, trend_strength,
            consecutive, mean_reversion_speed
        ]

    def get_options_flow_data(self, ticker, lookback_days=30):
        """
        Get options flow data using the scoring script's working yahooquery setup

        Returns dictionary with options flow metrics
        """
        try:
            print(f"   [DEBUG] Fetching options flow for {ticker}...")

            # Import from scoring script to use its properly-patched yahooquery
            import sys
            scoring_path = str(Path(__file__).parent.parent / "scoring")
            if scoring_path not in sys.path:
                sys.path.insert(0, scoring_path)
                print(f"   [DEBUG] Added scoring path: {scoring_path}")

            # Import the scoring module to trigger its yahooquery patching
            print(f"   [DEBUG] Importing scoring module...")
            import stock_Screener_MultiFactor_25_new
            from yahooquery import Ticker
            print(f"   [DEBUG] Yahooquery imported successfully")

            # Get current stock info for price
            print(f"   [DEBUG] Creating Ticker object for {ticker}...")
            t = Ticker(ticker)

            print(f"   [DEBUG] Fetching summary_detail...")
            info = t.summary_detail.get(ticker, {})
            print(f"   [DEBUG] Info keys: {list(info.keys()) if isinstance(info, dict) else type(info)}")

            # Try multiple price fields in order of preference
            current_price = (
                info.get('regularMarketPrice') or
                info.get('currentPrice') or
                info.get('regularMarketPreviousClose') or
                info.get('previousClose') or
                None
            )

            # If still None, try the price endpoint as fallback
            if current_price is None:
                print(f"   [DEBUG] No price in summary_detail, trying price endpoint...")
                price_info = t.price.get(ticker, {})
                current_price = price_info.get('regularMarketPrice') or price_info.get('regularMarketPreviousClose')
                print(f"   [DEBUG] Price endpoint result: ${current_price}")

            print(f"   [DEBUG] Current price: ${current_price}")

            if current_price is None or current_price <= 0:
                print(f"   [DEBUG] ✗ Failed: No valid current price found")
                return None

            # Fetch options chain
            print(f"   [DEBUG] Fetching options chain...")
            options_chain = t.option_chain
            print(f"   [DEBUG] Options chain type: {type(options_chain)}")

            if options_chain is None or not isinstance(options_chain, pd.DataFrame):
                print(f"   [DEBUG] ✗ Failed: Options chain is not a DataFrame")
                return None

            print(f"   [DEBUG] Options chain length: {len(options_chain)} contracts")

            if len(options_chain) < 10:
                print(f"   [DEBUG] ✗ Failed: Too few contracts ({len(options_chain)} < 10)")
                return None

            # Parse options data
            calls_data = []
            puts_data = []

            for idx, contract in options_chain.iterrows():
                if not isinstance(idx, tuple) or len(idx) < 3:
                    continue

                symbol, exp_date, option_type = idx[0], idx[1], idx[2]

                # Get contract details
                strike = contract.get('strike', 0)
                oi = contract.get('openInterest', 0)
                volume = contract.get('volume', 0)
                last_price = contract.get('lastPrice', 0)

                # Handle NaN values
                if pd.isna(strike) or pd.isna(oi):
                    continue
                if pd.isna(volume):
                    volume = 0
                if pd.isna(last_price):
                    last_price = 0

                strike = float(strike)
                oi = int(oi)
                volume = int(volume)
                last_price = float(last_price)

                if oi <= 0 and volume <= 0:
                    continue

                # Calculate moneyness
                moneyness = strike / current_price

                contract_info = {
                    'strike': strike,
                    'oi': oi,
                    'volume': volume,
                    'price': last_price,
                    'moneyness': moneyness,
                    'notional_oi': oi * 100 * last_price,
                    'notional_vol': volume * 100 * last_price
                }

                if option_type == 'calls':
                    calls_data.append(contract_info)
                else:
                    puts_data.append(contract_info)

            print(f"   [DEBUG] Parsed {len(calls_data)} calls, {len(puts_data)} puts")

            if len(calls_data) < 5 or len(puts_data) < 5:
                print(f"   [DEBUG] ✗ Failed: Insufficient calls ({len(calls_data)}) or puts ({len(puts_data)})")
                return None

            # Calculate metrics
            total_call_oi = sum(c['oi'] for c in calls_data)
            total_put_oi = sum(p['oi'] for p in puts_data)
            total_call_vol = sum(c['volume'] for c in calls_data)
            total_put_vol = sum(p['volume'] for p in puts_data)

            pc_ratio_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0
            pc_ratio_vol = total_put_vol / total_call_vol if total_call_vol > 0 else 1.0

            print(f"   [DEBUG] P/C Ratio OI: {pc_ratio_oi:.3f}, P/C Ratio Vol: {pc_ratio_vol:.3f}")

            # Near-the-money (±10%)
            ntm_calls = [c for c in calls_data if 0.90 <= c['moneyness'] <= 1.10]
            ntm_puts = [p for p in puts_data if 0.90 <= p['moneyness'] <= 1.10]

            ntm_call_oi = sum(c['oi'] for c in ntm_calls)
            ntm_put_oi = sum(p['oi'] for p in ntm_puts)

            # If OI data is missing/zero, fall back to volume as proxy
            if ntm_call_oi == 0 and ntm_put_oi == 0:
                print(f"   [DEBUG] ⚠ WARNING: Near-the-money OI is zero! Falling back to volume...")
                ntm_call_vol = sum(c['volume'] for c in ntm_calls)
                ntm_put_vol = sum(p['volume'] for p in ntm_puts)
                ntm_pc_ratio = ntm_put_vol / ntm_call_vol if ntm_call_vol > 0 else pc_ratio_vol
                print(f"   [DEBUG] Using volume proxy: NTM Call Vol: {ntm_call_vol}, NTM Put Vol: {ntm_put_vol}")
            else:
                ntm_pc_ratio = ntm_put_oi / ntm_call_oi if ntm_call_oi > 0 else pc_ratio_oi

            print(f"   [DEBUG] Near-the-money: {len(ntm_calls)} calls, {len(ntm_puts)} puts")
            print(f"   [DEBUG] NTM Call OI: {ntm_call_oi}, NTM Put OI: {ntm_put_oi}, NTM P/C: {ntm_pc_ratio:.3f}")

            # Large positions (>2 std above mean)
            all_oi = [c['oi'] for c in calls_data] + [p['oi'] for p in puts_data]
            avg_oi = np.mean(all_oi) if all_oi else 0
            std_oi = np.std(all_oi) if len(all_oi) > 1 else 0

            print(f"   [DEBUG] All OI stats: count={len(all_oi)}, mean={avg_oi:.1f}, std={std_oi:.1f}")

            # If average OI is very low (<10), data quality is poor - use volume instead
            if avg_oi < 10:
                print(f"   [DEBUG] ⚠ WARNING: Average OI too low ({avg_oi:.1f}), using volume for large position detection...")
                all_vol = [c['volume'] for c in calls_data] + [p['volume'] for p in puts_data]
                avg_vol = np.mean(all_vol) if all_vol else 0
                std_vol = np.std(all_vol) if len(all_vol) > 1 else 0

                # Use lower threshold when using volume (1 std instead of 2)
                threshold = avg_vol + 1 * std_vol if std_vol > 0 else avg_vol * 1.5

                large_calls = [c for c in ntm_calls if c['volume'] > threshold]
                large_puts = [p for p in ntm_puts if p['volume'] > threshold]

                print(f"   [DEBUG] Volume stats: mean={avg_vol:.1f}, std={std_vol:.1f}, threshold={threshold:.1f}")
                print(f"   [DEBUG] Large positions (by volume, threshold={threshold:.0f}): {len(large_calls)} calls, {len(large_puts)} puts")
                if large_calls:
                    print(f"   [DEBUG] Large call strikes: {[c['strike'] for c in large_calls]} with Vol: {[c['volume'] for c in large_calls]}")
                if large_puts:
                    print(f"   [DEBUG] Large put strikes: {[p['strike'] for p in large_puts]} with Vol: {[p['volume'] for p in large_puts]}")

                # Use volume for the return data too
                large_call_oi = sum(c['volume'] for c in large_calls)
                large_put_oi = sum(p['volume'] for p in large_puts)
            else:
                threshold = avg_oi + 2 * std_oi if std_oi > 0 else avg_oi * 2

                large_calls = [c for c in ntm_calls if c['oi'] > threshold]
                large_puts = [p for p in ntm_puts if p['oi'] > threshold]

                print(f"   [DEBUG] OI threshold={threshold:.1f}")
                print(f"   [DEBUG] Large positions (by OI, threshold={threshold:.0f}): {len(large_calls)} calls, {len(large_puts)} puts")
                if large_calls:
                    print(f"   [DEBUG] Large call strikes: {[c['strike'] for c in large_calls]} with OI: {[c['oi'] for c in large_calls]}")
                if large_puts:
                    print(f"   [DEBUG] Large put strikes: {[p['strike'] for p in large_puts]} with OI: {[p['oi'] for p in large_puts]}")

                large_call_oi = sum(c['oi'] for c in large_calls)
                large_put_oi = sum(p['oi'] for p in large_puts)
            print(f"   [DEBUG] ✓ Successfully fetched options flow data!")

            return {
                'current_price': current_price,
                'total_call_oi': total_call_oi,
                'total_put_oi': total_put_oi,
                'total_call_vol': total_call_vol,
                'total_put_vol': total_put_vol,
                'pc_ratio_oi': pc_ratio_oi,
                'pc_ratio_vol': pc_ratio_vol,
                'ntm_call_oi': ntm_call_oi,
                'ntm_put_oi': ntm_put_oi,
                'ntm_pc_ratio': ntm_pc_ratio,
                'large_calls': large_calls,
                'large_puts': large_puts,
                'large_call_oi': large_call_oi,
                'large_put_oi': large_put_oi
            }

        except Exception as e:
            print(f"   [DEBUG] ✗ EXCEPTION in get_options_flow_data for {ticker}:")
            print(f"   [DEBUG] Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def get_insider_transaction_data(self, ticker, lookback_days=90):
        """
        Get insider transaction data using yahooquery
        Returns dictionary with insider transaction metrics

        Args:
            ticker: Stock ticker symbol
            lookback_days: Number of days to look back (default: 90)

        Returns:
            Dictionary with insider metrics or None if no data
        """
        try:
            print(f"   [DEBUG] Fetching insider transactions for {ticker} ({lookback_days} days)...")

            # Import from scoring script to use its properly-patched yahooquery
            import sys
            from pathlib import Path
            scoring_path = str(Path(__file__).parent.parent / "scoring")
            if scoring_path not in sys.path:
                sys.path.insert(0, scoring_path)

            # Import the scoring module to trigger its yahooquery patching
            print(f"   [DEBUG] Importing scoring module...")
            import stock_Screener_MultiFactor_25_new
            from yahooquery import Ticker
            from datetime import datetime, timedelta
            import pandas as pd
            import numpy as np

            # Create ticker object
            t = Ticker(ticker, session=stock_Screener_MultiFactor_25_new._global_curl_session)

            # Get insider transactions
            df = t.insider_transactions

            # Handle various return types from yahooquery
            if df is None:
                print(f"   [DEBUG] ✗ Failed: No insider data returned")
                return None

            if isinstance(df, dict):
                if ticker not in df:
                    print(f"   [DEBUG] ✗ Failed: Ticker not in insider data")
                    return None
                df = df[ticker]

            if df is None or (hasattr(df, 'empty') and df.empty):
                print(f"   [DEBUG] ✗ Failed: Empty insider data")
                return None

            print(f"   [DEBUG] ✓ Got {len(df)} total insider transactions")

            # Filter to lookback window
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            cutoff_str = cutoff_date.strftime("%Y-%m-%d")

            # Check if startDate column exists
            if 'startDate' not in df.columns:
                print(f"   [DEBUG] ✗ Failed: No 'startDate' column in insider data")
                return None

            df = df[df["startDate"] >= cutoff_str]

            if df.empty:
                print(f"   [DEBUG] ✗ Failed: No recent insider transactions in {lookback_days} days")
                return None

            print(f"   [DEBUG] ✓ Found {len(df)} transactions in last {lookback_days} days")

            # Calculate transaction values
            df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)

            # Parse transaction text to identify buys and sells
            txt = df["transactionText"].str.lower()

            buy_mask = txt.str.contains("buy|purchase", na=False)
            sell_mask = txt.str.contains("sell|sale", na=False)

            buys_df = df[buy_mask]
            sells_df = df[sell_mask]

            buys = buys_df["value"].sum()
            sells = sells_df["value"].sum()
            total = buys + sells

            print(f"   [DEBUG] Buys: ${buys:,.0f}, Sells: ${sells:,.0f}, Total: ${total:,.0f}")

            if total == 0:
                print(f"   [DEBUG] ✗ Failed: No buy/sell transaction values")
                return None

            # Calculate metrics
            net = buys - sells
            buy_ratio = buys / total

            # Calculate unique insiders (breadth)
            unique_insiders = df['filerName'].nunique()
            unique_buyers = buys_df['filerName'].nunique() if not buys_df.empty else 0
            unique_sellers = sells_df['filerName'].nunique() if not sells_df.empty else 0

            # Calculate recency (average days ago)
            days_ago = []
            for date in df['startDate']:
                try:
                    dt = pd.Timestamp(date)
                    delta = (datetime.now() - dt).days
                    days_ago.append(delta)
                except:
                    days_ago.append(lookback_days)

            avg_days_ago = np.mean(days_ago) if days_ago else lookback_days

            # Get transaction counts
            num_buys = len(buys_df)
            num_sells = len(sells_df)

            print(f"   [DEBUG] ✓ Unique insiders: {unique_insiders} ({unique_buyers} buyers, {unique_sellers} sellers)")
            print(f"   [DEBUG] ✓ Transactions: {num_buys} buys, {num_sells} sells")
            print(f"   [DEBUG] ✓ Buy ratio: {buy_ratio:.2%}, Avg recency: {avg_days_ago:.1f} days ago")

            # Prepare transaction details for visualization
            transactions = []
            for _, row in df.iterrows():
                try:
                    trans_type = 'Buy' if 'buy' in row['transactionText'].lower() or 'purchase' in row['transactionText'].lower() else \
                                 'Sell' if 'sell' in row['transactionText'].lower() or 'sale' in row['transactionText'].lower() else \
                                 'Other'

                    transactions.append({
                        'date': row['startDate'],
                        'insider': row['filerName'],
                        'type': trans_type,
                        'value': row['value'],
                        'shares': row.get('shares', 0)
                    })
                except:
                    continue

            return {
                'buys': buys,
                'sells': sells,
                'net': net,
                'total': total,
                'buy_ratio': buy_ratio,
                'num_buys': num_buys,
                'num_sells': num_sells,
                'unique_insiders': unique_insiders,
                'unique_buyers': unique_buyers,
                'unique_sellers': unique_sellers,
                'avg_days_ago': avg_days_ago,
                'lookback_days': lookback_days,
                'transactions': transactions
            }

        except Exception as e:
            print(f"   [DEBUG] ✗ EXCEPTION in get_insider_transaction_data for {ticker}:")
            print(f"   [DEBUG] Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def calculate_sentiment_indicators(self, ticker):
        """
        Calculate sentiment indicators (Options Flow)
        Returns dictionary with sentiment scores
        """
        sentiment_data = {}

        # Get options flow data (30-day window)
        options_data = self.get_options_flow_data(ticker, lookback_days=30)

        if options_data:
            # Calculate options flow score using the same logic as scoring script
            pc_ratio_oi = options_data['pc_ratio_oi']
            pc_ratio_vol = options_data['pc_ratio_vol']
            ntm_pc_ratio = options_data['ntm_pc_ratio']
            large_call_oi = options_data['large_call_oi']
            large_put_oi = options_data['large_put_oi']

            signals = []

            # Signal 1: Overall P/C ratio (30%)
            pc_signal = 1.0 - min(max(pc_ratio_oi - 0.7, 0) / 0.6, 1.0)
            signals.append(('pc_oi', pc_signal, 0.30))

            # Signal 2: Volume P/C ratio (20%)
            pc_vol_signal = 1.0 - min(max(pc_ratio_vol - 0.7, 0) / 0.6, 1.0)
            signals.append(('pc_vol', pc_vol_signal, 0.20))

            # Signal 3: Near-the-money P/C (30%)
            ntm_signal = 1.0 - min(max(ntm_pc_ratio - 0.7, 0) / 0.6, 1.0)
            signals.append(('ntm', ntm_signal, 0.30))

            # Signal 4: Large position imbalance (20%)
            if large_call_oi + large_put_oi > 0:
                large_imbalance = large_call_oi / (large_call_oi + large_put_oi)
                signals.append(('large', large_imbalance, 0.20))
            else:
                signals.append(('large', 0.5, 0.20))

            # Calculate weighted score
            score = sum(signal * weight for _, signal, weight in signals)
            score = max(0.0, min(1.0, score))

            # Convert signals to dict (name -> value), excluding weights
            signals_dict = {name: signal for name, signal, weight in signals}

            sentiment_data['options_flow'] = {
                'score': round(score, 3),
                'raw_data': options_data,
                'signals': signals_dict
            }
        else:
            sentiment_data['options_flow'] = {
                'score': 0.5,  # Neutral
                'raw_data': None,
                'signals': {}
            }

        # ====================
        # 2. INSIDER TRANSACTIONS (90-day window)
        # ====================
        insider_data = self.get_insider_transaction_data(ticker, lookback_days=90)

        if insider_data:
            # Calculate insider score using the same logic as scoring script
            import math

            buys = insider_data['buys']
            sells = insider_data['sells']
            total = insider_data['total']
            buy_ratio = insider_data['buy_ratio']
            unique_insiders = insider_data['unique_insiders']
            avg_days_ago = insider_data['avg_days_ago']
            lookback_days = insider_data['lookback_days']

            # Calculate magnitude factor (larger transactions = stronger signal)
            magnitude = math.log(total + 1) / math.log(10000000)  # 0 to 1 range
            magnitude = min(magnitude, 1.0)

            # Calculate breadth factor (more insiders = stronger signal)
            breadth = min(unique_insiders / 5, 1.0)  # Cap at 5 insiders

            # Calculate recency factor (more recent = stronger signal)
            recency = 1 - (min(avg_days_ago, lookback_days) / lookback_days)

            # Combine factors into final score
            raw_score = (buy_ratio - 0.5) * 2  # -1 to +1 range
            weighted_score = raw_score * (0.5 + 0.2 * magnitude + 0.2 * breadth + 0.1 * recency)

            # Scale to 0-1 range (from -1 to +1)
            # -1 → 0, 0 → 0.5, +1 → 1
            score = (weighted_score + 1) / 2
            score = max(0.0, min(1.0, score))

            sentiment_data['insider'] = {
                'score': round(score, 3),
                'raw_data': insider_data,
                'factors': {
                    'buy_ratio': buy_ratio,
                    'magnitude': magnitude,
                    'breadth': breadth,
                    'recency': recency,
                    'raw_score': raw_score,
                    'weighted_score': weighted_score
                }
            }
        else:
            sentiment_data['insider'] = {
                'score': 0.5,  # Neutral
                'raw_data': None,
                'factors': {}
            }

        return sentiment_data

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

        # 01b - Velocity & Acceleration Analysis (NEW)
        accel_data = self.technical_results.get(ticker, {}).get('acceleration')
        if accel_data:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            fig.suptitle(f'{ticker} - Score Velocity & Acceleration Analysis',
                         fontsize=14, fontweight='bold')

            # Top panel: Score with Velocity bars
            ax1_twin = ax1.twinx()

            # Plot scores as line
            ax1.plot(indices, scores, 'b-', label='Score', linewidth=2, zorder=3)
            ax1.set_ylabel('Score', color='blue', fontsize=11)
            ax1.tick_params(axis='y', labelcolor='blue')

            # Plot velocity as bars
            velocity = accel_data['velocity']
            colors_vel = ['green' if v > 0 else 'red' if v < 0 else 'gray'
                          for v in velocity]
            ax1_twin.bar(indices, velocity, alpha=0.4, color=colors_vel,
                         label='Velocity', zorder=1)
            ax1_twin.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax1_twin.set_ylabel('Velocity (Score Change)', color='gray', fontsize=11)
            ax1_twin.tick_params(axis='y', labelcolor='gray')

            ax1.set_title('Score Trend with Velocity (1st Derivative)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')

            # Bottom panel: Acceleration
            acceleration = accel_data['acceleration']
            colors_accel = ['green' if a > 0 else 'red' if a < 0 else 'gray'
                            for a in acceleration]
            ax2.bar(indices, acceleration, alpha=0.6, color=colors_accel)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax2.fill_between(indices, 0, acceleration,
                             where=[a > 0 if not np.isnan(a) else False for a in acceleration],
                             alpha=0.3, color='green', label='Momentum Increasing')
            ax2.fill_between(indices, 0, acceleration,
                             where=[a < 0 if not np.isnan(a) else False for a in acceleration],
                             alpha=0.3, color='red', label='Momentum Decreasing')

            ax2.set_ylabel('Acceleration (Velocity Change)', fontsize=11)
            ax2.set_xlabel('Trading Days', fontsize=11)
            ax2.set_title('Acceleration (2nd Derivative) - Momentum Change Rate', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper left')

            # Add momentum phase indicator
            phase_display = accel_data['phase_display']
            phase_color = accel_data['phase_color']
            recent_vel = accel_data['recent_velocity']
            recent_accel = accel_data['recent_acceleration']

            textstr = f'Current Phase: {phase_display}\n'
            textstr += f'Recent Velocity: {recent_vel:+.3f}\n'
            textstr += f'Recent Acceleration: {recent_accel:+.3f}'

            props = dict(boxstyle='round', facecolor=phase_color, alpha=0.3)
            ax2.text(0.98, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
                     verticalalignment='top', horizontalalignment='right', bbox=props)

            # Add date labels
            tick_indices = indices[::max(1, len(indices) // 8)]
            tick_dates = [dates[i] for i in range(0, len(dates), max(1, len(dates) // 8))]
            ax2.set_xticks(tick_indices)
            ax2.set_xticklabels(tick_dates, rotation=45, fontsize=8)

            plt.tight_layout()
            plt.savefig(ticker_dir / f'{ticker}_01b_acceleration.png', dpi=100)
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

        # Plot 7: Options Flow Sentiment (30 days)
        print(f"   Creating Plot 7: Options Flow Sentiment...")
        sentiment_data = self.calculate_sentiment_indicators(ticker)

        if sentiment_data and 'options_flow' in sentiment_data:
            options_data = sentiment_data['options_flow']

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            fig.suptitle(f'{ticker} - Options Flow Analysis (30-Day Window)', fontsize=14, fontweight='bold')

            # Top panel: Put/Call Ratio and Large Positions
            if options_data['raw_data']:
                raw = options_data['raw_data']

                # Create bar chart for Put/Call ratios
                categories = ['Total OI\nP/C Ratio', 'Volume\nP/C Ratio', 'Near-Money\nP/C Ratio']
                values = [raw['pc_ratio_oi'], raw['pc_ratio_vol'], raw['ntm_pc_ratio']]
                colors = ['red' if v > 1.0 else 'green' for v in values]

                bars = ax1.bar(categories, values, color=colors, alpha=0.6, edgecolor='black')
                ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='Neutral (1.0)')
                ax1.axhline(y=0.7, color='green', linestyle=':', linewidth=1, alpha=0.5, label='Bullish (<0.7)')
                ax1.axhline(y=1.3, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Bearish (>1.3)')

                ax1.set_ylabel('Put/Call Ratio', fontsize=11, fontweight='bold')
                ax1.set_title('Put/Call Ratios - Lower = More Bullish', fontsize=12)
                ax1.legend(loc='upper right', fontsize=9)
                ax1.grid(True, alpha=0.3)

                # Add value labels on bars
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

                # Bottom panel: Large Positions (OI > 2σ)
                large_call_oi = raw['large_call_oi']
                large_put_oi = raw['large_put_oi']

                if large_call_oi > 0 or large_put_oi > 0:
                    positions = ['Large Calls', 'Large Puts']
                    oi_values = [large_call_oi, large_put_oi]
                    colors = ['green', 'red']

                    bars2 = ax2.barh(positions, oi_values, color=colors, alpha=0.6, edgecolor='black')
                    ax2.set_xlabel('Open Interest (Contracts)', fontsize=11, fontweight='bold')
                    ax2.set_title('Large Positions (OI > 2σ above mean) - Near-the-Money Only', fontsize=12)
                    ax2.grid(True, alpha=0.3, axis='x')

                    # Add value labels
                    for bar, val in zip(bars2, oi_values):
                        width = bar.get_width()
                        ax2.text(width, bar.get_y() + bar.get_height()/2.,
                                f' {int(val):,}', ha='left', va='center', fontsize=10, fontweight='bold')

                    # Add imbalance ratio
                    total_large = large_call_oi + large_put_oi
                    if total_large > 0:
                        call_pct = (large_call_oi / total_large) * 100
                        put_pct = (large_put_oi / total_large) * 100
                        imbalance_text = f'Call: {call_pct:.1f}% | Put: {put_pct:.1f}%'
                        ax2.text(0.98, 0.95, imbalance_text, transform=ax2.transAxes,
                                fontsize=11, verticalalignment='top', horizontalalignment='right',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
                else:
                    ax2.text(0.5, 0.5, 'No Large Positions Detected',
                            ha='center', va='center', fontsize=14, color='gray',
                            transform=ax2.transAxes)
                    ax2.set_xlim(0, 1)
                    ax2.set_ylim(0, 1)

                # Add overall sentiment score
                score = options_data['score']
                sentiment_label = 'BULLISH' if score > 0.6 else ('BEARISH' if score < 0.4 else 'NEUTRAL')
                sentiment_color = 'green' if score > 0.6 else ('red' if score < 0.4 else 'orange')

                fig.text(0.5, 0.02, f'Options Flow Sentiment: {sentiment_label} (Score: {score:.3f})',
                        ha='center', fontsize=13, fontweight='bold', color=sentiment_color,
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            else:
                # No data available
                for ax in [ax1, ax2]:
                    ax.text(0.5, 0.5, 'Options Data Not Available',
                           ha='center', va='center', fontsize=14, color='gray',
                           transform=ax.transAxes)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)

            plt.tight_layout(rect=[0, 0.04, 1, 0.98])
            plt.savefig(ticker_dir / f'{ticker}_07_options_flow.png', dpi=100, bbox_inches='tight')
            plt.close()

        # Plot 8: Insider Transactions (90 days)
        print(f"   Creating Plot 8: Insider Transactions...")
        sentiment_data = self.calculate_sentiment_indicators(ticker)

        if sentiment_data and 'insider' in sentiment_data:
            insider_info = sentiment_data['insider']

            fig = plt.figure(figsize=(14, 10))
            gs = gridspec.GridSpec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.3, wspace=0.3, figure=fig)
            fig.suptitle(f'{ticker} - Insider Transaction Analysis (90-Day Window)',
                         fontsize=14, fontweight='bold', y=0.98)

            if insider_info['raw_data']:
                raw = insider_info['raw_data']
                factors = insider_info['factors']

                # === TOP LEFT: Buy vs Sell Comparison ===
                ax1 = fig.add_subplot(gs[0, 0])
                categories = ['Insider Buys', 'Insider Sells']
                values = [raw['buys'], raw['sells']]
                colors = ['green', 'red']

                bars = ax1.barh(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
                ax1.set_xlabel('Transaction Value ($)', fontsize=11, fontweight='bold')
                ax1.set_title('Insider Buy vs Sell Activity', fontsize=12, fontweight='bold')
                ax1.grid(True, alpha=0.3, axis='x')

                # Add value labels
                for bar, val in zip(bars, values):
                    width = bar.get_width()
                    ax1.text(width, bar.get_y() + bar.get_height()/2.,
                            f' ${val:,.0f}', ha='left', va='center', fontsize=10, fontweight='bold')

                # Add transaction counts
                ax1.text(0.02, 0.98, f"Transactions: {raw['num_buys']} buys, {raw['num_sells']} sells",
                        transform=ax1.transAxes, fontsize=9, va='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

                # === TOP RIGHT: Buy Ratio Gauge ===
                ax2 = fig.add_subplot(gs[0, 1])
                buy_ratio = factors['buy_ratio']

                # Create horizontal bar showing buy ratio
                ax2.barh([0], [buy_ratio], height=0.6, color='green', alpha=0.7, label='Buy %')
                ax2.barh([0], [1-buy_ratio], height=0.6, left=buy_ratio, color='red', alpha=0.7, label='Sell %')

                # Add reference lines
                ax2.axvline(x=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Neutral (50%)')

                # Format
                ax2.set_xlim(0, 1)
                ax2.set_ylim(-0.5, 0.5)
                ax2.set_xlabel('Buy Ratio', fontsize=11, fontweight='bold')
                ax2.set_title(f'Buy Ratio: {buy_ratio:.1%}', fontsize=12, fontweight='bold')
                ax2.set_yticks([])
                ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=9)
                ax2.grid(True, alpha=0.3, axis='x')

                # Add percentage labels
                if buy_ratio > 0.05:
                    ax2.text(buy_ratio/2, 0, f'{buy_ratio:.1%}', ha='center', va='center',
                            fontsize=12, fontweight='bold', color='white')
                if (1-buy_ratio) > 0.05:
                    ax2.text(buy_ratio + (1-buy_ratio)/2, 0, f'{1-buy_ratio:.1%}', ha='center', va='center',
                            fontsize=12, fontweight='bold', color='white')

                # === MIDDLE: Net Position & Insider Details ===
                ax3 = fig.add_subplot(gs[1, :])
                ax3.axis('off')

                net = raw['net']
                net_label = "Net Buying" if net >= 0 else "Net Selling"
                net_color = 'green' if net >= 0 else 'red'

                info_text = f"""NET POSITION: {net_label.upper()} = ${abs(net):,.0f}

INSIDER DETAILS:
  • Total Insiders: {raw['unique_insiders']} ({raw['unique_buyers']} buyers, {raw['unique_sellers']} sellers)
  • Average Recency: {raw['avg_days_ago']:.1f} days ago (of {raw['lookback_days']} day window)
  • Total Value: ${raw['total']:,.0f}

SCORING FACTORS (see bottom panel for visualization):
  • Buy Ratio: {buy_ratio:.1%} → Raw Score: {factors['raw_score']:.3f} (from -1 to +1)
  • Magnitude: {factors['magnitude']:.3f} (transaction size impact)
  • Breadth: {factors['breadth']:.3f} (number of insiders)
  • Recency: {factors['recency']:.3f} (how recent transactions are)

FORMULA: raw_score × (0.5 + 0.2×magnitude + 0.2×breadth + 0.1×recency)
FINAL SCORE: {insider_info['score']:.3f} (scaled to 0-1 range)"""

                ax3.text(0.5, 0.5, info_text, transform=ax3.transAxes,
                        fontsize=9, fontfamily='monospace', ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor=net_color, alpha=0.15, edgecolor=net_color, linewidth=2))

                # === BOTTOM: Scoring Factors ===
                ax4 = fig.add_subplot(gs[2, :])

                factor_names = ['Buy Ratio\n(Base)', 'Magnitude\nFactor', 'Breadth\nFactor', 'Recency\nFactor']
                factor_values = [
                    (buy_ratio - 0.5) * 2,  # Normalized to -1 to +1
                    factors['magnitude'],
                    factors['breadth'],
                    factors['recency']
                ]
                factor_colors = ['blue' if factor_values[0] >= 0 else 'red', 'purple', 'orange', 'cyan']

                bars = ax4.bar(factor_names, factor_values, color=factor_colors, alpha=0.7, edgecolor='black', linewidth=2)
                ax4.set_ylabel('Factor Value', fontsize=11, fontweight='bold')
                ax4.set_title('Scoring Factors Breakdown', fontsize=12, fontweight='bold')
                ax4.set_ylim(-1.1, 1.1)
                ax4.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
                ax4.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.3, label='Bullish threshold')
                ax4.axhline(y=-0.5, color='gray', linestyle='--', linewidth=1, alpha=0.3, label='Bearish threshold')
                ax4.grid(True, alpha=0.3, axis='y')
                ax4.legend(loc='upper right', fontsize=9)

                # Add value labels
                for bar, val in zip(bars, factor_values):
                    height = bar.get_height()
                    label_y = height + (0.05 if height >= 0 else -0.05)
                    va = 'bottom' if height >= 0 else 'top'
                    ax4.text(bar.get_x() + bar.get_width()/2., label_y,
                            f'{val:.2f}', ha='center', va=va, fontsize=10, fontweight='bold')

                # === OVERALL SENTIMENT LABEL ===
                score = insider_info['score']
                if score > 0.6:
                    sentiment_label = 'BULLISH (Strong Insider Buying)'
                    sentiment_color = 'darkgreen'
                elif score > 0.55:
                    sentiment_label = 'MODERATELY BULLISH'
                    sentiment_color = 'green'
                elif score >= 0.45:
                    sentiment_label = 'NEUTRAL'
                    sentiment_color = 'orange'
                elif score >= 0.4:
                    sentiment_label = 'MODERATELY BEARISH'
                    sentiment_color = 'orangered'
                else:
                    sentiment_label = 'BEARISH (Strong Insider Selling)'
                    sentiment_color = 'darkred'

                fig.text(0.5, 0.01, f'Insider Sentiment: {sentiment_label} (Score: {score:.3f})',
                        ha='center', fontsize=13, fontweight='bold', color='white',
                        bbox=dict(boxstyle='round', facecolor=sentiment_color, alpha=0.9,
                                 edgecolor='black', linewidth=3, pad=0.7))

            else:
                # No data available
                ax = fig.add_subplot(gs[:, :])
                ax.text(0.5, 0.5, 'Insider Transaction Data Not Available',
                       ha='center', va='center', fontsize=16, color='gray',
                       transform=ax.transAxes)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')

            plt.tight_layout(rect=[0, 0.03, 1, 0.98])
            plt.savefig(ticker_dir / f'{ticker}_08_insider_transactions.png', dpi=100, bbox_inches='tight')
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

        # 7. XGBoost Reversion Analysis Plot
        self._plot_xgboost_reversion(ticker, indices, scores, dates, ticker_dir)

        # 8. Comprehensive Overview
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

    def _plot_xgboost_reversion(self, ticker, indices, scores, dates, output_dir):
        """Create XGBoost reversion analysis plot"""
        xgboost_results = self.technical_results.get(ticker, {}).get('xgboost_reversion')
        if not xgboost_results or 'xgboost' not in xgboost_results:
            return

        xgb_data = xgboost_results['xgboost']
        if xgb_data is None:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Score history with reversion probability overlay
        ax1 = axes[0, 0]
        ax1.plot(indices, scores, 'b-', linewidth=1.5, alpha=0.7, label='Score')
        ax1.set_xlabel('Trading Days')
        ax1.set_ylabel('Score', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Add current reversion probability as annotation
        current_prob = xgb_data['reversion_probability']
        prob_color = 'red' if current_prob > 0.6 else ('orange' if current_prob > 0.4 else 'green')
        ax1.axhline(y=scores[-1], color=prob_color, linestyle='--', alpha=0.5)

        textstr = f"Current Reversion Prob: {current_prob:.1%}\nPrediction: {'Reversion Likely' if xgb_data['reversion_prediction'] else 'No Reversion'}"
        props = dict(boxstyle='round', facecolor=prob_color, alpha=0.3)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes,
                fontsize=11, verticalalignment='top', bbox=props, fontweight='bold')

        ax1.set_title(f'{ticker} - Current Reversion Probability', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')

        # 2. Feature Importance
        ax2 = axes[0, 1]
        feature_imp = xgb_data['feature_importance']
        sorted_features = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)
        feature_names = [f[0].replace('_', '\n') for f in sorted_features]
        importances = [f[1] for f in sorted_features]

        bars = ax2.barh(range(len(feature_names)), importances, color='steelblue', alpha=0.8)
        ax2.set_yticks(range(len(feature_names)))
        ax2.set_yticklabels(feature_names, fontsize=8)
        ax2.set_xlabel('Importance')
        ax2.set_title('Feature Importance for Reversion Prediction', fontsize=12, fontweight='bold')
        ax2.invert_yaxis()

        # Highlight top 3 features
        for i in range(min(3, len(bars))):
            bars[i].set_color('darkred')
            bars[i].set_alpha(1.0)

        # 3. Historical Prediction Accuracy
        ax3 = axes[1, 0]
        hist_preds = xgb_data['historical_predictions']

        if hist_preds:
            pred_indices = [p['index'] for p in hist_preds]
            pred_probs = [p['predicted_prob'] for p in hist_preds]
            actuals = [p['actual'] for p in hist_preds]

            # Plot probabilities
            ax3.plot(pred_indices, pred_probs, 'purple', linewidth=2, label='Predicted Prob', alpha=0.8)
            ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold')

            # Mark actual reversions
            reversion_indices = [pred_indices[i] for i in range(len(actuals)) if actuals[i]]
            reversion_probs = [pred_probs[i] for i in range(len(actuals)) if actuals[i]]
            no_reversion_indices = [pred_indices[i] for i in range(len(actuals)) if not actuals[i]]
            no_reversion_probs = [pred_probs[i] for i in range(len(actuals)) if not actuals[i]]

            if reversion_indices:
                ax3.scatter(reversion_indices, reversion_probs, color='red', s=80,
                           marker='^', label='Actual Reversion', zorder=5, edgecolors='black')
            if no_reversion_indices:
                ax3.scatter(no_reversion_indices, no_reversion_probs, color='green', s=60,
                           marker='o', label='No Reversion', zorder=5, alpha=0.6)

        ax3.set_xlabel('Trading Days')
        ax3.set_ylabel('Reversion Probability')
        ax3.set_title('Historical Prediction Performance', fontsize=12, fontweight='bold')
        ax3.set_ylim(-0.05, 1.05)
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best', fontsize=9)

        # 4. Model Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')

        stats_text = f"""
XGBoost Reversion Model Statistics
{'='*40}

Model Performance:
  • CV Accuracy: {xgb_data['cv_accuracy']:.1%}
  • Training Samples: {xgb_data['training_samples']}
  • Base Reversion Rate: {xgb_data['positive_rate']:.1%}

Prediction Parameters:
  • Forecast Horizon: {xgb_data['forecast_horizon']} days
  • Reversion Threshold: {xgb_data['reversion_threshold']:.0%}

Current Prediction:
  • Reversion Probability: {xgb_data['reversion_probability']:.1%}
  • Prediction: {'REVERSION LIKELY' if xgb_data['reversion_prediction'] else 'NO REVERSION'}

Top 3 Predictive Features:
"""
        top_features = sorted(xgb_data['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:3]
        for i, (fname, fimp) in enumerate(top_features, 1):
            stats_text += f"  {i}. {fname}: {fimp:.3f}\n"

        # Interpretation
        cv_acc = xgb_data['cv_accuracy']
        prob = xgb_data['reversion_probability']

        if prob > 0.7:
            interpretation = "\nInterpretation: HIGH reversion risk.\nConsider reducing position or waiting for pullback."
        elif prob > 0.5:
            interpretation = "\nInterpretation: MODERATE reversion risk.\nMonitor closely for trend exhaustion signs."
        elif prob > 0.3:
            interpretation = "\nInterpretation: LOW reversion risk.\nTrend likely to continue in short term."
        else:
            interpretation = "\nInterpretation: VERY LOW reversion risk.\nStrong trend continuation expected."

        stats_text += interpretation

        # Add threshold guides
        stats_text += f"\n\n{'='*40}\nModel Confidence Thresholds:\n"
        stats_text += f"  CV > 65%: HIGH confidence\n"
        stats_text += f"  CV 55-65%: MODERATE confidence\n"
        stats_text += f"  CV 50-55%: LOW confidence\n"
        stats_text += f"  CV < 50%: DO NOT TRUST\n"

        stats_text += f"\nReversion Risk Thresholds:\n"
        stats_text += f"  Prob > 70%: HIGH risk\n"
        stats_text += f"  Prob 50-70%: MODERATE risk\n"
        stats_text += f"  Prob 30-50%: LOW risk\n"
        stats_text += f"  Prob < 30%: VERY LOW risk\n"

        # Add current model assessment
        if cv_acc > 0.65:
            model_assessment = "✓ HIGH confidence"
        elif cv_acc > 0.55:
            model_assessment = "✓ MODERATE confidence"
        elif cv_acc > 0.50:
            model_assessment = "⚠ LOW confidence"
        else:
            model_assessment = "✗ DO NOT TRUST"

        stats_text += f"\nCurrent Model: {model_assessment} ({cv_acc:.1%})"

        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(f'{ticker} - XGBoost Reversion Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / f'{ticker}_07_xgboost_reversion.png', dpi=100, bbox_inches='tight')
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

        # 17. XGBoost Reversion Risk (inverted scoring - low reversion risk is good)
        xgboost_results = self.technical_results.get(ticker, {}).get('xgboost_reversion')
        if xgboost_results and 'xgboost' in xgboost_results and xgboost_results['xgboost']:
            xgb_data = xgboost_results['xgboost']
            reversion_prob = xgb_data['reversion_probability']
            cv_accuracy = xgb_data['cv_accuracy']

            # Only trust the prediction if model has reasonable accuracy
            if cv_accuracy >= 0.55:  # Model is better than random
                if reversion_prob < 0.3:
                    score_details['xgboost_reversion'] = f"XGB Low Risk ({reversion_prob:.0%}) (+1)"
                    total_score += 1
                elif reversion_prob < 0.5:
                    score_details['xgboost_reversion'] = f"XGB Moderate Risk ({reversion_prob:.0%}) (+0.5)"
                    total_score += 0.5
                elif reversion_prob < 0.7:
                    score_details['xgboost_reversion'] = f"XGB Elevated Risk ({reversion_prob:.0%}) (+0.3)"
                    total_score += 0.3
                else:
                    score_details['xgboost_reversion'] = f"XGB High Risk ({reversion_prob:.0%}) (0)"
            else:
                score_details['xgboost_reversion'] = f"XGB Low Confidence (CV={cv_accuracy:.0%}) (+0.3)"
                total_score += 0.3
            max_possible += 1

<<<<<<< HEAD
        # 18. Acceleration Analysis (NEW)
        accel_data = self.technical_results.get(ticker, {}).get('acceleration')
        if accel_data:
            velocity = accel_data['recent_velocity']
            acceleration = accel_data['recent_acceleration']
            phase = accel_data['phase']

            # Score based on momentum phase
            if phase == 'accelerating_up':
                score_details['acceleration'] = f"Accelerating Up (v={velocity:+.3f}, a={acceleration:+.3f}) (+1.5)"
                total_score += 1.5
            elif phase == 'decelerating_up':
                score_details['acceleration'] = f"Decelerating Up (v={velocity:+.3f}, a={acceleration:+.3f}) (+0.5)"
                total_score += 0.5
            elif phase == 'decelerating_down':
                score_details['acceleration'] = f"Decelerating Down (v={velocity:+.3f}, a={acceleration:+.3f}) (+0.3)"
                total_score += 0.3
            elif phase == 'accelerating_down':
                score_details['acceleration'] = f"Accelerating Down (v={velocity:+.3f}, a={acceleration:+.3f}) (0)"
                total_score += 0
            else:  # stable
                score_details['acceleration'] = f"Stable (v={velocity:+.3f}, a={acceleration:+.3f}) (+0.7)"
                total_score += 0.7
            max_possible += 1.5

=======
>>>>>>> 19af94f24028bca828033df5510260094d76cda2
        # Store technical-only scores
        technical_score = total_score
        technical_max = max_possible

        # === SENTIMENT INDICATORS (NEW) ===
        sentiment_score = 0
        sentiment_max = 0

        # 18. Options Flow Sentiment (30-day window)
        sentiment_data = self.calculate_sentiment_indicators(ticker)
        if sentiment_data and 'options_flow' in sentiment_data:
            options_flow = sentiment_data['options_flow']
            flow_score = options_flow['score']

            # Convert 0-1 score to points (0-1 scale)
            if flow_score > 0.7:
                score_details['options_flow'] = f"Options Flow Bullish ({flow_score:.3f}) (+1)"
                sentiment_score += 1
            elif flow_score > 0.55:
                score_details['options_flow'] = f"Options Flow Mod Bullish ({flow_score:.3f}) (+0.5)"
                sentiment_score += 0.5
            elif flow_score >= 0.45:
                score_details['options_flow'] = f"Options Flow Neutral ({flow_score:.3f}) (0)"
                sentiment_score += 0
            elif flow_score >= 0.3:
                score_details['options_flow'] = f"Options Flow Mod Bearish ({flow_score:.3f}) (-0.5)"
                sentiment_score -= 0.5
            else:
                score_details['options_flow'] = f"Options Flow Bearish ({flow_score:.3f}) (-1)"
                sentiment_score -= 1
            sentiment_max += 1

        # 19. Insider Transaction Sentiment (90-day window)
        if sentiment_data and 'insider' in sentiment_data:
            insider_info = sentiment_data['insider']
            insider_score = insider_info['score']

            # Convert 0-1 score to points (0-1 scale, with negative for bearish)
            if insider_score > 0.6:
                score_details['insider'] = f"Insider Bullish ({insider_score:.3f}) (+1)"
                sentiment_score += 1
            elif insider_score > 0.55:
                score_details['insider'] = f"Insider Mod Bullish ({insider_score:.3f}) (+0.5)"
                sentiment_score += 0.5
            elif insider_score >= 0.45:
                score_details['insider'] = f"Insider Neutral ({insider_score:.3f}) (0)"
                sentiment_score += 0
            elif insider_score >= 0.4:
                score_details['insider'] = f"Insider Mod Bearish ({insider_score:.3f}) (-0.5)"
                sentiment_score -= 0.5
            else:
                score_details['insider'] = f"Insider Bearish ({insider_score:.3f}) (-1)"
                sentiment_score -= 1
            sentiment_max += 1

        # Add sentiment to total
        total_score += sentiment_score
        max_possible += sentiment_max

        # Calculate percentage score
        technical_percentage = (technical_score / technical_max * 100) if technical_max > 0 else 0
        sentiment_percentage = (sentiment_score / sentiment_max * 100) if sentiment_max > 0 else 0
        percentage = (total_score / max_possible * 100) if max_possible > 0 else 0

        # Determine recommendation based on combined score
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
            'details': score_details,
            # Add breakdown for technical vs sentiment
            'technical_score': technical_score,
            'technical_max': technical_max,
            'technical_percentage': technical_percentage,
            'sentiment_score': sentiment_score,
            'sentiment_max': sentiment_max,
            'sentiment_percentage': sentiment_percentage
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