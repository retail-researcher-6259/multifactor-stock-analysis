# Prophet Integration - Score Trend Analysis System

## Overview
Prophet (developed by Facebook/Meta) has been successfully integrated into the MSAS Score Trend Analysis system as an advanced forecasting method alongside ARIMA and Exponential Smoothing.

## Why Prophet?

Prophet is particularly well-suited for our stock scoring system because:

1. **Works with Limited Data**: Effective with 50+ data points (our current situation)
2. **Automatic Changepoint Detection**: Identifies regime changes in scoring patterns
3. **Built-in Uncertainty Intervals**: Provides confidence bands for predictions
4. **Robust to Missing Data**: Handles gaps in historical data gracefully
5. **Trend Decomposition**: Separates trend, seasonality, and residuals
6. **Minimal Overfitting Risk**: Conservative defaults prevent overfitting on short series

## What Was Added

### 1. Forecasting Method (`stock_score_trend_technical_03.py`)
- **Prophet forecasting** in `perform_statistical_forecasting()` method
- Conservative hyperparameters optimized for short time series:
  - `changepoint_prior_scale=0.05` (less flexible, prevents overfitting)
  - `seasonality_prior_scale=0.1` (minimal seasonal variation)
  - Disabled yearly/weekly/daily seasonality (not applicable for short series)
  - 80% confidence intervals
  - Changepoint range limited to first 80% of data

### 2. Enhanced Visualizations
All forecasting plots now include:
- **Prophet forecast line** (green with diamond markers)
- **80% confidence intervals** (shaded green area)
- Comparison with ARIMA and Exponential Smoothing forecasts

Updated plots:
- Individual forecasting plot (`{ticker}_06_forecasting.png`)
- Comprehensive forecasting plot (subplot version)
- Overview dashboard (shows Prophet in 5-day forecast panel)

### 3. Technical Scoring System
Prophet predictions now contribute to the technical score:

**Scoring Criteria:**
- **Prophet Strong Bullish (+1.5 points)**: Bullish forecast with narrow confidence interval (< 10% of current score)
- **Prophet Bullish (+1.0 point)**: Bullish forecast with wider confidence interval
- **Prophet Neutral (+0.3 points)**: Flat forecast
- **Prophet Bearish (0 points)**: Bearish forecast

This adds up to 1.5 points to the maximum possible technical score.

### 4. Single Ticker Analysis
The `stock_score_trend_technical_single_enhanced.py` script now includes:
- Prophet in the ranking CSV output
- Prophet signals in the detailed indicator breakdown
- Prophet scoring in the technical ranking

## Usage

### Installation
First, install Prophet:
```bash
pip install prophet
```

### Running with Prophet
The Prophet integration is **automatic** - no code changes needed!

Simply run your existing analysis scripts:

```python
# For comprehensive analysis
python src/trend_analysis/stock_score_trend_technical_03.py

# For single ticker analysis
python src/trend_analysis/stock_score_trend_technical_single_enhanced.py
```

### Configuration
Prophet parameters are set conservatively for short time series. If you have more data (200+ points), you can adjust in `perform_statistical_forecasting()`:

```python
prophet_model = Prophet(
    changepoint_prior_scale=0.1,   # Increase for more flexibility
    seasonality_prior_scale=0.5,   # Increase to detect seasonality
    interval_width=0.95,            # Wider confidence intervals
)
```

## Advantages Over ARIMA

| Feature | ARIMA | Prophet |
|---------|-------|---------|
| **Data Requirements** | 30-50 points | 50+ points (similar) |
| **Changepoint Detection** | Manual | Automatic |
| **Confidence Intervals** | Complex to compute | Built-in |
| **Handles Missing Data** | Poor | Excellent |
| **Trend Flexibility** | Linear/polynomial | Piecewise linear |
| **Parameter Tuning** | Grid search required | Good defaults |
| **Interpretability** | Moderate | High (trend decomposition) |

## Output Examples

### Forecasting Plot Features:
1. **Historical scores** (blue line)
2. **ARIMA forecast** (red line with circles)
3. **Prophet forecast** (green line with diamonds)
4. **Prophet confidence band** (shaded green area)
5. **Forecast boundary line** (vertical dashed line)

### Technical Score Example:
```
Forecasting:
  • ARIMA                : ARIMA Bullish (+1)
  • Exp Smooth           : Exp Smooth Bullish (+1)
  • Prophet              : Prophet Strong Bullish (+1.5)
```

## Performance Considerations

- **Speed**: Prophet is slower than ARIMA for single predictions (~2-5 seconds per ticker)
- **Memory**: Minimal additional memory usage
- **Scalability**: For 100 tickers, expect ~5-10 minutes additional processing time

## Troubleshooting

### If Prophet fails:
The system will gracefully skip Prophet and continue with ARIMA/ES:
```
Prophet failed for AAPL: [error message]
```

Common issues:
1. **Not enough data** - Prophet requires minimum 10 points
2. **Missing dependencies** - Install pystan: `pip install pystan`
3. **Convergence issues** - Rare with conservative parameters

## Future Enhancements

Potential improvements for when you have more data:

1. **Seasonal Patterns** (when you have 6+ months of data):
   - Enable weekly seasonality for trading week patterns
   - Custom seasonalities for quarter-end effects

2. **Exogenous Variables** (when market data is available):
   - Add market index as regressor
   - Include sector performance data

3. **Ensemble Forecasting** (next priority):
   - Weighted average of ARIMA, Prophet, and ES
   - Based on historical accuracy

## Files Modified

1. `src/trend_analysis/stock_score_trend_technical_03.py`
   - Added Prophet import
   - Enhanced `perform_statistical_forecasting()` method
   - Updated all plotting functions
   - Updated `calculate_technical_score()` method

2. `src/trend_analysis/stock_score_trend_technical_single_enhanced.py`
   - Added Prophet to indicator columns
   - Updated ranking data structure
   - Added Prophet to display functions

## Validation

To validate Prophet is working:

1. Check forecasting plots for green Prophet line with confidence bands
2. Look for Prophet scores in technical ranking CSV
3. Check console output for "Prophet failed" messages (should be none)

## Questions?

If you encounter issues or want to adjust Prophet parameters for your specific data characteristics, the main configuration is in the `perform_statistical_forecasting()` method around line 376-420 in `stock_score_trend_technical_03.py`.
