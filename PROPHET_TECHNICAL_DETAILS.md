# Prophet Integration - Technical Deep Dive

## What is Prophet?

Prophet is Facebook's (now Meta) open-source time series forecasting library. It's specifically designed for:
- Business metrics with strong seasonal patterns
- Multiple seasonalities (daily, weekly, yearly)
- Handling missing data and outliers
- Forecasting with uncertainty intervals
- Automatic trend change detection

**Why Prophet for Stock Scores?**
1. Works with limited data (50+ points) - perfect for 130 days
2. Automatic changepoint detection - identifies regime shifts
3. Built-in confidence intervals - quantifies uncertainty
4. Robust to outliers and missing data
5. Low overfitting risk with conservative defaults

## Implementation in MSAS

### File Location
`/home/user/multifactor-stock-analysis/src/trend_analysis/stock_score_trend_technical_03.py`
Lines: 376-420 (perform_statistical_forecasting method)

### Core Code

```python
def perform_statistical_forecasting(self, ticker):
    """Perform ARIMA and Exponential Smoothing forecasts"""
    data = self.score_history.get(ticker)
    if not data or len(data['scores']) < 10:
        return None  # Minimum 10 data points

    scores = np.array(data['scores'])
    results = {}
    forecast_periods = min(5, max(1, len(scores) // 10))

    # ... ARIMA code ...
    # ... Exponential Smoothing code ...

    # Prophet (Facebook's forecasting model)
    try:
        if len(scores) >= 10:
            # Step 1: Prepare data for Prophet
            prophet_df = pd.DataFrame({
                'ds': pd.date_range(start='2024-01-01', periods=len(scores), freq='D'),
                'y': scores
            })

            # Step 2: Initialize with conservative hyperparameters
            prophet_model = Prophet(
                changepoint_prior_scale=0.05,      # Very conservative
                seasonality_prior_scale=0.1,        # Minimal seasonality
                yearly_seasonality=False,           # Disable
                weekly_seasonality=False,           # Disable
                daily_seasonality=False,            # Disable
                interval_width=0.80,                # 80% confidence
                changepoint_range=0.8               # First 80% only
            )

            # Step 3: Fit the model (suppress output)
            import logging
            logging.getLogger('prophet').setLevel(logging.ERROR)
            prophet_model.fit(prophet_df)

            # Step 4: Create future dataframe for forecasting
            future = prophet_model.make_future_dataframe(
                periods=forecast_periods, freq='D'
            )
            
            # Step 5: Make predictions
            forecast = prophet_model.predict(future)

            # Step 6: Extract forecast values
            forecast_values = forecast['yhat'].values[-forecast_periods:]
            lower_bound = forecast['yhat_lower'].values[-forecast_periods:]
            upper_bound = forecast['yhat_upper'].values[-forecast_periods:]

            # Step 7: Store results
            results['prophet'] = {
                'forecast': forecast_values,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'fitted_values': forecast['yhat'].values[:len(scores)],
                'trend': forecast['trend'].values[:len(scores)]
            }

    except Exception as e:
        print(f"Prophet failed for {ticker}: {str(e)}")
        results['prophet'] = None

    return results
```

## Prophet Hyperparameter Details

### 1. changepoint_prior_scale (Default: 0.05)
**What it does**: Controls how flexible the trend is to changes
**Conservative value**: 0.05
**Aggressive value**: 0.1 or higher

**Explanation**:
- Low (0.05): Fewer, larger changepoints detected
- High (0.1+): More changepoints, more flexible trend
- For 130-day data: Low value prevents false changes from noise

### 2. seasonality_prior_scale (Default: 0.1)
**What it does**: Controls the strength of seasonal patterns
**Conservative value**: 0.1
**For data with seasonality**: 0.5+

**Explanation**:
- Low (0.1): Weak seasonal components
- High (0.5+): Strong seasonal components
- For 130 days: Too short for meaningful seasonality

### 3. yearly_seasonality, weekly_seasonality, daily_seasonality
**All disabled** for short time series
- Need 2+ years for yearly patterns
- Need 6+ weeks for weekly patterns
- Need 2+ weeks for daily patterns
- At 130 days, enabling these causes overfitting

### 4. interval_width (Default: 0.80)
**What it does**: Width of confidence intervals
**Current value**: 0.80 (80% confidence)
**Alternative**: 0.95 (95% confidence - wider bands)

**Explanation**:
- 80%: Narrower bands, more confidence in point forecast
- 95%: Wider bands, less certainty
- 80% selected for actionable signals

### 5. changepoint_range (Default: 0.8)
**What it does**: Proportion of historical data used for changepoint detection
**Current value**: 0.8 (first 80%)
**Alternative**: Full range (1.0)

**Explanation**:
- 0.8: Only detect changes in first 80% of data
- Prevents fitting changepoints to end effects
- Especially important for short series

## How Prophet Works

### Step 1: Decomposition
Prophet decomposes the time series into:
```
y(t) = g(t) + s(t) + h(t) + ε(t)

Where:
- g(t) = Trend (piecewise linear)
- s(t) = Seasonality (Fourier series)
- h(t) = Holiday effects (not used here)
- ε(t) = Noise
```

### Step 2: Trend Modeling
Uses piecewise linear trend with automatic changepoint detection:
```
Trend = base_trend + changepoints + automatic_regressors
```

For short series:
- Fewer changepoints detected (due to low prior)
- Smoother trend lines
- Less responsive to noise

### Step 3: Seasonality Modeling
Usually uses Fourier series to model repeating patterns
**Disabled for 130-day data** because:
- No complete annual cycles
- No complete weekly cycles
- Risk of overfitting

### Step 4: Uncertainty Intervals
Prophet computes predictive intervals by:
1. Sampling future trend scenarios
2. Sampling noise from residuals
3. Computing percentiles (80th and 95th)

Formula:
```
Upper/Lower = Forecast ± (zscore * std_error)
Where zscore depends on interval_width (0.84 for 80%)
```

## Integration with Technical Scoring

### Forecast Output Format
```python
results['prophet'] = {
    'forecast': array([f1, f2, f3, f4, f5]),           # 5-day forecast
    'lower_bound': array([l1, l2, l3, l4, l5]),        # Lower 80% CI
    'upper_bound': array([u1, u2, u3, u4, u5]),        # Upper 80% CI
    'fitted_values': array([...]),                       # Historical fit
    'trend': array([...])                                # Trend component
}
```

### Scoring Logic (in calculate_technical_score)

```python
# Extract forecast values for scoring
if results.get('prophet'):
    prophet_forecast = results['prophet']['forecast'][-1]  # Last forecast
    current_score = scores[-1]  # Most recent score
    
    # Calculate forecast direction and confidence
    forecast_direction = 'bullish' if prophet_forecast > current_score else 'bearish'
    
    # Calculate confidence interval width
    ci_width = results['prophet']['upper_bound'][-1] - results['prophet']['lower_bound'][-1]
    ci_percent = (ci_width / current_score) * 100
    
    # Score assignment
    if forecast_direction == 'bullish' and ci_percent < 10:
        score_details['prophet'] = "Prophet Strong Bullish (+1.5)"
        prophet_score = 1.5
    elif forecast_direction == 'bullish':
        score_details['prophet'] = "Prophet Bullish (+1)"
        prophet_score = 1.0
    elif forecast_direction == 'neutral':
        score_details['prophet'] = "Prophet Neutral (+0.3)"
        prophet_score = 0.3
    else:
        score_details['prophet'] = "Prophet Bearish (0)"
        prophet_score = 0
```

## Comparison: Prophet vs ARIMA vs Exponential Smoothing

### ARIMA (AutoRegressive Integrated Moving Average)
**Pros**:
- Well-established statistical method
- Good for simple univariate forecasting
- Generates confidence intervals

**Cons**:
- Requires parameter selection (p, d, q)
- No automatic seasonality detection
- Can overfit on short series
- Struggles with structural breaks

**In MSAS**: Grid search for optimal (p,d,q), select by AIC

### Exponential Smoothing
**Pros**:
- Simple and interpretable
- Works with limited data
- Fast computation
- Good for smooth trends

**Cons**:
- Assumes homogeneous dynamics
- Poor with structural breaks
- Limited flexibility

**In MSAS**: Simple ES (4+ pts) or Holt's (10+ pts)

### Prophet
**Pros**:
- Handles multiple seasonalities
- Automatic changepoint detection
- Robust to outliers and missing data
- Good defaults for short series
- Built-in uncertainty quantification
- Interpretable components (trend, seasonality)

**Cons**:
- Slower than ARIMA/ES
- Can be over-complex for very simple data
- Less traditional in quantitative finance

**In MSAS**: Added as third forecast method with scoring weight

## Visualizations with Prophet

### Individual Forecasting Plot
Filename: `{ticker}_06_forecasting.png`

```
Shows:
- Blue line: Historical scores
- Red line: ARIMA forecast with circles
- Green line: Prophet forecast with diamonds
- Green shaded area: Prophet 80% confidence interval
- Vertical dashed line: Historical/forecast boundary
```

### Comprehensive Dashboard
Subplot 6 (5-Day Forecast):
```
Shows same as above but condensed
Emphasizes Prophet forecast and uncertainty bands
```

### Technical Ranking CSV
Columns include:
- `Prophet`: Forecast signal text (e.g., "Prophet Strong Bullish (+1.5)")
- Integration with other technical indicators

## Configuration for Different Data Sizes

### For 130 Days (Current)
```python
Prophet(
    changepoint_prior_scale=0.05,      # Very conservative
    seasonality_prior_scale=0.1,        # No seasonality
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
    interval_width=0.80,
    changepoint_range=0.8
)
```

### For 200+ Days
```python
Prophet(
    changepoint_prior_scale=0.1,        # More flexible
    seasonality_prior_scale=0.5,        # Can detect seasonality
    yearly_seasonality=False,           # Still no yearly
    weekly_seasonality=True,            # Can detect weekly
    daily_seasonality=False,
    interval_width=0.95,                # Wider intervals
    changepoint_range=0.9
)
```

### For 1+ Year
```python
Prophet(
    changepoint_prior_scale=0.05,       # Standard
    seasonality_prior_scale=10,         # Strong seasonality
    yearly_seasonality=True,            # Detect yearly
    weekly_seasonality=True,            # Detect weekly
    daily_seasonality=False,
    interval_width=0.95,
    changepoint_range=0.8,
    seasonality_mode='additive'         # or 'multiplicative'
)
```

## Troubleshooting Prophet Issues

### Issue: "Prophet failed for AAPL: ..."
**Cause**: Usually insufficient data (< 10 points)
**Solution**: Check minimum data requirements
**Code Fallback**: System continues with ARIMA/ES

### Issue: Forecast is flat line
**Possible causes**:
- No trend in recent data
- Excessive smoothing
- All changepoints disabled

**Solution**: Check if real data has trend

### Issue: Confidence intervals too narrow
**Cause**: ci_width parameter or low noise
**Solution**: Increase interval_width (0.8 → 0.95)

### Issue: Convergence warnings
**Cause**: Poor data quality or conflicts with parameters
**Solution**: Prophet still works, warnings can be ignored

## Performance Notes

### Speed
- ARIMA: ~0.1-0.2 seconds per ticker
- Exponential Smoothing: ~0.05-0.1 seconds
- Prophet: ~2-5 seconds per ticker (sampling-based)
- **Total for 100 tickers**: ~5-10 minutes additional

### Memory
- ARIMA: Minimal
- Exponential Smoothing: Minimal
- Prophet: ~50-100 MB per model
- **Total for 100 tickers**: ~5-10 MB

### Output Size
- Plots: ~200 KB per ticker
- CSVs: ~10 KB per ticker
- **Total for 100 tickers**: ~20 MB output

## References

1. **Official Prophet Documentation**: https://facebook.github.io/prophet/
2. **Prophet Paper**: "Forecasting at Scale" (Facebook Research)
3. **MSAS Implementation**: PROPHET_INTEGRATION.md
4. **Configuration Guide**: CODEBASE_EXPLORATION.md (Section 4 & 5)

---

**Last Updated**: November 12, 2025
**Prophet Version**: Latest stable
**MSAS Integration**: Active in stock_score_trend_technical_03.py
