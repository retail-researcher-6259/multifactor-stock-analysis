# Quick Start Guide for 130-Day Analysis

## System Overview
- **GUI Entry Point**: `python MSAS_UI.py`
- **Core Scoring**: `src/scoring/stock_Screener_MultiFactor_25_new.py`
- **Score Trend Analysis (with Prophet)**: `src/trend_analysis/stock_score_trend_technical_03.py`

## File Structure
```
/home/user/multifactor-stock-analysis/
├── MSAS_UI.py                              # Main GUI
├── CODEBASE_EXPLORATION.md                 # Detailed documentation (NEW)
├── PROPHET_INTEGRATION.md                  # Prophet implementation details
├── requirements.txt                         # Dependencies (includes prophet)
│
├── config/
│   ├── Buyable_stocks_template.txt         # 998-stock ticker list
│   └── marketstack_config.json             # Optional API config
│
├── src/
│   ├── scoring/
│   │   └── stock_Screener_MultiFactor_25_new.py  # Main scoring engine
│   │       ├── REGIME_WEIGHTS (3 regimes with OptionsFlow)
│   │       └── Main factors: Value, Quality, Growth, Financial Health,
│   │           Technical, Liquidity, OptionsFlow, Insider, Size, etc.
│   │
│   ├── trend_analysis/
│   │   ├── stock_score_trend_technical_03.py     # Prophet + ARIMA + ES
│   │   │   └── perform_statistical_forecasting() (line 304-421)
│   │   └── stock_score_trend_technical_single_enhanced.py
│   │
│   ├── classes/                            # PyQt5 widgets
│   │   ├── multifactor_scoring_widget_new.py
│   │   └── score_trend_analysis_widget.py
│   │
│   ├── regime_detection/                   # Market state classification
│   ├── portfolio_selection/                # Portfolio construction
│   └── portfolio_optimization/             # HRP/HERC/MHRP/NCO
│
└── output/
    ├── Ranked_Lists/{regime_name}/         # Daily score CSVs
    │   └── top_ranked_stocks_{regime}_{MMDD}.csv
    └── Score_Trend_Analysis_Results/       # Technical analysis results
        └── {regime_name}/
            ├── stability_analysis_results_{MMDD}.csv
            └── technical_plots/{ticker}/
                ├── {ticker}_01_regression.png
                ├── {ticker}_02_moving_averages.png
                ├── {ticker}_03_momentum.png
                ├── {ticker}_04_bollinger_bands.png
                ├── {ticker}_05_trend_strength.png
                └── {ticker}_06_forecasting.png (INCLUDES PROPHET)
```

## Key Data Configurations

### 1. Minimum Data Requirements
| Component | Minimum Points |
|-----------|----------------|
| Prophet/ARIMA/ES | 10 |
| Bollinger Bands | 20 |
| MACD | 26 |
| Technical Indicators | Varies (3-26) |

### 2. Default Lookback Periods
```python
# In stock_Screener_MultiFactor_25_new.py:
INSIDER_LOOKBACK_DAYS = 90
lookback_days = 365 * 2  # Can change to 130

# Score trend analysis uses CSV file dates (MMDD format)
start_date = "0601"  # June 1st
end_date = "0621"    # June 21st
```

### 3. Prophet Configuration (Already Optimized for Limited Data)
```python
Prophet(
    changepoint_prior_scale=0.05,      # Conservative
    seasonality_prior_scale=0.1,        # No seasonality
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
    interval_width=0.80,                # 80% confidence bands
    changepoint_range=0.8               # First 80% only
)
```

## Scoring Weights Summary

### Three Market Regimes:
1. **Steady_Growth**: Balanced risk/reward environment
2. **Strong_Bull**: Bullish conditions (momentum favored)
3. **Crisis_Bear**: Risk-off environment (quality emphasized)

### Key Factor Weights:
| Factor | Steady Growth | Strong Bull | Crisis Bear |
|--------|---|---|---|
| **momentum** | 38.8 | 35.0 | 46.6 |
| **technical** | 19.5 | 22.0 | 25.9 |
| **value** | 17.5 | 15.0 | 11.0 |
| **options_flow** | 11.0 | 12.0 | 2.9 |
| **liquidity** | 7.6 | 12.0 | 4.7 |
| **insider** | 5.1 | 2.0 | 1.8 |
| **quality** | -6.8 | -6.0 | -6.0 |
| **stability** | -4.1 | -6.0 | -4.5 |

## Score Trend Analysis with Prophet Workflow

### Step 1: Load Score History
- Reads ranked score CSVs from date range
- Builds {ticker: {indices, scores, dates}} dictionary

### Step 2: Technical Analyses
- Regression: Linear, Polynomial (R² scoring)
- Moving Averages: SMA 5/10/20, EMA 5/10, Crossovers
- Momentum: RSI(14), MACD, ROC(10)
- Bollinger Bands: %B position, Bandwidth
- Trend Strength: ADX, Directional Indicators (+DI/-DI)

### Step 3: Forecasting (NEW)
**ARIMA**: Grid search (p:0-2, d:0-1, q:0-2), select by lowest AIC
**Exp Smoothing**: Simple (4+ pts) or Holt's (10+ pts)
**Prophet** (NEW):
- Forecast 5 days ahead
- 80% confidence intervals
- Detect changepoints automatically
- Score: Strong Bullish (+1.5) → Bullish (+1) → Neutral (+0.3) → Bearish (0)

### Step 4: Technical Score Calculation
- Combines all technical indicators
- Weights Prophet forecast signals (+0 to +1.5 points)
- Returns comprehensive ranking CSV

### Step 5: Visualization
- 7 individual plots per ticker
- Prophet shown with green line + shaded confidence bands
- Comprehensive 6-subplot dashboard

## How to Run for 130-Day Analysis

### Option 1: Using GUI
```bash
python MSAS_UI.py
# Click "Score Trend Analysis" tab
# Configure date range (within 130 days)
# Select regime and run analysis
```

### Option 2: Direct Script
```bash
# Requires pre-existing ranked score CSVs
python src/trend_analysis/stock_score_trend_technical_03.py

# Or for single ticker
python src/trend_analysis/stock_score_trend_technical_single_enhanced.py
```

### Option 3: Generate New Ranked Lists (Optional)
```bash
# Modify lookback_days in stock_Screener_MultiFactor_25_new.py:
# Change: lookback_days = 365 * 2
# To:     lookback_days = 130

python src/scoring/stock_Screener_MultiFactor_25_new.py
```

## Key Modifications for 130-Day Window

### File 1: `src/scoring/stock_Screener_MultiFactor_25_new.py`
**Line ~85**: Change lookback from 2 years to 130 days
```python
# Before:
hist = yahooquery_hist(tk, years=2)

# After:
hist = yahooquery_hist(tk, days=130)
```

### File 2: `src/trend_analysis/stock_score_trend_technical_03.py`
**No changes needed** - Prophet already configured for limited data

### File 3: Widget configuration
**File**: `src/classes/score_trend_analysis_widget.py` (line ~59)
```python
# Set date range within 130-day window:
start_date="0701",  # July 1st
end_date=datetime.now().strftime("%m%d")
```

## Important Notes for 130-Day Analysis

✓ **Works Well**:
- Short-term momentum analysis
- Technical indicators (RSI, MACD, Bollinger Bands)
- Score stability assessment
- Prophet forecasting (conservative settings)
- Institutional options flow (snapshot-based)

✗ **Limited**:
- Seasonal patterns (need 6+ months)
- Multi-year trends (incomplete)
- Earnings seasonality (quarterly)
- Credit scores (removed, now OptionsFlow)

## Data Sources

| Source | Purpose | Fallback |
|--------|---------|----------|
| Yahoo Finance | Prices, fundamentals, options | yahooquery wrapper |
| FRED API | Macroeconomic data | Manual input |
| SEC EDGAR | Insider trading | 90-day recent data |

## Output Files Generated

After running Score Trend Analysis:
```
output/Score_Trend_Analysis_Results/{regime_name}/
├── stability_analysis_results_{MMDD}.csv        # Main rankings
└── technical_plots/
    └── {ticker}/
        ├── {ticker}_01_regression.png
        ├── {ticker}_02_moving_averages.png
        ├── {ticker}_03_momentum.png
        ├── {ticker}_04_bollinger_bands.png
        ├── {ticker}_05_trend_strength.png
        ├── {ticker}_06_forecasting.png          # Includes Prophet
        ├── {ticker}_07_technical_ranking.png
        └── {ticker}_technical_ranking.csv        # Results CSV
```

## Troubleshooting

**Q: Prophet installation issues?**
```bash
pip install prophet pystan
```

**Q: Not enough score data?**
- Ensure ranked score CSVs exist in output/Ranked_Lists/{regime}/
- Check date range covers 130+ days of available scores

**Q: OptionsFlow score missing?**
- Options chains may not be available for all stocks
- System defaults to 0.5 (neutral) if options data unavailable

**Q: Technical analysis skipped for ticker?**
- Likely insufficient score history (<3 points for regression)
- Need more snapshot dates in ranked score files

---

**Last Updated**: November 12, 2025
**System Version**: MSAS with Prophet Integration + OptionsFlow Factor
**Tested Data Range**: 130+ days compatible
