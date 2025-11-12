# Multifactor Stock Analysis System (MSAS) - Comprehensive Codebase Exploration

## 1. MAIN ENTRY POINTS AND HOW TO RUN THE SYSTEM

### Primary Entry Point: GUI Application
**File**: `/home/user/multifactor-stock-analysis/MSAS_UI.py`
- **Launch Command**: `python MSAS_UI.py`
- **Interface**: PyQt5-based GUI with tabbed navigation
- **Tabs**:
  1. **Regime Detection** - Market state classification (Crisis/Bear, Steady Growth, Strong Bull)
  2. **Multifactor Scoring** - Stock ranking with regime-specific weights
  3. **Score Trend Analysis** - Technical analysis with Prophet forecasting (NEW)
  4. **Portfolio Selection** - Dynamic portfolio construction
  5. **Portfolio Optimization** - HRP/HERC/MHRP/NCO allocation methods

### Alternative Script Entry Points

#### A. Multifactor Scoring (Core System)
**File**: `/home/user/multifactor-stock-analysis/src/scoring/stock_Screener_MultiFactor_25_new.py`
```python
python src/scoring/stock_Screener_MultiFactor_25_new.py

# Functions:
- main(regime="Steady_Growth", mode="daily/historical", target_date=None, progress_callback=None)
- run_daily_scoring(regime="Steady_Growth", progress_callback=None)
- run_scoring_for_regime(regime_name, progress_callback=None)
```

#### B. Score Trend Analysis with Prophet (NEW)
**File**: `/home/user/multifactor-stock-analysis/src/trend_analysis/stock_score_trend_technical_03.py`
- Analyzes score stability and technical patterns
- Includes Prophet forecasting, ARIMA, and Exponential Smoothing
- Requires ranked score CSV files as input

#### C. Single Ticker Technical Analysis
**File**: `/home/user/multifactor-stock-analysis/src/trend_analysis/stock_score_trend_technical_single_enhanced.py`
- Enhanced single ticker analyzer
- Uses ranked approach from stock_score_trend_technical_03.py

## 2. RECENT CHANGES: PROPHET FORECASTING & SCORING UPDATES

### Latest Commit: Add Prophet Forecasting (4384eea - Nov 10, 2025)

**Files Modified**:
1. `src/trend_analysis/stock_score_trend_technical_03.py`
2. `src/trend_analysis/stock_score_trend_technical_single_enhanced.py`
3. `PROPHET_INTEGRATION.md` (new documentation)

**Key Features Added**:
- **Prophet Forecasting Integration** (lines 376-420)
  - Conservative hyperparameters optimized for short time series (50+ data points)
  - Parameters:
    - `changepoint_prior_scale=0.05` (less flexible, prevents overfitting)
    - `seasonality_prior_scale=0.1` (minimal seasonal variation)
    - `yearly/weekly/daily_seasonality=False` (disabled for short series)
    - `interval_width=0.80` (80% confidence intervals)
    - `changepoint_range=0.8` (only fit changepoints in first 80%)

- **Enhanced Visualizations**
  - Prophet forecast line (green with diamond markers)
  - 80% confidence intervals (shaded green area)
  - Comparison with ARIMA and Exponential Smoothing

- **Technical Score Integration**
  - Prophet Strong Bullish: +1.5 points (high confidence, narrow intervals < 10%)
  - Prophet Bullish: +1.0 point (wider confidence interval)
  - Prophet Neutral: +0.3 points (flat forecast)
  - Prophet Bearish: 0 points

### Previous Commit: Scoring Overhaul (a73a724 - Nov 10, 2025)

**Major Updates**:
1. **Removed Credit Factor** (get_credit_score function)
   - Was using Altman Z-Score components
   - Also evaluated debt service capacity and interest coverage

2. **Added OptionsFlow Factor** (get_options_flow_score function)
   - Uses instant options chain data (current snapshot)
   - Analyzes Put/Call Ratios (OI and Volume based)
   - Detects near-the-money concentration
   - Identifies large institutional positions
   - Returns score 0-1 (0.5=neutral, >0.5=bullish, <0.5=bearish)

3. **Re-optimized Weights** (REGIME_WEIGHTS dictionary)

#### Weight Changes by Regime:

**Steady_Growth Regime**:
- momentum: 38.8 (was varied)
- technical: 19.5
- value: 17.5
- **options_flow: 11.0** (NEW)
- liquidity: 7.6
- quality: -6.8
- insider: 5.1
- size: 4.5
- stability: -4.1
- financial_health: 3.8
- growth: 2.5
- carry: 0.6

**Strong_Bull Regime**:
- momentum: 35.0
- technical: 22.0
- value: 15.0
- **options_flow: 12.0** (NEW)
- liquidity: 12.0
- financial_health: 8.0
- growth: 5.0
- size: 3.0
- insider: 2.0
- carry: -2.0
- stability: -6.0
- quality: -6.0

**Crisis_Bear Regime**:
- momentum: 46.6
- technical: 25.9
- value: 11.0
- growth: 8.2
- quality: -6.0
- financial_health: 5.9
- liquidity: 4.7
- stability: -4.5
- size: 3.5
- **options_flow: 2.9** (NEW)
- insider: 1.8
- carry: 0.0

## 3. DATA REQUIREMENTS & WHERE DATA IS STORED/LOADED

### Minimum Data Requirements

**Score History Data**:
- **Minimum 10 data points** for Prophet, ARIMA, and Exponential Smoothing
- **Minimum 3 data points** for regression analysis
- **Minimum 5 data points** for moving averages
- **Minimum 14 data points** for RSI
- **Minimum 20 data points** for Bollinger Bands
- **Minimum 26 data points** for MACD

### Historical Data Fetching

**Default Lookback Period**: 730 days (2 years)
- Function: `fetch_historical_data_for_date(ticker, target_date, lookback_days=365*2)`
- Location: `/src/scoring/stock_Screener_MultiFactor_25_new.py` (line ~371)
- Data Source: yahooquery with curl_cffi wrapper (bypasses rate limiting)

**Limited Data Handling**:
- Line 2705 in stock_Screener_MultiFactor_25_new.py:
  ```
  "# Fallback: Use as much data as available (minimum 180 days)"
  ```
- Can operate with less than 180 days if needed

### Data Storage Locations

1. **Ranked Lists (Daily Scores)**
   - Directory: `/output/Ranked_Lists/{regime_name}/`
   - Format: `top_ranked_stocks_{regime_name}_{MMDD}.csv`
   - Columns: Ticker, Score, Company factors (Value, Technical, Momentum, etc.)

2. **Score Trend Analysis Results**
   - Directory: `/output/Score_Trend_Analysis_Results/{regime_name}/`
   - Contains:
     - `stability_analysis_results_{MMDD}.csv` (main results)
     - `technical_plots/{ticker}/` (individual plots and rankings)
     - Technical analysis plots: regression, MA, momentum, Bollinger Bands, forecasting

3. **Regime Detection Data**
   - Directory: `/output/Regime_Detection_Results/`
   - Files:
     - `regime_model.pkl` (trained HMM model)
     - `regime_periods.csv` (historical regime periods)
     - `historical_market_data.csv` (market data used for training)

4. **Portfolio Results**
   - `/output/Dynamic_Portfolio_Selection/`
   - `/output/Portfolio_Optimization_Results/`

### Data Input Files

**Ticker Lists**:
- Location: `/config/Buyable_stocks_template.txt`
- Format: One ticker per line (998 stocks across all sectors)
- Usage: Can be customized per analysis

**API Configuration**:
- `/config/marketstack_config.json` (optional Marketstack API)

### External Data Sources

1. **Yahoo Finance** (Primary)
   - Historical prices, fundamentals, info
   - Options chains (for OptionsFlow factor)

2. **yahooquery Library**
   - Enhanced Yahoo Finance API access
   - Uses curl_cffi wrapper to avoid rate limits

3. **FRED** (Federal Reserve Data)
   - Macroeconomic indicators
   - API Key: `a6472bcc951dc72f091984a09a36fc9e`
   - Series: GDP (quarterly)

4. **SEC EDGAR** (Insider Trading)
   - Lookback period: 90 days (INSIDER_LOOKBACK_DAYS)

## 4. CONFIGURATION FILES & MINIMUM DATA REQUIREMENTS

### Primary Configuration

**File**: `/src/scoring/stock_Screener_MultiFactor_25_new.py` (lines 64-123)

**Key Configuration Parameters**:

```python
# Data lookback periods
INSIDER_LOOKBACK_DAYS = 90
TOP_N = 20  # Number of top stocks to return

# Default data lookback for historical data
lookback_days = 365 * 2  # 2 years (730 days)

# Fallback minimum
MINIMUM_DATA_POINTS = 180  # Can fall back to this

# Scoring weights (3 regime-specific configurations)
REGIME_WEIGHTS = {
    "Steady_Growth": {...},
    "Strong_Bull": {...},
    "Crisis_Bear": {...}
}

# Fundamental thresholds
FUND_THRESHOLDS = {
    "pe": 20,
    "pb": 2.5,
    "ev_to_ebitda": 12,
    "roe": 0.12,
    "roic": 0.10,
    "gross_margin": 0.35,
    "de_ratio": 100,
    "current_ratio": 1.5,
    "rev_growth": 0.05,
    # ... and more
}
```

### Score Trend Analysis Configuration

**File**: `/src/trend_analysis/stock_score_trend_technical_03.py`

**Initialization Parameters**:
```python
class StockScoreTrendAnalyzerTechnical:
    def __init__(self, 
                 csv_directory="./ranked_lists",
                 start_date="0611",           # MMDD format
                 end_date="0621",             # MMDD format
                 sigmoid_sensitivity=5)       # For scaling
```

**Data Requirement Validation Points**:
- Line 307-308: `if not data or len(data['scores']) < 10: return None`
  - Prophet, ARIMA, Exp Smoothing require minimum 10 points
- Line 241: `if not data or len(data['scores']) < 20: return None`
  - Bollinger Bands require minimum 20 points
- Line 130-131: `if not data or len(data['indices']) < 3: return None`
  - Regression requires minimum 3 points

### Prophet-Specific Configuration

**File**: `/PROPHET_INTEGRATION.md` and line 376-419 in stock_score_trend_technical_03.py

**Conservative Parameters for Limited Data**:
```python
Prophet(
    changepoint_prior_scale=0.05,      # Conservative (prevent overfitting)
    seasonality_prior_scale=0.1,        # Minimal seasonal detection
    yearly_seasonality=False,           # Disabled
    weekly_seasonality=False,           # Disabled
    daily_seasonality=False,            # Disabled
    interval_width=0.80,                # 80% confidence intervals
    changepoint_range=0.8               # First 80% of data only
)
```

**Recommendations for More Data** (from PROPHET_INTEGRATION.md):
- With 200+ data points: Increase `changepoint_prior_scale=0.1`, `seasonality_prior_scale=0.5`
- With 6+ months: Enable weekly seasonality for trading week patterns
- With market data: Add exogenous variables (sector performance, market index)

## 5. HOW THE SCORE TREND ANALYSIS SYSTEM WORKS (WITH PROPHET)

### Architecture Overview

```
Ranked Score CSV Files (daily scores)
         ↓
  Load Score History (build_score_history)
         ↓
  Perform Analyses (in parallel):
  ├─ Regression Analysis (linear, polynomial)
  ├─ Moving Averages (SMA 5/10/20, EMA 5/10)
  ├─ Momentum Indicators (RSI, MACD, ROC)
  ├─ Bollinger Bands
  ├─ Trend Strength (ADX, Directional Indicators)
  └─ Statistical Forecasting:
     ├─ ARIMA (grid search on p,d,q)
     ├─ Exponential Smoothing (Holt's method)
     └─ Prophet (NEW - Facebook's time series library)
         ↓
  Calculate Technical Score
         ↓
  Generate Visualizations & Rankings
```

### Phase 1: Data Loading

**Function**: `load_ranking_files()` (line 64-103)
- Reads CSV files from date range (MMDD format)
- Example: `top_ranked_stocks_Steady_Growth_0611.csv`
- Returns sorted list of files with dates

**Function**: `build_score_history()` (line 105-125)
- Iterates through ranked files
- Builds dictionary: `{ticker: {indices, scores, dates}}`
- Tracks temporal progression of scores

### Phase 2: Technical Analysis Functions

**1. Regression Analysis** (line 127-158)
- **Linear Regression**: `y = mx + b`
  - R² score calculated
- **Polynomial Regression**: degree 2
  - `y = ax² + bx + c`
- Returns predictions and R² values

**2. Moving Averages** (line 160-199)
- **Simple Moving Averages (SMA)**: 5, 10, 20-period
- **Exponential Moving Averages (EMA)**: 5, 10-period
- **Crossover Signals**:
  - Bullish: 5-period SMA crosses above 10-period SMA
  - Bearish: 5-period SMA crosses below 10-period SMA
  - Neutral: no cross

**3. Momentum Indicators** (line 201-236)
- **RSI (Relative Strength Index)**: 14-period
  - Overbought: RSI > 70
  - Oversold: RSI < 30
- **MACD (Moving Average Convergence Divergence)**:
  - EMA(12) - EMA(26)
  - Signal line: EMA(9) of MACD
  - Histogram: MACD - Signal
- **Rate of Change (ROC)**: 10-period percentage change

**4. Bollinger Bands** (line 238-265)
- **Middle Band**: 20-period SMA
- **Upper/Lower Bands**: ±2 standard deviations
- **%B**: Position within bands (0=lower, 1=upper)
- **Bandwidth**: (Upper - Lower) / Middle

**5. Trend Strength (ADX)** (line 267-302)
- **Plus DI / Minus DI**: Directional indicators
- **ADX (Average Directional Index)**: 14-period
  - ADX > 25: Strong trend
  - ADX < 20: Weak trend

### Phase 3: Statistical Forecasting (NEW - WITH PROPHET)

**Function**: `perform_statistical_forecasting()` (line 304-421)

**ARIMA** (line 316-347):
- Grid search: p in [0,1,2], d in [0,1], q in [0,1,2]
- Selection criterion: Lowest AIC
- Forecast periods: `min(5, max(1, len(scores) // 10))`
- Returns: Best parameters, AIC, forecast, fitted values

**Exponential Smoothing** (line 349-374):
- Simple exponential smoothing (no trend/seasonality)
  - Requires ≥4 data points
- Holt's method (with trend)
  - Requires ≥10 data points

**Prophet** (line 376-420) - NEW:
```python
# Prepare data
prophet_df = pd.DataFrame({
    'ds': pd.date_range(start='2024-01-01', periods=len(scores), freq='D'),
    'y': scores
})

# Initialize with conservative hyperparameters
prophet_model = Prophet(
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=0.1,
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
    interval_width=0.80,
    changepoint_range=0.8
)

# Fit and forecast
prophet_model.fit(prophet_df)
future = prophet_model.make_future_dataframe(periods=forecast_periods, freq='D')
forecast = prophet_model.predict(future)

# Extract: forecast values, confidence bounds, fitted values, trend
```

### Phase 4: Technical Score Calculation

**Function**: `calculate_technical_score()` (line ~1260)

Returns dict with scores for:
- **Trend Direction**: Linear regression slope
- **Trend Quality**: Polynomial R² vs Linear R²
- **SMA Position**: Current price vs SMA levels
- **Recent Cross**: SMA 5/10 crossover signal
- **MACD Signal**: Signal line direction
- **MACD Histogram**: Divergence
- **RSI Level**: Position (0-100)
- **RSI Trend**: Increasing/decreasing
- **ROC**: Rate of change
- **%B Position**: Bollinger Band position
- **BB Width**: Bollinger Band width
- **ADX Strength**: Trend strength
- **DI Direction**: Plus/Minus DI direction

**Forecasting Contributions**:
- **ARIMA**: Bullish/Bearish based on trend (+1 or 0 points)
- **Exp Smooth**: Similar scoring
- **Prophet**: Enhanced scoring
  - Strong Bullish: +1.5 points (narrow CI < 10% of current score)
  - Bullish: +1.0 point (wider CI)
  - Neutral: +0.3 points
  - Bearish: 0 points

### Phase 5: Visualization & Output

**Individual Plots** (function `create_individual_plots()`):
1. `{ticker}_01_regression.png` - Linear vs Polynomial
2. `{ticker}_02_moving_averages.png` - SMA/EMA and crossovers
3. `{ticker}_03_momentum.png` - RSI, MACD, ROC
4. `{ticker}_04_bollinger_bands.png` - Price oscillation within bands
5. `{ticker}_05_trend_strength.png` - ADX and Directional Indicators
6. `{ticker}_06_forecasting.png` - **ARIMA + Prophet + Exp Smoothing**
7. `{ticker}_07_technical_ranking.png` - Comprehensive overview

**Comprehensive Dashboard** (function `create_comprehensive_plots()`):
- 6 subplots showing all analyses
- Subplot 6: 5-Day Forecast including Prophet with confidence intervals

**Output CSV**:
- `{ticker}_technical_ranking.csv`
- Columns: Ticker, Score, Technical indicators, Forecasting methods, Total Technical Score

## 6. KEY COMPONENTS & DATA FLOW

### Scoring System Architecture

```
Input Tickers (from config/Buyable_stocks_template.txt)
         ↓
For each ticker:
  ├─ Fetch company info (yahooquery)
  ├─ Fetch 2-year historical prices
  ├─ Calculate factors:
  │  ├─ Value (P/E, P/B, EV/EBITDA)
  │  ├─ Quality (ROE, ROA, ROIC)
  │  ├─ Growth (Revenue, EPS momentum)
  │  ├─ Financial Health (Debt ratios, interest coverage)
  │  ├─ Technical (trend, momentum, volatility)
  │  ├─ Liquidity (daily volume, spreads)
  │  ├─ **OptionsFlow** (Put/Call ratios from options chain) - NEW
  │  ├─ Insider (insider buying/selling)
  │  ├─ Size (market cap)
  │  ├─ Stability (volatility, beta)
  │  ├─ Carry (dividend yield)
  │  └─ Growth (earnings growth rate)
  │
  └─ Apply regime-specific weights → Final Score (0-120)
         ↓
Output: top_ranked_stocks_{regime}_{MMDD}.csv
```

### Component Summary

| Component | File | Purpose | Data Requirements |
|-----------|------|---------|-------------------|
| **Regime Detection** | `src/regime_detection/regime_detector.py` | Market state classification using HMM | 10-year historical data |
| **Multifactor Scoring** | `src/scoring/stock_Screener_MultiFactor_25_new.py` | Stock ranking with 50+ factors | 2 years historical prices + fundamentals |
| **Score Trend Analysis** | `src/trend_analysis/stock_score_trend_technical_03.py` | Technical analysis + forecasting | Daily ranked scores (CSV files) |
| **Dynamic Portfolio** | `src/portfolio_selection/dynamic_portfolio_selector_*.py` | Portfolio construction & backtesting | Ranked scores + price history |
| **Portfolio Optimization** | `src/portfolio_optimization/portfolio_optimizer.py` | Risk parity allocation (HRP/HERC) | Portfolio tickers + return covariance |

---

## 7. RUNNING WITH LIMITED DATA (130 DAYS)

### Compatibility Check ✓

The system **CAN operate with 130 days of data**:

**Requirement Analysis**:
```
130 days of data:
  ✓ Prophet: Requires 10+ points (works great with 130 days)
  ✓ ARIMA: Requires 10+ points (works)
  ✓ Exponential Smoothing: Requires 4+ points (works)
  ✓ Technical Indicators: Min 26 for MACD (works)
  ✓ Score Trend Analysis: Operates with score snapshots
  ✓ Regime-Adaptive Weights: Already stored in configuration
```

**Potential Limitations**:
```
Limited to:
  ✗ Full 2-year lookback for some fundamental metrics
    → Fallback to available data (180+ days minimum)
  ✗ Very long-term technical patterns (1-year+ trends)
    → But sufficient for medium-term trend analysis
  ✗ Seasonal patterns (Prophet won't detect with 130 days)
    → Feature already disabled in conservative config
```

### Recommended Adjustments for 130-Day Window

1. **In stock_Screener_MultiFactor_25_new.py**:
   - Change `lookback_days = 365 * 2` to `lookback_days = 130`
   - System will adapt if data unavailable

2. **In stock_score_trend_technical_03.py**:
   - No changes needed - Prophet already configured for limited data
   - Conservative hyperparameters already in place

3. **In regime_detection systems**:
   - May need recent regime data only
   - Historical 10-year regime detection can use cached model

### Data Preparation Steps

```
1. Ensure ranked score CSV files available
   Path: output/Ranked_Lists/{regime_name}/
   Format: top_ranked_stocks_{regime}_{MMDD}.csv
   
2. Run Score Trend Analysis:
   - Loads score history from CSVs (date range configurable)
   - Analyzes technical patterns and forecasts
   - Generates plots and rankings
   
3. (Optional) Run full multifactor scoring:
   - Will fetch 130 days of historical data
   - Calculate all factors with available data
   - Generate new ranked lists
```

---

## 8. CRITICAL NOTES FOR 130-DAY ANALYSIS

### Prophet Configuration is Already Optimized ✓
- Conservative defaults prevent overfitting on short series
- 80% confidence intervals provide uncertainty quantification
- No seasonality detection (inappropriate for 130 days)
- Low `changepoint_prior_scale` (0.05) prevents false trend changes

### What Works Well
- **Technical indicators** (RSI, MACD, Bollinger Bands, ADX)
- **Short-term momentum** analysis
- **Score stability** assessment over 130 days
- **Institutional options flow** (snapshot-based)
- **Regime-adaptive weights** (pre-optimized)

### What's Limited
- **Seasonal patterns** (too short)
- **Multi-year trends** (limited history)
- **Earnings seasonality** (quarterly cycles cut off)
- **Credit factor** (removed; replaced with OptionsFlow)

### Files to Modify for 130-Day Analysis
1. `src/scoring/stock_Screener_MultiFactor_25_new.py`:
   - Line ~85: Change `yahooquery_hist(tk, years=2)` to fetch only 130 days
   - Or change lookback_days parameter if using `fetch_historical_data_for_date()`

2. `src/trend_analysis/stock_score_trend_technical_03.py`:
   - No changes needed (Prophet already configured for limited data)
   - Just configure start_date and end_date parameters

3. Configuration in widget:
   - Line ~59 in score_trend_analysis_widget.py:
     `start_date="0601", end_date=datetime.now().strftime("%m%d")`
     Can be adjusted to date range within 130 days

