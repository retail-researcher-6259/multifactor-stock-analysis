# MSAS Codebase Documentation Index

## Overview
This folder contains the **Multifactor Stock Analysis System (MSAS)** - a comprehensive Python-based quantitative stock analysis framework with PyQt5 GUI. 

**Key Recent Updates** (November 2025):
- Prophet forecasting integrated into Score Trend Analysis
- OptionsFlow factor added (replaces Credit factor)
- Re-optimized regime-specific weights
- Ready for analysis with 130+ days of historical data

---

## Documentation Files

### 1. **CODEBASE_EXPLORATION.md** (21 KB) - COMPREHENSIVE GUIDE
**What it covers**:
- All 5 main entry points and how to run them
- Detailed breakdown of Prophet forecasting integration
- Recent changes: Prophet implementation + OptionsFlow factor
- Complete data requirements and storage structure
- Configuration parameters and thresholds
- Deep explanation of Score Trend Analysis workflow
- Component summary and data flow diagrams
- 130-day compatibility analysis

**Best for**: Understanding the entire system architecture, learning what changed, data flow details

---

### 2. **QUICK_START_130DAYS.md** (8.5 KB) - PRACTICAL GUIDE
**What it covers**:
- Visual file structure overview
- Key configuration parameters
- Scoring weights summary (3 regimes)
- Step-by-step Score Trend Analysis workflow
- Practical usage instructions (3 options: GUI, Script, Direct)
- Specific file modifications needed for 130-day analysis
- Quick troubleshooting checklist

**Best for**: Getting started quickly, making modifications, running analyses

---

### 3. **PROPHET_TECHNICAL_DETAILS.md** (13 KB) - TECHNICAL DEEP DIVE
**What it covers**:
- What Prophet is and why it's used
- Complete implementation code breakdown
- All 5 hyperparameters explained in detail
- How Prophet works (4-step decomposition)
- Scoring logic with confidence intervals
- Comparison with ARIMA and Exponential Smoothing
- Visualization examples
- Configuration recommendations for different data sizes
- Troubleshooting common issues
- Performance benchmarks

**Best for**: Understanding Prophet technically, tuning parameters, debugging issues

---

### 4. **PROPHET_INTEGRATION.md** (6.3 KB) - ORIGINAL IMPLEMENTATION DOCS
**What it covers**:
- Overview of Prophet integration
- Why Prophet is suited for the system
- Installation instructions
- Advantages over ARIMA (comparison table)
- Output examples
- Performance considerations
- Validation steps

**Best for**: Understanding integration rationale, installation, validation

---

### 5. **README.md** (10 KB) - PROJECT OVERVIEW
**What it covers**:
- System key features (6 major components)
- Quick start guide
- Project structure
- Workflow overview
- Performance metrics
- Known issues
- Future enhancements

**Best for**: High-level project understanding, feature overview

---

## Quick Navigation by Task

### "I want to understand the entire system"
1. Start: **README.md** (overview)
2. Then: **CODEBASE_EXPLORATION.md** (detailed guide)
3. Reference: **QUICK_START_130DAYS.md** (practical map)

### "I want to run the system with 130 days of data"
1. Start: **QUICK_START_130DAYS.md** (follow steps)
2. Reference: **CODEBASE_EXPLORATION.md** (Section 7-8)
3. If issues: **QUICK_START_130DAYS.md** (troubleshooting)

### "I want to understand Prophet integration"
1. Start: **PROPHET_TECHNICAL_DETAILS.md** (technical deep dive)
2. Overview: **PROPHET_INTEGRATION.md** (rationale)
3. Code: Line 376-420 in `src/trend_analysis/stock_score_trend_technical_03.py`

### "I want to understand recent scoring changes"
1. Start: **CODEBASE_EXPLORATION.md** (Section 2)
2. Deep dive: **QUICK_START_130DAYS.md** (weights summary)
3. Code: `src/scoring/stock_Screener_MultiFactor_25_new.py` (lines 80-123)

### "I want to modify system for 130-day analysis"
1. Start: **QUICK_START_130DAYS.md** (modifications section)
2. Details: **CODEBASE_EXPLORATION.md** (Section 7)
3. Code: Follow specific file locations and line numbers

---

## Key Files to Know

### Entry Points
```
MSAS_UI.py                          # Main GUI (python MSAS_UI.py)
src/scoring/stock_Screener_MultiFactor_25_new.py      # Scoring engine
src/trend_analysis/stock_score_trend_technical_03.py  # Trend analysis
```

### Configuration
```
config/Buyable_stocks_template.txt  # 998 stocks
src/scoring/stock_Screener_MultiFactor_25_new.py      # REGIME_WEIGHTS, FUND_THRESHOLDS
src/trend_analysis/stock_score_trend_technical_03.py  # Prophet configuration
```

### Widget Orchestration
```
src/classes/multifactor_scoring_widget_new.py         # Scoring UI
src/classes/score_trend_analysis_widget.py            # Trend analysis UI
```

---

## Data Flow Summary

```
User Input (Tickers, Date Range, Regime)
         ↓
[MULTIFACTOR SCORING]
  → Fetch data (yahooquery, 730 days default)
  → Calculate 50+ factors
  → Apply regime weights
  → Output: top_ranked_stocks_{regime}_{MMDD}.csv
         ↓
[SCORE TREND ANALYSIS] ← Reads ranked CSVs
  → Load score history
  → Technical analysis:
     - Regression, MA, MACD, RSI, Bollinger Bands, ADX
  → Forecasting:
     - ARIMA (grid search)
     - Exponential Smoothing
     - Prophet (NEW - with confidence bands)
  → Generate 7 plots per ticker
  → Output: technical_ranking CSVs + visualizations
         ↓
[DYNAMIC PORTFOLIO]
  → Construct portfolios from top scores
  → Backtest with walk-forward analysis
         ↓
[PORTFOLIO OPTIMIZATION]
  → Apply HRP/HERC/MHRP/NCO methods
  → Output: Allocation weights
```

---

## Key Metrics & Thresholds

### Minimum Data Requirements (130 days compatible)
- Prophet/ARIMA/ES: 10 points
- Bollinger Bands: 20 points
- MACD: 26 points
- Technical indicators: Varies (3-26)

### Scoring System
- Total score scale: 0-120
- 3 market regimes: Crisis/Bear, Steady Growth, Strong Bull
- 12 main factors, 50+ subfactors
- Prophet adds: 0 to +1.5 points

### Processing Speed
- Per ticker: ~2-5 seconds (Prophet slowest)
- 100 tickers: ~5-10 minutes
- 1000 tickers: ~50-100 minutes

---

## Most Important Changes (November 2025)

### 1. Prophet Forecasting Added
- **File**: `src/trend_analysis/stock_score_trend_technical_03.py` (lines 376-420)
- **Impact**: Adds uncertainty quantification, automatic trend detection
- **Scoring**: Strong Bullish (+1.5) to Bearish (0)
- **Visualization**: Green forecast line with 80% confidence bands

### 2. Credit Factor Removed → OptionsFlow Added
- **File**: `src/scoring/stock_Screener_MultiFactor_25_new.py` (line 876, 1933)
- **Old**: get_credit_score (Altman Z-Score based)
- **New**: get_options_flow_score (Put/Call ratios from options chains)
- **Impact**: Better institutional sentiment detection, instant metrics

### 3. Weights Re-optimized
- **File**: `src/scoring/stock_Screener_MultiFactor_25_new.py` (lines 80-123)
- **Impact**: Better risk-adjusted returns across 3 regimes
- **Details**: See QUICK_START_130DAYS.md (weights table)

---

## System Requirements

### Python Packages (from requirements.txt)
- Core: numpy, pandas, scipy, scikit-learn
- Time Series: prophet, statsmodels
- Data: yfinance, yahooquery, curl_cffi
- Technical: ta, ta-lib
- GUI: PyQt5
- Optimization: riskfolio-lib, cvxpy

### External Libraries
- TA-Lib C library (required for ta-lib package)
- Linux: `sudo apt-get install ta-lib`
- macOS: `brew install ta-lib`
- Windows: Download .whl from lfd.uci.edu

---

## Troubleshooting Quick Links

| Problem | Solution | Doc |
|---------|----------|-----|
| "Prophet not found" | `pip install prophet pystan` | PROPHET_INTEGRATION.md |
| Not enough score data | Check Ranked_Lists folder dates | QUICK_START_130DAYS.md |
| OptionsFlow missing | Options may unavailable, defaults to 0.5 | CODEBASE_EXPLORATION.md |
| 130-day lookback | Change line in stock_Screener_MultiFactor_25_new.py | QUICK_START_130DAYS.md |
| Prophet slow | Expected 2-5 sec/ticker, process in parallel | PROPHET_TECHNICAL_DETAILS.md |

---

## Document Statistics

| Document | Size | Sections | Best For |
|----------|------|----------|----------|
| CODEBASE_EXPLORATION.md | 21 KB | 8 + 142 lines | Complete understanding |
| QUICK_START_130DAYS.md | 8.5 KB | 12 sections | Quick execution |
| PROPHET_TECHNICAL_DETAILS.md | 13 KB | 12 sections | Technical mastery |
| PROPHET_INTEGRATION.md | 6.3 KB | 8 sections | Implementation rationale |
| README.md | 10 KB | Multiple | Project overview |

**Total Documentation**: 58.8 KB of comprehensive guides

---

## Version Information

**System**: MSAS (Multifactor Stock Analysis System)
**As of**: November 12, 2025
**Latest Commits**:
- 4384eea: Add Prophet forecasting to Score Trend Analysis system
- a73a724: Revised scoring functions, removed Credit, added OptionsFlow

**Prophet Version**: Latest stable (pip install prophet)
**Python Version**: 3.8+ (recommended 3.9+)
**Data Range Tested**: 130+ days compatible

---

## Getting Help

1. **System Overview**: See README.md + CODEBASE_EXPLORATION.md (Section 1)
2. **Data Issues**: See CODEBASE_EXPLORATION.md (Section 3)
3. **Configuration**: See QUICK_START_130DAYS.md (Key Data Configurations)
4. **Prophet Questions**: See PROPHET_TECHNICAL_DETAILS.md (all sections)
5. **Implementation Details**: See code files with line number references in docs
6. **Troubleshooting**: See QUICK_START_130DAYS.md (Troubleshooting section)

---

**Happy analyzing!** Start with README.md if new to the project, or jump to QUICK_START_130DAYS.md if you want to run it immediately.
