# Multifactor Stock Analysis System (MSAS)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyQt5](https://img.shields.io/badge/PyQt5-GUI-green)](https://pypi.org/project/PyQt5/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success)](https://github.com/)

A comprehensive, regime-aware quantitative stock analysis and portfolio optimization system that combines machine learning, technical analysis, and modern portfolio theory to deliver a well-rounded investment insights.

## Key Features

### 1. Market Regime Detection System
- **Hidden Markov Models (HMM)** for market state classification
- Three distinct market regimes: Crisis/Bear, Steady Growth, Strong Bull
- 10-year historical analysis with 80-90% validation accuracy
- Real-time regime detection and probabilistic transitions
- PCA-based feature reduction for robust model convergence

### 2. Multifactor Stock Scoring System
- **Comprehensive factor analysis** across 50+ subfactors:
  - **Value**: P/E, P/B, EV/EBITDA ratios
  - **Quality**: ROE, ROA, ROIC metrics
  - **Growth**: Revenue, EPS, and earnings momentum
  - **Financial Health**: Debt ratios, current ratio, cash flow
  - **Technical**: Trend, breakout, and momentum indicators
  - **Stability**: Volatility, beta, and risk metrics
  - **Credit**: Altman Z-score and default probability
  - **Size, Liquidity, Carry, Insider** factors
- **Regime-adaptive weighting** optimized through backtesting
- Industry and sector-relative adjustments
- Continuous scoring (0-1 scale) for all metrics

### 3. Score Trend Analysis System
- Multi-period stability analysis with linear regression
- **Technical indicator suite**:
  - SMA/EMA crossovers with trend quality scoring
  - MACD signal analysis and histogram divergence
  - RSI with overbought/oversold detection
  - Bollinger Bands with squeeze detection
  - ADX trend strength measurement
- **Advanced forecasting** with ARIMA and exponential smoothing
- R² adjusted scoring for trend reliability

### 4. Dynamic Portfolio Selection System
- **Multiple construction strategies**:
  - Top-ranked selection
  - Sector-balanced diversification
  - Industry-balanced allocation
  - Factor-balanced portfolios
  - Hybrid stability-sector approach
- Backtesting engine with:
  - Walk-forward analysis
  - Multiple lookback periods (252, 504, 756 days)
  - Risk-adjusted performance metrics
  - Transaction cost modeling
- Monte Carlo simulations for robustness testing

### 5. Portfolio Optimization
- **Hierarchical Risk Parity (HRP)** methods:
  - **HRP**: Standard hierarchical risk parity
  - **HERC**: Hierarchical Equal Risk Contribution
  - **MHRP**: Modified HRP with equal volatility
  - **NCO**: Nested Clustered Optimization
- **Ledoit-Wolf shrinkage** for robust covariance estimation
- Machine learning clustering for asset relationships
- No unstable matrix inversions required

### 6. PyQt5 Interface
- Dark-themed GUI with tabbed navigation
- Real-time progress tracking and status updates
- Interactive charts and visualizations
- Export capabilities for all analyses
- Multi-threaded processing for responsive UX

## Quick Start

### Prerequisites
- Python 3.8 or higher
- TA-Lib C library (required for technical indicators)
  - Windows: Download from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
  - macOS: `brew install ta-lib`
  - Linux: `sudo apt-get install ta-lib`

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MSAS.git
cd MSAS
```

2. Create and activate virtual environment:
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux  
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure API keys (optional):
```bash
# Edit config/marketstack_config.json if using Marketstack API
{
  "api_key": "YOUR_API_KEY_HERE"
}
```

### Launch the Application

```bash
python MSAS_UI_simplified.py
```

## Project Structure

```
MSAS/
├── MSAS_UI.py									# Main application entry point
├── requirements.txt								# Python dependencies
├── config/									# Configuration files
│   ├── marketstack_config.json   # API configuration
│   └── Buyable_stocks_template.txt         
├── src/                          						# Source code
│   ├── classes/                  						# GUI widget classes
│   │   ├── regime_detection_widget.py
│   │   ├── multifactor_scoring_widget_new.py
│   │   ├── score_trend_analysis_widget.py
│   │   ├── dynamic_portfolio_selection_widget_v2.py
│   │   └── portfolio_optimizer_widget.py
│   ├── regime_detection/         						# Market Regime Detection System
│   │   ├── regime_detector.py
│   │   └── current_regime_detector.py
│   ├── scoring/                  						# Multifactor Stock Scoring System
│   │   └── stock_Screener_MultiFactor_25_new.py
│   ├── trend_analysis/           						# Score Trend Analysis System
│   │   ├── stock_score_trend_technical_03.py
│   │   ├── stock_score_trend_analyzer_04.py
│   │   └── stock_score_trend_technical_single_enhanced.py
│   ├── portfolio_selection/                					# Dynamic Portfolio Selection System
│   │   ├── dynamic_portfolio_selector_06.py
│   │   ├── dynamic_portfolio_selector_yfinance.py
│   │   ├── Backtest_advanced_02.py
│   │   └── marketstack_integration.py
│   └── portfolio_optimization/                					# Advanced Portfolio Optimization System
│       └── portfolio_optimizer.py
├── output/                       						# Analysis results
│   ├── Dynamic_Portfolio_Selection/
│   ├── Portfolio_Optimization_Results/
│   ├── Ranked_Lists/
│   ├── Regime_Detection_Analysis/
│   ├── Regime_Detection_Results/
│   └── Score_Trend_Analysis_Results/
└── docs/                         						# Documentation
    ├── MSAS_complete_documentation.pdf
    └── msas_complete_documentation.html

```

## Workflow Overview

1. **Market Regime Detection System** → Identify historical and current market state.
2. **Multifactor Stock Scoring System** → Rank stocks with the scores obtained by multifactor screeners and regime-specific weights.
3. **Score Trend Analysis System** → Analyze score stability and technical patterns.
4. **Dynamic Portfolio Selection System** → Create diversified portfolios from top stocks.
5. **Portfolio Optimization System** → Apply portfolio optimization methods for allocation.

## Usage Guide

### 1. Market Regime Detection System
Navigate to the **Regime Detection** tab to analyze current market conditions:
- Click "Fetch Regime Data" to obtain historical regime data.
- Click "Run Detection" to obtain current regime status, and recommendation for the following systems.

### 2. Multifactor Stock Scoring System
Use the **Multifactor Scoring** tab to rank stocks:
- Select test mode or full analysis
- Configure regime-specific factor weights
- Generate ranked stock lists with composite scores

### 3. Score Trend Analysis System
Access the **Score Trend Analysis** tab for stability assessment:
- Run stability analysis on the ranked lists
- Generate technical indicator plots
- Review investment recommendations

### 4. Dynamic Portfolio Selection System
Build portfolios in the **Portfolio Selection** tab:
- Choose data source (Yahoo Finance or Marketstack)
- Configure portfolio sizes and selection pools
- Run backtests with multiple lookback periods
- Export portfolio compositions

### 5. Portfolio Optimization System
Optimize allocations in the **Portfolio Optimization** tab:
- Select optimization method (MHRP, HRP, HERC, NCO)
- Input portfolio tickers or load from file
- Generate allocation weights and performance metrics
- Visualize hierarchical clustering dendrograms

## Data Sources

- **Yahoo Finance** (primary via yfinance/yahooquery)
- **Marketstack API** (optional for enhanced data)
- **SEC EDGAR** for insider trading data
- **FRED** for macroeconomic indicators

## Performance Metrics

- **Processing Speed**: ~900-1000 stocks in 5-10 minutes
- **Regime Detection Accuracy**: 80-90% validation
- **Backtesting**: Superior Sharpe ratios vs. benchmarks
- **Portfolio Optimization**: O(N² log N) computational complexity

## Contributing

Contributions are welcome! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
3. Push to the branch (git push origin feature/AmazingFeature)
4. Open a Pull Request

## Documentation

- **Complete documentation**: See `docs/msas_complete_documentation.html`

## Known Issues

- TA-Lib installation can be challenging on some systems
- Large ticker lists (>1000) may require memory optimization
- Some technical indicators require minimum 20 data points

## Future Enhancements

- [ ] Web-based interface using Flask/React
- [ ] Real-time data streaming integration
- [ ] Options strategy analysis
- [ ] Multi-asset class support (crypto, forex)
- [ ] Cloud deployment capabilities
- [ ] Advanced factor research tools
- [ ] Automated trading integration

## License

MIT License - See LICENSE file for details

## Authors

- **MSAS Development Team** - *Initial work and ongoing development*

## Acknowledgments

- Marcos López de Prado - Hierarchical Risk Parity methodology
- scikit-learn - Machine learning algorithms
- PyQt5 - GUI framework
- yfinance - Market data access
- TA-Lib for technical indicators
- scikit-learn for machine learning algorithms
- yfinance for market data access
- PyQt5 for GUI framework
- riskfolio-lib for portfolio optimization

## Support

For support, please:
1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed information

---

**Built with passion for quantitative finance and open-source collaboration**

*Last Updated: October 2025*