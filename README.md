# 📊 Multifactor Stock Analysis System (MSAS)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyQt5](https://img.shields.io/badge/PyQt5-GUI-green)](https://pypi.org/project/PyQt5/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success)](https://github.com/)

A comprehensive, regime-aware quantitative stock analysis and portfolio optimization system that combines machine learning, technical analysis, and modern portfolio theory to deliver institutional-grade investment insights.

## 🌟 Key Features

### 1. 📈 Market Regime Detection
- **Hidden Markov Models (HMM)** for market state classification
- Three distinct market regimes: Crisis/Bear, Steady Growth, Strong Bull
- 10-year historical analysis with 80-90% validation accuracy
- Real-time regime detection and probabilistic transitions
- PCA-based feature reduction for robust model convergence

### 2. 🎯 12-Factor Stock Scoring System
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

### 3. 📊 Score Trend Analysis
- **Multi-period stability analysis** with linear regression
- **Technical indicator suite**:
  - SMA/EMA crossovers with trend quality scoring
  - MACD signal analysis and histogram divergence
  - RSI with overbought/oversold detection
  - Bollinger Bands with squeeze detection
  - ADX trend strength measurement
- **Advanced forecasting** with ARIMA and exponential smoothing
- R² adjusted scoring for trend reliability

### 4. 🎲 Dynamic Portfolio Selection
- **Multiple construction strategies**:
  - Top-ranked selection
  - Sector-balanced diversification
  - Industry-balanced allocation
  - Factor-balanced portfolios
  - Hybrid stability-sector approach
- **Sophisticated backtesting engine** with:
  - Walk-forward analysis
  - Multiple lookback periods (252, 504, 756 days)
  - Risk-adjusted performance metrics
  - Transaction cost modeling
- **Monte Carlo simulations** for robustness testing

### 5. 🔧 Portfolio Optimization
- **Hierarchical Risk Parity (HRP)** methods:
  - **HRP**: Standard hierarchical risk parity
  - **HERC**: Hierarchical Equal Risk Contribution
  - **MHRP**: Modified HRP with equal volatility
  - **NCO**: Nested Clustered Optimization
- **Ledoit-Wolf shrinkage** for robust covariance estimation
- Machine learning clustering for asset relationships
- No unstable matrix inversions required

### 6. 💻 Modern PyQt5 Interface
- **Dark-themed professional GUI** with tabbed navigation
- Real-time progress tracking and status updates
- Interactive charts and visualizations
- Export capabilities for all analyses
- Multi-threaded processing for responsive UX

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- TA-Lib C library (see installation notes)
- 8GB+ RAM recommended
- Windows/macOS/Linux

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/MSAS.git
cd MSAS
```

2. **Create virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. **Install TA-Lib C library**
- **Windows**: Download from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)
- **macOS**: `brew install ta-lib`
- **Linux**: `sudo apt-get install ta-lib`

4. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

5. **Configure API keys (optional)**
```bash
# Edit config/marketstack_config.json
{
    "api_key": "your_api_key_here"
}
```

6. **Launch the application**
```bash
python launch.py
# Or use the batch file on Windows:
launch.bat
```

## 📁 Project Structure

```
MSAS/
├── src/
│   ├── classes/           # UI widget classes
│   ├── regime_detection/  # Market regime detection
│   ├── scoring/           # Multifactor scoring engine
│   ├── trend_analysis/    # Score trend analysis
│   ├── portfolio/         # Portfolio selection & optimization
│   └── backend/           # Flask API backend
├── output/                # Analysis results
│   ├── Regime_Detection_Results/
│   ├── Ranked_Lists/
│   ├── Score_Trend_Analysis_Results/
│   └── Portfolio_Optimization_Results/
├── config/                # Configuration files
│   ├── Buyable_stock.txt
│   └── marketstack_config.json
├── docs/                  # Documentation
├── tests/                 # Unit tests
└── requirements.txt       # Python dependencies
```

## 📊 Workflow Overview

1. **Regime Detection** → Identify current market state
2. **Stock Scoring** → Rank stocks using regime-specific weights
3. **Trend Analysis** → Analyze score stability and technical patterns
4. **Portfolio Selection** → Create diversified portfolios from top stocks
5. **Optimization** → Apply HRP methods for final allocation
6. **Backtesting** → Validate performance across multiple periods

## 🔬 Advanced Features

### Machine Learning Integration
- **LSTM models** for factor prediction (optional)
- **Transformer architecture** for sequence modeling (optional)
- **PCA** for dimensionality reduction
- **Clustering algorithms** for portfolio construction

### Risk Management
- **Multi-dimensional risk metrics**: VaR, CVaR, maximum drawdown
- **Regime-specific risk adjustments**
- **Correlation-based diversification**
- **Tail risk hedging indicators**

### Data Sources
- **Yahoo Finance** (primary via yfinance/yahooquery)
- **Marketstack API** (optional for enhanced data)
- **SEC EDGAR** for insider trading data
- **FRED** for macroeconomic indicators

## 📈 Performance Metrics

- **Processing Speed**: ~900-1000 stocks in 5-10 minutes
- **Regime Detection Accuracy**: 80-90% validation
- **Backtesting**: Superior Sharpe ratios vs. benchmarks
- **Portfolio Optimization**: O(N² log N) computational complexity

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 Documentation

- **Complete documentation**: See `docs/msas_complete_documentation.html`
- **User guide**: Available in `docs/user_guide.md`
- **API reference**: Coming soon

## 🐛 Known Issues

- TA-Lib installation can be challenging on some systems
- Large ticker lists (>1000) may require memory optimization
- Some technical indicators require minimum 20 data points

## 🔮 Future Enhancements

- [ ] Web-based interface using Flask/React
- [ ] Real-time data streaming integration
- [ ] Options strategy analysis
- [ ] Multi-asset class support (crypto, forex)
- [ ] Cloud deployment capabilities
- [ ] Advanced factor research tools
- [ ] Automated trading integration

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **MSAS Development Team** - *Initial work and ongoing development*

## 🙏 Acknowledgments

- **Marcos López de Prado** - Hierarchical Risk Parity methodology
- **scikit-learn** - Machine learning algorithms
- **PyQt5** - GUI framework
- **yfinance** - Market data access
- All contributors and open-source library maintainers

## 📞 Support

For support, please:
1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed information

---

**⚡ Built with passion for quantitative finance and open-source collaboration**

*Last Updated: September 2025*