"""
Macro View Engine for Black-Litterman Portfolio Optimization
Generates objective, data-driven views from market data

Author: MSAS Team
Date: 2025-11-20
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class MacroViewEngine:
    """
    Generate objective macro views from:
    1. Analyst consensus (price targets)
    2. Sector momentum analysis
    3. Commodity price trends (for energy/materials)
    4. VIX-based risk adjustment
    """

    def __init__(self):
        self.views = {}
        self.confidence_levels = {}
        self.view_sources = {}

    def generate_views(self, tickers: list, lookback_days: int = 90) -> Dict[str, Dict]:
        """
        Generate macro views for a list of tickers

        Args:
            tickers: List of stock tickers
            lookback_days: Days of historical data to analyze

        Returns:
            Dictionary with structure:
            {
                'VNOM': {
                    'expected_return': -0.10,
                    'confidence': 0.6,
                    'source': 'Analyst consensus + Sector momentum',
                    'details': {...}
                }
            }
        """
        print("\n" + "=" * 60)
        print("GENERATING MACRO VIEWS")
        print("=" * 60)

        all_views = {}

        for ticker in tickers:
            try:
                print(f"\nAnalyzing {ticker}...")
                view = self._generate_ticker_view(ticker, lookback_days)

                if view is not None:
                    all_views[ticker] = view
                    print(f"  View: {view['expected_return']*100:+.1f}% "
                          f"(Confidence: {view['confidence']:.1%})")
                    print(f"  Source: {view['source']}")

            except Exception as e:
                print(f"  Error analyzing {ticker}: {e}")
                continue

        # Apply VIX-based risk adjustment
        self._apply_vix_adjustment(all_views)

        return all_views

    def _generate_ticker_view(self, ticker: str, lookback_days: int) -> Optional[Dict]:
        """Generate view for a single ticker"""

        views_components = []
        confidence_components = []

        # 1. Analyst consensus view
        analyst_view = self._get_analyst_view(ticker)
        if analyst_view:
            views_components.append(analyst_view['expected_return'])
            confidence_components.append(analyst_view['confidence'])

        # 2. Sector momentum view
        sector_view = self._get_sector_momentum_view(ticker, lookback_days)
        if sector_view:
            views_components.append(sector_view['expected_return'])
            confidence_components.append(sector_view['confidence'])

        # 3. Commodity/factor-specific view (for energy, metals, etc.)
        factor_view = self._get_factor_view(ticker)
        if factor_view:
            views_components.append(factor_view['expected_return'])
            confidence_components.append(factor_view['confidence'])

        # Combine views if we have any
        if not views_components:
            return None

        # Weighted average by confidence
        total_confidence = sum(confidence_components)
        if total_confidence > 0:
            weighted_return = sum(
                v * c for v, c in zip(views_components, confidence_components)
            ) / total_confidence
        else:
            weighted_return = np.mean(views_components)

        # Average confidence
        avg_confidence = np.mean(confidence_components) if confidence_components else 0.5

        # Determine source description
        sources = []
        if analyst_view:
            sources.append("Analyst consensus")
        if sector_view:
            sources.append("Sector momentum")
        if factor_view:
            sources.append(factor_view['source'])

        return {
            'expected_return': weighted_return,
            'confidence': avg_confidence,
            'source': " + ".join(sources),
            'components': {
                'analyst': analyst_view,
                'sector': sector_view,
                'factor': factor_view
            }
        }

    def _get_analyst_view(self, ticker: str) -> Optional[Dict]:
        """Get view from analyst price targets"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Get analyst target price
            target_price = info.get('targetMeanPrice')
            current_price = info.get('currentPrice')

            if target_price and current_price and target_price > 0 and current_price > 0:
                expected_return = (target_price - current_price) / current_price

                # Get number of analysts
                num_analysts = info.get('numberOfAnalystOpinions', 0)

                # Confidence based on number of analysts
                # More analysts = higher confidence
                if num_analysts >= 10:
                    confidence = 0.8
                elif num_analysts >= 5:
                    confidence = 0.6
                elif num_analysts >= 2:
                    confidence = 0.4
                else:
                    confidence = 0.2

                # Cap expected returns to reasonable range (-30% to +30%)
                expected_return = np.clip(expected_return, -0.30, 0.30)

                return {
                    'expected_return': expected_return,
                    'confidence': confidence,
                    'target_price': target_price,
                    'current_price': current_price,
                    'num_analysts': num_analysts
                }

        except Exception as e:
            # print(f"    Could not get analyst view for {ticker}: {e}")
            pass

        return None

    def _get_sector_momentum_view(self, ticker: str, lookback_days: int) -> Optional[Dict]:
        """Get view from sector momentum analysis with multi-level fallback"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Get sector and country
            sector = info.get('sector', '')
            country = info.get('country', '')
            industry = info.get('industry', '')

            # Try multi-level fallback strategy
            sector_etf = None
            fallback_level = "Unknown"

            # Level 1: Specific industry/subsector ETFs
            industry_etf_map = {
                'Fintech': 'FINX',
                'Semiconductors': 'SOXX',
                'Software': 'IGV',
                'Biotechnology': 'IBB',
                'Cloud Computing': 'SKYY',
                'Cybersecurity': 'HACK',
                'Renewable Energy': 'ICLN',
                'Battery Technology': 'LIT'
            }

            for key, etf in industry_etf_map.items():
                if key.lower() in industry.lower():
                    sector_etf = etf
                    fallback_level = f"Industry ({key})"
                    break

            # Level 2: Country-specific sector ETFs
            if not sector_etf and country:
                country_sector_etf_map = {
                    ('China', 'Financial Services'): 'KWEB',  # China Internet/Fintech (includes financials)
                    ('China', 'Technology'): 'CQQQ',  # China Tech
                    ('China', 'Consumer Cyclical'): 'CHIQ',  # China Consumer
                    ('China', 'Consumer Defensive'): 'CHIQ',  # China Consumer
                }
                sector_etf = country_sector_etf_map.get((country, sector))
                if sector_etf:
                    fallback_level = f"{country} {sector}"

            # Level 3: Country/region broad market ETFs
            if not sector_etf and country:
                country_etf_map = {
                    'China': 'FXI',  # iShares China Large-Cap
                    'India': 'INDA',
                    'Brazil': 'EWZ',
                    'Japan': 'EWJ',
                    'United Kingdom': 'EWU',
                    'Germany': 'EWG',
                    'South Korea': 'EWY',
                    'Taiwan': 'EWT',
                    'Hong Kong': 'EWH'
                }
                sector_etf = country_etf_map.get(country)
                if sector_etf:
                    fallback_level = f"{country} Market"

            # Level 4: US sector ETFs (for US stocks or as final fallback)
            if not sector_etf and sector:
                sector_etf_map = {
                    'Energy': 'XLE',
                    'Technology': 'XLK',
                    'Financial Services': 'XLF',
                    'Healthcare': 'XLV',
                    'Consumer Cyclical': 'XLY',
                    'Consumer Defensive': 'XLP',
                    'Industrials': 'XLI',
                    'Real Estate': 'XLRE',
                    'Utilities': 'XLU',
                    'Basic Materials': 'XLB',
                    'Communication Services': 'XLC'
                }
                sector_etf = sector_etf_map.get(sector)
                if sector_etf:
                    fallback_level = f"US {sector}"

            # Level 5: Emerging markets (if non-US)
            if not sector_etf and country and country != 'United States':
                sector_etf = 'EEM'  # Emerging Markets
                fallback_level = "Emerging Markets"

            if not sector_etf:
                return None

            # Get sector ETF data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)

            sector_data = yf.download(sector_etf, start=start_date, end=end_date, progress=False)

            if sector_data.empty:
                return None

            # Calculate momentum (return over period)
            sector_return = (sector_data['Close'].iloc[-1] / sector_data['Close'].iloc[0]) - 1

            # Compare to SPY (market benchmark)
            spy_data = yf.download('SPY', start=start_date, end=end_date, progress=False)
            market_return = (spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[0]) - 1

            # Relative momentum
            relative_momentum = sector_return - market_return

            # Scale to expected return view (-20% to +20%)
            expected_return = np.clip(relative_momentum * 0.5, -0.20, 0.20)

            # Confidence based on strength of momentum
            abs_momentum = abs(relative_momentum)
            if abs_momentum > 0.10:
                confidence = 0.7
            elif abs_momentum > 0.05:
                confidence = 0.5
            else:
                confidence = 0.3

            return {
                'expected_return': expected_return,
                'confidence': confidence,
                'sector': sector,
                'sector_etf': sector_etf,
                'fallback_level': fallback_level,
                'sector_return': sector_return,
                'market_return': market_return,
                'relative_momentum': relative_momentum
            }

        except Exception as e:
            # print(f"    Could not get sector momentum for {ticker}: {e}")
            pass

        return None

    def _get_factor_view(self, ticker: str) -> Optional[Dict]:
        """Get view from factor-specific analysis (commodity prices, etc.)"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            sector = info.get('sector', '')

            # Energy sector: Use oil price trend
            if sector == 'Energy':
                return self._get_oil_price_view(ticker)

            # Basic Materials: Use commodity indices
            elif sector == 'Basic Materials':
                return self._get_commodity_view(ticker)

            # TODO: Add more factor-specific views
            # - Interest rate sensitivity for financials
            # - Dollar strength for international stocks
            # - etc.

        except Exception as e:
            pass

        return None

    def _get_oil_price_view(self, ticker: str) -> Optional[Dict]:
        """Analyze oil price trend for energy stocks"""
        try:
            # Use USO (US Oil Fund) as oil price proxy
            oil = yf.download('USO', period='6mo', progress=False)

            if oil.empty:
                return None

            # Calculate moving averages
            oil_sma_50 = oil['Close'].rolling(50).mean().iloc[-1]
            oil_sma_200 = oil['Close'].rolling(200).mean().iloc[-1] if len(oil) >= 200 else oil_sma_50
            current_oil = oil['Close'].iloc[-1]

            # Calculate trend strength
            if oil_sma_200 > 0:
                trend_strength = (oil_sma_50 - oil_sma_200) / oil_sma_200
            else:
                trend_strength = 0

            # Generate view based on trend
            if current_oil < oil_sma_50 < oil_sma_200:
                # Strong downtrend
                expected_return = -0.15
                confidence = 0.7
                description = "Oil in strong downtrend"
            elif current_oil < oil_sma_50:
                # Mild downtrend
                expected_return = -0.08
                confidence = 0.5
                description = "Oil in mild downtrend"
            elif current_oil > oil_sma_50 > oil_sma_200:
                # Strong uptrend
                expected_return = 0.12
                confidence = 0.7
                description = "Oil in strong uptrend"
            elif current_oil > oil_sma_50:
                # Mild uptrend
                expected_return = 0.06
                confidence = 0.5
                description = "Oil in mild uptrend"
            else:
                # Neutral/choppy
                expected_return = 0.0
                confidence = 0.3
                description = "Oil range-bound"

            return {
                'expected_return': expected_return,
                'confidence': confidence,
                'source': f"Oil price trend ({description})",
                'current_oil': float(current_oil),
                'oil_sma_50': float(oil_sma_50),
                'trend_strength': float(trend_strength)
            }

        except Exception as e:
            # print(f"    Could not get oil price view: {e}")
            pass

        return None

    def _get_commodity_view(self, ticker: str) -> Optional[Dict]:
        """Analyze commodity trends for materials stocks"""
        try:
            # Use DBC (commodity index) as proxy
            commodities = yf.download('DBC', period='6mo', progress=False)

            if commodities.empty:
                return None

            # Simple momentum
            commodity_return = (commodities['Close'].iloc[-1] / commodities['Close'].iloc[0]) - 1

            # Scale to view
            expected_return = np.clip(commodity_return * 0.5, -0.15, 0.15)

            # Confidence based on momentum strength
            confidence = min(0.7, 0.4 + abs(commodity_return) * 2)

            return {
                'expected_return': expected_return,
                'confidence': confidence,
                'source': "Commodity price trend",
                'commodity_return': float(commodity_return)
            }

        except Exception as e:
            pass

        return None

    def _apply_vix_adjustment(self, views: Dict):
        """Adjust view confidence based on VIX level"""
        try:
            vix = yf.download('^VIX', period='1mo', progress=False)

            if vix.empty or len(vix) == 0:
                print("\nCould not fetch VIX data for risk adjustment")
                return

            # Handle both Series and DataFrame
            if isinstance(vix, pd.DataFrame):
                current_vix = float(vix['Close'].iloc[-1])
            else:
                current_vix = float(vix.iloc[-1])

            # VIX-based risk adjustment
            if current_vix > 30:
                # High volatility: reduce confidence
                adjustment = 0.7
                print(f"\nHigh VIX ({current_vix:.1f}): Reducing view confidence by 30%")
            elif current_vix > 20:
                # Elevated volatility: slight reduction
                adjustment = 0.85
                print(f"\nElevated VIX ({current_vix:.1f}): Reducing view confidence by 15%")
            elif current_vix < 12:
                # Low volatility: increase confidence
                adjustment = 1.15
                print(f"\nLow VIX ({current_vix:.1f}): Increasing view confidence by 15%")
            else:
                # Normal range: no adjustment
                adjustment = 1.0

            # Apply adjustment to all views
            for ticker in views:
                views[ticker]['confidence'] *= adjustment
                views[ticker]['confidence'] = min(0.95, views[ticker]['confidence'])  # Cap at 95%

        except Exception as e:
            print(f"Could not apply VIX adjustment: {e}")

    def save_views_report(self, views: Dict, output_path: str):
        """Save detailed views report to file"""
        import json
        from pathlib import Path

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        report = {
            'generation_date': datetime.now().isoformat(),
            'views': views,
            'summary': {
                'num_tickers': len(views),
                'avg_expected_return': np.mean([v['expected_return'] for v in views.values()]),
                'avg_confidence': np.mean([v['confidence'] for v in views.values()])
            }
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nViews report saved to: {output_file}")


def test_macro_views():
    """Test the macro view engine"""
    print("Testing Macro View Engine")
    print("=" * 60)

    # Test with energy and tech stocks
    test_tickers = ['VNOM', 'FSLR', 'AAPL', 'MSFT', 'XOM']

    engine = MacroViewEngine()
    views = engine.generate_views(test_tickers, lookback_days=90)

    print("\n" + "=" * 60)
    print("GENERATED VIEWS SUMMARY")
    print("=" * 60)

    for ticker, view in views.items():
        print(f"\n{ticker}:")
        print(f"  Expected Return: {view['expected_return']*100:+.1f}%")
        print(f"  Confidence: {view['confidence']:.1%}")
        print(f"  Source: {view['source']}")

    return views


if __name__ == "__main__":
    views = test_macro_views()
