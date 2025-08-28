# ========================================
# compare_data_sources.py - Save this in your project root
# ========================================

"""
Script to compare regime detection accuracy between Marketstack and yfinance data
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from regime_detection.regime_detector import MarketRegimeDetector

def compare_data_sources():
    """Compare regime detection results from different data sources"""

    print("=" * 80)
    print("DATA SOURCE COMPARISON: MARKETSTACK vs YFINANCE")
    print("=" * 80)

    results = {}

    # Test with yfinance
    print("\n" + "=" * 40)
    print("TESTING WITH YFINANCE")
    print("=" * 40)

    detector_yf = MarketRegimeDetector(n_regimes=3, lookback_years=10)

    # Fetch with yfinance
    detector_yf.fetch_market_data(
        symbols=['SPY', '^VIX', '^TNX', 'GLD'],
        use_marketstack=False  # Force yfinance
    )

    print(f"Data range: {detector_yf.data.index[0]} to {detector_yf.data.index[-1]}")
    print(f"Total days: {len(detector_yf.data)}")

    # Prepare and fit
    if hasattr(detector_yf, 'prepare_features'):
        detector_yf.prepare_features()
    elif hasattr(detector_yf, 'engineer_features'):
        detector_yf.engineer_features()

    detector_yf.fit_hmm(use_pca=True, n_components=5)
    detector_yf.characterize_regimes()

    # Validate
    accuracy_yf = detector_yf.validate_against_known_events()

    results['yfinance'] = {
        'data_days': len(detector_yf.data),
        'date_range': f"{detector_yf.data.index[0]} to {detector_yf.data.index[-1]}",
        'accuracy': accuracy_yf
    }

    # Test with Marketstack
    print("\n" + "=" * 40)
    print("TESTING WITH MARKETSTACK")
    print("=" * 40)

    detector_ms = MarketRegimeDetector(n_regimes=3, lookback_years=10)

    # Fetch with Marketstack
    detector_ms.fetch_market_data(
        symbols=['SPY', '^VIX', '^TNX', 'GLD'],
        use_marketstack=True
    )

    print(f"Data range: {detector_ms.data.index[0]} to {detector_ms.data.index[-1]}")
    print(f"Total days: {len(detector_ms.data)}")

    # Prepare and fit
    if hasattr(detector_ms, 'prepare_features'):
        detector_ms.prepare_features()
    elif hasattr(detector_ms, 'engineer_features'):
        detector_ms.engineer_features()

    detector_ms.fit_hmm(use_pca=True, n_components=5)
    detector_ms.characterize_regimes()

    # Validate
    accuracy_ms = detector_ms.validate_against_known_events()

    results['marketstack'] = {
        'data_days': len(detector_ms.data),
        'date_range': f"{detector_ms.data.index[0]} to {detector_ms.data.index[-1]}",
        'accuracy': accuracy_ms
    }

    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    for source, data in results.items():
        print(f"\n{source.upper()}:")
        print(f"  Data range: {data['date_range']}")
        print(f"  Total days: {data['data_days']}")
        if data['accuracy'] is not None:
            print(f"  Validation accuracy: {data['accuracy']:.1%}")
        else:
            print(f"  Validation accuracy: N/A")

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if results['yfinance']['accuracy'] and results['marketstack']['accuracy']:
        if results['yfinance']['accuracy'] > results['marketstack']['accuracy']:
            diff = (results['yfinance']['accuracy'] - results['marketstack']['accuracy']) * 100
            print(f"✓ Yfinance shows {diff:.1f}% better accuracy")
            print("  - Free and unlimited data access")
            print("  - Better historical coverage")
            print("  → Recommended for regime detection")
        else:
            diff = (results['marketstack']['accuracy'] - results['yfinance']['accuracy']) * 100
            print(f"✓ Marketstack shows {diff:.1f}% better accuracy")
            print("  - Professional data source")
            print("  - More reliable for production")

    # Data coverage comparison
    yf_days = results['yfinance']['data_days']
    ms_days = results['marketstack']['data_days']

    if yf_days > ms_days:
        print(f"\n⚠️ Yfinance provides {yf_days - ms_days} more days of historical data")
    elif ms_days > yf_days:
        print(f"\n⚠️ Marketstack provides {ms_days - yf_days} more days of historical data")

    return results

if __name__ == "__main__":
    compare_data_sources()