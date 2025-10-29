"""
Simplified current regime detector that reads from historical analysis
This ensures consistency between historical and current regime detection
"""
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import numpy as np

class CurrentRegimeDetector:
    def __init__(self):
        """Initialize by reading the latest historical analysis"""
        self.project_root = Path(__file__).parent.parent.parent
        self.output_dir = self.project_root / "output" / "Regime_Detection_Analysis"
        self.results_dir = self.project_root / "output" / "Regime_Detection_Results"

        # Load the latest regime data
        self.load_historical_data()

    def load_historical_data(self):
        """Load the latest historical regime analysis"""
        # Try to load regime periods
        regime_periods_file = self.results_dir / "regime_periods.csv"
        if not regime_periods_file.exists():
            raise FileNotFoundError(
                f"Regime periods file not found at {regime_periods_file}\n"
                "Please run regime detection first (Fetch Regime Data button)"
            )

        # Load regime periods
        self.regime_periods = pd.read_csv(regime_periods_file)
        self.regime_periods['start_date'] = pd.to_datetime(self.regime_periods['start_date'])
        self.regime_periods['end_date'] = pd.to_datetime(self.regime_periods['end_date'])

        # Sort by end date to get the most recent
        self.regime_periods = self.regime_periods.sort_values('end_date')

        # Load model data for characteristics if available
        model_file = self.results_dir / "regime_model.pkl"
        if model_file.exists():
            import pickle
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
                self.regime_characteristics = model_data.get('regime_characteristics', {})
        else:
            self.regime_characteristics = None

        print(" Loaded historical regime data")

    def get_current_regime(self):
        """Get the current regime from the latest historical data"""
        if self.regime_periods.empty:
            raise ValueError("No regime data available")

        # Get the last (most recent) regime period
        latest_period = self.regime_periods.iloc[-1]

        # Prepare the output
        current_regime = {
            'regime': int(latest_period.get('regime', -1)),
            'regime_name': latest_period['regime_name'],
            'start_date': latest_period['start_date'].strftime('%Y-%m-%d'),
            'end_date': latest_period['end_date'].strftime('%Y-%m-%d'),
            'days_in_regime': int(latest_period['days']),
            'as_of': latest_period['end_date'].strftime('%Y-%m-%d'),
        }

        # Add characteristics if available
        if self.regime_characteristics and current_regime['regime'] >= 0:
            regime_chars = self.regime_characteristics.get(current_regime['regime'], {})
            current_regime['characteristics'] = {
                'mean_return': regime_chars.get('mean_return', 0),
                'volatility': regime_chars.get('volatility', 0),
                'sharpe': regime_chars.get('sharpe', 0)
            }

        # Calculate regime probabilities from historical distribution
        regime_counts = self.regime_periods['regime_name'].value_counts()
        total_periods = len(self.regime_periods)

        all_probabilities = {}
        for regime_name in ['Strong Bull', 'Steady Growth', 'Crisis/Bear']:
            if regime_name in regime_counts:
                # Use historical frequency as proxy for probability
                all_probabilities[regime_name] = regime_counts[regime_name] / total_periods
            else:
                all_probabilities[regime_name] = 0.0

        # Set current regime probability higher to indicate confidence
        if current_regime['regime_name'] in all_probabilities:
            # Boost current regime probability
            current_prob = all_probabilities[current_regime['regime_name']]
            boost_factor = 2.0  # Make current regime 2x more likely

            # Renormalize probabilities
            for regime in all_probabilities:
                if regime == current_regime['regime_name']:
                    all_probabilities[regime] = min(current_prob * boost_factor, 0.8)
                else:
                    all_probabilities[regime] *= (1 - all_probabilities[current_regime['regime_name']]) / (1 - current_prob) if current_prob < 1 else 0.1

        # Ensure probabilities sum to 1
        total = sum(all_probabilities.values())
        if total > 0:
            all_probabilities = {k: v/total for k, v in all_probabilities.items()}

        current_regime['all_probabilities'] = all_probabilities
        current_regime['confidence'] = all_probabilities.get(current_regime['regime_name'], 0.5)

        return current_regime

    def detect_current_regime(self):
        """Wrapper method for compatibility"""
        return self.get_current_regime()

    def get_recommended_weights(self, regime_name):
        """Get recommended weights for the current regime"""
        REGIME_WEIGHTS = {
            'Strong Bull': {
                "momentum": 60.0,
                "growth": 15.0,
                "technical": 15.0,
                "quality": 10.0,
                "credit": 5.0,
                "financial_health": 5.0,
                "value": -5.0,
                "size": -5.0,
                "volatility_penalty": -3.0,
                "carry": -2.0,
                "liquidity": 5.0,
                "insider": 0.0
            },
            'Steady Growth': {
                "momentum": 45.0,
                "quality": 20.0,
                "growth": 10.0,
                "technical": 10.0,
                "credit": 10.0,
                "financial_health": 10.0,
                "value": 5.0,
                "size": -5.0,
                "volatility_penalty": -5.0,
                "carry": 0.0,
                "liquidity": 3.0,
                "insider": -3.0
            },
            'Crisis/Bear': {
                "quality": 30.0,
                "financial_health": 25.0,
                "credit": 20.0,
                "value": 15.0,
                "carry": 5.0,
                "liquidity": 10.0,
                "momentum": 10.0,
                "growth": 0.0,
                "technical": 0.0,
                "volatility_penalty": -15.0,
                "size": -5.0,
                "insider": 5.0
            }
        }

        return REGIME_WEIGHTS.get(regime_name, REGIME_WEIGHTS['Steady Growth'])


def main():
    """Main function for standalone execution"""
    try:
        print("=" * 60)
        print("CURRENT REGIME DETECTION (from Historical Analysis)")
        print("=" * 60)

        # Initialize detector
        detector = CurrentRegimeDetector()

        # Get current regime
        current_regime = detector.get_current_regime()

        print(f"\n CURRENT MARKET REGIME")
        print("-" * 40)
        print(f"Regime: {current_regime['regime_name']}")
        print(f"Started: {current_regime['start_date']}")
        print(f"Days in regime: {current_regime['days_in_regime']}")
        print(f"Confidence: {current_regime['confidence']:.1%}")

        print(f"\n Regime Probabilities:")
        for regime, prob in current_regime['all_probabilities'].items():
            indicator = "→" if regime == current_regime['regime_name'] else " "
            print(f"  {indicator} {regime}: {prob:.1%}")

        # Get recommended weights
        weights = detector.get_recommended_weights(current_regime['regime_name'])

        print(f"\n Recommended Factor Weights:")
        for factor, weight in sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True):
            if weight != 0:
                sign = "+" if weight > 0 else ""
                print(f"  {factor}: {sign}{weight:.1f}%")

        # Save results for UI
        output_dir = Path(__file__).parent.parent.parent / "output" / "Regime_Detection_Analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        output = {
            'timestamp': datetime.now().isoformat(),
            'regime_detection': current_regime,
            'recommended_weights': weights,
            'source': 'historical_analysis',
            'note': 'Current regime based on latest historical regime analysis'
        }

        output_file = output_dir / "current_regime_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n Results saved to {output_file}")

        return current_regime, weights

    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()