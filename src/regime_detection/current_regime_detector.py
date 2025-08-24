"""
Real-time market regime detection using trained model
"""
import os
import pickle
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
from sklearn.decomposition import PCA
import pickle
from pathlib import Path

class CurrentRegimeDetector:
    def __init__(self, model_path='regime_model.pkl'):
        # Create Analysis directory if it doesn't exist
        self.output_dir = "./Analysis"
        os.makedirs(self.output_dir, exist_ok=True)

        # Update model path to look in Analysis directory
        model_path = os.path.join(self.output_dir, model_path)

        """Load the trained regime model"""
        with open(model_path, 'rb') as f:
            self.model_data = pickle.load(f)

        self.model = self.model_data['model']
        self.scaler = self.model_data['scaler']
        self.pca = self.model_data.get('pca', None)
        self.use_pca = self.model_data.get('use_pca', False)
        self.feature_columns = self.model_data['feature_columns']
        self.regime_characteristics = self.model_data['regime_characteristics']

        if self.use_pca and self.pca is not None:
            print(f"📊 Loaded model with PCA ({self.pca.n_components} components)")
        else:
            print("📊 Loaded model without PCA")

    def fetch_recent_data(self, lookback_days=500):  # Increased for rolling calculations
        """Fetch recent market data for regime detection"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        symbols = ['SPY', '^VIX', '^TNX', 'GLD']  # Match training data
        data = {}

        print("📊 Fetching market data...")
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)
                if not hist.empty:
                    data[symbol] = hist['Close']
                    print(f"✓ Downloaded {symbol} ({len(hist)} days)")
            except Exception as e:
                print(f"✗ Failed to download {symbol}: {e}")

        df = pd.DataFrame(data)
        # Forward fill then backward fill
        df = df.ffill().bfill()
        return df

    def engineer_current_features(self, data):
        """Create features matching the training process"""
        features = pd.DataFrame(index=data.index)

        # Use SPY as base (matching training)
        base_col = 'SPY' if 'SPY' in data.columns else data.columns[0]

        # Core return features - fix the warning
        returns = data[base_col].pct_change(fill_method=None)
        features['return_1d'] = returns
        features['return_5d'] = returns.rolling(5).mean()
        features['return_21d'] = returns.rolling(21).mean()

        # Volatility features
        features['volatility_21d'] = returns.rolling(21).std() * np.sqrt(252)
        features['volatility_63d'] = returns.rolling(63).std() * np.sqrt(252)

        # Volatility change
        features['vol_change'] = features['volatility_21d'].pct_change(21, fill_method=None)

        # VIX normalization if available
        if '^VIX' in data.columns:
            vix_mean = data['^VIX'].rolling(252).mean()
            vix_std = data['^VIX'].rolling(252).std()
            features['vix_norm'] = (data['^VIX'] - vix_mean) / vix_std

        # Trend
        sma_50 = data[base_col].rolling(50).mean()
        sma_200 = data[base_col].rolling(200).mean()
        features['trend'] = (sma_50 - sma_200) / sma_200

        # Momentum
        features['momentum'] = data[base_col] / data[base_col].shift(63) - 1

        # Drop NaN values
        features = features.dropna()

        print(f"✓ Created {len(features.columns)} features")
        print(f"✓ Available data points: {len(features)}")

        if len(features) == 0:
            raise ValueError("No valid features after cleaning. Try increasing lookback period.")

        return features

    def detect_current_regime(self):
        """Detect the current market regime"""
        # Fetch recent data
        data = self.fetch_recent_data()

        # Engineer features
        features = self.engineer_current_features(data)

        # Ensure we have the same features as training
        # Reorder columns to match training if needed
        if hasattr(self, 'feature_columns') and self.feature_columns:
            # Make sure we have all required columns
            for col in self.feature_columns:
                if col not in features.columns:
                    print(f"Warning: Missing feature {col}, setting to 0")
                    features[col] = 0

            # Select only the columns used in training
            features = features[self.feature_columns]

        # Scale features
        features_scaled = pd.DataFrame(
            self.scaler.transform(features),
            index=features.index,
            columns=features.columns
        )

        # Apply PCA if needed (using the PCA set up in __init__)
        if self.use_pca:
            print("📊 Applying PCA transformation...")

            # Fit PCA on all available scaled features
            self.pca.fit(features_scaled)

            # Transform features
            features_pca = self.pca.transform(features_scaled)

            # Get latest observation
            current_features = features_pca[-1:, :]

            print(f"✓ PCA features shape: {current_features.shape}")
        else:
            # No PCA - use scaled features directly
            current_features = features_scaled.iloc[-1:].values
            print(f"✓ Features shape: {current_features.shape}")

        # Predict regime
        current_regime = self.model.predict(current_features)[0]
        regime_proba = self.model.predict_proba(current_features)[0]

        # Get regime characteristics
        regime_info = self.regime_characteristics[current_regime]

        return {
            'regime': current_regime,
            'regime_name': regime_info['name'],
            'confidence': regime_proba[current_regime],
            'all_probabilities': {
                self.regime_characteristics[i]['name']: regime_proba[i]
                for i in range(len(regime_proba))
            },
            'regime_characteristics': regime_info,
            'as_of': features.index[-1].strftime('%Y-%m-%d'),
            'latest_features': {
                'volatility_21d': features['volatility_21d'].iloc[-1],
                'momentum': features['momentum'].iloc[-1],
                'trend': features['trend'].iloc[-1]
            }
        }

    def get_recommended_weights(self, current_regime_name):
        """Get recommended factor weights for current regime"""

        # Define regime-specific weights based on your optimization results
        # These should be updated after you run the optimizer on each regime
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

        return REGIME_WEIGHTS.get(current_regime_name, REGIME_WEIGHTS['Steady Growth'])

    def get_current_regime_for_api(self):
        """
        Function to get current regime in API-friendly format
        Add this to your current_regime_detector.py
        """
        try:
            # Run your existing current regime detection
            # current_regime = your_existing_detection_function()

            # Format for API
            result = {
                "current_regime": "Steady Growth",  # Replace with actual
                "probabilities": {
                    "Steady Growth": 0.68,
                    "Strong Bull": 0.22,
                    "Crisis/Bear": 0.10
                },
                "last_updated": datetime.now().isoformat(),
                "confidence_score": 0.85
            }

            # Save to JSON for API access
            with open("Analysis/current_regime_analysis.json", 'w') as f:
                json.dump(result, f, indent=2)

            return result

        except Exception as e:
            return {"error": str(e)}

def main():
    try:
        # Initialize detector
        detector = CurrentRegimeDetector()

        # Detect current regime
        print("\n🔍 Detecting current market regime...")
        result = detector.detect_current_regime()

        print(f"\n{'='*60}")
        print("🎯 CURRENT MARKET REGIME ANALYSIS")
        print(f"{'='*60}")
        print(f"Date: {result['as_of']}")
        print(f"Regime: {result['regime_name']}")
        print(f"Confidence: {result['confidence']:.1%}")

        print("\nAll Regime Probabilities:")
        for regime, prob in result['all_probabilities'].items():
            print(f"  {regime}: {prob:.1%}")

        print("\nRegime Characteristics:")
        chars = result['regime_characteristics']
        print(f"  Expected Return: {chars['mean_return']:.1%}")
        print(f"  Expected Volatility: {chars['volatility']:.1%}")
        print(f"  Sharpe Ratio: {chars['sharpe']:.2f}")

        print("\nCurrent Market Conditions:")
        features = result['latest_features']
        print(f"  21-day Volatility: {features['volatility_21d']:.1%}")
        print(f"  63-day Momentum: {features['momentum']:.1%}")
        print(f"  Trend Strength: {features['trend']:.3f}")

        # Get recommended weights
        weights = detector.get_recommended_weights(result['regime_name'])
        print(f"\n📊 RECOMMENDED FACTOR WEIGHTS FOR {result['regime_name'].upper()}")
        print("-"*40)

        sorted_weights = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)
        for factor, weight in sorted_weights:
            if weight != 0:
                print(f"  {factor:<20} {weight:>6.1f}%")

        # Convert numpy types to Python native types
        def convert_numpy_types(obj):
            """Convert numpy types to Python native types for JSON serialization"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        # Save results
        output = {
            'timestamp': datetime.now().isoformat(),
            'regime_detection': convert_numpy_types(result),
            'recommended_weights': convert_numpy_types(weights)
        }

        # Save results - update the path
        output_path = os.path.join(detector.output_dir, 'current_regime_analysis.json')
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n✅ Analysis saved to {output_path}")

        # with open('current_regime_analysis.json', 'w') as f:
        #     json.dump(output, f, indent=2)
        #
        # print(f"\n✅ Analysis saved to current_regime_analysis.json")

        return result, weights

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def get_current_regime_api():
    """API wrapper to get current regime"""
    # Your existing code already creates current_regime_analysis.json
    # Just run it and return the path
    # your_existing_function()  # Replace with your actual function

    import json
    with open("Analysis/current_regime_analysis.json", 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    import sys

    # Check if script is being called with API flag
    if "--api" in sys.argv:
        # Run in API mode (returns JSON-friendly output)
        detector = RegimeDetectorAPI()
        result = detector.run_detection()
        print(json.dumps(result))
    else:
        # Run your existing code normally
        # your_existing_main_function()
        pass

    main()