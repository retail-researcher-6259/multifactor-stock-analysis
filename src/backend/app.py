"""
backend/app.py - Flask API to bridge Python scripts with HTML dashboard
Place this in src/backend/app.py
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import json
import os
import sys
from pathlib import Path
import pandas as pd
import pickle
from datetime import datetime
import traceback

# Add parent directories to path so we can import your scripts
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Import your existing scripts (adjust paths as needed)
try:
    from regime_detection import regime_detector
    from regime_detection import current_regime_detector
    from weight_optimization import MultiFactor_optimizer_03 as optimizer
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Make sure your scripts are in the correct folders")

app = Flask(__name__)
CORS(app)  # Enable CORS for browser requests

# Configure paths (adjust these to match your folder structure)
ANALYSIS_DIR = project_root / "Analysis"
RESULTS_DIR = project_root / "Results"
DATA_DIR = project_root / "data"
UI_DIR = project_root / "ui"

# Ensure directories exist
for dir_path in [ANALYSIS_DIR, RESULTS_DIR, DATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# REGIME DETECTION ENDPOINTS
# ------------------------------------------------------------

@app.route('/api/regime/detect', methods=['POST'])
def detect_regime():
    """
    Endpoint to run regime detection
    Calls your existing regime_detector.py functions
    """
    try:
        # Get parameters from request
        data = request.json or {}
        data_source = data.get('data_source', 'marketstack')

        # Call your existing regime detection function
        # You'll need to modify this based on your actual function names
        # Example:
        # result = regime_detector.detect_current_regime(data_source)

        # For now, simulate calling your script
        import subprocess
        result = subprocess.run(
            [sys.executable, str(project_root / 'src' / 'regime_detection' / 'regime_detector.py')],
            capture_output=True,
            text=True
        )

        # Read the generated files
        regime_data = {}

        # Read regime_periods.csv if it exists
        regime_file = ANALYSIS_DIR / "regime_periods.csv"
        if regime_file.exists():
            df = pd.read_csv(regime_file)
            regime_data['periods'] = df.to_dict('records')

        # Read current_regime_analysis.json if it exists
        analysis_file = ANALYSIS_DIR / "current_regime_analysis.json"
        if analysis_file.exists():
            with open(analysis_file, 'r') as f:
                regime_data['current'] = json.load(f)

        return jsonify({
            'status': 'success',
            'data': regime_data,
            'message': 'Regime detection completed'
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/regime/current', methods=['GET'])
def get_current_regime():
    """
    Get the current regime from saved analysis
    """
    try:
        # Read from your current_regime_analysis.json
        analysis_file = ANALYSIS_DIR / "current_regime_analysis.json"

        if analysis_file.exists():
            with open(analysis_file, 'r') as f:
                data = json.load(f)

            # Extract key information
            current_regime = data.get('current_regime', 'Unknown')
            probabilities = data.get('regime_probabilities', {})

            return jsonify({
                'status': 'success',
                'current_regime': current_regime,
                'probabilities': probabilities,
                'last_updated': data.get('timestamp', 'Unknown')
            })
        else:
            return jsonify({
                'status': 'warning',
                'message': 'No regime analysis found. Please run detection first.'
            })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/regime/historical', methods=['GET'])
def get_historical_regimes():
    """
    Get historical regime periods
    """
    try:
        # Get time period from query parameters
        period = request.args.get('period', '30')  # days

        regime_file = ANALYSIS_DIR / "regime_periods.csv"
        if regime_file.exists():
            df = pd.read_csv(regime_file)

            # Filter by period if needed
            if period != 'all':
                # Add date filtering logic here based on your data structure
                pass

            # Calculate regime statistics
            stats = df['regime'].value_counts(normalize=True).to_dict()

            return jsonify({
                'status': 'success',
                'periods': df.to_dict('records'),
                'statistics': stats
            })
        else:
            return jsonify({
                'status': 'warning',
                'message': 'No historical data found'
            })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/regime/probabilities', methods=['GET'])
def get_regime_probabilities():
    """
    Get regime probabilities from the model
    """
    try:
        model_file = ANALYSIS_DIR / "regime_model.pkl"
        if model_file.exists():
            with open(model_file, 'rb') as f:
                model = pickle.load(f)

            # Get probabilities from your model
            # This depends on your model structure
            # Example:
            # probs = model.predict_proba(latest_data)

            # For now, return from the JSON file
            analysis_file = ANALYSIS_DIR / "current_regime_analysis.json"
            if analysis_file.exists():
                with open(analysis_file, 'r') as f:
                    data = json.load(f)
                    probabilities = data.get('regime_probabilities', {
                        'Steady Growth': 0.68,
                        'Strong Bull': 0.22,
                        'Crisis/Bear': 0.10
                    })
            else:
                probabilities = {
                    'Steady Growth': 0.33,
                    'Strong Bull': 0.33,
                    'Crisis/Bear': 0.34
                }

            return jsonify({
                'status': 'success',
                'probabilities': probabilities
            })
        else:
            return jsonify({
                'status': 'warning',
                'message': 'Model not found'
            })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ------------------------------------------------------------
# WEIGHT OPTIMIZATION ENDPOINTS
# ------------------------------------------------------------

@app.route('/api/weights/optimize', methods=['POST'])
def optimize_weights():
    """
    Run weight optimization for a specific regime
    """
    try:
        data = request.json or {}
        regime = data.get('regime', 'Steady Growth')
        backtest_period = data.get('backtest_period', 12)
        sample_companies = data.get('sample_companies', 150)

        # Call your optimizer script
        # You'll need to modify based on your actual function
        # Example:
        # results = optimizer.optimize_for_regime(regime, backtest_period, sample_companies)

        # For now, simulate the call
        import subprocess
        result = subprocess.run(
            [sys.executable, str(project_root / 'src' / 'weight_optimization' / 'MultiFactor_optimizer_03.py'),
             '--regime', regime],
            capture_output=True,
            text=True
        )

        # Read the generated results
        results_file = RESULTS_DIR / f"factor_analysis_results_{regime.lower().replace(' ', '_').replace('/', '_')}.txt"

        weights = {}
        if results_file.exists():
            with open(results_file, 'r') as f:
                content = f.read()
                # Parse your results file to extract weights
                # This depends on your file format
                weights = parse_weights_from_results(content)

        return jsonify({
            'status': 'success',
            'regime': regime,
            'weights': weights,
            'message': f'Optimization completed for {regime}'
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/weights/current', methods=['GET'])
def get_current_weights():
    """
    Get current factor weights for all regimes
    """
    try:
        # Read weights from your saved files or configuration
        weights = {
            'Steady Growth': {
                'Value': 0.12, 'Quality': 0.10, 'FinancialHealth': 0.08,
                'Technical': 0.09, 'Insider': 0.07, 'Momentum': 0.11,
                'Stability': 0.09, 'Size': 0.06, 'Credit': 0.07,
                'Liquidity': 0.08, 'Carry': 0.06, 'Growth': 0.07
            },
            'Strong Bull': {
                'Value': 0.08, 'Quality': 0.09, 'FinancialHealth': 0.07,
                'Technical': 0.11, 'Insider': 0.08, 'Momentum': 0.13,
                'Stability': 0.06, 'Size': 0.07, 'Credit': 0.06,
                'Liquidity': 0.09, 'Carry': 0.08, 'Growth': 0.08
            },
            'Crisis/Bear': {
                'Value': 0.15, 'Quality': 0.13, 'FinancialHealth': 0.12,
                'Technical': 0.06, 'Insider': 0.09, 'Momentum': 0.05,
                'Stability': 0.11, 'Size': 0.08, 'Credit': 0.10,
                'Liquidity': 0.07, 'Carry': 0.04, 'Growth': 0.00
            }
        }

        # Try to read from actual results files
        for regime in weights.keys():
            filename = f"factor_analysis_results_{regime.lower().replace(' ', '_').replace('/', '_')}.txt"
            results_file = RESULTS_DIR / filename
            if results_file.exists():
                with open(results_file, 'r') as f:
                    content = f.read()
                    parsed_weights = parse_weights_from_results(content)
                    if parsed_weights:
                        weights[regime] = parsed_weights

        return jsonify({
            'status': 'success',
            'weights': weights
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ------------------------------------------------------------
# UTILITY FUNCTIONS
# ------------------------------------------------------------

def parse_weights_from_results(content):
    """
    Parse weights from your results text file
    Modify this based on your actual file format
    """
    weights = {}
    # Example parsing logic - adjust based on your file format
    lines = content.split('\n')
    for line in lines:
        if ':' in line and any(factor in line for factor in ['Value', 'Quality', 'Momentum']):
            parts = line.split(':')
            if len(parts) == 2:
                factor = parts[0].strip()
                try:
                    weight = float(parts[1].strip().rstrip('%')) / 100
                    weights[factor] = weight
                except ValueError:
                    pass
    return weights

# ------------------------------------------------------------
# SERVE UI FILES
# ------------------------------------------------------------

@app.route('/')
def serve_ui():
    """Serve the dashboard HTML"""
    return send_from_directory(UI_DIR, 'dashboard.html')

@app.route('/ui/<path:path>')
def serve_ui_files(path):
    """Serve UI static files"""
    return send_from_directory(UI_DIR, path)

# ------------------------------------------------------------
# HEALTH CHECK
# ------------------------------------------------------------

@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if the API is running"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

if __name__ == '__main__':
    print("=" * 50)
    print("Starting Multifactor Stock Analysis Backend")
    print(f"API running at: http://localhost:5000")
    print(f"Dashboard at: http://localhost:5000/")
    print("=" * 50)

    app.run(debug=True, port=5000)