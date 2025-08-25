"""
backend/app.py - Fixed Flask API with correct paths
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
import subprocess

# Add parent directories to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'config'))

app = Flask(__name__)
CORS(app)  # Enable CORS for browser requests

# Configure paths
REGIME_ANALYSIS_DIR = project_root / "output" / "Regime_Detection_Analysis"
REGIME_RESULTS_DIR = project_root / "output" / "Regime_Detection_Results"
WEIGHT_RESULTS_DIR = project_root / "output" / "Weight_Optimization_Results"
DATA_DIR = project_root / "data"
UI_DIR = project_root / "UI"
CONFIG_DIR = project_root / "config"

# Ensure directories exist
for dir_path in [REGIME_ANALYSIS_DIR, REGIME_RESULTS_DIR, WEIGHT_RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# REGIME DETECTION ENDPOINTS
# ------------------------------------------------------------

@app.route('/api/regime/detect', methods=['POST'])
def detect_regime():
    """
    Endpoint to run regime detection
    """
    try:
        data = request.json or {}
        data_source = data.get('data_source', 'marketstack')

        print(f"Running regime detection with data source: {data_source}")

        # Call the regime detector script directly
        script_path = project_root / 'src' / 'regime_detection' / 'regime_detector.py'

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=str(project_root)
        )

        print(f"Script return code: {result.returncode}")
        if result.stdout:
            print(f"Script output: {result.stdout[:500]}")  # First 500 chars
        if result.stderr:
            print(f"Script error: {result.stderr[:500]}")

        # Read the generated files
        regime_data = {}

        # Read regime_periods.csv
        regime_file = REGIME_RESULTS_DIR / "regime_periods.csv"
        if regime_file.exists():
            df = pd.read_csv(regime_file)
            regime_data['periods'] = df.to_dict('records')
            print(f"Loaded {len(df)} regime periods")
        else:
            print(f"Warning: {regime_file} not found")

        # Read current_regime_analysis.json
        analysis_file = REGIME_ANALYSIS_DIR / "current_regime_analysis.json"
        if not analysis_file.exists():
            # Try the Results directory as fallback
            analysis_file = REGIME_RESULTS_DIR / "current_regime_analysis.json"

        if analysis_file.exists():
            with open(analysis_file, 'r') as f:
                regime_data['current'] = json.load(f)
            print("Loaded current regime analysis")
        else:
            print(f"Warning: current_regime_analysis.json not found")

        return jsonify({
            'status': 'success',
            'data': regime_data,
            'message': 'Regime detection completed'
        })

    except Exception as e:
        print(f"Error in detect_regime: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/regime/current', methods=['GET'])
def get_current_regime():
    """Get the current regime from saved analysis"""
    try:
        # Try multiple locations
        possible_paths = [
            REGIME_ANALYSIS_DIR / "current_regime_analysis.json",
            REGIME_RESULTS_DIR / "current_regime_analysis.json"
        ]

        for analysis_file in possible_paths:
            if analysis_file.exists():
                with open(analysis_file, 'r') as f:
                    data = json.load(f)

                # Handle different data formats
                if 'regime_detection' in data:
                    regime_info = data['regime_detection']
                    current_regime = regime_info.get('current_regime', 'Unknown')
                    probabilities = regime_info.get('regime_probabilities', {})
                else:
                    current_regime = data.get('current_regime', 'Unknown')
                    probabilities = data.get('regime_probabilities', {})

                return jsonify({
                    'status': 'success',
                    'current_regime': current_regime,
                    'probabilities': probabilities,
                    'last_updated': data.get('timestamp', 'Unknown')
                })

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
    """Get historical regime periods"""
    try:
        period = request.args.get('period', '30')

        regime_file = REGIME_RESULTS_DIR / "regime_periods.csv"
        if regime_file.exists():
            df = pd.read_csv(regime_file)

            # Calculate statistics
            if 'regime_name' in df.columns:
                stats = df['regime_name'].value_counts(normalize=True).to_dict()
            else:
                stats = {}

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
        # Check for model in Regime_Detection_Results
        model_file = REGIME_RESULTS_DIR / "regime_model.pkl"

        # Also check in Analysis directory as fallback
        if not model_file.exists():
            model_file = REGIME_ANALYSIS_DIR / "regime_model.pkl"

        if model_file.exists():
            # Read probabilities from current analysis
            analysis_file = REGIME_ANALYSIS_DIR / "current_regime_analysis.json"
            if analysis_file.exists():
                with open(analysis_file, 'r') as f:
                    data = json.load(f)

                    if 'regime_detection' in data:
                        probabilities = data['regime_detection'].get('regime_probabilities', {})
                    else:
                        probabilities = data.get('regime_probabilities', {})
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
                'message': 'Model not found. Please run regime detection first.'
            })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/regime/fetch-historical', methods=['POST'])
def fetch_historical_regime_data():
    """
    Fetch historical data from Marketstack and detect regimes
    """
    try:
        data = request.json or {}
        data_source = data.get('data_source', 'marketstack')
        years = data.get('years', 10)

        print(f"Fetching {years} years of historical data from {data_source}")

        # Run the regime detector with historical data fetching
        script_path = project_root / 'src' / 'regime_detection' / 'regime_detector.py'

        # Run with special flag to fetch historical data
        result = subprocess.run(
            [sys.executable, str(script_path), '--fetch-historical', f'--years={years}'],
            capture_output=True,
            text=True,
            cwd=str(project_root)
        )

        print(f"Script return code: {result.returncode}")
        if result.stdout:
            print(f"Script output: {result.stdout[:500]}")

        # Read the saved historical analysis
        historical_file = REGIME_ANALYSIS_DIR / "historical_regimes.json"
        data_summary_file = REGIME_ANALYSIS_DIR / "data_summary.json"

        response_data = {
            'status': 'success',
            'message': 'Historical data fetched and analyzed'
        }

        if historical_file.exists():
            with open(historical_file, 'r') as f:
                historical_data = json.load(f)
                response_data['statistics'] = historical_data.get('statistics', {})
                response_data['regime_periods'] = historical_data.get('regime_periods', [])

        if data_summary_file.exists():
            with open(data_summary_file, 'r') as f:
                summary = json.load(f)
                response_data['data_range'] = summary.get('data_range', {})
                response_data['symbols'] = summary.get('symbols', [])

        return jsonify(response_data)

    except Exception as e:
        print(f"Error fetching historical data: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/regime/historical-saved', methods=['GET'])
def get_saved_historical_regimes():
    """
    Get saved historical regime data without re-fetching
    """
    try:
        # Debug: Print the paths being checked
        print(f"Checking for historical data in: {REGIME_ANALYSIS_DIR}")

        # Read saved historical analysis
        historical_file = REGIME_ANALYSIS_DIR / "historical_regimes.json"
        data_summary_file = REGIME_ANALYSIS_DIR / "data_summary.json"

        print(f"Looking for historical_regimes.json at: {historical_file}")
        print(f"File exists: {historical_file.exists()}")

        if not historical_file.exists():
            # Also check if the files might be in Results directory
            alt_historical = REGIME_RESULTS_DIR / "historical_regimes.json"
            alt_summary = REGIME_RESULTS_DIR / "data_summary.json"

            print(f"Checking alternative location: {alt_historical}")
            print(f"Alternative exists: {alt_historical.exists()}")

            if alt_historical.exists():
                historical_file = alt_historical
                data_summary_file = alt_summary
            else:
                return jsonify({
                    'status': 'warning',
                    'message': 'No historical data found. Please fetch data first.'
                })

        # Read the historical data
        with open(historical_file, 'r') as f:
            historical_data = json.load(f)

        response_data = {
            'status': 'success',
            'statistics': historical_data.get('statistics', {}),
            'regime_periods': historical_data.get('regime_periods', []),
            'last_updated': historical_data.get('last_updated', 'Unknown')
        }

        # Add data range if available
        if data_summary_file.exists():
            print(f"Reading data summary from: {data_summary_file}")
            with open(data_summary_file, 'r') as f:
                summary = json.load(f)
                response_data['data_range'] = summary.get('data_range', {})
                response_data['symbols'] = summary.get('symbols', [])

        print(f"Returning historical data with {len(response_data.get('regime_periods', []))} periods")

        return jsonify(response_data)

    except Exception as e:
        print(f"Error in get_saved_historical_regimes: {str(e)}")
        import traceback
        traceback.print_exc()

        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/debug/files', methods=['GET'])
def debug_files():
    """Debug endpoint to check what files exist"""
    try:
        files_info = {
            'regime_analysis_dir': str(REGIME_ANALYSIS_DIR),
            'regime_results_dir': str(REGIME_RESULTS_DIR),
            'analysis_files': [],
            'results_files': []
        }

        # List files in Analysis directory
        if REGIME_ANALYSIS_DIR.exists():
            files_info['analysis_files'] = [f.name for f in REGIME_ANALYSIS_DIR.iterdir() if f.is_file()]

        # List files in Results directory
        if REGIME_RESULTS_DIR.exists():
            files_info['results_files'] = [f.name for f in REGIME_RESULTS_DIR.iterdir() if f.is_file()]

        return jsonify(files_info)

    except Exception as e:
        return jsonify({
            'error': str(e)
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

        # Call optimizer with regime parameter
        import subprocess
        result = subprocess.run(
            [sys.executable, str(project_root / 'src' / 'weight_optimization' / 'MultiFactor_optimizer_03.py'),
             '--regime', regime],
            capture_output=True,
            text=True,
            cwd=str(project_root)
        )

        if result.returncode != 0:
            print(f"Optimizer error: {result.stderr}")

        # Read results - check multiple possible locations
        regime_filename = regime.lower().replace(' ', '_').replace('/', '_')

        # Try Weight_Optimization_Results first
        results_file = WEIGHT_RESULTS_DIR / f"factor_analysis_results_{regime_filename}.txt"

        # Fallback to other output directories
        if not results_file.exists():
            results_file = project_root / "output" / f"factor_analysis_results_{regime_filename}.txt"

        weights = {}
        if results_file.exists():
            with open(results_file, 'r') as f:
                content = f.read()
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
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/weights/current', methods=['GET'])
def get_current_weights():
    """
    Get current factor weights for all regimes
    """
    try:
        # Default weights
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

            # Check multiple locations
            possible_paths = [
                WEIGHT_RESULTS_DIR / filename,
                project_root / "output" / filename
            ]

            for results_file in possible_paths:
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        content = f.read()
                        parsed_weights = parse_weights_from_results(content)
                        if parsed_weights:
                            weights[regime] = parsed_weights
                    break

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
    """
    weights = {}
    lines = content.split('\n')

    # Look for lines with factor weights
    in_weights_section = False
    for line in lines:
        if 'Optimized Factor Weights' in line or 'Factor Weights' in line:
            in_weights_section = True
            continue

        if in_weights_section and ':' in line:
            # Parse lines like "Value: 0.1200 (12.00%)"
            parts = line.split(':')
            if len(parts) == 2:
                factor = parts[0].strip()
                value_part = parts[1].strip()

                # Extract the decimal value
                try:
                    # Remove percentage part if present
                    if '(' in value_part:
                        value_part = value_part.split('(')[0].strip()
                    weight = float(value_part)

                    # Convert to decimal if it's a percentage
                    if weight > 1:
                        weight = weight / 100

                    weights[factor] = weight
                except ValueError:
                    continue

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
        'version': '1.0.0',
        'paths': {
            'regime_analysis': str(REGIME_ANALYSIS_DIR),
            'regime_results': str(REGIME_RESULTS_DIR),
            'weight_results': str(WEIGHT_RESULTS_DIR)
        }
    })

if __name__ == '__main__':
    print("=" * 50)
    print("Starting Multifactor Stock Analysis Backend")
    print(f"Project root: {project_root}")
    print(f"API running at: http://localhost:5000")
    print(f"Dashboard at: http://localhost:5000/")
    print("=" * 50)
    print("Output directories:")
    print(f"  Regime Analysis: {REGIME_ANALYSIS_DIR}")
    print(f"  Regime Results: {REGIME_RESULTS_DIR}")
    print(f"  Weight Results: {WEIGHT_RESULTS_DIR}")
    print("=" * 50)

    app.run(debug=True, port=5000, host='0.0.0.0')