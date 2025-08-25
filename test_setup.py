"""
test_setup.py - Test script to verify all paths and configs are correct
Save this in your project root and run it
"""

from pathlib import Path
import json
import sys


def test_setup():
    """Test if all required files and directories exist"""

    print("=" * 60)
    print("TESTING PROJECT SETUP")
    print("=" * 60)

    # Get project root
    project_root = Path.cwd()
    print(f"\n✓ Project root: {project_root}")

    # Test directory structure
    print("\n📁 Checking directories:")
    required_dirs = [
        "src/regime_detection",
        "src/weight_optimization",
        "src/backend",
        "output/Regime_Detection_Analysis",
        "output/Regime_Detection_Results",
        "output/Weight_Optimization_Results",
        "config",
        "UI",
        "data"
    ]

    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} - MISSING!")
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"    Created: {full_path}")

    # Test config files
    print("\n📄 Checking config files:")
    config_files = [
        "config/marketstack_config.json",
        "config/test_tickers_stocks.txt"
    ]

    for config_file in config_files:
        full_path = project_root / config_file
        if full_path.exists():
            print(f"  ✓ {config_file}")

            # Check API key if it's the marketstack config
            if "marketstack" in config_file:
                with open(full_path, 'r') as f:
                    config = json.load(f)
                    api_key = config.get('api_key', '')
                    if 'xxxxx' in api_key or not api_key:
                        print(f"    ⚠️  WARNING: Update your API key in {config_file}")
                    else:
                        print(f"    ✓ API key configured")
        else:
            print(f"  ✗ {config_file} - MISSING!")

            # Create example config files
            if "marketstack" in config_file:
                config = {
                    "api_key": "YOUR_API_KEY_HERE",
                    "description": "Replace with your actual Marketstack API key"
                }
                full_path.parent.mkdir(parents=True, exist_ok=True)
                with open(full_path, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"    Created template: {full_path}")
            elif "test_tickers" in config_file:
                tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
                full_path.parent.mkdir(parents=True, exist_ok=True)
                with open(full_path, 'w') as f:
                    f.write('\n'.join(tickers))
                print(f"    Created example tickers: {full_path}")

    # Test Python scripts
    print("\n🐍 Checking Python scripts:")
    scripts = [
        "src/regime_detection/regime_detector.py",
        "src/regime_detection/current_regime_detector.py",
        "src/weight_optimization/MultiFactor_optimizer_03.py",
        "src/backend/app.py",
        "UI/dashboard.html"
    ]

    for script in scripts:
        full_path = project_root / script
        if full_path.exists():
            print(f"  ✓ {script}")
        else:
            print(f"  ✗ {script} - MISSING!")

    # Test Python imports
    print("\n📦 Testing Python imports:")
    try:
        sys.path.insert(0, str(project_root / 'src'))

        # Try importing modules
        imports_to_test = [
            ("flask", "Flask"),
            ("flask_cors", "CORS"),
            ("pandas", "pandas"),
            ("numpy", "numpy"),
            ("yfinance", "yfinance"),
        ]

        for module_name, display_name in imports_to_test:
            try:
                __import__(module_name)
                print(f"  ✓ {display_name}")
            except ImportError:
                print(f"  ✗ {display_name} - NOT INSTALLED!")
                print(f"    Run: pip install {module_name}")
    except Exception as e:
        print(f"  Error testing imports: {e}")

    # Test Flask server
    print("\n🌐 Testing Flask server:")
    try:
        import requests
        response = requests.get("http://localhost:5000/api/health", timeout=2)
        if response.status_code == 200:
            print("  ✓ Flask server is running")
            data = response.json()
            print(f"    Version: {data.get('version', 'unknown')}")
        else:
            print("  ⚠️  Flask server responded with error")
    except:
        print("  ✗ Flask server is not running")
        print("    Run: python src/backend/app.py")

    print("\n" + "=" * 60)
    print("SETUP TEST COMPLETE")
    print("=" * 60)

    # Summary
    print("\nNext steps:")
    print("1. Update your API key in config/marketstack_config.json")
    print("2. Start the Flask server: python src/backend/app.py")
    print("3. Open dashboard in browser: http://localhost:5000")
    print("4. Click 'Run Detection' to test regime detection")


if __name__ == "__main__":
    test_setup()