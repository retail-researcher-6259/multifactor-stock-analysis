from pathlib import Path
import json

def test_file_locations():
    """Test script to verify file locations"""

    project_root = Path.cwd()

    print("=" * 60)
    print("FILE LOCATION TEST")
    print("=" * 60)

    # Check Analysis directory
    analysis_dir = project_root / "output" / "Regime_Detection_Analysis"
    print(f"\n📁 Analysis Directory: {analysis_dir}")
    print(f"   Exists: {analysis_dir.exists()}")

    if analysis_dir.exists():
        files = list(analysis_dir.glob("*"))
        print(f"   Files found: {len(files)}")
        for f in files:
            print(f"     - {f.name} ({f.stat().st_size} bytes)")

    # Check for specific files
    historical_file = analysis_dir / "historical_regimes.json"
    summary_file = analysis_dir / "data_summary.json"

    print(f"\n📄 historical_regimes.json:")
    print(f"   Path: {historical_file}")
    print(f"   Exists: {historical_file.exists()}")

    if historical_file.exists():
        with open(historical_file, 'r') as f:
            data = json.load(f)
            print(f"   Contains: {len(data.get('regime_periods', []))} periods")
            print(f"   Statistics: {list(data.get('statistics', {}).keys())}")

    print(f"\n📄 data_summary.json:")
    print(f"   Path: {summary_file}")
    print(f"   Exists: {summary_file.exists()}")

    if summary_file.exists():
        with open(summary_file, 'r') as f:
            data = json.load(f)
            dr = data.get('data_range', {})
            print(f"   Date range: {dr.get('start', 'N/A')} to {dr.get('end', 'N/A')}")
            print(f"   Days: {dr.get('days', 'N/A')}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_file_locations()