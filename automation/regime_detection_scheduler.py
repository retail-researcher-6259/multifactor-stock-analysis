#!/usr/bin/env python3
"""
Market Regime Detection Automation Scheduler
Runs regime detection scripts automatically at 8AM on trading days
"""

import sys
import os
import time
import subprocess
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import pandas_market_calendars as mcal
from typing import Optional, Dict, Any

# Setup project paths
AUTOMATION_DIR = Path(__file__).parent
PROJECT_ROOT = AUTOMATION_DIR.parent
SRC_DIR = PROJECT_ROOT / 'src'
OUTPUT_DIR = PROJECT_ROOT / 'output'

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

# Setup logging
LOG_DIR = AUTOMATION_DIR / 'logs'
LOG_DIR.mkdir(exist_ok=True)


def setup_logging():
    """Configure logging for the automation system"""
    log_file = LOG_DIR / f"regime_detection_{datetime.now().strftime('%Y%m%d')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


class MarketCalendar:
    """Handle market calendar and trading day checks"""

    def __init__(self, exchange='NYSE'):
        self.exchange = exchange
        self.calendar = mcal.get_calendar(exchange)

    def is_trading_day(self, date: datetime) -> bool:
        """Check if given date is a trading day"""
        # Convert to pandas timestamp
        pd_date = pd.Timestamp(date)

        # Get valid trading days for a range around the date
        start = pd_date - pd.Timedelta(days=5)
        end = pd_date + pd.Timedelta(days=5)

        try:
            valid_days = self.calendar.valid_days(start_date=start, end_date=end)
            # Convert to timezone-naive for comparison
            valid_days = pd.DatetimeIndex([d.tz_localize(None) if d.tz else d for d in valid_days])
            return pd_date.normalize() in valid_days.normalize()
        except Exception as e:
            logger.warning(f"Could not check trading calendar: {e}")
            # Fallback to simple weekend check
            return date.weekday() < 5

    def get_previous_trading_day(self, date: datetime) -> datetime:
        """Get the most recent trading day before the given date"""
        pd_date = pd.Timestamp(date)

        for i in range(1, 10):  # Check up to 10 days back
            check_date = pd_date - pd.Timedelta(days=i)
            if self.is_trading_day(check_date):
                return check_date.to_pydatetime()

        # Fallback
        return (date - timedelta(days=1))

    def was_market_open_yesterday(self, date: datetime) -> bool:
        """Check if market was open on the previous trading day"""
        yesterday = date - timedelta(days=1)
        return self.is_trading_day(yesterday)


class RegimeDetectionAutomation:
    """Automate regime detection system"""

    def __init__(self):
        self.calendar = MarketCalendar()
        self.regime_detector_script = SRC_DIR / 'regime_detection' / 'regime_detector.py'
        self.current_detector_script = SRC_DIR / 'regime_detection' / 'current_regime_detector.py'
        self.status_file = AUTOMATION_DIR / 'status' / 'regime_detection_status.json'

        # Create status directory
        self.status_file.parent.mkdir(exist_ok=True)

    def check_prerequisites(self) -> bool:
        """Check if all required scripts and dependencies exist"""
        logger.info("Checking prerequisites...")

        # Check if scripts exist
        if not self.regime_detector_script.exists():
            logger.error(f"Script not found: {self.regime_detector_script}")
            return False

        if not self.current_detector_script.exists():
            logger.error(f"Script not found: {self.current_detector_script}")
            return False

        logger.info("All prerequisites met")
        return True

    def run_script(self, script_path: Path, script_name: str, args: list = None) -> bool:
        """Run a Python script and capture output"""
        logger.info(f"Running {script_name}...")

        cmd = [sys.executable, str(script_path)]
        if args:
            cmd.extend(args)

        try:
            # Run the script
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                cwd=str(PROJECT_ROOT),
                timeout=600  # 10 minute timeout
            )

            # Log output
            if result.stdout:
                logger.info(f"{script_name} output:\n{result.stdout}")

            if result.stderr:
                # Check if it's a real error or just warnings
                if "Error:" in result.stderr or "Traceback" in result.stderr:
                    logger.error(f"{script_name} error:\n{result.stderr}")
                    return False
                else:
                    logger.warning(f"{script_name} warnings:\n{result.stderr}")

            # Check return code
            if result.returncode != 0:
                logger.error(f"{script_name} failed with return code {result.returncode}")
                return False

            logger.info(f"{script_name} completed successfully")
            return True

        except subprocess.TimeoutExpired:
            logger.error(f"{script_name} timed out after 10 minutes")
            return False
        except Exception as e:
            logger.error(f"Failed to run {script_name}: {e}")
            return False

    def run_historical_regime_detection(self) -> bool:
        """Run the historical regime detection script"""
        logger.info("=" * 60)
        logger.info("Starting Historical Regime Detection")
        logger.info("=" * 60)

        # Run with yfinance by default (add --use-marketstack flag if you have API key)
        success = self.run_script(
            self.regime_detector_script,
            "Historical Regime Detector",
            args=["--use-yfinance", "--years=10"]
        )

        if success:
            # Check if output files were created
            regime_periods = OUTPUT_DIR / 'Regime_Detection_Results' / 'regime_periods.csv'
            regime_model = OUTPUT_DIR / 'Regime_Detection_Results' / 'regime_model.pkl'

            if regime_periods.exists() and regime_model.exists():
                logger.info(f" Regime periods saved to: {regime_periods}")
                logger.info(f" Regime model saved to: {regime_model}")
                return True
            else:
                logger.error("Output files not generated")
                return False

        return False

    def run_current_regime_detection(self) -> bool:
        """Run the current regime detection script"""
        logger.info("=" * 60)
        logger.info("Starting Current Regime Detection")
        logger.info("=" * 60)

        success = self.run_script(
            self.current_detector_script,
            "Current Regime Detector"
        )

        if success:
            # Check if output file was created
            current_analysis = OUTPUT_DIR / 'Regime_Detection_Analysis' / 'current_regime_analysis.json'

            if current_analysis.exists():
                logger.info(f" Current analysis saved to: {current_analysis}")

                # Read and log the current regime
                try:
                    with open(current_analysis, 'r') as f:
                        data = json.load(f)
                        regime = data.get('regime_detection', {})
                        logger.info(f"Current Regime: {regime.get('regime_name', 'Unknown')}")
                        logger.info(f"Confidence: {regime.get('confidence', 0):.1%}")
                except Exception as e:
                    logger.warning(f"Could not read analysis file: {e}")

                return True
            else:
                logger.error("Current analysis file not generated")
                return False

        return False

    def save_status(self, status: Dict[str, Any]):
        """Save execution status to file"""
        try:
            with open(self.status_file, 'w') as f:
                json.dump(status, f, indent=2, default=str)
            logger.info(f"Status saved to {self.status_file}")
        except Exception as e:
            logger.error(f"Failed to save status: {e}")

    def load_last_status(self) -> Optional[Dict[str, Any]]:
        """Load last execution status"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load last status: {e}")
        return None

    def should_run_today(self) -> bool:
        """Check if we should run today based on market calendar and last run"""
        today = datetime.now()

        # Check if market was open yesterday
        if not self.calendar.was_market_open_yesterday(today):
            logger.info(f"Market was closed yesterday. Skipping today's run.")
            return False

        # Check last run status
        last_status = self.load_last_status()
        if last_status:
            last_run = datetime.fromisoformat(last_status.get('last_run', '2000-01-01'))
            if last_run.date() == today.date():
                logger.info(f"Already ran today at {last_run.strftime('%H:%M:%S')}")
                return False

        return True

    def run_full_detection(self) -> bool:
        """Run the complete regime detection workflow"""
        start_time = datetime.now()

        logger.info("=" * 60)
        logger.info(f"REGIME DETECTION AUTOMATION - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

        # Check prerequisites
        if not self.check_prerequisites():
            logger.error("Prerequisites check failed. Aborting.")
            return False

        # Check if we should run today
        if not self.should_run_today():
            return False

        status = {
            'last_run': start_time.isoformat(),
            'historical_detection': False,
            'current_detection': False,
            'success': False,
            'duration_seconds': 0
        }

        try:
            # Step 1: Run historical regime detection
            if self.run_historical_regime_detection():
                status['historical_detection'] = True

                # Step 2: Run current regime detection
                if self.run_current_regime_detection():
                    status['current_detection'] = True
                    status['success'] = True

            # Calculate duration
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            status['duration_seconds'] = duration

            logger.info("=" * 60)
            logger.info(f"EXECUTION COMPLETED - Duration: {duration:.2f} seconds")
            logger.info(f"Status: {'SUCCESS' if status['success'] else 'FAILED'}")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Unexpected error during execution: {e}")
            import traceback
            logger.error(traceback.format_exc())
            status['error'] = str(e)

        finally:
            # Save status
            self.save_status(status)

        return status['success']


def main():
    """Main entry point for manual execution"""
    automation = RegimeDetectionAutomation()
    success = automation.run_full_detection()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()