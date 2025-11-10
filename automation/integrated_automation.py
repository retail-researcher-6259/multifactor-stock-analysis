#!/usr/bin/env python3
"""
Integrated MSAS Automation
Runs complete workflow: Regime Detection → Multifactor Scoring
"""

import sys
import os
from pathlib import Path

# Fix Unicode issues on Windows
if sys.platform == 'win32':
    import locale

    # Try to set UTF-8 locale
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        try:
            locale.setlocale(locale.LC_ALL, '')
        except:
            pass

    # Set environment variables
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'

import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import json
from typing import Dict, Any, Optional

# Setup paths
AUTOMATION_DIR = Path(__file__).parent
PROJECT_ROOT = AUTOMATION_DIR.parent

# Add to path
sys.path.insert(0, str(AUTOMATION_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# Import automation modules
from regime_detection_scheduler import RegimeDetectionAutomation
from scoring_scheduler import MultifactorScoringAutomation
from trend_analysis_scheduler import ScoreTrendAnalysisAutomation

# Try to load config
try:
    from config_loader import load_config
    CONFIG = load_config()
except:
    CONFIG = {}

# Setup logging
LOG_DIR = AUTOMATION_DIR / 'logs'
LOG_DIR.mkdir(exist_ok=True)


# def setup_logging():
#     """Configure logging for integrated automation"""
#     log_file = LOG_DIR / f"integrated_{datetime.now().strftime('%Y%m%d')}.log"
#
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(log_file),
#             logging.StreamHandler()
#         ]
#     )
#     return logging.getLogger(__name__)

def setup_logging(log_name="integrated"):
    """Configure logging for integrated automation"""
    log_file = LOG_DIR / f"{log_name}_{datetime.now().strftime('%Y%m%d')}.log"

    # Get a unique logger for this module
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()

    # Add new handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()


class IntegratedMSASAutomation:
    """Complete MSAS automation pipeline"""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize integrated automation

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or CONFIG
        self.regime_automation = RegimeDetectionAutomation()
        # self.scoring_automation = MultifactorScoringAutomation()

        # Pass ticker file from config to scoring
        scoring_config = self.config.get('multifactor_scoring', {})
        ticker_file = scoring_config.get('ticker_file')
        self.scoring_automation = MultifactorScoringAutomation(ticker_file=ticker_file)

        self.trend_automation = ScoreTrendAnalysisAutomation()  # Add this line

        self.status_file = AUTOMATION_DIR / 'status' / 'integrated_status.json'

        # Create status directory
        self.status_file.parent.mkdir(exist_ok=True)

    def run_complete_pipeline(self, skip_technical_analysis: bool = True) -> Dict[str, Any]:
        """
        Run the complete MSAS pipeline:
        1. Regime Detection (historical + current)
        2. Multifactor Scoring (all three regimes)
        3. Score Trend Analysis (stability analysis for all regimes)

        Returns:
            Summary of the entire pipeline execution
        """
        start_time = datetime.now()

        logger.info("=" * 70)
        logger.info("INTEGRATED MSAS AUTOMATION PIPELINE")
        logger.info(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70)

        pipeline_status = {
            'start_time': start_time.isoformat(),
            'regime_detection': {},
            'multifactor_scoring': {},
            'trend_analysis': {},
            'success': False
        }

        # Initialize status dictionary
        status = {
            'start_time': start_time.isoformat(),
            'regime_detection': {},
            'multifactor_scoring': {},
            'trend_analysis': {},
            'success': True
        }

        try:
            # Step 1: Run Regime Detection
            logger.info("\n" + "=" * 60)
            logger.info("STEP 1: REGIME DETECTION")
            logger.info("=" * 60)

            # Check if regime detection already ran today
            regime_already_ran = self.regime_automation.load_last_status()
            if regime_already_ran:
                last_run = datetime.fromisoformat(regime_already_ran.get('last_run', '2000-01-01'))
                if last_run.date() == datetime.now().date() and regime_already_ran.get('success'):
                    # Already ran successfully today - mark as skipped
                    logger.info("Regime detection already completed today - skipping")
                    pipeline_status['regime_detection'] = {
                        'status': 'skipped',
                        'success': True,
                        'timestamp': datetime.now().isoformat(),
                        'note': f'Already ran at {last_run.strftime("%H:%M:%S")}'
                    }
                    regime_success = True
                else:
                    # Try to run
                    regime_success = self.regime_automation.run_full_detection()
                    pipeline_status['regime_detection'] = {
                        'status': 'executed' if regime_success else 'failed',
                        'success': regime_success,
                        'timestamp': datetime.now().isoformat()
                    }
            else:
                # No previous run, execute now
                regime_success = self.regime_automation.run_full_detection()
                pipeline_status['regime_detection'] = {
                    'status': 'executed' if regime_success else 'failed',
                    'success': regime_success,
                    'timestamp': datetime.now().isoformat()
                }

            if not regime_success:
                logger.error("Regime detection failed. Pipeline will continue but results may be incomplete.")

            # Step 2: Run Multifactor Scoring
            logger.info("\n" + "=" * 60)
            logger.info("STEP 2: MULTIFACTOR STOCK SCORING")
            logger.info("=" * 60)

            # Don't wait for regime detection since we just ran it
            scoring_summary = self.scoring_automation.run_all_regimes(wait_for_regime=False)
            pipeline_status['multifactor_scoring'] = scoring_summary

            if scoring_summary.get('success'):
                logger.info(" Multifactor scoring completed successfully")
            else:
                logger.error("Some scoring regimes failed. Check logs for details.")

            # Step 3: Run Score Trend Analysis
            logger.info("\n" + "=" * 60)
            logger.info("STEP 3: SCORE TREND ANALYSIS")
            logger.info("=" * 60)

            trend_status = self.trend_automation.run_all_regimes(skip_technical=skip_technical_analysis)
            status['trend_analysis'] = trend_status

            if trend_status.get('success'):  # Changed from trend_summary to trend_status
                logger.info(" Score trend analysis completed successfully")
            else:
                logger.error(" Score trend analysis failed")
                status['success'] = False

            # Overall success if all three completed
            # pipeline_status['success'] = (
            #         regime_success and
            #         scoring_summary.get('success', False) and
            #         trend_summary.get('success', False)
            # )

            pipeline_status['success'] = (
                    regime_success and
                    scoring_summary.get('success', False) and
                    trend_status.get('success', False)
            )

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            pipeline_status['error'] = str(e)

        finally:
            # Calculate total duration
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            pipeline_status['end_time'] = end_time.isoformat()
            pipeline_status['duration_seconds'] = duration
            pipeline_status['duration_readable'] = f"{duration / 60:.1f} minutes"

            # Save status
            self.save_status(pipeline_status)

            # Print summary
            self.print_summary(pipeline_status)

        return pipeline_status

    def print_summary(self, status: Dict[str, Any]):
        """Print execution summary"""
        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 70)

        # Regime Detection Summary
        regime_status = status.get('regime_detection', {})
        logger.info("\nRegime Detection:")
        # logger.info(f"  Status: {' SUCCESS' if regime_status.get('success') else ' FAILED'}")

        status_text = {
            'skipped': ' SKIPPED',
            'executed': ' SUCCESS' if regime_status.get('success') else ' FAILED',
            'failed': ' FAILED'
        }.get(regime_status.get('status', 'failed'), ' FAILED')
        logger.info(f"  Status: {status_text}")

        if regime_status.get('current_regime'):
            logger.info(f"  Current Regime: {regime_status['current_regime']}")
            logger.info(f"  Confidence: {regime_status.get('confidence', 0):.1%}")

        # Scoring Summary
        scoring_status = status.get('multifactor_scoring', {})
        if scoring_status:
            logger.info("\nMultifactor Scoring:")
            logger.info(f"  Status: {' SUCCESS' if scoring_status.get('success') else ' PARTIAL'}")
            logger.info(
                f"  Regimes Processed: {scoring_status.get('successful', 0)}/{scoring_status.get('total_regimes', 3)}")

            if scoring_status.get('successful_regimes'):
                logger.info(f"  Completed: {', '.join(scoring_status['successful_regimes'])}")
            if scoring_status.get('failed_regimes'):
                logger.error(f"  Failed: {', '.join(scoring_status['failed_regimes'])}")

        # Trend Analysis Summary
        trend_status = status.get('trend_analysis', {})
        if trend_status:
            logger.info("\nScore Trend Analysis:")
            logger.info(f"  Status: {' SUCCESS' if trend_status.get('success') else ' PARTIAL'}")
            logger.info(
                f"  Regimes Analyzed: {trend_status.get('successful', 0)}/{trend_status.get('total_regimes', 3)}")

            if trend_status.get('successful_regimes'):
                logger.info(f"  Completed: {', '.join(trend_status['successful_regimes'])}")
            if trend_status.get('failed_regimes'):
                logger.error(f"  Failed: {', '.join(trend_status['failed_regimes'])}")

        # Overall Status
        logger.info("\nOverall Pipeline:")
        # logger.info(f"  Status: {' COMPLETE' if status.get('success') else ' INCOMPLETE'}")

        # Consider pipeline complete if all required components ran or were skipped successfully
        regime_ok = status.get('regime_detection', {}).get('success', False)
        scoring_ok = status.get('multifactor_scoring', {}).get('success', False)
        overall_success = regime_ok and scoring_ok
        logger.info(f"  Status: {' COMPLETE' if overall_success else ' INCOMPLETE'}")

        logger.info(f"  Duration: {status.get('duration_readable', 'Unknown')}")
        logger.info(f"  Ended: {status.get('end_time', 'Unknown')}")

        logger.info("=" * 70)

    def save_status(self, status: Dict[str, Any]):
        """Save pipeline status"""
        try:
            with open(self.status_file, 'w') as f:
                json.dump(status, f, indent=2, default=str)
            logger.info(f"Pipeline status saved to {self.status_file}")
        except Exception as e:
            logger.error(f"Failed to save status: {e}")

    def should_run_today(self) -> bool:
        """Check if pipeline should run based on market calendar"""
        try:
            import pandas as pd
            import pandas_market_calendars as mcal
            from datetime import datetime, timedelta

            today = datetime.now()
            yesterday = today - timedelta(days=1)

            # Get NYSE calendar
            nyse = mcal.get_calendar('NYSE')

            # Check if yesterday was a trading day
            start = yesterday - pd.Timedelta(days=5)
            end = yesterday + pd.Timedelta(days=1)

            valid_days = nyse.valid_days(start_date=start, end_date=end)

            # Convert to timezone-naive for comparison
            valid_days = pd.DatetimeIndex([d.tz_localize(None) if d.tz else d for d in valid_days])
            yesterday_pd = pd.Timestamp(yesterday).normalize()

            if yesterday_pd not in valid_days.normalize():
                logger.info(f"Market was closed yesterday ({yesterday.strftime('%Y-%m-%d')}). Skipping pipeline.")
                return False

            logger.info(f"Market was open yesterday. Proceeding with pipeline.")
            return True

        except Exception as e:
            logger.warning(f"Could not check market calendar: {e}. Proceeding anyway.")
            return True  # Default to running if we can't check

    def check_should_run(self) -> bool:
        """Check if pipeline should run based on schedule and market"""
        # First check if already ran today
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    last_status = json.load(f)
                    last_run = datetime.fromisoformat(last_status.get('start_time', '2000-01-01'))

                    # Check if already ran today
                    if last_run.date() == datetime.now().date():
                        logger.info(f"Pipeline already ran today at {last_run.strftime('%H:%M:%S')}")
                        return False
            except Exception as e:
                logger.warning(f"Could not check last run: {e}")

        # Then check market calendar
        if not self.should_run_today():
            return False

        return True


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Integrated MSAS Automation Pipeline')
    parser.add_argument('--force', action='store_true',
                        help='Force run even if already ran today')
    parser.add_argument('--regime-only', action='store_true',
                        help='Run only regime detection')
    parser.add_argument('--scoring-only', action='store_true',
                        help='Run only multifactor scoring')
    parser.add_argument('--skip-market-check', action='store_true',
                        help='Skip market calendar check')
    parser.add_argument('--include-technical', action='store_true',
                        help='Include technical analysis in trend analysis (default: skip)')

    args = parser.parse_args()

    # Create automation
    automation = IntegratedMSASAutomation()

    # # Check if should run
    if not args.force and not automation.check_should_run():
        logger.info("Skipping run. Use --force to override.")
        sys.exit(0)

    # Check if should run (unless forced)
    if not args.force:
        if not args.skip_market_check and not automation.check_should_run():
            logger.info("Skipping run based on schedule/market check. Use --force to override.")
            sys.exit(0)

    # Run based on arguments
    if args.regime_only:
        logger.info("Running regime detection only")
        success = automation.regime_automation.run_full_detection()
    elif args.scoring_only:
        logger.info("Running multifactor scoring only")
        summary = automation.scoring_automation.run_all_regimes(wait_for_regime=False)
        success = summary.get('success', False)
    else:
        # Run complete pipeline
        # Use include_technical flag (inverse of skip_technical)
        status = automation.run_complete_pipeline(skip_technical_analysis=not args.include_technical)
        success = status.get('success', False)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()