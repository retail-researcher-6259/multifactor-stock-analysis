#!/usr/bin/env python3
"""
Market Regime Detection Daemon
Runs continuously and executes regime detection at 8AM local time on trading days
"""

import os
import sys

# Set UTF-8 environment variables
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

# Reconfigure stdout/stderr to use UTF-8 (Python 3.7+)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


import sys
import time
import signal
import logging
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
import schedule
import pytz

# Setup project paths
AUTOMATION_DIR = Path(__file__).parent
PROJECT_ROOT = AUTOMATION_DIR.parent

# Add project root to path
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(AUTOMATION_DIR))

# Import our automation module
from regime_detection_scheduler import RegimeDetectionAutomation, setup_logging

# Global flag for graceful shutdown
running = True


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global running
    logger.info(f"Received signal {signum}. Shutting down...")
    running = False


# Setup signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Setup logging
logger = setup_logging()


class RegimeDetectionDaemon:
    """Daemon to run regime detection on schedule"""

    def __init__(self, run_time="08:00", timezone="US/Eastern"):
        """
        Initialize daemon

        Args:
            run_time: Time to run detection (HH:MM format)
            timezone: Timezone for scheduling (default: US/Eastern for NYSE)
        """
        self.automation = RegimeDetectionAutomation()
        self.run_time = run_time
        self.timezone = pytz.timezone(timezone)
        self.last_run_date = None

    def run_detection_job(self):
        """Job to run regime detection"""
        try:
            current_date = datetime.now().date()

            # Avoid running multiple times on the same day
            if self.last_run_date == current_date:
                logger.info(f"Already ran today ({current_date})")
                return

            logger.info("=" * 60)
            logger.info(f"SCHEDULED RUN TRIGGERED - {datetime.now()}")
            logger.info("=" * 60)

            success = self.automation.run_full_detection()

            if success:
                self.last_run_date = current_date
                logger.info("Scheduled run completed successfully")
            else:
                logger.error("Scheduled run failed")

        except Exception as e:
            logger.error(f"Error in scheduled job: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def setup_schedule(self):
        """Setup the daily schedule"""
        logger.info(f"Setting up schedule to run at {self.run_time} {self.timezone}")

        # Schedule daily run at specified time
        schedule.every().day.at(self.run_time).do(self.run_detection_job)

        # Log next scheduled run
        next_run = schedule.next_run()
        if next_run:
            logger.info(f"Next scheduled run: {next_run}")

    def run_once_if_needed(self):
        """Run once on startup if conditions are met"""
        now = datetime.now()
        scheduled_time = datetime.strptime(self.run_time, "%H:%M").time()

        # If it's past the scheduled time and we haven't run today
        if now.time() > scheduled_time and self.last_run_date != now.date():
            logger.info("Past scheduled time. Running initial detection...")
            self.run_detection_job()

    def start(self):
        """Start the daemon"""
        logger.info("=" * 60)
        logger.info("REGIME DETECTION DAEMON STARTING")
        logger.info(f"Schedule: Daily at {self.run_time}")
        logger.info(f"Timezone: {self.timezone}")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 60)

        # Setup schedule
        self.setup_schedule()

        # Run once if we're past today's scheduled time
        self.run_once_if_needed()

        # Main loop
        while running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                time.sleep(60)

        logger.info("Daemon stopped")


def main():
    """Main entry point"""
    import argparse

    # Try to load config
    try:
        from config_loader import get_schedule_config
        schedule_config = get_schedule_config()
        default_time = schedule_config.get('time', '08:00')
        default_timezone = schedule_config.get('timezone', 'Asia/Taipei')
    except:
        default_time = '08:00'
        default_timezone = 'Asia/Taipei'

    parser = argparse.ArgumentParser(description='Regime Detection Daemon')
    parser.add_argument('--time', default=default_time,
                        help=f'Time to run detection (HH:MM format, default: {default_time})')
    parser.add_argument('--timezone', default=default_timezone,
                        help=f'Timezone for scheduling (default: {default_timezone})')
    parser.add_argument('--run-once', action='store_true',
                        help='Run once immediately and exit')

    args = parser.parse_args()

    # Create daemon
    daemon = RegimeDetectionDaemon(run_time=args.time, timezone=args.timezone)

    if args.run_once:
        # Run once and exit
        logger.info("Running once (--run-once flag)")
        success = daemon.automation.run_full_detection()
        sys.exit(0 if success else 1)
    else:
        # Start daemon
        daemon.start()


if __name__ == "__main__":
    main()