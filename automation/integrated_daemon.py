#!/usr/bin/env python3
"""
Integrated MSAS Automation Daemon
Runs the complete pipeline continuously on schedule
"""

import sys
import os
import time
import signal
import logging
from datetime import datetime, time as dt_time
from pathlib import Path
import schedule
import pytz

# Setup paths
AUTOMATION_DIR = Path(__file__).parent
PROJECT_ROOT = AUTOMATION_DIR.parent

# Add to path
sys.path.insert(0, str(AUTOMATION_DIR))

# Import automation modules
from integrated_automation import IntegratedMSASAutomation
from config_loader import get_schedule_config

# Global flag for graceful shutdown
running = True


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global running
    logging.info(f"Received signal {signum}. Shutting down...")
    running = False


# Setup signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class IntegratedDaemon:
    """Daemon to run integrated pipeline on schedule"""

    def __init__(self, run_time="08:30", timezone="Asia/Taipei", skip_market_check=False):
        """Initialize daemon"""
        self.automation = IntegratedMSASAutomation()
        self.run_time = run_time
        self.timezone = pytz.timezone(timezone)
        self.last_run_date = None
        self.skip_market_check = skip_market_check

    # def run_pipeline_job(self):
    #     """Job to run the complete pipeline"""
    #     try:
    #         current_date = datetime.now().date()
    #
    #         # Avoid running multiple times on the same day
    #         if self.last_run_date == current_date:
    #             logging.info(f"Already ran today ({current_date})")
    #             return
    #
    #         # Check if market was open yesterday (unless skipping check)
    #         if not self.skip_market_check:
    #             if not self.automation.should_run_today():
    #                 logging.info("Market was closed yesterday. Skipping scheduled run.")
    #                 return
    #
    #         # Check if already executed today by checking status files
    #         if self.check_already_executed_today():
    #             logging.info("Pipeline already executed today. Skipping.")
    #             self.last_run_date = current_date
    #             return
    #
    #         logging.info("=" * 60)
    #         logging.info(f"SCHEDULED PIPELINE RUN - {datetime.now()}")
    #         logging.info("=" * 60)
    #
    #         status = self.automation.run_complete_pipeline()
    #
    #         if status.get('success'):
    #             self.last_run_date = current_date
    #             logging.info("Pipeline completed successfully")
    #         else:
    #             logging.warning("Pipeline completed with issues")
    #
    #     except Exception as e:
    #         logging.error(f"Error in scheduled job: {e}")
    #         import traceback
    #         logging.error(traceback.format_exc())

    def run_pipeline_job(self):
        """Job to run the complete pipeline"""
        try:
            current_date = datetime.now().date()

            # Avoid running multiple times on the same day
            if self.last_run_date == current_date:
                logging.info(f"Already checked today ({current_date})")
                return

            # IMPORTANT: Mark that we've checked today, regardless of whether we run
            # This prevents the scheduler from repeatedly trying to run
            self.last_run_date = current_date

            # Check if market was open yesterday (unless skipping check)
            if not self.skip_market_check:
                if not self.automation.should_run_today():
                    logging.info("Market was closed yesterday. Skipping scheduled run.")
                    logging.info(f"Next check will be tomorrow at {self.run_time}")
                    # Show next scheduled run
                    next_run = schedule.next_run()
                    if next_run:
                        logging.info(f"Next scheduled run: {next_run}")
                    return

            # Check if already executed today by checking status files
            if self.check_already_executed_today():
                logging.info("Pipeline already executed today (yesterday's data already processed). Skipping.")
                logging.info(f"Next check will be tomorrow at {self.run_time}")
                # Show next scheduled run
                next_run = schedule.next_run()
                if next_run:
                    logging.info(f"Next scheduled run: {next_run}")
                return

            # If we get here, we need to run the pipeline
            logging.info("=" * 60)
            logging.info(f"SCHEDULED PIPELINE RUN - {datetime.now()}")
            logging.info("=" * 60)

            status = self.automation.run_complete_pipeline()

            if status.get('success'):
                logging.info("Pipeline completed successfully")
                # Show next scheduled run
                next_run = schedule.next_run()
                if next_run:
                    logging.info(f"Next scheduled run: {next_run}")
            else:
                logging.warning("Pipeline completed with issues")

        except Exception as e:
            logging.error(f"Error in scheduled job: {e}")
            import traceback
            logging.error(traceback.format_exc())
            # Even on error, mark that we tried today to avoid infinite retries
            self.last_run_date = datetime.now().date()

    # def check_already_executed_today(self):
    #     """Check if the pipeline has already been executed today"""
    #     try:
    #         # Check integrated status file
    #         status_file = self.automation.status_file
    #         if status_file.exists():
    #             with open(status_file, 'r') as f:
    #                 status = json.load(f)
    #
    #             # Check if it was run today
    #             if 'start_time' in status:
    #                 run_date = datetime.fromisoformat(status['start_time']).date()
    #                 today = datetime.now().date()
    #
    #                 if run_date == today:
    #                     # Check if scoring was successful
    #                     scoring_status = status.get('multifactor_scoring', {})
    #                     if scoring_status.get('success', False):
    #                         logging.info("Scoring already completed successfully today")
    #                         return True
    #                     else:
    #                         logging.info("Scoring was attempted but failed today - will retry")
    #                         return False
    #
    #         return False
    #
    #     except Exception as e:
    #         logging.debug(f"Error checking execution status: {e}")
    #         return False

    def check_already_executed_today(self):
        """
        Simplified and more robust version.
        Check if we have already processed the most recent trading day's data.
        """
        try:
            import json
            import pandas_market_calendars as mcal
            from datetime import timedelta
            import pandas as pd

            # Check integrated status file
            status_file = self.automation.status_file
            if not status_file.exists():
                logging.info("No status file found - need to run pipeline")
                return False

            with open(status_file, 'r') as f:
                status = json.load(f)

            # Get NYSE calendar to find last trading day
            nyse = mcal.get_calendar('NYSE')
            today = datetime.now()
            yesterday = today - timedelta(days=1)

            # Find the most recent trading day (could be yesterday or earlier if weekend/holiday)
            start = yesterday - pd.Timedelta(days=10)
            end = yesterday + pd.Timedelta(days=1)
            valid_days = nyse.valid_days(start_date=start, end_date=end)

            if len(valid_days) == 0:
                logging.warning("No valid trading days found in the past 10 days")
                return False

            # Get the most recent trading day
            valid_days = pd.DatetimeIndex([d.tz_localize(None) if d.tz else d for d in valid_days])
            valid_days = valid_days[valid_days <= pd.Timestamp(yesterday).normalize()]

            if len(valid_days) == 0:
                logging.warning("No trading days found before today")
                return False

            last_trading_day = valid_days[-1].date()

            logging.info(f"Last trading day was: {last_trading_day}")

            # Check when we last ran successfully
            if 'start_time' not in status:
                logging.info("No previous run recorded - need to run pipeline")
                return False

            run_datetime = datetime.fromisoformat(status['start_time'])
            run_date = run_datetime.date()

            # Check if the last run successfully processed data
            scoring_status = status.get('multifactor_scoring', {})
            if not scoring_status.get('success', False):
                logging.info(f"Last run on {run_date} was not successful - will retry")
                return False

            # The critical logic:
            # If we ran on or after the last trading day + 1 (next day), we have that trading day's data
            # This accounts for running after market close (data becomes available next day at ~4-5 AM Asia time)
            day_after_last_trading = last_trading_day + timedelta(days=1)

            if run_date >= day_after_last_trading:
                # We ran on or after the day following the last trading day
                # This means we should have the last trading day's data
                logging.info(f"Already have data for last trading day ({last_trading_day})")
                logging.info(f"  (processed on {run_date})")
                return True
            elif run_date == last_trading_day:
                # We ran on the last trading day itself
                # Check if it was after market close (roughly 4 PM ET = 5 AM next day in Taipei)
                # For Asia/Taipei, if we ran after 5 AM, we might have yesterday's data
                run_hour = run_datetime.hour
                if run_hour >= 5:  # After 5 AM Taipei = after market close
                    # But this would be for the PREVIOUS trading day, not the last trading day
                    # So we still need to run
                    logging.info(f"Ran on {run_date} at {run_hour}:00 - need data for {last_trading_day}")
                    return False
                else:
                    logging.info(f"Ran on {run_date} before market close - need to run again")
                    return False
            else:
                # We ran before the last trading day
                logging.info(f"Last run was {run_date}, need data for {last_trading_day}")
                return False

        except Exception as e:
            logging.error(f"Error checking execution status: {e}")
            import traceback
            logging.error(traceback.format_exc())
            # On error, default to running the pipeline
            return False

    # def setup_schedule(self):
    #     """Setup the daily schedule"""
    #     logging.info(f"Setting up schedule to run at {self.run_time} {self.timezone}")
    #     schedule.every().day.at(self.run_time).do(self.run_pipeline_job)
    #
    #     next_run = schedule.next_run()
    #     if next_run:
    #         logging.info(f"Next scheduled run: {next_run}")

    def setup_schedule(self):
        """Setup the daily schedule with better logging"""
        logging.info(f"Setting up schedule to run at {self.run_time} {self.timezone}")

        # Clear any existing jobs (in case of restart)
        schedule.clear()

        # Schedule the daily job
        job = schedule.every().day.at(self.run_time).do(self.run_pipeline_job)

        # Log when the job will run
        next_run = schedule.next_run()
        if next_run:
            logging.info(f"Job scheduled successfully. Next run: {next_run}")
        else:
            logging.warning("Warning: No scheduled job found after setup")

        return job

    # def start(self):
    #     """Start the daemon"""
    #     logging.info("=" * 60)
    #     logging.info("INTEGRATED MSAS DAEMON STARTING")
    #     logging.info(f"Schedule: Daily at {self.run_time}")
    #     logging.info(f"Timezone: {self.timezone}")
    #     logging.info("Press Ctrl+C to stop")
    #     logging.info("=" * 60)
    #
    #     # Setup schedule
    #     self.setup_schedule()
    #
    #     # Check if should run immediately
    #     now = datetime.now()
    #     scheduled_time = datetime.strptime(self.run_time, "%H:%M").time()
    #     if now.time() > scheduled_time and self.last_run_date != now.date():
    #         logging.info("Past scheduled time. Running initial pipeline...")
    #         self.run_pipeline_job()
    #
    #     # Main loop
    #     while running:
    #         try:
    #             schedule.run_pending()
    #             time.sleep(60)  # Check every minute
    #
    #         except KeyboardInterrupt:
    #             logging.info("Keyboard interrupt received")
    #             break
    #         except Exception as e:
    #             logging.error(f"Unexpected error in main loop: {e}")
    #             time.sleep(60)
    #
    #     logging.info("Daemon stopped")

    def start(self):
        """Start the daemon with improved logging and scheduling"""
        logging.info("=" * 60)
        logging.info("INTEGRATED MSAS DAEMON STARTING")
        logging.info(f"Schedule: Daily at {self.run_time}")
        logging.info(f"Timezone: {self.timezone}")
        logging.info("Press Ctrl+C to stop")
        logging.info("=" * 60)

        # Setup schedule
        self.setup_schedule()

        # Check if should run immediately
        now = datetime.now()
        scheduled_time = datetime.strptime(self.run_time, "%H:%M").time()

        # Log the next scheduled run
        next_run = schedule.next_run()
        if next_run:
            logging.info(f"Next scheduled check: {next_run}")

        # Check if we're past today's scheduled time and haven't run yet
        if now.time() > scheduled_time and self.last_run_date != now.date():
            logging.info(f"Past scheduled time ({self.run_time}). Checking if pipeline needs to run...")
            self.run_pipeline_job()
        else:
            if now.time() <= scheduled_time:
                logging.info(f"Before today's scheduled time. Will run at {self.run_time}")
            else:
                logging.info(f"Already checked today. Next check tomorrow at {self.run_time}")

        # Main loop with periodic status updates
        last_status_time = datetime.now()
        status_interval_minutes = 60  # Log status every hour

        logging.info("Entering main scheduler loop...")

        while running:
            try:
                # Run any pending scheduled jobs
                schedule.run_pending()

                # Periodic status update
                current_time = datetime.now()
                if (current_time - last_status_time).total_seconds() > status_interval_minutes * 60:
                    next_run = schedule.next_run()
                    if next_run:
                        logging.info(f"[Status] Daemon running. Next scheduled check: {next_run}")
                    else:
                        logging.info("[Status] Daemon running. No scheduled jobs pending.")
                    last_status_time = current_time

                # Sleep for a short interval
                time.sleep(60)  # Check every minute

            except KeyboardInterrupt:
                logging.info("Keyboard interrupt received")
                break
            except Exception as e:
                logging.error(f"Unexpected error in main loop: {e}")
                import traceback
                logging.error(traceback.format_exc())
                time.sleep(60)  # Continue running even on errors

        logging.info("Daemon stopped")

def main():
    """Main entry point"""
    import argparse

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Try to load config
    try:
        config = get_schedule_config()
        default_time = config.get('time', '08:30')
        default_timezone = config.get('timezone', 'Asia/Taipei')
    except:
        default_time = '08:30'
        default_timezone = 'Asia/Taipei'

    parser = argparse.ArgumentParser(description='Integrated Pipeline Daemon')
    parser.add_argument('--time', default=default_time,
                        help=f'Time to run (HH:MM format, default: {default_time})')
    parser.add_argument('--timezone', default=default_timezone,
                        help=f'Timezone (default: {default_timezone})')
    parser.add_argument('--run-once', action='store_true',
                        help='Run once immediately and exit')
    parser.add_argument('--skip-market-check', action='store_true',
                        help='Skip market calendar check')

    args = parser.parse_args()

    if args.run_once:
        # Run once and exit
        automation = IntegratedMSASAutomation()

        # Check market unless skipping
        if not args.skip_market_check:
            if not automation.should_run_today():
                logging.info("Market was closed yesterday. Skipping run.")
                sys.exit(0)

        status = automation.run_complete_pipeline()
        sys.exit(0 if status.get('success') else 1)
    else:
        # # Start daemon
        # daemon = IntegratedDaemon(run_time=args.time, timezone=args.timezone)
        # daemon.start()

        # Start daemon with market check setting
        daemon = IntegratedDaemon(
            run_time=args.time,
            timezone=args.timezone,
            skip_market_check=args.skip_market_check
        )
        daemon.start()


if __name__ == "__main__":
    main()