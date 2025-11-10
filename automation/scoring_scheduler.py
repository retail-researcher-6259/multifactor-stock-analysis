#!/usr/bin/env python3
"""
Multifactor Stock Scoring Automation Scheduler
Runs scoring for all three regimes after regime detection completes
"""

import sys
import os

# Fix Unicode issues on Windows
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

import subprocess
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List, Optional

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


# def setup_logging():
#     """Configure logging for the scoring automation"""
#     log_file = LOG_DIR / f"scoring_{datetime.now().strftime('%Y%m%d')}.log"
#
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(log_file),
#             logging.StreamHandler()
#         ]
#     )
#     return logging.getLogger(__name__)


def setup_logging():
    """Configure logging for the scoring automation"""
    log_file = LOG_DIR / f"scoring_{datetime.now().strftime('%Y%m%d')}.log"

    # Create a unique logger for scoring
    logger = logging.getLogger('scoring_scheduler')
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()

    # Create handlers
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # IMPORTANT: Prevent propagation to avoid duplicate logs
    logger.propagate = False

    return logger


logger = setup_logging()


class MultifactorScoringAutomation:
    """Automate multifactor stock scoring system"""

    # Define the three regimes to process
    REGIMES = [
        "Steady_Growth",
        "Strong_Bull",
        "Crisis_Bear"  # Note: script uses underscore instead of slash
    ]

    # Alternative regime names (for compatibility)
    REGIME_MAPPINGS = {
        "Steady Growth": "Steady_Growth",
        "Strong Bull": "Strong_Bull",
        "Crisis/Bear": "Crisis_Bear",
        "Crisis": "Crisis_Bear",
        "Bear": "Crisis_Bear"
    }

    def __init__(self, ticker_file: Optional[str] = None):
        """
        Initialize scoring automation

        Args:
            ticker_file: Optional path to custom ticker file
        """
        self.scoring_script = SRC_DIR / 'scoring' / 'stock_Screener_MultiFactor_25_new.py'
        self.status_file = AUTOMATION_DIR / 'status' / 'scoring_status.json'

        # Handle ticker file - ensure it's a Path object
        if ticker_file:
            if isinstance(ticker_file, str):
                # Convert string to Path
                if not ticker_file.startswith('/') and not ':' in ticker_file:
                    # Relative path
                    self.ticker_file = PROJECT_ROOT / ticker_file
                else:
                    # Absolute path
                    self.ticker_file = Path(ticker_file)
            else:
                self.ticker_file = Path(ticker_file)
        else:
            # Default ticker file
            self.ticker_file = PROJECT_ROOT / 'config' / 'Buyable_stocks_0901.txt'

        # Create status directory
        self.status_file.parent.mkdir(exist_ok=True)

    def check_prerequisites(self) -> bool:
        """Check if all required files exist"""
        logger.info("Checking prerequisites...")

        # Check if scoring script exists
        if not self.scoring_script.exists():
            logger.error(f"Scoring script not found: {self.scoring_script}")
            return False

        # Check if ticker file exists
        if not self.ticker_file.exists():
            logger.error(f"Ticker file not found: {self.ticker_file}")
            logger.info(f"Please create {self.ticker_file} with list of tickers to analyze")
            return False

        # Count tickers in file
        try:
            with open(self.ticker_file, 'r') as f:
                tickers = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                logger.info(f"Found {len(tickers)} tickers in {self.ticker_file.name}")
        except Exception as e:
            logger.error(f"Error reading ticker file: {e}")
            return False

        # Check if regime detection results exist
        regime_analysis_file = OUTPUT_DIR / 'Regime_Detection_Analysis' / 'current_regime_analysis.json'
        if not regime_analysis_file.exists():
            logger.warning("Current regime analysis not found. Run regime detection first!")
            logger.warning(f"Expected file: {regime_analysis_file}")
            # Don't fail - we can still run scoring without current regime
        else:
            logger.info("Current regime analysis found")

        logger.info("All prerequisites met")
        return True

    def get_current_regime(self) -> Optional[str]:
        """Get the current detected regime"""
        try:
            regime_file = OUTPUT_DIR / 'Regime_Detection_Analysis' / 'current_regime_analysis.json'
            if regime_file.exists():
                with open(regime_file, 'r') as f:
                    data = json.load(f)
                    regime = data.get('regime_detection', {}).get('regime_name')
                    confidence = data.get('regime_detection', {}).get('confidence', 0)

                    logger.info(f"Current regime: {regime} (confidence: {confidence:.1%})")

                    # Map to standard regime name
                    if regime in self.REGIME_MAPPINGS:
                        return self.REGIME_MAPPINGS[regime]
                    return regime
        except Exception as e:
            logger.warning(f"Could not read current regime: {e}")

        return None

    # def run_scoring_for_regime(self, regime: str) -> Dict[str, Any]:
    #     """
    #     Run scoring for a specific regime
    #
    #     Args:
    #         regime: Regime name (e.g., "Steady_Growth")
    #
    #     Returns:
    #         Dict with results
    #     """
    #     logger.info(f"Running scoring for {regime} regime...")
    #
    #     # Build command
    #     cmd = [
    #         sys.executable,
    #         str(self.scoring_script),
    #         regime  # Pass regime as argument
    #     ]
    #
    #     logger.info(f"Command: {' '.join(cmd)}")
    #     logger.info(f"Using ticker file: {self.ticker_file}")
    #
    #     logger.info(f"Script exists: {self.scoring_script.exists()}")
    #     logger.info(f"Ticker file exists: {self.ticker_file.exists()}")
    #
    #     # Set environment to include ticker file path if custom
    #     env = os.environ.copy()
    #     env['PYTHONIOENCODING'] = 'utf-8'
    #
    #     # Always set the ticker file environment variable
    #     env['TICKER_FILE'] = str(self.ticker_file)
    #
    #     env['PYTHONUNBUFFERED'] = '1'  # Force unbuffered output
    #
    #     try:
    #         logger.info(f"Launching subprocess for {regime}...")
    #         # Run the scoring script
    #         result = subprocess.run(
    #             cmd,
    #             capture_output=True,
    #             text=True,
    #             encoding='utf-8',
    #             cwd=str(PROJECT_ROOT),
    #             timeout=7200,  # 30 minute timeout
    #             env=env
    #         )
    #
    #         # Check for success
    #         success = result.returncode == 0
    #
    #         # Log output
    #         if result.stdout:
    #             # Only log summary, not full output
    #             lines = result.stdout.split('\n')
    #             for line in lines:
    #                 if 'Saved' in line or 'Top' in line or 'Finished' in line:
    #                     logger.info(f"  {line}")
    #
    #         if result.stderr and ('Error:' in result.stderr or 'Traceback' in result.stderr):
    #             logger.error(f"Scoring error for {regime}:\n{result.stderr}")
    #             success = False
    #
    #         # Check if output file was created
    #         output_dir = OUTPUT_DIR / 'Ranked_Lists' / regime
    #         if output_dir.exists():
    #             # Find the most recent file
    #             csv_files = list(output_dir.glob('*.csv'))
    #             if csv_files:
    #                 latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
    #                 logger.info(f"Output saved to: {latest_file}")
    #
    #                 # Read and get statistics
    #                 try:
    #                     df = pd.read_csv(latest_file)
    #                     stats = {
    #                         'total_stocks': len(df),
    #                         'top_stock': df.iloc[0]['Ticker'] if len(df) > 0 else None,
    #                         'top_score': df.iloc[0]['Score'] if len(df) > 0 else None,
    #                         'avg_score': df['Score'].mean() if 'Score' in df.columns else None
    #                     }
    #
    #                     logger.info(f"  Analyzed {stats['total_stocks']} stocks")
    #                     logger.info(f"  Top stock: {stats['top_stock']} (score: {stats['top_score']})")
    #
    #                     return {
    #                         'success': True,
    #                         'regime': regime,
    #                         'output_file': str(latest_file),
    #                         'stats': stats,
    #                         'timestamp': datetime.now().isoformat()
    #                     }
    #                 except Exception as e:
    #                     logger.warning(f"Could not read output file: {e}")
    #
    #         if success:
    #             return {
    #                 'success': True,
    #                 'regime': regime,
    #                 'timestamp': datetime.now().isoformat()
    #             }
    #         else:
    #             return {
    #                 'success': False,
    #                 'regime': regime,
    #                 'error': 'Script failed',
    #                 'timestamp': datetime.now().isoformat()
    #             }
    #
    #     except subprocess.TimeoutExpired:
    #         logger.error(f"Scoring for {regime} timed out after 30 minutes")
    #         return {
    #             'success': False,
    #             'regime': regime,
    #             'error': 'Timeout',
    #             'timestamp': datetime.now().isoformat()
    #         }
    #     except Exception as e:
    #         logger.error(f"Failed to run scoring for {regime}: {e}")
    #         return {
    #             'success': False,
    #             'regime': regime,
    #             'error': str(e),
    #             'timestamp': datetime.now().isoformat()
    #         }

    # def run_scoring_for_regime(self, regime: str) -> Dict[str, Any]:
    #     """Run scoring for a specific regime with enhanced debugging"""
    #     logger.info(f"Running scoring for {regime} regime...")
    #
    #     # Prepare command
    #     cmd = [
    #         sys.executable,
    #         str(self.scoring_script),
    #         regime,
    #         '--ticker-file', str(self.ticker_file)  # Pass ticker file as argument
    #     ]
    #
    #     logger.info(f"Command: {' '.join(cmd)}")
    #     logger.info(f"Using ticker file: {self.ticker_file}")
    #     logger.info(f"Script exists: {self.scoring_script.exists()}")
    #     logger.info(f"Ticker file exists: {self.ticker_file.exists()}")
    #
    #     # Set environment variables
    #     env = os.environ.copy()
    #     env['PYTHONPATH'] = str(PROJECT_ROOT)
    #     env['TICKER_FILE'] = str(self.ticker_file)
    #     env['PYTHONUNBUFFERED'] = '1'  # Force unbuffered output
    #
    #     try:
    #         logger.info(f"Launching subprocess for {regime}...")
    #
    #         # Create the process with pipes
    #         process = subprocess.Popen(
    #             cmd,
    #             stdout=subprocess.PIPE,
    #             stderr=subprocess.PIPE,
    #             text=True,
    #             encoding='utf-8',
    #             cwd=str(PROJECT_ROOT),
    #             env=env,
    #             bufsize=1,  # Line buffered
    #             universal_newlines=True
    #         )
    #
    #         logger.info(f"Process started with PID: {process.pid}")
    #
    #         # Set up for real-time output monitoring
    #         start_time = datetime.now()
    #         timeout_seconds = 300  # 5 minutes for test file
    #         output_lines = []
    #         error_lines = []
    #         last_output_time = start_time
    #
    #         # Monitor the process
    #         import select
    #         import time
    #
    #         while True:
    #             # Check if process has ended
    #             poll_status = process.poll()
    #             if poll_status is not None:
    #                 logger.info(f"Process ended with return code: {poll_status}")
    #                 break
    #
    #             # Check for timeout
    #             elapsed = (datetime.now() - start_time).total_seconds()
    #             if elapsed > timeout_seconds:
    #                 logger.error(f"Timeout after {elapsed:.1f} seconds")
    #                 process.terminate()
    #                 time.sleep(2)
    #                 if process.poll() is None:
    #                     process.kill()
    #                 return {
    #                     'success': False,
    #                     'regime': regime,
    #                     'error': f'Timeout after {timeout_seconds} seconds'
    #                 }
    #
    #             # Try to read output (non-blocking)
    #             try:
    #                 # For Windows, we need a different approach
    #                 if sys.platform == 'win32':
    #                     # Check if there's output available
    #                     line = process.stdout.readline()
    #                     if line:
    #                         line = line.strip()
    #                         output_lines.append(line)
    #                         logger.info(f"  [{regime}] {line}")
    #                         last_output_time = datetime.now()
    #
    #                     # Also check stderr
    #                     err_line = process.stderr.readline()
    #                     if err_line:
    #                         err_line = err_line.strip()
    #                         error_lines.append(err_line)
    #                         logger.error(f"  [{regime} ERROR] {err_line}")
    #                 else:
    #                     # Unix-like systems can use select
    #                     readable, _, _ = select.select([process.stdout, process.stderr], [], [], 0.1)
    #
    #                     if process.stdout in readable:
    #                         line = process.stdout.readline()
    #                         if line:
    #                             line = line.strip()
    #                             output_lines.append(line)
    #                             logger.info(f"  [{regime}] {line}")
    #                             last_output_time = datetime.now()
    #
    #                     if process.stderr in readable:
    #                         err_line = process.stderr.readline()
    #                         if err_line:
    #                             err_line = err_line.strip()
    #                             error_lines.append(err_line)
    #                             logger.error(f"  [{regime} ERROR] {err_line}")
    #             except Exception as e:
    #                 logger.debug(f"Read exception (normal): {e}")
    #
    #             # Check for hanging (no output for 30 seconds)
    #             silence_duration = (datetime.now() - last_output_time).total_seconds()
    #             if silence_duration > 30:
    #                 logger.warning(f"No output from {regime} for {silence_duration:.0f} seconds...")
    #
    #                 # Every minute of silence, log status
    #                 if int(silence_duration) % 60 == 0:
    #                     logger.warning(f"Process {process.pid} still running but silent...")
    #
    #             # Small sleep to prevent CPU spinning
    #             time.sleep(0.1)
    #
    #         # Process has ended, get final output
    #         final_stdout, final_stderr = process.communicate(timeout=5)
    #         if final_stdout:
    #             output_lines.extend(final_stdout.strip().split('\n'))
    #         if final_stderr:
    #             error_lines.extend(final_stderr.strip().split('\n'))
    #
    #         success = process.returncode == 0
    #
    #         # Log final status
    #         logger.info(f"Process completed with return code: {process.returncode}")
    #         if error_lines:
    #             logger.error(f"Error output:\n" + '\n'.join(error_lines[-10:]))
    #
    #         # Check for output file...
    #         output_dir = OUTPUT_DIR / 'Ranked_Lists' / regime
    #         if output_dir.exists():
    #             # Find the most recent file
    #             csv_files = list(output_dir.glob('*.csv'))
    #             if csv_files:
    #                 latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
    #                 logger.info(f"Output saved to: {latest_file}")
    #
    #                 # Read and get statistics
    #                 try:
    #                     df = pd.read_csv(latest_file)
    #                     stats = {
    #                         'total_stocks': len(df),
    #                         'top_stock': df.iloc[0]['Ticker'] if len(df) > 0 else None,
    #                         'top_score': df.iloc[0]['Score'] if len(df) > 0 else None,
    #                         'avg_score': df['Score'].mean() if 'Score' in df.columns else None
    #                     }
    #
    #                     logger.info(f"  Analyzed {stats['total_stocks']} stocks")
    #                     logger.info(f"  Top stock: {stats['top_stock']} (score: {stats['top_score']})")
    #
    #                     return {
    #                         'success': True,
    #                         'regime': regime,
    #                         'output_file': str(latest_file),
    #                         'stats': stats,
    #                         'timestamp': datetime.now().isoformat()
    #                     }
    #                 except Exception as e:
    #                     logger.warning(f"Could not read output file: {e}")
    #
    #         if success:
    #             return {
    #                 'success': True,
    #                 'regime': regime,
    #                 'timestamp': datetime.now().isoformat()
    #             }
    #         else:
    #             return {
    #                 'success': False,
    #                 'regime': regime,
    #                 'error': 'Script failed',
    #                 'timestamp': datetime.now().isoformat()
    #             }
    #
    #     except subprocess.TimeoutExpired:
    #         logger.error(f"Scoring for {regime} timed out after 30 minutes")
    #         return {
    #             'success': False,
    #             'regime': regime,
    #             'error': 'Timeout',
    #             'timestamp': datetime.now().isoformat()
    #         }
    #     except Exception as e:
    #         logger.error(f"Failed to run scoring for {regime}: {e}")
    #         return {
    #             'success': False,
    #             'regime': regime,
    #             'error': str(e),
    #             'timestamp': datetime.now().isoformat()
    #         }

    def run_scoring_for_regime(self, regime: str) -> Dict[str, Any]:
        """Run scoring for a specific regime with real-time output streaming"""
        logger.info(f"Running scoring for {regime} regime...")

        # Prepare command with ticker file argument
        cmd = [
            sys.executable,
            str(self.scoring_script),
            regime,
            '--ticker-file', str(self.ticker_file)
        ]

        logger.info(f"Command: {' '.join(cmd)}")
        logger.info(f"Using ticker file: {self.ticker_file}")

        # Set environment variables
        env = os.environ.copy()
        env['PYTHONPATH'] = str(PROJECT_ROOT)
        env['PYTHONUNBUFFERED'] = '1'  # Force unbuffered output

        try:
            start_time = datetime.now()

            # Determine timeout based on ticker file
            if 'test' in str(self.ticker_file).lower():
                timeout_seconds = 300  # 5 minutes for test file
            else:
                timeout_seconds = 7200  # 2 hours for full file

            logger.info(f"Starting subprocess with {timeout_seconds}s timeout...")

            # Create process with proper output handling for Windows
            if sys.platform == 'win32':
                # Windows-specific: use CREATE_NO_WINDOW to avoid popup
                import subprocess
                from subprocess import CREATE_NO_WINDOW

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # Combine stderr with stdout
                    text=True,
                    encoding='utf-8',
                    cwd=str(PROJECT_ROOT),
                    env=env,
                    bufsize=1,
                    universal_newlines=True,
                    creationflags=CREATE_NO_WINDOW if 'CREATE_NO_WINDOW' in dir(subprocess) else 0
                )
            else:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    cwd=str(PROJECT_ROOT),
                    env=env,
                    bufsize=1,
                    universal_newlines=True
                )

            logger.info(f"Process started with PID: {process.pid}")

            # Track progress
            last_output_time = start_time
            output_lines = []
            ticker_count = 0
            total_tickers = 0

            # Read output line by line
            while True:
                # Check if process has ended
                poll_status = process.poll()
                if poll_status is not None:
                    logger.info(f"Process ended with return code: {poll_status}")
                    break

                # Check timeout
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > timeout_seconds:
                    logger.error(f"Timeout after {elapsed / 60:.1f} minutes")
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    return {
                        'success': False,
                        'regime': regime,
                        'error': f'Timeout after {timeout_seconds / 60:.0f} minutes',
                        'timestamp': datetime.now().isoformat()
                    }

                # Read line (with timeout to prevent hanging)
                try:
                    line = process.stdout.readline()
                    if line:
                        line = line.strip()
                        if line:  # Skip empty lines
                            output_lines.append(line)
                            last_output_time = datetime.now()

                            # Parse progress information
                            if '/' in line and 'scraping' in line.lower():
                                # Extract ticker progress like "[106/999] scraping G"
                                try:
                                    parts = line.split(']')[0].split('[')[1].split('/')
                                    ticker_count = int(parts[0])
                                    total_tickers = int(parts[1])

                                    # Log every 10 tickers or important milestones
                                    if ticker_count % 10 == 0 or ticker_count in [1, total_tickers]:
                                        progress_pct = (ticker_count / total_tickers) * 100
                                        eta_seconds = (elapsed / ticker_count) * (
                                                    total_tickers - ticker_count) if ticker_count > 0 else 0
                                        eta_minutes = eta_seconds / 60

                                        logger.info(f"  [{regime}] Progress: {ticker_count}/{total_tickers} "
                                                    f"({progress_pct:.1f}%) - ETA: {eta_minutes:.1f} min")
                                        logger.info(f"  [{regime}] {line}")
                                except:
                                    pass

                            # Log important messages
                            elif any(keyword in line.lower() for keyword in ['error', 'warning', 'failed']):
                                logger.warning(f"  [{regime}] {line}")
                            elif any(keyword in line.lower() for keyword in ['saved', 'finished', 'complete', 'top']):
                                logger.info(f"  [{regime}] {line}")
                            elif 'successfully added' in line.lower():
                                # Log successful additions less frequently
                                if ticker_count % 50 == 0:
                                    logger.debug(f"  [{regime}] {line}")

                except Exception as e:
                    # Timeout or read error - check if process is still alive
                    if process.poll() is None:
                        # Still running, check for silence
                        silence_duration = (datetime.now() - last_output_time).total_seconds()
                        if silence_duration > 60:
                            logger.warning(f"No output from {regime} for {silence_duration:.0f}s. "
                                           f"Last progress: {ticker_count}/{total_tickers if total_tickers else '?'}")
                            last_output_time = datetime.now()  # Reset to avoid spam

                    time.sleep(0.1)  # Small delay before retry

            # Process has ended - get final output
            remaining_output = process.stdout.read()
            if remaining_output:
                output_lines.extend(remaining_output.strip().split('\n'))

            success = process.returncode == 0

            # Check for output file and get statistics
            if success:
                output_dir = OUTPUT_DIR / 'Ranked_Lists' / regime
                if output_dir.exists():
                    csv_files = list(output_dir.glob('*.csv'))
                    if csv_files:
                        latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
                        logger.info(f"Output saved to: {latest_file}")

                        try:
                            df = pd.read_csv(latest_file)
                            stats = {
                                'total_stocks': len(df),
                                'top_stock': df.iloc[0]['Ticker'] if len(df) > 0 else None,
                                'top_score': df.iloc[0]['Score'] if len(df) > 0 else None,
                            }

                            logger.info(f"  Analyzed {stats['total_stocks']} stocks")
                            logger.info(f"  Top stock: {stats['top_stock']} (score: {stats['top_score']})")

                            return {
                                'success': True,
                                'regime': regime,
                                'output_file': str(latest_file),
                                'stats': stats,
                                'timestamp': datetime.now().isoformat()
                            }
                        except Exception as e:
                            logger.warning(f"Could not read output file: {e}")

            return {
                'success': success,
                'regime': regime,
                'error': 'Check logs for details' if not success else None,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to run scoring for {regime}: {e}")
            return {
                'success': False,
                'regime': regime,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def check_regime_detection_complete(self) -> bool:
        """Check if regime detection has completed today"""
        try:
            # Check regime detection status file
            regime_status_file = AUTOMATION_DIR / 'status' / 'regime_detection_status.json'
            if regime_status_file.exists():
                with open(regime_status_file, 'r') as f:
                    status = json.load(f)

                    # Check if successful and recent
                    if status.get('success'):
                        last_run = datetime.fromisoformat(status.get('last_run', '2000-01-01'))
                        if last_run.date() == datetime.now().date():
                            logger.info("Regime detection completed successfully today")
                            return True
                        else:
                            logger.info(f"Regime detection last run: {last_run.date()}")

            # Also check if current regime file exists and is recent
            regime_file = OUTPUT_DIR / 'Regime_Detection_Analysis' / 'current_regime_analysis.json'
            if regime_file.exists():
                # Check file modification time
                mtime = datetime.fromtimestamp(regime_file.stat().st_mtime)
                if mtime.date() == datetime.now().date():
                    logger.info("Current regime analysis is from today")
                    return True

        except Exception as e:
            logger.warning(f"Could not check regime detection status: {e}")

        return False

    def wait_for_regime_detection(self, max_wait_minutes: int = 60) -> bool:
        """
        Wait for regime detection to complete

        Args:
            max_wait_minutes: Maximum time to wait

        Returns:
            True if regime detection completed, False if timeout
        """
        import time

        logger.info("Waiting for regime detection to complete...")
        start_time = datetime.now()
        check_interval = 60  # Check every minute

        while (datetime.now() - start_time).total_seconds() < max_wait_minutes * 60:
            if self.check_regime_detection_complete():
                return True

            logger.info(f"Regime detection not complete. Waiting {check_interval} seconds...")
            time.sleep(check_interval)

        logger.warning(f"Timeout: Regime detection did not complete within {max_wait_minutes} minutes")
        return False

    def run_all_regimes(self, wait_for_regime: bool = True) -> Dict[str, Any]:
        """
        Run scoring for all three regimes

        Args:
            wait_for_regime: Whether to wait for regime detection to complete

        Returns:
            Summary of results
        """
        start_time = datetime.now()

        logger.info("=" * 60)
        logger.info(f"MULTIFACTOR SCORING AUTOMATION - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

        # Check prerequisites
        if not self.check_prerequisites():
            logger.error("Prerequisites check failed. Aborting.")
            return {'success': False, 'error': 'Prerequisites check failed'}

        # Wait for regime detection if requested
        if wait_for_regime:
            if not self.check_regime_detection_complete():
                logger.info("Regime detection not complete. Waiting...")
                if not self.wait_for_regime_detection():
                    logger.warning("Proceeding without waiting for regime detection")

        # Get current regime for reference
        current_regime = self.get_current_regime()
        if current_regime:
            logger.info(f"Current detected regime: {current_regime}")

        # Run scoring for each regime
        results = {}
        successful_regimes = []
        failed_regimes = []

        for i, regime in enumerate(self.REGIMES, 1):
            logger.info(f"\n[{i}/{len(self.REGIMES)}] Processing {regime} regime")
            logger.info("-" * 40)

            result = self.run_scoring_for_regime(regime)
            results[regime] = result

            if result['success']:
                successful_regimes.append(regime)
                logger.info(f" {regime} completed successfully")
            else:
                failed_regimes.append(regime)
                logger.error(f" {regime} failed: {result.get('error', 'Unknown error')}")

        # Calculate duration
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Create summary
        summary = {
            'timestamp': start_time.isoformat(),
            'duration_seconds': duration,
            'current_regime': current_regime,
            'total_regimes': len(self.REGIMES),
            'successful': len(successful_regimes),
            'failed': len(failed_regimes),
            'successful_regimes': successful_regimes,
            'failed_regimes': failed_regimes,
            'results': results,
            'success': len(failed_regimes) == 0
        }

        # Save status
        self.save_status(summary)

        # Log summary
        logger.info("\n" + "=" * 60)
        logger.info("SCORING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Successful: {len(successful_regimes)}/{len(self.REGIMES)}")
        if successful_regimes:
            logger.info(f"Completed: {', '.join(successful_regimes)}")
        if failed_regimes:
            logger.error(f"Failed: {', '.join(failed_regimes)}")
        logger.info("=" * 60)

        return summary

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


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Multifactor Stock Scoring Automation')
    parser.add_argument('--regime', type=str,
                        help='Run specific regime only (e.g., Steady_Growth)')
    parser.add_argument('--ticker-file', type=str,
                        help='Path to custom ticker file')
    parser.add_argument('--no-wait', action='store_true',
                        help='Do not wait for regime detection to complete')

    args = parser.parse_args()

    # Create automation
    automation = MultifactorScoringAutomation(ticker_file=args.ticker_file)

    if args.regime:
        # Run single regime
        logger.info(f"Running scoring for {args.regime} only")
        result = automation.run_scoring_for_regime(args.regime)
        success = result['success']
    else:
        # Run all regimes
        summary = automation.run_all_regimes(wait_for_regime=not args.no_wait)
        success = summary['success']

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()