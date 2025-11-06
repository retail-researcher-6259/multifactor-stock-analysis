#!/usr/bin/env python3
"""
Score Trend Analysis Automation Scheduler
Runs trend analysis for all three regimes after scoring completes
Analyzes stability and generates technical reports
"""

import sys
import os

# # Fix Unicode issues on Windows
# if sys.platform == 'win32':
#     os.system('chcp 65001 > nul')
#     import io
#
#     sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
#     sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
#     os.environ['PYTHONIOENCODING'] = 'utf-8'

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List, Optional
import numpy as np

# Setup project paths
AUTOMATION_DIR = Path(__file__).parent
PROJECT_ROOT = AUTOMATION_DIR.parent
SRC_DIR = PROJECT_ROOT / 'src'
OUTPUT_DIR = PROJECT_ROOT / 'output'

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR / 'trend_analysis'))

# Setup logging
LOG_DIR = AUTOMATION_DIR / 'logs'
LOG_DIR.mkdir(exist_ok=True)


# def setup_logging():
#     """Configure logging for trend analysis automation"""
#     log_file = LOG_DIR / f"trend_analysis_{datetime.now().strftime('%Y%m%d')}.log"
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
    """Configure logging for trend analysis automation"""
    log_file = LOG_DIR / f"trend_analysis_{datetime.now().strftime('%Y%m%d')}.log"

    # Create a unique logger for trend analysis
    logger = logging.getLogger('trend_analysis')
    logger.setLevel(logging.INFO)

    # Clear any existing handlers to avoid duplication
    logger.handlers.clear()

    # Create handlers
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    console_handler = logging.StreamHandler()

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Prevent propagation to parent logger
    logger.propagate = False

    return logger

logger = setup_logging()


class ScoreTrendAnalysisAutomation:
    """Automate score trend analysis system"""

    # Define the three regimes to process
    REGIMES = [
        "Steady_Growth",
        "Strong_Bull",
        "Crisis_Bear"
    ]

    def __init__(self, lookback_days: int = 90, min_appearances: int = 5):
        """
        Initialize trend analysis automation

        Args:
            lookback_days: Number of days to look back for analysis
            min_appearances: Minimum appearances required for analysis
        """
        self.analyzer_script = SRC_DIR / 'trend_analysis' / 'stock_score_trend_analyzer_04.py'
        self.technical_script = SRC_DIR / 'trend_analysis' / 'stock_score_trend_technical_03.py'
        self.status_file = AUTOMATION_DIR / 'status' / 'trend_analysis_status.json'
        self.lookback_days = lookback_days
        self.min_appearances = min_appearances

        # Create status directory
        self.status_file.parent.mkdir(exist_ok=True)

    def check_prerequisites(self) -> bool:
        """Check if all required files exist"""
        logger.info("Checking prerequisites...")

        # Check if analyzer scripts exist
        if not self.analyzer_script.exists():
            logger.error(f"Analyzer script not found: {self.analyzer_script}")
            return False

        # Check if scoring results exist for each regime
        for regime in self.REGIMES:
            ranked_dir = OUTPUT_DIR / 'Ranked_Lists' / regime
            if not ranked_dir.exists():
                logger.warning(f"Ranked lists directory not found for {regime}: {ranked_dir}")
                logger.warning(f"Run scoring for {regime} first!")
                return False

            # Check if there are CSV files
            csv_files = list(ranked_dir.glob('*.csv'))
            if not csv_files:
                logger.warning(f"No ranking files found for {regime}")
                return False

            logger.info(f"Found {len(csv_files)} ranking files for {regime}")

        logger.info("All prerequisites met")
        return True

    def get_date_range_for_regime(self, regime: str) -> tuple:
        """
        Get the date range for analysis based on available files

        Returns:
            Tuple of (start_date, end_date) in MMDD format
        """
        ranked_dir = OUTPUT_DIR / 'Ranked_Lists' / regime
        csv_files = sorted(ranked_dir.glob(f'top_ranked_stocks_{regime}_*.csv'))

        if not csv_files:
            # Fallback to any CSV files
            csv_files = sorted(ranked_dir.glob('*.csv'))

        if not csv_files:
            return None, None

        # Extract dates from filenames
        dates = []
        for csv_file in csv_files:
            date_str = csv_file.stem.split('_')[-1]
            if len(date_str) == 4 and date_str.isdigit():
                dates.append(date_str)

        if dates:
            # Use the range from oldest to newest
            return min(dates), max(dates)

        # Fallback to last N days
        end_date = datetime.now().strftime("%m%d")
        start_date = (datetime.now() - timedelta(days=self.lookback_days)).strftime("%m%d")
        return start_date, end_date

    def run_stability_analysis(self, regime: str) -> Dict[str, Any]:
        """
        Run stability analysis for a specific regime

        Args:
            regime: Regime name (e.g., "Steady_Growth")

        Returns:
            Dict with results
        """
        logger.info(f"Running stability analysis for {regime} regime...")

        try:
            # Import analyzer module
            import stock_score_trend_analyzer_04 as analyzer_module

            # Get date range
            start_date, end_date = self.get_date_range_for_regime(regime)
            if not start_date:
                logger.error(f"Could not determine date range for {regime}")
                return {'success': False, 'error': 'No date range'}

            logger.info(f"Date range for {regime}: {start_date} to {end_date}")

            # Setup paths
            csv_directory = OUTPUT_DIR / 'Ranked_Lists' / regime
            output_directory = OUTPUT_DIR / 'Score_Trend_Analysis_Results' / regime
            output_directory.mkdir(parents=True, exist_ok=True)

            # Initialize analyzer
            analyzer = analyzer_module.StockScoreTrendAnalyzer(
                csv_directory=str(csv_directory),
                start_date=start_date,
                end_date=end_date,
                sigmoid_sensitivity=5
            )

            # Run analysis
            today = datetime.now().strftime("%m%d")
            output_file = output_directory / f"stability_analysis_results_{today}.csv"

            # Find latest ranked file for sector info
            ranked_files = sorted(csv_directory.glob(f"top_ranked_stocks_{regime}*.csv"))
            if not ranked_files:
                ranked_files = sorted(csv_directory.glob("*.csv"))

            ranked_file = ranked_files[-1] if ranked_files else None

            # Export results
            results_df = analyzer.export_results(
                output_path=str(output_file),
                ranked_file_path=ranked_file
            )

            # Generate plot for top stable stocks
            if not results_df.empty:
                top_stable = results_df[
                    (results_df['linear_slope'] > 0) &
                    (results_df['linear_r2'] > 0.5)
                    ].head(5)

                if not top_stable.empty:
                    top_tickers = top_stable['ticker'].head(5).tolist()
                    plot_path = output_directory / f"top_stable_trends_{today}.png"
                    analyzer.plot_stock_trends(top_tickers, save_path=str(plot_path))
                    logger.info(f"Plot saved to: {plot_path}")

            # Get summary statistics
            stats = {
                'total_stocks_analyzed': len(results_df),
                'stable_positive_trends': len(results_df[
                                                  (results_df['linear_slope'] > 0) &
                                                  (results_df['linear_r2'] > 0.5)
                                                  ]),
                'high_volatility': len(results_df[results_df['score_cv'] > 0.15]),
                'top_stock': results_df.iloc[0]['ticker'] if len(results_df) > 0 else None,
                'top_stability_score': results_df.iloc[0]['stability_adjusted_score'] if len(results_df) > 0 else None
            }

            logger.info(f"Analyzed {stats['total_stocks_analyzed']} stocks for {regime}")
            logger.info(f"Found {stats['stable_positive_trends']} stocks with stable positive trends")

            return {
                'success': True,
                'regime': regime,
                'output_file': str(output_file),
                'stats': stats,
                'date_range': f"{start_date}-{end_date}",
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error in stability analysis for {regime}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'regime': regime,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def run_technical_analysis(self, regime: str, top_n: int = 10) -> Dict[str, Any]:
        """
        Run technical analysis for top stocks in a regime

        Args:
            regime: Regime name
            top_n: Number of top stocks to analyze

        Returns:
            Dict with results
        """
        logger.info(f"Running technical analysis for top {top_n} stocks in {regime}...")

        try:
            # Import technical module
            import stock_score_trend_technical_03 as technical_module

            # Get date range
            start_date, end_date = self.get_date_range_for_regime(regime)
            if not start_date:
                return {'success': False, 'error': 'No date range'}

            # Setup paths
            csv_directory = OUTPUT_DIR / 'Ranked_Lists' / regime
            output_directory = OUTPUT_DIR / 'Score_Trend_Analysis_Results' / regime / 'technical'
            output_directory.mkdir(parents=True, exist_ok=True)

            # Initialize analyzer
            analyzer = technical_module.StockScoreTrendAnalyzerTechnical(
                csv_directory=str(csv_directory),
                start_date=start_date,
                end_date=end_date
            )

            # Load ranking files
            files_data = analyzer.load_ranking_files()
            analyzer.build_score_history(files_data)

            # Find top stocks with enough appearances
            eligible_stocks = [
                ticker for ticker in analyzer.score_history
                if len(analyzer.score_history[ticker]['indices']) >= self.min_appearances
            ]

            # Sort by average score
            stock_scores = []
            for ticker in eligible_stocks:
                avg_score = np.mean(analyzer.score_history[ticker]['scores'])
                stock_scores.append((ticker, avg_score))

            stock_scores.sort(key=lambda x: x[1], reverse=True)
            top_stocks = [ticker for ticker, _ in stock_scores[:top_n]]

            logger.info(f"Analyzing technical indicators for: {', '.join(top_stocks)}")

            # Run technical analysis for each stock
            for ticker in top_stocks:
                try:
                    analyzer.plot_technical_indicators(
                        ticker,
                        output_base_dir=str(output_directory.parent)
                    )
                except Exception as e:
                    logger.warning(f"Technical analysis failed for {ticker}: {e}")

            return {
                'success': True,
                'regime': regime,
                'stocks_analyzed': top_stocks,
                'output_dir': str(output_directory),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error in technical analysis for {regime}: {e}")
            return {
                'success': False,
                'regime': regime,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def check_scoring_complete(self) -> bool:
        """Check if scoring has completed for all regimes"""
        try:
            # Check scoring status file
            scoring_status_file = AUTOMATION_DIR / 'status' / 'scoring_status.json'
            if scoring_status_file.exists():
                with open(scoring_status_file, 'r') as f:
                    status = json.load(f)

                    # Check if all regimes were successful
                    if status.get('successful', 0) == len(self.REGIMES):
                        last_run = datetime.fromisoformat(status.get('timestamp', '2000-01-01'))
                        # Check if recent (within last hour)
                        if (datetime.now() - last_run).total_seconds() < 3600:
                            logger.info("Scoring completed successfully for all regimes")
                            return True

        except Exception as e:
            logger.warning(f"Could not check scoring status: {e}")

        return False

    # def run_all_regimes(self, skip_technical: bool = False) -> Dict[str, Any]:
    #     """
    #     Run trend analysis for all three regimes
    #
    #     Args:
    #         skip_technical: Whether to skip technical analysis
    #
    #     Returns:
    #         Summary of results
    #     """
    #     start_time = datetime.now()
    #
    #     logger.info("=" * 60)
    #     logger.info(f"SCORE TREND ANALYSIS AUTOMATION - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    #     logger.info("=" * 60)
    #
    #     # Check prerequisites
    #     if not self.check_prerequisites():
    #         logger.error("Prerequisites check failed. Run scoring first!")
    #         return {'success': False, 'error': 'Prerequisites check failed'}
    #
    #     # Run analysis for each regime
    #     results = {}
    #     successful_regimes = []
    #     failed_regimes = []
    #
    #     for i, regime in enumerate(self.REGIMES, 1):
    #         logger.info(f"\n[{i}/{len(self.REGIMES)}] Processing {regime} regime")
    #         logger.info("-" * 40)
    #
    #         # Run stability analysis
    #         stability_result = self.run_stability_analysis(regime)
    #         results[f"{regime}_stability"] = stability_result
    #
    #         if stability_result['success']:
    #             # Optionally run technical analysis
    #             if not skip_technical:
    #                 technical_result = self.run_technical_analysis(regime)
    #                 results[f"{regime}_technical"] = technical_result
    #
    #                 if technical_result['success']:
    #                     successful_regimes.append(regime)
    #                     logger.info(f" {regime} completed successfully")
    #                 else:
    #                     logger.warning(f" {regime} stability succeeded but technical failed")
    #                     successful_regimes.append(regime)  # Count as success if stability worked
    #             else:
    #                 successful_regimes.append(regime)
    #                 logger.info(f" {regime} stability analysis completed")
    #         else:
    #             failed_regimes.append(regime)
    #             logger.error(f" {regime} failed: {stability_result.get('error', 'Unknown error')}")
    #
    #     # Calculate duration
    #     end_time = datetime.now()
    #     duration = (end_time - start_time).total_seconds()
    #
    #     # Create summary
    #     summary = {
    #         'timestamp': start_time.isoformat(),
    #         'duration_seconds': duration,
    #         'total_regimes': len(self.REGIMES),
    #         'successful': len(successful_regimes),
    #         'failed': len(failed_regimes),
    #         'successful_regimes': successful_regimes,
    #         'failed_regimes': failed_regimes,
    #         'results': results,
    #         'success': len(failed_regimes) == 0
    #     }
    #
    #     # Save status
    #     self.save_status(summary)
    #
    #     # Log summary
    #     logger.info("\n" + "=" * 60)
    #     logger.info("TREND ANALYSIS SUMMARY")
    #     logger.info("=" * 60)
    #     logger.info(f"Duration: {duration:.2f} seconds")
    #     logger.info(f"Successful: {len(successful_regimes)}/{len(self.REGIMES)}")
    #     if successful_regimes:
    #         logger.info(f"Completed: {', '.join(successful_regimes)}")
    #     if failed_regimes:
    #         logger.error(f"Failed: {', '.join(failed_regimes)}")
    #     logger.info("=" * 60)
    #
    #     return summary

    def run_all_regimes(self, skip_technical: bool = True) -> Dict[str, Any]:
        """
        Run trend analysis for all regimes

        Args:
            skip_technical: If True, skip technical analysis (default: True)

        Returns:
            Summary of results
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info(f"SCORE TREND ANALYSIS AUTOMATION - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

        # Check prerequisites
        if not self.check_prerequisites():
            logger.error("Prerequisites check failed. Aborting.")
            return {'success': False, 'error': 'Prerequisites check failed'}

        results = {}
        successful_regimes = []
        failed_regimes = []

        for i, regime in enumerate(self.REGIMES, 1):
            logger.info(f"\n[{i}/{len(self.REGIMES)}] Processing {regime} regime")
            logger.info("-" * 40)

            # Run stability analysis
            stability_result = self.run_stability_analysis(regime)
            results[f"{regime}_stability"] = stability_result

            if stability_result['success']:
                # Only run technical if not skipping
                if not skip_technical:
                    technical_result = self.run_technical_analysis(regime)
                    results[f"{regime}_technical"] = technical_result

                    if technical_result['success']:
                        successful_regimes.append(regime)
                        logger.info(f"✓ {regime} completed successfully")
                    else:
                        logger.warning(f"✓ {regime} stability succeeded but technical failed")
                        # Still count as success if stability worked
                        successful_regimes.append(regime)
                else:
                    # Just stability analysis
                    successful_regimes.append(regime)
                    logger.info(f"✓ {regime} stability analysis completed successfully")
            else:
                failed_regimes.append(regime)
                logger.error(f"✗ {regime} failed: {stability_result.get('error', 'Unknown error')}")

        # Calculate duration
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Create summary
        summary = {
            'timestamp': start_time.isoformat(),
            'duration_seconds': duration,
            'total_regimes': len(self.REGIMES),
            'successful': len(successful_regimes),
            'failed': len(failed_regimes),
            'successful_regimes': successful_regimes,
            'failed_regimes': failed_regimes,
            'results': results,
            'success': len(failed_regimes) == 0,
            'technical_analysis_skipped': skip_technical
        }

        # Save status
        self.save_status(summary)

        # Log summary
        logger.info("\n" + "=" * 60)
        logger.info("TREND ANALYSIS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Successful: {len(successful_regimes)}/{len(self.REGIMES)}")
        if successful_regimes:
            logger.info(f"Completed: {', '.join(successful_regimes)}")
        if failed_regimes:
            logger.error(f"Failed: {', '.join(failed_regimes)}")
        if skip_technical:
            logger.info("Technical analysis: SKIPPED (stability only)")
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


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Score Trend Analysis Automation')
    parser.add_argument('--regime', type=str,
                        help='Run specific regime only (e.g., Steady_Growth)')
    parser.add_argument('--lookback-days', type=int, default=90,
                        help='Number of days to look back (default: 90)')
    parser.add_argument('--min-appearances', type=int, default=5,
                        help='Minimum appearances for analysis (default: 5)')
    parser.add_argument('--skip-technical', action='store_true',
                        help='Skip technical analysis')

    args = parser.parse_args()

    # Create automation
    automation = ScoreTrendAnalysisAutomation(
        lookback_days=args.lookback_days,
        min_appearances=args.min_appearances
    )

    if args.regime:
        # Run single regime
        logger.info(f"Running trend analysis for {args.regime} only")
        result = automation.run_stability_analysis(args.regime)
        if not args.skip_technical and result['success']:
            automation.run_technical_analysis(args.regime)
        success = result['success']
    else:
        # Run all regimes
        summary = automation.run_all_regimes(skip_technical=args.skip_technical)
        success = summary['success']

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    # Import required modules
    import numpy as np

    main()