"""
Dynamic Portfolio Selection Widget with Data Source Selection
Supports both yfinance (fast) and Marketstack (standard) data sources
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))


class DynamicPortfolioThread(QThread):
    """Thread for running portfolio selection without blocking UI"""
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            self.status_update.emit("Initializing portfolio selector...")
            self.progress_update.emit(10)

            # Add the portfolio selector directory to Python path
            selector_path = PROJECT_ROOT / 'src' / 'portfolio_selection'
            sys.path.insert(0, str(selector_path))

            data_source = self.params.get('data_source', 'yfinance')

            if data_source == 'yfinance':
                # Use the optimized yfinance version
                self.status_update.emit("Using yfinance for data fetching (fast mode)...")
                from dynamic_portfolio_selector_yfinance import DynamicPortfolioSelectorYFinance

                # Initialize yfinance selector
                selector = DynamicPortfolioSelectorYFinance(
                    self.params['stability_file'],
                    exclude_real_estate=self.params.get('exclude_real_estate', True)
                )

            else:  # marketstack
                # Use the original Marketstack version
                self.status_update.emit("Using Marketstack API for data fetching...")
                from dynamic_portfolio_selector_06 import DynamicPortfolioSelector

                # Load Marketstack API key
                config_path = PROJECT_ROOT / 'config' / 'marketstack_config.json'
                marketstack_api_key = None

                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        marketstack_api_key = config.get('api_key')

                if not marketstack_api_key or marketstack_api_key == "YOUR_API_KEY_HERE":
                    self.error_occurred.emit("Marketstack API key not configured!")
                    return

                # Initialize Marketstack selector
                selector = DynamicPortfolioSelector(
                    self.params['stability_file'],
                    exclude_real_estate=self.params.get('exclude_real_estate', True),
                    marketstack_api_key=marketstack_api_key
                )

            self.progress_update.emit(20)
            self.status_update.emit("Loading stability analysis results...")

            # Apply industry exclusions
            if self.params.get('excluded_industries'):
                original_count = len(selector.stocks_df)
                excluded_industries = self.params['excluded_industries']
                selector.stocks_df = selector.stocks_df[~selector.stocks_df['Industry'].isin(excluded_industries)]
                excluded_count = original_count - len(selector.stocks_df)
                self.status_update.emit(f"Excluded {excluded_count} stocks from {len(excluded_industries)} industries")

            self.progress_update.emit(30)
            self.status_update.emit("Creating portfolio combinations...")

            # Create portfolios with specified parameters
            portfolio_sizes = self.params.get('portfolio_sizes', [10, 15, 20])
            top_pools = self.params.get('top_pools', [50, 100, 150])

            portfolios = selector.create_multiple_portfolios(
                portfolio_sizes=portfolio_sizes,
                top_pools=top_pools
            )

            self.progress_update.emit(50)

            # Backtest if requested
            backtest_results = {}
            if self.params.get('run_backtest', True):
                self.status_update.emit("Running backtests...")

                lookback_periods = self.params.get('lookback_periods', [252])

                if data_source == 'yfinance':
                    # Use quick backtest for yfinance
                    for lookback in lookback_periods:
                        self.status_update.emit(f"Backtesting with {lookback}-day lookback...")
                        results = selector.quick_backtest_portfolios(
                            lookback_days=lookback,
                            rebalance_frequency='monthly'
                        )

                        # Format results for display
                        if results:
                            best_portfolio = max(results.items(),
                                                 key=lambda x: x[1]['metrics']['sharpe_ratio'])
                            backtest_results[lookback] = {
                                'portfolio_name': best_portfolio[0],
                                'tickers': best_portfolio[1]['tickers'],
                                'metrics': best_portfolio[1]['metrics']
                            }
                else:
                    # Use original backtest for Marketstack
                    for i, lookback in enumerate(lookback_periods):
                        progress = 50 + (i + 1) * 30 / len(lookback_periods)
                        self.progress_update.emit(int(progress))

                        start_date = '2020-01-01'
                        end_date = datetime.now().strftime('%Y-%m-%d')

                        results = selector.backtest_all_portfolios_MHRP(
                            start_date=start_date,
                            end_date=end_date,
                            lookback_days=lookback
                        )

                        if results:
                            best_portfolio_name = max(results.keys(),
                                                      key=lambda k: results[k]['metrics']['sharpe_ratio'])
                            backtest_results[lookback] = results[best_portfolio_name]

            self.progress_update.emit(90)

            # Export results
            self.status_update.emit("Exporting results...")

            # After backtesting is complete, export best portfolio to JSON
            json_file = None
            if backtest_results and data_source == 'yfinance':
                json_file = selector.export_best_portfolio_json(
                    backtest_results,
                    regime=self.params['regime']
                )

            regime_name = self.params['regime'].replace(" ", "_").replace("/", "_")

            if data_source == 'yfinance':
                selector.backtest_results = backtest_results
                # Pass regime to export_results, don't specify output_dir
                output_file = selector.export_results(regime=regime_name)
            else:
                # For Marketstack, also use the regime-based directory
                output_file = selector.export_results(
                    regime_name,
                    output_dir=None  # Let the method use default directory structure
                )

            self.progress_update.emit(100)

            # Prepare results for UI
            results = {
                'portfolios': selector.portfolios,
                'backtest_results': backtest_results,
                'total_stocks': len(selector.stocks_df),
                'params': self.params,
                'output_file': str(output_file) if output_file else None,
                'data_source': data_source,
                'json_file': json_file,  # Add this line
            }

            self.result_ready.emit(results)

        except Exception as e:
            self.error_occurred.emit(f"Portfolio selection failed: {str(e)}")


class DynamicPortfolioSelectionWidget(QWidget):
    """Widget for dynamic portfolio selection with data source selection"""

    def __init__(self):
        super().__init__()
        self.portfolio_thread = None
        self.current_results = None
        self.recommended_regime = "Steady Growth"
        self.available_industries = set()
        self.init_ui()

    def init_ui(self):
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)

        # Title
        title_label = QLabel("Dynamic Portfolio Selection System")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont("Arial", 18, QFont.Bold)
        title_label.setFont(title_font)
        main_layout.addWidget(title_label)

        # Subtitle
        subtitle_label = QLabel("Create optimized portfolios with flexible data sources")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("color: #888;")
        main_layout.addWidget(subtitle_label)

        # Create two main sections
        sections_layout = QHBoxLayout()
        sections_layout.setSpacing(15)

        # Left section: Parameters
        params_section = self.create_parameters_section()
        sections_layout.addWidget(params_section, 1)

        # Right section: Industry Exclusions
        exclusions_section = self.create_exclusions_section()
        sections_layout.addWidget(exclusions_section, 1)

        main_layout.addLayout(sections_layout)

        # Run button
        self.run_btn = QPushButton("Generate Portfolios")
        self.run_btn.clicked.connect(self.run_portfolio_selection)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 15px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        main_layout.addWidget(self.run_btn)

        # Results section
        results_section = self.create_results_section()
        main_layout.addWidget(results_section)

        # Progress bar
        self.create_progress_bar()
        main_layout.addWidget(self.progress_group)

        self.setLayout(main_layout)

    def on_data_source_changed(self, source):
        """Update parameters based on selected data source"""
        if "yfinance" in source:
            # YFinance optimized settings
            self.pool_input.setText("50, 100")
            self.size_input.setText("10, 15, 20")
            self.lookback_input.setText("252, 504")  # 1 and 2 years only
            self.backtest_cb.setChecked(True)

            # Update tooltips
            self.lookback_input.setToolTip("Shorter lookbacks for better yfinance compatibility")

            # Show info message
            self.data_source_info.setText("✓ Fast mode: Using yfinance (no API limits)")
            self.data_source_info.setStyleSheet("color: #4CAF50;")

        else:  # Marketstack
            # Standard Marketstack settings
            self.pool_input.setText("50, 75, 100")
            self.size_input.setText("10, 15, 20")
            self.lookback_input.setText("252, 504, 756")  # 1, 2, 3 years
            self.backtest_cb.setChecked(True)

            # Update tooltips
            self.lookback_input.setToolTip("Full lookback range with Marketstack API")

            # Show info message
            self.data_source_info.setText("✓ Standard mode: Using Marketstack API")
            self.data_source_info.setStyleSheet("color: #2196F3;")

    def create_parameters_section(self):
        """Create parameters input section with data source selection"""
        group = QGroupBox("Portfolio Parameters")
        layout = QVBoxLayout()

        # Data source selection (NEW)
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Data Source:"))

        self.data_source_combo = QComboBox()
        self.data_source_combo.addItems(["yfinance (fast)", "Marketstack (standard)"])
        self.data_source_combo.setCurrentText("yfinance (fast)")
        self.data_source_combo.currentTextChanged.connect(self.on_data_source_changed)
        source_layout.addWidget(self.data_source_combo)

        # Info label for data source
        self.data_source_info = QLabel("✓ Fast mode: Using yfinance (no API limits)")
        self.data_source_info.setStyleSheet("color: #4CAF50; font-style: italic;")
        source_layout.addWidget(self.data_source_info)

        source_layout.addStretch()
        layout.addLayout(source_layout)

        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: #444;")
        layout.addWidget(separator)

        # Regime selection
        regime_layout = QHBoxLayout()
        regime_layout.addWidget(QLabel("Target Regime:"))

        self.regime_combo = QComboBox()
        self.regime_combo.addItems(["Steady Growth", "Strong Bull", "Crisis/Bear"])
        self.regime_combo.setCurrentText(self.recommended_regime)
        self.regime_combo.currentTextChanged.connect(self.on_regime_changed)
        regime_layout.addWidget(self.regime_combo)

        self.regime_recommendation_label = QLabel("(Recommended)")
        self.regime_recommendation_label.setStyleSheet("color: #4CAF50; font-style: italic;")
        regime_layout.addWidget(self.regime_recommendation_label)

        regime_layout.addStretch()
        layout.addLayout(regime_layout)

        # Stability file selection
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Stability File:"))

        self.stability_file_label = QLabel("No file selected")
        self.stability_file_label.setStyleSheet("color: #888; font-style: italic;")
        file_layout.addWidget(self.stability_file_label, 1)

        self.browse_btn = QPushButton("Auto-Select Latest")
        self.browse_btn.clicked.connect(self.auto_select_stability_file)
        file_layout.addWidget(self.browse_btn)

        layout.addLayout(file_layout)

        # Top pools parameter
        pool_layout = QHBoxLayout()
        pool_layout.addWidget(QLabel("Top Pools:"))

        self.pool_input = QLineEdit("50, 100, 150")
        self.pool_input.setPlaceholderText("e.g., 50, 100, 150")
        self.pool_input.setToolTip("Number of top stocks to consider")
        pool_layout.addWidget(self.pool_input)

        layout.addLayout(pool_layout)

        # Portfolio sizes
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Portfolio Sizes:"))

        self.size_input = QLineEdit("10, 15, 20")
        self.size_input.setPlaceholderText("e.g., 10, 15, 20")
        self.size_input.setToolTip("Number of stocks per portfolio")
        size_layout.addWidget(self.size_input)

        layout.addLayout(size_layout)

        # Lookback periods
        lookback_layout = QHBoxLayout()
        lookback_layout.addWidget(QLabel("Lookback Periods (days):"))

        self.lookback_input = QLineEdit("252, 504")
        self.lookback_input.setPlaceholderText("e.g., 252, 504")
        self.lookback_input.setToolTip("Shorter lookbacks for better yfinance compatibility")
        lookback_layout.addWidget(self.lookback_input)

        layout.addLayout(lookback_layout)

        # Max stocks per sector
        sector_layout = QHBoxLayout()
        sector_layout.addWidget(QLabel("Max Stocks Per Sector:"))

        self.max_sector_spin = QSpinBox()
        self.max_sector_spin.setMinimum(1)
        self.max_sector_spin.setMaximum(10)
        self.max_sector_spin.setValue(3)  # Default
        self.max_sector_spin.setToolTip("Maximum number of stocks from the same sector")
        sector_layout.addWidget(self.max_sector_spin)

        sector_layout.addStretch()
        layout.addLayout(sector_layout)

        # Backtest option
        self.backtest_cb = QCheckBox("Run Backtests")
        self.backtest_cb.setChecked(True)
        layout.addWidget(self.backtest_cb)

        # Export Results checkbox
        self.export_cb = QCheckBox("Export Results to JSON")
        self.export_cb.setChecked(True)
        layout.addWidget(self.export_cb)

        group.setLayout(layout)
        return group

    def create_exclusions_section(self):
        """Create industry exclusions section"""
        group = QGroupBox("Industry Exclusions")
        layout = QVBoxLayout()

        # Buttons
        button_layout = QHBoxLayout()

        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all_exclusions)
        button_layout.addWidget(select_all_btn)

        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self.deselect_all_exclusions)
        button_layout.addWidget(deselect_all_btn)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Scrollable area for checkboxes
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        self.exclusions_layout = QVBoxLayout()

        # Add default exclusion for REITs
        self.real_estate_cb = QCheckBox("Exclude all Real Estate/REITs")
        self.real_estate_cb.setChecked(True)
        self.real_estate_cb.setStyleSheet("font-weight: bold;")
        self.exclusions_layout.addWidget(self.real_estate_cb)

        # Placeholder for industry checkboxes
        self.exclusion_checkboxes = {}

        self.exclusions_layout.addStretch()
        scroll_widget.setLayout(self.exclusions_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

        group.setLayout(layout)
        return group

    def create_results_section(self):
        """Create results display section"""
        group = QGroupBox("Results")
        layout = QVBoxLayout()

        # Results text area
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(200)
        self.results_text.setPlaceholderText("Portfolio results will appear here...")
        layout.addWidget(self.results_text)

        # Summary label
        self.summary_label = QLabel("")
        self.summary_label.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(self.summary_label)

        group.setLayout(layout)
        return group

    def create_progress_bar(self):
        """Create progress bar section"""
        self.progress_group = QGroupBox("Progress")
        layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.progress_status = QLabel("Ready")
        self.progress_status.setStyleSheet("color: #888;")
        layout.addWidget(self.progress_status)

        self.progress_group.setLayout(layout)
        self.progress_group.setVisible(False)

    def on_regime_changed(self, regime):
        """Update recommendation label based on regime selection"""
        if regime == self.recommended_regime:
            self.regime_recommendation_label.setText("(Recommended)")
            self.regime_recommendation_label.setStyleSheet("color: #4CAF50; font-style: italic;")
        else:
            self.regime_recommendation_label.setText("")

    def set_recommended_regime(self, regime):
        """Set the recommended regime from external detection"""
        self.recommended_regime = regime
        self.regime_combo.setCurrentText(regime)
        self.on_regime_changed(regime)

    def auto_select_stability_file(self):
        """Automatically select the latest stability analysis file"""
        # Look for stability files in the appropriate directory
        regime_dirs = {
            "Steady Growth": "Steady_Growth",
            "Strong Bull": "Strong_Bull",
            "Crisis/Bear": "Crisis_Bear"
        }

        current_regime = self.regime_combo.currentText()
        # regime_dir = PROJECT_ROOT / 'output/Score_Trend_Analysis_Results' / regime_dirs.get(current_regime, "Steady_Growth")
        regime_dir = PROJECT_ROOT / 'output' / 'Score_Trend_Analysis_Results' / regime_dirs.get(current_regime,"Steady_Growth")

        if regime_dir.exists():
            # First try to find common_top files
            common_files = list(regime_dir.glob("common_top*.csv"))
            # If not found, look for stability analysis files
            if not common_files:
                stability_files = list(regime_dir.glob("stability_analysis_results_*.csv"))
                if stability_files:
                    common_files = stability_files
            if common_files:
                # Sort by modification time and get the latest
                latest_file = max(common_files, key=lambda p: p.stat().st_mtime)
                self.current_stability_file = str(latest_file)
                self.stability_file_label.setText(latest_file.name)
                self.stability_file_label.setStyleSheet("color: #4CAF50;")

                # Load industries from the file
                self.load_industries_from_file(latest_file)
            else:
                self.stability_file_label.setText("No stability analysis found")
                self.stability_file_label.setStyleSheet("color: #f44336;")
                self.current_stability_file = None
        else:
            self.stability_file_label.setText("Regime directory not found")
            self.stability_file_label.setStyleSheet("color: #f44336;")
            self.current_stability_file = None

    def load_industries_from_file(self, file_path):
        """Load unique industries from the stability file"""
        try:
            df = pd.read_csv(file_path)
            if 'Industry' in df.columns:
                industries = df['Industry'].unique()

                # Clear existing checkboxes
                for cb in self.exclusion_checkboxes.values():
                    cb.setParent(None)
                self.exclusion_checkboxes.clear()

                # Add checkboxes for all industries
                for industry in sorted(industries):
                    if pd.notna(industry):
                        cb = QCheckBox(str(industry))
                        # Pre-check if it's a real estate or REIT industry
                        if 'REIT' in str(industry) or 'Real Estate' in str(industry):
                            cb.setChecked(True)
                        self.exclusion_checkboxes[str(industry)] = cb
                        self.exclusions_layout.insertWidget(
                            self.exclusions_layout.count() - 1, cb
                        )
        except Exception as e:
            print(f"Error loading industries: {e}")

    def select_all_exclusions(self):
        """Select all industry exclusions"""
        for cb in self.exclusion_checkboxes.values():
            cb.setChecked(True)

    def deselect_all_exclusions(self):
        """Deselect all industry exclusions"""
        for cb in self.exclusion_checkboxes.values():
            cb.setChecked(False)

    def run_portfolio_selection(self):
        """Run the portfolio selection process"""
        # Validate inputs
        if not hasattr(self, 'current_stability_file') or not self.current_stability_file:
            QMessageBox.warning(self, "Warning", "Please select a stability analysis file first.")
            return

        # Parse parameters
        try:
            top_pools = [int(x.strip()) for x in self.pool_input.text().split(',')]
            portfolio_sizes = [int(x.strip()) for x in self.size_input.text().split(',')]
            lookback_periods = [int(x.strip()) for x in self.lookback_input.text().split(',')]
        except ValueError:
            QMessageBox.warning(self, "Invalid Input",
                                "Please enter valid comma-separated numbers for pools, sizes, and lookback periods.")
            return

        # Get excluded industries
        excluded_industries = []
        if self.real_estate_cb.isChecked():
            excluded_industries.extend(['REIT', 'Real Estate'])

        for industry, cb in self.exclusion_checkboxes.items():
            if cb.isChecked() and industry not in excluded_industries:
                excluded_industries.append(industry)

        # Get data source
        data_source = 'yfinance' if 'yfinance' in self.data_source_combo.currentText() else 'marketstack'

        # Prepare parameters
        params = {
            'stability_file': self.current_stability_file,
            'regime': self.regime_combo.currentText(),
            'top_pools': top_pools,
            'portfolio_sizes': portfolio_sizes,
            'lookback_periods': lookback_periods,
            'exclude_real_estate': self.real_estate_cb.isChecked(),
            'excluded_industries': excluded_industries,
            'run_backtest': self.backtest_cb.isChecked(),
            'data_source': data_source,
            'max_per_sector': self.max_sector_spin.value(),
            'export_results': self.export_cb.isChecked(),
        }

        # Show progress bar
        self.progress_group.setVisible(True)
        self.progress_bar.setValue(0)
        self.run_btn.setEnabled(False)

        # Create and start thread
        self.portfolio_thread = DynamicPortfolioThread(params)
        self.portfolio_thread.progress_update.connect(self.update_progress)
        self.portfolio_thread.status_update.connect(self.update_status)
        self.portfolio_thread.result_ready.connect(self.handle_results)
        self.portfolio_thread.error_occurred.connect(self.handle_error)
        self.portfolio_thread.finished.connect(self.on_thread_finished)
        self.portfolio_thread.start()

    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)

    def update_status(self, message):
        """Update status message"""
        self.progress_status.setText(message)

    def handle_results(self, results):
        """Handle portfolio selection results"""
        self.current_results = results

        # Format and display results
        results_text = "PORTFOLIO SELECTION RESULTS\n"
        results_text += "=" * 60 + "\n\n"

        # Data source info
        results_text += f"Data Source: {results.get('data_source', 'Unknown').upper()}\n"
        results_text += f"Regime: {results['params']['regime']}\n"
        results_text += f"Total Stocks Available: {results['total_stocks']}\n"
        results_text += f"Portfolio Sizes: {results['params']['portfolio_sizes']}\n"
        results_text += f"Top Pools: {results['params']['top_pools']}\n"
        results_text += f"Excluded Industries: {len(results['params']['excluded_industries'])}\n"
        results_text += "\n"

        # Portfolio combinations
        results_text += f"PORTFOLIO COMBINATIONS\n"
        results_text += "-" * 40 + "\n"
        results_text += f"Total Portfolios Created: {len(results['portfolios'])}\n\n"

        # Group portfolios by strategy
        strategy_counts = {}
        for name in results['portfolios'].keys():
            strategy = name.split('_')[0]
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        for strategy, count in sorted(strategy_counts.items()):
            results_text += f"  {strategy}: {count} variations\n"

        # Backtest results if available
        if results.get('backtest_results'):
            results_text += f"\nBACKTEST RESULTS\n"
            results_text += "-" * 40 + "\n"

            best_portfolio_name = None
            best_tickers = []

            for lookback, backtest_data in results['backtest_results'].items():
                results_text += f"\n{lookback}-Day Lookback Period:\n"
                results_text += f"  Best Portfolio: {backtest_data['portfolio_name']}\n"

                # Store the best portfolio info for later
                if not best_portfolio_name or backtest_data['metrics']['sharpe_ratio'] > best_sharpe:
                    best_portfolio_name = backtest_data['portfolio_name']
                    best_tickers = backtest_data.get('tickers', [])
                    best_sharpe = backtest_data['metrics']['sharpe_ratio']

                # Display metrics
                metrics = backtest_data['metrics']
                results_text += f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}\n"
                results_text += f"  Annual Return: {metrics.get('annual_return', 0) * 100:.2f}%\n"
                results_text += f"  Max Drawdown: {metrics.get('max_drawdown', 0) * 100:.2f}%\n"
                results_text += f"  Volatility: {metrics.get('volatility', 0) * 100:.2f}%\n"

            # Show best portfolio composition
            if best_portfolio_name and best_tickers:
                results_text += f"\nBEST PORTFOLIO COMPOSITION\n"
                results_text += "-" * 40 + "\n"
                results_text += f"{best_portfolio_name}:\n"
                results_text += f"Tickers ({len(best_tickers)}): {', '.join(best_tickers[:10])}"
                if len(best_tickers) > 10:
                    results_text += f"\n... and {len(best_tickers) - 10} more: {', '.join(best_tickers[10:])}"
                results_text += "\n"

        # Output file info
        if results.get('output_file'):
            results_text += f"\nResults exported to:\n{results['output_file']}\n"

        if results.get('json_file'):
            results_text += f"Portfolio JSON saved to:\n{results['json_file']}\n"

        self.results_text.setText(results_text)

        # Update summary
        if results.get('backtest_results'):
            self.summary_label.setText(
                f"✓ Analysis complete - {len(results['portfolios'])} portfolios created | "
                f"Best Sharpe: {best_sharpe:.3f} | Source: {results.get('data_source', 'Unknown').upper()}"
            )
        else:
            self.summary_label.setText(
                f"✓ Analysis complete - {len(results['portfolios'])} portfolios created | "
                f"Source: {results.get('data_source', 'Unknown').upper()}"
            )
        self.summary_label.setStyleSheet("color: #4CAF50;")

    def handle_error(self, error_message):
        """Handle errors from portfolio thread"""
        QMessageBox.critical(self, "Error", f"Portfolio selection failed:\n{error_message}")
        self.progress_group.setVisible(False)
        self.run_btn.setEnabled(True)

    def on_thread_finished(self):
        """Clean up after thread finishes"""
        self.progress_group.setVisible(False)
        self.run_btn.setEnabled(True)