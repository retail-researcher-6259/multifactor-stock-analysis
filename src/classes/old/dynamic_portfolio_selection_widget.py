# src/classes/dynamic_portfolio_selection_widget.py
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json
import subprocess

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

            # Import the portfolio selector
            from dynamic_portfolio_selector_06 import DynamicPortfolioSelector

            # Load Marketstack API key from config
            config_path = PROJECT_ROOT / 'config' / 'marketstack_config.json'
            marketstack_api_key = None

            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    marketstack_api_key = config.get('api_key')

            if not marketstack_api_key or marketstack_api_key == "YOUR_API_KEY_HERE":
                self.status_update.emit("Warning: Marketstack API key not configured. Using Yahoo Finance fallback.")

            self.progress_update.emit(20)
            self.status_update.emit("Loading stability analysis results...")

            # Get the stability results file
            stability_file = self.params['stability_file']
            if not Path(stability_file).exists():
                self.error_occurred.emit(f"Stability file not found: {stability_file}")
                return

            # Initialize selector with the stability file
            selector = DynamicPortfolioSelector(
                stability_file,
                exclude_real_estate=self.params.get('exclude_real_estate', True),
                marketstack_api_key=marketstack_api_key  # Add this parameter
            )

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
            top_pools = self.params.get('top_pools', [50, 75, 100])

            portfolios = selector.create_multiple_portfolios(
                portfolio_sizes=portfolio_sizes,
                top_pools=top_pools
            )

            self.progress_update.emit(50)
            self.status_update.emit(f"Created {len(portfolios)} portfolio combinations")

            # Run backtesting if requested
            if self.params.get('run_backtest', True):
                self.status_update.emit("Running backtests...")

                lookback_periods = self.params.get('lookback_periods', [252, 504, 756])

                best_results = {}
                for i, lookback in enumerate(lookback_periods):
                    progress = 50 + int(40 * (i + 1) / len(lookback_periods))
                    self.progress_update.emit(progress)
                    self.status_update.emit(f"Backtesting with {lookback}-day lookback...")

                    backtest_results = selector.backtest_all_portfolios(
                        lookback_days=lookback,
                        # use_yfinance=True
                    )

                    if backtest_results:
                        # Find best portfolio for this lookback
                        best_portfolio = max(backtest_results.items(),
                                             key=lambda x: x[1].get('sharpe_ratio', float('-inf')))

                        best_results[lookback] = {
                            'portfolio_name': best_portfolio[0],
                            'metrics': best_portfolio[1],
                            'all_results': backtest_results
                        }

            self.progress_update.emit(90)
            self.status_update.emit("Generating results...")

            # Prepare results
            results = {
                'success': True,
                'portfolios': portfolios,  # This contains the ticker lists
                'backtest_results': best_results if self.params.get('run_backtest') else None,
                'selector': selector,
                'total_stocks': len(selector.stocks_df),
                'params': self.params,
                'portfolio_compositions': {}  # Add this line
            }

            # Add portfolio compositions with tickers
            for portfolio_name, ticker_list in portfolios.items():
                results['portfolio_compositions'][portfolio_name] = ticker_list

            # Also add the best portfolio tickers to backtest results
            if best_results:
                for lookback, backtest_data in best_results.items():
                    portfolio_name = backtest_data['portfolio_name']
                    if portfolio_name in portfolios:
                        backtest_data['tickers'] = portfolios[portfolio_name]

            # Export results to JSON
            if self.params.get('export_results', True):
                output_dir = PROJECT_ROOT / 'output' / 'Dynamic_Portfolio_Selection' / self.params['regime']
                output_dir.mkdir(parents=True, exist_ok=True)

                timestamp = datetime.now().strftime("%m%d")
                output_file = output_dir / f"portfolio_selection_results_{timestamp}.json"

                selector.export_detailed_results(str(output_file))
                results['output_file'] = str(output_file)

            self.progress_update.emit(100)
            self.status_update.emit("Portfolio selection complete!")
            self.result_ready.emit(results)

        except Exception as e:
            self.error_occurred.emit(f"Portfolio selection failed: {str(e)}")


class DynamicPortfolioSelectionWidget(QWidget):
    """Widget for dynamic portfolio selection"""

    def __init__(self):
        super().__init__()
        self.portfolio_thread = None
        self.current_results = None
        self.recommended_regime = "Steady Growth"  # Default
        self.available_industries = set()
        self.init_ui()

    def init_ui(self):
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)

        # Title
        title_label = QLabel("Dynamic Portfolio Selection")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont("Arial", 18, QFont.Bold)
        title_label.setFont(title_font)
        main_layout.addWidget(title_label)

        # Subtitle
        subtitle_label = QLabel("Create optimized portfolios based on stability analysis")
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

    def on_mode_changed(self, mode):
        """Update parameters based on selected mode"""
        if "Quick" in mode:
            self.pool_input.setText("75")  # Single pool
            self.size_input.setText("15, 20")  # 2 sizes
            self.lookback_input.setText("252")  # 1 year only
            self.backtest_cb.setChecked(True)

        elif "Standard" in mode:
            self.pool_input.setText("50, 75")  # 2 pools
            self.size_input.setText("15, 20")  # 2 sizes
            self.lookback_input.setText("252, 504")  # 1 and 2 years
            self.backtest_cb.setChecked(True)

        else:  # Comprehensive
            self.pool_input.setText("50, 75, 100")  # 3 pools
            self.size_input.setText("10, 15, 20")  # 3 sizes
            self.lookback_input.setText("252, 504, 756")  # 1, 2, 3 years
            self.backtest_cb.setChecked(True)

    def create_parameters_section(self):
        """Create parameters input section"""
        group = QGroupBox("Portfolio Parameters")
        layout = QVBoxLayout()

        # Analysis mode
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Analysis Mode:"))

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Quick (5 min)", "Standard (15 min)", "Comprehensive (45 min)"])
        self.mode_combo.setCurrentText("Standard (15 min)")
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(self.mode_combo)

        layout.addLayout(mode_layout)

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

        self.pool_input = QLineEdit("50, 75, 100")
        self.pool_input.setPlaceholderText("e.g., 50, 75, 100")
        self.pool_input.setToolTip("Comma-separated list of top N stocks to consider")
        pool_layout.addWidget(self.pool_input)

        layout.addLayout(pool_layout)

        # Portfolio sizes
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Portfolio Sizes:"))

        self.size_input = QLineEdit("10, 15, 20")
        self.size_input.setPlaceholderText("e.g., 10, 15, 20")
        self.size_input.setToolTip("Comma-separated list of portfolio sizes")
        size_layout.addWidget(self.size_input)

        layout.addLayout(size_layout)

        # Lookback periods
        lookback_layout = QHBoxLayout()
        lookback_layout.addWidget(QLabel("Lookback Periods (days):"))

        self.lookback_input = QLineEdit("252, 504, 756")
        self.lookback_input.setPlaceholderText("e.g., 252, 504, 756")
        self.lookback_input.setToolTip("Comma-separated list of lookback periods in days")
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

        # Options
        options_label = QLabel("Options")
        options_label.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(options_label)

        self.backtest_cb = QCheckBox("Run Backtesting")
        self.backtest_cb.setChecked(True)
        layout.addWidget(self.backtest_cb)

        self.export_cb = QCheckBox("Export Results to JSON")
        self.export_cb.setChecked(True)
        layout.addWidget(self.export_cb)

        self.exclude_realestate_cb = QCheckBox("Exclude Real Estate Stocks")
        self.exclude_realestate_cb.setChecked(True)
        layout.addWidget(self.exclude_realestate_cb)

        layout.addStretch()
        group.setLayout(layout)
        return group

    def create_exclusions_section(self):
        """Create industry exclusions section"""
        group = QGroupBox("Industry Exclusions")
        layout = QVBoxLayout()

        instructions = QLabel("Select industries to exclude from portfolio:")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(instructions)

        # Scroll area for checkboxes
        scroll = QScrollArea()
        scroll_widget = QWidget()
        self.exclusions_layout = QVBoxLayout(scroll_widget)

        # Common industries to potentially exclude
        common_exclusions = [
            "Banks—Regional",
            "Banks—Diversified",
            "Insurance—Life",
            "Insurance—Property & Casualty",
            "REIT—Residential",
            "REIT—Retail",
            "REIT—Industrial",
            "Real Estate Services",
            "Utilities—Regulated Electric",
            "Utilities—Regulated Gas"
        ]

        self.exclusion_checkboxes = {}
        for industry in common_exclusions:
            cb = QCheckBox(industry)
            self.exclusion_checkboxes[industry] = cb
            self.exclusions_layout.addWidget(cb)

        # Select/Deselect all buttons
        buttons_layout = QHBoxLayout()

        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all_exclusions)
        buttons_layout.addWidget(select_all_btn)

        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self.deselect_all_exclusions)
        buttons_layout.addWidget(deselect_all_btn)

        layout.addLayout(buttons_layout)

        self.exclusions_layout.addStretch()
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        group.setLayout(layout)
        return group

    def create_results_section(self):
        """Create results display section"""
        group = QGroupBox("Portfolio Selection Results")
        layout = QVBoxLayout()

        # Results text area
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(300)
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: #2b2b2b;
                border: 1px solid #444;
                border-radius: 5px;
                color: #ffffff;
                font-family: 'Courier New', monospace;
                font-size: 11px;
            }
        """)
        self.results_text.setPlaceholderText("Portfolio selection results will appear here...")

        layout.addWidget(self.results_text)

        # Summary info
        self.summary_label = QLabel("No results available")
        self.summary_label.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(self.summary_label)

        group.setLayout(layout)
        return group

    def create_progress_bar(self):
        """Create progress bar for portfolio selection"""
        self.progress_group = QGroupBox("Progress")
        self.progress_group.setVisible(False)
        layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        layout.addWidget(self.progress_bar)

        self.progress_status = QLabel("Initializing...")
        self.progress_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_status)

        self.progress_group.setLayout(layout)

    def set_recommended_regime(self, regime):
        """Set the recommended regime from regime detection"""
        self.recommended_regime = regime
        self.regime_combo.setCurrentText(regime)
        self.regime_recommendation_label.setText("(Recommended)")
        self.auto_select_stability_file()

    def on_regime_changed(self):
        """Handle regime selection change"""
        current = self.regime_combo.currentText()
        if current == self.recommended_regime:
            self.regime_recommendation_label.setText("(Recommended)")
        else:
            self.regime_recommendation_label.setText("")
        self.auto_select_stability_file()

    def auto_select_stability_file(self):
        """Automatically select the latest stability analysis file"""
        regime = self.regime_combo.currentText().replace(" ", "_").replace("/", "_")

        # Look for stability analysis results
        stability_dir = PROJECT_ROOT / 'output' / 'Score_Trend_Analysis_Results' / regime

        if stability_dir.exists():
            # Find the latest stability file
            stability_files = list(stability_dir.glob("stability_analysis_results_*.csv"))

            if stability_files:
                # Sort by modification time and get the latest
                latest_file = max(stability_files, key=lambda p: p.stat().st_mtime)
                self.stability_file_label.setText(f".../{latest_file.parent.name}/{latest_file.name}")
                self.stability_file_label.setStyleSheet("color: #4CAF50;")
                self.current_stability_file = str(latest_file)

                # Load industries from the file for exclusion list
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
        excluded_industries = [
            industry for industry, cb in self.exclusion_checkboxes.items()
            if cb.isChecked()
        ]

        # Prepare parameters
        params = {
            'stability_file': self.current_stability_file,
            'regime': self.regime_combo.currentText().replace(" ", "_").replace("/", "_"),
            'top_pools': top_pools,
            'portfolio_sizes': portfolio_sizes,
            'lookback_periods': lookback_periods,
            'max_per_sector': self.max_sector_spin.value(),
            'run_backtest': self.backtest_cb.isChecked(),
            'export_results': self.export_cb.isChecked(),
            'exclude_real_estate': self.exclude_realestate_cb.isChecked(),
            'excluded_industries': excluded_industries
        }

        # Show progress
        self.progress_group.setVisible(True)
        self.run_btn.setEnabled(False)

        # Start thread
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

        # Parameters summary
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

            for lookback, backtest_data in results['backtest_results'].items():
                results_text += f"\n{lookback}-Day Lookback Period:\n"
                results_text += f"  Best Portfolio: {backtest_data['portfolio_name']}\n"

                # Display the tickers for the best portfolio
                if 'tickers' in backtest_data:
                    tickers = backtest_data['tickers']
                    results_text += f"  Tickers ({len(tickers)}): {', '.join(tickers[:10])}"
                    if len(tickers) > 10:
                        results_text += f"... and {len(tickers) - 10} more"
                    results_text += "\n"

                metrics = backtest_data['metrics']
                results_text += f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}\n"
                results_text += f"  Annual Return: {metrics.get('annual_return', 0) * 100:.2f}%\n"
                results_text += f"  Max Drawdown: {metrics.get('max_drawdown', 0) * 100:.2f}%\n"
                results_text += f"  Volatility: {metrics.get('volatility', 0) * 100:.2f}%\n"

        # Show example portfolio compositions
        results_text += f"\nEXAMPLE PORTFOLIO COMPOSITIONS\n"
        results_text += "-" * 40 + "\n"

        # Show top 3 portfolios with their tickers
        example_count = 0
        for name, tickers in list(results['portfolios'].items())[:3]:
            results_text += f"\n{name}:\n"
            results_text += f"  {', '.join(tickers[:10])}"
            if len(tickers) > 10:
                results_text += f"... and {len(tickers) - 10} more"
            results_text += "\n"
            example_count += 1

        # Output file info
        if results.get('output_file'):
            results_text += f"\nResults exported to:\n{results['output_file']}\n"

        self.results_text.setText(results_text)

        # Update summary with best portfolio info
        if results.get('backtest_results'):
            best_sharpe = max(
                (data['metrics'].get('sharpe_ratio', 0)
                 for data in results['backtest_results'].values())
            )
            best_portfolio_name = None
            for data in results['backtest_results'].values():
                if data['metrics'].get('sharpe_ratio', 0) == best_sharpe:
                    best_portfolio_name = data['portfolio_name']
                    break

            self.summary_label.setText(
                f"✔ Analysis complete - {len(results['portfolios'])} portfolios created\n"
                f"Best Portfolio: {best_portfolio_name} (Sharpe: {best_sharpe:.3f})"
            )
        else:
            self.summary_label.setText(
                f"✔ Analysis complete - {len(results['portfolios'])} portfolios created"
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