# src/classes/multifactor_scoring_widget.py
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

class MultifactorScoringThread(QThread):
    """Enhanced thread for running multifactor scoring with detailed progress tracking"""
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, regime_name, ticker_file=None):
        super().__init__()
        self.regime_name = regime_name
        self.current_progress = 0
        self.ticker_file = ticker_file  # NEW

    def progress_callback(self, update_type, value):
        """Callback function for progress updates from the scoring script"""
        if update_type == 'progress':
            self.current_progress = value
            self.progress_update.emit(value)
        elif update_type == 'status':
            self.status_update.emit(value)
        elif update_type == 'error':
            self.error_occurred.emit(value)

    def run(self):
        try:
            self.status_update.emit("Initializing scoring process...")
            self.progress_update.emit(0)

            # Import the scoring script
            scoring_script_path = PROJECT_ROOT / 'src' / 'scoring' / 'stock_Screener_MultiFactor_25_new.py'

            if not scoring_script_path.exists():
                self.error_occurred.emit(f"Scoring script not found: {scoring_script_path}")
                return

            # Add the scoring directory to Python path
            sys.path.insert(0, str(scoring_script_path.parent))

            try:
                # Import the scoring module
                import stock_Screener_MultiFactor_25_new as scorer

                # Set the ticker file if provided
                if self.ticker_file:
                    scorer.set_ticker_file(self.ticker_file)  # Use the set_ticker_file function

                self.status_update.emit("Starting multifactor analysis...")
                self.progress_update.emit(2)

                # # Pass ticker file to the scoring function
                # if self.ticker_file:
                #     # Temporarily override the TICKER_FILE in the module
                #     scorer.TICKER_FILE = self.ticker_file

                # Run the scoring with progress callback
                result = scorer.run_scoring_for_regime(self.regime_name, self.progress_callback)

                if result['success']:
                    self.status_update.emit("Analysis completed successfully!")
                    self.result_ready.emit(result)
                else:
                    self.error_occurred.emit(result['error'])

            except ImportError as e:
                self.error_occurred.emit(f"Failed to import scoring module: {e}")
            except Exception as e:
                self.error_occurred.emit(f"Scoring process failed: {e}")
            finally:
                # Clean up sys.path
                if str(scoring_script_path.parent) in sys.path:
                    sys.path.remove(str(scoring_script_path.parent))

        except Exception as e:
            self.error_occurred.emit(f"Thread execution failed: {e}")


class MultifactorScoringWidget(QWidget):
    """Main widget for multifactor scoring system"""

    def __init__(self):
        super().__init__()
        self.scoring_thread = None
        self.current_results = None
        self.regime_weights = self.get_regime_weights_dict()
        self.init_ui()

    def set_detected_regime(self, regime):
        """Set the detected regime from regime detection system"""
        # Map the regime name to the combo box format
        regime_map = {
            "Steady Growth": "Steady Growth",
            "Strong Bull": "Strong Bull",
            "Crisis/Bear": "Crisis/Bear",
            "Crisis_Bear": "Crisis/Bear"  # Handle underscore variant
        }

        detected_regime = regime_map.get(regime, regime)

        # Update the combo box to the detected regime
        index = self.regime_combo.findText(detected_regime)
        if index >= 0:
            self.regime_combo.setCurrentIndex(index)

            # Update visual indicator
            self.status_display.setText(f"Detected regime: {detected_regime}")
            self.status_display.setStyleSheet("""
                QLabel {
                    padding: 10px;
                    border-radius: 5px;
                    background-color: #4CAF50;
                    color: #ffffff;
                }
            """)

            # Update weights display for the new regime
            self.update_weights_display(detected_regime)

    def get_regime_weights_dict(self):
        """Define regime-specific weights"""
        return {
            "Steady Growth": {
                "momentum": 130.7,
                "stability": -30.8,
                "options_flow": -18.6,
                "financial_health": 15.3,
                "technical": 14.9,
                "value": -10.6,
                "carry": -7.0,
                "growth": 3.5,
                "quality": 2.6,
                "size": 2.3,
                "liquidity": -2.3,
                "insider": 0.0,
            },
            "Strong Bull": {
                "momentum": 138.9,
                "stability": -60.0,
                "financial_health": 40.1,
                "carry": -33.4,
                "technical": 32.3,
                "value": -23.2,
                "options_flow": -19.6,
                "growth": 13.4,
                "quality": 8.2,
                "size": 3.3,
                "insider": 0.0,
                "liquidity": 0.0,
            },
            "Crisis/Bear": {
                "momentum": 118.0,
                "stability": -50.4,
                "financial_health": 34.5,
                "technical": 28.8,
                "carry": -21.2,
                "options_flow": -18.6,
                "value": -13.3,
                "growth": 13.3,
                "size": 8.0,
                "quality": 3.6,
                "liquidity": -2.7,
                "insider": 0.0,
            }
        }

    def init_ui(self):
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)

        # Title
        title_label = QLabel("Multifactor Stock Scoring System")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont("Arial", 18, QFont.Bold)
        title_label.setFont(title_font)
        main_layout.addWidget(title_label)

        # Subtitle
        subtitle_label = QLabel("Advanced multi-factor stock analysis with regime-specific weights")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("color: #888;")
        main_layout.addWidget(subtitle_label)

        # Create horizontal layout for controls - NOW WITH THREE PANELS
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(15)

        # Control Panel (left)
        control_panel = self.create_control_panel()
        controls_layout.addWidget(control_panel, 1)

        # Status Panel (center) - THIS WAS MISSING!
        status_panel = self.create_status_panel()
        controls_layout.addWidget(status_panel, 1)

        # Weights Display Panel (right)
        weights_panel = self.create_weights_panel()
        controls_layout.addWidget(weights_panel, 1)

        main_layout.addLayout(controls_layout)

        # Results section
        results_section = self.create_results_section()
        main_layout.addWidget(results_section)

        # Progress and status bar
        self.create_progress_bar()
        main_layout.addWidget(self.progress_group)

        self.setLayout(main_layout)

    def create_weights_panel(self):
        """Create panel to display regime-specific weights"""
        group = QGroupBox("Factor Weights for Selected Regime")
        layout = QVBoxLayout()

        # Weights display area
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        self.weights_layout = QVBoxLayout(scroll_widget)

        # Initialize with default regime weights
        self.update_weights_display("Steady Growth")

        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(400)
        layout.addWidget(scroll_area)

        # Weights information
        weights_info = QLabel("Weights are optimized for each market regime based on historical backtesting")
        weights_info.setStyleSheet("color: #888; font-style: italic; font-size: 11px;")
        weights_info.setWordWrap(True)
        layout.addWidget(weights_info)

        group.setLayout(layout)
        return group

    def create_control_panel(self):
        """Create control panel with mode selection and regime selector"""
        group = QGroupBox("Scoring Configuration")
        layout = QVBoxLayout()

        # Ticker File Selection (NEW)
        ticker_layout = QHBoxLayout()
        ticker_label = QLabel("Ticker List:")
        ticker_label.setFont(QFont("Arial", 10))
        ticker_layout.addWidget(ticker_label)

        self.ticker_file_combo = QComboBox()
        self.ticker_file_combo.setEditable(False)  # Allow custom file entry
        self.ticker_file_combo.setMinimumWidth(200)

        # Find available ticker files in the config directory
        config_dir = PROJECT_ROOT / "config"
        ticker_files = []

        # Look for .txt files in config directory
        if config_dir.exists():
            for file in config_dir.glob("*.txt"):
                if "stock" in file.name.lower() or "ticker" in file.name.lower():
                    ticker_files.append(file.name)

        # Add default files if they exist
        default_files = [
            "Buyable_stocks_test.txt",
            "Buyable_stocks_0901.txt",
        ]

        for default_file in default_files:
            if default_file not in ticker_files and (config_dir / default_file).exists():
                ticker_files.append(default_file)

        # Sort and add to combo box
        ticker_files.sort()
        self.ticker_file_combo.addItems(ticker_files)

        # Set current file based on what's in the script
        current_file = "Buyable_stocks_0901.txt"  # Default
        if current_file in ticker_files:
            self.ticker_file_combo.setCurrentText(current_file)

        self.ticker_file_combo.currentTextChanged.connect(self.on_ticker_file_changed)
        ticker_layout.addWidget(self.ticker_file_combo)

        # Add a refresh button to reload ticker list
        self.refresh_ticker_btn = QPushButton("↻")
        self.refresh_ticker_btn.setMaximumWidth(30)
        self.refresh_ticker_btn.setToolTip("Refresh ticker file list")
        self.refresh_ticker_btn.clicked.connect(self.refresh_ticker_files)
        ticker_layout.addWidget(self.refresh_ticker_btn)

        ticker_layout.addStretch()
        layout.addLayout(ticker_layout)

        # Display ticker count
        self.ticker_count_label = QLabel("Tickers loaded: --")
        self.ticker_count_label.setStyleSheet("color: #888; font-size: 10px; margin-left: 60px;")
        layout.addWidget(self.ticker_count_label)

        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        # Mode Selection
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Mode:")
        mode_label.setFont(QFont("Arial", 10))
        mode_layout.addWidget(mode_label)

        self.mode_combo = QComboBox()
        # Removed "Historical (Past Year)" and added "Historical (Specific)"
        self.mode_combo.addItems(["Daily (Current)", "Historical (Range)", "Historical (Specific)"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        # Date Selection for Historical Range Mode
        self.date_range_group = QGroupBox("Historical Date Range")
        date_range_layout = QGridLayout()

        # Start Date
        date_range_layout.addWidget(QLabel("Start Date:"), 0, 0)
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setCalendarPopup(True)
        # Default to past year
        self.start_date_edit.setDate(QDate.currentDate().addDays(-365))
        self.start_date_edit.setDisplayFormat("yyyy-MM-dd")
        date_range_layout.addWidget(self.start_date_edit, 0, 1)

        # End Date
        date_range_layout.addWidget(QLabel("End Date:"), 1, 0)
        self.end_date_edit = QDateEdit()
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDate(QDate.currentDate())
        self.end_date_edit.setDisplayFormat("yyyy-MM-dd")
        date_range_layout.addWidget(self.end_date_edit, 1, 1)

        # Resolution Note
        resolution_note = QLabel("Resolution: Daily (optimized for trend analysis)")
        resolution_note.setStyleSheet("color: #888; font-style: italic; margin-top: 5px;")
        date_range_layout.addWidget(resolution_note, 2, 0, 1, 2)

        self.date_range_group.setLayout(date_range_layout)
        self.date_range_group.setVisible(False)  # Hidden by default
        layout.addWidget(self.date_range_group)

        # Date Selection for Historical Specific Mode (NEW)
        self.date_specific_group = QGroupBox("Date Range with Manual Regime")
        date_specific_layout = QGridLayout()

        # Start Date - DEFAULT TO 7 DAYS BEFORE TODAY
        date_specific_layout.addWidget(QLabel("Start Date:"), 0, 0)
        self.specific_start_date_edit = QDateEdit()
        self.specific_start_date_edit.setCalendarPopup(True)
        self.specific_start_date_edit.setDate(QDate.currentDate().addDays(-7))  # Changed from -30 to -7
        self.specific_start_date_edit.setDisplayFormat("yyyy-MM-dd")
        date_specific_layout.addWidget(self.specific_start_date_edit, 0, 1)

        # End Date - DEFAULT TO TODAY
        date_specific_layout.addWidget(QLabel("End Date:"), 1, 0)
        self.specific_end_date_edit = QDateEdit()
        self.specific_end_date_edit.setCalendarPopup(True)
        self.specific_end_date_edit.setDate(QDate.currentDate())  # This is already today's date
        self.specific_end_date_edit.setDisplayFormat("yyyy-MM-dd")
        date_specific_layout.addWidget(self.specific_end_date_edit, 1, 1)

        # Note about regime selection
        specific_note = QLabel("Note: The selected regime below will be used for ALL dates in this range")
        specific_note.setStyleSheet("color: #888; font-style: italic; margin-top: 5px;")
        date_specific_layout.addWidget(specific_note, 2, 0, 1, 2)

        self.date_specific_group.setLayout(date_specific_layout)
        self.date_specific_group.setVisible(False)  # Hidden by default
        layout.addWidget(self.date_specific_group)

        # Market Regime Selection
        regime_layout = QHBoxLayout()
        regime_label = QLabel("Market Regime:")
        regime_label.setFont(QFont("Arial", 10))
        regime_layout.addWidget(regime_label)

        self.regime_combo = QComboBox()
        self.regime_combo.addItems(["Steady Growth", "Strong Bull", "Crisis/Bear"])
        self.regime_combo.setCurrentText("Steady Growth")
        self.regime_combo.currentTextChanged.connect(self.on_regime_changed)
        self.regime_combo.setToolTip("Select the market regime for optimized factor weights")
        regime_layout.addWidget(self.regime_combo)
        regime_layout.addStretch()
        layout.addLayout(regime_layout)

        # Mode-specific notes
        self.daily_note = QLabel("Note: Using current market data with selected regime weights")
        self.daily_note.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.daily_note)

        self.historical_range_note = QLabel("Note: Regimes are auto-detected from regime_periods.csv for each date")
        self.historical_range_note.setStyleSheet("color: #888; font-size: 10px;")
        self.historical_range_note.setVisible(False)
        layout.addWidget(self.historical_range_note)

        self.historical_specific_note = QLabel(
            "Note: Using historical data for specific date with manually selected regime")
        self.historical_specific_note.setStyleSheet("color: #888; font-size: 10px;")
        self.historical_specific_note.setVisible(False)
        layout.addWidget(self.historical_specific_note)

        # Main Action Button
        self.run_scoring_btn = QPushButton("Run Multifactor Scoring Process")
        self.run_scoring_btn.clicked.connect(self.run_scoring_process)
        self.run_scoring_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 15px;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3e8e41;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        layout.addWidget(self.run_scoring_btn)

        # Process Information
        info_label = QLabel("Process Information")
        info_label.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(info_label)

        self.process_info = QLabel(
            f"• Data Source: {PROJECT_ROOT / 'config' / 'Buyable_stocks.txt'}\n"
            f"• Output Directory: ./output/Ranked_Lists/[Regime]/\n"
            f"• Analysis: Multifactor scoring with regime-specific weights\n"
            f"• Historical Resolution: Daily intervals for optimal trend analysis"
        )
        self.process_info.setStyleSheet("""
            QLabel {
                background-color: #3c3c3c;
                padding: 10px;
                border-radius: 5px;
                color: #cccccc;
                font-size: 11px;
            }
        """)
        self.process_info.setWordWrap(True)
        layout.addWidget(self.process_info)

        # Update Process Information to show selected ticker file
        self.update_process_info()

        layout.addStretch()
        group.setLayout(layout)
        return group

    def on_ticker_file_changed(self, filename):
        """Handle ticker file selection change"""
        if filename:
            ticker_path = PROJECT_ROOT / "config" / filename
            if ticker_path.exists():
                # Count tickers in the file
                try:
                    with open(ticker_path, 'r') as f:
                        tickers = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                        count = len(tickers)
                        self.ticker_count_label.setText(f"Tickers loaded: {count}")

                        # Update process info
                        self.update_process_info()

                        # Store the selected file path for use in scoring
                        self.selected_ticker_file = ticker_path

                        print(f" Loaded {count} tickers from {filename}")
                except Exception as e:
                    self.ticker_count_label.setText(f"Error reading file: {str(e)}")
                    QMessageBox.warning(self, "File Error", f"Could not read ticker file:\n{str(e)}")
            else:
                self.ticker_count_label.setText("File not found")
                QMessageBox.warning(self, "File Not Found", f"Ticker file not found:\n{ticker_path}")

    def refresh_ticker_files(self):
        """Refresh the list of available ticker files"""
        current_selection = self.ticker_file_combo.currentText()
        self.ticker_file_combo.clear()

        config_dir = PROJECT_ROOT / "config"
        ticker_files = []

        if config_dir.exists():
            for file in config_dir.glob("*.txt"):
                ticker_files.append(file.name)

        ticker_files.sort()
        self.ticker_file_combo.addItems(ticker_files)

        # Restore previous selection if it still exists
        if current_selection in ticker_files:
            self.ticker_file_combo.setCurrentText(current_selection)

        QMessageBox.information(self, "Refreshed", f"Found {len(ticker_files)} ticker files")

    def update_process_info(self):
        """Update process information display"""
        if hasattr(self, 'process_info'):
            ticker_file = self.ticker_file_combo.currentText() if hasattr(self,
                                                                          'ticker_file_combo') else "Buyable_stocks.txt"
            self.process_info.setText(
                f"• Data Source: {PROJECT_ROOT / 'config' / ticker_file}\n"
                f"• Output Directory: ./output/Ranked_Lists/[Regime]/\n"
                f"• Analysis: Multifactor scoring with regime-specific weights\n"
                f"• Historical Resolution: Daily intervals for optimal trend analysis"
            )

    def get_selected_ticker_file(self):
        """Get the currently selected ticker file path"""
        if hasattr(self, 'selected_ticker_file'):
            return self.selected_ticker_file
        else:
            # Default fallback
            return PROJECT_ROOT / "config" / "Buyable_stocks_0901.txt"

    def create_status_panel(self):
        """Create status panel showing current process info"""
        group = QGroupBox("Process Status")
        layout = QVBoxLayout()

        # Current Status
        status_label = QLabel("Current Status")
        status_label.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(status_label)

        self.status_display = QLabel("Ready to run scoring process")
        self.status_display.setStyleSheet("""
            QLabel {
                padding: 10px;
                border-radius: 5px;
                background-color: #3c3c3c;
                color: #ffffff;
            }
        """)
        layout.addWidget(self.status_display)

        # Statistics (will be populated after scoring)
        stats_label = QLabel("Scoring Statistics")
        stats_label.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(stats_label)

        self.stats_layout = QVBoxLayout()

        # Placeholder stats
        self.stats_labels = {
            'total_stocks': QLabel("Total Stocks Analyzed: --"),
            'top_performers': QLabel("Top 10% Performers: --"),
            'avg_score': QLabel("Average Score: --"),
            'processing_time': QLabel("Processing Time: --")
        }

        for label in self.stats_labels.values():
            label.setStyleSheet("color: #cccccc; font-size: 12px;")
            self.stats_layout.addWidget(label)

        layout.addLayout(self.stats_layout)
        layout.addStretch()

        group.setLayout(layout)
        return group

    def create_results_section(self):
        """Create results display section"""
        group = QGroupBox("Scoring Results")
        layout = QVBoxLayout()

        # Results table
        self.results_table = QTableWidget()
        self.results_table.setMinimumHeight(400)

        # Set table properties
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.horizontalHeader().setStretchLastSection(True)

        # Style the table
        self.results_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #444;
                background-color: #2b2b2b;
                alternate-background-color: #353535;
            }
            QHeaderView::section {
                background-color: #404040;
                padding: 8px;
                border: none;
                font-weight: bold;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #444;
            }
            QTableWidget::item:selected {
                background-color: #7c4dff;
            }
        """)

        # Set initial placeholder
        self.setup_placeholder_table()
        layout.addWidget(self.results_table)

        # Results actions
        actions_layout = QHBoxLayout()

        actions_layout.addStretch()

        self.results_info_label = QLabel("No results available")
        self.results_info_label.setStyleSheet("color: #888; font-style: italic;")
        actions_layout.addWidget(self.results_info_label)

        layout.addLayout(actions_layout)

        group.setLayout(layout)
        return group

    def setup_placeholder_table(self):
        """Set up placeholder table"""
        self.results_table.setRowCount(1)
        self.results_table.setColumnCount(1)
        self.results_table.setHorizontalHeaderLabels(['Status'])
        placeholder_item = QTableWidgetItem("Run scoring process to see results here")
        placeholder_item.setFlags(Qt.ItemIsEnabled)
        self.results_table.setItem(0, 0, placeholder_item)

    def create_progress_bar(self):
        """Create progress bar and status area"""
        self.progress_group = QGroupBox()
        self.progress_group.setVisible(False)
        layout = QVBoxLayout()

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Status message
        self.progress_status = QLabel("Initializing...")
        self.progress_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_status)

        self.progress_group.setLayout(layout)

    def toggle_technical(self):
        """Toggle technical analysis option"""
        if self.include_technical_cb.isChecked():
            self.include_technical_cb.setText("Technical Analysis: Enabled")
        else:
            self.include_technical_cb.setText("Technical Analysis: Disabled")

    def run_scoring_process(self):
        """Run the multifactor scoring process with enhanced progress tracking"""
        mode = self.mode_combo.currentText()

        # Get the selected ticker file
        ticker_file = self.get_selected_ticker_file()

        if mode == "Historical (Range)":
            # Run historical batch scoring with auto-detected regimes
            start_date = self.start_date_edit.date().toPyDate()
            end_date = self.end_date_edit.date().toPyDate()
            interval_days = 1  # Always daily

            # Add a label to show current processing date
            if not hasattr(self, 'current_date_label'):
                self.current_date_label = QLabel("")
                self.current_date_label.setAlignment(Qt.AlignCenter)
                self.current_date_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
                # Add it to the progress group layout
                self.progress_group.layout().addWidget(self.current_date_label)

            # Calculate estimated trading days (rough estimate)
            total_days = (end_date - start_date).days + 1
            estimated_trading_days = int(total_days * 5 / 7)  # Roughly 5/7 are weekdays

            # Update UI
            self.progress_group.setVisible(True)
            self.progress_bar.setValue(0)
            self.progress_status.setText(
                f"Initializing analysis for ~{estimated_trading_days} trading days "
                f"({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})..."
            )
            self.current_date_label.setText("")

            # Disable button during processing
            self.run_scoring_btn.setEnabled(False)
            self.run_scoring_btn.setText("Processing Historical Range...")

            # Start historical batch scoring thread with ticker file
            self.scoring_thread = HistoricalBatchScoringThread(
                start_date, end_date, interval_days, ticker_file
            )

        elif mode == "Historical (Specific)":
            # Run scoring for date range with manually selected regime
            start_date = self.specific_start_date_edit.date().toPyDate()
            end_date = self.specific_end_date_edit.date().toPyDate()
            selected_regime = self.regime_combo.currentText()
            regime_key = selected_regime.replace(" ", "_").replace("/", "_")
            interval_days = 1  # Always daily

            # Calculate estimated trading days
            total_days = (end_date - start_date).days + 1
            estimated_trading_days = int(total_days * 5 / 7)  # Roughly 5/7 are weekdays

            # Update UI
            self.progress_group.setVisible(True)
            self.progress_bar.setValue(0)
            self.progress_status.setText(
                f"Processing ~{estimated_trading_days} trading days "
                f"({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}) "
                f"with {selected_regime} regime..."
            )

            # Disable button during processing
            self.run_scoring_btn.setEnabled(False)
            self.run_scoring_btn.setText(f"Processing with {selected_regime}...")
            # Start historical specific batch scoring thread with manual regime
            self.scoring_thread = HistoricalSpecificBatchScoringThread(
                start_date, end_date, interval_days, regime_key, ticker_file
            )

        else:  # Daily (Current)
            # Run daily scoring (existing logic)
            selected_regime = self.regime_combo.currentText()
            regime_key = selected_regime.replace(" ", "_").replace("/", "_")

            # Record start time for ETA calculation
            import time
            self.start_time = time.time()

            # Show progress bar
            self.progress_group.setVisible(True)
            self.progress_bar.setValue(0)
            self.progress_status.setText(f"Initializing {selected_regime} regime analysis...")

            # Reset statistics
            if hasattr(self, 'stats_labels'):
                self.stats_labels['total_stocks'].setText("Total Stocks to Analyze: --")
                self.stats_labels['top_performers'].setText("Currently Processing: --")
                self.stats_labels['avg_score'].setText("Average Score: --")
                self.stats_labels['processing_time'].setText("Processing Time: --")

            # Disable button during processing
            self.run_scoring_btn.setEnabled(False)
            self.run_scoring_btn.setText("Processing...")

            # Update info label
            self.results_info_label.setText(f"Running analysis for {selected_regime} regime...")

            # Start the enhanced scoring thread with ticker file
            self.scoring_thread = MultifactorScoringThread(regime_key, ticker_file)

        # Connect signals (same for all modes)
        self.scoring_thread.progress_update.connect(self.update_progress)
        self.scoring_thread.status_update.connect(self.update_status_with_details)
        self.scoring_thread.result_ready.connect(self.handle_results)
        self.scoring_thread.error_occurred.connect(self.handle_error)
        self.scoring_thread.finished.connect(self.on_scoring_finished)
        self.scoring_thread.start()

    def export_results(self):
        """Export results to CSV"""
        QMessageBox.information(self, "Export", "Export functionality will be implemented tomorrow!")

    def refresh_results(self):
        """Refresh the results display"""
        QMessageBox.information(self, "Refresh", "Refresh functionality will be implemented tomorrow!")

    def update_results_table(self, data):
        """Update the results table with scoring data"""
        # This method will be implemented tomorrow when we have actual data
        pass

    def update_weights_display(self, regime_name):
        """Update the weights display for the selected regime"""
        # Clear existing weights display
        for i in reversed(range(self.weights_layout.count())):
            child = self.weights_layout.itemAt(i).widget()
            if child:
                child.setParent(None)

        weights = self.regime_weights.get(regime_name, {})

        # Sort weights by absolute value (descending) for better visualization
        sorted_weights = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)

        for factor, weight in sorted_weights:
            # Create weight display row
            weight_row = QHBoxLayout()

            # Factor name
            factor_label = QLabel(factor.replace('_', ' ').title() + ":")
            factor_label.setMinimumWidth(120)
            weight_row.addWidget(factor_label)

            # Weight value with color coding
            weight_label = QLabel(f"{weight:+.1f}")
            weight_label.setAlignment(Qt.AlignRight)

            # Color coding: positive weights in green, negative in red
            if weight > 0:
                color = "#4CAF50"  # Green
            elif weight < 0:
                color = "#f44336"  # Red
            else:
                color = "#888888"  # Gray

            weight_label.setStyleSheet(f"""
                QLabel {{
                    color: {color};
                    font-weight: bold;
                    font-size: 13px;
                    padding: 3px 8px;
                    border-radius: 3px;
                    background-color: rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1);
                }}
            """)
            weight_row.addWidget(weight_label)

            # Add to layout
            weight_widget = QWidget()
            weight_widget.setLayout(weight_row)
            self.weights_layout.addWidget(weight_widget)

    def on_regime_changed(self, regime_name):
        """Handle regime selection change"""
        self.update_weights_display(regime_name)

    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)

    def update_status(self, message):
        """Update status message"""
        self.progress_status.setText(message)

    def handle_results(self, results):  # Note: parameter is 'results' not 'result'
        """Handle successful scoring results with enhanced statistics"""
        self.current_results = results

        if results.get('mode') == 'historical':  # Changed 'result' to 'results'
            # Handle historical results
            total = results.get('total_processed', 0)  # Changed 'result' to 'results'
            successful = len([r for r in results.get('results', []) if r.get('status') == 'success'])

            self.results_info_label.setText(
                f"Historical analysis complete!\n"
                f"Period: {results.get('start_date')} to {results.get('end_date')}\n"
                f"Processed: {successful}/{total} dates successfully"
            )

            # Update statistics
            if hasattr(self, 'stats_labels'):
                self.stats_labels['total_stocks'].setText(f"Total Dates Processed: {total}")
                self.stats_labels['top_performers'].setText(f"Successful: {successful}")

                # Show regime distribution
                regime_counts = {}
                for r in results.get('results', []):  # Changed 'result' to 'results'
                    if 'regime' in r:
                        regime = r['regime']
                        regime_counts[regime] = regime_counts.get(regime, 0) + 1

                regime_str = ", ".join([f"{k}: {v}" for k, v in regime_counts.items()])
                self.stats_labels['avg_score'].setText(f"Regimes: {regime_str}")
        else:
            # Handle daily results (existing logic)
            if results.get('success'):
                # Extract dataframe
                df = results.get('results_df')
                if df is not None and not df.empty:
                    # Display in table
                    self.display_results(df)

                    # Update statistics
                    if hasattr(self, 'stats_labels'):
                        self.stats_labels['total_stocks'].setText(
                            f"Total Stocks Analyzed: {results.get('total_stocks', len(df))}")

                        # Calculate top performers
                        top_10_pct = int(len(df) * 0.1)
                        top_performers = df.head(top_10_pct) if len(df) > 0 else df
                        self.stats_labels['top_performers'].setText(f"Top 10% Stocks: {len(top_performers)}")

                        # Calculate average score
                        if 'Score' in df.columns:
                            avg_score = df['Score'].mean()
                            self.stats_labels['avg_score'].setText(f"Average Score: {avg_score:.2f}")

                    # Update info label
                    regime = results.get('regime', 'Unknown')
                    weights_used = results.get('weights', {})
                    self.results_info_label.setText(
                        f"Analysis complete for {regime} regime\n"
                        f"Output: {results.get('output_path', 'N/A')}"
                    )
                else:
                    self.results_info_label.setText("No results to display")
            else:
                error_msg = results.get('error', 'Unknown error')
                self.results_info_label.setText(f"Analysis failed: {error_msg}")

    def handle_error(self, error_message):
        """Handle scoring errors"""
        QMessageBox.critical(self, "Scoring Error", f"An error occurred:\n{error_message}")
        self.results_info_label.setText("Error occurred during scoring")

    def on_scoring_finished(self):
        """Clean up after scoring process"""
        self.progress_group.setVisible(False)
        self.run_scoring_btn.setEnabled(True)
        self.run_scoring_btn.setText("Run Multifactor Scoring Process")

    def display_results(self, df):
        """Display results in the table"""
        if df is None or df.empty:
            self.setup_placeholder_table()
            return

        # Set up table with actual data
        self.results_table.setRowCount(len(df))
        self.results_table.setColumnCount(len(df.columns))
        self.results_table.setHorizontalHeaderLabels(df.columns.tolist())

        # Populate table
        for row in range(len(df)):
            for col in range(len(df.columns)):
                value = df.iloc[row, col]
                item = QTableWidgetItem(str(value))
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self.results_table.setItem(row, col, item)

        # Resize columns to content
        self.results_table.resizeColumnsToContents()

    def export_results(self):
        """Export results to CSV"""
        if self.current_results:
            output_path = self.current_results['output_path']
            QMessageBox.information(self, "Export Complete",
                                    f"Results already saved to:\n{output_path}")
        else:
            QMessageBox.information(self, "Export", "No results to export!")

    def refresh_results(self):
        """Refresh the results display"""
        if self.current_results:
            self.display_results(self.current_results['results_df'])
            QMessageBox.information(self, "Refresh", "Results refreshed!")
        else:
            QMessageBox.information(self, "Refresh", "No results to refresh!")

    def update_status_with_details(self, message):
        """Enhanced status update with more detailed information for historical processing"""
        self.progress_status.setText(message)

        # Extract and display current date being processed
        if "Processing" in message and "Day" in message:
            try:
                # Extract date from message like "Processing 2024-09-17 (Day 3/5)..."
                import re
                date_match = re.search(r'Processing (\d{4}-\d{2}-\d{2})', message)
                day_match = re.search(r'Day (\d+)/(\d+)', message)

                if date_match and day_match:
                    current_date = date_match.group(1)
                    current_day = day_match.group(1)
                    total_days = day_match.group(2)

                    # Update the current date label if it exists
                    if hasattr(self, 'current_date_label'):
                        self.current_date_label.setText(
                            f" Current: {current_date} (Trading Day {current_day} of {total_days})"
                        )

                    # Update statistics if available
                    if hasattr(self, 'stats_labels'):
                        self.stats_labels['total_stocks'].setText(f"Total Trading Days: {total_days}")
                        self.stats_labels['top_performers'].setText(f"Currently Processing: Day {current_day}")

                        # Calculate ETA
                        if hasattr(self, 'start_time') and int(current_day) > 0:
                            elapsed = time.time() - self.start_time
                            avg_time_per_day = elapsed / int(current_day)
                            remaining_days = int(total_days) - int(current_day)
                            estimated_remaining = remaining_days * avg_time_per_day

                            if estimated_remaining > 60:
                                eta_text = f"ETA: ~{estimated_remaining / 60:.1f} min"
                            else:
                                eta_text = f"ETA: ~{estimated_remaining:.0f} sec"

                            self.stats_labels['processing_time'].setText(eta_text)
            except Exception as e:
                pass  # Ignore parsing errors

        # Handle completion message
        elif "complete" in message.lower():
            if hasattr(self, 'current_date_label'):
                self.current_date_label.setText(" All trading days processed!")

            # Parse summary statistics if available
            try:
                if "Processed:" in message:
                    import re
                    processed = re.search(r'Processed: (\d+)', message)
                    skipped = re.search(r'Skipped existing: (\d+)', message)
                    errors = re.search(r'Errors: (\d+)', message)

                    if processed and hasattr(self, 'stats_labels'):
                        self.stats_labels['avg_score'].setText(
                            f"Summary: {processed.group(1)} new, "
                            f"{skipped.group(1) if skipped else '0'} existing, "
                            f"{errors.group(1) if errors else '0'} errors"
                        )
            except:
                pass

    def on_mode_changed(self, mode_text):
        """Handle mode selection change"""
        # Hide all date groups and notes first
        self.date_range_group.setVisible(False)
        self.date_specific_group.setVisible(False)
        self.daily_note.setVisible(False)
        self.historical_range_note.setVisible(False)
        self.historical_specific_note.setVisible(False)

        if mode_text == "Daily (Current)":
            # Daily mode: enable regime selection, show daily note
            self.regime_combo.setEnabled(True)
            self.daily_note.setVisible(True)

        elif mode_text == "Historical (Range)":
            # Historical range: show date range, disable regime (auto-detect)
            self.date_range_group.setVisible(True)
            self.regime_combo.setEnabled(False)
            self.historical_range_note.setVisible(True)

        elif mode_text == "Historical (Specific)":
            # Historical specific: show single date, enable regime selection
            self.date_specific_group.setVisible(True)
            self.regime_combo.setEnabled(True)
            self.historical_specific_note.setVisible(True)

    def get_interval_days(self):
        """Always returns 1 for daily interval"""
        return 1  # Always daily for optimal resolution


class HistoricalBatchScoringThread(QThread):
    """Thread for running historical scoring"""
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, start_date, end_date, interval_days, ticker_file=None):
        super().__init__()
        # Convert QDate to pandas Timestamp if needed
        if hasattr(start_date, 'toPyDate'):
            self.start_date = pd.Timestamp(start_date.toPyDate())
        else:
            self.start_date = pd.Timestamp(start_date)

        if hasattr(end_date, 'toPyDate'):
            self.end_date = pd.Timestamp(end_date.toPyDate())
        else:
            self.end_date = pd.Timestamp(end_date)

        self.interval_days = interval_days
        self.ticker_file = ticker_file  # NEW

    def run(self):
        """Execute historical scoring process"""
        try:
            # Add the scoring script path
            scoring_script_path = PROJECT_ROOT / 'src' / 'scoring' / 'stock_Screener_MultiFactor_25_new.py'

            if not scoring_script_path.exists():
                self.error_occurred.emit(f"Scoring script not found at {scoring_script_path}")
                return

            # Add parent directory to sys.path
            sys.path.insert(0, str(scoring_script_path.parent))

            try:
                # Import the module
                import stock_Screener_MultiFactor_25_new as screener

                # Set the ticker file if provided
                if self.ticker_file:
                    screener.set_ticker_file(self.ticker_file)  # Use the set_ticker_file function

                # Progress callback
                def progress_callback(callback_type, value):
                    if callback_type == 'progress':
                        self.progress_update.emit(value)
                    elif callback_type == 'status':
                        self.status_update.emit(value)
                    elif callback_type == 'error':
                        self.error_occurred.emit(value)

                # Run historical batch processing
                results = screener.run_historical_batch(
                    start_date=self.start_date,
                    end_date=self.end_date,
                    interval_days=self.interval_days,
                    progress_callback=progress_callback
                )

                # Emit results
                self.result_ready.emit({
                    'success': True,
                    'mode': 'historical',
                    'results': results,
                    'start_date': self.start_date.strftime('%Y-%m-%d'),
                    'end_date': self.end_date.strftime('%Y-%m-%d'),
                    'interval_days': self.interval_days,
                    'total_processed': len(results)
                })

            except ImportError as e:
                self.error_occurred.emit(f"Failed to import scoring module: {e}")
            except Exception as e:
                self.error_occurred.emit(f"Historical scoring failed: {e}")
            finally:
                # Clean up sys.path
                if str(scoring_script_path.parent) in sys.path:
                    sys.path.remove(str(scoring_script_path.parent))

        except Exception as e:
            self.error_occurred.emit(f"Thread execution failed: {e}")


class HistoricalSpecificScoringThread(QThread):
    """Thread for running scoring on a specific historical date with manual regime"""
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, target_date, regime_key, ticker_file=None):
        super().__init__()
        if hasattr(target_date, 'toPyDate'):
            self.target_date = pd.Timestamp(target_date.toPyDate())
        else:
            self.target_date = pd.Timestamp(target_date)
        self.regime_key = regime_key
        self.ticker_file = ticker_file  # NEW

    def run(self):
        """Execute scoring for specific historical date"""
        try:
            # Add the scoring script path
            scoring_script_path = PROJECT_ROOT / 'src' / 'scoring' / 'stock_Screener_MultiFactor_25_new.py'

            if not scoring_script_path.exists():
                self.error_occurred.emit(f"Scoring script not found at {scoring_script_path}")
                return

            # Add parent directory to sys.path
            sys.path.insert(0, str(scoring_script_path.parent))

            try:
                # Import the module
                import stock_Screener_MultiFactor_25_new as screener

                # Set the ticker file if provided
                if self.ticker_file:
                    screener.set_ticker_file(self.ticker_file)  # Use the set_ticker_file function

                # Progress callback
                def progress_callback(callback_type, value):
                    if callback_type == 'progress':
                        self.progress_update.emit(value)
                    elif callback_type == 'status':
                        self.status_update.emit(value)
                    elif callback_type == 'error':
                        self.error_occurred.emit(value)

                # Run scoring for specific date with manual regime
                result = screener.run_historical_specific_date(
                    target_date=self.target_date,
                    regime=self.regime_key,
                    progress_callback=progress_callback
                )

                # Emit results
                self.result_ready.emit({
                    'success': True,
                    'mode': 'historical_specific',
                    'date': self.target_date.strftime('%Y-%m-%d'),
                    'regime': self.regime_key,
                    'output_path': result.get('output_path'),
                    'results_df': result.get('results_df')
                })

            except ImportError as e:
                self.error_occurred.emit(f"Failed to import scoring module: {e}")
            except Exception as e:
                self.error_occurred.emit(f"Historical specific scoring failed: {e}")
            finally:
                # Clean up sys.path
                if str(scoring_script_path.parent) in sys.path:
                    sys.path.remove(str(scoring_script_path.parent))

        except Exception as e:
            self.error_occurred.emit(f"Thread execution failed: {e}")


class HistoricalSpecificBatchScoringThread(QThread):
    """Thread for running scoring on a date range with manual regime"""
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, start_date, end_date, interval_days, regime_key, ticker_file=None):
        super().__init__()
        # Convert QDate to pandas Timestamp if needed
        if hasattr(start_date, 'toPyDate'):
            self.start_date = pd.Timestamp(start_date.toPyDate())
        else:
            self.start_date = pd.Timestamp(start_date)

        if hasattr(end_date, 'toPyDate'):
            self.end_date = pd.Timestamp(end_date.toPyDate())
        else:
            self.end_date = pd.Timestamp(end_date)

        self.interval_days = interval_days
        self.regime_key = regime_key
        self.ticker_file = ticker_file

    def run(self):
        """Execute scoring for date range with manual regime"""
        try:
            # Add the scoring script path
            scoring_script_path = PROJECT_ROOT / 'src' / 'scoring' / 'stock_Screener_MultiFactor_25_new.py'

            if not scoring_script_path.exists():
                self.error_occurred.emit(f"Scoring script not found at {scoring_script_path}")
                return

            # Add parent directory to sys.path
            sys.path.insert(0, str(scoring_script_path.parent))

            try:
                # Import the module
                import stock_Screener_MultiFactor_25_new as screener

                # Set the ticker file if provided
                if self.ticker_file:
                    screener.set_ticker_file(self.ticker_file)

                # Progress callback
                def progress_callback(callback_type, value):
                    if callback_type == 'progress':
                        self.progress_update.emit(value)
                    elif callback_type == 'status':
                        self.status_update.emit(value)
                    elif callback_type == 'error':
                        self.error_occurred.emit(value)

                # Run historical batch processing with manual regime
                results = screener.run_historical_batch_with_regime(
                    start_date=self.start_date,
                    end_date=self.end_date,
                    manual_regime=self.regime_key,
                    interval_days=self.interval_days,
                    progress_callback=progress_callback
                )

                # Emit results
                self.result_ready.emit({
                    'success': True,
                    'mode': 'historical_specific_range',
                    'results': results,
                    'start_date': self.start_date.strftime('%Y-%m-%d'),
                    'end_date': self.end_date.strftime('%Y-%m-%d'),
                    'regime': self.regime_key,
                    'interval_days': self.interval_days,
                    'total_processed': len(results)
                })

            except ImportError as e:
                self.error_occurred.emit(f"Failed to import scoring module: {e}")
            except Exception as e:
                self.error_occurred.emit(f"Historical specific batch scoring failed: {e}")
            finally:
                # Clean up sys.path
                if str(scoring_script_path.parent) in sys.path:
                    sys.path.remove(str(scoring_script_path.parent))

        except Exception as e:
            self.error_occurred.emit(f"Thread execution failed: {e}")