import sys
import os
import json
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QTextEdit,
                             QGroupBox, QSplitter, QTabWidget, QTableWidget,
                             QTableWidgetItem, QProgressBar, QComboBox,
                             QMessageBox, QHeaderView, QGridLayout, QTableView,
                             QAbstractItemView)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QAbstractTableModel
from PyQt5.QtGui import QFont, QPixmap, QPalette, QColor
from PyQt5.QtWidgets import QScrollArea
import pyqtgraph as pg
from pyqtgraph import DateAxisItem
import matplotlib.cm as cm

# Get project root directory (Current directory)
PROJECT_ROOT = Path(file).parent if 'file' in globals() else Path.cwd()


class MultifactorScoringWidget(QWidget):
    """Main widget for multifactor scoring system"""

    def __init__(self):
        super().__init__()
        self.scoring_thread = None
        self.current_results = None
        self.init_ui()

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
        subtitle_label = QLabel("Advanced multi-factor stock analysis and ranking system")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("color: #888;")
        main_layout.addWidget(subtitle_label)

        # Create horizontal layout for controls
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(20)

        # Control Panel
        control_panel = self.create_control_panel()
        controls_layout.addWidget(control_panel, 1)

        # Status Panel
        status_panel = self.create_status_panel()
        controls_layout.addWidget(status_panel, 1)

        main_layout.addLayout(controls_layout)

        # Results section
        results_section = self.create_results_section()
        main_layout.addWidget(results_section)

        # Progress and status bar
        self.create_progress_bar()
        main_layout.addWidget(self.progress_group)

        self.setLayout(main_layout)

    def create_control_panel(self):
        """Create control panel with button and regime selector"""
        group = QGroupBox("Scoring Controls")
        layout = QVBoxLayout()

        # Regime Selection
        regime_layout = QHBoxLayout()
        regime_layout.addWidget(QLabel("Target Regime:"))

        self.regime_combo = QComboBox()
        self.regime_combo.addItems(["Strong Bull", "Steady Growth", "Crisis/Bear"])
        self.regime_combo.setCurrentText("Steady Growth")
        self.regime_combo.setToolTip("Select the market regime for optimized factor weights")
        regime_layout.addWidget(self.regime_combo)
        regime_layout.addStretch()
        layout.addLayout(regime_layout)

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

        # Additional Options (for future expansion)
        options_label = QLabel("Additional Options")
        options_label.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(options_label)

        # Placeholder for additional controls
        options_layout = QVBoxLayout()

        self.include_technical_cb = QPushButton("Technical Analysis: Enabled")
        self.include_technical_cb.setCheckable(True)
        self.include_technical_cb.setChecked(True)
        self.include_technical_cb.clicked.connect(self.toggle_technical)
        self.include_technical_cb.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 8px;
                border-radius: 4px;
                text-align: left;
            }
            QPushButton:checked {
                background-color: #1976D2;
            }
            QPushButton:unchecked {
                background-color: #757575;
            }
        """)
        options_layout.addWidget(self.include_technical_cb)

        layout.addLayout(options_layout)
        layout.addStretch()

        group.setLayout(layout)
        return group

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
        self.results_table.setRowCount(1)
        self.results_table.setColumnCount(1)
        self.results_table.setHorizontalHeaderLabels(['Status'])
        placeholder_item = QTableWidgetItem("Run scoring process to see results here")
        placeholder_item.setFlags(Qt.ItemIsEnabled)  # Make it non-selectable
        self.results_table.setItem(0, 0, placeholder_item)

        layout.addWidget(self.results_table)

        # Results actions
        actions_layout = QHBoxLayout()

        self.export_btn = QPushButton("Export to CSV")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)  # Disabled until we have results
        actions_layout.addWidget(self.export_btn)

        self.refresh_btn = QPushButton("Refresh Results")
        self.refresh_btn.clicked.connect(self.refresh_results)
        self.refresh_btn.setEnabled(False)  # Disabled until we have results
        actions_layout.addWidget(self.refresh_btn)

        actions_layout.addStretch()

        self.results_info_label = QLabel("No results available")
        self.results_info_label.setStyleSheet("color: #888; font-style: italic;")
        actions_layout.addWidget(self.results_info_label)

        layout.addLayout(actions_layout)

        group.setLayout(layout)
        return group

    def create_progress_bar(self):
        """Create progress bar and status area"""
        self.progress_group = QGroupBox()
        self.progress_group.setVisible(False)  # Hidden by default
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
        """Run the multifactor scoring process"""
        # This will be implemented tomorrow
        selected_regime = self.regime_combo.currentText()
        technical_enabled = self.include_technical_cb.isChecked()

        # Show progress bar
        self.progress_group.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_status.setText(f"Starting scoring process for {selected_regime} regime...")

        # Disable button during processing
        self.run_scoring_btn.setEnabled(False)
        self.run_scoring_btn.setText("Processing...")

        # Update status
        self.status_display.setText(f"Running scoring for {selected_regime} regime...")

        # TODO: Tomorrow we'll implement the actual scoring logic
        # For now, just show a placeholder message
        QMessageBox.information(self, "Scoring Process",
                                f"Multifactor scoring will be implemented tomorrow!\n\n"
                                f"Selected regime: {selected_regime}\n"
                                f"Technical analysis: {'Enabled' if technical_enabled else 'Disabled'}")

        # Reset UI state
        self.progress_group.setVisible(False)
        self.run_scoring_btn.setEnabled(True)
        self.run_scoring_btn.setText("Run Multifactor Scoring Process")
        self.status_display.setText("Ready to run scoring process")

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

class RegimeDetectorThread(QThread):
    """Thread for running regime detection without blocking UI"""
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, detector_type='current', action='detect'):
        super().__init__()
        self.detector_type = detector_type
        self.action = action  # 'detect', 'fetch', or 'load'

    def run(self):
        try:
            if self.detector_type == 'current':
                self.run_current_detection()
            elif self.detector_type == 'historical':
                if self.action == 'fetch':
                    self.fetch_historical_data()
                elif self.action == 'load':
                    self.load_historical_data()

        except Exception as e:
            self.error_occurred.emit(str(e))

    def run_current_detection(self):
        """Run current regime detection"""
        self.status_update.emit("Starting current regime detection...")
        self.progress_update.emit(20)

        # Path to current regime detector script
        script_path = PROJECT_ROOT / 'src' / 'regime_detection' / 'current_regime_detector.py'

        if not script_path.exists():
            self.error_occurred.emit(f"Script not found: {script_path}")
            return

        self.status_update.emit("Running detection script...")
        self.progress_update.emit(40)

        # Run the script
        # result = subprocess.run(
        #     [sys.executable, str(script_path)],
        #     capture_output=True,
        #     text=True,
        #     cwd=str(PROJECT_ROOT)
        # )

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            encoding='utf-8',  # Add this line
            cwd=str(PROJECT_ROOT)
        )

        if result.returncode != 0 and result.stderr:
            self.error_occurred.emit(f"Script error: {result.stderr}")
            return

        self.progress_update.emit(80)

        # Read the output file
        output_file = PROJECT_ROOT / 'output' / 'Regime_Detection_Analysis' / 'current_regime_analysis.json'

        if output_file.exists():
            with open(output_file, 'r') as f:
                data = json.load(f)

            self.progress_update.emit(100)
            self.status_update.emit("Detection complete!")
            self.result_ready.emit({'type': 'current', 'data': data})
        else:
            self.error_occurred.emit("Output file not generated")

    def fetch_historical_data(self):
        """Fetch historical regime data"""
        self.status_update.emit("Fetching historical regime data...")
        self.progress_update.emit(20)

        # Path to regime detector script
        script_path = PROJECT_ROOT / 'src' / 'regime_detection' / 'regime_detector.py'

        if not script_path.exists():
            self.error_occurred.emit(f"Script not found: {script_path}")
            return

        self.status_update.emit("Fetching market data...")
        self.progress_update.emit(40)

        # Run the script
        # result = subprocess.run(
        #     [sys.executable, str(script_path)],
        #     capture_output=True,
        #     text=True,
        #     cwd=str(PROJECT_ROOT)
        # )

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            encoding='utf-8',  # Add this line
            cwd=str(PROJECT_ROOT)
        )

        self.progress_update.emit(60)

        if result.returncode != 0 and result.stderr:
            # Check if it's just a warning
            if "WARNING" not in result.stderr:
                self.error_occurred.emit(f"Script error: {result.stderr}")
                return

        self.status_update.emit("Processing results...")
        self.progress_update.emit(80)

        # Read the output files
        output_dir = PROJECT_ROOT / 'output' / 'Regime_Detection_Analysis'
        results = {}

        # Read each output file
        files_to_read = {
            'data_summary': 'data_summary.json',
            'validation_results': 'validation_results.json',
            'regime_periods': 'regime_periods.csv'
        }

        for key, filename in files_to_read.items():
            file_path = output_dir / filename
            if file_path.exists():
                if filename.endswith('.json'):
                    with open(file_path, 'r') as f:
                        results[key] = json.load(f)
                elif filename.endswith('.csv'):
                    results[key] = pd.read_csv(file_path).to_dict('records')

        # Also check for the plot image
        plot_path = PROJECT_ROOT / 'output' / 'Regime_Detection_Results' / 'regime_detection_plot.png'
        if plot_path.exists():
            results['plot_path'] = str(plot_path)

        self.progress_update.emit(100)
        self.status_update.emit("Historical data fetched successfully!")
        self.result_ready.emit({'type': 'historical', 'data': results})

    def load_historical_data(self):
        """Load saved historical data"""
        self.status_update.emit("Loading saved data...")
        self.progress_update.emit(50)

        output_dir = PROJECT_ROOT / 'output' / 'Regime_Detection_Analysis'
        results = {}

        # Load saved files
        files_to_load = {
            'data_summary': 'data_summary.json',
            'validation_results': 'validation_results.json',
            'regime_periods': 'regime_periods.csv'
        }

        for key, filename in files_to_load.items():
            file_path = output_dir / filename
            if file_path.exists():
                if filename.endswith('.json'):
                    with open(file_path, 'r') as f:
                        results[key] = json.load(f)
                elif filename.endswith('.csv'):
                    results[key] = pd.read_csv(file_path).to_dict('records')

        # Check for plot
        plot_path = PROJECT_ROOT / 'output' / 'Regime_Detection_Results' / 'regime_detection_plot.png'
        if plot_path.exists():
            results['plot_path'] = str(plot_path)

        if results:
            self.progress_update.emit(100)
            self.status_update.emit("Data loaded successfully!")
            self.result_ready.emit({'type': 'historical_load', 'data': results})
        else:
            self.error_occurred.emit("No saved data found")


class RegimeDetectionWidget(QWidget):
    """Main widget for regime detection visualization"""

    def __init__(self):
        super().__init__()
        self.detector_thread = None
        self.current_results = None
        self.historical_results = None
        self.init_ui()

    def init_ui(self):
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)

        # Title
        title_label = QLabel("Market Regime Detection")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont("Arial", 18, QFont.Bold)
        title_label.setFont(title_font)
        main_layout.addWidget(title_label)

        # Subtitle
        subtitle_label = QLabel("Using Hidden Markov Models to identify current market conditions")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("color: #888;")
        main_layout.addWidget(subtitle_label)

        # Create horizontal layout for the three sections
        sections_layout = QHBoxLayout()
        sections_layout.setSpacing(15)

        # Section 1: Current Regime
        current_section = self.create_current_regime_section()
        sections_layout.addWidget(current_section, 1)

        # Section 2: Historical Regimes
        historical_section = self.create_historical_regime_section()
        sections_layout.addWidget(historical_section, 1)

        main_layout.addLayout(sections_layout)

        # Section 3: Regime Timeline Visualization
        timeline_section = self.create_timeline_section()
        main_layout.addWidget(timeline_section)

        # Status bar
        self.create_status_bar()
        main_layout.addWidget(self.status_group)

        self.setLayout(main_layout)

    def create_current_regime_section(self):
        """Create current regime analysis section"""
        group = QGroupBox("Current Regime Analysis")
        layout = QVBoxLayout()

        # Data source selector
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Data Source:"))
        self.current_source_combo = QComboBox()
        self.current_source_combo.addItems(["Yahoo Finance", "Marketstack"])
        self.current_source_combo.setCurrentText("Yahoo Finance")
        source_layout.addWidget(self.current_source_combo)
        source_layout.addStretch()
        layout.addLayout(source_layout)

        # Run Detection button
        self.detect_current_btn = QPushButton("Run Detection")
        self.detect_current_btn.clicked.connect(self.detect_current_regime)
        self.detect_current_btn.setStyleSheet("""
            QPushButton {
                background-color: #7c4dff;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #651fff;
            }
        """)
        layout.addWidget(self.detect_current_btn)

        # Current regime display
        self.current_regime_label = QLabel("Current Regime: Not Detected")
        self.current_regime_label.setAlignment(Qt.AlignCenter)
        regime_font = QFont("Arial", 14, QFont.Bold)
        self.current_regime_label.setFont(regime_font)
        self.current_regime_label.setStyleSheet("""
            QLabel {
                padding: 15px;
                border-radius: 8px;
                background-color: #f0f0f0;
                color: #333;
            }
        """)
        layout.addWidget(self.current_regime_label)

        # Regime Probabilities
        prob_label = QLabel("Regime Probabilities")
        prob_label.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(prob_label)

        # Probability displays
        self.prob_labels = {}
        regimes = ["Steady Growth", "Strong Bull", "Crisis/Bear"]
        for regime in regimes:
            prob_layout = QHBoxLayout()
            regime_label = QLabel(regime + ":")
            regime_label.setMinimumWidth(100)
            prob_layout.addWidget(regime_label)

            value_label = QLabel("--")
            value_label.setAlignment(Qt.AlignRight)
            self.prob_labels[regime] = value_label
            prob_layout.addWidget(value_label)

            layout.addLayout(prob_layout)

        layout.addStretch()
        group.setLayout(layout)
        return group

    def calculate_regime_percentages(self):
        """Calculate regime percentages from regime_periods.csv"""
        try:
            # Try different possible locations for the CSV file
            possible_paths = [
                PROJECT_ROOT / 'output' / 'Regime_Detection_Analysis' / 'regime_periods.csv',
                PROJECT_ROOT / 'regime_periods.csv'
            ]

            regime_periods_path = None
            for path in possible_paths:
                if path.exists():
                    regime_periods_path = path
                    break

            if regime_periods_path:
                df = pd.read_csv(regime_periods_path)

                # Group by regime_name and sum days
                regime_stats = df.groupby('regime_name')['days'].sum()
                total_days = regime_stats.sum()

                # Calculate percentages
                regime_percentages = {}
                for regime_name, days in regime_stats.items():
                    percentage = (days / total_days) * 100
                    regime_percentages[regime_name] = percentage

                return regime_percentages
            else:
                return {}
        except Exception as e:
            print(f"Error calculating regime percentages: {e}")
            return {}

    def create_historical_regime_section(self):
        """Create historical regime section"""
        group = QGroupBox("Historical Regimes")
        layout = QVBoxLayout()

        # Data source selector
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Data Source:"))
        self.historical_source_combo = QComboBox()
        self.historical_source_combo.addItems(["Yahoo Finance (10 years)", "Marketstack (10 years)"])
        self.historical_source_combo.setCurrentText("Yahoo Finance (10 years)")
        source_layout.addWidget(self.historical_source_combo)
        source_layout.addStretch()
        layout.addLayout(source_layout)

        # Buttons
        button_layout = QHBoxLayout()

        self.fetch_data_btn = QPushButton("Fetch Regime Data")
        self.fetch_data_btn.clicked.connect(self.fetch_historical_data)
        self.fetch_data_btn.setStyleSheet("""
            QPushButton {
                background-color: #7c4dff;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #651fff;
            }
        """)
        button_layout.addWidget(self.fetch_data_btn)

        self.load_data_btn = QPushButton("Load Saved Data")
        self.load_data_btn.clicked.connect(self.load_historical_data)
        self.load_data_btn.setStyleSheet("""
            QPushButton {
                background-color: #00acc1;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0097a7;
            }
        """)
        button_layout.addWidget(self.load_data_btn)

        layout.addLayout(button_layout)

        # Results display
        results_label = QLabel("Analysis Results")
        results_label.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(results_label)

        # Regime probabilities for historical
        self.hist_prob_labels = {}
        regimes = ["Steady Growth", "Strong Bull", "Crisis/Bear"]
        for regime in regimes:
            prob_layout = QHBoxLayout()
            regime_label = QLabel(regime + ":")
            regime_label.setMinimumWidth(100)
            prob_layout.addWidget(regime_label)

            value_label = QLabel("--")
            value_label.setAlignment(Qt.AlignRight)
            self.hist_prob_labels[regime] = value_label
            prob_layout.addWidget(value_label)

            layout.addLayout(prob_layout)

        # Validation accuracy
        accuracy_layout = QHBoxLayout()
        accuracy_label = QLabel("Validation Accuracy:")
        accuracy_label.setMinimumWidth(100)
        accuracy_layout.addWidget(accuracy_label)

        self.accuracy_label = QLabel("--")
        self.accuracy_label.setAlignment(Qt.AlignRight)
        accuracy_layout.addWidget(self.accuracy_label)
        layout.addLayout(accuracy_layout)

        layout.addStretch()
        group.setLayout(layout)
        return group

    def update_timeline_plot(self):
        """Update the regime timeline plot based on selected period"""
        try:
            # Clear existing plot and legend
            self.regime_plot.clear()

            # Load data files - try multiple possible locations
            possible_market_paths = [
                PROJECT_ROOT / 'output' / 'Regime_Detection_Analysis' / 'historical_market_data.csv',
                PROJECT_ROOT / 'historical_market_data.csv'
            ]
            possible_regime_paths = [
                PROJECT_ROOT / 'output' / 'Regime_Detection_Analysis' / 'regime_periods.csv',
                PROJECT_ROOT / 'regime_periods.csv'
            ]

            market_data_path = None
            regime_periods_path = None

            for path in possible_market_paths:
                if path.exists():
                    market_data_path = path
                    break

            for path in possible_regime_paths:
                if path.exists():
                    regime_periods_path = path
                    break

            if not market_data_path or not regime_periods_path:
                self.update_status("Data files not found. Please run historical detection first.")
                return

            # Load market data
            market_df = pd.read_csv(market_data_path)
            market_df['Date'] = pd.to_datetime(market_df['Date'])
            market_df = market_df.sort_values('Date')

            # Load regime periods
            regime_df = pd.read_csv(regime_periods_path)
            regime_df['start_date'] = pd.to_datetime(regime_df['start_date'])
            regime_df['end_date'] = pd.to_datetime(regime_df['end_date'])

            # Filter data based on selected time period
            period_text = self.period_combo.currentText()
            end_date = market_df['Date'].max()

            if period_text == "30 Days":
                start_date = end_date - pd.Timedelta(days=30)
            elif period_text == "3 Months":
                start_date = end_date - pd.Timedelta(days=90)
            elif period_text == "1 Year":
                start_date = end_date - pd.Timedelta(days=365)
            elif period_text == "5 Years":
                start_date = end_date - pd.Timedelta(days=365 * 5)
            else:  # All Time
                start_date = market_df['Date'].min()

            # Filter market data
            filtered_data = market_df[
                (market_df['Date'] >= start_date) &
                (market_df['Date'] <= end_date)
                ].copy()

            if filtered_data.empty:
                self.update_status("No data available for selected period")
                return

            # Check for valid SPY data and handle NaN values
            if 'SPY' not in filtered_data.columns or filtered_data['SPY'].isna().all():
                self.update_status("No valid SPY data available")
                return

            # Remove NaN values
            filtered_data = filtered_data.dropna(subset=['SPY'])

            if filtered_data.empty:
                self.update_status("No valid data after removing NaN values")
                return

            # Convert dates to timestamps for plotting
            x = filtered_data['Date'].apply(lambda x: x.timestamp()).values
            y = filtered_data['SPY'].values

            # Validate data ranges
            if len(x) == 0 or len(y) == 0:
                self.update_status("No valid data points to plot")
                return

            # Define colors for regimes (only three main colors to avoid legend duplication)
            regime_colors = {
                'Steady Growth': '#2196F3',  # Blue
                'Strong Bull': '#4CAF50',  # Green
                'Crisis/Bear': '#f44336'  # Red
            }

            # Keep track of which regimes we've already added to legend
            legend_added = set()

            # Plot each regime period
            for _, regime_period in regime_df.iterrows():
                regime_name = regime_period['regime_name']
                period_start = regime_period['start_date']
                period_end = regime_period['end_date']

                # Check if this regime period overlaps with our filtered time range
                if period_end < start_date or period_start > end_date:
                    continue

                # Get the regime data for this period
                regime_mask = (
                        (filtered_data['Date'] >= period_start) &
                        (filtered_data['Date'] <= period_end)
                )
                regime_data = filtered_data[regime_mask]

                if not regime_data.empty and len(regime_data) > 1:  # Need at least 2 points to plot
                    regime_x = regime_data['Date'].apply(lambda x: x.timestamp()).values
                    regime_y = regime_data['SPY'].values

                    # Skip if data contains NaN values
                    if pd.isna(regime_x).any() or pd.isna(regime_y).any():
                        continue

                    # Get color for this regime
                    color = regime_colors.get(regime_name, '#888888')

                    # Create pen for plotting
                    pen = pg.mkPen(color=color, width=2)

                    # Only add to legend if this regime hasn't been added yet
                    legend_name = regime_name if regime_name not in legend_added else None
                    if legend_name:
                        legend_added.add(regime_name)

                    # Plot this regime segment
                    self.regime_plot.plot(
                        regime_x,
                        regime_y,
                        pen=pen,
                        name=legend_name
                    )

            # Set axis ranges with proper bounds checking
            if len(x) > 0 and len(y) > 0 and not (pd.isna(x).any() or pd.isna(y).any()):
                x_min, x_max = float(x.min()), float(x.max())
                y_min, y_max = float(y.min()), float(y.max())

                # Ensure we don't have NaN values
                if not (pd.isna(x_min) or pd.isna(x_max) or pd.isna(y_min) or pd.isna(y_max)):
                    # Add some padding to y-axis
                    y_padding = (y_max - y_min) * 0.05

                    self.regime_plot.setXRange(x_min, x_max)
                    self.regime_plot.setYRange(y_min - y_padding, y_max + y_padding)

            self.update_status(f"Plot updated: {period_text}")

        except Exception as e:
            self.update_status(f"Error updating plot: {str(e)}")
            print(f"Plot error details: {e}")
            import traceback
            traceback.print_exc()

    def create_timeline_section(self):
        """Create timeline visualization section with interactive plot"""
        group = QGroupBox("Regime Timeline Visualization")
        layout = QVBoxLayout()

        # Add controls for time period selection
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Time Period:"))

        self.period_combo = QComboBox()
        self.period_combo.addItems(["30 Days", "3 Months", "1 Year", "5 Years", "All Time"])
        self.period_combo.currentTextChanged.connect(self.update_timeline_plot)
        controls_layout.addWidget(self.period_combo)

        self.refresh_plot_btn = QPushButton("Refresh Plot")
        self.refresh_plot_btn.clicked.connect(self.update_timeline_plot)
        controls_layout.addWidget(self.refresh_plot_btn)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Create plot widget using pyqtgraph for interactive plotting
        self.regime_plot = pg.PlotWidget(
            title="SPY Price Colored by Regime",
            axisItems={'bottom': DateAxisItem(orientation='bottom')}
        )
        self.regime_plot.setLabel('left', 'Price ($)')
        self.regime_plot.setLabel('bottom', 'Date')
        self.regime_plot.showGrid(x=True, y=True, alpha=0.3)
        self.regime_plot.setMinimumHeight(400)

        # Add legend
        self.regime_plot.addLegend()

        layout.addWidget(self.regime_plot)

        group.setLayout(layout)
        return group

    def create_status_bar(self):
        """Create status bar for showing progress and messages"""
        self.status_group = QGroupBox()
        status_layout = QHBoxLayout()

        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label)

        status_layout.addStretch()

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(300)
        status_layout.addWidget(self.progress_bar)

        self.status_group.setLayout(status_layout)

    def detect_current_regime(self):
        """Run current regime detection"""
        self.detect_current_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.detector_thread = RegimeDetectorThread(detector_type='current')
        self.detector_thread.progress_update.connect(self.update_progress)
        self.detector_thread.status_update.connect(self.update_status)
        self.detector_thread.result_ready.connect(self.handle_results)
        self.detector_thread.error_occurred.connect(self.handle_error)
        self.detector_thread.finished.connect(lambda: self.detect_current_btn.setEnabled(True))
        self.detector_thread.start()

    def fetch_historical_data(self):
        """Fetch historical regime data"""
        self.fetch_data_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.detector_thread = RegimeDetectorThread(detector_type='historical', action='fetch')
        self.detector_thread.progress_update.connect(self.update_progress)
        self.detector_thread.status_update.connect(self.update_status)
        self.detector_thread.result_ready.connect(self.handle_results)
        self.detector_thread.error_occurred.connect(self.handle_error)
        self.detector_thread.finished.connect(lambda: self.fetch_data_btn.setEnabled(True))
        self.detector_thread.start()

    def load_historical_data(self):
        """Load saved historical data"""
        self.load_data_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.detector_thread = RegimeDetectorThread(detector_type='historical', action='load')
        self.detector_thread.progress_update.connect(self.update_progress)
        self.detector_thread.status_update.connect(self.update_status)
        self.detector_thread.result_ready.connect(self.handle_results)
        self.detector_thread.error_occurred.connect(self.handle_error)
        self.detector_thread.finished.connect(lambda: self.load_data_btn.setEnabled(True))
        self.detector_thread.start()

    def handle_results(self, results):
        """Handle results from detection threads"""
        result_type = results['type']
        data = results['data']

        if result_type == 'current':
            self.display_current_results(data)
        elif result_type in ['historical', 'historical_load']:
            self.display_historical_results(data)
            # Update the timeline plot after loading historical data
            self.update_timeline_plot()

        self.progress_bar.setVisible(False)

    def display_current_results(self, data):
        """Display current regime detection results"""
        # Extract regime information
        regime_info = data.get('regime_detection', {})

        # Update current regime label
        current_regime = regime_info.get('regime_name', 'Unknown')
        self.current_regime_label.setText(f"Current Regime: {current_regime}")

        # Color coding based on regime
        if 'Bull' in current_regime:
            color = '#4CAF50'  # Green
        elif 'Growth' in current_regime:
            color = '#2196F3'  # Blue
        elif 'Bear' in current_regime or 'Crisis' in current_regime:
            color = '#f44336'  # Red
        else:
            color = '#757575'  # Grey

        self.current_regime_label.setStyleSheet(f"""
            QLabel {{
                padding: 15px;
                border-radius: 8px;
                background-color: {color};
                color: white;
            }}
        """)

        # Update probabilities
        # self.prob_labels["Steady Growth"].setText(f"{regime_info.get('prob_growth', 0):.2%}")
        # self.prob_labels["Strong Bull"].setText(f"{regime_info.get('prob_bull', 0):.2%}")
        # self.prob_labels["Crisis/Bear"].setText(f"{regime_info.get('prob_bear', 0):.2%}")

        # Update probabilities - FIX: Use all_probabilities instead
        all_probs = regime_info.get('all_probabilities', {})
        self.prob_labels["Steady Growth"].setText(f"{all_probs.get('Steady Growth', 0):.2%}")
        self.prob_labels["Strong Bull"].setText(f"{all_probs.get('Strong Bull', 0):.2%}")
        self.prob_labels["Crisis/Bear"].setText(f"{all_probs.get('Crisis/Bear', 0):.2%}")

        self.update_status("Current regime detected successfully")

    # def display_historical_results(self, data):
    #     """Display historical regime analysis results"""
    #     # Update validation accuracy if available
    #     if 'validation_results' in data:
    #         validation = data['validation_results']
    #         accuracy = validation.get('accuracy', 0)
    #         self.accuracy_label.setText(f"{accuracy:.2%}")
    #
    #     # Update regime statistics if available
    #     if 'data_summary' in data:
    #         summary = data['data_summary']
    #         regime_stats = summary.get('regime_statistics', {})
    #
    #         # Map regime statistics to display labels
    #         total_days = 0
    #         regime_days = {}
    #
    #         for regime_name, stats in regime_stats.items():
    #             if 'total_days' in stats:
    #                 regime_days[regime_name] = stats['total_days']
    #                 total_days += stats['total_days']
    #
    #         # Calculate percentages
    #         if total_days > 0:
    #             for regime_name, days in regime_days.items():
    #                 percentage = (days / total_days) * 100
    #
    #                 if 'Growth' in regime_name:
    #                     self.hist_prob_labels["Steady Growth"].setText(f"{percentage:.1f}%")
    #                 elif 'Bull' in regime_name:
    #                     self.hist_prob_labels["Strong Bull"].setText(f"{percentage:.1f}%")
    #                 elif 'Bear' in regime_name or 'Crisis' in regime_name:
    #                     self.hist_prob_labels["Crisis/Bear"].setText(f"{percentage:.1f}%")
    #
    #     # Display plot if available
    #     if 'plot_path' in data:
    #         self.display_timeline_plot(data['plot_path'])
    #
    #     self.update_status("Historical data loaded successfully")

    def display_historical_results(self, data):
        """Display historical regime analysis results"""
        # Update validation accuracy if available
        if 'validation_results' in data:
            validation = data['validation_results']
            accuracy = validation.get('accuracy', 0)
            self.accuracy_label.setText(f"{accuracy:.2%}")

        # Calculate and display regime percentages from CSV file
        regime_percentages = self.calculate_regime_percentages()

        if regime_percentages:
            # Update the percentage labels
            for regime_name, percentage in regime_percentages.items():
                if 'Growth' in regime_name or regime_name == 'Steady Growth':
                    self.hist_prob_labels["Steady Growth"].setText(f"{percentage:.1f}%")
                elif 'Bull' in regime_name or regime_name == 'Strong Bull':
                    self.hist_prob_labels["Strong Bull"].setText(f"{percentage:.1f}%")
                elif 'Bear' in regime_name or 'Crisis' in regime_name or regime_name == 'Crisis/Bear':
                    self.hist_prob_labels["Crisis/Bear"].setText(f"{percentage:.1f}%")

        self.update_status("Historical data loaded successfully")

    # def display_timeline_plot(self, plot_path):
    #     """Display the regime timeline plot"""
    #     if os.path.exists(plot_path):
    #         pixmap = QPixmap(plot_path)
    #         # Scale the image to fit
    #         scaled_pixmap = pixmap.scaled(1200, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    #         self.timeline_image_label.setPixmap(scaled_pixmap)
    #     else:
    #         self.timeline_image_label.setText("Plot file not found")

    def display_timeline_plot(self, plot_path):
        """Display the regime timeline plot - Updated for interactive plots"""
        # This method is no longer needed since we use interactive plots
        # Just update the plot instead
        self.update_timeline_plot()

    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)

    def update_status(self, message):
        """Update status message"""
        self.status_label.setText(message)

    def handle_error(self, error_message):
        """Handle errors from detection thread"""
        self.progress_bar.setVisible(False)
        self.update_status(f"Error: {error_message}")
        QMessageBox.critical(self, "Detection Error", f"An error occurred:\n{error_message}")


# class MainWindow(QMainWindow):
#     """Main application window"""
#
#     def __init__(self):
#         super().__init__()
#         self.init_ui()
#
#     def init_ui(self):
#         self.setWindowTitle("Multifactor Stock Analysis System - Regime Detection")
#         self.setGeometry(100, 100, 1400, 900)
#
#         # Set modern dark theme
#         self.setStyleSheet("""
#             QMainWindow {
#                 background-color: #1a1a1a;
#             }
#             QWidget {
#                 background-color: #2b2b2b;
#                 color: #ffffff;
#                 font-family: 'Segoe UI', Arial, sans-serif;
#             }
#             QGroupBox {
#                 border: 2px solid #444;
#                 border-radius: 8px;
#                 margin-top: 1.5ex;
#                 font-weight: bold;
#                 font-size: 12px;
#                 padding-top: 10px;
#             }
#             QGroupBox::title {
#                 subcontrol-origin: margin;
#                 left: 10px;
#                 padding: 0 10px 0 10px;
#                 color: #fff;
#             }
#             QPushButton {
#                 background-color: #3c3c3c;
#                 border: 1px solid #555;
#                 padding: 8px 15px;
#                 border-radius: 5px;
#                 font-weight: bold;
#             }
#             QPushButton:hover {
#                 background-color: #4c4c4c;
#             }
#             QPushButton:pressed {
#                 background-color: #2c2c2c;
#             }
#             QPushButton:disabled {
#                 background-color: #2c2c2c;
#                 color: #777;
#             }
#             QComboBox {
#                 background-color: #3c3c3c;
#                 border: 1px solid #555;
#                 padding: 5px;
#                 border-radius: 3px;
#                 min-width: 150px;
#             }
#             QComboBox::drop-down {
#                 border: none;
#             }
#             QComboBox::down-arrow {
#                 image: none;
#                 border-left: 4px solid transparent;
#                 border-right: 4px solid transparent;
#                 border-top: 6px solid #888;
#                 margin-right: 5px;
#             }
#             QProgressBar {
#                 border: 1px solid #444;
#                 border-radius: 3px;
#                 text-align: center;
#                 background-color: #2b2b2b;
#             }
#             QProgressBar::chunk {
#                 background-color: #7c4dff;
#                 border-radius: 3px;
#             }
#             QLabel {
#                 color: #ffffff;
#             }
#             QScrollArea {
#                 border: none;
#                 background-color: #2b2b2b;
#             }
#             QScrollBar:vertical {
#                 background-color: #2b2b2b;
#                 width: 12px;
#                 border-radius: 6px;
#             }
#             QScrollBar::handle:vertical {
#                 background-color: #555;
#                 border-radius: 6px;
#                 min-height: 20px;
#             }
#             QScrollBar::handle:vertical:hover {
#                 background-color: #666;
#             }
#         """)
#
#         # Create central widget
#         self.regime_widget = RegimeDetectionWidget()
#         self.setCentralWidget(self.regime_widget)
#
#         # Create menu bar
#         self.create_menu_bar()
#
#     def create_menu_bar(self):
#         """Create application menu bar"""
#         menubar = self.menuBar()
#         menubar.setStyleSheet("""
#             QMenuBar {
#                 background-color: #2b2b2b;
#                 color: #ffffff;
#             }
#             QMenuBar::item:selected {
#                 background-color: #3c3c3c;
#             }
#             QMenu {
#                 background-color: #2b2b2b;
#                 color: #ffffff;
#                 border: 1px solid #444;
#             }
#             QMenu::item:selected {
#                 background-color: #3c3c3c;
#             }
#         """)
#
#         # File menu
#         file_menu = menubar.addMenu('File')
#
#         export_action = file_menu.addAction('Export Results')
#         export_action.setShortcut('Ctrl+E')
#         export_action.triggered.connect(self.export_results)
#
#         file_menu.addSeparator()
#
#         exit_action = file_menu.addAction('Exit')
#         exit_action.setShortcut('Ctrl+Q')
#         exit_action.triggered.connect(self.close)
#
#         # Tools menu
#         tools_menu = menubar.addMenu('Tools')
#
#         refresh_action = tools_menu.addAction('Refresh Current')
#         refresh_action.setShortcut('F5')
#         refresh_action.triggered.connect(self.regime_widget.detect_current_regime)
#
#         # Help menu
#         help_menu = menubar.addMenu('Help')
#
#         about_action = help_menu.addAction('About')
#         about_action.triggered.connect(self.show_about)
#
#     def export_results(self):
#         """Export current results to CSV"""
#         if self.regime_widget.historical_results:
#             QMessageBox.information(self, "Export",
#                                     "Results would be exported to:\n" +
#                                     str(PROJECT_ROOT / "output" / "exported_results.csv"))
#         else:
#             QMessageBox.warning(self, "Export",
#                                 "No results to export. Please run detection first.")
#
#     def show_about(self):
#         """Show about dialog"""
#         QMessageBox.about(self, "About",
#                           "Multifactor Stock Analysis System\n"
#                           "Regime Detection Module v2.0\n\n"
#                           "This system uses Hidden Markov Models (HMM) to detect and analyze "
#                           "market regimes: Steady Growth, Strong Bull, and Crisis/Bear periods.\n\n"
#                           "Features:\n"
#                           "• Current regime detection with confidence scores\n"
#                           "• Historical regime analysis with validation\n"
#                           "• Visual timeline of regime transitions\n\n"
#                           "Developed for intelligent portfolio management.")

# Modified MainWindow class to use tabs
class MainWindow(QMainWindow):
    """Main application window with tabs"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Multifactor Stock Analysis System (MSAS)")
        self.setGeometry(100, 100, 1400, 900)

        # Set modern dark theme (same as before)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QTabWidget::pane {
                border: 1px solid #444;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #3c3c3c;
                color: #ffffff;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: #7c4dff;
                color: #ffffff;
            }
            QTabBar::tab:hover {
                background-color: #4c4c4c;
            }
            QGroupBox {
                border: 2px solid #444;
                border-radius: 8px;
                margin-top: 1.5ex;
                font-weight: bold;
                font-size: 12px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
                color: #fff;
            }
            QPushButton {
                background-color: #3c3c3c;
                border: 1px solid #555;
                padding: 8px 15px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4c4c4c;
            }
            QPushButton:pressed {
                background-color: #2c2c2c;
            }
            QPushButton:disabled {
                background-color: #2c2c2c;
                color: #777;
            }
            QComboBox {
                background-color: #3c3c3c;
                border: 1px solid #555;
                padding: 5px;
                border-radius: 3px;
                min-width: 150px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #888;
                margin-right: 5px;
            }
            QProgressBar {
                border: 1px solid #444;
                border-radius: 3px;
                text-align: center;
                background-color: #2b2b2b;
            }
            QProgressBar::chunk {
                background-color: #7c4dff;
                border-radius: 3px;
            }
            QLabel {
                color: #ffffff;
            }
            QScrollArea {
                border: none;
                background-color: #2b2b2b;
            }
            QScrollBar:vertical {
                background-color: #2b2b2b;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #555;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #666;
            }
        """)

        # Create tab widget as central widget
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # Add tabs
        self.regime_widget = RegimeDetectionWidget()
        self.scoring_widget = MultifactorScoringWidget()

        self.tab_widget.addTab(self.regime_widget, "Regime Detection")
        self.tab_widget.addTab(self.scoring_widget, "Multifactor Scoring")

        # Set the Multifactor Scoring tab as default for now
        self.tab_widget.setCurrentIndex(1)

        # Create menu bar
        self.create_menu_bar()

    def create_menu_bar(self):
        """Create application menu bar"""
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QMenuBar::item:selected {
                background-color: #3c3c3c;
            }
            QMenu {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #444;
            }
            QMenu::item:selected {
                background-color: #3c3c3c;
            }
        """)

        # File menu
        file_menu = menubar.addMenu('File')

        export_action = file_menu.addAction('Export Results')
        export_action.setShortcut('Ctrl+E')
        export_action.triggered.connect(self.export_results)

        file_menu.addSeparator()

        exit_action = file_menu.addAction('Exit')
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)

        # Tools menu
        tools_menu = menubar.addMenu('Tools')

        refresh_regime_action = tools_menu.addAction('Refresh Regime Detection')
        refresh_regime_action.setShortcut('F5')
        refresh_regime_action.triggered.connect(lambda: self.regime_widget.detect_current_regime())

        tools_menu.addSeparator()

        run_scoring_action = tools_menu.addAction('Run Scoring Process')
        run_scoring_action.setShortcut('Ctrl+R')
        run_scoring_action.triggered.connect(lambda: self.scoring_widget.run_scoring_process())

        # View menu
        view_menu = menubar.addMenu('View')

        regime_tab_action = view_menu.addAction('Regime Detection Tab')
        regime_tab_action.setShortcut('Ctrl+1')
        regime_tab_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(0))

        scoring_tab_action = view_menu.addAction('Multifactor Scoring Tab')
        scoring_tab_action.setShortcut('Ctrl+2')
        scoring_tab_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(1))

        # Help menu
        help_menu = menubar.addMenu('Help')

        about_action = help_menu.addAction('About')
        about_action.triggered.connect(self.show_about)

    def export_results(self):
        """Export current results to CSV"""
        current_tab = self.tab_widget.currentIndex()
        if current_tab == 0:  # Regime Detection tab
            if hasattr(self.regime_widget, 'historical_results') and self.regime_widget.historical_results:
                QMessageBox.information(self, "Export",
                                        "Regime results would be exported to:\n" +
                                        str(PROJECT_ROOT / "output" / "regime_results.csv"))
            else:
                QMessageBox.warning(self, "Export",
                                    "No regime results to export. Please run detection first.")
        elif current_tab == 1:  # Multifactor Scoring tab
            if hasattr(self.scoring_widget, 'current_results') and self.scoring_widget.current_results:
                QMessageBox.information(self, "Export",
                                        "Scoring results would be exported to:\n" +
                                        str(PROJECT_ROOT / "output" / "scoring_results.csv"))
            else:
                QMessageBox.warning(self, "Export",
                                    "No scoring results to export. Please run scoring first.")

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About",
                          "Multifactor Stock Analysis System (MSAS)\n"
                          "Version 2.0\n\n"
                          "This comprehensive system provides:\n"
                          "• Market regime detection using Hidden Markov Models\n"
                          "• Advanced multifactor stock scoring and ranking\n"
                          "• Intelligent portfolio optimization\n\n"
                          "Tabs:\n"
                          "• Regime Detection: Analyze current market conditions\n"
                          "• Multifactor Scoring: Score and rank stocks using multiple factors\n\n"
                          "Developed for intelligent investment analysis.")

def main():
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    # Set application icon if available
    icon_path = PROJECT_ROOT / 'assets' / 'icon.png'
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    # Create and show main window
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()