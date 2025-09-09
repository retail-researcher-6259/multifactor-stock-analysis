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
                             QAbstractItemView, QLineEdit, QScrollArea, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QAbstractTableModel
from PyQt5.QtGui import QFont, QPixmap, QPalette, QColor
import pyqtgraph as pg
from pyqtgraph import DateAxisItem
import matplotlib.cm as cm
import time
import re

# Get project root directory (Current directory)
PROJECT_ROOT = Path(file).parent if 'file' in globals() else Path.cwd()

class MultifactorScoringThread(QThread):
    """Enhanced thread for running multifactor scoring with detailed progress tracking"""
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, regime_name):
        super().__init__()
        self.regime_name = regime_name
        self.current_progress = 0

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
            scoring_script_path = PROJECT_ROOT / 'src' / 'scoring' / 'stock_Screener_MultiFactor_24.py'

            if not scoring_script_path.exists():
                self.error_occurred.emit(f"Scoring script not found: {scoring_script_path}")
                return

            # Add the scoring directory to Python path
            sys.path.insert(0, str(scoring_script_path.parent))

            try:
                # Import the scoring module
                import stock_Screener_MultiFactor_24 as scorer

                self.status_update.emit("Starting multifactor analysis...")
                self.progress_update.emit(2)

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

    def get_regime_weights_dict(self):
        """Define regime-specific weights"""
        return {
            "Steady Growth": {
                "credit": 48.6,
                "quality": 18.1,
                "momentum": 18.1,
                "financial_health": 14.4,
                "growth": 10.9,
                "value": -7.3,
                "technical": 3.8,
                "liquidity": 3.6,
                "carry": -3.6,
                "stability": -3.0,
                "size": -2.2,
                "insider": -1.4,
            },
            "Strong Bull": {
                "credit": 40.0,
                "quality": 20.0,
                "momentum": 25.0,
                "financial_health": 15.0,
                "growth": 15.0,
                "value": -5.0,
                "technical": 8.0,
                "liquidity": 5.0,
                "carry": -2.0,
                "stability": -5.0,
                "size": 0.0,
                "insider": 2.0,
            },
            "Crisis/Bear": {
                "momentum": 72.0,      # Slight increase from 70.7
                "size": 20.0,          # Rounded up from 17.6
                "financial_health": 13.0,  # Reduced to minimize drag
                "credit": 0.0,         # Eliminated (negative correlation)
                "insider": 5.0,        # Rounded up from 4.4
                "growth": 5.0,         # Reduced from 8.8 (negative correlation)
                "quality": 0.0,        # Keep at zero
                "liquidity": 5.0,      # Add for positive correlation
                "value": 0.0,          # Keep at zero
                "technical": -5.0,     # Unchanged
                "carry": -5.0,         # Rounded from -4.4
                "stability": -10.0,    # Rounded from -10.3
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
        """Create control panel with button and regime selector"""
        group = QGroupBox("Scoring Controls")
        layout = QVBoxLayout()

        # Regime Selection
        regime_layout = QHBoxLayout()
        regime_layout.addWidget(QLabel("Target Regime:"))

        self.regime_combo = QComboBox()
        self.regime_combo.addItems(["Steady Growth", "Strong Bull", "Crisis/Bear"])
        self.regime_combo.setCurrentText("Steady Growth")
        self.regime_combo.setToolTip("Select the market regime for optimized factor weights")
        self.regime_combo.currentTextChanged.connect(self.on_regime_changed)
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

        # Process Information
        info_label = QLabel("Process Information")
        info_label.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(info_label)

        self.process_info = QLabel(
            f"• Data Source: {PROJECT_ROOT / 'config' / 'Buyable_stock.txt'}\n"
            f"• Output Directory: ./output/Ranked_Lists/[Regime]/\n"
            f"• Analysis: Multifactor scoring with regime-specific weights"
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
        selected_regime = self.regime_combo.currentText()

        # Convert regime name to match file naming convention
        regime_key = selected_regime.replace(" ", "_").replace("/", "_")

        # Record start time for ETA calculation
        import time
        self.start_time = time.time()

        # Show progress bar
        self.progress_group.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_status.setText(f"Initializing {selected_regime} regime analysis...")

        # Reset statistics
        self.stats_labels['total_stocks'].setText("Total Stocks to Analyze: --")
        self.stats_labels['top_performers'].setText("Currently Processing: --")
        self.stats_labels['avg_score'].setText("Average Score: --")
        self.stats_labels['processing_time'].setText("Processing Time: --")

        # Disable button during processing
        self.run_scoring_btn.setEnabled(False)
        self.run_scoring_btn.setText("Processing...")

        # Update info label
        self.results_info_label.setText(f"Running analysis for {selected_regime} regime...")

        # Start the enhanced scoring thread
        self.scoring_thread = MultifactorScoringThread(regime_key)
        self.scoring_thread.progress_update.connect(self.update_progress)
        self.scoring_thread.status_update.connect(self.update_status_with_details)  # Use enhanced version
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

        # Add placeholder note for regimes with placeholder weights
        if regime_name in ["Strong Bull", "Crisis/Bear"]:
            placeholder_note = QLabel("⚠️ These are placeholder weights. Optimization pending.")
            placeholder_note.setStyleSheet("""
                QLabel {
                    color: #ff9800;
                    font-style: italic;
                    font-size: 11px;
                    padding: 5px;
                    background-color: rgba(255, 152, 0, 0.1);
                    border-radius: 3px;
                    margin-top: 10px;
                }
            """)
            self.weights_layout.addWidget(placeholder_note)

    def on_regime_changed(self, regime_name):
        """Handle regime selection change"""
        self.update_weights_display(regime_name)

    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)

    def update_status(self, message):
        """Update status message"""
        self.progress_status.setText(message)

    def handle_results(self, results):
        """Handle successful scoring results with enhanced statistics"""
        self.current_results = results
        self.display_results(results['results_df'])

        # Calculate final statistics
        df = results['results_df']
        total_stocks = results['total_stocks']
        regime = results['regime']

        # Update final statistics
        self.stats_labels['total_stocks'].setText(f"Total Stocks Analyzed: {total_stocks}")

        if not df.empty:
            top_10_percent = int(len(df) * 0.1)
            avg_score = df['Score'].mean()

            self.stats_labels['top_performers'].setText(f"Top 10% Count: {top_10_percent}")
            self.stats_labels['avg_score'].setText(f"Average Score: {avg_score:.2f}")

        # Calculate total processing time
        if hasattr(self, 'start_time'):
            total_time = time.time() - self.start_time
            if total_time > 60:
                time_text = f"Total Time: {total_time / 60:.1f} min"
            else:
                time_text = f"Total Time: {total_time:.0f} sec"
            self.stats_labels['processing_time'].setText(time_text)

        # Update info label
        self.results_info_label.setText(f"{total_stocks} stocks analyzed for {regime} regime")

        # Enable action buttons
        # self.export_btn.setEnabled(True)
        # self.refresh_btn.setEnabled(True)

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
        """Enhanced status update with more detailed information"""
        self.progress_status.setText(message)

        # Parse ticker information from status message for additional display
        if "Processing" in message and "(" in message:
            # Extract current/total from messages like "Processing XOM (1/10)..."
            try:
                match = re.search(r'Processing \w+ \((\d+)/(\d+)\)', message)
                if match:
                    current, total = match.groups()

                    # Update statistics in real-time
                    self.stats_labels['total_stocks'].setText(f"Total Stocks to Analyze: {total}")
                    self.stats_labels['top_performers'].setText(f"Currently Processing: {current}/{total}")

                    # Calculate estimated time remaining
                    if hasattr(self, 'start_time') and int(current) > 0:
                        elapsed = time.time() - self.start_time
                        avg_time_per_stock = elapsed / int(current)
                        remaining_stocks = int(total) - int(current)
                        estimated_remaining = remaining_stocks * avg_time_per_stock

                        if estimated_remaining > 60:
                            eta_text = f"ETA: ~{estimated_remaining / 60:.1f} min"
                        else:
                            eta_text = f"ETA: ~{estimated_remaining:.0f} sec"

                        self.stats_labels['processing_time'].setText(eta_text)
            except:
                pass  # Ignore parsing errors


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

    # ADD THIS SIGNAL to the class
    regime_detected = pyqtSignal(str)  # Signal to emit when regime is detected

    def __init__(self):
        super().__init__()
        self.detector_thread = None
        self.current_results = None
        self.historical_results = None
        self.current_regime = "Steady Growth"  # Default regime
        self.init_ui()
        # Load previous regime result at startup
        self.load_previous_regime()

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

    def load_previous_regime(self):
        """Load the previous regime detection result at startup"""
        try:
            regime_file = PROJECT_ROOT / 'output' / 'Regime_Detection_Analysis' / 'current_regime_analysis.json'

            if regime_file.exists():
                with open(regime_file, 'r') as f:
                    data = json.load(f)

                # Extract regime information
                regime_info = data.get('regime_detection', {})
                detected_regime = regime_info.get('regime_name', 'Steady Growth')

                # Update current regime
                self.current_regime = detected_regime

                # Update UI display
                self.current_regime_label.setText(f"Current Regime: {detected_regime}")

                # Set color coding
                if 'Bull' in detected_regime:
                    color = '#4CAF50'  # Green
                elif 'Growth' in detected_regime:
                    color = '#2196F3'  # Blue
                elif 'Bear' in detected_regime or 'Crisis' in detected_regime:
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

                # Update probabilities if available
                all_probs = regime_info.get('all_probabilities', {})
                if all_probs:
                    self.prob_labels["Steady Growth"].setText(f"{all_probs.get('Steady Growth', 0):.2%}")
                    self.prob_labels["Strong Bull"].setText(f"{all_probs.get('Strong Bull', 0):.2%}")
                    self.prob_labels["Crisis/Bear"].setText(f"{all_probs.get('Crisis/Bear', 0):.2%}")

                # Emit signal for other widgets
                self.regime_detected.emit(detected_regime)

                print(f"✅ Loaded previous regime: {detected_regime}")

            else:
                print("ℹ️ No previous regime analysis found, using default")

        except Exception as e:
            print(f"⚠️ Error loading previous regime: {e}")
            # Use default regime
            self.current_regime = "Steady Growth"

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
        self.current_regime = current_regime  # Store current regime
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
        all_probs = regime_info.get('all_probabilities', {})
        self.prob_labels["Steady Growth"].setText(f"{all_probs.get('Steady Growth', 0):.2%}")
        self.prob_labels["Strong Bull"].setText(f"{all_probs.get('Strong Bull', 0):.2%}")
        self.prob_labels["Crisis/Bear"].setText(f"{all_probs.get('Crisis/Bear', 0):.2%}")

        self.update_status("Current regime detected successfully")

        # Emit signal for other widgets to update
        self.regime_detected.emit(current_regime)


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


class StabilityAnalysisThread(QThread):
    """Thread for running stability analysis without blocking UI"""
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, regime_name):
        super().__init__()
        self.regime_name = regime_name

    def run(self):
        try:
            self.status_update.emit("Starting stability analysis...")
            self.progress_update.emit(10)

            # Import and run the stability analysis script
            script_path = PROJECT_ROOT / 'src' / 'trend_analysis' / 'stock_score_trend_analyzer_04.py'

            if not script_path.exists():
                self.error_occurred.emit(f"Script not found: {script_path}")
                return

            self.progress_update.emit(30)
            self.status_update.emit("Loading ranking files...")

            # Add script directory to path
            sys.path.insert(0, str(script_path.parent))

            try:
                import stock_score_trend_analyzer_04 as analyzer_module

                self.progress_update.emit(50)
                self.status_update.emit("Running stability analysis...")

                # Create analyzer instance
                csv_directory = PROJECT_ROOT / "output" / "Ranked_Lists" / self.regime_name
                output_directory = PROJECT_ROOT / "output" / "Score_Trend_Analysis_Results" / self.regime_name
                output_directory.mkdir(parents=True, exist_ok=True)

                analyzer = analyzer_module.StockScoreTrendAnalyzer(
                    csv_directory=str(csv_directory),
                    start_date="0601",  # Configurable
                    end_date=datetime.now().strftime("%m%d"),
                    sigmoid_sensitivity=5
                )

                self.progress_update.emit(70)
                self.status_update.emit("Analyzing stock trends...")

                # Run analysis and export results
                today = datetime.now().strftime("%m%d")
                output_file = output_directory / f"stability_analysis_results_{today}.csv"
                results_df = analyzer.export_results(output_path=str(output_file))

                self.progress_update.emit(90)
                self.status_update.emit("Finalizing results...")

                # Prepare result data
                result = {
                    'success': True,
                    'output_path': str(output_file),
                    'results_df': results_df,
                    'regime': self.regime_name,
                    'total_stocks': len(results_df),
                    'analyzer': analyzer
                }

                self.progress_update.emit(100)
                self.result_ready.emit(result)

            except Exception as e:
                self.error_occurred.emit(f"Analysis failed: {str(e)}")
            finally:
                if str(script_path.parent) in sys.path:
                    sys.path.remove(str(script_path.parent))

        except Exception as e:
            self.error_occurred.emit(f"Thread execution failed: {str(e)}")


class TechnicalPlotsThread(QThread):
    """Thread for generating technical plots for a specific ticker"""
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, ticker, regime_name):
        super().__init__()
        self.ticker = ticker.upper()
        self.regime_name = regime_name

    def run(self):
        try:
            self.status_update.emit(f"Initializing technical analysis for {self.ticker}...")
            self.progress_update.emit(10)

            # Import technical analysis script
            script_path = PROJECT_ROOT / 'src' / 'trend_analysis' / 'stock_score_trend_technical_single.py'

            if not script_path.exists():
                self.error_occurred.emit(f"Technical script not found: {script_path}")
                return

            self.progress_update.emit(30)
            self.status_update.emit("Loading historical data...")

            sys.path.insert(0, str(script_path.parent))

            try:
                import stock_score_trend_technical_single as tech_module

                self.progress_update.emit(50)
                self.status_update.emit("Running technical analysis...")

                # Set up directories
                csv_directory = PROJECT_ROOT / "output" / "Ranked_Lists" / self.regime_name
                output_base_dir = PROJECT_ROOT / "output" / "Score_Trend_Analysis_Results"
                tech_plots_dir = output_base_dir / "technical_plots" / self.ticker
                tech_plots_dir.mkdir(parents=True, exist_ok=True)

                # Create analyzer instance
                analyzer = tech_module.SingleTickerTechnicalAnalyzer(
                    csv_directory=str(csv_directory),
                    start_date="0601",
                    end_date=datetime.now().strftime("%m%d"),
                    sigmoid_sensitivity=5
                )

                self.progress_update.emit(70)
                self.status_update.emit("Generating technical plots...")

                # Run analysis
                success = analyzer.analyze_single_ticker(
                    ticker=self.ticker,
                    output_base_dir=str(output_base_dir),
                    min_appearances=3
                )

                if not success:
                    self.error_occurred.emit(f"Technical analysis failed for {self.ticker}")
                    return

                self.progress_update.emit(90)
                self.status_update.emit("Generating recommendation...")

                # Get technical score and recommendation
                recommendation = analyzer.generate_recommendation(self.ticker)

                # Find plot files
                plot_files = list(tech_plots_dir.glob("*.png"))

                result = {
                    'success': True,
                    'ticker': self.ticker,
                    'regime': self.regime_name,
                    'recommendation': recommendation,
                    'plot_directory': str(tech_plots_dir),
                    'plot_files': [str(p) for p in plot_files],
                    'analyzer': analyzer
                }

                self.progress_update.emit(100)
                self.result_ready.emit(result)

            except Exception as e:
                self.error_occurred.emit(f"Technical analysis failed: {str(e)}")
            finally:
                if str(script_path.parent) in sys.path:
                    sys.path.remove(str(script_path.parent))

        except Exception as e:
            self.error_occurred.emit(f"Thread execution failed: {str(e)}")


class ScoreTrendAnalysisWidget(QWidget):
    """Main widget for score trend analysis system"""

    def __init__(self):
        super().__init__()
        self.stability_thread = None
        self.technical_thread = None
        self.current_stability_results = None
        self.current_technical_results = None
        self.recommended_regime = "Steady Growth"  # Default
        self.init_ui()

    def init_ui(self):
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)

        # Title
        title_label = QLabel("Score Trend Analysis System")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont("Arial", 18, QFont.Bold)
        title_label.setFont(title_font)
        main_layout.addWidget(title_label)

        # Subtitle
        subtitle_label = QLabel("Stability analysis and technical plotting for stock score trends")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("color: #888;")
        main_layout.addWidget(subtitle_label)

        # Create two main sections
        sections_layout = QHBoxLayout()
        sections_layout.setSpacing(15)

        # Left section: Stability Analysis
        stability_section = self.create_stability_section()
        sections_layout.addWidget(stability_section, 1)

        # Right section: Technical Analysis
        technical_section = self.create_technical_section()
        sections_layout.addWidget(technical_section, 1)

        main_layout.addLayout(sections_layout)

        # Results display area
        results_section = self.create_results_section()
        main_layout.addWidget(results_section)

        # Progress bars
        self.create_progress_bars()
        main_layout.addWidget(self.stability_progress_group)
        main_layout.addWidget(self.technical_progress_group)

        self.setLayout(main_layout)

    def create_stability_section(self):
        """Create stability analysis section"""
        group = QGroupBox("Stability Analysis")
        layout = QVBoxLayout()

        # Regime selection with recommendation
        regime_layout = QHBoxLayout()
        regime_layout.addWidget(QLabel("Target Regime:"))

        self.stability_regime_combo = QComboBox()
        self.stability_regime_combo.addItems(["Steady Growth", "Strong Bull", "Crisis/Bear"])
        self.stability_regime_combo.setCurrentText(self.recommended_regime)
        regime_layout.addWidget(self.stability_regime_combo)

        # Recommendation indicator
        self.regime_recommendation_label = QLabel("(Recommended)")
        self.regime_recommendation_label.setStyleSheet("color: #4CAF50; font-style: italic;")
        regime_layout.addWidget(self.regime_recommendation_label)

        regime_layout.addStretch()
        layout.addLayout(regime_layout)

        # Run Analysis Button
        self.run_stability_btn = QPushButton("Run Stability Analysis")
        self.run_stability_btn.clicked.connect(self.run_stability_analysis)
        self.run_stability_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                font-size: 13px;
                padding: 12px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        layout.addWidget(self.run_stability_btn)

        # Analysis Info
        analysis_info = QLabel(
            "• Analyzes all ranked lists in selected regime folder\n"
            "• Generates stability metrics and trend analysis\n"
            "• Outputs results to Score_Trend_Analysis_Results/[Regime]/"
        )
        analysis_info.setStyleSheet("""
            QLabel {
                background-color: #3c3c3c;
                padding: 10px;
                border-radius: 5px;
                color: #cccccc;
                font-size: 11px;
            }
        """)
        analysis_info.setWordWrap(True)
        layout.addWidget(analysis_info)

        # Stability Statistics
        stats_label = QLabel("Analysis Statistics")
        stats_label.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(stats_label)

        self.stability_stats = {
            'total_analyzed': QLabel("Total Stocks Analyzed: --"),
            'strong_buy': QLabel("Strong Buy Recommendations: --"),
            'avg_stability': QLabel("Average Stability Score: --")
        }

        for label in self.stability_stats.values():
            label.setStyleSheet("color: #cccccc; font-size: 12px;")
            layout.addWidget(label)

        layout.addStretch()
        group.setLayout(layout)
        return group

    def create_technical_section(self):
        """Create technical analysis section"""
        group = QGroupBox("Technical Plot Analysis")
        layout = QVBoxLayout()

        # Ticker input
        ticker_layout = QHBoxLayout()
        ticker_layout.addWidget(QLabel("Ticker Symbol:"))

        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("Enter ticker (e.g., AAPL)")
        self.ticker_input.setStyleSheet("""
            QLineEdit {
                background-color: #3c3c3c;
                border: 1px solid #555;
                padding: 8px;
                border-radius: 4px;
                font-size: 13px;
            }
        """)
        self.ticker_input.returnPressed.connect(self.run_technical_analysis)
        ticker_layout.addWidget(self.ticker_input)

        layout.addLayout(ticker_layout)

        # Generate Plots Button
        self.run_technical_btn = QPushButton("Generate Technical Plots")
        self.run_technical_btn.clicked.connect(self.run_technical_analysis)
        self.run_technical_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-weight: bold;
                font-size: 13px;
                padding: 12px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        layout.addWidget(self.run_technical_btn)

        # Technical Results Display
        results_label = QLabel("Technical Analysis Results")
        results_label.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(results_label)

        self.technical_results_area = QTextEdit()
        self.technical_results_area.setMaximumHeight(150)
        self.technical_results_area.setStyleSheet("""
            QTextEdit {
                background-color: #3c3c3c;
                border: 1px solid #555;
                border-radius: 5px;
                color: #ffffff;
                font-family: 'Courier New', monospace;
                font-size: 11px;
            }
        """)
        self.technical_results_area.setPlaceholderText("Technical analysis results will appear here...")
        layout.addWidget(self.technical_results_area)

        # Plot Files Display
        plots_label = QLabel("Generated Plots")
        plots_label.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(plots_label)

        self.plots_list_widget = QTextEdit()
        self.plots_list_widget.setMaximumHeight(100)
        self.plots_list_widget.setStyleSheet("""
            QTextEdit {
                background-color: #2b2b2b;
                border: 1px solid #444;
                border-radius: 5px;
                color: #cccccc;
                font-size: 11px;
            }
        """)
        self.plots_list_widget.setPlaceholderText("Plot files will be listed here...")
        layout.addWidget(self.plots_list_widget)

        layout.addStretch()
        group.setLayout(layout)
        return group

    def create_results_section(self):
        """Create unified results display section with proper horizontal scroll"""
        group = QGroupBox("Analysis Results")
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 30)  # Add bottom margin to prevent taskbar overlap

        # Tab-like buttons for switching views
        view_buttons_layout = QHBoxLayout()
        view_buttons_layout.setSpacing(5)

        self.stability_view_btn = QPushButton("Stability Results")
        self.stability_view_btn.setCheckable(True)
        self.stability_view_btn.setChecked(True)
        self.stability_view_btn.clicked.connect(lambda: self.switch_results_view("stability"))
        view_buttons_layout.addWidget(self.stability_view_btn)

        # Technical plots button with dropdown
        technical_button_widget = QWidget()
        technical_button_layout = QHBoxLayout(technical_button_widget)
        technical_button_layout.setContentsMargins(0, 0, 0, 0)
        technical_button_layout.setSpacing(0)

        self.technical_view_btn = QPushButton("Technical Plots")
        self.technical_view_btn.setCheckable(True)
        self.technical_view_btn.clicked.connect(lambda: self.switch_results_view("technical"))
        technical_button_layout.addWidget(self.technical_view_btn)

        # Add dropdown for plot selection
        self.plot_selector = QComboBox()
        self.plot_selector.setMinimumWidth(200)
        self.plot_selector.setMaximumWidth(300)
        self.plot_selector.addItem("All Plots")
        self.plot_selector.currentTextChanged.connect(self.on_plot_selection_changed)
        self.plot_selector.setVisible(False)  # Initially hidden
        self.plot_selector.setStyleSheet("""
            QComboBox {
                background-color: #3c3c3c;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px;
                color: white;
                margin-left: 10px;
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
            QComboBox QAbstractItemView {
                background-color: #2b2b2b;
                border: 1px solid #555;
                selection-background-color: #7c4dff;
                color: white;
            }
        """)
        technical_button_layout.addWidget(self.plot_selector)

        view_buttons_layout.addWidget(technical_button_widget)
        view_buttons_layout.addStretch()

        # Style the view buttons
        button_style = """
            QPushButton {
                background-color: #3c3c3c;
                color: white;
                padding: 8px 15px;
                border-radius: 4px;
                border: none;
            }
            QPushButton:checked {
                background-color: #7c4dff;
            }
            QPushButton:hover {
                background-color: #4c4c4c;
            }
        """
        self.stability_view_btn.setStyleSheet(button_style)
        self.technical_view_btn.setStyleSheet(button_style)

        # ADD THIS LINE - IT WAS COMMENTED OUT!
        layout.addLayout(view_buttons_layout)

        # Results display area
        self.results_stack = QWidget()
        results_stack_layout = QVBoxLayout(self.results_stack)

        # ENHANCED: Stability results table with proper horizontal scroll
        self.stability_table = QTableWidget()
        self.stability_table.setMinimumHeight(350)
        self.stability_table.setAlternatingRowColors(True)
        self.stability_table.setSelectionBehavior(QAbstractItemView.SelectRows)

        # CRITICAL: Proper horizontal scroll configuration
        self.stability_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.stability_table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.stability_table.horizontalHeader().setStretchLastSection(False)

        # IMPORTANT: Hide row numbers (index column) to save space
        self.stability_table.verticalHeader().setVisible(False)

        # Configure resize modes for better scrolling
        header = self.stability_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)  # Allow manual resizing
        header.setDefaultSectionSize(100)  # Default column width

        # Enhanced table styling with proper scroll bars
        self.stability_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #444;
                background-color: #2b2b2b;
                alternate-background-color: #353535;
                selection-background-color: #7c4dff;
            }
            QHeaderView::section {
                background-color: #404040;
                padding: 8px;
                border: 1px solid #555;
                font-weight: bold;
                color: #ffffff;
                min-height: 25px;
            }
            QTableWidget::item {
                padding: 6px;
                border-bottom: 1px solid #444;
                color: #ffffff;
            }
            QTableWidget::item:selected {
                background-color: #7c4dff;
                color: #ffffff;
            }
            QScrollBar:horizontal {
                background-color: #3c3c3c;
                height: 15px;
                border-radius: 7px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background-color: #7c4dff;
                border-radius: 7px;
                min-width: 30px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #8c5fff;
            }
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {
                border: none;
                background: none;
            }
            QScrollBar:vertical {
                background-color: #3c3c3c;
                width: 15px;
                border-radius: 7px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #7c4dff;
                border-radius: 7px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #8c5fff;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
        """)

        # Technical plots display area
        self.technical_plots_area = QScrollArea()
        self.technical_plots_area.setWidgetResizable(True)
        self.technical_plots_area.setMinimumHeight(350)
        self.technical_plots_area.setStyleSheet("""
            QScrollArea {
                border: 1px solid #444;
                border-radius: 5px;
                background-color: #2b2b2b;
            }
        """)

        # Initially show stability table
        results_stack_layout.addWidget(self.stability_table)
        results_stack_layout.addWidget(self.technical_plots_area)
        self.technical_plots_area.hide()

        layout.addWidget(self.results_stack)

        # SIMPLIFIED: Action area with just the summary label (NO BUTTONS)
        actions_layout = QHBoxLayout()
        actions_layout.addStretch()

        self.results_summary_label = QLabel("No analysis results available")
        self.results_summary_label.setStyleSheet("color: #888; font-style: italic;")
        actions_layout.addWidget(self.results_summary_label)

        layout.addLayout(actions_layout)

        # layout.setContentsMargins(10, 10, 10, 30)  # Add 30px bottom margin

        group.setLayout(layout)
        return group

    def populate_plot_selector(self):
        """Populate the plot selector dropdown with available plots"""
        self.plot_selector.clear()
        self.plot_selector.addItem("All Plots")

        if hasattr(self, 'filtered_plot_files') and self.filtered_plot_files:
            # Extract ticker name from first file
            first_file = Path(self.filtered_plot_files[0]).name
            ticker = first_file.split('_')[0]

            # Add individual plot options
            plot_names = []
            for plot_file in self.filtered_plot_files:
                filename = Path(plot_file).stem
                # Format the display name nicely
                parts = filename.replace(ticker + '_', '').split('_')

                # Create readable name
                if len(parts) >= 2:
                    plot_number = parts[0]
                    plot_name = ' '.join(parts[1:])
                    display_name = f"{ticker} {plot_number} {plot_name.title()}"
                else:
                    display_name = filename.replace('_', ' ').title()

                plot_names.append((display_name, plot_file))

            # Sort by plot number
            plot_names.sort(key=lambda x: x[0])

            for display_name, plot_file in plot_names:
                self.plot_selector.addItem(display_name, plot_file)

    def on_plot_selection_changed(self, selected_text):
        """Handle plot selection from dropdown"""
        if selected_text == "All Plots":
            self.load_technical_plots()
        else:
            # Load single plot
            current_index = self.plot_selector.currentIndex()
            if current_index > 0:  # Skip "All Plots"
                plot_file = self.plot_selector.itemData(current_index)
                if plot_file:
                    self.load_single_plot(plot_file)

    def load_single_plot(self, plot_file):
        """Load and display a single plot"""
        if not Path(plot_file).exists():
            return

        # Create a widget to hold the single plot
        plot_widget = QWidget()
        layout = QVBoxLayout(plot_widget)
        layout.setAlignment(Qt.AlignCenter)

        # Create a frame for the plot
        frame = QFrame()
        frame.setFrameStyle(QFrame.Box)
        frame.setStyleSheet("QFrame { border: 2px solid #555; background-color: #2b2b2b; }")
        frame_layout = QVBoxLayout(frame)

        # Load and display the image at full resolution
        label = QLabel()
        pixmap = QPixmap(plot_file)

        # Scale to fit the available space while maintaining aspect ratio
        # Use larger max dimensions for single plot view
        scaled_pixmap = pixmap.scaled(1200, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)
        label.setAlignment(Qt.AlignCenter)

        # Add filename as title
        title = QLabel(Path(plot_file).stem.replace('_', ' ').title())
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 10px; color: #ffffff;")

        frame_layout.addWidget(title)
        frame_layout.addWidget(label)

        layout.addWidget(frame)

        # Set the widget in the scroll area
        self.technical_plots_area.setWidget(plot_widget)

    def create_progress_bars(self):
        """Create progress bars for both analysis types"""
        # Stability analysis progress
        self.stability_progress_group = QGroupBox("Stability Analysis Progress")
        self.stability_progress_group.setVisible(False)
        stability_layout = QVBoxLayout()

        self.stability_progress_bar = QProgressBar()
        self.stability_progress_bar.setMinimum(0)
        self.stability_progress_bar.setMaximum(100)
        stability_layout.addWidget(self.stability_progress_bar)

        self.stability_progress_status = QLabel("Ready...")
        self.stability_progress_status.setAlignment(Qt.AlignCenter)
        stability_layout.addWidget(self.stability_progress_status)

        self.stability_progress_group.setLayout(stability_layout)

        # Technical analysis progress
        self.technical_progress_group = QGroupBox("Technical Analysis Progress")
        self.technical_progress_group.setVisible(False)
        technical_layout = QVBoxLayout()

        self.technical_progress_bar = QProgressBar()
        self.technical_progress_bar.setMinimum(0)
        self.technical_progress_bar.setMaximum(100)
        technical_layout.addWidget(self.technical_progress_bar)

        self.technical_progress_status = QLabel("Ready...")
        self.technical_progress_status.setAlignment(Qt.AlignCenter)
        technical_layout.addWidget(self.technical_progress_status)

        self.technical_progress_group.setLayout(technical_layout)

    def switch_results_view(self, view_type):
        """Switch between stability and technical results views"""
        if view_type == "stability":
            self.stability_view_btn.setChecked(True)
            self.technical_view_btn.setChecked(False)
            self.stability_table.show()
            self.technical_plots_area.hide()
            self.plot_selector.setVisible(False)  # Hide dropdown
        else:  # technical
            self.stability_view_btn.setChecked(False)
            self.technical_view_btn.setChecked(True)
            self.stability_table.hide()
            self.technical_plots_area.show()
            self.plot_selector.setVisible(True)  # Show dropdown
            # Load plots when switching to technical view
            self.load_technical_plots()
            # Populate the dropdown
            self.populate_plot_selector()

    def set_recommended_regime(self, regime):
        """Set the recommended regime from regime detection tab"""
        print(f"📊 Updating recommended regime to: {regime}")

        self.recommended_regime = regime

        # Map regime names properly
        regime_mapping = {
            "Steady Growth": "Steady Growth",
            "Strong Bull": "Strong Bull",
            "Crisis/Bear": "Crisis/Bear",
            "Crisis": "Crisis/Bear",  # Handle variations
            "Bear": "Crisis/Bear"
        }

        mapped_regime = regime_mapping.get(regime, regime)

        # Update combo box if the regime is in our list
        combo_items = [self.stability_regime_combo.itemText(i) for i in range(self.stability_regime_combo.count())]
        if mapped_regime in combo_items:
            self.stability_regime_combo.setCurrentText(mapped_regime)
            # self.regime_recommendation_label.setText("(Recommended from Regime Detection)")
            self.regime_recommendation_label.setText("")
            self.regime_recommendation_label.setStyleSheet("color: #4CAF50; font-style: italic; font-weight: bold;")
        else:
            self.regime_recommendation_label.setText("(Regime not in analysis list)")
            self.regime_recommendation_label.setStyleSheet("color: #ff9800; font-style: italic;")

    def run_stability_analysis(self):
        """Run enhanced stability analysis for selected regime"""
        selected_regime = self.stability_regime_combo.currentText()
        regime_key = selected_regime.replace(" ", "_").replace("/", "_")

        # Show progress
        self.stability_progress_group.setVisible(True)
        self.stability_progress_bar.setValue(0)
        self.stability_progress_status.setText("Starting enhanced stability analysis...")

        # Disable button
        self.run_stability_btn.setEnabled(False)
        self.run_stability_btn.setText("Analyzing...")

        # Start enhanced thread
        self.stability_thread = EnhancedStabilityAnalysisThread(regime_key)
        self.stability_thread.progress_update.connect(self.update_stability_progress)
        self.stability_thread.status_update.connect(self.update_stability_status)
        self.stability_thread.result_ready.connect(self.handle_stability_results)
        self.stability_thread.error_occurred.connect(self.handle_stability_error)
        self.stability_thread.finished.connect(self.on_stability_finished)
        self.stability_thread.start()

    def run_technical_analysis(self):
        """Run enhanced technical analysis for entered ticker"""
        ticker = self.ticker_input.text().strip().upper()

        if not ticker:
            QMessageBox.warning(self, "Input Error", "Please enter a ticker symbol!")
            return

        selected_regime = self.stability_regime_combo.currentText()
        regime_key = selected_regime.replace(" ", "_").replace("/", "_")

        # Show progress
        self.technical_progress_group.setVisible(True)
        self.technical_progress_bar.setValue(0)
        self.technical_progress_status.setText(f"Starting enhanced technical analysis for {ticker}...")

        # Disable button
        self.run_technical_btn.setEnabled(False)
        self.run_technical_btn.setText("Generating...")

        # Start enhanced thread
        self.technical_thread = EnhancedTechnicalPlotsThread(ticker, regime_key)
        self.technical_thread.progress_update.connect(self.update_technical_progress)
        self.technical_thread.status_update.connect(self.update_technical_status)
        self.technical_thread.result_ready.connect(self.handle_technical_results)
        self.technical_thread.error_occurred.connect(self.handle_technical_error)
        self.technical_thread.finished.connect(self.on_technical_finished)
        self.technical_thread.start()

    # Progress update methods
    def update_stability_progress(self, value):
        self.stability_progress_bar.setValue(value)

    def update_stability_status(self, message):
        self.stability_progress_status.setText(message)

    def update_technical_progress(self, value):
        self.technical_progress_bar.setValue(value)

    def update_technical_status(self, message):
        self.technical_progress_status.setText(message)

    # Results handlers
    def handle_stability_results(self, results):
        """Handle stability analysis results (no export button)"""
        self.current_stability_results = results
        self.display_stability_results(results['results_df'])

        # Update statistics
        df = results['results_df']
        total_stocks = len(df)
        strong_buy_count = len(df[df['recommendation'].str.contains('STRONG BUY', na=False)])
        avg_stability = df['stability_adjusted_score'].mean()

        self.stability_stats['total_analyzed'].setText(f"Total Stocks Analyzed: {total_stocks}")
        self.stability_stats['strong_buy'].setText(f"Strong Buy Recommendations: {strong_buy_count}")
        self.stability_stats['avg_stability'].setText(f"Average Stability Score: {avg_stability:.2f}")

        # Show where results are saved
        output_path = results['output_path']
        self.results_summary_label.setText(
            f"Analysis complete: {total_stocks} stocks analyzed. Results saved to output directory."
        )

        print(f"✅ Stability results saved to: {output_path}")

    def handle_technical_results(self, results):
        """Enhanced handler for technical analysis results (no folder button)"""
        self.current_technical_results = results

        # Display enhanced technical analysis results
        ticker = results['ticker']
        recommendation = results['recommendation']
        technical_score = results.get('technical_score', 0)
        total_plots = results.get('total_plots', 0)
        ranking_data = results.get('ranking_data', {})
        plot_directory = results['plot_directory']

        results_text = f"ENHANCED TECHNICAL ANALYSIS FOR {ticker}\n"
        results_text += "=" * 50 + "\n"
        results_text += f"Technical Score: {technical_score:.1f}\n"
        results_text += f"Recommendation: {recommendation}\n"
        results_text += f"Regime: {results['regime']}\n"
        results_text += f"Plots Generated: {total_plots}\n"
        results_text += f"📁 Plots Directory: {plot_directory}\n"

        if results.get('ranking_file'):
            results_text += f"📊 Ranking CSV: {Path(results['ranking_file']).name}\n"

        # Add detailed breakdown if available
        if ranking_data:
            results_text += f"\n🔍 DETAILED INDICATOR BREAKDOWN:\n"
            results_text += "-" * 30 + "\n"

            indicators = ['Trend Direction', 'SMA Position', 'MACD Signal', 'RSI Level',
                          'ADX Strength', 'ARIMA', 'Exp Smooth']

            for indicator in indicators:
                if indicator in ranking_data:
                    value = ranking_data[indicator]
                    if value == 1:
                        status = "🟢 BULLISH"
                    elif value >= 0.5:
                        status = "🟡 NEUTRAL"
                    else:
                        status = "🔴 BEARISH"
                    results_text += f"{indicator}: {value} {status}\n"

        self.technical_results_area.setText(results_text)

        # Display plot files with directory info
        plot_files_text = "Plot files generated:\n" + "\n".join([Path(p).name for p in results['plot_files']])
        if not results['plot_files']:
            plot_files_text = "No plot files generated"
        self.plots_list_widget.setText(plot_files_text)

        # Switch to technical view
        self.switch_results_view("technical")

        # self.results_summary_label.setText(
        #     f"Analysis complete for {ticker} - Score: {technical_score:.1f}. Check plots directory."
        # )
        self.results_summary_label.setText(
            f"✓ {ticker}: Score {technical_score:.1f}, {recommendation}"
        )

    def handle_stability_error(self, error_message):
        QMessageBox.critical(self, "Stability Analysis Error", f"Error occurred:\n{error_message}")

    def handle_technical_error(self, error_message):
        QMessageBox.critical(self, "Technical Analysis Error", f"Error occurred:\n{error_message}")

    def on_stability_finished(self):
        """Clean up after stability analysis"""
        self.stability_progress_group.setVisible(False)
        self.run_stability_btn.setEnabled(True)
        self.run_stability_btn.setText("Run Stability Analysis")

    def on_technical_finished(self):
        """Clean up after technical analysis"""
        self.technical_progress_group.setVisible(False)
        self.run_technical_btn.setEnabled(True)
        self.run_technical_btn.setText("Generate Technical Plots")

    def display_stability_results(self, df):
        """Enhanced display of stability analysis results with proper scrolling"""
        if df is None or df.empty:
            return

        # Set up table dimensions
        self.stability_table.setRowCount(len(df))
        self.stability_table.setColumnCount(len(df.columns))
        self.stability_table.setHorizontalHeaderLabels(df.columns.tolist())

        # Populate table with formatted data
        for row in range(len(df)):
            for col in range(len(df.columns)):
                value = df.iloc[row, col]

                # Enhanced formatting for different data types
                if isinstance(value, float):
                    if abs(value) < 0.0001:
                        display_value = "0"
                    elif abs(value) < 0.01:
                        display_value = f"{value:.4f}"
                    elif abs(value) < 1:
                        display_value = f"{value:.3f}"
                    elif abs(value) < 100:
                        display_value = f"{value:.2f}"
                    else:
                        display_value = f"{value:.1f}"
                else:
                    display_value = str(value)

                item = QTableWidgetItem(display_value)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

                # Add tooltips for long content
                if len(display_value) > 20:
                    item.setToolTip(display_value)

                self.stability_table.setItem(row, col, item)

        # Smart column sizing with better widths
        header = self.stability_table.horizontalHeader()

        # Set specific widths for different column types
        for i, column_name in enumerate(df.columns):
            column_lower = column_name.lower()

            if column_lower == 'ticker':
                header.resizeSection(i, 80)  # Compact ticker column
            elif column_lower in ['companyname', 'company_name']:
                header.resizeSection(i, 150)  # Company name
            elif column_lower in ['sector']:
                header.resizeSection(i, 120)  # Sector
            elif column_lower in ['industry']:
                header.resizeSection(i, 140)  # Industry
            elif column_lower == 'recommendation':
                header.resizeSection(i, 180)  # Recommendation
            elif 'score' in column_lower:
                header.resizeSection(i, 90)  # Score columns
            elif column_lower in ['appearances', 'avg_rank']:
                header.resizeSection(i, 80)  # Numeric columns
            else:
                header.resizeSection(i, 100)  # Default width

        # Ensure horizontal scrolling is working
        self.stability_table.horizontalHeader().setStretchLastSection(False)

        print(f"📊 Stability table updated: {len(df)} rows × {len(df.columns)} columns")
        print(f"🔄 Horizontal scroll enabled, row numbers hidden")

    def export_stability_results(self):
        """Export stability results - DISABLED (results auto-saved)"""
        # This method can be removed or kept disabled
        pass

    def open_plots_folder(self):
        """Open plots folder - DISABLED (path shown in results)"""
        # This method can be removed or kept disabled
        pass

    def load_technical_plots(self):
        """Load and display technical plots in the plots area"""
        if not self.current_technical_results:
            return

        # Reset dropdown to "All Plots" without triggering signal
        if hasattr(self, 'plot_selector'):
            self.plot_selector.blockSignals(True)
            self.plot_selector.setCurrentIndex(0)
            self.plot_selector.blockSignals(False)

        # Create a widget to hold all plots
        plots_widget = QWidget()
        layout = QGridLayout(plots_widget)
        layout.setSpacing(10)

        plot_files = self.current_technical_results.get('plot_files', [])

        # Filter and sort plot files (exclude overview and original subplot files)
        excluded_patterns = [
            '00_overview',
            '_03_momentum.png',  # Original momentum subplot
            '_04_bollinger_bands.png',  # Original bollinger subplot
            '_05_trend_strength.png'  # Original trend strength subplot
        ]

        filtered_files = []
        for f in plot_files:
            filename = Path(f).name
            if not any(pattern in filename for pattern in excluded_patterns):
                filtered_files.append(f)

        filtered_files.sort()

        # Store filtered files for dropdown
        self.filtered_plot_files = filtered_files

        # Display plots in a grid (2 columns)
        row, col = 0, 0
        for plot_file in filtered_files:
            if Path(plot_file).exists():
                # Create a frame for each plot
                frame = QFrame()
                frame.setFrameStyle(QFrame.Box)
                frame.setStyleSheet("QFrame { border: 1px solid #555; }")
                frame_layout = QVBoxLayout(frame)

                # Load and display the image
                label = QLabel()
                pixmap = QPixmap(plot_file)
                # Scale to fit (max width 500px)
                scaled_pixmap = pixmap.scaledToWidth(500, Qt.SmoothTransformation)
                label.setPixmap(scaled_pixmap)
                label.setAlignment(Qt.AlignCenter)

                # Add filename as title
                title = QLabel(Path(plot_file).stem.replace('_', ' ').title())
                title.setAlignment(Qt.AlignCenter)
                title.setStyleSheet("font-weight: bold; padding: 5px;")

                frame_layout.addWidget(title)
                frame_layout.addWidget(label)

                layout.addWidget(frame, row, col)

                col += 1
                if col >= 2:
                    col = 0
                    row += 1

        # Set the widget in the scroll area
        self.technical_plots_area.setWidget(plots_widget)

class EnhancedTechnicalPlotsThread(QThread):
    """Enhanced thread using the ranked technical analysis approach"""
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, ticker, regime_name):
        super().__init__()
        self.ticker = ticker.upper()
        self.regime_name = regime_name

    def run(self):
        try:
            self.status_update.emit(f"Initializing enhanced technical analysis for {self.ticker}...")
            self.progress_update.emit(10)

            # Set up directories
            csv_directory = PROJECT_ROOT / "output" / "Ranked_Lists" / self.regime_name
            output_base_dir = PROJECT_ROOT / "output" / "Score_Trend_Analysis_Results"

            self.progress_update.emit(30)
            self.status_update.emit("Loading historical data...")

            # Add trend analysis directory to Python path
            trend_analysis_path = PROJECT_ROOT / 'src' / 'trend_analysis'
            sys.path.insert(0, str(trend_analysis_path))

            try:
                # Import the enhanced analyzer
                from stock_score_trend_technical_single_enhanced import EnhancedSingleTickerTechnicalAnalyzer

                self.progress_update.emit(50)
                self.status_update.emit("Running enhanced technical analysis...")

                # Create enhanced analyzer instance
                analyzer = EnhancedSingleTickerTechnicalAnalyzer(
                    csv_directory=str(csv_directory),
                    start_date="0601",
                    end_date=datetime.now().strftime("%m%d"),
                    sigmoid_sensitivity=5
                )

                # Run enhanced analysis
                success, ranking_data = analyzer.analyze_single_ticker_with_ranking(
                    ticker=self.ticker,
                    output_base_dir=str(output_base_dir),
                    min_appearances=3
                )

                if not success:
                    self.error_occurred.emit(f"Enhanced technical analysis failed for {self.ticker}")
                    return

                self.progress_update.emit(90)
                self.status_update.emit("Generating final results...")

                # Find plot files
                tech_plots_dir = output_base_dir / "technical_plots" / self.ticker
                plot_files = list(tech_plots_dir.glob("*.png")) if tech_plots_dir.exists() else []

                # Find ranking file
                ranking_file = tech_plots_dir / f"{self.ticker}_technical_ranking.csv"

                result = {
                    'success': True,
                    'ticker': self.ticker,
                    'regime': self.regime_name,
                    'recommendation': ranking_data['Recommendation'] if ranking_data else 'Unknown',
                    'technical_score': ranking_data['Score'] if ranking_data else 0,
                    'plot_directory': str(tech_plots_dir),
                    'plot_files': [str(p) for p in plot_files],
                    'ranking_file': str(ranking_file) if ranking_file.exists() else None,
                    'ranking_data': ranking_data,
                    'total_plots': len(plot_files)
                }

                self.progress_update.emit(100)
                self.result_ready.emit(result)

            except ImportError as e:
                self.error_occurred.emit(
                    f"Failed to import enhanced analyzer: {str(e)}\nMake sure stock_score_trend_technical_single_enhanced.py is in {trend_analysis_path}")
            except Exception as e:
                self.error_occurred.emit(f"Enhanced technical analysis failed: {str(e)}")
            finally:
                # Clean up sys.path
                if str(trend_analysis_path) in sys.path:
                    sys.path.remove(str(trend_analysis_path))

        except Exception as e:
            self.error_occurred.emit(f"Thread execution failed: {str(e)}")


class EnhancedStabilityAnalysisThread(QThread):
    """Enhanced stability analysis thread that includes sector/industry info"""
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, regime_name):
        super().__init__()
        self.regime_name = regime_name

    def run(self):
        try:
            self.status_update.emit("Starting enhanced stability analysis...")
            self.progress_update.emit(10)

            # Find the most recent ranked stocks file to get sector/industry info
            ranked_lists_dir = PROJECT_ROOT / "output" / "Ranked_Lists" / self.regime_name

            if not ranked_lists_dir.exists():
                self.error_occurred.emit(f"Ranked lists directory not found: {ranked_lists_dir}")
                return

            # Find the most recent CSV file
            csv_files = sorted(ranked_lists_dir.glob("top_ranked_stocks_*.csv"))
            if not csv_files:
                self.error_occurred.emit(f"No ranked stocks files found in {ranked_lists_dir}")
                return

            most_recent_file = csv_files[-1]

            self.progress_update.emit(30)
            self.status_update.emit("Loading sector/industry information...")

            # Load sector/industry mapping from most recent file
            sector_industry_df = pd.read_csv(most_recent_file)[['Ticker', 'CompanyName', 'Sector', 'Industry']]

            self.progress_update.emit(50)
            self.status_update.emit("Running stability analysis...")

            # Run original stability analysis
            script_path = PROJECT_ROOT / 'src' / 'trend_analysis' / 'stock_score_trend_analyzer_04.py'

            sys.path.insert(0, str(script_path.parent))

            try:
                import stock_score_trend_analyzer_04 as analyzer_module

                # Create analyzer instance
                csv_directory = PROJECT_ROOT / "output" / "Ranked_Lists" / self.regime_name
                output_directory = PROJECT_ROOT / "output" / "Score_Trend_Analysis_Results" / self.regime_name
                output_directory.mkdir(parents=True, exist_ok=True)

                analyzer = analyzer_module.StockScoreTrendAnalyzer(
                    csv_directory=str(csv_directory),
                    start_date="0601",
                    end_date=datetime.now().strftime("%m%d"),
                    sigmoid_sensitivity=5
                )

                self.progress_update.emit(70)
                self.status_update.emit("Analyzing stock trends...")

                # Run analysis and export results
                today = datetime.now().strftime("%m%d")
                output_file = output_directory / f"stability_analysis_results_{today}.csv"
                results_df = analyzer.export_results(output_path=str(output_file))

                self.progress_update.emit(90)
                self.status_update.emit("Adding sector/industry information...")

                # Merge with sector/industry information
                enhanced_df = results_df.merge(
                    sector_industry_df,
                    left_on='ticker',
                    right_on='Ticker',
                    how='left'
                ).drop('Ticker', axis=1)  # Remove duplicate ticker column

                # Reorder columns to put sector/industry info early
                column_order = ['ticker', 'CompanyName', 'Sector', 'Industry'] + \
                               [col for col in enhanced_df.columns if
                                col not in ['ticker', 'CompanyName', 'Sector', 'Industry']]
                enhanced_df = enhanced_df[column_order]

                # Fill missing sector/industry info
                enhanced_df['CompanyName'] = enhanced_df['CompanyName'].fillna('Unknown')
                enhanced_df['Sector'] = enhanced_df['Sector'].fillna('Unknown')
                enhanced_df['Industry'] = enhanced_df['Industry'].fillna('Unknown')

                # Save enhanced results
                enhanced_output_file = output_directory / f"stability_analysis_enhanced_{today}.csv"
                enhanced_df.to_csv(enhanced_output_file, index=False)

                self.progress_update.emit(100)

                # Prepare result data
                result = {
                    'success': True,
                    'output_path': str(enhanced_output_file),
                    'original_output_path': str(output_file),
                    'results_df': enhanced_df,
                    'regime': self.regime_name,
                    'total_stocks': len(enhanced_df),
                    'analyzer': analyzer,
                    'has_sector_info': True
                }

                self.result_ready.emit(result)

            except Exception as e:
                self.error_occurred.emit(f"Enhanced analysis failed: {str(e)}")
            finally:
                if str(script_path.parent) in sys.path:
                    sys.path.remove(str(script_path.parent))

        except Exception as e:
            self.error_occurred.emit(f"Thread execution failed: {str(e)}")


# Modified MainWindow class to use tabs
class MainWindow(QMainWindow):
    """Main application window with three tabs"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Multifactor Stock Analysis System (MSAS)")
        self.setGeometry(100, 100, 1500, 1000)  # Increased window size

        # Set modern dark theme (same styling as before)
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
                        padding: 12px 20px;
                        margin-right: 2px;
                        border-top-left-radius: 5px;
                        border-top-right-radius: 5px;
                        min-width: 120px;
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

        # Add all three tabs
        self.regime_widget = RegimeDetectionWidget()
        self.scoring_widget = MultifactorScoringWidget()
        self.trend_analysis_widget = ScoreTrendAnalysisWidget()

        self.tab_widget.addTab(self.regime_widget, "Regime Detection")
        self.tab_widget.addTab(self.scoring_widget, "Multifactor Scoring")
        self.tab_widget.addTab(self.trend_analysis_widget, "Score Trend Analysis")

        # FIXED: Connect to the correct signal from RegimeDetectionWidget
        self.regime_widget.regime_detected.connect(self.update_trend_regime_recommendation)

        # Set default tab
        self.tab_widget.setCurrentIndex(0)  # Start with Regime Detection

        # Create enhanced menu bar
        self.create_menu_bar()

    def update_trend_regime_recommendation(self, detected_regime):
        """Update trend analysis with detected regime recommendation"""
        print(f"📡 Regime detected: {detected_regime}")

        # Update the trend analysis widget with the detected regime
        self.trend_analysis_widget.set_recommended_regime(detected_regime)

    def create_menu_bar(self):
        """Create enhanced application menu bar"""
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

        export_action = file_menu.addAction('Export Current Results')
        export_action.setShortcut('Ctrl+E')
        export_action.triggered.connect(self.export_current_results)

        file_menu.addSeparator()

        exit_action = file_menu.addAction('Exit')
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)

        # Analysis menu
        analysis_menu = menubar.addMenu('Analysis')

        run_regime_action = analysis_menu.addAction('Run Regime Detection')
        run_regime_action.setShortcut('F5')
        run_regime_action.triggered.connect(lambda: self.regime_widget.detect_current_regime())

        run_scoring_action = analysis_menu.addAction('Run Scoring Process')
        run_scoring_action.setShortcut('Ctrl+R')
        run_scoring_action.triggered.connect(lambda: self.scoring_widget.run_scoring_process())

        analysis_menu.addSeparator()

        run_stability_action = analysis_menu.addAction('Run Stability Analysis')
        run_stability_action.setShortcut('Ctrl+T')
        run_stability_action.triggered.connect(lambda: self.trend_analysis_widget.run_stability_analysis())

        # View menu
        view_menu = menubar.addMenu('View')

        regime_tab_action = view_menu.addAction('Regime Detection Tab')
        regime_tab_action.setShortcut('Ctrl+1')
        regime_tab_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(0))

        scoring_tab_action = view_menu.addAction('Multifactor Scoring Tab')
        scoring_tab_action.setShortcut('Ctrl+2')
        scoring_tab_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(1))

        trend_tab_action = view_menu.addAction('Score Trend Analysis Tab')
        trend_tab_action.setShortcut('Ctrl+3')
        trend_tab_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(2))

        # Help menu
        help_menu = menubar.addMenu('Help')

        about_action = help_menu.addAction('About')
        about_action.triggered.connect(self.show_about)

    def export_current_results(self):
        """Export results from currently active tab"""
        current_tab = self.tab_widget.currentIndex()

        if current_tab == 0:  # Regime Detection
            if hasattr(self.regime_widget, 'historical_results') and self.regime_widget.historical_results:
                QMessageBox.information(self, "Export", "Regime results exported!")
            else:
                QMessageBox.warning(self, "Export", "No regime results to export.")

        elif current_tab == 1:  # Multifactor Scoring
            if hasattr(self.scoring_widget, 'current_results') and self.scoring_widget.current_results:
                self.scoring_widget.export_results()
            else:
                QMessageBox.warning(self, "Export", "No scoring results to export.")

        elif current_tab == 2:  # Score Trend Analysis
            if hasattr(self.trend_analysis_widget,
                       'current_stability_results') and self.trend_analysis_widget.current_stability_results:
                self.trend_analysis_widget.export_stability_results()
            else:
                QMessageBox.warning(self, "Export", "No trend analysis results to export.")

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
        """Show enhanced about dialog"""
        QMessageBox.about(self, "About",
                          "Multifactor Stock Analysis System (MSAS)\n"
                          "Version 2.0\n\n"
                          "This comprehensive system provides:\n"
                          "• Market regime detection using Hidden Markov Models\n"
                          "• Advanced multifactor stock scoring and ranking\n"
                          "• Score trend analysis and stability assessment\n"
                          "• Technical analysis with detailed plotting\n\n"
                          "Tabs:\n"
                          "• Regime Detection: Analyze current market conditions\n"
                          "• Multifactor Scoring: Score and rank stocks using multiple factors\n"
                          "• Score Trend Analysis: Assess stability and generate technical plots\n\n"
                          "Keyboard Shortcuts:\n"
                          "• Ctrl+1/2/3: Switch between tabs\n"
                          "• Ctrl+R: Run scoring process\n"
                          "• Ctrl+T: Run stability analysis\n"
                          "• F5: Refresh regime detection\n\n"
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