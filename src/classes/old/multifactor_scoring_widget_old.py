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
            f"• Data Source: {PROJECT_ROOT / 'config' / 'Buyable_stocks.txt'}\n"
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