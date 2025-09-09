# src/classes/score_trend_analysis_widget.py
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

        # ENSURE HORIZONTAL SCROLLBAR
        self.stability_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.stability_table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.stability_table.horizontalHeader().setStretchLastSection(False)
        self.stability_table.setSizeAdjustPolicy(QAbstractItemView.AdjustToContents)

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

    # def display_stability_results(self, df):
    #     """Enhanced display of stability analysis results with proper scrolling"""
    #     if df is None or df.empty:
    #         return
    #
    #     # Set up table dimensions
    #     self.stability_table.setRowCount(len(df))
    #     self.stability_table.setColumnCount(len(df.columns))
    #     self.stability_table.setHorizontalHeaderLabels(df.columns.tolist())
    #
    #     # Populate table with formatted data
    #     for row in range(len(df)):
    #         for col in range(len(df.columns)):
    #             value = df.iloc[row, col]
    #
    #             # Enhanced formatting for different data types
    #             if isinstance(value, float):
    #                 if abs(value) < 0.0001:
    #                     display_value = "0"
    #                 elif abs(value) < 0.01:
    #                     display_value = f"{value:.4f}"
    #                 elif abs(value) < 1:
    #                     display_value = f"{value:.3f}"
    #                 elif abs(value) < 100:
    #                     display_value = f"{value:.2f}"
    #                 else:
    #                     display_value = f"{value:.1f}"
    #             else:
    #                 display_value = str(value)
    #
    #             item = QTableWidgetItem(display_value)
    #             item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
    #
    #             # Add tooltips for long content
    #             if len(display_value) > 20:
    #                 item.setToolTip(display_value)
    #
    #             self.stability_table.setItem(row, col, item)
    #
    #     # Smart column sizing with better widths
    #     header = self.stability_table.horizontalHeader()
    #
    #     # Set specific widths for different column types
    #     for i, column_name in enumerate(df.columns):
    #         column_lower = column_name.lower()
    #
    #         if column_lower == 'ticker':
    #             header.resizeSection(i, 80)  # Compact ticker column
    #         elif column_lower in ['companyname', 'company_name']:
    #             header.resizeSection(i, 150)  # Company name
    #         elif column_lower in ['sector']:
    #             header.resizeSection(i, 120)  # Sector
    #         elif column_lower in ['industry']:
    #             header.resizeSection(i, 140)  # Industry
    #         elif column_lower == 'recommendation':
    #             header.resizeSection(i, 180)  # Recommendation
    #         elif 'score' in column_lower:
    #             header.resizeSection(i, 90)  # Score columns
    #         elif column_lower in ['appearances', 'avg_rank']:
    #             header.resizeSection(i, 80)  # Numeric columns
    #         else:
    #             header.resizeSection(i, 100)  # Default width
    #
    #     # Ensure horizontal scrolling is working
    #     self.stability_table.horizontalHeader().setStretchLastSection(False)
    #
    #     print(f"📊 Stability table updated: {len(df)} rows × {len(df.columns)} columns")
    #     print(f"🔄 Horizontal scroll enabled, row numbers hidden")

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

        # AUTO-ADJUST COLUMN WIDTHS - IMPROVED VERSION
        header = self.stability_table.horizontalHeader()

        # Method 1: Resize to contents first
        self.stability_table.resizeColumnsToContents()

        # Method 2: Set minimum and maximum widths for better readability
        for i in range(self.stability_table.columnCount()):
            current_width = self.stability_table.columnWidth(i)
            column_name = df.columns[i].lower()

            # Set minimum widths based on column type
            if column_name == 'ticker':
                min_width, max_width = 60, 100
            elif column_name in ['companyname', 'company_name']:
                min_width, max_width = 150, 300
            elif column_name == 'recommendation':
                min_width, max_width = 180, 250
            elif column_name in ['sector', 'industry']:
                min_width, max_width = 100, 200
            elif 'score' in column_name:
                min_width, max_width = 80, 120
            else:
                min_width, max_width = 60, 150

            # Apply constraints
            final_width = max(min_width, min(current_width, max_width))
            self.stability_table.setColumnWidth(i, final_width)

        # Ensure horizontal scrolling is working
        self.stability_table.horizontalHeader().setStretchLastSection(False)
        self.stability_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        print(f"📊 Stability table updated: {len(df)} rows × {len(df.columns)} columns")
        print(f"📏 Columns auto-sized with constraints")

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
            csv_files = sorted(ranked_lists_dir.glob(f"top_ranked_stocks_{self.regime_name.replace(' ', '_')}*.csv"))

            # If no files found, try old pattern as fallback
            if not csv_files:
                csv_files = sorted(ranked_lists_dir.glob("top_ranked_stocks_*.csv"))

            if not csv_files:
                self.error_occurred.emit(f"No ranked stocks files found in {ranked_lists_dir}")
                return

            most_recent_file = csv_files[-1]

            self.progress_update.emit(30)
            self.status_update.emit("Loading sector/industry information...")

            # Check what columns are available in the ranked file
            ranked_df = pd.read_csv(most_recent_file)
            available_cols = ranked_df.columns.tolist()

            # Determine which columns we can use
            required_cols = ['Ticker']
            optional_cols = ['CompanyName', 'Sector', 'Industry', 'Country']

            # Only select columns that actually exist
            cols_to_load = required_cols + [col for col in optional_cols if col in available_cols]

            if len(cols_to_load) == 1:  # Only Ticker found
                self.status_update.emit("Warning: No sector/industry info in ranked file, using basic analysis...")
                sector_industry_df = None
            else:
                sector_industry_df = ranked_df[cols_to_load]

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
                # The export_results method now handles sector/industry info internally
                today = datetime.now().strftime("%m%d")
                output_file = output_directory / f"stability_analysis_results_{today}.csv"

                # Pass the ranked file path so the analyzer can add sector info
                results_df = analyzer.export_results(
                    output_path=str(output_file),
                    ranked_file_path=str(most_recent_file)
                )

                self.progress_update.emit(90)

                # Check if sector info was already added
                has_sector_info = 'Sector' in results_df.columns and 'Industry' in results_df.columns

                if has_sector_info:
                    self.status_update.emit("Analysis complete with sector information")
                    enhanced_df = results_df
                    enhanced_output_file = output_file  # Use the same file since it already has sector info
                else:
                    self.status_update.emit("Analysis complete (no sector information available)")
                    enhanced_df = results_df
                    enhanced_output_file = output_file

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
                    'has_sector_info': has_sector_info
                }

                self.result_ready.emit(result)

            except Exception as e:
                self.error_occurred.emit(f"Enhanced analysis failed: {str(e)}")
            finally:
                if str(script_path.parent) in sys.path:
                    sys.path.remove(str(script_path.parent))

        except Exception as e:
            self.error_occurred.emit(f"Thread execution failed: {str(e)}")