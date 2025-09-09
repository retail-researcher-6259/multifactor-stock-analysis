# src/classes/regime_detection_widget.py
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json
import pyqtgraph as pg
from pyqtgraph import DateAxisItem
import subprocess
# For dynamic plotting
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

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

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            encoding='utf-8',
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

        # Read the output files from BOTH directories
        analysis_dir = PROJECT_ROOT / 'output' / 'Regime_Detection_Analysis'
        results_dir = PROJECT_ROOT / 'output' / 'Regime_Detection_Results'
        results = {}

        # Files can be in either directory, so check both
        files_to_read = {
            'data_summary': ('data_summary.json', analysis_dir),
            'validation_results': ('validation_results.json', analysis_dir),
            'historical_market_data': ('historical_market_data.csv', analysis_dir),
            'regime_periods': ('regime_periods.csv', results_dir),  # This is in Results dir
            'regime_model': ('regime_model.pkl', results_dir)  # This is also in Results dir
        }

        for key, (filename, directory) in files_to_read.items():
            file_path = directory / filename

            # If not found in primary location, check the other directory
            if not file_path.exists():
                alt_dir = results_dir if directory == analysis_dir else analysis_dir
                file_path = alt_dir / filename

            if file_path.exists():
                if filename.endswith('.json'):
                    with open(file_path, 'r') as f:
                        results[key] = json.load(f)
                elif filename.endswith('.csv'):
                    results[key] = pd.read_csv(file_path).to_dict('records')
                elif filename.endswith('.pkl'):
                    # Just confirm the model exists
                    results[key] = {'exists': True, 'path': str(file_path)}
                print(f"✓ Loaded {filename} from {file_path.parent.name}")
            else:
                print(f"✗ Could not find {filename}")

        # Check for the plot image
        plot_path = results_dir / 'regime_detection_plot.png'
        if plot_path.exists():
            results['plot_path'] = str(plot_path)
            print(f"✓ Found plot at {plot_path}")

        # Verify we have the essential files
        if 'regime_periods' in results and 'regime_model' in results:
            self.progress_update.emit(100)
            self.status_update.emit("Historical data fetched successfully!")
            self.result_ready.emit({'type': 'historical', 'data': results})
        else:
            missing = []
            if 'regime_periods' not in results:
                missing.append('regime_periods.csv')
            if 'regime_model' not in results:
                missing.append('regime_model.pkl')

            self.error_occurred.emit(f"Missing essential files: {', '.join(missing)}")

    def load_historical_data(self):
        """Load saved historical data"""
        self.status_update.emit("Loading saved data...")
        self.progress_update.emit(50)

        analysis_dir = PROJECT_ROOT / 'output' / 'Regime_Detection_Analysis'
        results_dir = PROJECT_ROOT / 'output' / 'Regime_Detection_Results'
        results = {}

        # Load saved files from both directories
        files_to_load = {
            'data_summary': ('data_summary.json', analysis_dir),
            'validation_results': ('validation_results.json', analysis_dir),
            'historical_market_data': ('historical_market_data.csv', analysis_dir),
            'regime_periods': ('regime_periods.csv', results_dir),
            'regime_model': ('regime_model.pkl', results_dir)
        }

        for key, (filename, directory) in files_to_load.items():
            file_path = directory / filename

            # If not found in primary location, check the other directory
            if not file_path.exists():
                alt_dir = results_dir if directory == analysis_dir else analysis_dir
                file_path = alt_dir / filename

            if file_path.exists():
                if filename.endswith('.json'):
                    with open(file_path, 'r') as f:
                        results[key] = json.load(f)
                elif filename.endswith('.csv'):
                    results[key] = pd.read_csv(file_path).to_dict('records')
                elif filename.endswith('.pkl'):
                    results[key] = {'exists': True, 'path': str(file_path)}

        # Check for plot
        plot_path = results_dir / 'regime_detection_plot.png'
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
        """Create simplified historical regime section"""
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

        # Simplified statistics display (dark themed)
        self.historical_stats_label = QLabel("HISTORICAL REGIME ANALYSIS\n\nNo data loaded")
        self.historical_stats_label.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                padding: 10px;
                border-radius: 5px;
                border: 1px solid #444;
                color: #ffffff;
                font-family: 'Courier New', monospace;
                font-size: 11px;
            }
        """)
        self.historical_stats_label.setWordWrap(True)
        self.historical_stats_label.setMinimumHeight(100)
        self.historical_stats_label.setMaximumHeight(150)
        layout.addWidget(self.historical_stats_label)

        layout.addStretch()
        group.setLayout(layout)
        return group

    def update_timeline_plot(self):
        """Create clean SPY price plot colored by regime with CORRECT regime names"""
        try:
            # Check if we have the necessary data files
            regime_periods_path = PROJECT_ROOT / 'output' / 'Regime_Detection_Results' / 'regime_periods.csv'
            market_data_path = PROJECT_ROOT / 'output' / 'Regime_Detection_Analysis' / 'historical_market_data.csv'

            if not regime_periods_path.exists() or not market_data_path.exists():
                self.update_status("Data files not found. Please fetch historical data first.")
                return

            # Load the data
            regime_periods = pd.read_csv(regime_periods_path)
            market_data = pd.read_csv(market_data_path)

            # Convert date columns
            market_data['Date'] = pd.to_datetime(market_data['Date'])
            market_data = market_data.set_index('Date')

            regime_periods['start_date'] = pd.to_datetime(regime_periods['start_date'])
            regime_periods['end_date'] = pd.to_datetime(regime_periods['end_date'])

            # Filter based on selected period
            selected_period = self.period_combo.currentText()
            end_date = market_data.index[-1]

            if selected_period == "Last Year":
                start_date = end_date - pd.DateOffset(years=1)
            elif selected_period == "Last 3 Years":
                start_date = end_date - pd.DateOffset(years=3)
            elif selected_period == "Last 5 Years":
                start_date = end_date - pd.DateOffset(years=5)
            elif selected_period == "Last 10 Years":
                start_date = end_date - pd.DateOffset(years=10)
            else:  # All Time
                start_date = market_data.index[0]

            # Filter market data
            filtered_data = market_data[market_data.index >= start_date].copy()

            # Clear the figure
            self.figure.clear()

            # Create subplot with dark theme
            ax = self.figure.add_subplot(111, facecolor='#1e1e1e')

            # Get SPY data
            if 'SPY' not in filtered_data.columns:
                ax.text(0.5, 0.5, 'SPY data not available',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=14, color='#888')
                ax.axis('off')
                self.canvas.draw()
                return

            spy_data = filtered_data['SPY'].dropna()

            # Define regime colors - USE THE CORRECT NAMES FROM YOUR DATA
            regime_colors = {
                'Crisis/Bear': '#f44336',  # Red
                'Strong Bull': '#4caf50',  # Green
                'Steady Growth': '#2196f3',  # Blue
                # Add any other regime names that might appear in your data
                'High Volatility': '#4caf50',  # Green (if it appears)
                'Unknown': '#757575'  # Grey
            }

            # Create regime history for filtered period
            regime_history = pd.Series(index=spy_data.index, dtype=str)

            for _, period in regime_periods.iterrows():
                mask = (spy_data.index >= period['start_date']) & (spy_data.index <= period['end_date'])
                regime_history[mask] = period['regime_name']

            # Plot SPY price colored by regime (scatter plot for better color visibility)
            plotted_regimes = set()

            for regime_name in regime_history.dropna().unique():
                mask = regime_history == regime_name
                regime_data = spy_data[mask]

                if not regime_data.empty:
                    color = regime_colors.get(regime_name, '#757575')  # Default to grey if unknown

                    # Use scatter plot with small points for better color visibility
                    ax.scatter(regime_data.index, regime_data.values,
                               c=color, s=2, alpha=0.8, label=regime_name)
                    plotted_regimes.add(regime_name)

            # Formatting
            ax.set_xlabel('Date', color='#ccc')
            ax.set_ylabel('SPY Price ($)', color='#ccc')
            ax.set_title(f'SPY Price Colored by Regime - {selected_period}',
                         fontsize=14, color='white', pad=20)

            # Legend with only plotted regimes
            if plotted_regimes:
                ax.legend(loc='upper left', frameon=True,
                          facecolor='#2b2b2b', edgecolor='#444',
                          fontsize=9)

            # Grid
            ax.grid(True, alpha=0.2, color='#444', linestyle='-', linewidth=0.5)

            # Style the axes
            ax.tick_params(colors='#ccc')
            for spine in ax.spines.values():
                spine.set_color('#444')

            # Format x-axis dates
            import matplotlib.dates as mdates
            if selected_period == "Last Year":
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            elif selected_period == "Last 3 Years":
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            elif selected_period == "Last 5 Years":
                ax.xaxis.set_major_locator(mdates.YearLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            else:
                ax.xaxis.set_major_locator(mdates.YearLocator(2))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

            # Rotate x-axis labels
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            # Adjust layout
            self.figure.tight_layout()

            # Refresh canvas
            self.canvas.draw()

            self.update_status(f"Timeline plot updated: {selected_period}")

        except Exception as e:
            print(f"Error updating timeline plot: {e}")
            import traceback
            traceback.print_exc()

            # Show error on plot
            self.figure.clear()
            ax = self.figure.add_subplot(111, facecolor='#1e1e1e')
            ax.text(0.5, 0.5, f'Error creating plot:\n{str(e)}',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='#f44336')
            ax.axis('off')
            self.canvas.draw()

    # def create_timeline_section(self):
    #     """Create timeline visualization section with interactive plot"""
    #     group = QGroupBox("Regime Timeline Visualization")
    #     layout = QVBoxLayout()
    #
    #     # Add controls for time period selection
    #     controls_layout = QHBoxLayout()
    #     controls_layout.addWidget(QLabel("Time Period:"))
    #
    #     self.period_combo = QComboBox()
    #     self.period_combo.addItems(["30 Days", "3 Months", "1 Year", "5 Years", "All Time"])
    #     self.period_combo.currentTextChanged.connect(self.update_timeline_plot)
    #     controls_layout.addWidget(self.period_combo)
    #
    #     self.refresh_plot_btn = QPushButton("Refresh Plot")
    #     self.refresh_plot_btn.clicked.connect(self.update_timeline_plot)
    #     controls_layout.addWidget(self.refresh_plot_btn)
    #
    #     controls_layout.addStretch()
    #     layout.addLayout(controls_layout)
    #
    #     # Create plot widget using pyqtgraph for interactive plotting
    #     self.regime_plot = pg.PlotWidget(
    #         title="SPY Price Colored by Regime",
    #         axisItems={'bottom': DateAxisItem(orientation='bottom')}
    #     )
    #     self.regime_plot.setLabel('left', 'Price ($)')
    #     self.regime_plot.setLabel('bottom', 'Date')
    #     self.regime_plot.showGrid(x=True, y=True, alpha=0.3)
    #     self.regime_plot.setMinimumHeight(400)
    #
    #     # Add legend
    #     self.regime_plot.addLegend()
    #
    #     layout.addWidget(self.regime_plot)
    #
    #     group.setLayout(layout)
    #     return group

    def create_timeline_section(self):
        """Create the timeline visualization section with dynamic plotting"""
        group = QGroupBox("Regime Timeline Visualization")
        layout = QVBoxLayout()

        # Controls for the timeline
        controls_layout = QHBoxLayout()

        # Period selector
        period_label = QLabel("Display Period:")
        controls_layout.addWidget(period_label)

        self.period_combo = QComboBox()
        self.period_combo.addItems(["All Time", "Last 10 Years", "Last 5 Years", "Last 3 Years", "Last Year"])
        self.period_combo.setCurrentText("All Time")
        self.period_combo.currentTextChanged.connect(self.update_timeline_plot)
        controls_layout.addWidget(self.period_combo)

        controls_layout.addStretch()

        # Refresh button
        refresh_btn = QPushButton("Refresh Plot")
        refresh_btn.clicked.connect(self.update_timeline_plot)
        self.refresh_plot_btn = refresh_btn
        controls_layout.addWidget(refresh_btn)

        layout.addLayout(controls_layout)

        # Create the plot widget using matplotlib
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        import matplotlib.pyplot as plt

        # Set dark style for matplotlib
        plt.style.use('dark_background')

        # Create figure with dark background
        self.figure = Figure(figsize=(12, 6), facecolor='#1e1e1e')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: #1e1e1e;")

        # Add canvas to layout
        layout.addWidget(self.canvas)

        # Initialize with placeholder
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, 'No regime data loaded\nClick "Fetch Regime Data" to load historical data',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14, color='#888')
        ax.set_facecolor('#1e1e1e')
        ax.axis('off')
        self.canvas.draw()

        group.setLayout(layout)
        return group

    def export_timeline_plot(self):
        """Export the timeline plot to a file"""
        if self.historical_results and 'plot_path' in self.historical_results:
            # Copy the plot to a user-selected location
            from PyQt5.QtWidgets import QFileDialog
            import shutil

            save_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Regime Plot",
                "regime_timeline.png",
                "PNG Files (*.png);;All Files (*)"
            )

            if save_path:
                try:
                    shutil.copy(self.historical_results['plot_path'], save_path)
                    self.update_status(f"Plot exported to {save_path}")
                except Exception as e:
                    self.update_status(f"Error exporting plot: {str(e)}")

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
        """Display simplified historical regime results"""
        self.historical_results = data

        # Build simplified statistics text
        stats_text = "HISTORICAL REGIME ANALYSIS\n\n"

        # Validation accuracy
        if 'validation_results' in data:
            validation = data['validation_results']
            accuracy = validation.get('accuracy', 0)
            stats_text += f"Validation Accuracy: {accuracy:.1%}\n\n"

        # Regime distribution - SIMPLIFIED
        if 'regime_periods' in data:
            periods = data['regime_periods']

            # Calculate percentages for each regime
            regime_stats = {}
            total_days = 0

            for period in periods:
                regime_name = period.get('regime_name', 'Unknown')
                days = period.get('days', 0)

                if regime_name not in regime_stats:
                    regime_stats[regime_name] = 0

                regime_stats[regime_name] += days
                total_days += days

            # Display each regime with percentage
            for regime_name in ['Steady Growth', 'Strong Bull', 'Crisis/Bear']:
                if regime_name in regime_stats:
                    percentage = (regime_stats[regime_name] / total_days) * 100 if total_days > 0 else 0
                    stats_text += f"{regime_name}: {percentage:.1f}%\n"
                else:
                    stats_text += f"{regime_name}: 0.0%\n"

        # Update the label
        self.historical_stats_label.setText(stats_text)

        # Update the timeline plot
        self.update_timeline_plot()

        # Update status
        self.status_label.setText("Historical data loaded successfully")
        self.load_data_btn.setEnabled(True)

    def display_regime_plot(self, plot_path):
        """Display the regime detection plot in the timeline section"""
        try:
            from PyQt5.QtGui import QPixmap

            # Load the image
            pixmap = QPixmap(plot_path)

            if not pixmap.isNull():
                # Calculate the label size
                label_width = self.plot_label.width()
                label_height = self.plot_label.height()

                # Scale the pixmap to fit the label while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(
                    label_width if label_width > 0 else 800,
                    label_height if label_height > 0 else 400,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )

                self.plot_label.setPixmap(scaled_pixmap)
                self.plot_label.setAlignment(Qt.AlignCenter)

                # Update the label style to remove the dashed border
                self.plot_label.setStyleSheet("""
                    QLabel {
                        background-color: white;
                        border: 1px solid #ddd;
                        border-radius: 5px;
                    }
                """)

                print(f"✓ Displayed plot from {plot_path}")
            else:
                self.plot_label.setText("Error: Could not load plot image")
                print(f"✗ Failed to load plot from {plot_path}")

        except Exception as e:
            print(f"Error displaying plot: {e}")
            self.plot_label.setText(f"Error displaying plot: {str(e)}")

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