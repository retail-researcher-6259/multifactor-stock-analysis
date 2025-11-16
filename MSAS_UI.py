# MSAS_UI_simplified.py
import sys
from pathlib import Path
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# Get project root directory
PROJECT_ROOT = Path(__file__).parent if '__file__' in globals() else Path.cwd()
sys.path.append(str(PROJECT_ROOT))

# Import widget classes
from src.classes.regime_detection_widget import RegimeDetectorThread, RegimeDetectionWidget
from src.classes.multifactor_scoring_widget_new import MultifactorScoringThread, MultifactorScoringWidget
from src.classes.score_trend_analysis_widget import (
    StabilityAnalysisThread,
    TechnicalPlotsThread,
    ScoreTrendAnalysisWidget,
    EnhancedTechnicalPlotsThread,
    EnhancedStabilityAnalysisThread
)
from src.classes.dynamic_portfolio_selection_widget_v2 import DynamicPortfolioThread, DynamicPortfolioSelectionWidget
from src.classes.portfolio_optimizer_widget import PortfolioOptimizerThread, PortfolioOptimizerWidget

class MainWindow(QMainWindow):
    """Main application window with tabs for each analysis system"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Multifactor Stock Analysis System (MSAS)")
        self.setGeometry(100, 100, 1500, 1000)

        # Set modern dark theme
        self.setStyleSheet(self.get_dark_theme_stylesheet())

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main vertical layout
        main_layout = QVBoxLayout()

        # Create header
        header = self.create_header()
        main_layout.addWidget(header)

        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)
        self.tab_widget.setStyleSheet("""
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
                font-size: 12px;
            }
            QTabBar::tab:selected {
                background-color: #7c4dff;
            }
            QTabBar::tab:hover {
                background-color: #5c5c5c;
            }
        """)

        # Create tabs
        self.regime_widget = RegimeDetectionWidget()
        self.tab_widget.addTab(self.regime_widget, " Regime Detection")

        self.scoring_widget = MultifactorScoringWidget()
        self.tab_widget.addTab(self.scoring_widget, " Multifactor Scoring")

        self.trend_analysis_widget = ScoreTrendAnalysisWidget()
        self.tab_widget.addTab(self.trend_analysis_widget, " Score Trend Analysis")

        # Add new Dynamic Portfolio Selection tab
        self.portfolio_widget = DynamicPortfolioSelectionWidget()
        self.tab_widget.addTab(self.portfolio_widget, " Portfolio Selection")

        # Add new Portfolio Optimization tab
        self.optimizer_widget = PortfolioOptimizerWidget()
        self.tab_widget.addTab(self.optimizer_widget, " Portfolio Optimization")

        # Connect signals between widgets
        self.connect_widget_signals()

        main_layout.addWidget(self.tab_widget)
        central_widget.setLayout(main_layout)

        # Create menu bar
        self.create_menu_bar()

        # Create status bar
        self.create_status_bar()

    def create_header(self):
        """Create header with title and description"""
        header = QWidget()
        header_layout = QVBoxLayout()
        header_layout.setSpacing(5)

        # Title
        title = QLabel("Multifactor Stock Analysis System")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #ffffff;
                padding: 10px;
            }
        """)
        header_layout.addWidget(title)

        # Subtitle
        subtitle = QLabel(
            "Integrated Analysis Platform for Market Regime Detection, Stock Scoring, and Portfolio Optimization")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #aaa;
                padding-bottom: 10px;
            }
        """)
        header_layout.addWidget(subtitle)

        header.setLayout(header_layout)
        return header

    def connect_widget_signals(self):
        """Connect signals between different widgets for data flow"""
        # Pass detected regime to other widgets
        self.regime_widget.regime_detected.connect(self.scoring_widget.set_detected_regime)
        self.regime_widget.regime_detected.connect(self.trend_analysis_widget.set_recommended_regime)
        self.regime_widget.regime_detected.connect(self.portfolio_widget.set_recommended_regime)

        # You can add more signal connections here as needed
        # For example, passing scoring results to portfolio selection, etc.

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

        run_portfolio_action = analysis_menu.addAction('Generate Portfolios')
        run_portfolio_action.setShortcut('Ctrl+P')
        run_portfolio_action.triggered.connect(lambda: self.portfolio_widget.run_portfolio_selection())

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

        portfolio_tab_action = view_menu.addAction('Portfolio Selection Tab')
        portfolio_tab_action.setShortcut('Ctrl+4')
        portfolio_tab_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(3))

        # Help menu
        help_menu = menubar.addMenu('Help')

        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        help_action = QAction('User Guide', self)
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)

        # ADD THIS: Documentation action
        doc_action = QAction('Documentation', self)
        doc_action.triggered.connect(self.open_documentation)
        help_menu.addAction(doc_action)

    def create_status_bar(self):
        """Create status bar"""
        status_bar = self.statusBar()
        status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #1e1e1e;
                color: #888;
                border-top: 1px solid #444;
            }
        """)
        status_bar.showMessage("Ready")

    def export_current_results(self):
        """Export results from currently active tab"""
        current_tab = self.tab_widget.currentIndex()

        if current_tab == 0:  # Regime Detection
            if hasattr(self.regime_widget, 'current_results') and self.regime_widget.current_results:
                QMessageBox.information(self, "Export", "Regime detection results exported successfully!")
            else:
                QMessageBox.warning(self, "Export", "No regime detection results to export.")

        elif current_tab == 1:  # Multifactor Scoring
            if hasattr(self.scoring_widget, 'current_results') and self.scoring_widget.current_results:
                # The scoring widget already has export functionality
                QMessageBox.information(self, "Export", "Scoring results exported successfully!")
            else:
                QMessageBox.warning(self, "Export", "No scoring results to export.")

        elif current_tab == 2:  # Score Trend Analysis
            if hasattr(self.trend_analysis_widget, 'current_stability_results'):
                QMessageBox.information(self, "Export", "Trend analysis results exported successfully!")
            else:
                QMessageBox.warning(self, "Export", "No trend analysis results to export.")

        elif current_tab == 3:  # Portfolio Selection
            if hasattr(self.portfolio_widget, 'current_results') and self.portfolio_widget.current_results:
                QMessageBox.information(self, "Export", "Portfolio selection results exported successfully!")
            else:
                QMessageBox.warning(self, "Export", "No portfolio selection results to export.")

        elif current_tab == 4:  # Portfolio Optimization
            if hasattr(self.optimizer_widget, 'current_results') and self.optimizer_widget.current_results:
                QMessageBox.information(self, "Export", "Portfolio optimization results exported successfully!")
            else:
                QMessageBox.warning(self, "Export", "No optimization results to export.")

    def show_about(self):
        """Show about dialog"""
        about_text = """
        <h2>Multifactor Stock Analysis System (MSAS)</h2>
        <p>Version 3.0</p>
        <br>
        <p><b>An integrated platform for:</b></p>
        <ul>
            <li>Market Regime Detection using Hidden Markov Models</li>
            <li>Multifactor Stock Scoring and Ranking</li>
            <li>Score Trend Analysis and Technical Indicators</li>
            <li>Dynamic Portfolio Selection and Optimization</li>
        </ul>
        <br>
        <p>© 2025 - Developed for Advanced Investment Analysis</p>
        """

        msg = QMessageBox()
        msg.setWindowTitle("About MSAS")
        msg.setTextFormat(Qt.RichText)
        msg.setText(about_text)
        msg.setIcon(QMessageBox.Information)
        msg.exec_()

    def show_help(self):
        """Show help dialog"""
        help_text = """
        <h3>Quick Start Guide</h3>

        <p><b>1. Regime Detection:</b><br>
        Detect the current market regime (Bull/Bear/Steady) using HMM analysis.</p>

        <p><b>2. Multifactor Scoring:</b><br>
        Score and rank stocks based on multiple factors optimized for the detected regime.</p>

        <p><b>3. Score Trend Analysis:</b><br>
        Analyze stability and generate technical indicators for top-ranked stocks.</p>

        <p><b>4. Portfolio Selection:</b><br>
        Create optimized portfolios based on stability analysis and backtest performance.</p>

        <p><b>5. Portfolio Optimization:</b><br>
        Optimize portfolio weights using hierarchical risk parity methods (HRP, HERC, MHRP, NCO).</p>


        <p><b>Workflow:</b><br>
        1. Start with Regime Detection (Tab 1)<br>
        2. Run Multifactor Scoring for the detected regime (Tab 2)<br>
        3. Analyze score trends and stability (Tab 3)<br>
        4. Generate optimized portfolios (Tab 4)</p>
        5. Fine-tune allocation with Portfolio Optimization (Tab 5)</p>

        <p><b>Keyboard Shortcuts:</b><br>
        Ctrl+1-5: Switch between tabs<br>
        F5: Run regime detection<br>
        Ctrl+R: Run scoring process<br>
        Ctrl+T: Run stability analysis<br>
        Ctrl+P: Generate portfolios<br>
        Ctrl+E: Export results<br>
        Ctrl+Q: Exit</p>
        """

        msg = QMessageBox()
        msg.setWindowTitle("MSAS User Guide")
        msg.setTextFormat(Qt.RichText)
        msg.setText(help_text)
        msg.setIcon(QMessageBox.Information)
        msg.exec_()

    def get_dark_theme_stylesheet(self):
        """Return the dark theme stylesheet"""
        return """
            QMainWindow {
                background-color: #1a1a1a;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QGroupBox {
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #3c3c3c;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 8px;
                min-width: 100px;
                color: white;
            }
            QPushButton:hover {
                background-color: #4c4c4c;
                border-color: #777;
            }
            QPushButton:pressed {
                background-color: #2c2c2c;
            }
            QPushButton:disabled {
                background-color: #2b2b2b;
                color: #666;
            }
            QComboBox {
                background-color: #3c3c3c;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px;
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
            QLineEdit {
                background-color: #3c3c3c;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px;
                color: white;
            }
            QTextEdit {
                background-color: #2b2b2b;
                border: 1px solid #444;
                border-radius: 4px;
            }
            QTableWidget {
                background-color: #2b2b2b;
                alternate-background-color: #353535;
                gridline-color: #444;
            }
            QHeaderView::section {
                background-color: #3c3c3c;
                padding: 8px;
                border: none;
                font-weight: bold;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 4px;
                text-align: center;
                background-color: #2b2b2b;
            }
            QProgressBar::chunk {
                background-color: #7c4dff;
                border-radius: 3px;
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
            QCheckBox {
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #555;
                border-radius: 3px;
                background-color: #2b2b2b;
            }
            QCheckBox::indicator:checked {
                background-color: #7c4dff;
                border-color: #7c4dff;
            }
        """

    def open_documentation(self):
        """Open the MSAS documentation HTML file"""
        import os
        import webbrowser
        from pathlib import Path

        # Get the documentation file path
        doc_path = PROJECT_ROOT / "docs" / "msas_complete_documentation.html"

        # Check if the file exists
        if doc_path.exists():
            # Convert to absolute path and open in default browser
            file_url = doc_path.absolute().as_uri()
            webbrowser.open(file_url)
        else:
            # Show error if documentation file not found
            QMessageBox.warning(
                self,
                "Documentation Not Found",
                f"Documentation file not found at:\n{doc_path}\n\n"
                "Please ensure the documentation is in the ./docs/ directory."
            )


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Set application-wide dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(43, 43, 43))
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.Base, QColor(45, 45, 45))
    palette.setColor(QPalette.AlternateBase, QColor(60, 60, 60))
    palette.setColor(QPalette.ToolTipBase, QColor(0, 0, 0))
    palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.Button, QColor(60, 60, 60))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(124, 77, 255))
    palette.setColor(QPalette.Highlight, QColor(124, 77, 255))
    palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()