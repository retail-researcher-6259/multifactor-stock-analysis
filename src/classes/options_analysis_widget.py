# src/classes/options_analysis_widget.py
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
import re
import json
import subprocess
import os
from pathlib import Path
import pandas as pd
import pytz
from datetime import datetime, time as dt_time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

SCRIPT_DIR = PROJECT_ROOT / "src" / "options_flow_analysis"
MASSIVE_CONFIG = PROJECT_ROOT / "config" / "massive_config.json"
OPTIONS_LISTS_DIR = PROJECT_ROOT / "output" / "Options_Ranked_Lists"
ANALYSIS_DIR = PROJECT_ROOT / "output" / "Options_Flow_Analysis"

# Output patterns that indicate a Massive API credential problem
API_KEY_ERROR_PATTERNS = [
    "401 Unauthorized",
    "check your API key",
    "No valid api_key",
    "Missing S3 credentials",
    "S3 listing failed",
    "403 Forbidden",
]


def validate_massive_config(require_s3=False):
    """Check massive_config.json before launching any Massive-based script.

    Returns (ok, message)."""
    if not MASSIVE_CONFIG.exists():
        return False, f"Config file not found:\n{MASSIVE_CONFIG}"
    try:
        with open(MASSIVE_CONFIG) as f:
            cfg = json.load(f)
    except Exception as e:
        return False, f"Config file is not valid JSON:\n{e}"

    api_key = cfg.get("api_key", "")
    if not api_key or api_key in ("YOUR_API_KEY_HERE", "aaa"):
        return False, ("No valid 'api_key' in massive_config.json.\n"
                       "Paste your Massive.com (Polygon.io) REST API key.")
    if require_s3:
        s3_id = cfg.get("s3_access_key_id", "")
        s3_secret = cfg.get("s3_secret_access_key", "")
        if (not s3_id or "YOUR_S3" in s3_id
                or not s3_secret or "YOUR_S3" in s3_secret):
            return False, ("Historical backfill needs the S3 flat-file "
                           "credentials.\nFill in 's3_access_key_id' and "
                           "'s3_secret_access_key' in massive_config.json\n"
                           "(from the Massive dashboard, S3/Flat Files "
                           "section; NOT the REST key).")
    return True, "OK"


class ScriptRunnerThread(QThread):
    """Runs one options script as a subprocess with live output parsing."""
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    api_key_error = pyqtSignal(str)

    def __init__(self, script_name, args=None):
        super().__init__()
        self.script_name = script_name
        self.args = args or []

    def run(self):
        try:
            script_path = SCRIPT_DIR / self.script_name
            if not script_path.exists():
                self.error_occurred.emit(f"Script not found: {script_path}")
                return

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            cmd = [sys.executable, str(script_path)] + self.args

            self.status_update.emit(f"Launching {self.script_name}...")
            proc = subprocess.Popen(
                cmd, cwd=str(SCRIPT_DIR),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", env=env,
            )

            lines = []
            saved_path = None
            total_sessions = 0
            sessions_done = 0
            api_error_line = None

            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                lines.append(line)
                self.status_update.emit(line[-120:])

                for pat in API_KEY_ERROR_PATTERNS:
                    if pat in line:
                        api_error_line = line
                        break

                # Daily scorer progress: "  123/1027 scored ..."
                m = re.search(r"(\d+)/(\d+) scored", line)
                if m:
                    self.progress_update.emit(
                        int(100 * int(m.group(1)) / max(1, int(m.group(2)))))
                # Backfill progress: sessions announced then processed
                m = re.search(r"(\d+) trading sessions found", line)
                if m:
                    total_sessions = int(m.group(1))
                if total_sessions and re.match(
                        r"\d{4}-\d{2}-\d{2}:", line):
                    sessions_done += 1
                    self.progress_update.emit(
                        min(99, int(100 * sessions_done / total_sessions)))
                # Saved file line (all four scripts)
                m = re.search(r"saved to: (.+\.csv)\s*$", line,
                              re.IGNORECASE)
                if m:
                    saved_path = m.group(1).strip()

            proc.wait()
            output_text = "\n".join(lines)

            if api_error_line:
                self.api_key_error.emit(api_error_line)
                return
            if proc.returncode != 0:
                tail = "\n".join(lines[-15:])
                self.error_occurred.emit(
                    f"{self.script_name} exited with code "
                    f"{proc.returncode}:\n\n{tail}")
                return

            self.progress_update.emit(100)
            self.result_ready.emit({
                "success": True,
                "script": self.script_name,
                "saved_path": saved_path,
                "console": output_text,
            })

        except Exception as e:
            self.error_occurred.emit(f"Thread execution failed: {str(e)}")


class OptionsAnalysisWidget(QWidget):
    """Main widget for the options flow analysis system"""

    def __init__(self):
        super().__init__()
        self.fetch_thread = None
        self.recommender_thread = None
        self.analyzer_thread = None
        self.current_recommendations = None
        self.current_flow_report = None
        self.recommended_regime = "Steady Growth"
        self.init_ui()
        self.refresh_data_stats()

    # ------------------------------------------------------------- UI
    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)

        title_label = QLabel("Options Flow Analysis System")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        main_layout.addWidget(title_label)

        subtitle_label = QLabel(
            "Near-the-money options flow scoring, positioning trends, and "
            "confluence recommendations (Massive.com data)")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("color: #888;")
        main_layout.addWidget(subtitle_label)

        sections_layout = QHBoxLayout()
        sections_layout.setSpacing(15)
        sections_layout.addWidget(self.create_fetch_section(), 1)
        sections_layout.addWidget(self.create_analysis_section(), 1)
        main_layout.addLayout(sections_layout)

        main_layout.addWidget(self.create_results_section())

        self.create_progress_bars()
        main_layout.addWidget(self.fetch_progress_group)
        main_layout.addWidget(self.analysis_progress_group)

        self.setLayout(main_layout)

    def create_fetch_section(self):
        """Upper-left: options data fetching (daily snapshot or backfill)"""
        group = QGroupBox("Options Data Fetching")
        layout = QVBoxLayout()

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Data Source:"))
        self.fetch_mode_combo = QComboBox()
        self.fetch_mode_combo.addItems([
            "Today's Data (Chain Snapshot)",
            "Historical Data (Flat File Backfill)",
        ])
        self.fetch_mode_combo.currentTextChanged.connect(
            self.on_fetch_mode_changed)
        mode_layout.addWidget(self.fetch_mode_combo)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        # Date range (backfill mode only)
        self.date_range_widget = QWidget()
        date_layout = QGridLayout()
        date_layout.setContentsMargins(0, 0, 0, 0)
        date_layout.addWidget(QLabel("Start Date:"), 0, 0)
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDate(QDate.currentDate().addDays(-30))
        self.start_date_edit.setDisplayFormat("yyyy-MM-dd")
        date_layout.addWidget(self.start_date_edit, 0, 1)
        date_layout.addWidget(QLabel("End Date:"), 1, 0)
        self.end_date_edit = QDateEdit()
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDate(QDate.currentDate())
        self.end_date_edit.setDisplayFormat("yyyy-MM-dd")
        date_layout.addWidget(self.end_date_edit, 1, 1)
        date_layout.setColumnStretch(2, 1)
        self.date_range_widget.setLayout(date_layout)
        self.date_range_widget.setVisible(False)
        layout.addWidget(self.date_range_widget)

        self.run_fetch_btn = QPushButton("Fetch Options Data")
        self.run_fetch_btn.clicked.connect(self.run_fetch)
        self.run_fetch_btn.setStyleSheet("""
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
        layout.addWidget(self.run_fetch_btn)

        self.fetch_info = QLabel()
        self.fetch_info.setStyleSheet("""
            QLabel {
                background-color: #3c3c3c;
                padding: 10px;
                border-radius: 5px;
                color: #cccccc;
                font-size: 11px;
            }
        """)
        self.fetch_info.setWordWrap(True)
        layout.addWidget(self.fetch_info)
        self.on_fetch_mode_changed(self.fetch_mode_combo.currentText())

        stats_label = QLabel("Data Inventory")
        stats_label.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(stats_label)

        self.data_stats = {
            "sessions": QLabel("Stored Sessions: --"),
            "latest": QLabel("Latest Session: --"),
            "oi": QLabel("Sessions with Open Interest: --"),
        }
        for label in self.data_stats.values():
            label.setStyleSheet("color: #cccccc; font-size: 12px;")
            layout.addWidget(label)

        layout.addStretch()
        group.setLayout(layout)
        return group

    def create_analysis_section(self):
        """Upper-right: regime selection and analysis runs"""
        group = QGroupBox("Flow Analysis and Recommendations")
        layout = QVBoxLayout()

        regime_layout = QHBoxLayout()
        regime_layout.addWidget(QLabel("Target Regime:"))
        self.regime_combo = QComboBox()
        self.regime_combo.addItems(
            ["Steady Growth", "Strong Bull", "Crisis/Bear"])
        self.regime_combo.setCurrentText(self.recommended_regime)
        regime_layout.addWidget(self.regime_combo)

        self.regime_recommendation_label = QLabel("(Recommended)")
        self.regime_recommendation_label.setStyleSheet(
            "color: #4CAF50; font-style: italic;")
        regime_layout.addWidget(self.regime_recommendation_label)
        regime_layout.addStretch()
        layout.addLayout(regime_layout)

        self.run_analyzer_checkbox = QCheckBox(
            "Also run daily flow analyzer (events / OI builds report)")
        self.run_analyzer_checkbox.setChecked(True)
        layout.addWidget(self.run_analyzer_checkbox)

        self.run_analysis_btn = QPushButton("Run Leading Factor Analysis")
        self.run_analysis_btn.clicked.connect(self.run_analysis)
        self.run_analysis_btn.setStyleSheet("""
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
        layout.addWidget(self.run_analysis_btn)

        analysis_info = QLabel(
            "- Confluence of stability rank, 5-session OI trend, and "
            "abnormal-volume events\n"
            "- Tiers: STRONG BUY / BUY / HOLD / SELL / STRONG SELL / "
            "AMBIGUOUS\n"
            "- Results saved to output/Options_Flow_Analysis/"
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

        stats_label = QLabel("Analysis Statistics")
        stats_label.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(stats_label)

        self.analysis_stats = {
            "universe": QLabel("Stocks Analyzed: --"),
            "strong_buy": QLabel("Strong Buy: --"),
            "sells": QLabel("Sell / Strong Sell: --"),
            "session": QLabel("Options Session: --"),
        }
        for label in self.analysis_stats.values():
            label.setStyleSheet("color: #cccccc; font-size: 12px;")
            layout.addWidget(label)

        layout.addStretch()
        group.setLayout(layout)
        return group

    def create_results_section(self):
        """Lower: recommendations table / flow report table / console"""
        group = QGroupBox("Analysis Results")
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 30)

        view_buttons_layout = QHBoxLayout()
        view_buttons_layout.setSpacing(5)

        self.recommendations_view_btn = QPushButton("Recommendations")
        self.recommendations_view_btn.setCheckable(True)
        self.recommendations_view_btn.setChecked(True)
        self.recommendations_view_btn.clicked.connect(
            lambda: self.switch_results_view("recommendations"))
        view_buttons_layout.addWidget(self.recommendations_view_btn)

        self.flow_view_btn = QPushButton("Flow Report")
        self.flow_view_btn.setCheckable(True)
        self.flow_view_btn.clicked.connect(
            lambda: self.switch_results_view("flow"))
        view_buttons_layout.addWidget(self.flow_view_btn)

        self.console_view_btn = QPushButton("Console Log")
        self.console_view_btn.setCheckable(True)
        self.console_view_btn.clicked.connect(
            lambda: self.switch_results_view("console"))
        view_buttons_layout.addWidget(self.console_view_btn)

        view_buttons_layout.addStretch()
        layout.addLayout(view_buttons_layout)

        table_style = """
            QTableWidget {
                background-color: #2b2b2b;
                alternate-background-color: #353535;
                gridline-color: #444;
            }
        """
        self.recommendations_table = QTableWidget()
        self.recommendations_table.setAlternatingRowColors(True)
        self.recommendations_table.setStyleSheet(table_style)
        self.recommendations_table.verticalHeader().setVisible(False)
        layout.addWidget(self.recommendations_table)

        self.flow_table = QTableWidget()
        self.flow_table.setAlternatingRowColors(True)
        self.flow_table.setStyleSheet(table_style)
        self.flow_table.verticalHeader().setVisible(False)
        self.flow_table.hide()
        layout.addWidget(self.flow_table)

        self.console_area = QTextEdit()
        self.console_area.setReadOnly(True)
        self.console_area.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                border: 1px solid #444;
                color: #cccccc;
                font-family: 'Courier New', monospace;
                font-size: 11px;
            }
        """)
        self.console_area.setPlaceholderText(
            "Script console output will appear here...")
        self.console_area.hide()
        layout.addWidget(self.console_area)

        self.results_summary_label = QLabel(
            "Fetch options data, then run the leading factor analysis.")
        self.results_summary_label.setStyleSheet(
            "color: #888; font-size: 11px;")
        layout.addWidget(self.results_summary_label)

        group.setLayout(layout)
        return group

    def create_progress_bars(self):
        self.fetch_progress_group = QGroupBox("Data Fetch Progress")
        self.fetch_progress_group.setVisible(False)
        fetch_layout = QVBoxLayout()
        self.fetch_progress_bar = QProgressBar()
        self.fetch_progress_bar.setRange(0, 100)
        fetch_layout.addWidget(self.fetch_progress_bar)
        self.fetch_progress_status = QLabel("Ready...")
        self.fetch_progress_status.setAlignment(Qt.AlignCenter)
        fetch_layout.addWidget(self.fetch_progress_status)
        self.fetch_progress_group.setLayout(fetch_layout)

        self.analysis_progress_group = QGroupBox("Analysis Progress")
        self.analysis_progress_group.setVisible(False)
        analysis_layout = QVBoxLayout()
        self.analysis_progress_bar = QProgressBar()
        self.analysis_progress_bar.setRange(0, 100)
        analysis_layout.addWidget(self.analysis_progress_bar)
        self.analysis_progress_status = QLabel("Ready...")
        self.analysis_progress_status.setAlignment(Qt.AlignCenter)
        analysis_layout.addWidget(self.analysis_progress_status)
        self.analysis_progress_group.setLayout(analysis_layout)

    # ------------------------------------------------- helpers / slots
    def on_fetch_mode_changed(self, mode_text):
        is_backfill = "Historical" in mode_text
        self.date_range_widget.setVisible(is_backfill)
        if is_backfill:
            self.fetch_info.setText(
                "- Downloads flat files from Massive S3 (needs S3 "
                "credentials in massive_config.json)\n"
                "- One ranked list per U.S. trading session; weekends and "
                "holidays are skipped automatically\n"
                "- Existing session files are kept, only missing ones are "
                "generated\n"
                "- Note: open interest is NOT available historically "
                "(volume metrics only)")
        else:
            self.fetch_info.setText(
                "- Fetches the most recent completed U.S. session via "
                "chain snapshots (includes open interest)\n"
                "- Run after the U.S. close; ~1,030 API calls, about 5-10 "
                "minutes\n"
                "- Output: output/Options_Ranked_Lists/"
                "options_<session>.csv")

    def set_recommended_regime(self, regime):
        """Set the recommended regime from the regime detection tab"""
        print(f" Options tab: updating recommended regime to {regime}")
        self.recommended_regime = regime
        regime_mapping = {
            "Steady Growth": "Steady Growth",
            "Strong Bull": "Strong Bull",
            "Crisis/Bear": "Crisis/Bear",
            "Crisis": "Crisis/Bear",
            "Bear": "Crisis/Bear",
        }
        mapped = regime_mapping.get(regime, regime)
        items = [self.regime_combo.itemText(i)
                 for i in range(self.regime_combo.count())]
        if mapped in items:
            self.regime_combo.setCurrentText(mapped)
            self.regime_recommendation_label.setText(
                "(Recommended by Regime Detection)")
            self.regime_recommendation_label.setStyleSheet(
                "color: #4CAF50; font-style: italic; font-weight: bold;")
        else:
            self.regime_recommendation_label.setText(
                "(Regime not in analysis list)")
            self.regime_recommendation_label.setStyleSheet(
                "color: #ff9800; font-style: italic;")

    def selected_regime_key(self):
        return (self.regime_combo.currentText()
                .replace(" ", "_").replace("/", "_"))

    def refresh_data_stats(self):
        """Update the data inventory labels from the output folder."""
        try:
            files = sorted(OPTIONS_LISTS_DIR.glob("options_*.csv"))
            dates = set()
            oi_dates = set()
            for f in files:
                m = re.search(r"(\d{8})", f.name)
                if not m:
                    continue
                dates.add(m.group(1))
            self.data_stats["sessions"].setText(
                f"Stored Sessions: {len(dates)}")
            self.data_stats["latest"].setText(
                f"Latest Session: {max(dates) if dates else '--'}")
            # OI check only on the newest few files (cheap)
            for f in files[-8:]:
                try:
                    df = pd.read_csv(f, usecols=["CallOI"], nrows=50)
                    if df["CallOI"].notna().any():
                        m = re.search(r"(\d{8})", f.name)
                        if m:
                            oi_dates.add(m.group(1))
                except Exception:
                    pass
            self.data_stats["oi"].setText(
                f"Sessions with Open Interest: {len(oi_dates)}+ recent")
        except Exception as e:
            print(f"Data stats refresh failed: {e}")

    def show_api_key_error(self, detail):
        QMessageBox.critical(
            self, "Massive API Key Invalid",
            "A Massive.com API call failed with an authentication or "
            "permission error.\n\n"
            f"Details: {detail}\n\n"
            "Please check the credentials in:\n"
            f"{MASSIVE_CONFIG}\n\n"
            "- 'api_key': REST API key (daily fetch and analysis)\n"
            "- 's3_access_key_id' / 's3_secret_access_key': Flat Files "
            "keys (historical backfill)")

    # ----------------------------------------------------- fetch flow
    @staticmethod
    def us_session_incomplete():
        """True on U.S. weekdays between the ~3:30 AM ET snapshot reset
        and the end of the session: fetching in this window captures
        partial (or, right after the open on a delayed feed, empty)
        volumes for the in-progress session."""
        now_et = datetime.now(pytz.timezone("US/Eastern"))
        if now_et.weekday() >= 5:
            return False
        return dt_time(3, 30) <= now_et.time() < dt_time(16, 15)

    def run_fetch(self):
        is_backfill = "Historical" in self.fetch_mode_combo.currentText()

        ok, message = validate_massive_config(require_s3=is_backfill)
        if not ok:
            QMessageBox.critical(self, "Massive API Key Invalid", message)
            return

        if not is_backfill and self.us_session_incomplete():
            reply = QMessageBox.question(
                self, "U.S. Session Not Complete",
                "Today's U.S. session has not finished (or has just "
                "opened on a 15-minute delayed feed).\n\n"
                "Fetching now captures PARTIAL or EMPTY volumes for the "
                "in-progress session. The reliable window is after the "
                "U.S. close: 16:15 ET, i.e. 04:15/05:15 Taipei, or simply "
                "the evening (19:00 Taipei, the daemon's schedule).\n\n"
                "Fetch anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return

        if is_backfill:
            start = self.start_date_edit.date().toPyDate()
            end = self.end_date_edit.date().toPyDate()
            if start > end:
                QMessageBox.warning(self, "Input Error",
                                    "Start date must be before end date!")
                return
            args = ["--start", start.strftime("%Y-%m-%d"),
                    "--end", end.strftime("%Y-%m-%d")]
            script = "options_backfill_scorer.py"
        else:
            args = []
            script = "options_daily_scorer.py"

        self.fetch_progress_group.setVisible(True)
        self.fetch_progress_bar.setValue(0)
        self.fetch_progress_status.setText(f"Starting {script}...")
        self.run_fetch_btn.setEnabled(False)
        self.run_fetch_btn.setText("Fetching...")

        self.fetch_thread = ScriptRunnerThread(script, args)
        self.fetch_thread.progress_update.connect(
            self.fetch_progress_bar.setValue)
        self.fetch_thread.status_update.connect(
            self.fetch_progress_status.setText)
        self.fetch_thread.result_ready.connect(self.handle_fetch_results)
        self.fetch_thread.error_occurred.connect(self.handle_fetch_error)
        self.fetch_thread.api_key_error.connect(self.handle_api_key_error)
        self.fetch_thread.finished.connect(self.on_fetch_finished)
        self.fetch_thread.start()

    def handle_fetch_results(self, results):
        self.console_area.setText(results.get("console", ""))
        saved = results.get("saved_path")
        if saved:
            self.results_summary_label.setText(
                f"Data fetch complete. Saved: {saved}")
        else:
            self.results_summary_label.setText(
                "Data fetch complete (see Console Log for details).")
        self.refresh_data_stats()

    def handle_fetch_error(self, message):
        self.console_area.setText(message)
        QMessageBox.critical(self, "Options Data Fetch Error",
                             f"Error occurred:\n\n{message}")

    def handle_api_key_error(self, detail):
        self.show_api_key_error(detail)

    def on_fetch_finished(self):
        self.fetch_progress_group.setVisible(False)
        self.run_fetch_btn.setEnabled(True)
        self.run_fetch_btn.setText("Fetch Options Data")

    # -------------------------------------------------- analysis flow
    def run_analysis(self):
        regime_key = self.selected_regime_key()

        self.analysis_progress_group.setVisible(True)
        self.analysis_progress_bar.setValue(10)
        self.analysis_progress_status.setText(
            "Running leading factor recommender...")
        self.run_analysis_btn.setEnabled(False)
        self.run_analysis_btn.setText("Analyzing...")

        self.recommender_thread = ScriptRunnerThread(
            "leading_factor_recommender.py", ["--regime", regime_key])
        self.recommender_thread.status_update.connect(
            self.analysis_progress_status.setText)
        self.recommender_thread.result_ready.connect(
            self.handle_recommender_results)
        self.recommender_thread.error_occurred.connect(
            self.handle_analysis_error)
        self.recommender_thread.api_key_error.connect(
            self.handle_api_key_error)
        self.recommender_thread.start()

    def handle_recommender_results(self, results):
        self.analysis_progress_bar.setValue(
            50 if self.run_analyzer_checkbox.isChecked() else 100)
        console = results.get("console", "")
        self.console_area.setText(console)

        saved = results.get("saved_path")
        df = None
        if saved and Path(saved).exists():
            try:
                df = pd.read_csv(saved)
            except Exception as e:
                self.handle_analysis_error(
                    f"Could not read recommendations CSV:\n{e}")
                return
        if df is None:
            self.handle_analysis_error(
                "Recommender finished but no output CSV was found.\n"
                "Check the Console Log view.")
            return

        self.current_recommendations = {"df": df, "path": saved}
        self.display_recommendations(df)

        n = len(df)
        tiers = df["Recommendation"].str.split(" - ").str[0]
        n_sb = (tiers == "STRONG BUY").sum()
        n_sell = tiers.isin(["SELL", "STRONG SELL"]).sum()
        self.analysis_stats["universe"].setText(f"Stocks Analyzed: {n}")
        self.analysis_stats["strong_buy"].setText(f"Strong Buy: {n_sb}")
        self.analysis_stats["sells"].setText(
            f"Sell / Strong Sell: {n_sell}")
        m = re.search(r"Options session: (\d{4}-\d{2}-\d{2})", console)
        if m:
            self.analysis_stats["session"].setText(
                f"Options Session: {m.group(1)}")

        self.results_summary_label.setText(
            f"Recommendations ready: {n} stocks "
            f"({n_sb} STRONG BUY, {n_sell} SELL tier). Saved: {saved}")
        self.switch_results_view("recommendations")

        # Optionally chain the daily flow analyzer
        if self.run_analyzer_checkbox.isChecked():
            regime_key = self.selected_regime_key()
            self.analysis_progress_status.setText(
                "Running daily flow analyzer...")
            self.analyzer_thread = ScriptRunnerThread(
                "options_flow_analyzer.py", ["--regime", regime_key])
            self.analyzer_thread.status_update.connect(
                self.analysis_progress_status.setText)
            self.analyzer_thread.result_ready.connect(
                self.handle_analyzer_results)
            self.analyzer_thread.error_occurred.connect(
                self.handle_analysis_error)
            self.analyzer_thread.api_key_error.connect(
                self.handle_api_key_error)
            self.analyzer_thread.finished.connect(self.on_analysis_finished)
            self.analyzer_thread.start()
        else:
            self.on_analysis_finished()

    def handle_analyzer_results(self, results):
        self.analysis_progress_bar.setValue(100)
        console = results.get("console", "")
        self.console_area.append("\n\n" + "=" * 60 + "\n" + console)

        saved = results.get("saved_path")
        if saved and Path(saved).exists():
            try:
                df = pd.read_csv(saved)
                self.current_flow_report = {"df": df, "path": saved}
                self.display_flow_report(df)
            except Exception as e:
                print(f"Could not read flow report: {e}")

    def handle_analysis_error(self, message):
        self.console_area.setText(message)
        QMessageBox.critical(self, "Options Analysis Error",
                             f"Error occurred:\n\n{message}")
        self.on_analysis_finished()

    def on_analysis_finished(self):
        self.analysis_progress_group.setVisible(False)
        self.run_analysis_btn.setEnabled(True)
        self.run_analysis_btn.setText("Run Leading Factor Analysis")

    # -------------------------------------------------------- display
    def switch_results_view(self, view):
        self.recommendations_view_btn.setChecked(view == "recommendations")
        self.flow_view_btn.setChecked(view == "flow")
        self.console_view_btn.setChecked(view == "console")
        self.recommendations_table.setVisible(view == "recommendations")
        self.flow_table.setVisible(view == "flow")
        self.console_area.setVisible(view == "console")

    TIER_COLORS = {
        "STRONG BUY": QColor("#4CAF50"),
        "BUY": QColor("#8BC34A"),
        "HOLD": QColor("#cccccc"),
        "SELL": QColor("#FF9800"),
        "STRONG SELL": QColor("#F44336"),
        "AMBIGUOUS": QColor("#777777"),
    }

    def _fill_table(self, table, df, color_column=None):
        table.setRowCount(len(df))
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels(df.columns.tolist())

        color_col_idx = (df.columns.get_loc(color_column)
                         if color_column in df.columns else None)

        for row in range(len(df)):
            row_color = None
            if color_col_idx is not None:
                tier = str(df.iloc[row, color_col_idx]).split(" - ")[0]
                row_color = self.TIER_COLORS.get(tier)
            for col in range(len(df.columns)):
                value = df.iloc[row, col]
                if isinstance(value, float):
                    if pd.isna(value):
                        display_value = ""
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
                if len(display_value) > 25:
                    item.setToolTip(display_value)
                if row_color is not None and col == color_col_idx:
                    item.setForeground(row_color)
                table.setItem(row, col, item)

        table.resizeColumnsToContents()
        for i in range(table.columnCount()):
            table.setColumnWidth(i, max(60, min(
                table.columnWidth(i),
                320 if df.columns[i] == "Recommendation" else 160)))
        table.horizontalHeader().setStretchLastSection(False)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

    def display_recommendations(self, df):
        self._fill_table(self.recommendations_table, df,
                         color_column="Recommendation")

    def display_flow_report(self, df):
        self._fill_table(self.flow_table, df, color_column="Label")
