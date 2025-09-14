# portfolio_optimizer_widget.py
import sys
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd()
sys.path.append(str(PROJECT_ROOT))


class PortfolioOptimizerThread(QThread):
    """Thread for running portfolio optimization"""

    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, tickers, shares, method, lookback_days=180):
        super().__init__()
        self.tickers = tickers
        self.shares = shares  # None if not provided
        self.method = method
        self.lookback_days = lookback_days

    def run(self):
        """Run the portfolio optimization"""
        try:
            import yfinance as yf
            import riskfolio as rp
            from sklearn.covariance import LedoitWolf
            from scipy.cluster.hierarchy import linkage
            from scipy.spatial.distance import squareform
            from scipy.optimize import minimize
            import matplotlib.pyplot as plt
            from PIL import Image

            self.progress_update.emit(10)
            self.status_update.emit("Fetching price data...")

            # Fetch price data
            prices, dropped = self.fetch_prices(self.tickers, self.lookback_days)

            if dropped:
                self.status_update.emit(f"Warning: Skipping {dropped} due to missing data")
                self.tickers = [t for t in self.tickers if t not in dropped]

            if len(self.tickers) < 2:
                self.error_occurred.emit("Need at least 2 tickers with valid data")
                return

            returns = prices.pct_change().dropna()

            self.progress_update.emit(30)
            self.status_update.emit(f"Optimizing portfolio using {self.method}...")

            # Estimate covariance matrix
            lw = LedoitWolf()
            cov_matrix = lw.fit(returns).covariance_
            cov_df = pd.DataFrame(cov_matrix, index=returns.columns, columns=returns.columns)

            # Run optimization based on method
            if self.method == "HRP":
                weights = self.optimize_hrp(returns, cov_df)
            elif self.method == "HERC":
                weights = self.optimize_herc(returns, cov_df)
            elif self.method == "MHRP":
                weights = self.optimize_mhrp(returns, cov_df)
            elif self.method == "NCO":
                weights = self.optimize_nco(returns, cov_df)
            else:
                self.error_occurred.emit(f"Unknown optimization method: {self.method}")
                return

            self.progress_update.emit(60)

            # Calculate current portfolio weights if shares provided
            current_weights = None
            rebalance_info = None

            if self.shares is not None:
                self.status_update.emit("Calculating current portfolio weights...")
                current_weights, rebalance_info = self.calculate_current_weights(
                    self.tickers, self.shares, weights
                )

            self.progress_update.emit(80)
            self.status_update.emit("Generating visualizations...")

            # Generate visualizations
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = PROJECT_ROOT / "output" / "Portfolio_Optimization_Results"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save optimized weights
            optimized_img = output_dir / f"{self.method}_{date_str}.png"
            self.plot_donut_chart(weights, f"{self.method} Portfolio", optimized_img)

            combined_img = None
            if current_weights is not None:
                # Save current weights
                current_img = output_dir / f"current_{date_str}.png"
                self.plot_donut_chart(current_weights, "Current Portfolio", current_img)

                # Combine images
                combined_img = output_dir / f"combined_{date_str}.png"
                self.merge_images(current_img, optimized_img, combined_img)

            # Calculate portfolio metrics
            metrics = self.calculate_metrics(weights, cov_df, returns)

            self.progress_update.emit(100)

            # Prepare results
            results = {
                'success': True,
                'method': self.method,
                'weights': weights.to_dict(),
                'current_weights': current_weights.to_dict() if current_weights is not None else None,
                'metrics': metrics,
                'rebalance_info': rebalance_info,
                'optimized_img': str(optimized_img),
                'combined_img': str(combined_img) if combined_img else None,
                'tickers': self.tickers
            }

            self.result_ready.emit(results)

        except Exception as e:
            self.error_occurred.emit(f"Optimization failed: {str(e)}\n{traceback.format_exc()}")

    def fetch_prices(self, tickers, lookback_days, max_retries=3):
        """Fetch historical price data with retry logic"""
        import yfinance as yf
        import time

        tries = 0
        missing_cols = []

        while tries < max_retries and tickers:
            start_date = (datetime.now() - pd.Timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            data = yf.download(tickers, start=start_date, auto_adjust=True, progress=False)

            if len(tickers) == 1:
                close = data['Close'].to_frame(columns=[tickers[0]])
            else:
                close = data['Close']

            missing_cols = [t for t in tickers if t not in close.columns]

            if not missing_cols:
                break
            else:
                tries += 1
                tickers = [t for t in tickers if t not in missing_cols]
                time.sleep(1)

        prices = close.ffill(limit=2).dropna(axis=0, how="any")
        return prices, missing_cols

    def optimize_hrp(self, returns, cov):
        """Standard HRP optimization"""
        import riskfolio as rp

        port = rp.HCPortfolio(returns=returns)
        port.cov = cov

        weights = port.optimization(
            model='HRP',
            codependence='pearson',
            linkage='ward'
        )

        return weights.squeeze()

    def optimize_herc(self, returns, cov):
        """HERC optimization"""
        import riskfolio as rp

        port = rp.HCPortfolio(returns=returns)
        port.cov = cov

        weights = port.optimization(
            model='HERC',
            codependence='pearson',
            linkage='ward',
            max_k=10
        )

        return weights.squeeze()

    def optimize_mhrp(self, returns, cov):
        """Modified HRP with equal volatility weighting"""
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform

        # Build correlation matrix for clustering
        cor = returns.corr(method='spearman')
        distance = np.sqrt(0.5 * (1 - cor))
        link = linkage(squareform(distance), method='ward')

        # Get ticker order after clustering
        ticker_order = self.get_quasi_diag(link)
        sorted_tickers = [returns.columns[i] for i in ticker_order]

        # Apply equal volatility recursive bisection
        weights = self.recursive_bisection_equal_volatility(cov, sorted_tickers)
        weights = weights / weights.sum()

        return weights

    def optimize_nco(self, returns, cov):
        """Nested Clustered Optimization"""
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        from scipy.optimize import minimize

        # Hierarchical clustering
        cor = returns.corr(method='spearman')
        distance = np.sqrt(0.5 * (1 - cor))
        link = linkage(squareform(distance), method='ward')

        # Create 5 clusters
        n_clusters = min(5, len(returns.columns) // 2)
        clusters = fcluster(link, n_clusters, criterion='maxclust')

        # Optimize within each cluster (minimum variance)
        cluster_weights = {}
        for i in range(1, n_clusters + 1):
            cluster_assets = [returns.columns[j] for j in range(len(clusters)) if clusters[j] == i]
            if len(cluster_assets) > 0:
                sub_cov = cov.loc[cluster_assets, cluster_assets].values
                n = len(cluster_assets)

                # Minimum variance optimization
                def objective(w):
                    return w @ sub_cov @ w

                constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
                bounds = [(0, 1) for _ in range(n)]
                w0 = np.ones(n) / n

                result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)

                if result.success:
                    for j, asset in enumerate(cluster_assets):
                        cluster_weights[asset] = result.x[j] / n_clusters
                else:
                    # Fallback to equal weight
                    for asset in cluster_assets:
                        cluster_weights[asset] = 1.0 / (n * n_clusters)

        return pd.Series(cluster_weights)

    def get_quasi_diag(self, link):
        """Get quasi-diagonal ordering from linkage matrix"""
        link = np.asarray(link, order='c')
        n = link.shape[0] + 1
        order = np.zeros(n, dtype=int)

        def _get_leaves(i):
            if i < n:
                return [i]
            else:
                left = int(link[i - n, 0])
                right = int(link[i - n, 1])
                return _get_leaves(left) + _get_leaves(right)

        return _get_leaves(2 * n - 2)

    def recursive_bisection_equal_volatility(self, cov, sort_ix):
        """Recursive bisection with equal volatility weighting"""

        def get_cluster_volatility(cov, cluster_items):
            sub_cov = cov.loc[cluster_items, cluster_items]
            vol = np.sqrt(np.diag(sub_cov))
            inv_vol = 1 / vol
            weights = inv_vol / inv_vol.sum()
            cluster_vol = np.sqrt(np.dot(weights, np.dot(sub_cov.values, weights)))
            return cluster_vol

        w = pd.Series(1.0, index=sort_ix)
        clusters = [sort_ix]

        while len(clusters) > 0:
            clusters = [c[j:k] for c in clusters
                        for j, k in ((0, len(c) // 2), (len(c) // 2, len(c)))
                        if len(c) > 1]

            for i in range(0, len(clusters), 2):
                if i + 1 < len(clusters):
                    cluster0 = clusters[i]
                    cluster1 = clusters[i + 1]

                    vol0 = get_cluster_volatility(cov, cluster0)
                    vol1 = get_cluster_volatility(cov, cluster1)

                    total_vol = vol0 + vol1
                    w0 = vol1 / total_vol if total_vol > 0 else 0.5
                    w1 = 1 - w0

                    w[cluster0] *= w0
                    w[cluster1] *= w1

        # Final within-cluster equal volatility allocation
        vol = np.sqrt(np.diag(cov))
        inv_vol = 1 / vol
        w = w * inv_vol
        w = w / w.sum()

        return w

    def calculate_current_weights(self, tickers, shares, optimal_weights):
        """Calculate current portfolio weights and rebalancing needs"""
        import yfinance as yf

        # Get current prices
        prices = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1d")
                if not hist.empty:
                    prices[ticker] = hist['Close'].iloc[-1]
            except:
                prices[ticker] = 100  # Default price if fetch fails

        # Calculate current weights
        values = {t: shares.get(t, 0) * prices.get(t, 100) for t in tickers}
        total_value = sum(values.values())

        if total_value > 0:
            current_weights = pd.Series({t: values[t] / total_value for t in tickers})
        else:
            current_weights = pd.Series({t: 1.0 / len(tickers) for t in tickers})

        # Calculate rebalancing needs
        rebalance_info = self.calculate_rebalance_info(current_weights, optimal_weights)

        return current_weights, rebalance_info

    def calculate_rebalance_info(self, current_weights, optimal_weights, threshold=0.05):
        """Calculate rebalancing recommendations"""
        differences = {}
        needs_rebalance = False

        for ticker in current_weights.index:
            current = current_weights.get(ticker, 0)
            optimal = optimal_weights.get(ticker, 0)
            diff = optimal - current

            if abs(diff) > threshold:
                needs_rebalance = True
                differences[ticker] = diff

        if needs_rebalance:
            message = "Rebalance needed: " + ", ".join(
                [f"{t}: {d:+.1%}" for t, d in sorted(differences.items(), key=lambda x: abs(x[1]), reverse=True)]
            )
        else:
            message = "No rebalance needed"

        return {
            'needs_rebalance': needs_rebalance,
            'message': message,
            'differences': differences
        }

    def calculate_metrics(self, weights, cov, returns):
        """Calculate portfolio metrics"""
        portfolio_return = (returns @ weights).mean() * 252
        portfolio_vol = np.sqrt(weights @ cov @ weights) * np.sqrt(252)
        sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0

        return {
            'annual_return': portfolio_return,
            'annual_volatility': portfolio_vol,
            'sharpe_ratio': sharpe,
            'max_weight': weights.max(),
            'min_weight': weights.min(),
            'effective_n': 1 / (weights ** 2).sum()
        }

    def plot_donut_chart(self, weights, title, filepath):
        """Create donut chart visualization"""
        import matplotlib.pyplot as plt

        # Sort weights for better visualization
        weights_sorted = weights.sort_values(ascending=False)

        # Prepare labels with percentages
        labels = [f"{t} {w * 100:.1f}%" for t, w in weights_sorted.items()]
        sizes = weights_sorted.values

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))

        # Use a nice color palette
        colors = plt.cm.Set3(np.linspace(0, 1, len(weights_sorted)))

        # Create donut chart
        wedges, texts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            startangle=90,
            wedgeprops=dict(width=0.5, edgecolor='white')
        )

        # Improve text appearance
        plt.setp(texts, size=9, weight="bold")

        # Add title
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # Equal aspect ratio ensures circular donut
        ax.axis('equal')

        plt.tight_layout()
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()

    def merge_images(self, img1_path, img2_path, output_path):
        """Merge two images side by side"""
        from PIL import Image

        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        # Get dimensions
        width1, height1 = img1.size
        width2, height2 = img2.size

        # Create new image
        total_width = width1 + width2 + 20  # 20px gap
        max_height = max(height1, height2)

        new_img = Image.new('RGB', (total_width, max_height), 'white')

        # Paste images
        new_img.paste(img1, (0, (max_height - height1) // 2))
        new_img.paste(img2, (width1 + 20, (max_height - height2) // 2))

        new_img.save(output_path)


class PortfolioOptimizerWidget(QWidget):
    """Widget for portfolio optimization"""

    def __init__(self):
        super().__init__()
        self.optimizer_thread = None
        self.current_results = None
        self.portfolio_data = {}
        self.init_ui()

    def init_ui(self):
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)

        # Title
        title_label = QLabel("Portfolio Optimization")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont("Arial", 18, QFont.Bold)
        title_label.setFont(title_font)
        main_layout.addWidget(title_label)

        # Subtitle
        subtitle_label = QLabel("Optimize portfolio allocation using hierarchical risk parity methods")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("color: #888;")
        main_layout.addWidget(subtitle_label)

        # Create sections
        sections_layout = QHBoxLayout()
        sections_layout.setSpacing(15)

        # Left section: Method selection and portfolio input
        left_section = self.create_input_section()
        sections_layout.addWidget(left_section, 1)

        # Right section: Results display
        right_section = self.create_results_section()
        sections_layout.addWidget(right_section, 1)

        main_layout.addLayout(sections_layout)

        # Progress bar
        self.create_progress_bar()
        main_layout.addWidget(self.progress_group)

        self.setLayout(main_layout)

    def create_input_section(self):
        """Create input section for portfolio configuration"""
        group = QGroupBox("Portfolio Configuration")
        layout = QVBoxLayout()

        # Method selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Optimization Method:"))

        self.method_combo = QComboBox()
        self.method_combo.addItems(["MHRP", "HRP", "HERC", "NCO"])
        self.method_combo.setCurrentText("MHRP")
        self.method_combo.setToolTip(
            "MHRP: Modified HRP with equal volatility\n"
            "HRP: Hierarchical Risk Parity\n"
            "HERC: Hierarchical Equal Risk Contribution\n"
            "NCO: Nested Clustered Optimization"
        )
        method_layout.addWidget(self.method_combo)
        method_layout.addStretch()
        layout.addLayout(method_layout)

        # Include shares checkbox
        self.shares_checkbox = QCheckBox("Include share quantities for rebalancing analysis")
        self.shares_checkbox.stateChanged.connect(self.on_shares_checkbox_changed)
        layout.addWidget(self.shares_checkbox)

        # Portfolio input area
        portfolio_label = QLabel("Portfolio Holdings:")
        layout.addWidget(portfolio_label)

        # Table for ticker and shares input
        self.portfolio_table = QTableWidget()
        self.portfolio_table.setColumnCount(2)
        self.portfolio_table.setHorizontalHeaderLabels(["Ticker", "Shares"])
        self.portfolio_table.horizontalHeader().setStretchLastSection(True)
        self.portfolio_table.setMinimumHeight(200)

        # Initially hide shares column
        self.portfolio_table.setColumnHidden(1, True)

        # Add some initial rows
        self.portfolio_table.setRowCount(10)

        layout.addWidget(self.portfolio_table)

        # Add/Remove row buttons
        row_buttons_layout = QHBoxLayout()

        add_row_btn = QPushButton("Add Row")
        add_row_btn.clicked.connect(self.add_portfolio_row)
        row_buttons_layout.addWidget(add_row_btn)

        remove_row_btn = QPushButton("Remove Row")
        remove_row_btn.clicked.connect(self.remove_portfolio_row)
        row_buttons_layout.addWidget(remove_row_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.clear_portfolio)
        row_buttons_layout.addWidget(clear_btn)

        row_buttons_layout.addStretch()
        layout.addLayout(row_buttons_layout)

        # File operations
        file_layout = QHBoxLayout()

        self.save_btn = QPushButton("Save Portfolio")
        self.save_btn.clicked.connect(self.save_portfolio)
        file_layout.addWidget(self.save_btn)

        self.load_btn = QPushButton("Load Portfolio")
        self.load_btn.clicked.connect(self.load_portfolio)
        file_layout.addWidget(self.load_btn)

        file_layout.addStretch()
        layout.addLayout(file_layout)

        # Lookback period
        lookback_layout = QHBoxLayout()
        lookback_layout.addWidget(QLabel("Lookback Period (days):"))

        self.lookback_spin = QSpinBox()
        self.lookback_spin.setMinimum(30)
        self.lookback_spin.setMaximum(1000)
        self.lookback_spin.setValue(180)
        self.lookback_spin.setSingleStep(30)
        lookback_layout.addWidget(self.lookback_spin)

        lookback_layout.addStretch()
        layout.addLayout(lookback_layout)

        # Run optimization button
        self.run_btn = QPushButton("Run Optimization")
        self.run_btn.clicked.connect(self.run_optimization)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 15px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        layout.addWidget(self.run_btn)

        group.setLayout(layout)
        return group

    def create_results_section(self):
        """Create results display section"""
        group = QGroupBox("Optimization Results")
        layout = QVBoxLayout()

        # Results text area
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(200)
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
        self.results_text.setPlaceholderText("Optimization results will appear here...")
        layout.addWidget(self.results_text)

        # Rebalancing message
        self.rebalance_label = QLabel("")
        self.rebalance_label.setStyleSheet("font-weight: bold; padding: 10px;")
        self.rebalance_label.setWordWrap(True)
        layout.addWidget(self.rebalance_label)

        # Image display area
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(300)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
        """)
        self.image_label.setText("Visualization will appear here")

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

        # Export button
        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        layout.addWidget(self.export_btn)

        group.setLayout(layout)
        return group

    def create_progress_bar(self):
        """Create progress bar"""
        self.progress_group = QGroupBox()
        layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                text-align: center;
                background-color: #2b2b2b;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(self.status_label)

        self.progress_group.setLayout(layout)
        self.progress_group.hide()

    def on_shares_checkbox_changed(self, state):
        """Handle shares checkbox state change"""
        show_shares = state == Qt.Checked
        self.portfolio_table.setColumnHidden(1, not show_shares)

    def add_portfolio_row(self):
        """Add a new row to the portfolio table"""
        row_count = self.portfolio_table.rowCount()
        self.portfolio_table.insertRow(row_count)

    def remove_portfolio_row(self):
        """Remove the selected row from the portfolio table"""
        current_row = self.portfolio_table.currentRow()
        if current_row >= 0:
            self.portfolio_table.removeRow(current_row)

    def clear_portfolio(self):
        """Clear all portfolio entries"""
        self.portfolio_table.clearContents()
        self.portfolio_table.setRowCount(10)

    def get_portfolio_data(self):
        """Extract portfolio data from the table"""
        tickers = []
        shares = {} if self.shares_checkbox.isChecked() else None

        for row in range(self.portfolio_table.rowCount()):
            ticker_item = self.portfolio_table.item(row, 0)
            if ticker_item and ticker_item.text().strip():
                ticker = ticker_item.text().strip().upper()
                tickers.append(ticker)

                if self.shares_checkbox.isChecked():
                    shares_item = self.portfolio_table.item(row, 1)
                    if shares_item and shares_item.text().strip():
                        try:
                            shares[ticker] = float(shares_item.text().strip())
                        except ValueError:
                            shares[ticker] = 0
                    else:
                        shares[ticker] = 0

        return tickers, shares

    def save_portfolio(self):
        """Save portfolio to file"""
        try:
            tickers, shares = self.get_portfolio_data()

            if not tickers:
                QMessageBox.warning(self, "Warning", "No portfolio data to save")
                return

            # Create config directory
            config_dir = PROJECT_ROOT / "config" / "Portfolio_Lists"
            config_dir.mkdir(parents=True, exist_ok=True)

            # Get filename from user
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Portfolio",
                str(config_dir / f"portfolio_{datetime.now().strftime('%Y%m%d')}.json"),
                "JSON Files (*.json)"
            )

            if filename:
                portfolio_data = {
                    'tickers': tickers,
                    'shares': shares if shares else {},
                    'date_saved': datetime.now().isoformat(),
                    'has_shares': self.shares_checkbox.isChecked()
                }

                with open(filename, 'w') as f:
                    json.dump(portfolio_data, f, indent=2)

                QMessageBox.information(self, "Success", f"Portfolio saved to {Path(filename).name}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save portfolio: {str(e)}")

    def load_portfolio(self):
        """Load portfolio from file"""
        try:
            config_dir = PROJECT_ROOT / "config" / "Portfolio_Lists"
            config_dir.mkdir(parents=True, exist_ok=True)

            filename, _ = QFileDialog.getOpenFileName(
                self,
                "Load Portfolio",
                str(config_dir),
                "JSON Files (*.json)"
            )

            if filename:
                with open(filename, 'r') as f:
                    portfolio_data = json.load(f)

                # Clear existing data
                self.clear_portfolio()

                # Load tickers and shares
                tickers = portfolio_data.get('tickers', [])
                shares = portfolio_data.get('shares', {})
                has_shares = portfolio_data.get('has_shares', False)

                # Set checkbox state
                self.shares_checkbox.setChecked(has_shares)

                # Populate table
                self.portfolio_table.setRowCount(max(len(tickers), 10))

                for i, ticker in enumerate(tickers):
                    self.portfolio_table.setItem(i, 0, QTableWidgetItem(ticker))
                    if has_shares and ticker in shares:
                        self.portfolio_table.setItem(i, 1, QTableWidgetItem(str(shares[ticker])))

                QMessageBox.information(self, "Success", f"Portfolio loaded from {Path(filename).name}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load portfolio: {str(e)}")

    def run_optimization(self):
        """Run portfolio optimization"""
        try:
            # Get portfolio data
            tickers, shares = self.get_portfolio_data()

            if len(tickers) < 2:
                QMessageBox.warning(self, "Warning", "Please enter at least 2 tickers")
                return

            # Disable UI elements
            self.run_btn.setEnabled(False)
            self.progress_group.show()
            self.progress_bar.setValue(0)
            self.status_label.setText("Starting optimization...")

            # Clear previous results
            self.results_text.clear()
            self.rebalance_label.clear()
            self.image_label.clear()
            self.image_label.setText("Processing...")

            # Create and start optimization thread
            self.optimizer_thread = PortfolioOptimizerThread(
                tickers=tickers,
                shares=shares,
                method=self.method_combo.currentText(),
                lookback_days=self.lookback_spin.value()
            )

            # Connect signals
            self.optimizer_thread.progress_update.connect(self.update_progress)
            self.optimizer_thread.status_update.connect(self.update_status)
            self.optimizer_thread.result_ready.connect(self.handle_results)
            self.optimizer_thread.error_occurred.connect(self.handle_error)

            # Start thread
            self.optimizer_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start optimization: {str(e)}")
            self.run_btn.setEnabled(True)
            self.progress_group.hide()

    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)

    def update_status(self, message):
        """Update status message"""
        self.status_label.setText(message)

    def handle_results(self, results):
        """Handle optimization results"""
        try:
            self.current_results = results

            # Display weights
            weights_text = f"=== {results['method']} Optimization Results ===\n\n"
            weights_text += "Optimized Weights:\n"
            weights_text += "-" * 30 + "\n"

            for ticker, weight in sorted(results['weights'].items(), key=lambda x: x[1], reverse=True):
                weights_text += f"{ticker:<8} {weight * 100:>6.2f}%\n"

            # Add metrics
            metrics = results['metrics']
            weights_text += "\n" + "=" * 30 + "\n"
            weights_text += "Portfolio Metrics:\n"
            weights_text += "-" * 30 + "\n"
            weights_text += f"Annual Return:    {metrics['annual_return'] * 100:>6.2f}%\n"
            weights_text += f"Annual Volatility: {metrics['annual_volatility'] * 100:>6.2f}%\n"
            weights_text += f"Sharpe Ratio:     {metrics['sharpe_ratio']:>6.3f}\n"
            weights_text += f"Max Weight:       {metrics['max_weight'] * 100:>6.2f}%\n"
            weights_text += f"Min Weight:       {metrics['min_weight'] * 100:>6.2f}%\n"
            weights_text += f"Effective N:      {metrics['effective_n']:>6.1f}\n"

            # Display current weights if available
            if results['current_weights']:
                weights_text += "\n" + "=" * 30 + "\n"
                weights_text += "Current Weights:\n"
                weights_text += "-" * 30 + "\n"

                for ticker, weight in sorted(results['current_weights'].items(), key=lambda x: x[1], reverse=True):
                    weights_text += f"{ticker:<8} {weight * 100:>6.2f}%\n"

            self.results_text.setText(weights_text)

            # Display rebalancing info
            if results['rebalance_info']:
                rebalance_info = results['rebalance_info']
                if rebalance_info['needs_rebalance']:
                    self.rebalance_label.setText(rebalance_info['message'])
                    self.rebalance_label.setStyleSheet("color: #ff9800; font-weight: bold; padding: 10px;")
                else:
                    self.rebalance_label.setText(rebalance_info['message'])
                    self.rebalance_label.setStyleSheet("color: #4CAF50; font-weight: bold; padding: 10px;")

            # Display image
            img_path = results['combined_img'] if results['combined_img'] else results['optimized_img']
            if img_path and Path(img_path).exists():
                pixmap = QPixmap(img_path)
                # Scale to fit while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(
                    800, 600,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)

            # Enable export button
            self.export_btn.setEnabled(True)

            # Show success message
            self.status_label.setText("Optimization completed successfully!")
            self.status_label.setStyleSheet("color: #4CAF50; font-style: italic;")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to display results: {str(e)}")

        finally:
            self.run_btn.setEnabled(True)
            QTimer.singleShot(3000, self.progress_group.hide)

    def handle_error(self, error_message):
        """Handle optimization error"""
        QMessageBox.critical(self, "Optimization Error", error_message)
        self.run_btn.setEnabled(True)
        self.progress_group.hide()
        self.status_label.setText("Error occurred")
        self.status_label.setStyleSheet("color: #f44336; font-style: italic;")

    def export_results(self):
        """Export optimization results"""
        try:
            if not self.current_results:
                QMessageBox.warning(self, "Warning", "No results to export")
                return

            # Get export directory
            export_dir = PROJECT_ROOT / "output" / "Portfolio_Optimization_Results"
            export_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = export_dir / f"{self.current_results['method']}_results_{timestamp}.json"

            # Prepare export data
            export_data = {
                'method': self.current_results['method'],
                'tickers': self.current_results['tickers'],
                'weights': self.current_results['weights'],
                'current_weights': self.current_results['current_weights'],
                'metrics': self.current_results['metrics'],
                'rebalance_info': self.current_results['rebalance_info'],
                'optimization_date': datetime.now().isoformat(),
                'images': {
                    'optimized': self.current_results['optimized_img'],
                    'combined': self.current_results['combined_img']
                }
            }

            # Save to file
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)

            # Also export weights to CSV
            csv_filename = export_dir / f"{self.current_results['method']}_weights_{timestamp}.csv"
            weights_df = pd.DataFrame([
                {'Ticker': ticker, 'Weight': weight}
                for ticker, weight in self.current_results['weights'].items()
            ])
            weights_df.to_csv(csv_filename, index=False)

            QMessageBox.information(
                self,
                "Export Complete",
                f"Results exported to:\n{filename.name}\n{csv_filename.name}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export results: {str(e)}")


# Test the widget standalone
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    # Create and show widget
    widget = PortfolioOptimizerWidget()
    widget.setWindowTitle("Portfolio Optimizer")
    widget.resize(1200, 800)
    widget.show()

    sys.exit(app.exec_())