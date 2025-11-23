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

# Import macro view engine
try:
    from src.portfolio_optimization.macro_view_engine import MacroViewEngine
    MACRO_VIEW_AVAILABLE = True
except ImportError:
    MACRO_VIEW_AVAILABLE = False
    print("Warning: Macro view engine not available")


class PortfolioOptimizerThread(QThread):
    """Thread for running portfolio optimization"""

    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, tickers, shares, method, lookback_days=180, use_regime_filter=True, use_macro_views=False):
        super().__init__()
        self.tickers = tickers
        self.shares = shares  # None if not provided
        self.method = method
        self.lookback_days = lookback_days
        self.use_regime_filter = use_regime_filter
        self.use_macro_views = use_macro_views
        self.current_regime_info = None
        self.regime_periods = None
        self.macro_views = None

    def run(self):
        """Run the portfolio optimization"""
        try:
            import yfinance as yf
            import riskfolio as rp
            from sklearn.covariance import LedoitWolf
            from scipy.cluster.hierarchy import linkage
            from scipy.spatial.distance import squareform
            from scipy.optimize import minimize
            # import matplotlib.pyplot as plt
            from PIL import Image

            # STORE ORIGINAL TICKER ORDER AS INSTANCE VARIABLE
            self.original_tickers = self.tickers.copy()

            # Load regime detection data if regime filtering is enabled
            if self.use_regime_filter:
                self.progress_update.emit(5)
                self.status_update.emit("Loading regime detection data...")
                self.load_regime_data()

            # Generate macro views if enabled
            if self.use_macro_views and MACRO_VIEW_AVAILABLE:
                self.progress_update.emit(8)
                self.status_update.emit("Generating macro views...")
                self.generate_macro_views()

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

            # Apply regime filtering if enabled
            if self.use_regime_filter and self.current_regime_info:
                current_regime_name = self.current_regime_info.get('regime_name', 'Unknown')
                self.status_update.emit(f"Applying regime filter for '{current_regime_name}'...")
                returns = self.filter_returns_by_regime(returns, current_regime_name)

            self.progress_update.emit(30)
            self.status_update.emit(f"Optimizing portfolio using {self.method}...")

            # Estimate covariance matrix (now on regime-filtered data)
            lw = LedoitWolf()
            cov_matrix = lw.fit(returns).covariance_
            cov_df = pd.DataFrame(cov_matrix, index=returns.columns, columns=returns.columns)

            # Apply Black-Litterman adjustment if macro views are enabled
            if self.use_macro_views and self.macro_views:
                cov_df = self.apply_black_litterman(cov_df)

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
                'tickers': self.tickers,
                'regime_info': {
                    'used_regime_filter': self.use_regime_filter,
                    'regime_name': self.current_regime_info.get('regime_name', 'N/A') if self.current_regime_info else 'N/A',
                    'regime_confidence': self.current_regime_info.get('confidence', 0) if self.current_regime_info else 0
                } if self.use_regime_filter else None,
                'macro_views_info': {
                    'used_macro_views': self.use_macro_views,
                    'num_views': len(self.macro_views) if self.macro_views else 0,
                    'views': self.macro_views
                } if self.use_macro_views else None
            }

            self.result_ready.emit(results)

        except Exception as e:
            self.error_occurred.emit(f"Optimization failed: {str(e)}\n{traceback.format_exc()}")

    def load_regime_data(self):
        """Load current regime and historical regime periods"""
        try:
            # Load current regime analysis
            regime_file = PROJECT_ROOT / "output" / "Regime_Detection_Analysis" / "current_regime_analysis.json"
            if regime_file.exists():
                with open(regime_file, 'r') as f:
                    data = json.load(f)
                    self.current_regime_info = data.get('regime_detection', {})
                    self.status_update.emit(f"Current regime: {self.current_regime_info.get('regime_name', 'Unknown')}")
            else:
                self.status_update.emit("Warning: No regime data found, using all historical data")
                self.use_regime_filter = False
                return

            # Load historical regime periods
            periods_file = PROJECT_ROOT / "output" / "Regime_Detection_Results" / "regime_periods.csv"
            if periods_file.exists():
                self.regime_periods = pd.read_csv(periods_file)
                self.regime_periods['start_date'] = pd.to_datetime(self.regime_periods['start_date'])
                self.regime_periods['end_date'] = pd.to_datetime(self.regime_periods['end_date'])
                self.status_update.emit(f"Loaded {len(self.regime_periods)} regime periods")
            else:
                self.status_update.emit("Warning: No regime periods found, using all historical data")
                self.use_regime_filter = False

        except Exception as e:
            self.status_update.emit(f"Warning: Could not load regime data - {str(e)}")
            self.use_regime_filter = False

    def filter_returns_by_regime(self, returns, current_regime_name):
        """
        Filter returns to include only periods matching the current regime

        Args:
            returns: DataFrame of returns
            current_regime_name: Name of current regime (e.g., 'Steady Growth')

        Returns:
            Filtered returns DataFrame
        """
        if not self.use_regime_filter or self.regime_periods is None:
            return returns

        try:
            # Get all periods matching the current regime
            matching_periods = self.regime_periods[
                self.regime_periods['regime_name'] == current_regime_name
            ]

            if matching_periods.empty:
                self.status_update.emit(f"Warning: No matching periods for regime '{current_regime_name}', using all data")
                return returns

            # Create a mask for dates in matching regimes
            regime_mask = pd.Series(False, index=returns.index)

            for _, period in matching_periods.iterrows():
                # Include data from this regime period
                period_mask = (returns.index >= period['start_date']) & (returns.index <= period['end_date'])
                regime_mask = regime_mask | period_mask

            filtered_returns = returns[regime_mask]

            # Calculate statistics
            original_days = len(returns)
            filtered_days = len(filtered_returns)
            percentage = (filtered_days / original_days * 100) if original_days > 0 else 0

            self.status_update.emit(
                f"Regime filter: Using {filtered_days} days ({percentage:.1f}%) from '{current_regime_name}' periods"
            )

            # Ensure we have enough data
            if filtered_days < 60:  # Minimum 60 days for reliable covariance
                self.status_update.emit(
                    f"Warning: Only {filtered_days} days in regime '{current_regime_name}', using all data"
                )
                return returns

            return filtered_returns

        except Exception as e:
            self.status_update.emit(f"Warning: Regime filtering failed - {str(e)}, using all data")
            return returns

    def generate_macro_views(self):
        """Generate macro views using the macro view engine"""
        try:
            engine = MacroViewEngine()
            self.macro_views = engine.generate_views(self.tickers, lookback_days=90)

            if self.macro_views:
                num_views = len(self.macro_views)
                self.status_update.emit(f"Generated {num_views} macro views")

                # Show summary
                for ticker, view in self.macro_views.items():
                    self.status_update.emit(
                        f"  {ticker}: {view['expected_return']*100:+.1f}% "
                        f"(Confidence: {view['confidence']:.0%})"
                    )
            else:
                self.status_update.emit("No macro views generated")
                self.use_macro_views = False

        except Exception as e:
            self.status_update.emit(f"Warning: Could not generate macro views - {str(e)}")
            self.use_macro_views = False

    def apply_black_litterman(self, cov_df):
        """
        Apply Black-Litterman model to incorporate macro views

        Args:
            cov_df: Covariance matrix DataFrame

        Returns:
            Adjusted covariance matrix DataFrame
        """
        try:
            from pypfopt import BlackLittermanModel

            self.status_update.emit("Applying Black-Litterman model...")

            # Prepare views dictionary
            viewdict = {}
            for ticker in self.tickers:
                if ticker in self.macro_views:
                    viewdict[ticker] = self.macro_views[ticker]['expected_return']

            if not viewdict:
                self.status_update.emit("No views to apply, skipping Black-Litterman")
                return cov_df

            self.status_update.emit(f"Applying {len(viewdict)} macro views via Black-Litterman")

            # Get market cap weights as prior (or use equal weights as fallback)
            try:
                # Try to get market caps
                import yfinance as yf
                market_caps = {}
                for ticker in self.tickers:
                    try:
                        stock = yf.Ticker(ticker)
                        mc = stock.info.get('marketCap', None)
                        if mc and mc > 0:
                            market_caps[ticker] = mc
                    except:
                        pass

                if market_caps:
                    total_mc = sum(market_caps.values())
                    market_prior = pd.Series({t: mc/total_mc for t, mc in market_caps.items()})
                else:
                    # Equal weight fallback
                    market_prior = pd.Series(1.0/len(self.tickers), index=self.tickers)

            except:
                # Equal weight fallback
                market_prior = pd.Series(1.0/len(self.tickers), index=self.tickers)

            # Create Black-Litterman model
            bl = BlackLittermanModel(
                cov_matrix=cov_df,
                absolute_views=viewdict,
                pi="market",  # Use market equilibrium
                market_caps=market_prior if 'market_caps' in locals() else None
            )

            # Get posterior (BL-adjusted) covariance
            bl_cov = bl.bl_cov()

            # Convert back to DataFrame
            bl_cov_df = pd.DataFrame(bl_cov, index=cov_df.index, columns=cov_df.columns)

            self.status_update.emit("Black-Litterman adjustment complete")

            return bl_cov_df

        except ImportError:
            self.status_update.emit("Warning: pypfopt not available, skipping Black-Litterman")
            return cov_df
        except Exception as e:
            self.status_update.emit(f"Warning: Black-Litterman failed - {str(e)}, using original covariance")
            return cov_df

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

        weights = weights.squeeze()

        # Reindex to original ticker order
        if hasattr(self, 'original_tickers'):
            valid_tickers = [t for t in self.original_tickers if t in weights.index]
            weights = weights.reindex(valid_tickers).fillna(0)
        else:
            weights = weights.reindex(self.returns.columns).fillna(0)

        return weights

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

        weights = weights.squeeze()

        # Reindex to original ticker order
        if hasattr(self, 'original_tickers'):
            valid_tickers = [t for t in self.original_tickers if t in weights.index]
            weights = weights.reindex(valid_tickers).fillna(0)
        else:
            weights = weights.reindex(self.returns.columns).fillna(0)

        return weights

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

        # CRITICAL: Reindex to original ticker order (like the old script)
        if hasattr(self, 'original_tickers'):
            # Only keep tickers that are in our current data (some might have been dropped)
            valid_tickers = [t for t in self.original_tickers if t in weights.index]
            weights = weights.reindex(valid_tickers).fillna(0)
        else:
            weights = weights.reindex(self.returns.columns).fillna(0)

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

        # Convert to Series and reindex to original order
        weights = pd.Series(cluster_weights)

        if hasattr(self, 'original_tickers'):
            valid_tickers = [t for t in self.original_tickers if t in weights.index]
            weights = weights.reindex(valid_tickers).fillna(0)
        else:
            weights = weights.reindex(self.returns.columns).fillna(0)

        return weights

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

        # if total_value > 0:
        #     current_weights = pd.Series({t: values[t] / total_value for t in tickers})
        # else:
        #     current_weights = pd.Series({t: 1.0 / len(tickers) for t in tickers})

        if total_value > 0:
            # Create Series with explicit index order to preserve ticker order
            current_weights = pd.Series([values[t] / total_value for t in tickers], index=tickers)
        else:
            current_weights = pd.Series([1.0 / len(tickers) for t in tickers], index=tickers)

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
        """Create donut chart visualization with fixed 600x600 size"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # IMPORTANT: Don't filter or reorder - preserve exact input order
        weights_to_plot = weights.copy()

        # Prepare labels and sizes - preserve order
        labels = []
        sizes = []
        for ticker in weights_to_plot.index:
            if weights_to_plot[ticker] > 1e-4:  # Only show if weight > 0.01%
                labels.append(f"{ticker} {weights_to_plot[ticker] * 100:.1f}%")
                sizes.append(weights_to_plot[ticker])

        # Create figure with EXACT size
        # fig = plt.figure(figsize=(6, 6), dpi=100)
        fig = plt.figure(figsize=(6, 6), dpi=100, facecolor='white')
        ax = fig.add_subplot(111)

        # Use consistent color palette
        colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))

        # Create donut chart
        wedges, texts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            startangle=90,
            wedgeprops=dict(width=0.35, edgecolor='white'),
            textprops=dict(color='black', weight='bold', size=9)  # EXPLICITLY SET TEXT COLOR
        )

        # Text appearance
        # plt.setp(texts, size=9, weight="bold")

        # Add white center circle
        centre_circle = plt.Circle((0, 0), 0.70, fc='white', linewidth=0)
        ax.add_artist(centre_circle)

        # Title
        # ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10, color='black')

        # Set background
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')

        # Equal aspect ratio
        ax.axis('equal')

        # Save with exact size
        plt.tight_layout()
        # fig.savefig(filepath, dpi=100, facecolor='white')
        fig.savefig(filepath, dpi=100, facecolor='white', edgecolor='none')
        plt.close(fig)

    def merge_images(self, img1_path, img2_path, output_path):
        """Merge two 600x600 images with a 5-pixel black divider"""
        from PIL import Image

        TARGET_SIZE = (600, 600)
        DIVIDER_W = 5
        DIVIDER_COLOR = (0, 0, 0)  # Black

        # Open and convert images
        left = Image.open(img1_path).convert("RGB")
        right = Image.open(img2_path).convert("RGB")

        # Resize to exact target size if needed
        if left.size != TARGET_SIZE:
            left = left.resize(TARGET_SIZE, Image.LANCZOS)
        if right.size != TARGET_SIZE:
            right = right.resize(TARGET_SIZE, Image.LANCZOS)

        # Create canvas with white background
        w, h = TARGET_SIZE
        total_width = 2 * w + DIVIDER_W
        canvas = Image.new("RGB", (total_width, h), (255, 255, 255))  # White background

        # Paste left image
        canvas.paste(left, (0, 0))

        # Create and paste black divider
        for x in range(w, w + DIVIDER_W):
            for y in range(h):
                canvas.putpixel((x, y), DIVIDER_COLOR)

        # Paste right image
        canvas.paste(right, (w + DIVIDER_W, 0))

        # Save
        canvas.save(output_path)

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
        title_label = QLabel("Advanced Portfolio Optimization System")
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

        # Regime-aware optimization checkbox
        self.regime_filter_checkbox = QCheckBox("Use regime-aware optimization (filter by current market regime)")
        self.regime_filter_checkbox.setChecked(True)  # Enabled by default
        self.regime_filter_checkbox.setToolTip(
            "When enabled, uses only historical data from periods matching the current market regime.\n"
            "This makes the optimizer forward-looking and regime-aware."
        )
        layout.addWidget(self.regime_filter_checkbox)

        # Macro views checkbox
        self.macro_views_checkbox = QCheckBox("Use macro views with Black-Litterman (analyst + sector momentum)")
        self.macro_views_checkbox.setChecked(False)  # Disabled by default (experimental)
        self.macro_views_checkbox.setToolTip(
            "When enabled, generates objective macro views from:\n"
            "- Analyst consensus (price targets)\n"
            "- Sector momentum analysis\n"
            "- Factor-specific views (oil prices for energy stocks)\n"
            "These views are incorporated via Black-Litterman model."
        )
        if not MACRO_VIEW_AVAILABLE:
            self.macro_views_checkbox.setEnabled(False)
            self.macro_views_checkbox.setToolTip("Macro view engine not available")
        layout.addWidget(self.macro_views_checkbox)

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
                lookback_days=self.lookback_spin.value(),
                use_regime_filter=self.regime_filter_checkbox.isChecked(),
                use_macro_views=self.macro_views_checkbox.isChecked()
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

            # Show regime information if available
            if results.get('regime_info') and results['regime_info'].get('used_regime_filter'):
                regime_info = results['regime_info']
                weights_text += "Regime-Aware Optimization:\n"
                weights_text += "-" * 30 + "\n"
                weights_text += f"Market Regime: {regime_info['regime_name']}\n"
                weights_text += f"Confidence: {regime_info['regime_confidence']*100:.1f}%\n"
                weights_text += f"Data Filter: Using only '{regime_info['regime_name']}' periods\n"
                weights_text += "\n"

            # Show macro views information if available
            if results.get('macro_views_info') and results['macro_views_info'].get('used_macro_views'):
                macro_info = results['macro_views_info']
                weights_text += "Black-Litterman Macro Views:\n"
                weights_text += "-" * 30 + "\n"
                weights_text += f"Views Generated: {macro_info['num_views']}\n"

                if macro_info.get('views'):
                    weights_text += "\nMacro View Summary:\n"
                    for ticker, view in macro_info['views'].items():
                        weights_text += f"  {ticker}: {view['expected_return']*100:+.1f}% "
                        weights_text += f"({view['confidence']:.0%} confidence)\n"
                        weights_text += f"    Source: {view['source']}\n"

                weights_text += "\n"

            weights_text += "Optimized Weights:\n"
            weights_text += "-" * 30 + "\n"

            # Show weights in original ticker order (from results['tickers'])
            for ticker in results['tickers']:
                if ticker in results['weights']:
                    weight = results['weights'][ticker]
                    if weight > 0.0001:  # Only show non-zero weights
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

            # Display current weights if available (also in original order)
            if results.get('current_weights'):
                weights_text += "\n" + "=" * 30 + "\n"
                weights_text += "Current Weights:\n"
                weights_text += "-" * 30 + "\n"

                for ticker in results['tickers']:
                    if ticker in results['current_weights']:
                        weight = results['current_weights'][ticker]
                        if weight > 0.0001:
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