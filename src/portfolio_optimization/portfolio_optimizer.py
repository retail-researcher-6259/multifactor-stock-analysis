"""
Portfolio Optimizer
==================
Comprehensive portfolio optimization module supporting multiple hierarchical risk parity methods.

Methods Supported:
- HRP: Hierarchical Risk Parity (standard)
- HERC: Hierarchical Equal Risk Contribution
- MHRP: Modified Hierarchical Risk Parity (with equal volatility)
- NCO: Nested Clustered Optimization

Author: MSAS Team
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yfinance as yf
import riskfolio as rp
from sklearn.covariance import LedoitWolf
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import warnings
import time
import sys

warnings.filterwarnings('ignore')


class PortfolioOptimizer:
    """
    Unified portfolio optimizer supporting multiple hierarchical methods.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the portfolio optimizer.

        Args:
            output_dir: Directory for saving output files
        """
        self.output_dir = output_dir or Path("./output/Portfolio_Optimization_Results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Store latest results
        self.weights = None
        self.returns = None
        self.cov_matrix = None
        self.metrics = None

    def optimize(self,
                 tickers: List[str],
                 method: str = "MHRP",
                 shares: Optional[Dict[str, float]] = None,
                 lookback_days: int = 180,
                 save_images: bool = True) -> Dict:
        """
        Main optimization function.

        Args:
            tickers: List of ticker symbols
            method: Optimization method (HRP, HERC, MHRP, NCO)
            shares: Optional dictionary of current holdings {ticker: shares}
            lookback_days: Number of days for historical data
            save_images: Whether to save visualization images

        Returns:
            Dictionary containing optimization results
        """
        print(f"\n{'='*60}")
        print(f"Portfolio Optimization - {method}")
        print(f"{'='*60}\n")

        # STORE ORIGINAL TICKER ORDER AS INSTANCE VARIABLE
        self.original_tickers = tickers.copy()

        # Fetch and prepare data
        print("📊 Fetching price data...")
        prices, dropped_tickers = self._fetch_prices(tickers, lookback_days)

        if dropped_tickers:
            print(f"⚠️  Warning: Skipping {dropped_tickers} due to missing data")
            tickers = [t for t in tickers if t not in dropped_tickers]

        if len(tickers) < 2:
            raise ValueError("Need at least 2 tickers with valid data for optimization")

        # Calculate returns
        self.returns = prices.pct_change().dropna()

        # Estimate covariance matrix
        print("📈 Estimating covariance matrix (Ledoit-Wolf shrinkage)...")
        lw = LedoitWolf()
        cov_array = lw.fit(self.returns).covariance_
        self.cov_matrix = pd.DataFrame(cov_array,
                                       index=self.returns.columns,
                                       columns=self.returns.columns)

        # Run optimization based on method
        print(f"⚙️  Running {method} optimization...")
        if method == "HRP":
            self.weights = self._optimize_hrp()
        elif method == "HERC":
            self.weights = self._optimize_herc()
        elif method == "MHRP":
            self.weights = self._optimize_mhrp()
        elif method == "NCO":
            self.weights = self._optimize_nco()
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        # FORCE REINDEX TO ORIGINAL TICKER ORDER (exactly like the old script)
        # This ensures ticker order is preserved regardless of optimization method
        valid_tickers = [t for t in tickers if t in self.weights.index]
        self.weights = self.weights.reindex(valid_tickers).fillna(0)

        # Calculate metrics
        self.metrics = self._calculate_metrics()

        # Print results
        self._print_results(method)

        # Generate visualizations if requested
        images = {}
        if save_images:
            date_str = datetime.now().strftime("%m%d")

            # Save optimized portfolio image
            opt_image = self.output_dir / f"{method}_{date_str}.png"
            self._plot_donut_chart(self.weights, f"{method} Portfolio", opt_image)
            images['optimized'] = str(opt_image)
            print(f"✅ Saved optimization chart: {opt_image.name}")

            # If shares provided, create comparison
            if shares:
                current_weights, rebalance_info = self._calculate_current_weights(
                    tickers, shares
                )

                # Save current portfolio image
                current_image = self.output_dir / f"current_{date_str}.png"
                self._plot_donut_chart(current_weights, "Current Portfolio", current_image)

                # Create comparison image
                comparison_image = self.output_dir / f"comparison_{method}_{date_str}.png"
                self._merge_images(current_image, opt_image, comparison_image)
                images['comparison'] = str(comparison_image)
                print(f"✅ Saved comparison chart: {comparison_image.name}")

                # Print rebalancing suggestions
                self._print_rebalance_suggestions(current_weights, rebalance_info)

        # After optimization, reindex to original order
        self.weights = self.weights.reindex(self.original_tickers).fillna(0)

        # Prepare return dictionary
        results = {
            'method': method,
            'weights': self.weights.to_dict(),
            'metrics': self.metrics,
            'tickers': list(self.weights.index),
            'images': images
        }

        if shares:
            current_weights, rebalance_info = self._calculate_current_weights(tickers, shares)
            results['current_weights'] = current_weights.to_dict()
            results['rebalance_info'] = rebalance_info

        return results

    def _fetch_prices(self, tickers: List[str], lookback_days: int,
                     max_retries: int = 3) -> Tuple[pd.DataFrame, List[str]]:
        """
        Fetch historical price data with retry logic.
        """
        end_date = datetime.today()
        start_date = end_date - timedelta(days=int(lookback_days * 1.4))

        tries = 0
        dropped = []

        while tries < max_retries and tickers:
            data = yf.download(
                tickers,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
                threads=True
            )

            if len(tickers) == 1:
                close = data['Close'].to_frame(columns=[tickers[0]])
            else:
                close = data['Close'] if 'Close' in data else data

            # Check for missing data
            missing = [t for t in tickers if t not in close.columns]

            if not missing:
                break
            else:
                tries += 1
                dropped.extend(missing)
                tickers = [t for t in tickers if t not in missing]
                if tries < max_retries:
                    time.sleep(1)

        # Clean data
        prices = close.ffill(limit=2).dropna(axis=0, how="any")

        return prices, list(set(dropped))

    def _optimize_hrp(self) -> pd.Series:
        """
        Standard Hierarchical Risk Parity optimization.
        """
        # Hierarchical clustering
        cor = self.returns.corr()
        distance = np.sqrt(0.5 * (1 - cor))
        link = linkage(squareform(distance), method='ward')

        # Quasi-diagonalization
        ticker_order = self._get_quasi_diag(link)
        sorted_tickers = [self.returns.columns[i] for i in ticker_order]

        # Recursive bisection
        weights = self._recursive_bisection_hrp(sorted_tickers)

        # Normalize
        weights = weights / weights.sum()

        # Reindex to original ticker order
        if hasattr(self, 'original_tickers'):
            valid_tickers = [t for t in self.original_tickers if t in weights.index]
            weights = weights.reindex(valid_tickers).fillna(0)
        else:
            weights = weights.reindex(self.returns.columns).fillna(0)

        return weights

    def _optimize_herc(self) -> pd.Series:
        """
        Hierarchical Equal Risk Contribution optimization.
        """
        # Use riskfolio-lib for HERC
        port = rp.HCPortfolio(returns=self.returns)
        port.cov = self.cov_matrix

        weights = port.optimization(
            model='HERC',
            codependence='pearson',
            linkage='ward',
            max_k=10  # Maximum number of clusters
        )

        weights = weights.squeeze()

        # Reindex to original ticker order
        if hasattr(self, 'original_tickers'):
            valid_tickers = [t for t in self.original_tickers if t in weights.index]
            weights = pd.Series([weights.get(t, 0) for t in valid_tickers], index=valid_tickers)

        return weights

    def _optimize_mhrp(self) -> pd.Series:
        """
        Modified HRP with equal volatility weighting.
        """
        # Hierarchical clustering with Spearman correlation
        cor = self.returns.corr(method='spearman')
        distance = np.sqrt(0.5 * (1 - cor))
        link = linkage(squareform(distance), method='ward')

        # Quasi-diagonalization
        ticker_order = self._get_quasi_diag(link)
        sorted_tickers = [self.returns.columns[i] for i in ticker_order]

        # Recursive bisection with equal volatility
        weights = self._recursive_bisection_mhrp(sorted_tickers)

        # Normalize and reindex
        weights = weights / weights.sum()

        # CRITICAL: Reindex to original ticker order (like the old script)
        if hasattr(self, 'original_tickers'):
            # Only keep tickers that are in our current data (some might have been dropped)
            valid_tickers = [t for t in self.original_tickers if t in weights.index]
            weights = weights.reindex(valid_tickers).fillna(0)
        else:
            weights = weights.reindex(self.returns.columns).fillna(0)

        return weights

    def _optimize_nco(self) -> pd.Series:
        """
        Nested Clustered Optimization.
        """
        # Hierarchical clustering
        cor = self.returns.corr(method='spearman')
        distance = np.sqrt(0.5 * (1 - cor))
        link = linkage(squareform(distance), method='ward')

        # Determine optimal number of clusters (max 5 or n_assets/2)
        n_clusters = min(5, len(self.returns.columns) // 2)

        # Form clusters
        clusters = fcluster(link, n_clusters, criterion='maxclust')

        # Optimize within each cluster (minimum variance)
        cluster_weights = {}
        inter_cluster_weights = {}

        for i in range(1, n_clusters + 1):
            cluster_assets = [self.returns.columns[j]
                            for j in range(len(clusters)) if clusters[j] == i]

            if len(cluster_assets) > 0:
                # Get sub-covariance matrix
                sub_cov = self.cov_matrix.loc[cluster_assets, cluster_assets].values
                n = len(cluster_assets)

                # Minimum variance optimization within cluster
                def objective(w):
                    return w @ sub_cov @ w

                constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
                bounds = [(0, 1) for _ in range(n)]
                w0 = np.ones(n) / n

                result = minimize(objective, w0, method='SLSQP',
                                bounds=bounds, constraints=constraints)

                if result.success:
                    intra_weights = result.x
                else:
                    # Fallback to equal weight
                    intra_weights = np.ones(n) / n

                # Store intra-cluster weights
                for j, asset in enumerate(cluster_assets):
                    cluster_weights[asset] = intra_weights[j]

                # Calculate cluster variance for inter-cluster allocation
                cluster_var = intra_weights @ sub_cov @ intra_weights
                inter_cluster_weights[i] = 1.0 / (cluster_var + 1e-8)

        # Normalize inter-cluster weights
        total_inter = sum(inter_cluster_weights.values())
        for i in inter_cluster_weights:
            inter_cluster_weights[i] /= total_inter

        # Combine intra and inter cluster weights
        final_weights = {}
        for i in range(1, n_clusters + 1):
            cluster_assets = [self.returns.columns[j]
                            for j in range(len(clusters)) if clusters[j] == i]
            for asset in cluster_assets:
                final_weights[asset] = (cluster_weights[asset] *
                                       inter_cluster_weights[i])

        # Convert to Series and reindex to original order
        weights = pd.Series(final_weights)

        if hasattr(self, 'original_tickers'):
            valid_tickers = [t for t in self.original_tickers if t in weights.index]
            weights = weights.reindex(valid_tickers).fillna(0)
        else:
            weights = weights.reindex(self.returns.columns).fillna(0)

        return weights

    def _recursive_bisection_hrp(self, sorted_tickers: List[str]) -> pd.Series:
        """
        Standard recursive bisection for HRP.
        """
        def get_cluster_var(cov, cluster_items):
            sub_cov = cov.loc[cluster_items, cluster_items]
            weights = np.linalg.pinv(sub_cov.values).sum(axis=1)
            weights /= weights.sum()
            return np.dot(weights, np.dot(sub_cov.values, weights))

        weights = pd.Series(1.0, index=sorted_tickers, dtype='float64')
        clusters = [sorted_tickers]

        while clusters:
            cluster = clusters.pop(0)
            if len(cluster) <= 1:
                continue

            split = int(len(cluster) / 2)
            left = cluster[:split]
            right = cluster[split:]

            var_left = get_cluster_var(self.cov_matrix, left)
            var_right = get_cluster_var(self.cov_matrix, right)

            alpha = 1 - var_left / (var_left + var_right)
            weights[left] *= alpha
            weights[right] *= (1 - alpha)

            clusters += [left, right]

        return weights

    def _recursive_bisection_mhrp(self, sorted_tickers: List[str]) -> pd.Series:
        """
        Modified recursive bisection with equal volatility weighting.
        """
        def get_cluster_volatility(cov, cluster_items):
            sub_cov = cov.loc[cluster_items, cluster_items]
            vol = np.sqrt(np.diag(sub_cov))
            inv_vol = 1 / vol
            weights = inv_vol / inv_vol.sum()
            cluster_vol = np.sqrt(np.dot(weights, np.dot(sub_cov.values, weights)))
            return cluster_vol

        weights = pd.Series(1.0, index=sorted_tickers, dtype='float64')
        clusters = [sorted_tickers]

        while clusters:
            cluster = clusters.pop(0)
            if len(cluster) <= 1:
                continue

            split = int(len(cluster) / 2)
            left = cluster[:split]
            right = cluster[split:]

            vol_left = get_cluster_volatility(self.cov_matrix, left)
            vol_right = get_cluster_volatility(self.cov_matrix, right)

            total_vol = vol_left + vol_right
            if total_vol > 0:
                alpha = vol_right / total_vol
            else:
                alpha = 0.5

            weights[left] *= alpha
            weights[right] *= (1 - alpha)

            clusters += [left, right]

        # Apply equal volatility within final allocation
        vol = np.sqrt(np.diag(self.cov_matrix.loc[sorted_tickers, sorted_tickers]))
        inv_vol = 1 / vol
        weights = weights * inv_vol

        return weights

    def _get_quasi_diag(self, link: np.ndarray) -> List[int]:
        """
        Quasi-diagonalization for hierarchical clustering.
        """
        link = np.asarray(link, order='c')
        n = link.shape[0] + 1

        def _get_leaves(i):
            if i < n:
                return [i]
            else:
                left = int(link[i - n, 0])
                right = int(link[i - n, 1])
                return _get_leaves(left) + _get_leaves(right)

        return _get_leaves(2 * n - 2)

    def _calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics.
        """
        # Annual return
        portfolio_return = (self.returns @ self.weights).mean() * 252

        # Annual volatility
        portfolio_vol = np.sqrt(self.weights @ self.cov_matrix @ self.weights) * np.sqrt(252)

        # Sharpe ratio (assuming 0 risk-free rate)
        sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0

        # Concentration metrics
        max_weight = self.weights.max()
        min_weight = self.weights[self.weights > 1e-6].min() if (self.weights > 1e-6).any() else 0
        hhi = (self.weights ** 2).sum()
        effective_n = 1 / hhi if hhi > 0 else 1

        return {
            'annual_return': portfolio_return,
            'annual_volatility': portfolio_vol,
            'sharpe_ratio': sharpe,
            'max_weight': max_weight,
            'min_weight': min_weight,
            'herfindahl_index': hhi,
            'effective_n': effective_n
        }

    def _calculate_current_weights(self, tickers: List[str],
                                  shares: Dict[str, float]) -> Tuple[pd.Series, Dict]:
        """
        Calculate current portfolio weights based on shares.
        """
        # Get current prices
        prices = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1d")
                if not hist.empty:
                    prices[ticker] = hist['Close'].iloc[-1]
                else:
                    prices[ticker] = 100  # Default if fetch fails
            except:
                prices[ticker] = 100

        # Calculate values and weights
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
        rebalance_info = self._calculate_rebalance_info(current_weights)

        return current_weights, rebalance_info

    def _calculate_rebalance_info(self, current_weights: pd.Series,
                                 threshold: float = 0.05) -> Dict:
        """
        Calculate rebalancing recommendations.
        """
        differences = {}
        needs_rebalance = False
        suggestions = []

        for ticker in self.weights.index:
            current = current_weights.get(ticker, 0)
            optimal = self.weights.get(ticker, 0)
            diff = optimal - current

            if abs(diff) > threshold:
                needs_rebalance = True
                differences[ticker] = diff

                action = "Buy" if diff > 0 else "Sell"
                suggestions.append(f"{action} {ticker}: {abs(diff)*100:.1f}%")

        if needs_rebalance:
            message = "Rebalancing needed"
            suggestion_text = " | ".join(suggestions)
        else:
            message = "No rebalancing needed"
            suggestion_text = "Portfolio is within acceptable drift limits"

        return {
            'needs_rebalance': needs_rebalance,
            'message': message,
            'suggestions': suggestion_text,
            'differences': differences,
            'threshold': threshold
        }

    def _plot_donut_chart(self, weights: pd.Series, title: str, filepath: Path):
        """
        Create and save a donut chart visualization with fixed 600x600 size.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # IMPORTANT: Don't filter or reorder - preserve exact input order
        weights_to_plot = weights.copy()

        # Prepare labels with percentages - preserve order exactly
        labels = []
        sizes = []
        for ticker in weights_to_plot.index:
            if weights_to_plot[ticker] > 1e-4:  # Only show if weight > 0.01%
                labels.append(f"{ticker} {weights_to_plot[ticker] * 100:.1f}%")
                sizes.append(weights_to_plot[ticker])

        # Create figure with EXACT size
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

        # Title with explicit color
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10, color='black')

        # Set background
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')

        # Equal aspect ratio
        ax.axis('equal')

        # Save with exact size - no bbox_inches='tight' which changes dimensions
        plt.tight_layout()
        # fig.savefig(filepath, dpi=100, facecolor='white')
        fig.savefig(filepath, dpi=100, facecolor='white', edgecolor='none')
        plt.close(fig)

    def _merge_images(self, img1_path: Path, img2_path: Path, output_path: Path):
        """
        Merge two 600x600 images with a 5-pixel black divider.
        """
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
        print(f"✅ Saved merged image: {output_path.name}")

    def _print_results(self, method: str):
        """
        Print optimization results to console.
        """
        print(f"\n{'='*40}")
        print(f"{method} Optimization Results")
        print(f"{'='*40}")

        print("\nOptimized Weights:")
        print("-" * 30)
        # Don't sort - preserve original ticker order
        for ticker in self.weights.index:
            weight = self.weights[ticker]
            if weight > 1e-4:
                print(f"{ticker:<8} {weight*100:>6.2f}%")

        print(f"\n{'='*40}")
        print("Portfolio Metrics:")
        print("-" * 30)
        print(f"Annual Return:     {self.metrics['annual_return']*100:>6.2f}%")
        print(f"Annual Volatility: {self.metrics['annual_volatility']*100:>6.2f}%")
        print(f"Sharpe Ratio:      {self.metrics['sharpe_ratio']:>6.3f}")
        print(f"Max Weight:        {self.metrics['max_weight']*100:>6.2f}%")
        print(f"Min Weight:        {self.metrics['min_weight']*100:>6.2f}%")
        print(f"Effective N:       {self.metrics['effective_n']:>6.1f}")
        print(f"{'='*40}\n")

    def _print_rebalance_suggestions(self, current_weights: pd.Series,
                                    rebalance_info: Dict):
        """
        Print rebalancing suggestions.
        """
        print(f"\n{'='*40}")
        print("Rebalancing Analysis")
        print(f"{'='*40}")

        print("\nCurrent vs Optimal Weights:")
        print("-" * 40)
        print(f"{'Ticker':<8} {'Current':>10} {'Optimal':>10} {'Difference':>12}")
        print("-" * 40)

        for ticker in self.weights.index:
            current = current_weights.get(ticker, 0)
            optimal = self.weights.get(ticker, 0)
            diff = optimal - current

            if abs(diff) > 0.001:
                print(f"{ticker:<8} {current*100:>9.2f}% {optimal*100:>9.2f}% "
                      f"{diff*100:>+11.2f}%")

        print("\n" + "="*40)
        print(f"Status: {rebalance_info['message']}")
        if rebalance_info['needs_rebalance']:
            print(f"Action: {rebalance_info['suggestions']}")
        print("="*40 + "\n")


# Convenience functions for direct usage
def optimize_portfolio(tickers: List[str],
                       method: str = "MHRP",
                       shares: Optional[Dict[str, float]] = None,
                       lookback_days: int = 180,
                       output_dir: Optional[str] = None) -> Dict:
    """
    Convenience function for portfolio optimization.

    Args:
        tickers: List of ticker symbols
        method: Optimization method (HRP, HERC, MHRP, NCO)
        shares: Optional dictionary of current holdings
        lookback_days: Historical data period
        output_dir: Output directory for results

    Returns:
        Dictionary containing optimization results
    """
    optimizer = PortfolioOptimizer(Path(output_dir) if output_dir else None)
    return optimizer.optimize(tickers, method, shares, lookback_days)


if __name__ == "__main__":
    # Example usage
    test_tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B']

    # Example with shares for rebalancing analysis
    test_shares = {
        'AAPL': 100,
        'GOOGL': 50,
        'MSFT': 75,
        'AMZN': 30,
        'NVDA': 40,
        'TSLA': 25,
        'META': 60,
        'BRK-B': 20
    }

    # Test each method
    for method in ['MHRP', 'HRP', 'HERC', 'NCO']:
        print(f"\n{'#'*60}")
        print(f"Testing {method} Optimization")
        print(f"{'#'*60}")

        results = optimize_portfolio(
            tickers=test_tickers,
            method=method,
            shares=test_shares,
            lookback_days=180
        )

        print(f"\n✅ {method} optimization completed successfully!")
        print(f"   Images saved: {results.get('images', {})}")