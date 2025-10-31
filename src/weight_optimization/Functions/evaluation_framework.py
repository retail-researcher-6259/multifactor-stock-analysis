# Functions/evaluation_framework.py
"""
Enhanced Evaluation Framework for Multi-Factor Stock Screening
"""

import pandas as pd
import numpy as np
from scipy import stats
from arch.bootstrap import IIDBootstrap


class PerformanceEvaluator:
    """
    A comprehensive class for evaluating multi-factor model performance
    """

    def __init__(self):
        self.evaluation_results = {}

    def comprehensive_evaluation(self, returns, benchmark_returns=None):
        """
        Comprehensive performance evaluation

        Args:
            returns: Series of returns to evaluate
            benchmark_returns: Optional benchmark returns for comparison

        Returns:
            dict: Dictionary containing all performance metrics
        """
        results = {}

        # Basic metrics
        results['total_return'] = (1 + returns).prod() - 1
        results['mean_return'] = returns.mean()
        results['volatility'] = returns.std()
        results['sharpe_ratio'] = returns.mean() / returns.std() if returns.std() > 0 else 0

        # Risk metrics
        results['max_drawdown'] = self.calculate_max_drawdown(returns)
        results['var_95'] = returns.quantile(0.05)
        results['cvar_95'] = returns[returns <= results['var_95']].mean()
        results['skewness'] = stats.skew(returns)
        results['kurtosis'] = stats.kurtosis(returns)

        # Information ratio (if benchmark provided)
        if benchmark_returns is not None:
            active_returns = returns - benchmark_returns
            results[
                'information_ratio'] = active_returns.mean() / active_returns.std() if active_returns.std() > 0 else 0
            results['tracking_error'] = active_returns.std()
            results['beta'] = np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
            results['alpha'] = returns.mean() - results['beta'] * benchmark_returns.mean()

        # Stability metrics
        results['hit_rate'] = (returns > 0).mean()
        results['worst_month'] = returns.min()
        results['best_month'] = returns.max()

        # Risk-adjusted performance
        if abs(results['max_drawdown']) > 0:
            results['calmar_ratio'] = results['total_return'] / abs(results['max_drawdown'])
        else:
            results['calmar_ratio'] = np.inf

        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            results['sortino_ratio'] = returns.mean() / downside_returns.std()
        else:
            results['sortino_ratio'] = np.inf

        return results

    def calculate_max_drawdown(self, returns):
        """
        Calculate maximum drawdown

        Args:
            returns: Series of returns

        Returns:
            float: Maximum drawdown value
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def rolling_performance(self, returns, window=12):
        """
        Calculate rolling performance metrics

        Args:
            returns: Series of returns
            window: Rolling window size

        Returns:
            DataFrame: Rolling performance metrics
        """
        rolling_metrics = pd.DataFrame(index=returns.index)

        rolling_metrics['Return'] = returns.rolling(window).mean()
        rolling_metrics['Volatility'] = returns.rolling(window).std()
        rolling_metrics['Sharpe'] = rolling_metrics['Return'] / rolling_metrics['Volatility']

        # Rolling max drawdown
        def rolling_max_dd(series):
            if len(series) < 2:
                return np.nan
            return self.calculate_max_drawdown(series)

        rolling_metrics['Max_DD'] = returns.rolling(window).apply(
            rolling_max_dd, raw=False)

        return rolling_metrics

    def factor_performance_attribution(self, factor_scores, returns, weights):
        """
        Attributes performance to individual factors

        Args:
            factor_scores: Dictionary of factor scores
            returns: Series of returns
            weights: Dictionary of factor weights

        Returns:
            dict: Factor attribution analysis
        """
        attribution = {}

        # Calculate weighted factor contributions
        for factor, weight in weights.items():
            if weight > 0 and factor in factor_scores:
                # This is a simplified attribution
                factor_contribution = factor_scores.get(factor, 0) * weight / 100
                attribution[factor] = factor_contribution

        return attribution

    def performance_analytics_dashboard(self, results, save_path=None):
        """
        Creates comprehensive performance analytics dashboard

        Args:
            results: Dictionary containing performance data
            save_path: Optional path to save the dashboard
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.style.use('seaborn-v0_8')
        except ImportError:
            print("matplotlib/seaborn not available for visualization")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Cumulative returns
        if 'returns' in results and len(results['returns']) > 0:
            cumulative = (1 + results['returns']).cumprod()
            axes[0, 0].plot(cumulative.values)
            axes[0, 0].set_title('Cumulative Returns', fontsize=14)
            axes[0, 0].set_ylabel('Cumulative Return')
            axes[0, 0].grid(True, alpha=0.3)

        # 2. Rolling Sharpe ratio
        if 'rolling_sharpe' in results and len(results['rolling_sharpe']) > 0:
            rolling_sharpe = results['rolling_sharpe'].dropna()
            if len(rolling_sharpe) > 0:
                axes[0, 1].plot(rolling_sharpe.values)
                axes[0, 1].set_title('Rolling 12-Month Sharpe Ratio', fontsize=14)
                axes[0, 1].set_ylabel('Sharpe Ratio')
                axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
                axes[0, 1].grid(True, alpha=0.3)

        # 3. Return distribution
        if 'returns' in results and len(results['returns']) > 0:
            axes[0, 2].hist(results['returns'], bins=50, alpha=0.7, edgecolor='black')
            axes[0, 2].set_title('Return Distribution', fontsize=14)
            axes[0, 2].set_xlabel('Monthly Return')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].axvline(x=results['returns'].mean(), color='r', linestyle='--',
                               label=f'Mean: {results["returns"].mean():.4%}')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)

        # 4. Rolling drawdown
        if 'returns' in results and len(results['returns']) > 0:
            cumulative = (1 + results['returns']).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            axes[1, 0].fill_between(range(len(drawdown)), drawdown, 0,
                                    alpha=0.5, color='red')
            axes[1, 0].set_title('Drawdown', fontsize=14)
            axes[1, 0].set_ylabel('Drawdown %')
            axes[1, 0].set_xlabel('Time Period')
            axes[1, 0].grid(True, alpha=0.3)

        # 5. Factor contribution (if available)
        if 'factor_attribution' in results and results['factor_attribution']:
            factors = list(results['factor_attribution'].keys())
            contributions = list(results['factor_attribution'].values())
            axes[1, 1].bar(factors, contributions, alpha=0.7)
            axes[1, 1].set_title('Factor Contribution to Score', fontsize=14)
            axes[1, 1].set_ylabel('Contribution')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)

        # 6. Performance metrics summary
        if 'metrics' in results:
            metrics_text = ""
            key_metrics = ['mean_return', 'sharpe_ratio', 'max_drawdown', 'volatility',
                           'hit_rate', 'var_95', 'skewness']

            for metric in key_metrics:
                if metric in results['metrics']:
                    value = results['metrics'][metric]
                    if metric in ['mean_return', 'max_drawdown', 'var_95']:
                        metrics_text += f"{metric.replace('_', ' ').title()}: {value:.4%}\n"
                    elif metric in ['sharpe_ratio', 'skewness']:
                        metrics_text += f"{metric.replace('_', ' ').title()}: {value:.4f}\n"
                    else:
                        metrics_text += f"{metric.replace('_', ' ').title()}: {value:.4f}\n"

            axes[1, 2].text(0.1, 0.9, metrics_text, transform=axes[1, 2].transAxes,
                            fontsize=12, verticalalignment='top', fontfamily='monospace')
            axes[1, 2].set_title('Key Performance Metrics', fontsize=14)
            axes[1, 2].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Dashboard saved to {save_path}")
        plt.show()

    def statistical_significance_tests(self, returns1, returns2):
        """
        Tests statistical significance of performance differences

        Args:
            returns1: First series of returns
            returns2: Second series of returns

        Returns:
            dict: Statistical test results
        """
        results = {}

        # T-test for mean differences
        t_stat, p_value = stats.ttest_ind(returns1, returns2, equal_var=False)
        results['t_test'] = {'statistic': t_stat, 'p_value': p_value}

        # Wilcoxon rank-sum test (non-parametric)
        from scipy.stats import ranksums
        w_stat, w_p_value = ranksums(returns1, returns2)
        results['wilcoxon_test'] = {'statistic': w_stat, 'p_value': w_p_value}

        # Jarque-Bera test for normality
        jb_stat1, jb_p1 = stats.jarque_bera(returns1)
        jb_stat2, jb_p2 = stats.jarque_bera(returns2)
        results['normality_test'] = {
            'series1': {'statistic': jb_stat1, 'p_value': jb_p1},
            'series2': {'statistic': jb_stat2, 'p_value': jb_p2}
        }

        return results

    def bootstrap_confidence_intervals(self, returns, metrics=['mean', 'sharpe'], n_bootstrap=1000):
        """
        Calculate bootstrap confidence intervals for various metrics

        Args:
            returns: Series of returns
            metrics: List of metrics to bootstrap
            n_bootstrap: Number of bootstrap samples

        Returns:
            dict: Bootstrap confidence intervals
        """
        results = {}

        # Define metrics
        def mean_return(x):
            return np.mean(x)

        def sharpe_ratio(x):
            std = np.std(x, ddof=1)
            return np.nan if std == 0 else np.mean(x) / std

        metric_functions = {
            'mean': mean_return,
            'sharpe': sharpe_ratio
        }

        # Calculate confidence intervals
        for metric in metrics:
            if metric in metric_functions:
                bootstrap_results = IIDBootstrap(returns.values).conf_int(
                    metric_functions[metric], reps=n_bootstrap, size=0.95, method="percentile"
                )
                lower, upper = bootstrap_results.flatten()
                results[metric] = {'lower': lower, 'upper': upper}

        return results

    def analyze_temporal_stability(self, df):
        """
        Analyze temporal stability of performance

        Args:
            df: DataFrame with returns and dates

        Returns:
            dict: Temporal stability analysis results
        """
        results = {}

        # Split data into different periods
        split_point = len(df) // 2
        early_period = df[:split_point][df[:split_point].Decile == 9]['Ret']
        late_period = df[split_point:][df[split_point:].Decile == 9]['Ret']

        if len(early_period) > 0 and len(late_period) > 0:
            results['early_period'] = {
                'mean': early_period.mean(),
                'sharpe': early_period.mean() / early_period.std()
            }
            results['late_period'] = {
                'mean': late_period.mean(),
                'sharpe': late_period.mean() / late_period.std()
            }

            # Statistical significance test between periods
            stability_tests = self.statistical_significance_tests(early_period, late_period)
            results['period_tests'] = stability_tests

        return results