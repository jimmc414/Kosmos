"""
Extended statistical analysis for Phase 6.

Additional statistical analysis beyond Phase 5's execution statistics,
focused on interpretation and reporting.

Note: Phase 5 has kosmos/execution/statistics.py for computational statistics.
This module adds interpretive and reporting capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import logging

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


class DescriptiveStats:
    """Descriptive statistics computation and reporting."""

    @staticmethod
    def compute_full_descriptive(data: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive descriptive statistics.

        Args:
            data: Data array

        Returns:
            dict: Descriptive statistics
        """
        # Remove NaN values
        clean_data = data[~np.isnan(data)]

        if len(clean_data) == 0:
            return {"error": "All values are NaN"}

        stats_dict = {
            'n': len(clean_data),
            'n_missing': int(np.sum(np.isnan(data))),
            'mean': float(np.mean(clean_data)),
            'median': float(np.median(clean_data)),
            'std': float(np.std(clean_data, ddof=1)),  # Sample std
            'var': float(np.var(clean_data, ddof=1)),  # Sample variance
            'min': float(np.min(clean_data)),
            'max': float(np.max(clean_data)),
            'range': float(np.max(clean_data) - np.min(clean_data)),
            'q1': float(np.percentile(clean_data, 25)),
            'q3': float(np.percentile(clean_data, 75)),
            'iqr': float(np.percentile(clean_data, 75) - np.percentile(clean_data, 25)),
            'skewness': float(sp_stats.skew(clean_data)) if HAS_SCIPY else None,
            'kurtosis': float(sp_stats.kurtosis(clean_data)) if HAS_SCIPY else None,
            'cv': float(np.std(clean_data, ddof=1) / np.mean(clean_data)) if np.mean(clean_data) != 0 else None
        }

        # Add percentiles
        for p in [5, 10, 90, 95]:
            stats_dict[f'p{p}'] = float(np.percentile(clean_data, p))

        return stats_dict

    @staticmethod
    def generate_descriptive_report(data: Dict[str, np.ndarray]) -> str:
        """
        Generate text report of descriptive statistics.

        Args:
            data: Dictionary of {variable_name: data_array}

        Returns:
            str: Formatted report
        """
        report = "# Descriptive Statistics Report\n\n"

        for var_name, var_data in data.items():
            stats = DescriptiveStats.compute_full_descriptive(var_data)

            report += f"## {var_name}\n\n"
            report += f"- **N:** {stats.get('n', 'N/A')}\n"
            report += f"- **Mean:** {stats.get('mean', 'N/A'):.3f}\n"
            report += f"- **Median:** {stats.get('median', 'N/A'):.3f}\n"
            report += f"- **Std Dev:** {stats.get('std', 'N/A'):.3f}\n"
            report += f"- **Min:** {stats.get('min', 'N/A'):.3f}\n"
            report += f"- **Max:** {stats.get('max', 'N/A'):.3f}\n"
            report += f"- **Q1:** {stats.get('q1', 'N/A'):.3f}\n"
            report += f"- **Q3:** {stats.get('q3', 'N/A'):.3f}\n"
            report += f"- **Skewness:** {stats.get('skewness', 'N/A'):.3f}\n"
            report += f"- **Kurtosis:** {stats.get('kurtosis', 'N/A'):.3f}\n\n"

        return report


class DistributionAnalysis:
    """Distribution fitting and analysis."""

    @staticmethod
    def test_normality(data: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Test if data is normally distributed.

        Args:
            data: Data array
            alpha: Significance level

        Returns:
            dict: Normality test results
        """
        if not HAS_SCIPY:
            return {"error": "scipy required for normality testing"}

        clean_data = data[~np.isnan(data)]

        if len(clean_data) < 3:
            return {"error": "Insufficient data for normality test"}

        # Shapiro-Wilk test
        statistic, p_value = sp_stats.shapiro(clean_data)

        result = {
            'test': 'Shapiro-Wilk',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'is_normal': p_value > alpha,
            'alpha': alpha,
            'interpretation': (
                f"Data {'appears' if p_value > alpha else 'does not appear'} "
                f"normally distributed (p={p_value:.4f})"
            )
        }

        return result

    @staticmethod
    def fit_distribution(
        data: np.ndarray,
        dist_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Fit common distributions to data and find best fit.

        Args:
            data: Data array
            dist_names: Distribution names to try (default: ['norm', 'lognorm', 'expon'])

        Returns:
            dict: Best fitting distribution info
        """
        if not HAS_SCIPY:
            return {"error": "scipy required for distribution fitting"}

        clean_data = data[~np.isnan(data)]

        if len(clean_data) < 10:
            return {"error": "Insufficient data for distribution fitting"}

        if dist_names is None:
            dist_names = ['norm', 'lognorm', 'expon']

        best_fit = None
        best_aic = np.inf

        for dist_name in dist_names:
            try:
                dist = getattr(sp_stats, dist_name)
                params = dist.fit(clean_data)
                log_likelihood = np.sum(dist.logpdf(clean_data, *params))

                # Calculate AIC (Akaike Information Criterion)
                k = len(params)
                aic = 2 * k - 2 * log_likelihood

                if aic < best_aic:
                    best_aic = aic
                    best_fit = {
                        'distribution': dist_name,
                        'parameters': params,
                        'aic': aic,
                        'log_likelihood': log_likelihood
                    }

            except Exception as e:
                logger.debug(f"Failed to fit {dist_name}: {e}")
                continue

        if best_fit:
            best_fit['interpretation'] = f"Best fit: {best_fit['distribution']} (AIC={best_fit['aic']:.2f})"

        return best_fit or {"error": "Could not fit any distribution"}


class CorrelationAnalysis:
    """Extended correlation analysis."""

    @staticmethod
    def correlation_matrix(
        data: pd.DataFrame,
        method: str = 'pearson'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute correlation matrix with p-values.

        Args:
            data: DataFrame with variables as columns
            method: 'pearson' or 'spearman'

        Returns:
            Tuple of (correlation_matrix, p_value_matrix)
        """
        if not HAS_SCIPY:
            raise ImportError("scipy required for correlation analysis")

        n_vars = data.shape[1]
        corr_matrix = np.zeros((n_vars, n_vars))
        p_matrix = np.zeros((n_vars, n_vars))

        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    p_matrix[i, j] = 0.0
                else:
                    # Remove NaN pairs
                    mask = ~(np.isnan(data.iloc[:, i]) | np.isnan(data.iloc[:, j]))
                    x = data.iloc[:, i][mask]
                    y = data.iloc[:, j][mask]

                    if len(x) > 2:
                        if method == 'pearson':
                            r, p = sp_stats.pearsonr(x, y)
                        else:
                            r, p = sp_stats.spearmanr(x, y)

                        corr_matrix[i, j] = r
                        p_matrix[i, j] = p
                    else:
                        corr_matrix[i, j] = np.nan
                        p_matrix[i, j] = np.nan

        corr_df = pd.DataFrame(corr_matrix, index=data.columns, columns=data.columns)
        p_df = pd.DataFrame(p_matrix, index=data.columns, columns=data.columns)

        return corr_df, p_df

    @staticmethod
    def generate_correlation_report(
        corr_matrix: pd.DataFrame,
        p_matrix: pd.DataFrame,
        alpha: float = 0.05
    ) -> str:
        """
        Generate text report of correlation analysis.

        Args:
            corr_matrix: Correlation matrix
            p_matrix: P-value matrix
            alpha: Significance level

        Returns:
            str: Formatted report
        """
        report = "# Correlation Analysis Report\n\n"

        # Find significant correlations
        significant = []

        n = corr_matrix.shape[0]
        for i in range(n):
            for j in range(i+1, n):
                r = corr_matrix.iloc[i, j]
                p = p_matrix.iloc[i, j]

                if not np.isnan(p) and p < alpha:
                    var1 = corr_matrix.index[i]
                    var2 = corr_matrix.columns[j]
                    significant.append({
                        'var1': var1,
                        'var2': var2,
                        'r': r,
                        'p': p
                    })

        # Sort by absolute correlation
        significant.sort(key=lambda x: abs(x['r']), reverse=True)

        report += f"## Significant Correlations (α={alpha})\n\n"

        if significant:
            for item in significant:
                strength = "strong" if abs(item['r']) > 0.7 else "moderate" if abs(item['r']) > 0.4 else "weak"
                direction = "positive" if item['r'] > 0 else "negative"

                report += f"- **{item['var1']} vs {item['var2']}**: "
                report += f"r = {item['r']:.3f} (p = {item['p']:.4f}) - "
                report += f"{strength} {direction} correlation\n"
        else:
            report += "No significant correlations found.\n"

        report += f"\n## Correlation Matrix\n\n"
        report += corr_matrix.to_markdown() + "\n"

        return report


class RegressionAnalysis:
    """Regression analysis utilities."""

    @staticmethod
    def simple_linear_regression(
        x: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Perform simple linear regression with diagnostics.

        Args:
            x: Independent variable
            y: Dependent variable

        Returns:
            dict: Regression results and diagnostics
        """
        if not HAS_SCIPY:
            raise ImportError("scipy required for regression")

        # Remove NaN pairs
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]

        if len(x_clean) < 3:
            return {"error": "Insufficient data for regression"}

        # Fit regression
        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x_clean, y_clean)

        # Compute residuals
        y_pred = slope * x_clean + intercept
        residuals = y_clean - y_pred

        # Residual diagnostics
        residual_std = np.std(residuals)

        results = {
            'slope': float(slope),
            'intercept': float(intercept),
            'r': float(r_value),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value),
            'std_err': float(std_err),
            'n': len(x_clean),
            'equation': f'y = {slope:.3f}x + {intercept:.3f}',
            'residuals': {
                'mean': float(np.mean(residuals)),
                'std': float(residual_std),
                'min': float(np.min(residuals)),
                'max': float(np.max(residuals))
            },
            'interpretation': (
                f"{'Significant' if p_value < 0.05 else 'Non-significant'} linear relationship "
                f"(R² = {r_value**2:.3f}, p = {p_value:.4f})"
            )
        }

        return results


class StatisticalReporter:
    """Generate comprehensive statistical reports."""

    def __init__(self):
        """Initialize statistical reporter."""
        self.descriptive = DescriptiveStats()
        self.distribution = DistributionAnalysis()
        self.correlation = CorrelationAnalysis()
        self.regression = RegressionAnalysis()

    def generate_full_report(
        self,
        data: pd.DataFrame,
        include_correlations: bool = True,
        include_distributions: bool = True
    ) -> str:
        """
        Generate comprehensive statistical report.

        Args:
            data: DataFrame with variables
            include_correlations: Include correlation analysis
            include_distributions: Include distribution analysis

        Returns:
            str: Markdown formatted report
        """
        report = "# Comprehensive Statistical Report\n\n"
        report += f"**Dataset Shape:** {data.shape[0]} observations × {data.shape[1]} variables\n\n"

        # Descriptive statistics
        report += "# Descriptive Statistics\n\n"
        data_dict = {col: data[col].values for col in data.columns}
        report += self.descriptive.generate_descriptive_report(data_dict)

        # Distribution analysis
        if include_distributions:
            report += "\n# Distribution Analysis\n\n"
            for col in data.columns:
                normality = self.distribution.test_normality(data[col].values)
                if 'error' not in normality:
                    report += f"## {col}\n"
                    report += f"- {normality['interpretation']}\n\n"

        # Correlation analysis
        if include_correlations and data.shape[1] > 1:
            report += "\n# Correlation Analysis\n\n"
            try:
                corr_matrix, p_matrix = self.correlation.correlation_matrix(data)
                report += self.correlation.generate_correlation_report(corr_matrix, p_matrix)
            except Exception as e:
                report += f"Error in correlation analysis: {e}\n"

        return report
