"""
Independent recomputation of the statistical tests the code templates run.

build_analysis_fn() returns a function that reruns one test on a dataframe
with scipy, using the same formulas as the generated code in
kosmos/execution/code_generator.py. The research director uses it to check a
stored statistic against the data and, with shuffle_target(), as the analysis
that NullModelValidator reruns on permuted copies of the real data.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

CORRELATION_TESTS = ('pearson_correlation', 'spearman_correlation', 'linear_regression')
GROUP_TESTS = ('welch_t_test', 'mann_whitney', 'one_way_anova')
SUPPORTED_TESTS = CORRELATION_TESTS + GROUP_TESTS


def _group_samples(
    data: pd.DataFrame, x_col: str, y: np.ndarray, groups: Optional[Sequence[Any]]
) -> List[np.ndarray]:
    """Split y by the labels in x_col: the given group order, else sorted levels."""
    labels = data[x_col].astype(str)
    levels = [str(g) for g in groups] if groups else sorted(labels.unique())
    return [y[(labels == level).values] for level in levels]


def _cohens_d(g1: np.ndarray, g2: np.ndarray) -> float:
    """Mean difference over the pooled standard deviation (the templates' effect size)."""
    n1, n2 = len(g1), len(g2)
    pooled_sd = float(np.sqrt(
        ((n1 - 1) * np.var(g1, ddof=1) + (n2 - 1) * np.var(g2, ddof=1)) / (n1 + n2 - 2)
    ))
    mean_diff = float(np.mean(g1) - np.mean(g2))
    return mean_diff / pooled_sd if pooled_sd > 0 else 0.0


def build_analysis_fn(
    test_type: str,
    x_col: str,
    y_col: str,
    groups: Optional[Sequence[Any]] = None,
) -> Callable[[pd.DataFrame], Dict[str, Any]]:
    """
    Build a function that runs one statistical test of y_col on x_col.

    Args:
        test_type: One of SUPPORTED_TESTS
        x_col: Independent column (numeric for correlations and regression,
            group labels for group tests)
        y_col: Dependent numeric column
        groups: For group tests, the labels to compare in order (the t statistic
            is groups[0] minus groups[1]); default is every level, sorted

    Returns:
        A function df -> {'statistic', 'p_value', 'effect_size', 'test_type', 'n'}.
        Rows missing x_col or y_col are dropped first, as the templates do.
        For mann_whitney the statistic is the rank-biserial correlation (0 under
        the null), so a two-sided permutation test can compare absolute values;
        the U statistic is returned as 'u_statistic'.

    Raises:
        ValueError: if test_type is not supported
    """
    if test_type not in SUPPORTED_TESTS:
        raise ValueError(
            f"Unsupported test_type {test_type!r}; supported: {', '.join(SUPPORTED_TESTS)}"
        )

    def analysis(df: pd.DataFrame) -> Dict[str, Any]:
        data = df.dropna(subset=[x_col, y_col])
        y = data[y_col].astype(float).values
        out: Dict[str, Any] = {'test_type': test_type}

        if test_type in CORRELATION_TESTS:
            x = data[x_col].astype(float).values
            out['n'] = int(len(x))
            if test_type == 'pearson_correlation':
                r, p = stats.pearsonr(x, y)
                out.update(statistic=float(r), p_value=float(p), effect_size=float(r))
            elif test_type == 'spearman_correlation':
                rho, p = stats.spearmanr(x, y)
                out.update(statistic=float(rho), p_value=float(p), effect_size=float(rho))
            else:
                slope, intercept, r_value, p, _std_err = stats.linregress(x, y)
                out.update(
                    statistic=float(slope), p_value=float(p),
                    effect_size=float(r_value ** 2), intercept=float(intercept),
                )
            return out

        samples = _group_samples(data, x_col, y, groups)
        out['n'] = int(sum(len(s) for s in samples))
        if test_type == 'one_way_anova':
            f_stat, p = stats.f_oneway(*samples)
            pooled = np.concatenate(samples)
            grand_mean = float(np.mean(pooled))
            ss_between = float(sum(len(s) * (np.mean(s) - grand_mean) ** 2 for s in samples))
            ss_total = float(np.sum((pooled - grand_mean) ** 2))
            out.update(
                statistic=float(f_stat), p_value=float(p),
                effect_size=ss_between / ss_total if ss_total > 0 else 0.0,  # eta squared
            )
            return out

        if len(samples) != 2:
            raise ValueError(f"{test_type} needs exactly 2 groups in {x_col!r}, got {len(samples)}")
        g1, g2 = samples
        if test_type == 'welch_t_test':
            t_stat, p = stats.ttest_ind(g1, g2, equal_var=False)
            out.update(
                statistic=float(t_stat), p_value=float(p), effect_size=_cohens_d(g1, g2),
                mean_difference=float(np.mean(g1) - np.mean(g2)),
            )
        else:
            u_stat, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
            rank_biserial = 2.0 * float(u_stat) / (len(g1) * len(g2)) - 1.0
            out.update(
                statistic=rank_biserial, p_value=float(p), effect_size=rank_biserial,
                u_statistic=float(u_stat),
            )
        return out

    return analysis


def shuffle_target(
    df: pd.DataFrame,
    test_type: str,
    x_col: str,
    y_col: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Return a copy of df with the tested association broken.

    Permutes y_col for correlations and regression, and the group labels
    (x_col) for group tests. Every other column is left as it is.
    """
    column = x_col if test_type in GROUP_TESTS else y_col
    shuffled = df.copy()
    shuffled[column] = rng.permutation(shuffled[column].values)
    return shuffled
