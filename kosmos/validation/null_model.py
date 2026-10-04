"""
Null Model Statistical Validation for Kosmos.

Implements permutation testing to validate findings against null models,
addressing the "sycophancy loop" problem where LLM-based validation
(ScholarEval) lacks statistical grounding.

Paper Reference: Kosmos used null models to achieve 79.4% accuracy by
running analyses on randomized data to ensure discoveries disappear.

Key Features:
1. Permutation testing with configurable iterations (default: 1000)
2. Multiple shuffle strategies (column, row, label, residual)
3. Detection of findings that persist in noise (false positives)
4. Integration with ScholarEval validation framework

Issue: #70 - Null Model Statistical Validation
"""

import logging
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Callable, Union

import numpy as np
import pandas as pd

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


@dataclass
class NullModelResult:
    """
    Result of null model validation for a finding.

    Stores permutation test results including empirical p-value,
    percentile in null distribution, and persistence flag.
    """

    # Core results
    observed_statistic: float  # Original test statistic
    null_distribution: List[float]  # Distribution summary (percentiles: 5, 25, 50, 75, 95)
    permutation_p_value: float  # Empirical p-value from permutation test
    null_percentile: float  # Where observed falls in null distribution (0-100)

    # Validation outcome
    passes_null_test: bool  # True if permutation_p_value < threshold
    persists_in_noise: bool  # True if finding appears in shuffled data (BAD)

    # Metadata
    n_permutations: int  # Number of permutations performed
    shuffle_method: str  # 'column', 'row', 'label', or 'residual'
    alpha: float  # Significance threshold used

    # Effect size comparison (optional)
    observed_effect_size: Optional[float] = None
    null_effect_sizes: Optional[List[float]] = None  # Percentiles from permuted data

    # Timing
    computation_time_seconds: float = 0.0

    # Warning messages
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        for key, value in d.items():
            if isinstance(value, np.bool_):
                d[key] = bool(value)
            elif isinstance(value, np.integer):
                d[key] = int(value)
            elif isinstance(value, np.floating):
                d[key] = float(value)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NullModelResult':
        """Create NullModelResult from dictionary."""
        # Handle missing optional fields
        data.setdefault('observed_effect_size', None)
        data.setdefault('null_effect_sizes', None)
        data.setdefault('computation_time_seconds', 0.0)
        data.setdefault('warnings', [])
        return cls(**data)

    @property
    def is_valid(self) -> bool:
        """Check if the finding is statistically valid (passes null test, doesn't persist in noise)."""
        return self.passes_null_test and not self.persists_in_noise

    def get_summary(self) -> str:
        """Generate human-readable summary of validation result."""
        if self.is_valid:
            return (
                f"VALID: p={self.permutation_p_value:.4f} (< {self.alpha}), "
                f"percentile={self.null_percentile:.1f}%"
            )
        elif self.persists_in_noise:
            return (
                f"WARNING: Finding persists in noise (percentile={self.null_percentile:.1f}%). "
                f"Potential false positive."
            )
        else:
            return (
                f"INVALID: p={self.permutation_p_value:.4f} (>= {self.alpha}). "
                f"Finding not significant against null model."
            )


class NullModelValidator:
    """
    Validates findings against null models using permutation testing.

    Addresses the "sycophancy loop" problem where LLM-based validation
    (ScholarEval) lacks statistical grounding.

    Paper Reference: Kosmos used null models to achieve 79.4% accuracy
    by running analyses on randomized data to ensure discoveries disappear.

    Usage:
        validator = NullModelValidator(n_permutations=1000)
        result = validator.validate_finding(finding)
        if result.persists_in_noise:
            print("WARNING: Potential false positive!")
    """

    # Test statistic keys to look for in findings
    STATISTIC_KEYS = [
        'statistic', 't_statistic', 'f_statistic', 'chi2',
        'correlation', 'r', 'r_squared', 'effect_size',
        'coefficient', 'odds_ratio', 'hazard_ratio'
    ]

    def __init__(
        self,
        n_permutations: int = 1000,
        alpha: float = 0.05,
        random_seed: Optional[int] = None,
        persistence_threshold: float = 0.5  # IQR threshold for persistence
    ):
        """
        Initialize NullModelValidator.

        Args:
            n_permutations: Number of permutations for empirical null distribution
            alpha: Significance threshold for p-value (default: 0.05)
            random_seed: Random seed for reproducibility
            persistence_threshold: Threshold for detecting persistence in noise
                                   (fraction of null distribution)
        """
        self.n_permutations = n_permutations
        self.alpha = alpha
        self.persistence_threshold = persistence_threshold
        self.rng = np.random.default_rng(random_seed)

        if not HAS_SCIPY:
            logger.warning("scipy not available; parametric null distributions limited")

    def validate_finding(
        self,
        finding: Dict[str, Any],
        data: Optional[pd.DataFrame] = None,
        analysis_func: Optional[Callable[[pd.DataFrame], Dict[str, Any]]] = None,
        shuffle_func: Optional[Callable[[pd.DataFrame, np.random.Generator], pd.DataFrame]] = None
    ) -> NullModelResult:
        """
        Validate a finding using permutation testing.

        Args:
            finding: Finding dict with 'statistics' containing test results
            data: Original data (if available for full permutation test)
            analysis_func: Function to re-run analysis on permuted data
                          Should return dict with statistic in STATISTIC_KEYS
            shuffle_func: Optional (data, rng) -> permuted copy, used instead of the
                          heuristic shuffle methods (for example
                          kosmos.validation.analysis_fn.shuffle_target); recorded
                          as shuffle_method 'target'

        Returns:
            NullModelResult with permutation p-value and validation outcome
        """
        start_time = time.time()
        warnings: List[str] = []

        # Extract observed statistic from finding
        try:
            observed = self._extract_test_statistic(finding)
        except ValueError as e:
            logger.warning(f"Could not extract test statistic: {e}")
            # Return invalid result when no statistic found
            return NullModelResult(
                observed_statistic=0.0,
                null_distribution=[0.0] * 5,
                permutation_p_value=1.0,
                null_percentile=50.0,
                passes_null_test=False,
                persists_in_noise=True,
                n_permutations=0,
                shuffle_method='none',
                alpha=self.alpha,
                warnings=[str(e)],
                computation_time_seconds=time.time() - start_time
            )

        # Extract effect size if available
        effect_size = self._extract_effect_size(finding)

        # Determine shuffle method based on analysis type
        shuffle_method = 'target' if shuffle_func is not None else self._determine_shuffle_method(finding)

        # Generate null distribution
        if data is not None and analysis_func is not None:
            # Full permutation: shuffle data and re-run analysis
            null_dist, null_effects = self._full_permutation_test(
                data, analysis_func, shuffle_method, shuffle_func
            )
            if len(null_dist) < self.n_permutations * 0.5:
                warnings.append(
                    f"Only {len(null_dist)}/{self.n_permutations} permutations succeeded"
                )
        else:
            # Approximate: use parametric null distribution
            null_dist = self._parametric_null(finding)
            null_effects = None

        # Calculate empirical p-value
        p_value = self._calculate_permutation_pvalue(observed, null_dist)

        # Check if finding persists in noise (BAD sign)
        persists = self._check_persistence_in_noise(observed, null_dist)

        # Calculate percentile
        percentile = self._calculate_percentile(observed, null_dist)

        # Summarize null distribution (store as percentiles for efficiency)
        null_summary = self._summarize_distribution(null_dist)

        computation_time = time.time() - start_time

        return NullModelResult(
            observed_statistic=observed,
            null_distribution=null_summary,
            permutation_p_value=p_value,
            null_percentile=percentile,
            passes_null_test=bool(p_value < self.alpha),
            persists_in_noise=persists,
            n_permutations=len(null_dist),
            shuffle_method=shuffle_method,
            alpha=self.alpha,
            observed_effect_size=effect_size,
            null_effect_sizes=self._summarize_distribution(null_effects) if null_effects is not None else None,
            computation_time_seconds=computation_time,
            warnings=warnings
        )

    # === Shuffle Methods ===

    def shuffle_columns(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Shuffle specified columns independently (breaks correlations).

        Args:
            df: Input DataFrame
            columns: Columns to shuffle (default: all columns)

        Returns:
            DataFrame with shuffled columns
        """
        result = df.copy()
        cols_to_shuffle = columns or list(df.columns)
        for col in cols_to_shuffle:
            if col in result.columns:
                result[col] = self.rng.permutation(result[col].values)
        return result

    def shuffle_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Shuffle entire rows (breaks time-series structure).

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with shuffled rows
        """
        indices = self.rng.permutation(len(df))
        return df.iloc[indices].reset_index(drop=True)

    def shuffle_labels(
        self,
        df: pd.DataFrame,
        label_col: str
    ) -> pd.DataFrame:
        """
        Shuffle group labels (breaks group assignments for t-tests, ANOVA).

        Args:
            df: Input DataFrame
            label_col: Column containing group labels to shuffle

        Returns:
            DataFrame with shuffled labels
        """
        result = df.copy()
        if label_col in result.columns:
            result[label_col] = self.rng.permutation(result[label_col].values)
        return result

    def shuffle_residuals(
        self,
        y: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Shuffle regression residuals (for regression analyses).

        Args:
            y: Observed values
            y_pred: Predicted values

        Returns:
            New y values with shuffled residuals
        """
        residuals = y - y_pred
        shuffled_residuals = self.rng.permutation(residuals)
        return y_pred + shuffled_residuals

    # === Internal Methods ===

    def _extract_test_statistic(self, finding: Dict[str, Any]) -> float:
        """
        Extract primary test statistic from finding.

        Args:
            finding: Finding dictionary with 'statistics' field

        Returns:
            Absolute value of test statistic

        Raises:
            ValueError: If no test statistic found
        """
        stats = finding.get('statistics', {})

        # Try various common keys
        for key in self.STATISTIC_KEYS:
            if key in stats and stats[key] is not None:
                try:
                    return abs(float(stats[key]))
                except (ValueError, TypeError):
                    continue

        # Fall back to p-value if available (convert to z-score)
        if 'p_value' in stats and stats['p_value'] is not None and HAS_SCIPY:
            p = float(stats['p_value'])
            if 0 < p < 1:
                # Convert p-value to z-score (two-tailed)
                return abs(sp_stats.norm.ppf(p / 2))

        raise ValueError(
            f"No test statistic found in finding. "
            f"Expected one of: {self.STATISTIC_KEYS}"
        )

    def _extract_effect_size(self, finding: Dict[str, Any]) -> Optional[float]:
        """Extract effect size from finding if available."""
        stats = finding.get('statistics', {})
        for key in ['effect_size', 'cohens_d', 'd', 'eta_squared', 'r_squared']:
            if key in stats and stats[key] is not None:
                try:
                    return float(stats[key])
                except (ValueError, TypeError):
                    continue
        return None

    def _full_permutation_test(
        self,
        data: pd.DataFrame,
        analysis_func: Callable[[pd.DataFrame], Dict[str, Any]],
        shuffle_method: str,
        shuffle_func: Optional[Callable[[pd.DataFrame, np.random.Generator], pd.DataFrame]] = None
    ) -> tuple:
        """
        Run full permutation test by shuffling data and re-running analysis.

        Args:
            data: Original data
            analysis_func: Function to run analysis on data
            shuffle_method: Which shuffle method to use
            shuffle_func: Optional (data, rng) -> permuted copy; overrides shuffle_method

        Returns:
            Tuple of (null_statistics array, null_effect_sizes array or None)
        """
        null_statistics: List[float] = []
        null_effects: List[float] = []

        for _ in range(self.n_permutations):
            # Shuffle data based on method
            if shuffle_func is not None:
                shuffled = shuffle_func(data, self.rng)
            elif shuffle_method == 'column':
                shuffled = self.shuffle_columns(data)
            elif shuffle_method == 'row':
                shuffled = self.shuffle_rows(data)
            elif shuffle_method == 'label':
                # Try to detect label column
                label_col = self._detect_label_column(data)
                if label_col:
                    shuffled = self.shuffle_labels(data, label_col)
                else:
                    shuffled = self.shuffle_columns(data)
            else:
                shuffled = self.shuffle_columns(data)

            try:
                result = analysis_func(shuffled)
                stat = self._extract_test_statistic({'statistics': result})
                null_statistics.append(stat)

                # Also track effect size if available
                effect = self._extract_effect_size({'statistics': result})
                if effect is not None:
                    null_effects.append(effect)
            except Exception as e:
                logger.debug(f"Permutation failed: {e}")
                continue

        return (
            np.array(null_statistics),
            np.array(null_effects) if null_effects else None
        )

    def _parametric_null(self, finding: Dict[str, Any]) -> np.ndarray:
        """
        Generate approximate null distribution from parametric assumptions.

        Args:
            finding: Finding dictionary with test type information

        Returns:
            Array of null distribution values
        """
        if not HAS_SCIPY:
            # Fallback: uniform distribution
            return self.rng.uniform(0, 3, size=self.n_permutations)

        stats = finding.get('statistics', {})

        # Determine distribution based on test type
        test_type = str(stats.get('test_type', '')).lower()
        df = stats.get('degrees_of_freedom', 100)

        # Ensure df is valid
        if df is None or df < 1:
            df = 100

        if 't' in test_type or 't_test' in test_type:
            # T-distribution
            return np.abs(sp_stats.t.rvs(df, size=self.n_permutations, random_state=self.rng))
        elif 'f' in test_type or 'anova' in test_type:
            # F-distribution
            return sp_stats.f.rvs(1, max(1, df), size=self.n_permutations, random_state=self.rng)
        elif 'chi' in test_type:
            # Chi-squared distribution
            return sp_stats.chi2.rvs(max(1, df), size=self.n_permutations, random_state=self.rng)
        elif 'correlation' in test_type or 'pearson' in test_type or 'spearman' in test_type:
            # Correlation: use Fisher z-transformation null
            return np.abs(sp_stats.norm.rvs(size=self.n_permutations, random_state=self.rng))
        else:
            # Default: standard normal (absolute values)
            return np.abs(sp_stats.norm.rvs(size=self.n_permutations, random_state=self.rng))

    def _calculate_permutation_pvalue(
        self,
        observed: float,
        null_dist: np.ndarray
    ) -> float:
        """
        Calculate empirical p-value from permutation distribution.

        Uses the formula: p = (k + 1) / (n + 1) where k is the number
        of null values >= observed (correction for finite samples).

        Args:
            observed: Observed test statistic
            null_dist: Array of null distribution values

        Returns:
            Empirical p-value
        """
        if len(null_dist) == 0:
            return 1.0

        # Two-tailed: count how many null values >= observed
        n_extreme = np.sum(np.abs(null_dist) >= np.abs(observed))

        # Add 1 to numerator and denominator (correction)
        return float((n_extreme + 1) / (len(null_dist) + 1))

    def _check_persistence_in_noise(
        self,
        observed: float,
        null_dist: np.ndarray
    ) -> bool:
        """
        Check if finding persists even in shuffled/noise data.

        A finding "persists in noise" if the observed statistic falls
        within the middle portion of the null distribution, indicating
        it could easily arise by chance.

        Args:
            observed: Observed test statistic
            null_dist: Array of null distribution values

        Returns:
            True if finding persists in noise (potential false positive)
        """
        if len(null_dist) == 0:
            return True

        # If observed is within middle 50% of null distribution, it persists in noise
        lower_pct = (1 - self.persistence_threshold) / 2 * 100  # 25%
        upper_pct = 100 - lower_pct  # 75%

        q_low, q_high = np.percentile(null_dist, [lower_pct, upper_pct])

        # Check if observed falls within IQR (bad sign)
        return bool(q_low <= abs(observed) <= q_high)

    def _calculate_percentile(
        self,
        observed: float,
        null_dist: np.ndarray
    ) -> float:
        """
        Calculate where observed falls in null distribution.

        Args:
            observed: Observed test statistic
            null_dist: Array of null distribution values

        Returns:
            Percentile (0-100)
        """
        if len(null_dist) == 0:
            return 50.0

        return float(np.sum(null_dist <= abs(observed)) / len(null_dist) * 100)

    def _summarize_distribution(
        self,
        dist: Optional[np.ndarray]
    ) -> List[float]:
        """
        Summarize distribution as percentiles (for storage efficiency).

        Args:
            dist: Array of distribution values

        Returns:
            List of percentiles [5th, 25th, 50th, 75th, 95th]
        """
        if dist is None or len(dist) == 0:
            return [0.0] * 5

        percentiles = [5, 25, 50, 75, 95]
        return [float(np.percentile(dist, p)) for p in percentiles]

    def _determine_shuffle_method(self, finding: Dict[str, Any]) -> str:
        """
        Determine appropriate shuffle method based on analysis type.

        Args:
            finding: Finding dictionary

        Returns:
            Shuffle method: 'column', 'row', 'label', or 'residual'
        """
        stats = finding.get('statistics', {})
        test_type = str(stats.get('test_type', '')).lower()

        if 't_test' in test_type or 'anova' in test_type or 'group' in test_type:
            return 'label'
        elif 'correlation' in test_type or 'pearson' in test_type or 'spearman' in test_type:
            return 'column'
        elif 'regression' in test_type or 'linear' in test_type:
            return 'residual'
        elif 'time' in test_type or 'series' in test_type:
            return 'row'
        else:
            return 'column'  # Default

    def _detect_label_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Try to detect which column contains group labels.

        Args:
            df: Input DataFrame

        Returns:
            Column name or None if not detected
        """
        # Common label column names
        label_names = ['group', 'label', 'class', 'category', 'condition', 'treatment']

        for col in df.columns:
            col_lower = col.lower()
            for name in label_names:
                if name in col_lower:
                    return col

        # Check for columns with few unique values (likely categorical)
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].nunique() <= 5:
                return col

        return None

    def batch_validate(
        self,
        findings: List[Dict[str, Any]]
    ) -> List[NullModelResult]:
        """
        Validate multiple findings.

        Args:
            findings: List of finding dictionaries

        Returns:
            List of NullModelResult objects
        """
        return [self.validate_finding(f) for f in findings]

    def get_validation_statistics(
        self,
        results: List[NullModelResult]
    ) -> Dict[str, Any]:
        """
        Compute aggregate statistics from multiple validations.

        Args:
            results: List of NullModelResult objects

        Returns:
            Dictionary with validation statistics
        """
        if not results:
            return {'count': 0}

        valid_count = sum(1 for r in results if r.is_valid)
        passes_null = sum(1 for r in results if r.passes_null_test)
        persists_noise = sum(1 for r in results if r.persists_in_noise)

        p_values = [r.permutation_p_value for r in results]

        return {
            'count': len(results),
            'valid_count': valid_count,
            'valid_rate': valid_count / len(results),
            'passes_null_test_count': passes_null,
            'passes_null_test_rate': passes_null / len(results),
            'persists_in_noise_count': persists_noise,
            'persists_in_noise_rate': persists_noise / len(results),
            'mean_p_value': float(np.mean(p_values)),
            'median_p_value': float(np.median(p_values)),
            'min_p_value': float(np.min(p_values)),
            'max_p_value': float(np.max(p_values))
        }
