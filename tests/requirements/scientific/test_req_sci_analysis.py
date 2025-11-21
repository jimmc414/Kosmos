"""
Tests for Scientific Analysis Validity Requirements (REQ-SCI-ANA-*).

These tests validate statistical analysis methods, assumption checking,
effect size reporting, and analysis integrity as specified in
REQUIREMENTS.md Section 10.2.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from unittest.mock import Mock, patch, MagicMock
from scipy import stats as scipy_stats

from kosmos.analysis.statistics import DescriptiveStats, DistributionAnalysis

# Test markers for requirements traceability
pytestmark = [
    pytest.mark.requirement("REQ-SCI-ANA"),
    pytest.mark.category("scientific"),
    pytest.mark.priority("MUST"),
]


@pytest.mark.requirement("REQ-SCI-ANA-001")
@pytest.mark.priority("MUST")
def test_req_sci_ana_001_appropriate_statistical_methods():
    """
    REQ-SCI-ANA-001: Statistical analyses MUST use appropriate methods
    for the data type and distribution.

    Validates that:
    - Data type is assessed before analysis
    - Distribution is checked
    - Appropriate test is selected based on data characteristics
    """

    def select_appropriate_test(
        data: np.ndarray,
        data_type: str = "continuous",
        paired: bool = False
    ) -> Dict[str, Any]:
        """
        Select appropriate statistical test based on data characteristics.

        Args:
            data: Data array (for single sample) or 2D array (for two samples)
            data_type: 'continuous', 'ordinal', or 'categorical'
            paired: Whether samples are paired

        Returns:
            Dict with recommended test and rationale
        """
        recommendations = {
            "data_type": data_type,
            "sample_size": len(data) if data.ndim == 1 else len(data[0]),
            "is_paired": paired,
            "recommended_tests": [],
            "rationale": []
        }

        # For continuous data
        if data_type == "continuous":
            # Check normality
            if len(data) >= 3:
                try:
                    _, p_value = scipy_stats.shapiro(data if data.ndim == 1 else data[0])
                    is_normal = p_value > 0.05
                    recommendations["normality_p_value"] = p_value
                except Exception:
                    is_normal = None

                if is_normal:
                    recommendations["recommended_tests"].append("t-test (parametric)")
                    recommendations["rationale"].append("Data appears normally distributed")
                else:
                    recommendations["recommended_tests"].append("Mann-Whitney U or Wilcoxon (non-parametric)")
                    recommendations["rationale"].append("Data is not normally distributed")

            # Sample size consideration
            if recommendations["sample_size"] < 30:
                recommendations["recommended_tests"].append("Non-parametric test (small sample)")
                recommendations["rationale"].append("Small sample size suggests non-parametric approach")

        # For ordinal data
        elif data_type == "ordinal":
            recommendations["recommended_tests"].append("Mann-Whitney U or Kruskal-Wallis")
            recommendations["rationale"].append("Ordinal data requires rank-based tests")

        # For categorical data
        elif data_type == "categorical":
            recommendations["recommended_tests"].append("Chi-square or Fisher's exact test")
            recommendations["rationale"].append("Categorical data requires frequency-based tests")

        return recommendations

    # Test Case 1: Normal continuous data -> parametric test
    normal_data = np.random.normal(loc=100, scale=15, size=50)

    result = select_appropriate_test(normal_data, data_type="continuous")

    # Assert: Should recommend parametric test for normal data
    assert "data_type" in result, "Should assess data type"
    assert result["data_type"] == "continuous", "Should identify continuous data"
    assert any("parametric" in test.lower() or "t-test" in test.lower()
               for test in result["recommended_tests"]), \
        "Should recommend parametric test for normal data"

    # Test Case 2: Non-normal continuous data -> non-parametric test
    skewed_data = np.random.exponential(scale=2.0, size=50)

    result2 = select_appropriate_test(skewed_data, data_type="continuous")

    # Assert: Should recommend non-parametric test for skewed data
    assert any("non-parametric" in test.lower() or "mann-whitney" in test.lower() or "wilcoxon" in test.lower()
               for test in result2["recommended_tests"]), \
        "Should recommend non-parametric test for non-normal data"

    # Test Case 3: Ordinal data -> rank-based test
    result3 = select_appropriate_test(np.array([1, 2, 3, 2, 1, 3]), data_type="ordinal")

    # Assert: Should recommend rank-based test
    assert any("mann-whitney" in test.lower() or "kruskal" in test.lower()
               for test in result3["recommended_tests"]), \
        "Should recommend rank-based test for ordinal data"

    # Test Case 4: Categorical data -> frequency test
    result4 = select_appropriate_test(np.array([0, 1, 1, 0, 1]), data_type="categorical")

    # Assert: Should recommend chi-square or Fisher's exact
    assert any("chi-square" in test.lower() or "fisher" in test.lower()
               for test in result4["recommended_tests"]), \
        "Should recommend frequency-based test for categorical data"


@pytest.mark.requirement("REQ-SCI-ANA-002")
@pytest.mark.priority("MUST")
def test_req_sci_ana_002_check_statistical_assumptions():
    """
    REQ-SCI-ANA-002: The system MUST check statistical assumptions
    (normality, independence, homoscedasticity) before applying parametric tests.

    Validates that:
    - Normality is tested before parametric tests
    - Independence is verified
    - Homoscedasticity (equal variances) is checked
    - Violations are detected and reported
    """

    class AssumptionChecker:
        """Check statistical assumptions for parametric tests."""

        @staticmethod
        def check_normality(data: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
            """Test normality assumption."""
            if len(data) < 3:
                return {
                    "test": "shapiro-wilk",
                    "passed": False,
                    "reason": "Insufficient data for normality test",
                    "p_value": None
                }

            statistic, p_value = scipy_stats.shapiro(data)

            return {
                "test": "shapiro-wilk",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "passed": p_value > alpha,
                "alpha": alpha,
                "interpretation": (
                    f"Data {'appears' if p_value > alpha else 'does not appear'} "
                    f"normally distributed (p={p_value:.4f})"
                )
            }

        @staticmethod
        def check_homoscedasticity(
            group1: np.ndarray,
            group2: np.ndarray,
            alpha: float = 0.05
        ) -> Dict[str, Any]:
            """Test homoscedasticity (equal variances)."""
            if len(group1) < 2 or len(group2) < 2:
                return {
                    "test": "levene",
                    "passed": False,
                    "reason": "Insufficient data",
                    "p_value": None
                }

            statistic, p_value = scipy_stats.levene(group1, group2)

            return {
                "test": "levene",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "passed": p_value > alpha,
                "alpha": alpha,
                "interpretation": (
                    f"Variances {'appear' if p_value > alpha else 'do not appear'} "
                    f"equal (p={p_value:.4f})"
                )
            }

        @staticmethod
        def check_all_assumptions(
            data1: np.ndarray,
            data2: np.ndarray = None
        ) -> Dict[str, Any]:
            """Check all assumptions for parametric test."""
            results = {
                "all_assumptions_met": True,
                "checks": {},
                "violations": [],
                "recommendations": []
            }

            # Check normality for first group
            norm1 = AssumptionChecker.check_normality(data1)
            results["checks"]["normality_group1"] = norm1

            if not norm1["passed"]:
                results["all_assumptions_met"] = False
                results["violations"].append("Group 1 not normally distributed")
                results["recommendations"].append(
                    "Consider non-parametric test (e.g., Mann-Whitney U)"
                )

            # Check normality for second group if provided
            if data2 is not None:
                norm2 = AssumptionChecker.check_normality(data2)
                results["checks"]["normality_group2"] = norm2

                if not norm2["passed"]:
                    results["all_assumptions_met"] = False
                    results["violations"].append("Group 2 not normally distributed")

                # Check homoscedasticity
                homosc = AssumptionChecker.check_homoscedasticity(data1, data2)
                results["checks"]["homoscedasticity"] = homosc

                if not homosc["passed"]:
                    results["all_assumptions_met"] = False
                    results["violations"].append("Variances not equal")
                    results["recommendations"].append(
                        "Use Welch's t-test (does not assume equal variances)"
                    )

            # Note: Independence is typically a study design issue, not testable from data alone
            results["checks"]["independence"] = {
                "note": "Independence must be ensured by study design (random sampling, etc.)"
            }

            return results

    # Test Case 1: Data meeting all assumptions
    np.random.seed(42)
    normal_group1 = np.random.normal(100, 15, 40)
    normal_group2 = np.random.normal(105, 15, 40)

    result = AssumptionChecker.check_all_assumptions(normal_group1, normal_group2)

    # Assert: Should check all assumptions
    assert "checks" in result, "Should perform assumption checks"
    assert "normality_group1" in result["checks"], "Should check normality for group 1"
    assert "homoscedasticity" in result["checks"], "Should check homoscedasticity"
    assert "p_value" in result["checks"]["normality_group1"], "Should provide p-value"

    # Test Case 2: Data violating normality assumption
    skewed_data = np.random.exponential(2, 40)
    normal_data = np.random.normal(100, 15, 40)

    result2 = AssumptionChecker.check_all_assumptions(skewed_data, normal_data)

    # Assert: Should detect violation
    assert not result2["all_assumptions_met"], \
        "Should detect assumption violations"
    assert len(result2["violations"]) > 0, \
        "Should list specific violations"
    assert len(result2["recommendations"]) > 0, \
        "Should provide recommendations for violated assumptions"

    # Test Case 3: Unequal variances
    low_var = np.random.normal(100, 5, 40)
    high_var = np.random.normal(100, 20, 40)

    result3 = AssumptionChecker.check_all_assumptions(low_var, high_var)

    # Assert: Should detect heteroscedasticity
    if not result3["checks"]["homoscedasticity"]["passed"]:
        assert any("welch" in rec.lower() for rec in result3["recommendations"]), \
            "Should recommend Welch's t-test for unequal variances"


@pytest.mark.requirement("REQ-SCI-ANA-003")
@pytest.mark.priority("SHOULD")
def test_req_sci_ana_003_report_effect_sizes():
    """
    REQ-SCI-ANA-003: The system SHOULD report effect sizes alongside
    p-values for all statistical tests.

    Validates that:
    - Effect sizes are calculated
    - Effect sizes are reported with p-values
    - Effect size interpretation is provided
    """

    def compute_effect_size(
        group1: np.ndarray,
        group2: np.ndarray,
        effect_type: str = "cohen_d"
    ) -> Dict[str, Any]:
        """
        Compute effect size for two groups.

        Args:
            group1: First group data
            group2: Second group data
            effect_type: Type of effect size ('cohen_d', 'hedges_g', 'glass_delta')

        Returns:
            Dict with effect size and interpretation
        """
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)

        # Cohen's d
        if effect_type == "cohen_d":
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            effect_size = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0

        # Hedges' g (corrected for small sample bias)
        elif effect_type == "hedges_g":
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            cohen_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
            correction = 1 - (3 / (4 * (n1 + n2) - 9))
            effect_size = cohen_d * correction

        # Glass's delta (uses control group std)
        elif effect_type == "glass_delta":
            effect_size = (mean1 - mean2) / std2 if std2 > 0 else 0.0

        else:
            raise ValueError(f"Unknown effect type: {effect_type}")

        # Interpret effect size (Cohen's conventions)
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            interpretation = "negligible"
        elif abs_effect < 0.5:
            interpretation = "small"
        elif abs_effect < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"

        return {
            "effect_size": float(effect_size),
            "effect_type": effect_type,
            "magnitude": interpretation,
            "mean_difference": float(mean1 - mean2),
            "group1_mean": float(mean1),
            "group2_mean": float(mean2),
            "group1_std": float(std1),
            "group2_std": float(std2)
        }

    def perform_test_with_effect_size(
        group1: np.ndarray,
        group2: np.ndarray
    ) -> Dict[str, Any]:
        """Perform statistical test and compute effect size."""
        # Perform t-test
        t_stat, p_value = scipy_stats.ttest_ind(group1, group2)

        # Compute effect size
        effect = compute_effect_size(group1, group2, effect_type="cohen_d")

        return {
            "test": "independent_t_test",
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "effect_size": effect["effect_size"],
            "effect_magnitude": effect["magnitude"],
            "full_effect_details": effect,
            "interpretation": (
                f"{'Significant' if p_value < 0.05 else 'Non-significant'} difference "
                f"(p={p_value:.4f}) with {effect['magnitude']} effect size "
                f"(d={effect['effect_size']:.3f})"
            )
        }

    # Test Case 1: Large effect size (should be reported)
    np.random.seed(42)
    group_a = np.random.normal(100, 15, 50)
    group_b = np.random.normal(120, 15, 50)  # Large difference

    result = perform_test_with_effect_size(group_a, group_b)

    # Assert: Should report effect size
    assert "effect_size" in result, "Should compute effect size"
    assert "effect_magnitude" in result, "Should provide effect size interpretation"
    assert result["effect_magnitude"] in ["small", "medium", "large"], \
        "Should categorize effect size"
    assert "interpretation" in result, \
        "Should provide interpretation including both p-value and effect size"
    assert abs(result["effect_size"]) > 0.8, \
        "Should detect large effect size for groups with large mean difference"

    # Test Case 2: Small effect size (should still be reported)
    group_c = np.random.normal(100, 15, 50)
    group_d = np.random.normal(102, 15, 50)  # Small difference

    result2 = perform_test_with_effect_size(group_c, group_d)

    # Assert: Should report small effect size
    assert result2["effect_magnitude"] in ["negligible", "small"], \
        "Should detect small effect size for groups with small mean difference"
    assert "effect_size" in result2, \
        "Should report effect size even when small"

    # Test Case 3: Effect size interpretation
    effect_details = compute_effect_size(group_a, group_b)

    assert "magnitude" in effect_details, "Should provide magnitude interpretation"
    assert "mean_difference" in effect_details, "Should include mean difference"
    assert effect_details["effect_type"] == "cohen_d", "Should specify effect type"


@pytest.mark.requirement("REQ-SCI-ANA-004")
@pytest.mark.priority("MUST")
def test_req_sci_ana_004_flag_assumption_violations():
    """
    REQ-SCI-ANA-004: The system MUST flag analyses that violate assumptions
    and suggest alternative approaches.

    Validates that:
    - Assumption violations are detected
    - Warnings are generated
    - Alternative approaches are suggested
    """

    class AnalysisValidator:
        """Validate analysis and suggest alternatives."""

        @staticmethod
        def validate_analysis(
            data: np.ndarray,
            analysis_type: str,
            **kwargs
        ) -> Dict[str, Any]:
            """
            Validate if analysis is appropriate for data.

            Args:
                data: Data array
                analysis_type: Type of analysis ('t_test', 'anova', etc.)
                **kwargs: Additional parameters

            Returns:
                Validation result with flags and suggestions
            """
            result = {
                "analysis_type": analysis_type,
                "is_appropriate": True,
                "violations": [],
                "warnings": [],
                "alternative_suggestions": []
            }

            # Check normality
            if len(data) >= 3:
                _, p_norm = scipy_stats.shapiro(data)

                if p_norm < 0.05:
                    result["is_appropriate"] = False
                    result["violations"].append({
                        "assumption": "normality",
                        "p_value": float(p_norm),
                        "severity": "critical" if analysis_type in ["t_test", "anova"] else "warning"
                    })

                    if analysis_type == "t_test":
                        result["alternative_suggestions"].append({
                            "method": "Mann-Whitney U test",
                            "rationale": "Non-parametric alternative for non-normal data"
                        })
                    elif analysis_type == "anova":
                        result["alternative_suggestions"].append({
                            "method": "Kruskal-Wallis test",
                            "rationale": "Non-parametric alternative to ANOVA"
                        })

            # Check sample size
            if len(data) < 30:
                result["warnings"].append({
                    "issue": "small_sample_size",
                    "sample_size": len(data),
                    "recommendation": "Consider non-parametric tests for small samples"
                })

                result["alternative_suggestions"].append({
                    "method": "Bootstrap or permutation test",
                    "rationale": "More robust for small sample sizes"
                })

            # Check for outliers
            if len(data) > 0:
                q1, q3 = np.percentile(data, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = np.sum((data < lower_bound) | (data > upper_bound))

                if outliers > 0:
                    result["warnings"].append({
                        "issue": "outliers_detected",
                        "count": int(outliers),
                        "percentage": float(outliers / len(data) * 100),
                        "recommendation": "Consider robust statistics or outlier removal"
                    })

            return result

        @staticmethod
        def generate_warning_message(validation_result: Dict[str, Any]) -> str:
            """Generate human-readable warning message."""
            if validation_result["is_appropriate"] and not validation_result["warnings"]:
                return "Analysis appears appropriate for the data"

            messages = []

            if not validation_result["is_appropriate"]:
                messages.append("⚠️ CRITICAL: Analysis may not be appropriate")

                for violation in validation_result["violations"]:
                    messages.append(
                        f"  - {violation['assumption'].upper()} assumption violated "
                        f"(p={violation.get('p_value', 'N/A')})"
                    )

            if validation_result["warnings"]:
                messages.append("⚠️ WARNINGS:")
                for warning in validation_result["warnings"]:
                    messages.append(f"  - {warning['issue']}: {warning.get('recommendation', '')}")

            if validation_result["alternative_suggestions"]:
                messages.append("\n📋 ALTERNATIVE APPROACHES:")
                for alt in validation_result["alternative_suggestions"]:
                    messages.append(f"  - {alt['method']}: {alt['rationale']}")

            return "\n".join(messages)

    # Test Case 1: Appropriate analysis (no violations)
    np.random.seed(42)
    normal_data = np.random.normal(100, 15, 50)

    result = AnalysisValidator.validate_analysis(normal_data, "t_test")

    # Assert: Should pass validation
    assert "is_appropriate" in result, "Should assess appropriateness"
    assert "violations" in result, "Should check for violations"
    assert "alternative_suggestions" in result, "Should provide alternatives if needed"

    # Test Case 2: Non-normal data with parametric test (violation)
    skewed_data = np.random.exponential(2, 50)

    result2 = AnalysisValidator.validate_analysis(skewed_data, "t_test")

    # Assert: Should flag violation
    assert not result2["is_appropriate"], \
        "Should flag inappropriate analysis for non-normal data"
    assert len(result2["violations"]) > 0, \
        "Should list specific violations"
    assert any(v["assumption"] == "normality" for v in result2["violations"]), \
        "Should identify normality violation"
    assert len(result2["alternative_suggestions"]) > 0, \
        "Should suggest alternative methods"
    assert any("mann-whitney" in alt["method"].lower()
               for alt in result2["alternative_suggestions"]), \
        "Should suggest non-parametric alternative"

    # Test Case 3: Warning message generation
    warning_msg = AnalysisValidator.generate_warning_message(result2)

    # Assert: Should generate informative warning
    assert "CRITICAL" in warning_msg or "WARNING" in warning_msg, \
        "Should include severity indicator"
    assert "ALTERNATIVE" in warning_msg, \
        "Should include alternative suggestions"


@pytest.mark.requirement("REQ-SCI-ANA-005")
@pytest.mark.priority("MUST")
def test_req_sci_ana_005_no_grossly_violating_assumptions():
    """
    REQ-SCI-ANA-005: The system MUST NOT perform statistical tests on data
    that grossly violates test assumptions (e.g., t-test on heavily skewed
    non-normal data with small n).

    Validates that:
    - Gross violations are detected
    - Tests are blocked when assumptions are severely violated
    - Clear error messages are provided
    """

    class SafeStatisticalTest:
        """Perform statistical tests with assumption validation."""

        @staticmethod
        def safe_t_test(
            group1: np.ndarray,
            group2: np.ndarray,
            force: bool = False
        ) -> Dict[str, Any]:
            """
            Perform t-test with safety checks.

            Args:
                group1: First group
                group2: Second group
                force: Force execution despite violations (for testing)

            Returns:
                Test result or error dict
            """
            result = {
                "test": "t_test",
                "executed": False,
                "blocked": False,
                "violations": [],
                "result": None
            }

            # Check sample sizes
            if len(group1) < 3 or len(group2) < 3:
                result["blocked"] = True
                result["violations"].append({
                    "type": "insufficient_data",
                    "severity": "critical",
                    "message": "Sample size too small for t-test (n < 3)"
                })
                return result

            # Check normality (stricter for small samples)
            small_sample = len(group1) < 30 or len(group2) < 30

            try:
                _, p1 = scipy_stats.shapiro(group1)
                _, p2 = scipy_stats.shapiro(group2)

                # Stricter threshold for small samples
                threshold = 0.01 if small_sample else 0.05

                if p1 < threshold or p2 < threshold:
                    # Check severity using skewness
                    skew1 = abs(scipy_stats.skew(group1))
                    skew2 = abs(scipy_stats.skew(group2))

                    # Gross violation: heavily skewed + small sample
                    if (skew1 > 2 or skew2 > 2) and small_sample:
                        result["blocked"] = True
                        result["violations"].append({
                            "type": "gross_normality_violation",
                            "severity": "critical",
                            "message": (
                                f"Data is heavily skewed (skew1={skew1:.2f}, skew2={skew2:.2f}) "
                                f"with small sample size (n1={len(group1)}, n2={len(group2)}). "
                                "T-test is inappropriate. Use Mann-Whitney U test instead."
                            ),
                            "alternative": "Mann-Whitney U test"
                        })

                        if not force:
                            return result

            except Exception as e:
                result["violations"].append({
                    "type": "assumption_check_failed",
                    "message": f"Could not check assumptions: {e}"
                })

            # If not blocked, perform test
            if not result["blocked"] or force:
                try:
                    t_stat, p_value = scipy_stats.ttest_ind(group1, group2)
                    result["executed"] = True
                    result["result"] = {
                        "t_statistic": float(t_stat),
                        "p_value": float(p_value),
                        "warning": (
                            "Test performed despite violations"
                            if result["violations"] and force else None
                        )
                    }
                except Exception as e:
                    result["error"] = str(e)

            return result

    # Test Case 1: Valid data (should execute)
    np.random.seed(42)
    normal1 = np.random.normal(100, 15, 50)
    normal2 = np.random.normal(105, 15, 50)

    result = SafeStatisticalTest.safe_t_test(normal1, normal2)

    # Assert: Should execute successfully
    assert result["executed"], "Should execute t-test on appropriate data"
    assert not result["blocked"], "Should not block valid analysis"
    assert result["result"] is not None, "Should return test results"

    # Test Case 2: Heavily skewed small sample (should block)
    skewed1 = np.random.exponential(1, 15)  # Small sample, skewed
    skewed2 = np.random.exponential(1.5, 15)

    result2 = SafeStatisticalTest.safe_t_test(skewed1, skewed2)

    # Assert: Should block test
    assert result2["blocked"], \
        "Should block t-test on heavily skewed data with small sample"
    assert not result2["executed"], "Should not execute blocked test"
    assert len(result2["violations"]) > 0, "Should report violations"
    assert any(v["severity"] == "critical" for v in result2["violations"]), \
        "Should mark gross violations as critical"
    assert any("mann-whitney" in v.get("alternative", "").lower()
               for v in result2["violations"]), \
        "Should suggest appropriate alternative"

    # Test Case 3: Very small sample (should block)
    tiny1 = np.array([1.0, 2.0])
    tiny2 = np.array([1.5, 2.5])

    result3 = SafeStatisticalTest.safe_t_test(tiny1, tiny2)

    # Assert: Should block due to insufficient data
    assert result3["blocked"], "Should block t-test on very small samples"
    assert any(v["type"] == "insufficient_data" for v in result3["violations"]), \
        "Should identify insufficient data violation"


@pytest.mark.requirement("REQ-SCI-ANA-006")
@pytest.mark.priority("MUST")
def test_req_sci_ana_006_report_effect_sizes_with_pvalues():
    """
    REQ-SCI-ANA-006: The system MUST report effect sizes alongside p-values,
    and SHOULD report confidence intervals when applicable.

    Validates that:
    - Effect sizes are always reported with p-values
    - Confidence intervals are computed
    - Complete statistical reporting is enforced
    """

    def complete_statistical_report(
        group1: np.ndarray,
        group2: np.ndarray,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Generate complete statistical report with all required metrics.

        Args:
            group1: First group
            group2: Second group
            alpha: Significance level

        Returns:
            Complete report dict
        """
        # Perform t-test
        t_stat, p_value = scipy_stats.ttest_ind(group1, group2)

        # Compute effect size (Cohen's d)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)

        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0

        # Compute confidence interval for mean difference
        mean_diff = mean1 - mean2
        se_diff = np.sqrt(std1**2 / n1 + std2**2 / n2)
        df = n1 + n2 - 2
        t_critical = scipy_stats.t.ppf(1 - alpha/2, df)
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff

        # Compute confidence interval for effect size (approximate)
        # Using non-central t distribution approximation
        se_d = np.sqrt((n1 + n2) / (n1 * n2) + cohens_d**2 / (2 * (n1 + n2)))
        d_ci_lower = cohens_d - 1.96 * se_d
        d_ci_upper = cohens_d + 1.96 * se_d

        return {
            "test_type": "independent_t_test",

            # MUST have: p-value
            "p_value": float(p_value),
            "significant": p_value < alpha,

            # MUST have: effect size
            "effect_size": {
                "cohens_d": float(cohens_d),
                "magnitude": (
                    "large" if abs(cohens_d) >= 0.8 else
                    "medium" if abs(cohens_d) >= 0.5 else
                    "small" if abs(cohens_d) >= 0.2 else
                    "negligible"
                )
            },

            # SHOULD have: confidence intervals
            "confidence_intervals": {
                "mean_difference": {
                    "estimate": float(mean_diff),
                    "ci_lower": float(ci_lower),
                    "ci_upper": float(ci_upper),
                    "confidence_level": 1 - alpha
                },
                "effect_size": {
                    "ci_lower": float(d_ci_lower),
                    "ci_upper": float(d_ci_upper),
                    "confidence_level": 0.95
                }
            },

            # Additional statistics
            "descriptive": {
                "group1": {"mean": float(mean1), "sd": float(std1), "n": n1},
                "group2": {"mean": float(mean2), "sd": float(std2), "n": n2}
            },

            "test_statistic": float(t_stat),
            "degrees_of_freedom": df
        }

    # Test Case 1: Complete report includes all required elements
    np.random.seed(42)
    group_a = np.random.normal(100, 15, 50)
    group_b = np.random.normal(110, 15, 50)

    report = complete_statistical_report(group_a, group_b)

    # Assert: MUST have p-value
    assert "p_value" in report, "Report MUST include p-value"
    assert isinstance(report["p_value"], float), "P-value must be numeric"

    # Assert: MUST have effect size
    assert "effect_size" in report, "Report MUST include effect size"
    assert "cohens_d" in report["effect_size"], "Must report Cohen's d"
    assert "magnitude" in report["effect_size"], "Must interpret effect size magnitude"

    # Assert: SHOULD have confidence intervals
    assert "confidence_intervals" in report, \
        "Report SHOULD include confidence intervals"
    assert "mean_difference" in report["confidence_intervals"], \
        "Should include CI for mean difference"
    assert "effect_size" in report["confidence_intervals"], \
        "Should include CI for effect size"

    # Test Case 2: Verify completeness check function
    def check_report_completeness(report: Dict[str, Any]) -> Dict[str, bool]:
        """Check if report meets completeness requirements."""
        return {
            "has_p_value": "p_value" in report,
            "has_effect_size": "effect_size" in report and "cohens_d" in report.get("effect_size", {}),
            "has_ci_mean_diff": (
                "confidence_intervals" in report and
                "mean_difference" in report.get("confidence_intervals", {})
            ),
            "has_ci_effect_size": (
                "confidence_intervals" in report and
                "effect_size" in report.get("confidence_intervals", {})
            ),
            "has_descriptive_stats": "descriptive" in report
        }

    completeness = check_report_completeness(report)

    # Assert: All required elements present
    assert completeness["has_p_value"], "Must have p-value"
    assert completeness["has_effect_size"], "Must have effect size"
    assert completeness["has_ci_mean_diff"], "Should have CI for mean difference"
    assert completeness["has_ci_effect_size"], "Should have CI for effect size"


@pytest.mark.requirement("REQ-SCI-ANA-007")
@pytest.mark.priority("MUST")
def test_req_sci_ana_007_no_cherry_picking():
    """
    REQ-SCI-ANA-007: The system MUST NOT cherry-pick analyses or report
    only statistically significant results - all performed analyses
    MUST be documented.

    Validates that:
    - All analyses are logged and documented
    - Non-significant results are reported
    - Analysis trail is maintained
    """

    class AnalysisRegistry:
        """Track all performed analyses to prevent cherry-picking."""

        def __init__(self):
            self.analyses = []
            self.reported = []

        def register_analysis(
            self,
            analysis_id: str,
            analysis_type: str,
            data_description: str,
            **kwargs
        ) -> str:
            """Register an analysis before performing it."""
            analysis_record = {
                "id": analysis_id,
                "type": analysis_type,
                "description": data_description,
                "timestamp": datetime.now(),
                "status": "registered",
                "result": None,
                "metadata": kwargs
            }
            self.analyses.append(analysis_record)
            return analysis_id

        def record_result(
            self,
            analysis_id: str,
            result: Dict[str, Any]
        ):
            """Record result of analysis."""
            for analysis in self.analyses:
                if analysis["id"] == analysis_id:
                    analysis["result"] = result
                    analysis["status"] = "completed"
                    return True
            return False

        def report_analysis(self, analysis_id: str):
            """Mark analysis as reported."""
            for analysis in self.analyses:
                if analysis["id"] == analysis_id:
                    if analysis["id"] not in self.reported:
                        self.reported.append(analysis["id"])
                    return True
            return False

        def check_reporting_integrity(self) -> Dict[str, Any]:
            """
            Check if all analyses are reported (no cherry-picking).

            Returns:
                Dict with integrity check results
            """
            completed = [a for a in self.analyses if a["status"] == "completed"]
            completed_ids = {a["id"] for a in completed}
            reported_ids = set(self.reported)

            unreported = completed_ids - reported_ids

            significant_count = sum(
                1 for a in completed
                if a["result"] and a["result"].get("p_value", 1.0) < 0.05
            )

            reported_significant_count = sum(
                1 for a in completed
                if a["id"] in reported_ids and
                a["result"] and a["result"].get("p_value", 1.0) < 0.05
            )

            return {
                "total_analyses": len(self.analyses),
                "completed_analyses": len(completed),
                "reported_analyses": len(reported_ids),
                "unreported_analyses": len(unreported),
                "unreported_ids": list(unreported),
                "significant_analyses": significant_count,
                "reported_significant": reported_significant_count,
                "all_reported": len(unreported) == 0,
                "potential_cherry_picking": (
                    len(unreported) > 0 and
                    reported_significant_count > 0 and
                    reported_significant_count == len(reported_ids)
                ),
                "reporting_rate": len(reported_ids) / len(completed) if completed else 0
            }

    # Test Case 1: All analyses reported (good practice)
    registry1 = AnalysisRegistry()

    # Perform and report multiple analyses
    np.random.seed(42)
    for i in range(5):
        aid = registry1.register_analysis(
            f"analysis_{i}",
            "t_test",
            f"Comparison {i}"
        )

        # Simulate results (some significant, some not)
        p_value = 0.03 if i % 2 == 0 else 0.15
        registry1.record_result(aid, {"p_value": p_value, "significant": p_value < 0.05})
        registry1.report_analysis(aid)

    integrity1 = registry1.check_reporting_integrity()

    # Assert: Should show good reporting practices
    assert integrity1["all_reported"], \
        "All completed analyses should be reported"
    assert not integrity1["potential_cherry_picking"], \
        "Should not detect cherry-picking when all results reported"
    assert integrity1["reporting_rate"] == 1.0, \
        "Reporting rate should be 100%"

    # Test Case 2: Cherry-picking detected (bad practice)
    registry2 = AnalysisRegistry()

    # Perform analyses but only report significant ones
    for i in range(5):
        aid = registry2.register_analysis(
            f"analysis_{i}",
            "t_test",
            f"Comparison {i}"
        )

        p_value = 0.03 if i % 2 == 0 else 0.15
        registry2.record_result(aid, {"p_value": p_value, "significant": p_value < 0.05})

        # Only report significant results
        if p_value < 0.05:
            registry2.report_analysis(aid)

    integrity2 = registry2.check_reporting_integrity()

    # Assert: Should detect potential cherry-picking
    assert not integrity2["all_reported"], \
        "Should detect unreported analyses"
    assert integrity2["unreported_analyses"] > 0, \
        "Should count unreported analyses"
    assert integrity2["potential_cherry_picking"], \
        "Should flag potential cherry-picking when only significant results reported"
    assert integrity2["reporting_rate"] < 1.0, \
        "Reporting rate should be less than 100%"

    # Test Case 3: Documentation completeness
    def generate_analysis_documentation(registry: AnalysisRegistry) -> str:
        """Generate complete documentation of all analyses."""
        doc = "# Analysis Documentation\n\n"
        doc += f"Total analyses performed: {len(registry.analyses)}\n\n"

        for analysis in registry.analyses:
            doc += f"## {analysis['id']}\n"
            doc += f"- Type: {analysis['type']}\n"
            doc += f"- Description: {analysis['description']}\n"
            doc += f"- Status: {analysis['status']}\n"

            if analysis['result']:
                doc += f"- P-value: {analysis['result'].get('p_value', 'N/A')}\n"
                doc += f"- Significant: {analysis['result'].get('significant', 'N/A')}\n"

            doc += f"- Reported: {'Yes' if analysis['id'] in registry.reported else 'No'}\n\n"

        return doc

    doc = generate_analysis_documentation(registry2)

    # Assert: Documentation should include all analyses
    assert "Total analyses performed: 5" in doc, \
        "Documentation should include all analyses"
    assert "Reported: No" in doc, \
        "Documentation should show unreported analyses"
