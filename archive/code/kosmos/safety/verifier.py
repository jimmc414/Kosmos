"""
Result verification and validation.

Implements:
- Sanity checks for statistical results
- Outlier detection
- Cross-validation of findings
- Error detection in analysis
- Verification reports
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from scipy import stats as scipy_stats

from kosmos.models.result import ExperimentResult, ResultStatus
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class VerificationIssue(BaseModel):
    """An issue detected during verification."""

    severity: str  # "warning", "error", "critical"
    category: str  # "sanity", "outlier", "statistical", "consistency"
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class VerificationReport(BaseModel):
    """Report from result verification."""

    result_id: str
    passed: bool
    issues: List[VerificationIssue] = Field(default_factory=list)
    checks_performed: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

    def has_errors(self) -> bool:
        """Check if report has any errors."""
        return any(issue.severity in ["error", "critical"] for issue in self.issues)

    def has_warnings(self) -> bool:
        """Check if report has any warnings."""
        return any(issue.severity == "warning" for issue in self.issues)

    def get_issues_by_severity(self, severity: str) -> List[VerificationIssue]:
        """Get issues of a specific severity."""
        return [issue for issue in self.issues if issue.severity == severity]

    def get_issues_by_category(self, category: str) -> List[VerificationIssue]:
        """Get issues of a specific category."""
        return [issue for issue in self.issues if issue.category == category]

    def summary(self) -> str:
        """Generate human-readable summary."""
        if self.passed and not self.issues:
            return f"✓ Verification passed (no issues)"

        error_count = len(self.get_issues_by_severity("error")) + len(self.get_issues_by_severity("critical"))
        warning_count = len(self.get_issues_by_severity("warning"))

        if self.passed:
            return f"⚠ Verification passed with warnings: {warning_count} warnings"
        else:
            return f"✗ Verification failed: {error_count} errors, {warning_count} warnings"


class ResultVerifier:
    """
    Verifies experiment results for validity and consistency.

    Performs sanity checks, outlier detection, and cross-validation.
    """

    def __init__(
        self,
        enable_sanity_checks: bool = True,
        enable_outlier_detection: bool = True,
        enable_statistical_validation: bool = True,
        outlier_threshold: float = 3.0,  # Z-score threshold
        min_sample_size: int = 10
    ):
        """
        Initialize result verifier.

        Args:
            enable_sanity_checks: Enable sanity checks
            enable_outlier_detection: Enable outlier detection
            enable_statistical_validation: Enable statistical validation
            outlier_threshold: Z-score threshold for outlier detection
            min_sample_size: Minimum sample size for statistical tests
        """
        self.enable_sanity_checks = enable_sanity_checks
        self.enable_outlier_detection = enable_outlier_detection
        self.enable_statistical_validation = enable_statistical_validation
        self.outlier_threshold = outlier_threshold
        self.min_sample_size = min_sample_size

        logger.info(
            f"ResultVerifier initialized (sanity={enable_sanity_checks}, "
            f"outliers={enable_outlier_detection}, stats={enable_statistical_validation})"
        )

    def verify(self, result: ExperimentResult) -> VerificationReport:
        """
        Verify an experiment result.

        Args:
            result: Experiment result to verify

        Returns:
            VerificationReport with verification results
        """
        issues = []
        checks_performed = []

        # 1. Sanity checks
        if self.enable_sanity_checks:
            sanity_issues = self._check_sanity(result)
            issues.extend(sanity_issues)
            checks_performed.append("sanity_checks")

        # 2. Outlier detection
        if self.enable_outlier_detection:
            outlier_issues = self._detect_outliers(result)
            issues.extend(outlier_issues)
            checks_performed.append("outlier_detection")

        # 3. Statistical validation
        if self.enable_statistical_validation:
            stat_issues = self._validate_statistics(result)
            issues.extend(stat_issues)
            checks_performed.append("statistical_validation")

        # 4. Consistency checks
        consistency_issues = self._check_consistency(result)
        issues.extend(consistency_issues)
        checks_performed.append("consistency_checks")

        # Determine if passed (no errors or critical issues)
        passed = not any(
            issue.severity in ["error", "critical"]
            for issue in issues
        )

        report = VerificationReport(
            result_id=result.id or result.experiment_id,
            passed=passed,
            issues=issues,
            checks_performed=checks_performed,
            metadata={
                "result_status": result.status.value,
                "has_statistical_tests": len(result.statistical_tests) > 0,
                "hypothesis_id": result.hypothesis_id
            }
        )

        logger.info(f"Result verification: {report.summary()}")

        return report

    def _check_sanity(self, result: ExperimentResult) -> List[VerificationIssue]:
        """Perform sanity checks on result."""
        issues = []

        # Check p-value range
        if result.primary_p_value is not None:
            if not (0 <= result.primary_p_value <= 1):
                issues.append(VerificationIssue(
                    severity="error",
                    category="sanity",
                    message=f"P-value out of range [0, 1]: {result.primary_p_value}",
                    details={"p_value": result.primary_p_value}
                ))

        # Check for inf/nan in primary metrics
        if result.primary_effect_size is not None:
            if np.isnan(result.primary_effect_size) or np.isinf(result.primary_effect_size):
                issues.append(VerificationIssue(
                    severity="error",
                    category="sanity",
                    message=f"Effect size is NaN or Inf: {result.primary_effect_size}",
                    details={"effect_size": result.primary_effect_size}
                ))

        # Check statistical test results
        for test in result.statistical_tests:
            if test.p_value is not None and not (0 <= test.p_value <= 1):
                issues.append(VerificationIssue(
                    severity="error",
                    category="sanity",
                    message=f"Test '{test.test_name}' has invalid p-value: {test.p_value}",
                    details={"test_name": test.test_name, "p_value": test.p_value}
                ))

        # Check for empty results
        if result.status == ResultStatus.SUCCESS:
            if not result.raw_data and not result.processed_data:
                issues.append(VerificationIssue(
                    severity="warning",
                    category="sanity",
                    message="Successful result has no data",
                    details={}
                ))

        # Check for contradictions
        if result.supports_hypothesis is not None and result.primary_p_value is not None:
            # If p < 0.05, should generally support hypothesis
            if result.primary_p_value < 0.05 and result.supports_hypothesis is False:
                issues.append(VerificationIssue(
                    severity="warning",
                    category="sanity",
                    message="Significant p-value but hypothesis not supported",
                    details={
                        "p_value": result.primary_p_value,
                        "supports_hypothesis": result.supports_hypothesis
                    }
                ))

        return issues

    def _detect_outliers(self, result: ExperimentResult) -> List[VerificationIssue]:
        """Detect outliers in result data."""
        issues = []

        # Check processed data for numeric outliers
        if result.processed_data:
            for key, value in result.processed_data.items():
                if isinstance(value, (list, np.ndarray)):
                    try:
                        data = np.array(value, dtype=float)
                        if len(data) > 0:
                            outliers = self._find_outliers(data)
                            if len(outliers) > 0:
                                outlier_ratio = len(outliers) / len(data)
                                severity = "error" if outlier_ratio > 0.2 else "warning"
                                issues.append(VerificationIssue(
                                    severity=severity,
                                    category="outlier",
                                    message=f"Outliers detected in '{key}': {len(outliers)} / {len(data)}",
                                    details={
                                        "field": key,
                                        "outlier_count": len(outliers),
                                        "total_count": len(data),
                                        "outlier_ratio": outlier_ratio,
                                        "outlier_indices": outliers.tolist()
                                    }
                                ))
                    except (ValueError, TypeError):
                        # Not numeric data, skip
                        pass

        # Check variable results for outliers
        for var_result in result.variable_results:
            if var_result.value is not None:
                values = var_result.value if isinstance(var_result.value, list) else [var_result.value]
                try:
                    data = np.array(values, dtype=float)
                    if len(data) > 1:
                        outliers = self._find_outliers(data)
                        if len(outliers) > 0:
                            issues.append(VerificationIssue(
                                severity="warning",
                                category="outlier",
                                message=f"Outliers in variable '{var_result.name}': {len(outliers)} / {len(data)}",
                                details={
                                    "variable": var_result.name,
                                    "outlier_count": len(outliers),
                                    "total_count": len(data)
                                }
                            ))
                except (ValueError, TypeError):
                    pass

        return issues

    def _find_outliers(self, data: np.ndarray) -> np.ndarray:
        """Find outliers using Z-score method."""
        if len(data) < 3:
            return np.array([])

        # Calculate Z-scores
        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return np.array([])

        z_scores = np.abs((data - mean) / std)

        # Find indices where Z-score exceeds threshold
        outlier_indices = np.where(z_scores > self.outlier_threshold)[0]

        return outlier_indices

    def _validate_statistics(self, result: ExperimentResult) -> List[VerificationIssue]:
        """Validate statistical test results."""
        issues = []

        # Check sample sizes
        for test in result.statistical_tests:
            sample_size = test.details.get('sample_size')
            if sample_size is not None and sample_size < self.min_sample_size:
                issues.append(VerificationIssue(
                    severity="warning",
                    category="statistical",
                    message=f"Test '{test.test_name}' has small sample size: {sample_size}",
                    details={"test_name": test.test_name, "sample_size": sample_size}
                ))

        # Check effect size and significance alignment
        if result.primary_p_value is not None and result.primary_effect_size is not None:
            # Large effect size but not significant -> underpowered
            if abs(result.primary_effect_size) > 0.5 and result.primary_p_value > 0.05:
                issues.append(VerificationIssue(
                    severity="warning",
                    category="statistical",
                    message="Large effect size but not statistically significant (underpowered study)",
                    details={
                        "effect_size": result.primary_effect_size,
                        "p_value": result.primary_p_value
                    }
                ))

            # Small effect size but significant -> overpowered or noise
            if abs(result.primary_effect_size) < 0.1 and result.primary_p_value < 0.05:
                issues.append(VerificationIssue(
                    severity="warning",
                    category="statistical",
                    message="Statistically significant but small effect size (may not be practically significant)",
                    details={
                        "effect_size": result.primary_effect_size,
                        "p_value": result.primary_p_value
                    }
                ))

        # Check for multiple testing without correction
        if len(result.statistical_tests) > 1:
            # Check if any test mentions correction
            has_correction = any(
                "bonferroni" in test.test_name.lower() or
                "fdr" in test.test_name.lower() or
                "holm" in test.test_name.lower()
                for test in result.statistical_tests
            )

            if not has_correction:
                issues.append(VerificationIssue(
                    severity="warning",
                    category="statistical",
                    message=f"Multiple tests ({len(result.statistical_tests)}) without multiple testing correction",
                    details={"test_count": len(result.statistical_tests)}
                ))

        return issues

    def _check_consistency(self, result: ExperimentResult) -> List[VerificationIssue]:
        """Check internal consistency of results."""
        issues = []

        # Check status matches data
        if result.status == ResultStatus.SUCCESS:
            if result.stderr and "error" in result.stderr.lower():
                issues.append(VerificationIssue(
                    severity="warning",
                    category="consistency",
                    message="Result marked as SUCCESS but stderr contains errors",
                    details={"stderr_preview": result.stderr[:200]}
                ))

        if result.status == ResultStatus.FAILED:
            if result.primary_p_value is not None or result.supports_hypothesis is not None:
                issues.append(VerificationIssue(
                    severity="warning",
                    category="consistency",
                    message="Result marked as FAILED but has statistical results",
                    details={}
                ))

        # Check supports_hypothesis matches p_value
        if result.supports_hypothesis is True and result.primary_p_value is not None:
            if result.primary_p_value > 0.05:
                issues.append(VerificationIssue(
                    severity="warning",
                    category="consistency",
                    message="Hypothesis marked as supported but p-value not significant",
                    details={
                        "supports_hypothesis": True,
                        "p_value": result.primary_p_value
                    }
                ))

        # Check primary test exists if primary p-value exists
        if result.primary_p_value is not None and not result.primary_test:
            issues.append(VerificationIssue(
                severity="warning",
                category="consistency",
                message="Primary p-value present but no primary test specified",
                details={"primary_p_value": result.primary_p_value}
            ))

        return issues

    def cross_validate(
        self,
        original: ExperimentResult,
        replication: ExperimentResult
    ) -> VerificationReport:
        """
        Cross-validate results from replicated experiments.

        Args:
            original: Original experiment result
            replication: Replication experiment result

        Returns:
            VerificationReport with cross-validation results
        """
        issues = []
        checks_performed = ["cross_validation"]

        # Check p-values are consistent
        if original.primary_p_value is not None and replication.primary_p_value is not None:
            # Both should be on same side of significance threshold
            orig_sig = original.primary_p_value < 0.05
            repl_sig = replication.primary_p_value < 0.05

            if orig_sig != repl_sig:
                issues.append(VerificationIssue(
                    severity="error",
                    category="consistency",
                    message="Replication failed: significance differs",
                    details={
                        "original_p": original.primary_p_value,
                        "replication_p": replication.primary_p_value
                    }
                ))

        # Check effect sizes are consistent (within 50% of each other)
        if original.primary_effect_size is not None and replication.primary_effect_size is not None:
            ratio = abs(original.primary_effect_size) / (abs(replication.primary_effect_size) + 1e-10)
            if ratio < 0.5 or ratio > 2.0:
                issues.append(VerificationIssue(
                    severity="warning",
                    category="consistency",
                    message="Effect sizes differ substantially between original and replication",
                    details={
                        "original_effect": original.primary_effect_size,
                        "replication_effect": replication.primary_effect_size,
                        "ratio": ratio
                    }
                ))

        # Check hypothesis support is consistent
        if original.supports_hypothesis is not None and replication.supports_hypothesis is not None:
            if original.supports_hypothesis != replication.supports_hypothesis:
                issues.append(VerificationIssue(
                    severity="error",
                    category="consistency",
                    message="Replication failed: hypothesis support differs",
                    details={
                        "original_supports": original.supports_hypothesis,
                        "replication_supports": replication.supports_hypothesis
                    }
                ))

        passed = not any(issue.severity in ["error", "critical"] for issue in issues)

        return VerificationReport(
            result_id=f"{original.id or original.experiment_id}_vs_{replication.id or replication.experiment_id}",
            passed=passed,
            issues=issues,
            checks_performed=checks_performed,
            metadata={
                "original_id": original.id or original.experiment_id,
                "replication_id": replication.id or replication.experiment_id,
                "cross_validation": True
            }
        )

    def detect_errors(self, result: ExperimentResult) -> List[str]:
        """
        Detect errors in analysis output.

        Args:
            result: Experiment result to check

        Returns:
            List of detected error messages
        """
        errors = []

        # Check stderr
        if result.stderr:
            stderr_lower = result.stderr.lower()
            error_keywords = [
                "error", "exception", "traceback", "failed",
                "critical", "fatal", "abort"
            ]
            for keyword in error_keywords:
                if keyword in stderr_lower:
                    errors.append(f"Stderr contains '{keyword}'")
                    break

        # Check stdout for error indicators
        if result.stdout:
            stdout_lower = result.stdout.lower()
            if "nan" in stdout_lower or "inf" in stdout_lower:
                errors.append("Output contains NaN or Inf values")

        # Check for failed status
        if result.status == ResultStatus.FAILED:
            errors.append(f"Result status is FAILED")

        # Check metadata for errors
        if hasattr(result, 'metadata') and result.metadata:
            error_msg = result.metadata.error_message if hasattr(result.metadata, 'error_message') else None
            if error_msg:
                errors.append(f"Metadata error: {error_msg}")

        return errors
