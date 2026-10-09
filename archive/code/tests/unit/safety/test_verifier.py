"""
Tests for result verifier.

Tests sanity checks, outlier detection, statistical validation, and cross-validation.
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock

from kosmos.safety.verifier import ResultVerifier, VerificationReport, VerificationIssue
from kosmos.models.result import (
    ExperimentResult, ResultStatus, StatisticalTestResult,
    VariableResult, ExecutionMetadata
)


@pytest.fixture
def basic_metadata():
    """Create basic execution metadata."""
    return ExecutionMetadata(
        start_time=datetime.now(),
        end_time=datetime.now(),
        execution_time_seconds=10.0,
        library_versions={}
    )


@pytest.fixture
def valid_result(basic_metadata):
    """Create a valid experiment result."""
    return ExperimentResult(
        experiment_id="exp_123",
        protocol_id="proto_456",
        status=ResultStatus.SUCCESS,
        primary_p_value=0.03,
        primary_effect_size=0.5,
        supports_hypothesis=True,
        metadata=basic_metadata,
        raw_data={"values": [1, 2, 3, 4, 5]},
        processed_data={"mean": 3.0, "std": 1.41}
    )


class TestResultVerifierInitialization:
    """Tests for ResultVerifier initialization."""

    def test_init_default_settings(self):
        """Test initialization with default settings."""
        verifier = ResultVerifier()

        assert verifier.enable_sanity_checks is True
        assert verifier.enable_outlier_detection is True
        assert verifier.enable_statistical_validation is True
        assert verifier.outlier_threshold == 3.0
        assert verifier.min_sample_size == 10

    def test_init_custom_settings(self):
        """Test initialization with custom settings."""
        verifier = ResultVerifier(
            enable_sanity_checks=False,
            enable_outlier_detection=False,
            outlier_threshold=2.5,
            min_sample_size=20
        )

        assert verifier.enable_sanity_checks is False
        assert verifier.enable_outlier_detection is False
        assert verifier.outlier_threshold == 2.5
        assert verifier.min_sample_size == 20


class TestSanityChecks:
    """Tests for sanity checking."""

    def test_valid_result_passes_sanity_checks(self, valid_result):
        """Test that valid result passes sanity checks."""
        verifier = ResultVerifier()
        report = verifier.verify(valid_result)

        sanity_issues = report.get_issues_by_category("sanity")
        assert len(sanity_issues) == 0

    def test_invalid_p_value_detected(self, basic_metadata):
        """Test that invalid p-value is detected."""
        result = ExperimentResult(
            experiment_id="exp_123",
            protocol_id="proto_456",
            status=ResultStatus.SUCCESS,
            primary_p_value=1.5,  # Invalid!
            metadata=basic_metadata
        )

        verifier = ResultVerifier()
        report = verifier.verify(result)

        assert not report.passed
        sanity_issues = report.get_issues_by_category("sanity")
        assert len(sanity_issues) > 0
        assert any("p-value out of range" in issue.message.lower() for issue in sanity_issues)

    def test_nan_effect_size_detected(self, basic_metadata):
        """Test that NaN effect size is detected."""
        result = ExperimentResult(
            experiment_id="exp_123",
            protocol_id="proto_456",
            status=ResultStatus.SUCCESS,
            primary_effect_size=float('nan'),
            metadata=basic_metadata
        )

        verifier = ResultVerifier()
        report = verifier.verify(result)

        assert not report.passed
        sanity_issues = report.get_issues_by_category("sanity")
        assert any("nan or inf" in issue.message.lower() for issue in sanity_issues)

    def test_inf_effect_size_detected(self, basic_metadata):
        """Test that Inf effect size is detected."""
        result = ExperimentResult(
            experiment_id="exp_123",
            protocol_id="proto_456",
            status=ResultStatus.SUCCESS,
            primary_effect_size=float('inf'),
            metadata=basic_metadata
        )

        verifier = ResultVerifier()
        report = verifier.verify(result)

        assert not report.passed
        sanity_issues = report.get_issues_by_category("sanity")
        assert any("nan or inf" in issue.message.lower() for issue in sanity_issues)

    def test_empty_successful_result_generates_warning(self, basic_metadata):
        """Test that successful result with no data generates warning."""
        result = ExperimentResult(
            experiment_id="exp_123",
            protocol_id="proto_456",
            status=ResultStatus.SUCCESS,
            raw_data={},
            processed_data={},
            metadata=basic_metadata
        )

        verifier = ResultVerifier()
        report = verifier.verify(result)

        warnings = report.get_issues_by_severity("warning")
        assert any("no data" in warning.message.lower() for warning in warnings)

    def test_contradiction_detected(self, basic_metadata):
        """Test that contradictory results are detected."""
        result = ExperimentResult(
            experiment_id="exp_123",
            protocol_id="proto_456",
            status=ResultStatus.SUCCESS,
            primary_p_value=0.01,  # Significant
            supports_hypothesis=False,  # But not supported?
            metadata=basic_metadata
        )

        verifier = ResultVerifier()
        report = verifier.verify(result)

        warnings = report.get_issues_by_severity("warning")
        assert any("significant" in w.message.lower() and "not supported" in w.message.lower() for w in warnings)


class TestOutlierDetection:
    """Tests for outlier detection."""

    def test_no_outliers_in_clean_data(self, basic_metadata):
        """Test that clean data has no outliers."""
        result = ExperimentResult(
            experiment_id="exp_123",
            protocol_id="proto_456",
            status=ResultStatus.SUCCESS,
            processed_data={"values": [1.0, 1.1, 0.9, 1.2, 0.8]},  # No outliers
            metadata=basic_metadata
        )

        verifier = ResultVerifier()
        report = verifier.verify(result)

        outlier_issues = report.get_issues_by_category("outlier")
        assert len(outlier_issues) == 0

    def test_outliers_detected_in_data(self, basic_metadata):
        """Test that outliers are detected."""
        # Create data with outlier
        data = [1.0] * 20 + [100.0]  # 100.0 is clear outlier

        result = ExperimentResult(
            experiment_id="exp_123",
            protocol_id="proto_456",
            status=ResultStatus.SUCCESS,
            processed_data={"values": data},
            metadata=basic_metadata
        )

        verifier = ResultVerifier()
        report = verifier.verify(result)

        outlier_issues = report.get_issues_by_category("outlier")
        assert len(outlier_issues) > 0
        assert any("outliers detected" in issue.message.lower() for issue in outlier_issues)

    def test_many_outliers_generate_error(self, basic_metadata):
        """Test that many outliers generate error severity."""
        # Create data with >20% outliers
        data = [1.0] * 8 + [100.0] * 3  # 3/11 = 27% outliers

        result = ExperimentResult(
            experiment_id="exp_123",
            protocol_id="proto_456",
            status=ResultStatus.SUCCESS,
            processed_data={"values": data},
            metadata=basic_metadata
        )

        verifier = ResultVerifier()
        report = verifier.verify(result)

        errors = report.get_issues_by_severity("error")
        assert any("outlier" in e.message.lower() for e in errors)

    def test_outlier_detection_skips_non_numeric_data(self, basic_metadata):
        """Test that outlier detection skips non-numeric data."""
        result = ExperimentResult(
            experiment_id="exp_123",
            protocol_id="proto_456",
            status=ResultStatus.SUCCESS,
            processed_data={"labels": ["a", "b", "c"]},  # Non-numeric
            metadata=basic_metadata
        )

        verifier = ResultVerifier()
        report = verifier.verify(result)

        # Should not crash, should have no outlier issues
        outlier_issues = report.get_issues_by_category("outlier")
        assert len(outlier_issues) == 0

    def test_outlier_detection_handles_empty_data(self, basic_metadata):
        """Test that outlier detection handles empty data gracefully."""
        result = ExperimentResult(
            experiment_id="exp_123",
            protocol_id="proto_456",
            status=ResultStatus.SUCCESS,
            processed_data={"values": []},
            metadata=basic_metadata
        )

        verifier = ResultVerifier()
        report = verifier.verify(result)

        # Should not crash
        assert report is not None


class TestStatisticalValidation:
    """Tests for statistical validation."""

    def test_small_sample_size_warning(self, basic_metadata):
        """Test that small sample size generates warning."""
        result = ExperimentResult(
            experiment_id="exp_123",
            protocol_id="proto_456",
            status=ResultStatus.SUCCESS,
            statistical_tests=[
                StatisticalTestResult(
                    test_name="t-test",
                    statistic=2.5,
                    p_value=0.03,
                    details={"sample_size": 5}  # Too small
                )
            ],
            metadata=basic_metadata
        )

        verifier = ResultVerifier(min_sample_size=10)
        report = verifier.verify(result)

        warnings = report.get_issues_by_severity("warning")
        assert any("small sample size" in w.message.lower() for w in warnings)

    def test_large_effect_but_not_significant_warning(self, basic_metadata):
        """Test warning for large effect but not significant (underpowered)."""
        result = ExperimentResult(
            experiment_id="exp_123",
            protocol_id="proto_456",
            status=ResultStatus.SUCCESS,
            primary_p_value=0.10,  # Not significant
            primary_effect_size=0.8,  # Large effect
            metadata=basic_metadata
        )

        verifier = ResultVerifier()
        report = verifier.verify(result)

        warnings = report.get_issues_by_severity("warning")
        assert any("underpowered" in w.message.lower() for w in warnings)

    def test_small_effect_but_significant_warning(self, basic_metadata):
        """Test warning for small effect but significant."""
        result = ExperimentResult(
            experiment_id="exp_123",
            protocol_id="proto_456",
            status=ResultStatus.SUCCESS,
            primary_p_value=0.01,  # Significant
            primary_effect_size=0.05,  # Small effect
            metadata=basic_metadata
        )

        verifier = ResultVerifier()
        report = verifier.verify(result)

        warnings = report.get_issues_by_severity("warning")
        assert any("small effect size" in w.message.lower() for w in warnings)

    def test_multiple_testing_without_correction_warning(self, basic_metadata):
        """Test warning for multiple tests without correction."""
        result = ExperimentResult(
            experiment_id="exp_123",
            protocol_id="proto_456",
            status=ResultStatus.SUCCESS,
            statistical_tests=[
                StatisticalTestResult(test_name="t-test-1", statistic=2.0, p_value=0.04),
                StatisticalTestResult(test_name="t-test-2", statistic=2.5, p_value=0.03),
                StatisticalTestResult(test_name="t-test-3", statistic=1.8, p_value=0.07)
            ],
            metadata=basic_metadata
        )

        verifier = ResultVerifier()
        report = verifier.verify(result)

        warnings = report.get_issues_by_severity("warning")
        assert any("multiple testing correction" in w.message.lower() or "multiple tests" in w.message.lower() for w in warnings)

    def test_no_warning_with_correction_method(self, basic_metadata):
        """Test no warning when correction method is used."""
        result = ExperimentResult(
            experiment_id="exp_123",
            protocol_id="proto_456",
            status=ResultStatus.SUCCESS,
            statistical_tests=[
                StatisticalTestResult(test_name="t-test-bonferroni", statistic=2.0, p_value=0.01),
                StatisticalTestResult(test_name="t-test-bonferroni-2", statistic=2.5, p_value=0.02)
            ],
            metadata=basic_metadata
        )

        verifier = ResultVerifier()
        report = verifier.verify(result)

        warnings = report.get_issues_by_severity("warning")
        multiple_testing_warnings = [w for w in warnings if "multiple testing" in w.message.lower()]
        assert len(multiple_testing_warnings) == 0


class TestConsistencyChecks:
    """Tests for consistency checking."""

    def test_success_with_stderr_errors_warning(self, basic_metadata):
        """Test warning when SUCCESS status but stderr has errors."""
        result = ExperimentResult(
            experiment_id="exp_123",
            protocol_id="proto_456",
            status=ResultStatus.SUCCESS,
            stderr="ValueError: Something went wrong",
            metadata=basic_metadata
        )

        verifier = ResultVerifier()
        report = verifier.verify(result)

        warnings = report.get_issues_by_severity("warning")
        assert any("success" in w.message.lower() and "error" in w.message.lower() for w in warnings)

    def test_failed_with_statistical_results_warning(self, basic_metadata):
        """Test warning when FAILED status but has statistical results."""
        result = ExperimentResult(
            experiment_id="exp_123",
            protocol_id="proto_456",
            status=ResultStatus.FAILED,
            primary_p_value=0.05,  # Shouldn't have this if failed
            metadata=basic_metadata
        )

        verifier = ResultVerifier()
        report = verifier.verify(result)

        warnings = report.get_issues_by_severity("warning")
        assert any("failed" in w.message.lower() and "statistical" in w.message.lower() for w in warnings)

    def test_hypothesis_supported_but_not_significant_warning(self, basic_metadata):
        """Test warning when hypothesis supported but p-value not significant."""
        result = ExperimentResult(
            experiment_id="exp_123",
            protocol_id="proto_456",
            status=ResultStatus.SUCCESS,
            supports_hypothesis=True,
            primary_p_value=0.20,  # Not significant
            metadata=basic_metadata
        )

        verifier = ResultVerifier()
        report = verifier.verify(result)

        warnings = report.get_issues_by_severity("warning")
        assert any("supported" in w.message.lower() and "not significant" in w.message.lower() for w in warnings)


class TestCrossValidation:
    """Tests for cross-validation of replications."""

    def test_consistent_results_pass_cross_validation(self, basic_metadata):
        """Test that consistent results pass cross-validation."""
        original = ExperimentResult(
            experiment_id="exp_original",
            protocol_id="proto_456",
            status=ResultStatus.SUCCESS,
            primary_p_value=0.03,
            primary_effect_size=0.5,
            supports_hypothesis=True,
            metadata=basic_metadata
        )

        replication = ExperimentResult(
            experiment_id="exp_replication",
            protocol_id="proto_456",
            status=ResultStatus.SUCCESS,
            primary_p_value=0.04,  # Similar
            primary_effect_size=0.48,  # Similar
            supports_hypothesis=True,
            metadata=basic_metadata
        )

        verifier = ResultVerifier()
        report = verifier.cross_validate(original, replication)

        assert report.passed
        assert len(report.get_issues_by_severity("error")) == 0

    def test_different_significance_fails_cross_validation(self, basic_metadata):
        """Test that different significance fails cross-validation."""
        original = ExperimentResult(
            experiment_id="exp_original",
            protocol_id="proto_456",
            status=ResultStatus.SUCCESS,
            primary_p_value=0.03,  # Significant
            supports_hypothesis=True,
            metadata=basic_metadata
        )

        replication = ExperimentResult(
            experiment_id="exp_replication",
            protocol_id="proto_456",
            status=ResultStatus.SUCCESS,
            primary_p_value=0.10,  # Not significant
            supports_hypothesis=False,
            metadata=basic_metadata
        )

        verifier = ResultVerifier()
        report = verifier.cross_validate(original, replication)

        assert not report.passed
        errors = report.get_issues_by_severity("error")
        assert any("replication failed" in e.message.lower() for e in errors)

    def test_different_effect_sizes_generate_warning(self, basic_metadata):
        """Test that substantially different effect sizes generate warning."""
        original = ExperimentResult(
            experiment_id="exp_original",
            protocol_id="proto_456",
            status=ResultStatus.SUCCESS,
            primary_effect_size=0.8,  # Large
            metadata=basic_metadata
        )

        replication = ExperimentResult(
            experiment_id="exp_replication",
            protocol_id="proto_456",
            status=ResultStatus.SUCCESS,
            primary_effect_size=0.2,  # Much smaller
            metadata=basic_metadata
        )

        verifier = ResultVerifier()
        report = verifier.cross_validate(original, replication)

        warnings = report.get_issues_by_severity("warning")
        assert any("effect size" in w.message.lower() for w in warnings)

    def test_different_hypothesis_support_fails(self, basic_metadata):
        """Test that different hypothesis support fails cross-validation."""
        original = ExperimentResult(
            experiment_id="exp_original",
            protocol_id="proto_456",
            status=ResultStatus.SUCCESS,
            supports_hypothesis=True,
            metadata=basic_metadata
        )

        replication = ExperimentResult(
            experiment_id="exp_replication",
            protocol_id="proto_456",
            status=ResultStatus.SUCCESS,
            supports_hypothesis=False,
            metadata=basic_metadata
        )

        verifier = ResultVerifier()
        report = verifier.cross_validate(original, replication)

        assert not report.passed
        errors = report.get_issues_by_severity("error")
        assert any("hypothesis support differs" in e.message.lower() for e in errors)


class TestErrorDetection:
    """Tests for error detection in analysis output."""

    def test_detect_errors_in_stderr(self, basic_metadata):
        """Test error detection in stderr."""
        result = ExperimentResult(
            experiment_id="exp_123",
            protocol_id="proto_456",
            status=ResultStatus.SUCCESS,
            stderr="RuntimeError: Division by zero\nTraceback...",
            metadata=basic_metadata
        )

        verifier = ResultVerifier()
        errors = verifier.detect_errors(result)

        assert len(errors) > 0
        assert any("stderr" in e.lower() for e in errors)

    def test_detect_nan_in_stdout(self, basic_metadata):
        """Test detection of NaN in stdout."""
        result = ExperimentResult(
            experiment_id="exp_123",
            protocol_id="proto_456",
            status=ResultStatus.SUCCESS,
            stdout="Result: nan",
            metadata=basic_metadata
        )

        verifier = ResultVerifier()
        errors = verifier.detect_errors(result)

        assert any("nan" in e.lower() for e in errors)

    def test_detect_failed_status(self, basic_metadata):
        """Test detection of FAILED status."""
        result = ExperimentResult(
            experiment_id="exp_123",
            protocol_id="proto_456",
            status=ResultStatus.FAILED,
            metadata=basic_metadata
        )

        verifier = ResultVerifier()
        errors = verifier.detect_errors(result)

        assert any("failed" in e.lower() for e in errors)


class TestVerificationReportMethods:
    """Tests for VerificationReport helper methods."""

    def test_has_errors(self):
        """Test has_errors method."""
        report_with = VerificationReport(
            result_id="test",
            passed=False,
            issues=[
                VerificationIssue(
                    severity="error",
                    category="sanity",
                    message="Test error"
                )
            ]
        )
        report_without = VerificationReport(
            result_id="test",
            passed=True,
            issues=[]
        )

        assert report_with.has_errors() is True
        assert report_without.has_errors() is False

    def test_has_warnings(self):
        """Test has_warnings method."""
        report_with = VerificationReport(
            result_id="test",
            passed=True,
            issues=[
                VerificationIssue(
                    severity="warning",
                    category="sanity",
                    message="Test warning"
                )
            ]
        )
        report_without = VerificationReport(
            result_id="test",
            passed=True,
            issues=[]
        )

        assert report_with.has_warnings() is True
        assert report_without.has_warnings() is False

    def test_get_issues_by_severity(self):
        """Test get_issues_by_severity method."""
        report = VerificationReport(
            result_id="test",
            passed=False,
            issues=[
                VerificationIssue(severity="error", category="sanity", message="Error 1"),
                VerificationIssue(severity="warning", category="sanity", message="Warning 1"),
                VerificationIssue(severity="error", category="outlier", message="Error 2")
            ]
        )

        errors = report.get_issues_by_severity("error")
        warnings = report.get_issues_by_severity("warning")

        assert len(errors) == 2
        assert len(warnings) == 1

    def test_get_issues_by_category(self):
        """Test get_issues_by_category method."""
        report = VerificationReport(
            result_id="test",
            passed=False,
            issues=[
                VerificationIssue(severity="error", category="sanity", message="Error 1"),
                VerificationIssue(severity="warning", category="outlier", message="Warning 1"),
                VerificationIssue(severity="error", category="sanity", message="Error 2")
            ]
        )

        sanity = report.get_issues_by_category("sanity")
        outlier = report.get_issues_by_category("outlier")

        assert len(sanity) == 2
        assert len(outlier) == 1

    def test_summary_no_issues(self):
        """Test summary for report with no issues."""
        report = VerificationReport(
            result_id="test",
            passed=True,
            issues=[]
        )

        summary = report.summary()

        assert "✓" in summary or "passed" in summary.lower()
        assert "no issues" in summary.lower()

    def test_summary_with_warnings(self):
        """Test summary for report with warnings."""
        report = VerificationReport(
            result_id="test",
            passed=True,
            issues=[
                VerificationIssue(severity="warning", category="sanity", message="Warning")
            ]
        )

        summary = report.summary()

        assert "warning" in summary.lower()

    def test_summary_with_errors(self):
        """Test summary for failed report."""
        report = VerificationReport(
            result_id="test",
            passed=False,
            issues=[
                VerificationIssue(severity="error", category="sanity", message="Error"),
                VerificationIssue(severity="warning", category="sanity", message="Warning")
            ]
        )

        summary = report.summary()

        assert "✗" in summary or "failed" in summary.lower()
        assert "error" in summary.lower()
