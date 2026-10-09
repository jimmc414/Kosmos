"""
Unit tests for Paper Accuracy Validation (Issue #65).

Tests AccuracyTracker, AccuracyReporter, and related classes for
measuring accuracy by statement type against paper claims.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock
from typing import List

from kosmos.validation.accuracy_tracker import (
    AccuracyTarget,
    TypeAccuracyResult,
    AccuracyReport,
    AccuracyTracker,
    AccuracyReporter,
    ValidationRecord,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_finding():
    """Create a mock finding object."""
    finding = MagicMock()
    finding.finding_id = "test_001"
    finding.evidence_type = "data_analysis"
    finding.summary = "Test finding summary"
    return finding


@pytest.fixture
def sample_findings():
    """Create sample findings of different types."""
    findings = []

    # Data analysis findings
    for i in range(10):
        f = MagicMock()
        f.finding_id = f"da_{i:03d}"
        f.evidence_type = "data_analysis"
        f.summary = f"Data analysis finding {i}"
        findings.append(f)

    # Literature findings
    for i in range(10):
        f = MagicMock()
        f.finding_id = f"lit_{i:03d}"
        f.evidence_type = "literature"
        f.summary = f"Literature finding {i}"
        findings.append(f)

    # Interpretation findings
    for i in range(10):
        f = MagicMock()
        f.finding_id = f"int_{i:03d}"
        f.evidence_type = "interpretation"
        f.summary = f"Interpretation finding {i}"
        findings.append(f)

    return findings


@pytest.fixture
def tracker_with_findings(sample_findings):
    """Create a tracker with sample findings added."""
    tracker = AccuracyTracker()
    for finding in sample_findings:
        tracker.add_finding(finding)
    return tracker


# ============================================================================
# TestAccuracyTarget
# ============================================================================

class TestAccuracyTarget:
    """Tests for AccuracyTarget dataclass."""

    def test_create_accuracy_target(self):
        """Test creating an accuracy target."""
        target = AccuracyTarget(
            statement_type="data_analysis",
            paper_accuracy=0.855,
            target_threshold=0.80,
            description="Data analysis statements"
        )

        assert target.statement_type == "data_analysis"
        assert target.paper_accuracy == 0.855
        assert target.target_threshold == 0.80
        assert target.description == "Data analysis statements"

    def test_accuracy_target_to_dict(self):
        """Test converting accuracy target to dictionary."""
        target = AccuracyTarget(
            statement_type="literature",
            paper_accuracy=0.821,
            target_threshold=0.75,
            description="Literature statements"
        )

        result = target.to_dict()

        assert isinstance(result, dict)
        assert result["statement_type"] == "literature"
        assert result["paper_accuracy"] == 0.821
        assert result["target_threshold"] == 0.75

    def test_paper_targets_defined(self):
        """Test that paper targets are properly defined in tracker."""
        assert "data_analysis" in AccuracyTracker.PAPER_TARGETS
        assert "literature" in AccuracyTracker.PAPER_TARGETS
        assert "interpretation" in AccuracyTracker.PAPER_TARGETS
        assert "overall" in AccuracyTracker.PAPER_TARGETS

    def test_paper_targets_match_paper_claims(self):
        """Test that paper targets match paper claims from Section 8."""
        targets = AccuracyTracker.PAPER_TARGETS

        # Paper claims: 79.4% overall, 85.5% data, 82.1% lit, 57.9% interp
        assert targets["overall"].paper_accuracy == 0.794
        assert targets["data_analysis"].paper_accuracy == 0.855
        assert targets["literature"].paper_accuracy == 0.821
        assert targets["interpretation"].paper_accuracy == 0.579

    def test_implementation_targets_set(self):
        """Test that implementation targets are set appropriately."""
        targets = AccuracyTracker.PAPER_TARGETS

        # Implementation targets should be below paper claims
        assert targets["overall"].target_threshold == 0.75
        assert targets["data_analysis"].target_threshold == 0.80
        assert targets["literature"].target_threshold == 0.75
        assert targets["interpretation"].target_threshold == 0.50


# ============================================================================
# TestTypeAccuracyResult
# ============================================================================

class TestTypeAccuracyResult:
    """Tests for TypeAccuracyResult dataclass."""

    def test_create_type_accuracy_result(self):
        """Test creating a type accuracy result."""
        result = TypeAccuracyResult(
            statement_type="data_analysis",
            total_count=100,
            accurate_count=85,
            accuracy=0.85,
            target=0.80,
            meets_target=True,
            paper_claim=0.855,
            delta_from_paper=-0.005
        )

        assert result.statement_type == "data_analysis"
        assert result.total_count == 100
        assert result.accurate_count == 85
        assert result.accuracy == 0.85
        assert result.meets_target is True

    def test_accuracy_percentage_property(self):
        """Test accuracy percentage calculation."""
        result = TypeAccuracyResult(
            statement_type="test",
            total_count=100,
            accurate_count=85,
            accuracy=0.85,
            target=0.80,
            meets_target=True,
            paper_claim=0.855,
            delta_from_paper=-0.005
        )

        assert result.accuracy_percentage == 85.0

    def test_target_percentage_property(self):
        """Test target percentage calculation."""
        result = TypeAccuracyResult(
            statement_type="test",
            total_count=100,
            accurate_count=85,
            accuracy=0.85,
            target=0.80,
            meets_target=True,
            paper_claim=0.855,
            delta_from_paper=-0.005
        )

        assert result.target_percentage == 80.0

    def test_delta_percentage_property(self):
        """Test delta percentage calculation."""
        result = TypeAccuracyResult(
            statement_type="test",
            total_count=100,
            accurate_count=85,
            accuracy=0.85,
            target=0.80,
            meets_target=True,
            paper_claim=0.855,
            delta_from_paper=-0.005
        )

        assert result.delta_percentage == -0.5

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = TypeAccuracyResult(
            statement_type="literature",
            total_count=50,
            accurate_count=40,
            accuracy=0.80,
            target=0.75,
            meets_target=True,
            paper_claim=0.821,
            delta_from_paper=-0.021
        )

        d = result.to_dict()

        assert isinstance(d, dict)
        assert d["statement_type"] == "literature"
        assert d["accuracy"] == 0.80

    def test_meets_target_false_when_below(self):
        """Test meets_target is False when accuracy below target."""
        result = TypeAccuracyResult(
            statement_type="test",
            total_count=100,
            accurate_count=70,
            accuracy=0.70,
            target=0.80,
            meets_target=False,
            paper_claim=0.855,
            delta_from_paper=-0.155
        )

        assert result.meets_target is False


# ============================================================================
# TestAccuracyTracker
# ============================================================================

class TestAccuracyTracker:
    """Tests for AccuracyTracker class."""

    def test_create_tracker(self):
        """Test creating an accuracy tracker."""
        tracker = AccuracyTracker()

        assert tracker.findings_by_type is not None
        assert tracker.validations is not None
        assert len(tracker._finding_ids) == 0

    def test_add_finding(self, mock_finding):
        """Test adding a finding."""
        tracker = AccuracyTracker()
        tracker.add_finding(mock_finding)

        assert mock_finding.finding_id in tracker._finding_ids
        assert len(tracker.findings_by_type["data_analysis"]) == 1

    def test_add_finding_without_id_raises(self):
        """Test that adding finding without ID raises error."""
        tracker = AccuracyTracker()
        finding = MagicMock()
        finding.finding_id = None

        with pytest.raises(ValueError, match="finding_id"):
            tracker.add_finding(finding)

    def test_add_duplicate_finding_skipped(self, mock_finding):
        """Test that duplicate findings are skipped."""
        tracker = AccuracyTracker()
        tracker.add_finding(mock_finding)
        tracker.add_finding(mock_finding)  # Add again

        assert len(tracker.findings_by_type["data_analysis"]) == 1

    def test_add_finding_with_unknown_type(self):
        """Test that unknown evidence types default to data_analysis."""
        tracker = AccuracyTracker()
        finding = MagicMock()
        finding.finding_id = "test_001"
        finding.evidence_type = "unknown_type"

        tracker.add_finding(finding)

        assert len(tracker.findings_by_type["data_analysis"]) == 1

    def test_validate_finding(self, mock_finding):
        """Test validating a finding."""
        tracker = AccuracyTracker()
        tracker.add_finding(mock_finding)
        tracker.validate_finding("test_001", is_accurate=True)

        assert "test_001" in tracker.validations
        assert tracker.validations["test_001"].is_accurate is True

    def test_validate_finding_not_found_raises(self):
        """Test that validating unknown finding raises error."""
        tracker = AccuracyTracker()

        with pytest.raises(ValueError, match="not found"):
            tracker.validate_finding("unknown", is_accurate=True)

    def test_validate_finding_with_notes(self, mock_finding):
        """Test validating a finding with expert notes."""
        tracker = AccuracyTracker()
        tracker.add_finding(mock_finding)
        tracker.validate_finding("test_001", is_accurate=True, expert_notes="Well supported")

        assert tracker.validations["test_001"].expert_notes == "Well supported"

    def test_compute_type_accuracy_empty(self):
        """Test computing accuracy with no findings."""
        tracker = AccuracyTracker()
        result = tracker.compute_type_accuracy("data_analysis")

        assert result.total_count == 0
        assert result.accuracy == 0.0
        assert result.meets_target is False

    def test_compute_type_accuracy_with_findings(self, tracker_with_findings):
        """Test computing accuracy with validated findings."""
        # Validate 9/10 data analysis as accurate
        for i in range(10):
            tracker_with_findings.validate_finding(
                f"da_{i:03d}",
                is_accurate=(i < 9)  # 90% accurate
            )

        result = tracker_with_findings.compute_type_accuracy("data_analysis")

        assert result.total_count == 10
        assert result.accurate_count == 9
        assert result.accuracy == 0.9
        assert result.meets_target is True  # Target is 80%

    def test_compute_type_accuracy_unknown_type_raises(self):
        """Test that unknown type raises error."""
        tracker = AccuracyTracker()

        with pytest.raises(ValueError, match="Unknown evidence type"):
            tracker.compute_type_accuracy("invalid_type")

    def test_compute_accuracy_by_type(self, tracker_with_findings):
        """Test computing accuracy for all types."""
        # Validate all findings
        for i in range(10):
            tracker_with_findings.validate_finding(f"da_{i:03d}", is_accurate=True)
            tracker_with_findings.validate_finding(f"lit_{i:03d}", is_accurate=(i < 8))
            tracker_with_findings.validate_finding(f"int_{i:03d}", is_accurate=(i < 6))

        results = tracker_with_findings.compute_accuracy_by_type()

        assert "data_analysis" in results
        assert "literature" in results
        assert "interpretation" in results

        assert results["data_analysis"].accuracy == 1.0
        assert results["literature"].accuracy == 0.8
        assert results["interpretation"].accuracy == 0.6

    def test_compare_to_paper(self, tracker_with_findings):
        """Test comparing results to paper claims."""
        # Validate findings to match paper claims roughly
        for i in range(10):
            tracker_with_findings.validate_finding(f"da_{i:03d}", is_accurate=(i < 8))
            tracker_with_findings.validate_finding(f"lit_{i:03d}", is_accurate=(i < 8))
            tracker_with_findings.validate_finding(f"int_{i:03d}", is_accurate=(i < 6))

        deltas = tracker_with_findings.compare_to_paper()

        assert "data_analysis" in deltas
        assert "literature" in deltas
        assert "interpretation" in deltas
        assert "overall" in deltas

    def test_get_statistics(self, tracker_with_findings):
        """Test getting summary statistics."""
        # Validate some findings
        for i in range(5):
            tracker_with_findings.validate_finding(f"da_{i:03d}", is_accurate=True)

        stats = tracker_with_findings.get_statistics()

        assert stats["total_findings"] == 30
        assert stats["validated_findings"] == 5
        assert stats["accurate_findings"] == 5
        assert stats["validation_rate"] == 5 / 30

    def test_generate_report(self, tracker_with_findings):
        """Test generating a complete report."""
        # Validate all findings
        for i in range(10):
            tracker_with_findings.validate_finding(f"da_{i:03d}", is_accurate=(i < 9))
            tracker_with_findings.validate_finding(f"lit_{i:03d}", is_accurate=(i < 8))
            tracker_with_findings.validate_finding(f"int_{i:03d}", is_accurate=(i < 6))

        report = tracker_with_findings.generate_report()

        assert isinstance(report, AccuracyReport)
        assert report.total_findings == 30
        assert report.validated_findings == 30
        assert len(report.by_type) == 3
        assert len(report.recommendations) > 0


# ============================================================================
# TestAccuracyReport
# ============================================================================

class TestAccuracyReport:
    """Tests for AccuracyReport dataclass."""

    def test_create_report(self):
        """Test creating an accuracy report."""
        by_type = {
            "data_analysis": TypeAccuracyResult(
                statement_type="data_analysis",
                total_count=10,
                accurate_count=9,
                accuracy=0.9,
                target=0.80,
                meets_target=True,
                paper_claim=0.855,
                delta_from_paper=0.045
            )
        }

        report = AccuracyReport(
            overall_accuracy=0.8,
            overall_target=0.75,
            overall_meets_target=True,
            by_type=by_type,
            total_findings=30,
            validated_findings=30,
            paper_comparison={"overall": 0.006},
            timestamp=datetime.now().isoformat()
        )

        assert report.overall_accuracy == 0.8
        assert report.overall_meets_target is True

    def test_overall_accuracy_percentage(self):
        """Test overall accuracy percentage property."""
        report = AccuracyReport(
            overall_accuracy=0.85,
            overall_target=0.75,
            overall_meets_target=True,
            by_type={},
            total_findings=30,
            validated_findings=30,
            paper_comparison={},
            timestamp=datetime.now().isoformat()
        )

        assert report.overall_accuracy_percentage == 85.0

    def test_all_types_meet_targets_true(self):
        """Test all_types_meet_targets when all pass."""
        by_type = {
            "data_analysis": TypeAccuracyResult(
                "data_analysis", 10, 9, 0.9, 0.8, True, 0.855, 0.045
            ),
            "literature": TypeAccuracyResult(
                "literature", 10, 8, 0.8, 0.75, True, 0.821, -0.021
            ),
        }

        report = AccuracyReport(
            overall_accuracy=0.85,
            overall_target=0.75,
            overall_meets_target=True,
            by_type=by_type,
            total_findings=20,
            validated_findings=20,
            paper_comparison={},
            timestamp=datetime.now().isoformat()
        )

        assert report.all_types_meet_targets is True

    def test_all_types_meet_targets_false(self):
        """Test all_types_meet_targets when one fails."""
        by_type = {
            "data_analysis": TypeAccuracyResult(
                "data_analysis", 10, 9, 0.9, 0.8, True, 0.855, 0.045
            ),
            "literature": TypeAccuracyResult(
                "literature", 10, 6, 0.6, 0.75, False, 0.821, -0.221
            ),
        }

        report = AccuracyReport(
            overall_accuracy=0.75,
            overall_target=0.75,
            overall_meets_target=True,
            by_type=by_type,
            total_findings=20,
            validated_findings=20,
            paper_comparison={},
            timestamp=datetime.now().isoformat()
        )

        assert report.all_types_meet_targets is False

    def test_to_dict(self):
        """Test converting report to dictionary."""
        by_type = {
            "data_analysis": TypeAccuracyResult(
                "data_analysis", 10, 9, 0.9, 0.8, True, 0.855, 0.045
            ),
        }

        report = AccuracyReport(
            overall_accuracy=0.9,
            overall_target=0.75,
            overall_meets_target=True,
            by_type=by_type,
            total_findings=10,
            validated_findings=10,
            paper_comparison={"overall": 0.106},
            timestamp="2025-12-09T00:00:00"
        )

        d = report.to_dict()

        assert isinstance(d, dict)
        assert d["overall_accuracy"] == 0.9
        assert "data_analysis" in d["by_type"]


# ============================================================================
# TestAccuracyReporter
# ============================================================================

class TestAccuracyReporter:
    """Tests for AccuracyReporter class."""

    @pytest.fixture
    def sample_report(self):
        """Create a sample report for testing."""
        by_type = {
            "data_analysis": TypeAccuracyResult(
                "data_analysis", 40, 35, 0.875, 0.80, True, 0.855, 0.02
            ),
            "literature": TypeAccuracyResult(
                "literature", 35, 28, 0.80, 0.75, True, 0.821, -0.021
            ),
            "interpretation": TypeAccuracyResult(
                "interpretation", 25, 15, 0.60, 0.50, True, 0.579, 0.021
            ),
        }

        return AccuracyReport(
            overall_accuracy=0.78,
            overall_target=0.75,
            overall_meets_target=True,
            by_type=by_type,
            total_findings=100,
            validated_findings=100,
            paper_comparison={
                "overall": -0.014,
                "data_analysis": 0.02,
                "literature": -0.021,
                "interpretation": 0.021
            },
            timestamp="2025-12-09T00:00:00",
            recommendations=["All accuracy targets met."]
        )

    def test_create_reporter(self):
        """Test creating a reporter."""
        reporter = AccuracyReporter()
        assert reporter is not None

    def test_generate_markdown_report(self, sample_report):
        """Test generating markdown report."""
        reporter = AccuracyReporter()
        md = reporter.generate_markdown_report(sample_report)

        assert isinstance(md, str)
        assert "# Paper Accuracy Validation Report" in md
        assert "Overall Accuracy" in md
        assert "Data Analysis" in md
        assert "Literature" in md
        assert "Interpretation" in md

    def test_generate_markdown_report_contains_tables(self, sample_report):
        """Test that markdown report contains tables."""
        reporter = AccuracyReporter()
        md = reporter.generate_markdown_report(sample_report)

        # Should contain markdown table separators
        assert "|-----" in md or "|---" in md

    def test_generate_markdown_report_contains_status(self, sample_report):
        """Test that markdown report contains pass/fail status."""
        reporter = AccuracyReporter()
        md = reporter.generate_markdown_report(sample_report)

        assert "PASS" in md

    def test_generate_summary(self, sample_report):
        """Test generating executive summary."""
        reporter = AccuracyReporter()
        summary = reporter.generate_summary(sample_report)

        assert isinstance(summary, str)
        assert "Overall Accuracy" in summary
        assert "78.0%" in summary

    def test_generate_summary_failing_report(self):
        """Test generating summary for failing report."""
        by_type = {
            "data_analysis": TypeAccuracyResult(
                "data_analysis", 10, 7, 0.70, 0.80, False, 0.855, -0.155
            ),
        }

        report = AccuracyReport(
            overall_accuracy=0.70,
            overall_target=0.75,
            overall_meets_target=False,
            by_type=by_type,
            total_findings=10,
            validated_findings=10,
            paper_comparison={"overall": -0.094},
            timestamp="2025-12-09T00:00:00"
        )

        reporter = AccuracyReporter()
        summary = reporter.generate_summary(report)

        assert "FAIL" in summary

    def test_generate_json_report(self, sample_report):
        """Test generating JSON report."""
        reporter = AccuracyReporter()
        json_str = reporter.generate_json_report(sample_report)

        import json
        data = json.loads(json_str)

        assert data["overall_accuracy"] == 0.78
        assert "by_type" in data

    def test_generate_markdown_contains_recommendations(self, sample_report):
        """Test that markdown contains recommendations."""
        reporter = AccuracyReporter()
        md = reporter.generate_markdown_report(sample_report)

        assert "Recommendations" in md


# ============================================================================
# TestPaperComparison
# ============================================================================

class TestPaperComparison:
    """Tests for paper comparison functionality."""

    def test_compare_to_paper_all_types(self, tracker_with_findings):
        """Test that all types are compared to paper."""
        # Validate all findings
        for i in range(10):
            tracker_with_findings.validate_finding(f"da_{i:03d}", is_accurate=True)
            tracker_with_findings.validate_finding(f"lit_{i:03d}", is_accurate=True)
            tracker_with_findings.validate_finding(f"int_{i:03d}", is_accurate=True)

        deltas = tracker_with_findings.compare_to_paper()

        assert len(deltas) == 4  # 3 types + overall

    def test_compare_exceeds_paper(self, tracker_with_findings):
        """Test delta is positive when exceeding paper."""
        # Validate all data analysis as accurate (100% vs paper 85.5%)
        for i in range(10):
            tracker_with_findings.validate_finding(f"da_{i:03d}", is_accurate=True)

        result = tracker_with_findings.compute_type_accuracy("data_analysis")

        assert result.delta_from_paper > 0  # 1.0 - 0.855 = 0.145

    def test_compare_below_paper(self, tracker_with_findings):
        """Test delta is negative when below paper."""
        # Validate half data analysis as accurate (50% vs paper 85.5%)
        for i in range(10):
            tracker_with_findings.validate_finding(f"da_{i:03d}", is_accurate=(i < 5))

        result = tracker_with_findings.compute_type_accuracy("data_analysis")

        assert result.delta_from_paper < 0  # 0.5 - 0.855 = -0.355

    def test_recommendations_generated_when_below_target(self, tracker_with_findings):
        """Test recommendations are generated when below target."""
        # Validate to be below target
        for i in range(10):
            tracker_with_findings.validate_finding(f"da_{i:03d}", is_accurate=(i < 6))  # 60%
        for i in range(10):
            tracker_with_findings.validate_finding(f"lit_{i:03d}", is_accurate=True)
        for i in range(10):
            tracker_with_findings.validate_finding(f"int_{i:03d}", is_accurate=True)

        report = tracker_with_findings.generate_report()

        # Should have recommendation about data_analysis being below target
        recommendations_text = " ".join(report.recommendations)
        assert "Data Analysis" in recommendations_text or "data analysis" in recommendations_text

    def test_recommendations_when_all_pass(self, tracker_with_findings):
        """Test recommendations when all targets pass."""
        # Validate all to pass
        for i in range(10):
            tracker_with_findings.validate_finding(f"da_{i:03d}", is_accurate=(i < 9))  # 90%
            tracker_with_findings.validate_finding(f"lit_{i:03d}", is_accurate=(i < 8))  # 80%
            tracker_with_findings.validate_finding(f"int_{i:03d}", is_accurate=(i < 6))  # 60%

        report = tracker_with_findings.generate_report()

        # Should still have recommendations
        assert len(report.recommendations) > 0

    def test_validation_record_timestamp(self, mock_finding):
        """Test that validation records include timestamp."""
        tracker = AccuracyTracker()
        tracker.add_finding(mock_finding)
        tracker.validate_finding("test_001", is_accurate=True)

        record = tracker.validations["test_001"]

        assert record.timestamp is not None
        assert isinstance(record.timestamp, str)
