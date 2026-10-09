"""
Integration tests for Paper Accuracy Validation Pipeline (Issue #65).

Tests the complete accuracy validation workflow including:
- Loading benchmark datasets
- Tracking accuracy across finding types
- Generating validation reports
- Comparing to paper claims
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from kosmos.validation.accuracy_tracker import (
    AccuracyTracker,
    AccuracyReporter,
    AccuracyReport,
)
from kosmos.validation.benchmark_dataset import (
    BenchmarkDataset,
    BenchmarkFinding,
    BenchmarkGenerator,
    create_paper_benchmark,
)
from kosmos.world_model.artifacts import Finding


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def paper_benchmark():
    """Create the paper accuracy benchmark dataset."""
    return create_paper_benchmark()


@pytest.fixture
def real_findings():
    """Create real Finding objects for testing."""
    findings = []

    # Data analysis findings
    for i in range(10):
        finding = Finding(
            finding_id=f"da_{i:03d}",
            cycle=1,
            task_id=i,
            summary=f"Data analysis finding {i} with p-value 0.0{i+1}",
            statistics={"p_value": 0.01 * (i + 1), "effect_size": 0.5 + i * 0.05},
            evidence_type="data_analysis"
        )
        findings.append(finding)

    # Literature findings
    for i in range(10):
        finding = Finding(
            finding_id=f"lit_{i:03d}",
            cycle=1,
            task_id=10 + i,
            summary=f"Literature review finding {i} based on prior studies",
            statistics={},
            evidence_type="literature",
            citations=[{"author": f"Author{i}", "year": "2020"}]
        )
        findings.append(finding)

    # Interpretation findings
    for i in range(10):
        finding = Finding(
            finding_id=f"int_{i:03d}",
            cycle=1,
            task_id=20 + i,
            summary=f"Interpretation {i}: this suggests a potential mechanism",
            statistics={},
            evidence_type="interpretation"
        )
        findings.append(finding)

    return findings


# ============================================================================
# TestFullAccuracyPipeline
# ============================================================================

class TestFullAccuracyPipeline:
    """End-to-end tests for accuracy validation pipeline."""

    def test_full_pipeline_with_real_findings(self, real_findings):
        """Test complete pipeline with real Finding objects."""
        # Create tracker and add findings
        tracker = AccuracyTracker()
        for finding in real_findings:
            tracker.add_finding(finding)

        # Validate findings (simulate expert review)
        for i in range(10):
            # 90% data analysis accurate
            tracker.validate_finding(f"da_{i:03d}", is_accurate=(i < 9))
            # 80% literature accurate
            tracker.validate_finding(f"lit_{i:03d}", is_accurate=(i < 8))
            # 60% interpretation accurate
            tracker.validate_finding(f"int_{i:03d}", is_accurate=(i < 6))

        # Generate report
        report = tracker.generate_report()

        # Verify results
        assert report.total_findings == 30
        assert report.validated_findings == 30
        assert report.by_type["data_analysis"].accuracy == 0.9
        assert report.by_type["literature"].accuracy == 0.8
        assert report.by_type["interpretation"].accuracy == 0.6

    def test_pipeline_matches_paper_targets(self, real_findings):
        """Test that pipeline can validate against paper targets."""
        tracker = AccuracyTracker()
        for finding in real_findings:
            tracker.add_finding(finding)

        # Validate to match paper rates approximately
        for i in range(10):
            tracker.validate_finding(f"da_{i:03d}", is_accurate=(i < 9))  # 90% (target 80%)
            tracker.validate_finding(f"lit_{i:03d}", is_accurate=(i < 8))  # 80% (target 75%)
            tracker.validate_finding(f"int_{i:03d}", is_accurate=(i < 6))  # 60% (target 50%)

        report = tracker.generate_report()

        # All should meet targets
        assert report.by_type["data_analysis"].meets_target is True
        assert report.by_type["literature"].meets_target is True
        assert report.by_type["interpretation"].meets_target is True
        assert report.overall_meets_target is True

    def test_pipeline_with_failing_accuracy(self, real_findings):
        """Test pipeline with accuracy below target."""
        tracker = AccuracyTracker()
        for finding in real_findings:
            tracker.add_finding(finding)

        # Validate with low accuracy
        for i in range(10):
            tracker.validate_finding(f"da_{i:03d}", is_accurate=(i < 5))  # 50% (target 80%)
            tracker.validate_finding(f"lit_{i:03d}", is_accurate=(i < 5))  # 50% (target 75%)
            tracker.validate_finding(f"int_{i:03d}", is_accurate=(i < 3))  # 30% (target 50%)

        report = tracker.generate_report()

        # Should fail targets
        assert report.by_type["data_analysis"].meets_target is False
        assert report.by_type["literature"].meets_target is False
        assert report.by_type["interpretation"].meets_target is False
        assert report.overall_meets_target is False

    def test_pipeline_generates_recommendations(self, real_findings):
        """Test that pipeline generates actionable recommendations."""
        tracker = AccuracyTracker()
        for finding in real_findings:
            tracker.add_finding(finding)

        # Validate with mixed results
        for i in range(10):
            tracker.validate_finding(f"da_{i:03d}", is_accurate=(i < 6))  # 60% - below target
            tracker.validate_finding(f"lit_{i:03d}", is_accurate=(i < 9))  # 90% - above target
            tracker.validate_finding(f"int_{i:03d}", is_accurate=(i < 6))  # 60% - above target

        report = tracker.generate_report()

        # Should have recommendations for data_analysis
        assert len(report.recommendations) > 0
        rec_text = " ".join(report.recommendations).lower()
        assert "data" in rec_text or "target" in rec_text

    def test_pipeline_incremental_validation(self, real_findings):
        """Test validating findings incrementally."""
        tracker = AccuracyTracker()

        # Add all findings
        for finding in real_findings:
            tracker.add_finding(finding)

        # Validate in batches
        for i in range(5):
            tracker.validate_finding(f"da_{i:03d}", is_accurate=True)

        # Check intermediate state
        stats = tracker.get_statistics()
        assert stats["validated_findings"] == 5

        # Continue validation
        for i in range(5, 10):
            tracker.validate_finding(f"da_{i:03d}", is_accurate=True)

        stats = tracker.get_statistics()
        assert stats["validated_findings"] == 10

    def test_pipeline_with_expert_notes(self, real_findings):
        """Test that expert notes are preserved."""
        tracker = AccuracyTracker()
        for finding in real_findings[:5]:
            tracker.add_finding(finding)

        tracker.validate_finding(
            "da_000",
            is_accurate=True,
            expert_notes="Well-supported statistical analysis"
        )
        tracker.validate_finding(
            "da_001",
            is_accurate=False,
            expert_notes="P-value hacking suspected"
        )

        assert tracker.validations["da_000"].expert_notes == "Well-supported statistical analysis"
        assert tracker.validations["da_001"].expert_notes == "P-value hacking suspected"

    def test_pipeline_report_serialization(self, real_findings):
        """Test that reports can be serialized to JSON."""
        tracker = AccuracyTracker()
        for finding in real_findings:
            tracker.add_finding(finding)

        for i in range(10):
            tracker.validate_finding(f"da_{i:03d}", is_accurate=True)
            tracker.validate_finding(f"lit_{i:03d}", is_accurate=True)
            tracker.validate_finding(f"int_{i:03d}", is_accurate=True)

        report = tracker.generate_report()
        reporter = AccuracyReporter()

        json_str = reporter.generate_json_report(report)
        data = json.loads(json_str)

        assert data["overall_accuracy"] == 1.0
        assert "by_type" in data

    def test_pipeline_empty_validation(self):
        """Test pipeline with no validations."""
        tracker = AccuracyTracker()
        report = tracker.generate_report()

        assert report.total_findings == 0
        assert report.validated_findings == 0
        assert report.overall_accuracy == 0.0


# ============================================================================
# TestBenchmarkValidation
# ============================================================================

class TestBenchmarkValidation:
    """Tests for validating against benchmark datasets."""

    def test_validate_against_paper_benchmark(self, paper_benchmark):
        """Test validating findings against paper benchmark."""
        tracker = AccuracyTracker()

        # Convert benchmark findings to tracker findings
        for bf in paper_benchmark.findings:
            mock_finding = MagicMock()
            mock_finding.finding_id = bf.finding_id
            mock_finding.evidence_type = bf.evidence_type
            mock_finding.summary = bf.summary
            tracker.add_finding(mock_finding)

        # Validate using ground truth
        for bf in paper_benchmark.findings:
            tracker.validate_finding(bf.finding_id, is_accurate=bf.ground_truth_accurate)

        report = tracker.generate_report()

        # Should match benchmark accuracy rates
        benchmark_accuracy = paper_benchmark.get_accuracy_by_type()

        assert abs(
            report.by_type["data_analysis"].accuracy -
            benchmark_accuracy["data_analysis"]
        ) < 0.01
        assert abs(
            report.by_type["literature"].accuracy -
            benchmark_accuracy["literature"]
        ) < 0.01
        assert abs(
            report.by_type["interpretation"].accuracy -
            benchmark_accuracy["interpretation"]
        ) < 0.01

    def test_benchmark_save_and_reload(self, paper_benchmark):
        """Test saving and reloading benchmark dataset."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            paper_benchmark.save(path)
            reloaded = BenchmarkDataset.load(path)

            assert reloaded.name == paper_benchmark.name
            assert len(reloaded.findings) == len(paper_benchmark.findings)
            assert reloaded.get_overall_accuracy() == paper_benchmark.get_overall_accuracy()
        finally:
            Path(path).unlink()

    def test_benchmark_type_distribution(self, paper_benchmark):
        """Test that benchmark has correct type distribution."""
        dist = paper_benchmark.get_type_distribution()

        # 30 per type
        assert dist["data_analysis"] == 30
        assert dist["literature"] == 30
        assert dist["interpretation"] == 30

    def test_create_custom_benchmark_from_findings(self, real_findings):
        """Test creating custom benchmark from real findings."""
        ground_truth = {}
        for i, finding in enumerate(real_findings):
            ground_truth[finding.finding_id] = (i % 3 != 0)  # 2/3 accurate

        generator = BenchmarkGenerator()
        benchmark = generator.create_from_findings(
            real_findings,
            ground_truth,
            name="custom_test"
        )

        assert benchmark.name == "custom_test"
        assert len(benchmark.findings) == 30

    def test_benchmark_reproducibility(self):
        """Test that benchmark generation is reproducible."""
        gen1 = BenchmarkGenerator(seed=12345)
        gen2 = BenchmarkGenerator(seed=12345)

        ds1 = gen1.create_synthetic_benchmark(n_per_type=10)
        ds2 = gen2.create_synthetic_benchmark(n_per_type=10)

        assert ds1.get_overall_accuracy() == ds2.get_overall_accuracy()

    def test_benchmark_with_edge_case_accuracy(self):
        """Test benchmark with edge case accuracy rates."""
        gen = BenchmarkGenerator(seed=42)

        # Test with extreme rates
        rates = {
            "data_analysis": 1.0,  # 100%
            "literature": 0.5,     # 50%
            "interpretation": 0.0  # 0%
        }

        ds = gen.create_synthetic_benchmark(n_per_type=10, accuracy_rates=rates)
        accuracy = ds.get_accuracy_by_type()

        assert accuracy["data_analysis"] == 1.0
        assert accuracy["literature"] == 0.5
        assert accuracy["interpretation"] == 0.0


# ============================================================================
# TestReportGeneration
# ============================================================================

class TestReportGeneration:
    """Tests for report generation."""

    def test_markdown_report_structure(self, real_findings):
        """Test markdown report has correct structure."""
        tracker = AccuracyTracker()
        for finding in real_findings:
            tracker.add_finding(finding)

        for i in range(10):
            tracker.validate_finding(f"da_{i:03d}", is_accurate=True)
            tracker.validate_finding(f"lit_{i:03d}", is_accurate=True)
            tracker.validate_finding(f"int_{i:03d}", is_accurate=True)

        report = tracker.generate_report()
        reporter = AccuracyReporter()
        md = reporter.generate_markdown_report(report)

        # Check structure
        assert "# Paper Accuracy Validation Report" in md
        assert "## Summary" in md
        assert "## Accuracy by Statement Type" in md
        assert "## Comparison to Paper Claims" in md
        assert "## Recommendations" in md

    def test_markdown_report_contains_tables(self, real_findings):
        """Test markdown report contains properly formatted tables."""
        tracker = AccuracyTracker()
        for finding in real_findings:
            tracker.add_finding(finding)

        for i in range(10):
            tracker.validate_finding(f"da_{i:03d}", is_accurate=True)
            tracker.validate_finding(f"lit_{i:03d}", is_accurate=True)
            tracker.validate_finding(f"int_{i:03d}", is_accurate=True)

        report = tracker.generate_report()
        reporter = AccuracyReporter()
        md = reporter.generate_markdown_report(report)

        # Check for table markers
        assert "|" in md
        assert "---" in md or "-----" in md

    def test_summary_generation(self, real_findings):
        """Test executive summary generation."""
        tracker = AccuracyTracker()
        for finding in real_findings:
            tracker.add_finding(finding)

        for i in range(10):
            tracker.validate_finding(f"da_{i:03d}", is_accurate=True)
            tracker.validate_finding(f"lit_{i:03d}", is_accurate=True)
            tracker.validate_finding(f"int_{i:03d}", is_accurate=True)

        report = tracker.generate_report()
        reporter = AccuracyReporter()
        summary = reporter.generate_summary(report)

        assert "Overall Accuracy" in summary
        assert "PASS" in summary

    def test_json_report_validity(self, real_findings):
        """Test JSON report is valid JSON."""
        tracker = AccuracyTracker()
        for finding in real_findings:
            tracker.add_finding(finding)

        for i in range(10):
            tracker.validate_finding(f"da_{i:03d}", is_accurate=True)
            tracker.validate_finding(f"lit_{i:03d}", is_accurate=True)
            tracker.validate_finding(f"int_{i:03d}", is_accurate=True)

        report = tracker.generate_report()
        reporter = AccuracyReporter()
        json_str = reporter.generate_json_report(report)

        # Should be valid JSON
        data = json.loads(json_str)
        assert isinstance(data, dict)

    def test_report_with_paper_comparison(self, real_findings):
        """Test that report includes paper comparison."""
        tracker = AccuracyTracker()
        for finding in real_findings:
            tracker.add_finding(finding)

        for i in range(10):
            tracker.validate_finding(f"da_{i:03d}", is_accurate=True)
            tracker.validate_finding(f"lit_{i:03d}", is_accurate=True)
            tracker.validate_finding(f"int_{i:03d}", is_accurate=True)

        report = tracker.generate_report()

        assert "overall" in report.paper_comparison
        assert "data_analysis" in report.paper_comparison
        assert "literature" in report.paper_comparison
        assert "interpretation" in report.paper_comparison

    def test_report_timestamp(self, real_findings):
        """Test that report includes timestamp."""
        tracker = AccuracyTracker()
        for finding in real_findings[:5]:
            tracker.add_finding(finding)
            tracker.validate_finding(finding.finding_id, is_accurate=True)

        report = tracker.generate_report()

        assert report.timestamp is not None
        assert len(report.timestamp) > 0


# ============================================================================
# TestStatisticalValidation
# ============================================================================

class TestStatisticalValidation:
    """Tests for statistical aspects of validation."""

    def test_accuracy_calculation_edge_cases(self):
        """Test accuracy calculation with edge cases."""
        tracker = AccuracyTracker()

        # Add single finding
        finding = MagicMock()
        finding.finding_id = "single"
        finding.evidence_type = "data_analysis"
        tracker.add_finding(finding)

        # Validate as accurate
        tracker.validate_finding("single", is_accurate=True)

        result = tracker.compute_type_accuracy("data_analysis")
        assert result.accuracy == 1.0

    def test_zero_accuracy(self):
        """Test handling of zero accuracy."""
        tracker = AccuracyTracker()

        for i in range(5):
            finding = MagicMock()
            finding.finding_id = f"f_{i}"
            finding.evidence_type = "data_analysis"
            tracker.add_finding(finding)
            tracker.validate_finding(f"f_{i}", is_accurate=False)

        result = tracker.compute_type_accuracy("data_analysis")
        assert result.accuracy == 0.0
        assert result.meets_target is False

    def test_perfect_accuracy(self):
        """Test handling of perfect accuracy."""
        tracker = AccuracyTracker()

        for i in range(5):
            finding = MagicMock()
            finding.finding_id = f"f_{i}"
            finding.evidence_type = "data_analysis"
            tracker.add_finding(finding)
            tracker.validate_finding(f"f_{i}", is_accurate=True)

        result = tracker.compute_type_accuracy("data_analysis")
        assert result.accuracy == 1.0
        assert result.meets_target is True

    def test_delta_from_paper_calculation(self, real_findings):
        """Test delta from paper calculation."""
        tracker = AccuracyTracker()
        for finding in real_findings:
            tracker.add_finding(finding)

        # Validate to exactly match paper claim for data_analysis (85.5%)
        # With 10 findings, 8-9 accurate gives ~85%
        for i in range(10):
            tracker.validate_finding(f"da_{i:03d}", is_accurate=(i < 9))  # 90%
            tracker.validate_finding(f"lit_{i:03d}", is_accurate=True)
            tracker.validate_finding(f"int_{i:03d}", is_accurate=True)

        result = tracker.compute_type_accuracy("data_analysis")

        # 90% - 85.5% = 4.5%
        assert abs(result.delta_from_paper - 0.045) < 0.01
