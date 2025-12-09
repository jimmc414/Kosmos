"""
Unit tests for Benchmark Dataset (Issue #65).

Tests BenchmarkFinding, BenchmarkDataset, and BenchmarkGenerator for
managing ground truth datasets for accuracy validation.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from kosmos.validation.benchmark_dataset import (
    BenchmarkFinding,
    BenchmarkDataset,
    BenchmarkGenerator,
    create_paper_benchmark,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_finding():
    """Create a sample benchmark finding."""
    return BenchmarkFinding(
        finding_id="bench_001",
        summary="Treatment shows significant improvement (p<0.01)",
        evidence_type="data_analysis",
        ground_truth_accurate=True,
        domain="clinical",
        source="synthetic",
        statistics={"p_value": 0.008, "effect_size": 0.65}
    )


@pytest.fixture
def sample_dataset():
    """Create a sample benchmark dataset."""
    findings = [
        BenchmarkFinding("b001", "Data finding 1", "data_analysis", True, "biology", "synthetic"),
        BenchmarkFinding("b002", "Data finding 2", "data_analysis", True, "biology", "synthetic"),
        BenchmarkFinding("b003", "Data finding 3", "data_analysis", False, "biology", "synthetic"),
        BenchmarkFinding("b004", "Lit finding 1", "literature", True, "biology", "synthetic"),
        BenchmarkFinding("b005", "Lit finding 2", "literature", False, "biology", "synthetic"),
        BenchmarkFinding("b006", "Interp finding 1", "interpretation", True, "biology", "synthetic"),
        BenchmarkFinding("b007", "Interp finding 2", "interpretation", False, "biology", "synthetic"),
        BenchmarkFinding("b008", "Interp finding 3", "interpretation", False, "biology", "synthetic"),
    ]

    return BenchmarkDataset(
        name="test_benchmark",
        version="1.0.0",
        findings=findings,
        created_at="2025-12-09T00:00:00",
        description="Test benchmark dataset"
    )


# ============================================================================
# TestBenchmarkFinding
# ============================================================================

class TestBenchmarkFinding:
    """Tests for BenchmarkFinding dataclass."""

    def test_create_finding(self, sample_finding):
        """Test creating a benchmark finding."""
        assert sample_finding.finding_id == "bench_001"
        assert sample_finding.evidence_type == "data_analysis"
        assert sample_finding.ground_truth_accurate is True
        assert sample_finding.domain == "clinical"

    def test_finding_with_statistics(self, sample_finding):
        """Test that statistics are stored correctly."""
        assert sample_finding.statistics is not None
        assert sample_finding.statistics["p_value"] == 0.008

    def test_finding_to_dict(self, sample_finding):
        """Test converting finding to dictionary."""
        d = sample_finding.to_dict()

        assert isinstance(d, dict)
        assert d["finding_id"] == "bench_001"
        assert d["evidence_type"] == "data_analysis"
        assert d["ground_truth_accurate"] is True

    def test_finding_from_dict(self, sample_finding):
        """Test creating finding from dictionary."""
        d = sample_finding.to_dict()
        restored = BenchmarkFinding.from_dict(d)

        assert restored.finding_id == sample_finding.finding_id
        assert restored.evidence_type == sample_finding.evidence_type
        assert restored.ground_truth_accurate == sample_finding.ground_truth_accurate

    def test_invalid_evidence_type_raises(self):
        """Test that invalid evidence type raises error."""
        with pytest.raises(ValueError, match="Invalid evidence_type"):
            BenchmarkFinding(
                finding_id="test",
                summary="Test",
                evidence_type="invalid_type",
                ground_truth_accurate=True,
                domain="test",
                source="test"
            )

    def test_literature_finding(self):
        """Test creating a literature finding."""
        finding = BenchmarkFinding(
            finding_id="lit_001",
            summary="Prior studies show effect",
            evidence_type="literature",
            ground_truth_accurate=True,
            domain="biology",
            source="synthetic",
            citations=[{"author": "Smith", "year": "2020"}]
        )

        assert finding.evidence_type == "literature"
        assert finding.citations is not None

    def test_interpretation_finding(self):
        """Test creating an interpretation finding."""
        finding = BenchmarkFinding(
            finding_id="int_001",
            summary="This suggests a mechanism",
            evidence_type="interpretation",
            ground_truth_accurate=False,
            domain="biology",
            source="synthetic",
            expert_notes="Over-interpretation of data"
        )

        assert finding.evidence_type == "interpretation"
        assert finding.expert_notes is not None


# ============================================================================
# TestBenchmarkDataset
# ============================================================================

class TestBenchmarkDataset:
    """Tests for BenchmarkDataset class."""

    def test_create_dataset(self, sample_dataset):
        """Test creating a benchmark dataset."""
        assert sample_dataset.name == "test_benchmark"
        assert sample_dataset.version == "1.0.0"
        assert len(sample_dataset.findings) == 8

    def test_get_by_type(self, sample_dataset):
        """Test filtering findings by type."""
        data_findings = sample_dataset.get_by_type("data_analysis")
        lit_findings = sample_dataset.get_by_type("literature")
        interp_findings = sample_dataset.get_by_type("interpretation")

        assert len(data_findings) == 3
        assert len(lit_findings) == 2
        assert len(interp_findings) == 3

    def test_get_type_distribution(self, sample_dataset):
        """Test getting type distribution."""
        dist = sample_dataset.get_type_distribution()

        assert dist["data_analysis"] == 3
        assert dist["literature"] == 2
        assert dist["interpretation"] == 3

    def test_get_accuracy_by_type(self, sample_dataset):
        """Test getting accuracy by type."""
        accuracy = sample_dataset.get_accuracy_by_type()

        # data_analysis: 2/3 accurate = 0.667
        # literature: 1/2 accurate = 0.5
        # interpretation: 1/3 accurate = 0.333
        assert abs(accuracy["data_analysis"] - 0.667) < 0.01
        assert accuracy["literature"] == 0.5
        assert abs(accuracy["interpretation"] - 0.333) < 0.01

    def test_get_overall_accuracy(self, sample_dataset):
        """Test getting overall accuracy."""
        accuracy = sample_dataset.get_overall_accuracy()

        # 4/8 accurate = 0.5
        assert accuracy == 0.5

    def test_to_dict(self, sample_dataset):
        """Test converting dataset to dictionary."""
        d = sample_dataset.to_dict()

        assert isinstance(d, dict)
        assert d["name"] == "test_benchmark"
        assert len(d["findings"]) == 8

    def test_to_json(self, sample_dataset):
        """Test converting dataset to JSON."""
        json_str = sample_dataset.to_json()
        data = json.loads(json_str)

        assert data["name"] == "test_benchmark"
        assert len(data["findings"]) == 8

    def test_from_dict(self, sample_dataset):
        """Test creating dataset from dictionary."""
        d = sample_dataset.to_dict()
        restored = BenchmarkDataset.from_dict(d)

        assert restored.name == sample_dataset.name
        assert len(restored.findings) == len(sample_dataset.findings)

    def test_from_json(self, sample_dataset):
        """Test creating dataset from JSON."""
        json_str = sample_dataset.to_json()
        restored = BenchmarkDataset.from_json(json_str)

        assert restored.name == sample_dataset.name
        assert len(restored.findings) == len(sample_dataset.findings)

    def test_save_and_load(self, sample_dataset):
        """Test saving and loading dataset."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            sample_dataset.save(path)
            loaded = BenchmarkDataset.load(path)

            assert loaded.name == sample_dataset.name
            assert len(loaded.findings) == len(sample_dataset.findings)
        finally:
            Path(path).unlink()

    def test_empty_dataset(self):
        """Test handling empty dataset."""
        dataset = BenchmarkDataset(
            name="empty",
            version="1.0.0",
            findings=[],
            created_at="2025-12-09T00:00:00",
            description="Empty dataset"
        )

        assert dataset.get_overall_accuracy() == 0.0
        assert dataset.get_type_distribution() == {
            "data_analysis": 0,
            "literature": 0,
            "interpretation": 0
        }


# ============================================================================
# TestBenchmarkGenerator
# ============================================================================

class TestBenchmarkGenerator:
    """Tests for BenchmarkGenerator class."""

    def test_create_generator(self):
        """Test creating a generator."""
        generator = BenchmarkGenerator(seed=42)
        assert generator is not None

    def test_create_generator_with_seed(self):
        """Test that seed produces reproducible results."""
        gen1 = BenchmarkGenerator(seed=42)
        gen2 = BenchmarkGenerator(seed=42)

        dataset1 = gen1.create_synthetic_benchmark(n_per_type=5)
        dataset2 = gen2.create_synthetic_benchmark(n_per_type=5)

        # Same seed should produce same findings
        assert len(dataset1.findings) == len(dataset2.findings)
        for f1, f2 in zip(dataset1.findings, dataset2.findings):
            assert f1.finding_id == f2.finding_id

    def test_create_synthetic_benchmark(self):
        """Test creating synthetic benchmark."""
        generator = BenchmarkGenerator(seed=42)
        dataset = generator.create_synthetic_benchmark(n_per_type=10)

        assert len(dataset.findings) == 30  # 10 per type
        assert dataset.name == "paper_accuracy_benchmark"

    def test_synthetic_benchmark_distribution(self):
        """Test that synthetic benchmark has correct distribution."""
        generator = BenchmarkGenerator(seed=42)
        dataset = generator.create_synthetic_benchmark(n_per_type=10)

        dist = dataset.get_type_distribution()

        assert dist["data_analysis"] == 10
        assert dist["literature"] == 10
        assert dist["interpretation"] == 10

    def test_synthetic_benchmark_accuracy_rates(self):
        """Test that synthetic benchmark matches paper accuracy rates."""
        generator = BenchmarkGenerator(seed=42)
        dataset = generator.create_synthetic_benchmark(n_per_type=100)

        accuracy = dataset.get_accuracy_by_type()

        # Should be close to paper rates: 85.5%, 82.1%, 57.9%
        # With n=100, we expect close but not exact match
        assert abs(accuracy["data_analysis"] - 0.855) < 0.05
        assert abs(accuracy["literature"] - 0.821) < 0.05
        assert abs(accuracy["interpretation"] - 0.579) < 0.05

    def test_synthetic_benchmark_custom_accuracy(self):
        """Test synthetic benchmark with custom accuracy rates."""
        generator = BenchmarkGenerator(seed=42)
        custom_rates = {
            "data_analysis": 0.90,
            "literature": 0.80,
            "interpretation": 0.70
        }

        dataset = generator.create_synthetic_benchmark(
            n_per_type=100,
            accuracy_rates=custom_rates
        )

        accuracy = dataset.get_accuracy_by_type()

        assert abs(accuracy["data_analysis"] - 0.90) < 0.05
        assert abs(accuracy["literature"] - 0.80) < 0.05
        assert abs(accuracy["interpretation"] - 0.70) < 0.05

    def test_synthetic_benchmark_metadata(self):
        """Test that synthetic benchmark includes metadata."""
        generator = BenchmarkGenerator(seed=42)
        dataset = generator.create_synthetic_benchmark(n_per_type=10)

        assert "generator" in dataset.metadata
        assert "n_per_type" in dataset.metadata
        assert dataset.metadata["n_per_type"] == 10

    def test_create_from_findings(self):
        """Test creating benchmark from existing findings."""
        # Create mock findings
        findings = []
        for i in range(5):
            f = MagicMock()
            f.finding_id = f"f_{i:03d}"
            f.summary = f"Finding {i}"
            f.evidence_type = "data_analysis"
            f.domain = "biology"
            f.statistics = {"p_value": 0.01}
            f.citations = None
            findings.append(f)

        ground_truth = {
            "f_000": True,
            "f_001": True,
            "f_002": False,
            "f_003": True,
            "f_004": False
        }

        generator = BenchmarkGenerator()
        dataset = generator.create_from_findings(
            findings,
            ground_truth,
            name="custom_benchmark"
        )

        assert dataset.name == "custom_benchmark"
        assert len(dataset.findings) == 5
        assert dataset.get_overall_accuracy() == 0.6  # 3/5

    def test_create_from_findings_missing_ground_truth(self):
        """Test that findings without ground truth are skipped."""
        findings = []
        for i in range(3):
            f = MagicMock()
            f.finding_id = f"f_{i:03d}"
            f.summary = f"Finding {i}"
            f.evidence_type = "data_analysis"
            f.domain = "biology"
            f.statistics = None
            f.citations = None
            findings.append(f)

        ground_truth = {
            "f_000": True,
            # f_001 missing
            "f_002": False
        }

        generator = BenchmarkGenerator()
        dataset = generator.create_from_findings(findings, ground_truth)

        assert len(dataset.findings) == 2  # f_001 skipped

    def test_data_analysis_finding_generation(self):
        """Test that data analysis findings have statistics."""
        generator = BenchmarkGenerator(seed=42)
        dataset = generator.create_synthetic_benchmark(n_per_type=5)

        data_findings = dataset.get_by_type("data_analysis")

        for finding in data_findings:
            assert finding.statistics is not None

    def test_literature_finding_generation(self):
        """Test that literature findings are generated correctly."""
        generator = BenchmarkGenerator(seed=42)
        dataset = generator.create_synthetic_benchmark(n_per_type=5)

        lit_findings = dataset.get_by_type("literature")

        assert len(lit_findings) == 5
        for finding in lit_findings:
            assert finding.evidence_type == "literature"


# ============================================================================
# TestCreatePaperBenchmark
# ============================================================================

class TestCreatePaperBenchmark:
    """Tests for create_paper_benchmark utility function."""

    def test_create_paper_benchmark(self):
        """Test creating paper benchmark."""
        dataset = create_paper_benchmark()

        assert dataset.name == "paper_accuracy_benchmark"
        assert len(dataset.findings) == 90  # 30 per type

    def test_create_paper_benchmark_with_output(self):
        """Test creating paper benchmark with file output."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            dataset = create_paper_benchmark(output_path=path)

            assert Path(path).exists()
            loaded = BenchmarkDataset.load(path)
            assert loaded.name == dataset.name
        finally:
            Path(path).unlink()

    def test_create_paper_benchmark_accuracy_rates(self):
        """Test that paper benchmark has correct accuracy rates."""
        dataset = create_paper_benchmark()

        accuracy = dataset.get_accuracy_by_type()

        # Should match paper rates approximately
        # Using seed=42, with n=30, some variance expected
        assert accuracy["data_analysis"] >= 0.75
        assert accuracy["literature"] >= 0.70
        assert accuracy["interpretation"] >= 0.45

    def test_paper_benchmark_reproducible(self):
        """Test that paper benchmark is reproducible."""
        dataset1 = create_paper_benchmark()
        dataset2 = create_paper_benchmark()

        # Same seed should give same results
        assert len(dataset1.findings) == len(dataset2.findings)
        for f1, f2 in zip(dataset1.findings, dataset2.findings):
            assert f1.finding_id == f2.finding_id
