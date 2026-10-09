"""
Accuracy Validator for Benchmark Datasets.

Compares KOSMOS conclusions against ground truth from benchmark datasets
to validate paper accuracy claims.

Paper Reference (Section 8):
> "79.4% overall accuracy, 85.5% data analysis, 82.1% literature, 57.9% interpretation"
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

from kosmos.validation.benchmark_dataset import BenchmarkDataset, BenchmarkFinding

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a single finding."""
    finding_id: str
    evidence_type: str
    ground_truth: bool
    kosmos_conclusion: Optional[bool]
    correct: bool
    details: str = ""


@dataclass
class AccuracyValidationReport:
    """Report from running accuracy validation against a benchmark dataset."""
    dataset_name: str
    dataset_version: str
    total_findings: int
    validated_findings: int
    overall_accuracy: float
    accuracy_by_type: Dict[str, float]
    counts_by_type: Dict[str, Dict[str, int]]
    results: List[ValidationResult] = field(default_factory=list)

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"Accuracy Validation Report: {self.dataset_name} v{self.dataset_version}",
            f"  Overall: {self.overall_accuracy:.1%} ({self.validated_findings}/{self.total_findings})",
        ]
        for etype, acc in sorted(self.accuracy_by_type.items()):
            counts = self.counts_by_type.get(etype, {})
            lines.append(
                f"  {etype}: {acc:.1%} "
                f"({counts.get('correct', 0)}/{counts.get('total', 0)})"
            )
        return "\n".join(lines)


class AccuracyValidator:
    """
    Validates KOSMOS conclusions against benchmark ground truth.

    Usage:
        validator = AccuracyValidator()
        report = validator.validate_dataset(dataset, evaluate_fn)
    """

    def validate_dataset(
        self,
        dataset: BenchmarkDataset,
        evaluate_finding: Any,
    ) -> AccuracyValidationReport:
        """
        Run validation against a benchmark dataset.

        Args:
            dataset: BenchmarkDataset with ground truth findings.
            evaluate_finding: Callable(BenchmarkFinding) -> bool or None.
                Returns True/False for KOSMOS conclusion, None if unable to evaluate.

        Returns:
            AccuracyValidationReport with per-type and overall accuracy.
        """
        results: List[ValidationResult] = []
        type_correct: Dict[str, int] = {}
        type_total: Dict[str, int] = {}

        for finding in dataset.findings:
            try:
                conclusion = evaluate_finding(finding)
            except Exception as e:
                logger.warning(f"Failed to evaluate finding {finding.finding_id}: {e}")
                conclusion = None

            if conclusion is None:
                continue

            correct = conclusion == finding.ground_truth_accurate
            results.append(ValidationResult(
                finding_id=finding.finding_id,
                evidence_type=finding.evidence_type,
                ground_truth=finding.ground_truth_accurate,
                kosmos_conclusion=conclusion,
                correct=correct,
            ))

            etype = finding.evidence_type
            type_total[etype] = type_total.get(etype, 0) + 1
            if correct:
                type_correct[etype] = type_correct.get(etype, 0) + 1

        validated = len(results)
        overall_acc = sum(1 for r in results if r.correct) / validated if validated else 0.0

        accuracy_by_type = {}
        counts_by_type = {}
        for etype in sorted(set(type_total.keys())):
            total = type_total[etype]
            correct_count = type_correct.get(etype, 0)
            accuracy_by_type[etype] = correct_count / total if total else 0.0
            counts_by_type[etype] = {"correct": correct_count, "total": total}

        report = AccuracyValidationReport(
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            total_findings=len(dataset.findings),
            validated_findings=validated,
            overall_accuracy=overall_acc,
            accuracy_by_type=accuracy_by_type,
            counts_by_type=counts_by_type,
            results=results,
        )

        logger.info(f"Accuracy validation complete:\n{report.summary()}")
        return report

    @staticmethod
    def load_dataset(path: str) -> BenchmarkDataset:
        """Load a benchmark dataset from JSON file."""
        return BenchmarkDataset.load(path)
