"""
Paper Accuracy Validation Framework.

Implements accuracy tracking and reporting by statement type to validate
against paper claims (79.4% overall, 85.5% data analysis, 82.1% literature,
57.9% interpretation).

Paper Reference (Section 8):
> "79.4% overall accuracy, 85.5% data analysis, 82.1% literature, 57.9% interpretation"
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class AccuracyTarget:
    """Paper-defined accuracy targets by statement type."""
    statement_type: str
    paper_accuracy: float  # Paper claimed value (e.g., 0.794 for 79.4%)
    target_threshold: float  # Implementation target (usually paper - buffer)
    description: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class TypeAccuracyResult:
    """Accuracy result for a single statement type."""
    statement_type: str
    total_count: int
    accurate_count: int
    accuracy: float
    target: float
    meets_target: bool
    paper_claim: float
    delta_from_paper: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @property
    def accuracy_percentage(self) -> float:
        """Return accuracy as a percentage."""
        return self.accuracy * 100

    @property
    def target_percentage(self) -> float:
        """Return target as a percentage."""
        return self.target * 100

    @property
    def paper_claim_percentage(self) -> float:
        """Return paper claim as a percentage."""
        return self.paper_claim * 100

    @property
    def delta_percentage(self) -> float:
        """Return delta as a percentage."""
        return self.delta_from_paper * 100


@dataclass
class ValidationRecord:
    """Record of a single validation decision."""
    finding_id: str
    is_accurate: bool
    expert_notes: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass
class AccuracyReport:
    """Complete accuracy validation report."""
    overall_accuracy: float
    overall_target: float
    overall_meets_target: bool
    by_type: Dict[str, TypeAccuracyResult]
    total_findings: int
    validated_findings: int
    paper_comparison: Dict[str, float]  # Deltas from paper claims
    timestamp: str
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "overall_accuracy": self.overall_accuracy,
            "overall_target": self.overall_target,
            "overall_meets_target": self.overall_meets_target,
            "by_type": {k: v.to_dict() for k, v in self.by_type.items()},
            "total_findings": self.total_findings,
            "validated_findings": self.validated_findings,
            "paper_comparison": self.paper_comparison,
            "timestamp": self.timestamp,
            "recommendations": self.recommendations,
        }
        return result

    @property
    def overall_accuracy_percentage(self) -> float:
        """Return overall accuracy as a percentage."""
        return self.overall_accuracy * 100

    @property
    def all_types_meet_targets(self) -> bool:
        """Check if all statement types meet their targets."""
        return all(result.meets_target for result in self.by_type.values())


class AccuracyTracker:
    """
    Track and report accuracy by finding/statement type.

    Implements paper accuracy validation against Section 8 claims:
    - 79.4% overall accuracy
    - 85.5% data analysis accuracy
    - 82.1% literature accuracy
    - 57.9% interpretation accuracy

    Usage:
        tracker = AccuracyTracker()

        # Add findings
        for finding in findings:
            tracker.add_finding(finding)

        # Record expert validations
        tracker.validate_finding("f001", is_accurate=True)
        tracker.validate_finding("f002", is_accurate=False)

        # Generate report
        report = tracker.generate_report()
        print(f"Overall: {report.overall_accuracy_percentage:.1f}%")
    """

    # Paper-defined accuracy targets (Section 8)
    PAPER_TARGETS = {
        "data_analysis": AccuracyTarget(
            statement_type="data_analysis",
            paper_accuracy=0.855,
            target_threshold=0.80,
            description="Data analysis-based statements"
        ),
        "literature": AccuracyTarget(
            statement_type="literature",
            paper_accuracy=0.821,
            target_threshold=0.75,
            description="Literature review-based statements"
        ),
        "interpretation": AccuracyTarget(
            statement_type="interpretation",
            paper_accuracy=0.579,
            target_threshold=0.50,
            description="Interpretation/synthesis statements"
        ),
        "overall": AccuracyTarget(
            statement_type="overall",
            paper_accuracy=0.794,
            target_threshold=0.75,
            description="All statement types combined"
        ),
    }

    # Valid evidence types
    VALID_EVIDENCE_TYPES = {"data_analysis", "literature", "interpretation"}

    def __init__(self):
        """Initialize accuracy tracker."""
        self.findings_by_type: Dict[str, List[Any]] = defaultdict(list)
        self.validations: Dict[str, ValidationRecord] = {}
        self._finding_ids: set = set()

    def add_finding(self, finding: Any) -> None:
        """
        Add a finding for tracking.

        Args:
            finding: Finding object with finding_id and evidence_type attributes
        """
        finding_id = getattr(finding, 'finding_id', None)
        if finding_id is None:
            raise ValueError("Finding must have a finding_id attribute")

        if finding_id in self._finding_ids:
            logger.warning(f"Finding {finding_id} already added, skipping duplicate")
            return

        evidence_type = getattr(finding, 'evidence_type', 'data_analysis')
        if evidence_type not in self.VALID_EVIDENCE_TYPES:
            logger.warning(
                f"Unknown evidence_type '{evidence_type}' for finding {finding_id}, "
                f"using 'data_analysis'"
            )
            evidence_type = 'data_analysis'

        self.findings_by_type[evidence_type].append(finding)
        self._finding_ids.add(finding_id)
        logger.debug(f"Added finding {finding_id} with type {evidence_type}")

    def validate_finding(
        self,
        finding_id: str,
        is_accurate: bool,
        expert_notes: Optional[str] = None
    ) -> None:
        """
        Record expert validation for a finding.

        Args:
            finding_id: ID of the finding to validate
            is_accurate: Whether the finding is accurate according to expert review
            expert_notes: Optional notes from the expert reviewer
        """
        if finding_id not in self._finding_ids:
            raise ValueError(f"Finding {finding_id} not found in tracker")

        self.validations[finding_id] = ValidationRecord(
            finding_id=finding_id,
            is_accurate=is_accurate,
            expert_notes=expert_notes,
            timestamp=datetime.now().isoformat()
        )
        logger.debug(f"Validated finding {finding_id}: accurate={is_accurate}")

    def compute_type_accuracy(self, evidence_type: str) -> TypeAccuracyResult:
        """
        Compute accuracy for a specific finding type.

        Args:
            evidence_type: The type to compute accuracy for

        Returns:
            TypeAccuracyResult with accuracy metrics
        """
        if evidence_type not in self.PAPER_TARGETS:
            raise ValueError(f"Unknown evidence type: {evidence_type}")

        target = self.PAPER_TARGETS[evidence_type]

        if evidence_type == "overall":
            # Compute overall accuracy across all types
            all_findings = []
            for findings in self.findings_by_type.values():
                all_findings.extend(findings)
            findings_list = all_findings
        else:
            findings_list = self.findings_by_type.get(evidence_type, [])

        # Filter to validated findings
        validated = [
            f for f in findings_list
            if getattr(f, 'finding_id', None) in self.validations
        ]

        if not validated:
            return TypeAccuracyResult(
                statement_type=evidence_type,
                total_count=0,
                accurate_count=0,
                accuracy=0.0,
                target=target.target_threshold,
                meets_target=False,
                paper_claim=target.paper_accuracy,
                delta_from_paper=0.0
            )

        accurate = sum(
            1 for f in validated
            if self.validations[getattr(f, 'finding_id')].is_accurate
        )
        accuracy = accurate / len(validated)

        return TypeAccuracyResult(
            statement_type=evidence_type,
            total_count=len(validated),
            accurate_count=accurate,
            accuracy=accuracy,
            target=target.target_threshold,
            meets_target=accuracy >= target.target_threshold,
            paper_claim=target.paper_accuracy,
            delta_from_paper=accuracy - target.paper_accuracy
        )

    def compute_accuracy_by_type(self) -> Dict[str, TypeAccuracyResult]:
        """
        Compute accuracy breakdown by statement type.

        Returns:
            Dictionary mapping evidence type to TypeAccuracyResult
        """
        results = {}
        for evidence_type in self.VALID_EVIDENCE_TYPES:
            results[evidence_type] = self.compute_type_accuracy(evidence_type)
        return results

    def compare_to_paper(self) -> Dict[str, float]:
        """
        Compare results to paper claims and return deltas.

        Returns:
            Dictionary mapping evidence type to delta from paper claim
        """
        deltas = {}
        for evidence_type in list(self.VALID_EVIDENCE_TYPES) + ["overall"]:
            result = self.compute_type_accuracy(evidence_type)
            deltas[evidence_type] = result.delta_from_paper
        return deltas

    def _generate_recommendations(
        self,
        by_type: Dict[str, TypeAccuracyResult],
        overall_result: TypeAccuracyResult
    ) -> List[str]:
        """Generate recommendations based on accuracy results."""
        recommendations = []

        # Check overall accuracy
        if not overall_result.meets_target:
            recommendations.append(
                f"Overall accuracy ({overall_result.accuracy_percentage:.1f}%) "
                f"below target ({overall_result.target_percentage:.1f}%). "
                "Consider reviewing validation criteria."
            )

        # Check each type
        for evidence_type, result in by_type.items():
            if not result.meets_target:
                recommendations.append(
                    f"{evidence_type.replace('_', ' ').title()} accuracy "
                    f"({result.accuracy_percentage:.1f}%) below target "
                    f"({result.target_percentage:.1f}%). "
                    f"Focus improvement efforts on this category."
                )
            elif result.delta_from_paper < -0.05:  # More than 5% below paper
                recommendations.append(
                    f"{evidence_type.replace('_', ' ').title()} accuracy "
                    f"({result.accuracy_percentage:.1f}%) meets target but is "
                    f"{abs(result.delta_percentage):.1f}% below paper claim "
                    f"({result.paper_claim_percentage:.1f}%)."
                )

        if not recommendations:
            recommendations.append(
                "All accuracy targets met. System performs within expected parameters."
            )

        return recommendations

    def generate_report(self) -> AccuracyReport:
        """
        Generate comprehensive accuracy report.

        Returns:
            AccuracyReport with all accuracy metrics and recommendations
        """
        by_type = self.compute_accuracy_by_type()
        overall_result = self.compute_type_accuracy("overall")
        paper_comparison = self.compare_to_paper()

        # Count findings
        total_findings = sum(len(f) for f in self.findings_by_type.values())
        validated_findings = len(self.validations)

        recommendations = self._generate_recommendations(by_type, overall_result)

        return AccuracyReport(
            overall_accuracy=overall_result.accuracy,
            overall_target=overall_result.target,
            overall_meets_target=overall_result.meets_target,
            by_type=by_type,
            total_findings=total_findings,
            validated_findings=validated_findings,
            paper_comparison=paper_comparison,
            timestamp=datetime.now().isoformat(),
            recommendations=recommendations
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for the tracker.

        Returns:
            Dictionary with summary statistics
        """
        total = sum(len(f) for f in self.findings_by_type.values())
        validated = len(self.validations)
        accurate = sum(1 for v in self.validations.values() if v.is_accurate)

        return {
            "total_findings": total,
            "validated_findings": validated,
            "accurate_findings": accurate,
            "validation_rate": validated / total if total > 0 else 0.0,
            "accuracy_rate": accurate / validated if validated > 0 else 0.0,
            "by_type": {
                k: len(v) for k, v in self.findings_by_type.items()
            }
        }


class AccuracyReporter:
    """Generate markdown reports for accuracy validation."""

    def __init__(self):
        """Initialize reporter."""
        pass

    def generate_markdown_report(self, report: AccuracyReport) -> str:
        """
        Generate detailed markdown report.

        Args:
            report: AccuracyReport to render

        Returns:
            Markdown formatted report string
        """
        lines = [
            "# Paper Accuracy Validation Report",
            "",
            f"**Generated**: {report.timestamp[:10]}",
            f"**Total Findings**: {report.total_findings}",
            f"**Validated Findings**: {report.validated_findings}",
            "",
            "## Summary",
            "",
            "| Metric | Value | Target | Status |",
            "|--------|-------|--------|--------|",
            f"| Overall Accuracy | {report.overall_accuracy * 100:.1f}% | "
            f"{report.overall_target * 100:.1f}% | "
            f"{'PASS' if report.overall_meets_target else 'FAIL'} |",
            f"| Paper Claim | 79.4% | - | "
            f"{report.paper_comparison.get('overall', 0) * 100:+.1f}% |",
            "",
            "## Accuracy by Statement Type",
            "",
            "| Type | Count | Accurate | Accuracy | Target | Paper | Delta | Status |",
            "|------|-------|----------|----------|--------|-------|-------|--------|",
        ]

        # Add rows for each type
        for evidence_type in ["data_analysis", "literature", "interpretation"]:
            result = report.by_type.get(evidence_type)
            if result:
                status = "PASS" if result.meets_target else "FAIL"
                lines.append(
                    f"| {evidence_type.replace('_', ' ').title()} | "
                    f"{result.total_count} | {result.accurate_count} | "
                    f"{result.accuracy_percentage:.1f}% | "
                    f"{result.target_percentage:.0f}% | "
                    f"{result.paper_claim_percentage:.1f}% | "
                    f"{result.delta_percentage:+.1f}% | {status} |"
                )

        lines.extend([
            "",
            "## Comparison to Paper Claims",
            "",
        ])

        # Add comparison details
        overall_delta = report.paper_comparison.get("overall", 0)
        lines.append(
            f"- **Overall**: {report.overall_accuracy * 100:.1f}% vs 79.4% "
            f"({overall_delta * 100:+.1f}%) - "
            f"{'Within acceptable range' if abs(overall_delta) < 0.05 else 'Notable deviation'}"
        )

        for evidence_type in ["data_analysis", "literature", "interpretation"]:
            result = report.by_type.get(evidence_type)
            if result:
                delta_desc = "Exceeds paper" if result.delta_from_paper > 0 else "Below paper"
                if result.meets_target:
                    delta_desc += " but meets target"
                lines.append(
                    f"- **{evidence_type.replace('_', ' ').title()}**: "
                    f"{result.accuracy_percentage:.1f}% vs {result.paper_claim_percentage:.1f}% "
                    f"({result.delta_percentage:+.1f}%) - {delta_desc}"
                )

        lines.extend([
            "",
            "## Recommendations",
            "",
        ])

        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"{i}. {rec}")

        return "\n".join(lines)

    def generate_summary(self, report: AccuracyReport) -> str:
        """
        Generate executive summary.

        Args:
            report: AccuracyReport to summarize

        Returns:
            Brief summary string
        """
        status = "PASS" if report.overall_meets_target else "FAIL"
        all_types_pass = report.all_types_meet_targets

        summary_parts = [
            f"Overall Accuracy: {report.overall_accuracy * 100:.1f}% ({status})",
        ]

        if all_types_pass:
            summary_parts.append("All statement types meet targets.")
        else:
            failing = [
                k for k, v in report.by_type.items()
                if not v.meets_target
            ]
            summary_parts.append(f"Types below target: {', '.join(failing)}")

        delta = report.paper_comparison.get("overall", 0)
        if abs(delta) < 0.05:
            summary_parts.append("Results consistent with paper claims.")
        else:
            summary_parts.append(f"Delta from paper: {delta * 100:+.1f}%")

        return " | ".join(summary_parts)

    def generate_json_report(self, report: AccuracyReport) -> str:
        """
        Generate JSON report.

        Args:
            report: AccuracyReport to serialize

        Returns:
            JSON formatted report string
        """
        import json
        return json.dumps(report.to_dict(), indent=2)
