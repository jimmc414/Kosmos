"""
Benchmark Dataset Management for Paper Accuracy Validation.

Provides infrastructure for managing ground truth datasets used to validate
accuracy against paper claims.

Paper Reference (Section 8):
> "79.4% overall accuracy, 85.5% data analysis, 82.1% literature, 57.9% interpretation"
"""

import json
import logging
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkFinding:
    """A finding with known ground truth for validation."""
    finding_id: str
    summary: str
    evidence_type: str  # "data_analysis", "literature", "interpretation"
    ground_truth_accurate: bool
    domain: str
    source: str  # Where this benchmark came from (e.g., "synthetic", "expert", "paper")
    expert_notes: Optional[str] = None
    statistics: Optional[Dict[str, Any]] = None
    citations: Optional[List[Dict[str, str]]] = None

    def __post_init__(self):
        """Validate evidence type."""
        valid_types = {"data_analysis", "literature", "interpretation"}
        if self.evidence_type not in valid_types:
            raise ValueError(
                f"Invalid evidence_type '{self.evidence_type}'. "
                f"Must be one of: {valid_types}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkFinding':
        """Create BenchmarkFinding from dictionary."""
        return cls(**data)


@dataclass
class BenchmarkDataset:
    """Collection of benchmark findings for validation."""
    name: str
    version: str
    findings: List[BenchmarkFinding]
    created_at: str
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_by_type(self, evidence_type: str) -> List[BenchmarkFinding]:
        """
        Get findings filtered by evidence type.

        Args:
            evidence_type: Type to filter by

        Returns:
            List of findings with matching evidence_type
        """
        return [f for f in self.findings if f.evidence_type == evidence_type]

    def get_type_distribution(self) -> Dict[str, int]:
        """
        Get count by evidence type.

        Returns:
            Dictionary mapping evidence_type to count
        """
        distribution = {"data_analysis": 0, "literature": 0, "interpretation": 0}
        for finding in self.findings:
            if finding.evidence_type in distribution:
                distribution[finding.evidence_type] += 1
        return distribution

    def get_accuracy_by_type(self) -> Dict[str, float]:
        """
        Get ground truth accuracy rate by evidence type.

        Returns:
            Dictionary mapping evidence_type to accuracy rate (0-1)
        """
        accuracy = {}
        for evidence_type in ["data_analysis", "literature", "interpretation"]:
            findings = self.get_by_type(evidence_type)
            if findings:
                accurate = sum(1 for f in findings if f.ground_truth_accurate)
                accuracy[evidence_type] = accurate / len(findings)
            else:
                accuracy[evidence_type] = 0.0
        return accuracy

    def get_overall_accuracy(self) -> float:
        """
        Get overall ground truth accuracy rate.

        Returns:
            Overall accuracy rate (0-1)
        """
        if not self.findings:
            return 0.0
        accurate = sum(1 for f in self.findings if f.ground_truth_accurate)
        return accurate / len(self.findings)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "version": self.version,
            "findings": [f.to_dict() for f in self.findings],
            "created_at": self.created_at,
            "description": self.description,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """
        Serialize dataset to JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON formatted string
        """
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str) -> None:
        """
        Save dataset to JSON file.

        Args:
            path: File path to save to
        """
        with open(path, 'w') as f:
            f.write(self.to_json())
        logger.info(f"Saved benchmark dataset to {path}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkDataset':
        """
        Create BenchmarkDataset from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            BenchmarkDataset instance
        """
        findings = [BenchmarkFinding.from_dict(f) for f in data.get("findings", [])]
        return cls(
            name=data["name"],
            version=data["version"],
            findings=findings,
            created_at=data["created_at"],
            description=data["description"],
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'BenchmarkDataset':
        """
        Load dataset from JSON string.

        Args:
            json_str: JSON formatted string

        Returns:
            BenchmarkDataset instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: str) -> 'BenchmarkDataset':
        """
        Load dataset from JSON file.

        Args:
            path: File path to load from

        Returns:
            BenchmarkDataset instance
        """
        with open(path, 'r') as f:
            return cls.from_json(f.read())


class BenchmarkGenerator:
    """Generate benchmark datasets from various sources."""

    # Sample findings for different types and domains
    DATA_ANALYSIS_TEMPLATES = [
        ("Treatment group shows significant improvement (p={p_value:.3f}, d={effect_size:.2f})",
         {"p_value": (0.001, 0.05), "effect_size": (0.3, 0.9)}),
        ("Strong positive correlation between X and Y (r={correlation:.2f}, p<0.01)",
         {"correlation": (0.5, 0.9)}),
        ("Regression analysis reveals significant predictor (beta={beta:.2f}, p={p_value:.3f})",
         {"beta": (-0.8, 0.8), "p_value": (0.001, 0.05)}),
        ("ANOVA shows significant group differences (F={f_stat:.2f}, p={p_value:.3f})",
         {"f_stat": (3.0, 15.0), "p_value": (0.001, 0.05)}),
        ("Chi-square test indicates association (chi2={chi2:.2f}, p={p_value:.3f})",
         {"chi2": (5.0, 30.0), "p_value": (0.001, 0.05)}),
    ]

    LITERATURE_TEMPLATES = [
        "Prior studies have shown that {topic} leads to improved outcomes (Smith et al., 2020)",
        "Meta-analysis of {n_studies} studies confirms {finding} (Jones et al., 2021)",
        "Recent work demonstrates that {mechanism} underlies {phenomenon} (Chen et al., 2022)",
        "Literature review indicates {treatment} is effective for {condition} (Brown et al., 2019)",
        "Systematic review supports {hypothesis} with moderate evidence (Wilson et al., 2020)",
    ]

    INTERPRETATION_TEMPLATES = [
        "These findings suggest that {mechanism} may be a key driver of {outcome}",
        "The observed pattern indicates a potential role for {factor} in {process}",
        "Our results support the hypothesis that {hypothesis_detail}",
        "This evidence points to {conclusion} as a promising avenue for future research",
        "The data collectively suggest {interpretation} warrants further investigation",
    ]

    DOMAINS = ["biology", "clinical", "chemistry", "physics", "social_science"]
    TOPICS = ["gene expression", "metabolic pathways", "protein interactions",
              "cellular mechanisms", "treatment efficacy"]

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize benchmark generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.random = random.Random(seed)

    def _generate_data_analysis_finding(
        self,
        finding_id: str,
        domain: str,
        is_accurate: bool
    ) -> BenchmarkFinding:
        """Generate a data analysis finding."""
        template, params = self.random.choice(self.DATA_ANALYSIS_TEMPLATES)

        # Generate statistics
        statistics = {}
        summary_params = {}
        for param, (low, high) in params.items():
            value = self.random.uniform(low, high)
            statistics[param] = value
            summary_params[param] = value

        # For inaccurate findings, use implausible values
        if not is_accurate:
            if "p_value" in statistics:
                statistics["p_value"] = self.random.uniform(0.15, 0.5)
                summary_params["p_value"] = statistics["p_value"]

        summary = template.format(**summary_params)

        return BenchmarkFinding(
            finding_id=finding_id,
            summary=summary,
            evidence_type="data_analysis",
            ground_truth_accurate=is_accurate,
            domain=domain,
            source="synthetic",
            statistics=statistics,
        )

    def _generate_literature_finding(
        self,
        finding_id: str,
        domain: str,
        is_accurate: bool
    ) -> BenchmarkFinding:
        """Generate a literature finding."""
        template = self.random.choice(self.LITERATURE_TEMPLATES)
        topic = self.random.choice(self.TOPICS)

        summary = template.format(
            topic=topic,
            n_studies=self.random.randint(5, 50),
            finding="positive effects",
            mechanism="the proposed mechanism",
            phenomenon="the observed effect",
            treatment="the intervention",
            condition="the target condition",
            hypothesis="the primary hypothesis",
        )

        citations = [
            {"author": f"Author{i}", "year": str(self.random.randint(2018, 2024))}
            for i in range(self.random.randint(1, 5))
        ]

        return BenchmarkFinding(
            finding_id=finding_id,
            summary=summary,
            evidence_type="literature",
            ground_truth_accurate=is_accurate,
            domain=domain,
            source="synthetic",
            citations=citations,
            expert_notes="Synthetic literature finding" if not is_accurate else None,
        )

    def _generate_interpretation_finding(
        self,
        finding_id: str,
        domain: str,
        is_accurate: bool
    ) -> BenchmarkFinding:
        """Generate an interpretation finding."""
        template = self.random.choice(self.INTERPRETATION_TEMPLATES)

        summary = template.format(
            mechanism="the underlying process",
            outcome="the observed results",
            factor="this key variable",
            process="the biological process",
            hypothesis_detail="alterations in X contribute to Y",
            conclusion="this approach",
            interpretation="this relationship",
        )

        return BenchmarkFinding(
            finding_id=finding_id,
            summary=summary,
            evidence_type="interpretation",
            ground_truth_accurate=is_accurate,
            domain=domain,
            source="synthetic",
            expert_notes="Over-interpretation" if not is_accurate else None,
        )

    def create_synthetic_benchmark(
        self,
        n_per_type: int = 30,
        accuracy_rates: Optional[Dict[str, float]] = None
    ) -> BenchmarkDataset:
        """
        Create synthetic benchmark with known accuracy distribution.

        The default accuracy rates match the paper claims:
        - data_analysis: 85.5%
        - literature: 82.1%
        - interpretation: 57.9%

        Args:
            n_per_type: Number of findings per evidence type
            accuracy_rates: Optional custom accuracy rates by type

        Returns:
            BenchmarkDataset with synthetic findings
        """
        if accuracy_rates is None:
            # Paper claimed accuracy rates
            accuracy_rates = {
                "data_analysis": 0.855,
                "literature": 0.821,
                "interpretation": 0.579,
            }

        findings = []
        finding_counter = 0

        for evidence_type, rate in accuracy_rates.items():
            n_accurate = int(n_per_type * rate)
            n_inaccurate = n_per_type - n_accurate

            # Generate accurate findings
            for i in range(n_accurate):
                finding_counter += 1
                domain = self.random.choice(self.DOMAINS)

                if evidence_type == "data_analysis":
                    finding = self._generate_data_analysis_finding(
                        f"bench_{finding_counter:03d}", domain, True
                    )
                elif evidence_type == "literature":
                    finding = self._generate_literature_finding(
                        f"bench_{finding_counter:03d}", domain, True
                    )
                else:
                    finding = self._generate_interpretation_finding(
                        f"bench_{finding_counter:03d}", domain, True
                    )
                findings.append(finding)

            # Generate inaccurate findings
            for i in range(n_inaccurate):
                finding_counter += 1
                domain = self.random.choice(self.DOMAINS)

                if evidence_type == "data_analysis":
                    finding = self._generate_data_analysis_finding(
                        f"bench_{finding_counter:03d}", domain, False
                    )
                elif evidence_type == "literature":
                    finding = self._generate_literature_finding(
                        f"bench_{finding_counter:03d}", domain, False
                    )
                else:
                    finding = self._generate_interpretation_finding(
                        f"bench_{finding_counter:03d}", domain, False
                    )
                findings.append(finding)

        # Shuffle findings
        self.random.shuffle(findings)

        return BenchmarkDataset(
            name="paper_accuracy_benchmark",
            version="1.0.0",
            findings=findings,
            created_at=datetime.now().isoformat(),
            description=(
                f"Synthetic benchmark dataset with {len(findings)} findings "
                f"({n_per_type} per type) matching paper accuracy rates"
            ),
            metadata={
                "generator": "BenchmarkGenerator.create_synthetic_benchmark",
                "n_per_type": n_per_type,
                "target_accuracy_rates": accuracy_rates,
            }
        )

    def create_from_findings(
        self,
        findings: List[Any],
        ground_truth: Dict[str, bool],
        name: str = "custom_benchmark",
        description: str = "Custom benchmark from expert annotations"
    ) -> BenchmarkDataset:
        """
        Create benchmark from actual findings with expert annotations.

        Args:
            findings: List of Finding objects
            ground_truth: Dictionary mapping finding_id to accuracy (True/False)
            name: Name for the benchmark dataset
            description: Description of the dataset

        Returns:
            BenchmarkDataset with annotated findings
        """
        benchmark_findings = []

        for finding in findings:
            finding_id = getattr(finding, 'finding_id', None)
            if finding_id is None:
                continue

            if finding_id not in ground_truth:
                logger.warning(f"No ground truth for finding {finding_id}, skipping")
                continue

            benchmark_finding = BenchmarkFinding(
                finding_id=finding_id,
                summary=getattr(finding, 'summary', ''),
                evidence_type=getattr(finding, 'evidence_type', 'data_analysis'),
                ground_truth_accurate=ground_truth[finding_id],
                domain=getattr(finding, 'domain', 'unknown'),
                source="expert_annotation",
                statistics=getattr(finding, 'statistics', None),
                citations=getattr(finding, 'citations', None),
            )
            benchmark_findings.append(benchmark_finding)

        return BenchmarkDataset(
            name=name,
            version="1.0.0",
            findings=benchmark_findings,
            created_at=datetime.now().isoformat(),
            description=description,
            metadata={
                "generator": "BenchmarkGenerator.create_from_findings",
                "total_findings": len(findings),
                "annotated_findings": len(benchmark_findings),
            }
        )


def create_paper_benchmark(output_path: Optional[str] = None) -> BenchmarkDataset:
    """
    Create the standard paper accuracy benchmark dataset.

    This creates a synthetic dataset with accuracy rates matching the paper claims:
    - 85.5% data analysis
    - 82.1% literature
    - 57.9% interpretation

    Args:
        output_path: Optional path to save the dataset

    Returns:
        BenchmarkDataset instance
    """
    generator = BenchmarkGenerator(seed=42)
    dataset = generator.create_synthetic_benchmark(n_per_type=30)

    if output_path:
        dataset.save(output_path)

    return dataset
