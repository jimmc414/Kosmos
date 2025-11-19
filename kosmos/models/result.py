"""
Experiment result data models.

Defines Pydantic models for experiment results, extending the database Result model
with validation and structured data handling.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import json


class ResultStatus(str, Enum):
    """Status of experiment result."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    ERROR = "error"


class ExecutionMetadata(BaseModel):
    """Metadata about experiment execution."""

    start_time: datetime = Field(..., description="Execution start timestamp")
    end_time: datetime = Field(..., description="Execution end timestamp")
    duration_seconds: float = Field(..., ge=0, description="Execution duration in seconds")

    # System information
    python_version: str = Field(..., description="Python version used")
    platform: str = Field(..., description="Operating system platform")
    hostname: Optional[str] = Field(None, description="Machine hostname")

    # Resource usage
    cpu_time_seconds: Optional[float] = Field(None, ge=0, description="CPU time consumed")
    memory_peak_mb: Optional[float] = Field(None, ge=0, description="Peak memory usage in MB")

    # Execution environment
    random_seed: Optional[int] = Field(None, description="Random seed used")
    library_versions: Dict[str, str] = Field(
        default_factory=dict,
        description="Versions of key libraries used"
    )

    # Parameters
    experiment_id: str = Field(..., description="Experiment ID")
    protocol_id: str = Field(..., description="Protocol ID")
    hypothesis_id: Optional[str] = Field(None, description="Hypothesis ID")

    # Execution details
    sandbox_used: bool = Field(default=False, description="Whether sandbox was used")
    timeout_occurred: bool = Field(default=False, description="Whether timeout occurred")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    warnings: List[str] = Field(default_factory=list, description="Warnings generated")


class StatisticalTestResult(BaseModel):
    """Result of a statistical test."""

    test_type: str = Field(..., description="Type of test (t-test, ANOVA, etc.)")
    test_name: str = Field(..., description="Human-readable test name")

    # Test statistics
    statistic: float = Field(..., description="Test statistic value")
    p_value: float = Field(..., ge=0, le=1, description="P-value")

    # Effect size (optional)
    effect_size: Optional[float] = Field(None, description="Effect size measure")
    effect_size_type: Optional[str] = Field(None, description="Type of effect size (Cohen's d, etc.)")

    # Confidence intervals (optional)
    confidence_interval: Optional[Dict[str, float]] = Field(
        None,
        description="Confidence interval (lower, upper)"
    )
    confidence_level: Optional[float] = Field(None, ge=0, le=1, description="CI confidence level")

    # Significance
    significant_0_05: bool = Field(..., description="Significant at α=0.05")
    significant_0_01: bool = Field(..., description="Significant at α=0.01")
    significant_0_001: bool = Field(..., description="Significant at α=0.001")
    significance_label: str = Field(..., description="Significance label (**, *, ns)")

    # Sample information
    sample_size: Optional[int] = Field(None, ge=0, description="Total sample size")
    degrees_of_freedom: Optional[int] = Field(None, description="Degrees of freedom")

    # Additional test-specific data
    additional_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional test-specific statistics"
    )

    # Interpretation
    interpretation: Optional[str] = Field(None, description="Human-readable interpretation")
    is_primary: bool = Field(False, description="Whether this is the primary test for the experiment")


class VariableResult(BaseModel):
    """Result data for a specific variable."""

    variable_name: str = Field(..., description="Variable name")
    variable_type: str = Field(..., description="Variable type (independent, dependent, etc.)")

    # Summary statistics
    mean: Optional[float] = Field(None, description="Mean value")
    median: Optional[float] = Field(None, description="Median value")
    std: Optional[float] = Field(None, description="Standard deviation")
    min: Optional[float] = Field(None, description="Minimum value")
    max: Optional[float] = Field(None, description="Maximum value")

    # Data points
    values: Optional[List[Union[float, int, str]]] = Field(
        None,
        description="Raw data values (if not too large)"
    )
    n_samples: Optional[int] = Field(None, ge=0, description="Number of samples")
    n_missing: Optional[int] = Field(None, ge=0, description="Number of missing values")


class ExperimentResult(BaseModel):
    """
    Complete experiment result model.

    Extends database Result model with structured Pydantic validation.
    """

    # Identifiers
    id: Optional[str] = Field(None, description="Result ID")
    experiment_id: str = Field(..., description="Experiment ID")
    protocol_id: str = Field(..., description="Protocol ID used")
    hypothesis_id: Optional[str] = Field(None, description="Hypothesis ID tested")

    # Status
    status: ResultStatus = Field(..., description="Result status")

    # Data
    raw_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw output data from execution"
    )
    processed_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Processed/cleaned data"
    )

    # Variables
    variable_results: List[VariableResult] = Field(
        default_factory=list,
        description="Results for each variable"
    )

    # Statistical tests
    statistical_tests: List[StatisticalTestResult] = Field(
        default_factory=list,
        description="Statistical test results"
    )
    primary_test: Optional[str] = Field(
        None,
        description="Name of primary statistical test"
    )

    # Key results
    primary_p_value: Optional[float] = Field(None, ge=0, le=1, description="Primary p-value")
    primary_effect_size: Optional[float] = Field(None, description="Primary effect size")
    primary_ci_lower: Optional[float] = Field(None, description="Lower bound of the primary confidence interval")
    primary_ci_upper: Optional[float] = Field(None, description="Upper bound of the primary confidence interval")
    supports_hypothesis: Optional[bool] = Field(
        None,
        description="Whether results support the hypothesis"
    )

    # Metadata
    metadata: ExecutionMetadata = Field(..., description="Execution metadata")

    # Versioning
    version: int = Field(default=1, ge=1, description="Result version number")
    parent_result_id: Optional[str] = Field(None, description="ID of parent result if re-run")

    # Outputs
    stdout: Optional[str] = Field(None, description="Standard output from execution")
    stderr: Optional[str] = Field(None, description="Standard error from execution")
    generated_files: List[str] = Field(
        default_factory=list,
        description="Paths to generated files (plots, data, etc.)"
    )

    # Interpretation
    summary: Optional[str] = Field(None, description="Summary of results")
    interpretation: Optional[str] = Field(None, description="Interpretation of findings")
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for follow-up"
    )

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Result creation time")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")

    @field_validator('statistical_tests')
    @classmethod
    def validate_statistical_tests(cls, v: List[StatisticalTestResult]) -> List[StatisticalTestResult]:
        """Validate statistical tests list."""
        # Check for duplicate test names
        test_names = [test.test_name for test in v]
        if len(test_names) != len(set(test_names)):
            raise ValueError("Duplicate test names in statistical_tests")
        return v

    @field_validator('primary_test', mode='before')
    @classmethod
    def validate_primary_test(cls, v: Optional[str], info) -> Optional[str]:
        """Validate primary test exists in statistical_tests."""
        if v is not None and 'statistical_tests' in info.data:
            # Pydantic V2 validators run before the model is created, so we're working with dicts
            test_names = [test.get('test_name') for test in info.data['statistical_tests']]
            if v not in test_names:
                raise ValueError(f"Primary test '{v}' not found in statistical_tests")
        return v

    def get_primary_test_result(self) -> Optional[StatisticalTestResult]:
        """Get the primary statistical test result."""
        if self.primary_test is None:
            return None

        for test in self.statistical_tests:
            if test.test_name == self.primary_test:
                return test

        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return self.model_dump(mode='json', exclude_none=True)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return self.model_dump_json(indent=2, exclude_none=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentResult":
        """Create from dictionary."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> "ExperimentResult":
        """Create from JSON string."""
        return cls.model_validate_json(json_str)

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if primary result is statistically significant."""
        if self.primary_p_value is None:
            return False
        return self.primary_p_value < alpha

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for all variables."""
        stats = {}
        for var_result in self.variable_results:
            stats[var_result.variable_name] = {
                'mean': var_result.mean,
                'median': var_result.median,
                'std': var_result.std,
                'min': var_result.min,
                'max': var_result.max,
                'n_samples': var_result.n_samples
            }
        return stats


class ResultExport(BaseModel):
    """Model for exporting results in various formats."""

    result: ExperimentResult
    format: str = Field(..., description="Export format (json, csv, markdown)")

    def export_json(self) -> str:
        """Export as JSON."""
        return self.result.to_json()

    def export_csv(self) -> str:
        """Export variable results as CSV."""
        import pandas as pd

        # Create DataFrame from variable results
        data = []
        for var in self.result.variable_results:
            row = {
                'variable': var.variable_name,
                'type': var.variable_type,
                'mean': var.mean,
                'median': var.median,
                'std': var.std,
                'min': var.min,
                'max': var.max,
                'n_samples': var.n_samples
            }
            data.append(row)

        df = pd.DataFrame(data)
        return df.to_csv(index=False)

    def export_markdown(self) -> str:
        """Export as Markdown report."""
        lines = []

        # Header
        lines.append(f"# Experiment Result Report")
        lines.append(f"**Experiment ID:** {self.result.experiment_id}")
        lines.append(f"**Status:** {self.result.status.value}")
        lines.append(f"**Date:** {self.result.created_at.isoformat()}")
        lines.append("")

        # Summary
        if self.result.summary:
            lines.append("## Summary")
            lines.append(self.result.summary)
            lines.append("")

        # Statistical Tests
        if self.result.statistical_tests:
            lines.append("## Statistical Tests")
            lines.append("")
            for test in self.result.statistical_tests:
                lines.append(f"### {test.test_name}")
                lines.append(f"- **Statistic:** {test.statistic:.4f}")
                lines.append(f"- **P-value:** {test.p_value:.6f}")
                lines.append(f"- **Significance:** {test.significance_label}")
                if test.effect_size:
                    lines.append(f"- **Effect Size:** {test.effect_size:.4f} ({test.effect_size_type})")
                lines.append("")

        # Variables
        if self.result.variable_results:
            lines.append("## Variable Statistics")
            lines.append("")
            lines.append("| Variable | Mean | Median | Std | Min | Max | N |")
            lines.append("|----------|------|--------|-----|-----|-----|---|")
            for var in self.result.variable_results:
                lines.append(
                    f"| {var.variable_name} | "
                    f"{var.mean:.2f} | {var.median:.2f} | {var.std:.2f} | "
                    f"{var.min:.2f} | {var.max:.2f} | {var.n_samples} |"
                )
            lines.append("")

        # Interpretation
        if self.result.interpretation:
            lines.append("## Interpretation")
            lines.append(self.result.interpretation)
            lines.append("")

        # Recommendations
        if self.result.recommendations:
            lines.append("## Recommendations")
            for rec in self.result.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        # Metadata
        lines.append("## Execution Metadata")
        lines.append(f"- **Duration:** {self.result.metadata.duration_seconds:.2f}s")
        lines.append(f"- **Python Version:** {self.result.metadata.python_version}")
        if self.result.metadata.memory_peak_mb:
            lines.append(f"- **Peak Memory:** {self.result.metadata.memory_peak_mb:.1f} MB")
        lines.append("")

        return "\n".join(lines)
