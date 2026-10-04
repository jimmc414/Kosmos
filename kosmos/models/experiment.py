"""
Experiment data models for runtime use.

Provides Pydantic models for experiment design, protocols, and validation.
Complements the SQLAlchemy Experiment model in kosmos.db.models.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import datetime, timezone
from enum import Enum
import logging

from kosmos.models.hypothesis import ExperimentType
from kosmos.config import _DEFAULT_CLAUDE_SONNET_MODEL

_experiment_logger = logging.getLogger(__name__)

_MAX_SAMPLE_SIZE = 100_000

class VariableType(str, Enum):
    """Types of variables in an experiment."""
    INDEPENDENT = "independent"  # Manipulated variable
    DEPENDENT = "dependent"  # Measured outcome
    CONTROL = "control"  # Held constant
    CONFOUNDING = "confounding"  # Potential confound to track


class StatisticalTest(str, Enum):
    """Common statistical tests."""
    T_TEST = "t_test"
    ANOVA = "anova"
    CHI_SQUARE = "chi_square"
    CORRELATION = "correlation"
    REGRESSION = "regression"
    MANN_WHITNEY = "mann_whitney"
    KRUSKAL_WALLIS = "kruskal_wallis"
    WILCOXON = "wilcoxon"
    CUSTOM = "custom"


class Variable(BaseModel):
    """
    A variable in an experiment.

    Example:
        ```python
        var = Variable(
            name="attention_heads",
            type=VariableType.INDEPENDENT,
            description="Number of attention heads in transformer",
            values=[8, 12, 16],
            unit="count"
        )
        ```
    """
    name: str = Field(..., description="Variable name")
    type: VariableType
    description: str = Field(..., min_length=10, description="Clear description of the variable")

    # For independent variables: possible values to test
    # For dependent variables: expected measurement method
    # For control variables: fixed value
    values: Optional[List[Any]] = None
    fixed_value: Optional[Any] = None

    unit: Optional[str] = None  # e.g., "seconds", "percentage", "count"
    measurement_method: Optional[str] = None  # How to measure/compute this variable

    # Exact dataset column this variable is bound to (set when a dataset is supplied)
    column: Optional[str] = None

    @field_validator('description')
    @classmethod
    def validate_description(cls, v: str) -> str:
        """Ensure description is clear."""
        if not v or v.strip() == "":
            raise ValueError("Description cannot be empty")
        if len(v.strip()) < 10:
            raise ValueError("Description must be at least 10 characters")
        return v.strip()


class ControlGroup(BaseModel):
    """
    Control group specification for an experiment.

    Example:
        ```python
        control = ControlGroup(
            name="baseline_model",
            description="Standard transformer with 8 attention heads",
            variables={"attention_heads": 8},
            rationale="Industry standard baseline"
        )
        ```
    """
    name: str = Field(..., description="Control group name")
    description: str = Field(..., min_length=5, description="What this control group represents")

    # Variable values for this control group
    variables: Dict[str, Any] = Field(..., description="Variable settings for control group")

    # Why this control group is necessary
    rationale: str = Field(..., min_length=10, description="Scientific rationale for this control")

    sample_size: Optional[int] = Field(None, ge=1, description="Required sample size")

    @field_validator('sample_size', mode='before')
    @classmethod
    def coerce_sample_size(cls, v):
        """Coerce string sample_size from LLM output to int, enforce upper bound."""
        if v is None:
            return v
        if isinstance(v, str):
            try:
                v = int(v)
            except (ValueError, TypeError):
                _experiment_logger.warning(
                    f"Could not coerce sample_size '{v}' to int, setting to None"
                )
                return None
        if isinstance(v, (int, float)) and v > _MAX_SAMPLE_SIZE:
            _experiment_logger.warning(
                f"sample_size {v} exceeds maximum {_MAX_SAMPLE_SIZE}, clamping"
            )
            return _MAX_SAMPLE_SIZE
        return v

    @field_validator('description', 'rationale')
    @classmethod
    def validate_text_fields(cls, v: str) -> str:
        """Ensure text fields are substantive."""
        if not v or v.strip() == "":
            raise ValueError("Field cannot be empty")
        if len(v.strip()) < 5:
            raise ValueError("Field must be at least 5 characters")
        return v.strip()


class ProtocolStep(BaseModel):
    """
    A single step in an experimental protocol.

    Example:
        ```python
        step = ProtocolStep(
            step_number=1,
            title="Data Preparation",
            description="Load and preprocess the dataset...",
            action="Run preprocessing pipeline with sklearn.preprocessing",
            expected_duration_minutes=30,
            validation_check="Verify dataset shape is (N, 768)"
        )
        ```
    """
    step_number: int = Field(..., ge=1, description="Step order in protocol")
    title: str = Field(..., min_length=3, description="Step title")
    description: str = Field(..., min_length=10, description="Detailed step description")

    # Specific action to take
    action: str = Field(..., description="Concrete action or code to execute")

    # Dependencies
    requires_steps: List[int] = Field(default_factory=list, description="Step numbers that must complete first")
    requires_resources: List[str] = Field(default_factory=list, description="Required resources (data, compute, etc.)")

    # Expected outcomes
    expected_output: Optional[str] = None
    validation_check: Optional[str] = None

    # Time estimates
    expected_duration_minutes: Optional[float] = Field(None, ge=0)

    # Code generation hints (for Phase 5)
    code_template: Optional[str] = None
    library_imports: List[str] = Field(default_factory=list)

    @field_validator('title', mode='before')
    @classmethod
    def ensure_title(cls, v):
        """Provide default title when LLM returns empty string."""
        if not v or (isinstance(v, str) and len(v.strip()) < 3):
            return "Untitled Step"
        return v

    @field_validator('title', 'description', 'action')
    @classmethod
    def validate_text_fields(cls, v: str) -> str:
        """Ensure text fields are clear."""
        if not v or v.strip() == "":
            raise ValueError("Field cannot be empty")
        return v.strip()


class ResourceRequirements(BaseModel):
    """
    Resource requirements for an experiment.

    Example:
        ```python
        resources = ResourceRequirements(
            compute_hours=24.0,
            memory_gb=16,
            gpu_required=True,
            estimated_cost_usd=5.50,
            estimated_duration_days=2
        )
        ```
    """
    # Compute resources
    compute_hours: Optional[float] = Field(None, ge=0, description="Estimated CPU/GPU hours")
    memory_gb: Optional[float] = Field(None, ge=0, description="Peak memory requirement")
    gpu_required: bool = Field(default=False)
    gpu_memory_gb: Optional[float] = Field(None, ge=0)

    # Cost estimates
    estimated_cost_usd: Optional[float] = Field(None, ge=0, description="Total estimated cost")
    api_calls_estimated: Optional[int] = Field(None, ge=0)

    # Time estimates
    estimated_duration_days: Optional[float] = Field(None, ge=0)

    # Data requirements
    required_data_sources: List[str] = Field(default_factory=list)
    required_datasets: List[str] = Field(default_factory=list)
    data_size_gb: Optional[float] = Field(None, ge=0)

    # Dependencies
    required_libraries: List[str] = Field(default_factory=list)
    python_version: Optional[str] = None

    # Optimization suggestions
    can_parallelize: bool = Field(default=False)
    parallelization_factor: Optional[int] = Field(None, ge=1)


class StatisticalTestSpec(BaseModel):
    """
    Specification for a statistical test to perform.

    Example:
        ```python
        test = StatisticalTestSpec(
            test_type=StatisticalTest.T_TEST,
            description="Compare mean performance between groups",
            null_hypothesis="No difference in mean performance",
            alternative="two-sided",
            alpha=0.05,
            variables=["performance_score"]
        )
        ```
    """
    test_type: StatisticalTest
    description: str = Field(..., min_length=10)

    # Hypothesis testing
    null_hypothesis: str = Field(..., description="H0: statement")
    alternative: str = Field(default="two-sided", description="Alternative hypothesis type")
    alpha: float = Field(default=0.05, ge=0.0, le=1.0, description="Significance level")

    # Variables involved
    variables: List[str] = Field(..., description="Variable names to test")

    # Groups/conditions
    groups: Optional[List[str]] = None

    @field_validator('groups', mode='before')
    @classmethod
    def coerce_groups(cls, v):
        """Coerce comma-separated string groups from LLM output to list."""
        if v is None:
            return v
        if isinstance(v, str):
            return [g.strip() for g in v.split(',') if g.strip()]
        return v

    # Multiple testing correction
    correction_method: Optional[str] = None  # "bonferroni", "fdr", etc.

    # Power analysis
    required_power: float = Field(default=0.8, ge=0.0, le=1.0)
    expected_effect_size: Optional[float] = None

    @field_validator('expected_effect_size', mode='before')
    @classmethod
    def parse_effect_size(cls, v):
        """Parse effect size from string if needed (LLM may return text like 'Medium (Cohen's d = 0.5)')."""
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            # Try to extract a number from the string
            import re
            # Look for patterns like "0.5", "= 0.5", "d = 0.5"
            match = re.search(r'[-+]?\d*\.?\d+', v)
            if match:
                return float(match.group())
            # Return None if no number found
            return None
        return None


class ValidationCheck(BaseModel):
    """
    A validation check to ensure experimental rigor.

    Example:
        ```python
        check = ValidationCheck(
            check_type="control_group",
            description="Verify control group exists",
            severity="error",
            status="passed",
            message="Control group 'baseline_model' properly defined"
        )
        ```
    """
    check_type: str = Field(..., description="Type of validation (control_group, sample_size, bias, etc.)")
    description: str = Field(..., min_length=10)

    severity: str = Field(default="warning", description="error, warning, or info")
    status: str = Field(default="pending", description="passed, failed, or pending")

    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    recommendation: Optional[str] = None  # How to fix if failed


class ExperimentProtocol(BaseModel):
    """
    Complete experimental protocol.

    Specifies all details needed to execute an experiment to test a hypothesis.

    Example:
        ```python
        protocol = ExperimentProtocol(
            name="Attention Head Count Experiment",
            hypothesis_id="hyp_12345",
            experiment_type=ExperimentType.COMPUTATIONAL,
            description="Test whether increasing attention heads improves performance",
            steps=[...],
            variables={...},
            control_groups=[...],
            statistical_tests=[...],
            resource_requirements=ResourceRequirements(...),
            validation_checks=[...]
        )
        ```
    """
    id: Optional[str] = None
    name: str = Field(..., min_length=5, description="Experiment name")
    hypothesis_id: str = Field(..., description="ID of hypothesis being tested")

    experiment_type: ExperimentType
    domain: str = Field(..., description="Scientific domain")

    # Protocol details
    description: str = Field(..., min_length=20, description="Comprehensive experiment description")
    objective: str = Field(..., min_length=10, description="What this experiment aims to accomplish")

    # Experimental design
    steps: List[ProtocolStep] = Field(..., min_items=1, description="Ordered protocol steps")
    variables: Dict[str, Variable] = Field(..., description="All experiment variables")
    control_groups: List[ControlGroup] = Field(default_factory=list)

    # Statistical design
    statistical_tests: List[StatisticalTestSpec] = Field(default_factory=list)
    sample_size: Optional[int] = Field(None, ge=1, le=_MAX_SAMPLE_SIZE)
    sample_size_rationale: Optional[str] = None
    power_analysis_performed: bool = Field(default=False)

    @field_validator('sample_size', mode='before')
    @classmethod
    def coerce_protocol_sample_size(cls, v):
        """Coerce and bound sample_size for protocol."""
        if v is None:
            return v
        if isinstance(v, str):
            try:
                v = int(v)
            except (ValueError, TypeError):
                _experiment_logger.warning(
                    f"Could not coerce protocol sample_size '{v}' to int, setting to None"
                )
                return None
        if isinstance(v, (int, float)) and v > _MAX_SAMPLE_SIZE:
            _experiment_logger.warning(
                f"Protocol sample_size {v} exceeds maximum {_MAX_SAMPLE_SIZE}, clamping"
            )
            return _MAX_SAMPLE_SIZE
        return v

    # Resources & constraints
    resource_requirements: ResourceRequirements

    # Validation & rigor
    validation_checks: List[ValidationCheck] = Field(default_factory=list)
    rigor_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Scientific rigor score")

    # Reproducibility
    random_seed: Optional[int] = None
    reproducibility_notes: Optional[str] = None

    # Template information
    template_name: Optional[str] = None
    template_version: Optional[str] = None

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    generated_by: str = Field(default="experiment_designer")

    @field_validator('description')
    @classmethod
    def validate_description(cls, v: str) -> str:
        """Ensure description is comprehensive."""
        if not v or v.strip() == "":
            raise ValueError("Description cannot be empty")
        if len(v.strip()) < 20:
            raise ValueError("Description must be at least 20 characters for clarity")
        return v.strip()

    @field_validator('steps')
    @classmethod
    def validate_steps(cls, v: List[ProtocolStep]) -> List[ProtocolStep]:
        """Ensure steps are properly ordered."""
        if not v:
            raise ValueError("Protocol must have at least one step")

        # Check step numbers are sequential starting from 1
        expected_nums = set(range(1, len(v) + 1))
        actual_nums = set(step.step_number for step in v)

        if expected_nums != actual_nums:
            raise ValueError(f"Step numbers must be sequential 1-{len(v)}, got {sorted(actual_nums)}")

        return sorted(v, key=lambda s: s.step_number)

    def get_step(self, step_number: int) -> Optional[ProtocolStep]:
        """Get a specific step by number."""
        for step in self.steps:
            if step.step_number == step_number:
                return step
        return None

    def get_independent_variables(self) -> List[Variable]:
        """Get all independent variables."""
        return [v for v in self.variables.values() if v.type == VariableType.INDEPENDENT]

    def get_dependent_variables(self) -> List[Variable]:
        """Get all dependent variables."""
        return [v for v in self.variables.values() if v.type == VariableType.DEPENDENT]

    def has_control_group(self) -> bool:
        """Check if experiment has at least one control group."""
        return len(self.control_groups) > 0

    def total_duration_estimate_days(self) -> float:
        """Calculate total estimated duration from steps."""
        if self.resource_requirements.estimated_duration_days:
            return self.resource_requirements.estimated_duration_days

        # Fallback: sum step durations
        total_minutes = sum(
            step.expected_duration_minutes or 0
            for step in self.steps
        )
        return total_minutes / (24 * 60)  # Convert to days

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage or serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "hypothesis_id": self.hypothesis_id,
            "experiment_type": self.experiment_type.value,
            "domain": self.domain,
            "description": self.description,
            "objective": self.objective,
            "steps": [
                {
                    "step_number": s.step_number,
                    "title": s.title,
                    "description": s.description,
                    "action": s.action,
                    "requires_steps": s.requires_steps,
                    "requires_resources": s.requires_resources,
                    "expected_output": s.expected_output,
                    "validation_check": s.validation_check,
                    "expected_duration_minutes": s.expected_duration_minutes,
                    "code_template": s.code_template,
                    "library_imports": s.library_imports,
                }
                for s in self.steps
            ],
            "variables": {
                name: {
                    "name": v.name,
                    "type": v.type.value,
                    "description": v.description,
                    "values": v.values,
                    "fixed_value": v.fixed_value,
                    "unit": v.unit,
                    "measurement_method": v.measurement_method,
                    "column": v.column,
                }
                for name, v in self.variables.items()
            },
            "control_groups": [
                {
                    "name": cg.name,
                    "description": cg.description,
                    "variables": cg.variables,
                    "rationale": cg.rationale,
                    "sample_size": cg.sample_size,
                }
                for cg in self.control_groups
            ],
            "statistical_tests": [
                {
                    "test_type": st.test_type.value,
                    "description": st.description,
                    "null_hypothesis": st.null_hypothesis,
                    "alternative": st.alternative,
                    "alpha": st.alpha,
                    "variables": st.variables,
                    "groups": st.groups,
                    "correction_method": st.correction_method,
                    "required_power": st.required_power,
                    "expected_effect_size": st.expected_effect_size,
                }
                for st in self.statistical_tests
            ],
            "sample_size": self.sample_size,
            "sample_size_rationale": self.sample_size_rationale,
            "power_analysis_performed": self.power_analysis_performed,
            "resource_requirements": {
                "compute_hours": self.resource_requirements.compute_hours,
                "memory_gb": self.resource_requirements.memory_gb,
                "gpu_required": self.resource_requirements.gpu_required,
                "gpu_memory_gb": self.resource_requirements.gpu_memory_gb,
                "estimated_cost_usd": self.resource_requirements.estimated_cost_usd,
                "api_calls_estimated": self.resource_requirements.api_calls_estimated,
                "estimated_duration_days": self.resource_requirements.estimated_duration_days,
                "required_data_sources": self.resource_requirements.required_data_sources,
                "required_datasets": self.resource_requirements.required_datasets,
                "data_size_gb": self.resource_requirements.data_size_gb,
                "required_libraries": self.resource_requirements.required_libraries,
                "python_version": self.resource_requirements.python_version,
                "can_parallelize": self.resource_requirements.can_parallelize,
                "parallelization_factor": self.resource_requirements.parallelization_factor,
            },
            "validation_checks": [
                {
                    "check_type": vc.check_type,
                    "description": vc.description,
                    "severity": vc.severity,
                    "status": vc.status,
                    "message": vc.message,
                    "details": vc.details,
                    "recommendation": vc.recommendation,
                }
                for vc in self.validation_checks
            ],
            "rigor_score": self.rigor_score,
            "random_seed": self.random_seed,
            "reproducibility_notes": self.reproducibility_notes,
            "template_name": self.template_name,
            "template_version": self.template_version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "generated_by": self.generated_by,
        }

    model_config = ConfigDict(use_enum_values=False)


class ExperimentDesignRequest(BaseModel):
    """
    Request for experiment design.

    Example:
        ```python
        request = ExperimentDesignRequest(
            hypothesis_id="hyp_12345",
            preferred_experiment_type=ExperimentType.COMPUTATIONAL,
            max_cost_usd=100.0,
            max_duration_days=7
        )
        ```
    """
    hypothesis_id: str = Field(..., description="Hypothesis to design experiment for")

    # Optional preferences
    preferred_experiment_type: Optional[ExperimentType] = None
    domain: Optional[str] = None

    # Constraints
    max_cost_usd: Optional[float] = Field(None, ge=0)
    max_duration_days: Optional[float] = Field(None, ge=0)
    max_compute_hours: Optional[float] = Field(None, ge=0)

    # Design parameters
    require_control_group: bool = Field(default=True)
    require_power_analysis: bool = Field(default=True)
    min_rigor_score: float = Field(default=0.6, ge=0.0, le=1.0)

    # Template selection
    use_template: Optional[str] = None  # Specific template name
    allow_template_customization: bool = Field(default=True)

    # Additional context
    context: Optional[Dict[str, Any]] = None


class ExperimentDesignResponse(BaseModel):
    """
    Response from experiment design.

    Contains the generated experiment protocol and metadata.
    """
    protocol: ExperimentProtocol
    hypothesis_id: str

    # Design metadata
    design_time_seconds: float
    model_used: str = _DEFAULT_CLAUDE_SONNET_MODEL
    template_used: Optional[str] = None

    # Validation results
    validation_passed: bool
    validation_warnings: List[str] = Field(default_factory=list)
    validation_errors: List[str] = Field(default_factory=list)

    # Quality metrics
    rigor_score: float = Field(..., ge=0.0, le=1.0)
    completeness_score: float = Field(..., ge=0.0, le=1.0)

    # Resource summary
    estimated_cost_usd: Optional[float] = None
    estimated_duration_days: Optional[float] = None
    feasibility_assessment: str = Field(..., description="High/Medium/Low feasibility")

    # Recommendations
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def is_feasible(self, max_cost: Optional[float] = None, max_duration: Optional[float] = None) -> bool:
        """Check if experiment is feasible given constraints."""
        if max_cost and self.estimated_cost_usd and self.estimated_cost_usd > max_cost:
            return False
        if max_duration and self.estimated_duration_days and self.estimated_duration_days > max_duration:
            return False
        return self.validation_passed


class ValidationReport(BaseModel):
    """
    Scientific rigor validation report for an experiment protocol.

    Provides comprehensive assessment of experimental design quality.
    """
    protocol_id: str
    rigor_score: float = Field(..., ge=0.0, le=1.0, description="Overall scientific rigor score")

    # Validation checks performed
    checks_performed: List[ValidationCheck] = Field(default_factory=list)
    checks_passed: int = 0
    checks_failed: int = 0
    checks_warnings: int = 0

    # Specific assessments
    has_control_group: bool
    control_group_adequate: bool = False

    sample_size_adequate: bool = False
    sample_size: Optional[int] = None
    recommended_sample_size: Optional[int] = None

    power_analysis_performed: bool = False
    statistical_power: Optional[float] = None

    # Bias detection
    potential_biases: List[Dict[str, str]] = Field(default_factory=list)
    bias_mitigation_suggestions: List[str] = Field(default_factory=list)

    # Reproducibility
    is_reproducible: bool = False
    reproducibility_score: float = Field(..., ge=0.0, le=1.0)
    reproducibility_issues: List[str] = Field(default_factory=list)

    # Overall assessment
    validation_passed: bool
    severity_level: str = Field(..., description="critical, major, minor, or passed")

    summary: str = Field(..., description="Human-readable validation summary")
    recommendations: List[str] = Field(default_factory=list)

    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
