"""
Basic tests for Phase 4 functionality.

Tests core experiment design capabilities to verify the system works.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from kosmos.models.hypothesis import Hypothesis, ExperimentType
from kosmos.models.experiment import (
    ExperimentProtocol,
    ProtocolStep,
    Variable,
    VariableType,
    ResourceRequirements,
    ControlGroup,
    StatisticalTest,
    StatisticalTestSpec,
)
from kosmos.experiments.templates.base import TemplateBase, TemplateCustomizationParams, TemplateRegistry
from kosmos.experiments.templates.data_analysis import TTestComparisonTemplate
from kosmos.experiments.resource_estimator import ResourceEstimator, ComplexityLevel
from kosmos.experiments.statistical_power import PowerAnalyzer
from kosmos.experiments.validator import ExperimentValidator


class TestExperimentModels:
    """Test experiment data models."""

    def test_protocol_creation(self):
        """Test creating an experiment protocol."""
        protocol = ExperimentProtocol(
            name="Test Experiment",
            hypothesis_id="hyp_123",
            experiment_type=ExperimentType.DATA_ANALYSIS,
            domain="test_domain",
            description="This is a test experiment protocol with sufficient description length",
            objective="Test objective",
            steps=[
                ProtocolStep(
                    step_number=1,
                    title="Step 1",
                    description="Description of step 1 with adequate detail",
                    action="Perform action 1"
                )
            ],
            variables={
                "var1": Variable(
                    name="var1",
                    type=VariableType.INDEPENDENT,
                    description="Independent variable description"
                )
            },
            resource_requirements=ResourceRequirements(
                compute_hours=2.0,
                estimated_cost_usd=5.0,
                estimated_duration_days=1.0
            )
        )

        assert protocol.name == "Test Experiment"
        assert protocol.experiment_type == ExperimentType.DATA_ANALYSIS
        assert len(protocol.steps) == 1
        assert len(protocol.variables) == 1

    def test_protocol_validation(self):
        """Test protocol data validation."""
        # Should raise error for short description
        with pytest.raises(ValueError):
            ExperimentProtocol(
                name="Test",
                hypothesis_id="hyp_123",
                experiment_type=ExperimentType.DATA_ANALYSIS,
                domain="test",
                description="Too short",  # Less than 50 chars
                objective="Test",
                steps=[],
                variables={},
                resource_requirements=ResourceRequirements()
            )


class TestTemplateSystem:
    """Test template system."""

    def test_template_registry(self):
        """Test template registration and lookup."""
        # auto_discover=False: start empty instead of loading the built-in templates
        registry = TemplateRegistry(auto_discover=False)

        class TestTemplate(TemplateBase):
            def __init__(self):
                super().__init__(
                    name="test_template",
                    experiment_type=ExperimentType.DATA_ANALYSIS,
                    title="Test Template",
                    # TemplateMetadata.description requires >= 50 characters
                    description="Minimal data-analysis template used to exercise the template registry",
                )

            def is_applicable(self, hypothesis):
                return True

            def generate_protocol(self, params):
                return ExperimentProtocol(
                    name="Test",
                    hypothesis_id="hyp_123",
                    experiment_type=ExperimentType.DATA_ANALYSIS,
                    domain="test",
                    description="Test protocol from template with sufficient length for validation",
                    objective="Test objective",
                    steps=[ProtocolStep(step_number=1, title="Test", description="Test step description", action="Test")],
                    variables={},
                    resource_requirements=ResourceRequirements()
                )

        template = TestTemplate()
        registry.register(template)

        assert len(registry) == 1
        assert "test_template" in registry
        assert registry.get_template("test_template") is template

    def test_t_test_template(self):
        """Test T-Test template."""
        template = TTestComparisonTemplate()

        hypothesis = Hypothesis(
            statement="Group A scores will be higher than Group B scores",
            rationale="Based on prior research, we expect Group A to outperform Group B",
            domain="psychology",
            research_question="Do groups differ?",
            suggested_experiment_types=[ExperimentType.DATA_ANALYSIS]
        )

        assert template.is_applicable(hypothesis)

        params = TemplateCustomizationParams(hypothesis=hypothesis)
        protocol = template.generate_protocol(params)

        assert protocol.name is not None
        assert protocol.experiment_type == ExperimentType.DATA_ANALYSIS
        assert len(protocol.steps) >= 3  # Should have multiple steps
        assert len(protocol.control_groups) > 0  # Should have control group


class TestResourceEstimator:
    """Test resource estimation."""

    def test_basic_estimation(self):
        """Test basic resource estimation."""
        estimator = ResourceEstimator()

        resources = estimator.estimate(
            experiment_type=ExperimentType.DATA_ANALYSIS,
            num_steps=5,
            sample_size=100,
            complexity=ComplexityLevel.MODERATE
        )

        assert resources.compute_hours is not None
        assert resources.compute_hours > 0
        assert resources.estimated_cost_usd is not None
        assert resources.estimated_cost_usd > 0
        assert resources.estimated_duration_days is not None
        assert resources.estimated_duration_days > 0

    def test_complexity_affects_estimates(self):
        """Test that complexity affects estimates."""
        estimator = ResourceEstimator()

        simple = estimator.estimate(
            experiment_type=ExperimentType.COMPUTATIONAL,
            complexity=ComplexityLevel.SIMPLE
        )

        complex = estimator.estimate(
            experiment_type=ExperimentType.COMPUTATIONAL,
            complexity=ComplexityLevel.VERY_COMPLEX
        )

        assert complex.compute_hours > simple.compute_hours
        assert complex.estimated_cost_usd > simple.estimated_cost_usd

    def test_availability_checking(self):
        """Test resource availability checking."""
        estimator = ResourceEstimator()

        resources = ResourceRequirements(
            estimated_cost_usd=100.0,
            estimated_duration_days=5.0
        )

        # Should be available
        result = estimator.check_availability(
            resources,
            available_budget=200.0,
            available_time=10.0
        )
        assert result["available"] is True

        # Should not be available (exceeds budget)
        result = estimator.check_availability(
            resources,
            available_budget=50.0,
            available_time=10.0
        )
        assert result["available"] is False


class TestStatisticalPower:
    """Test statistical power analysis."""

    def test_ttest_sample_size(self):
        """Test t-test sample size calculation."""
        analyzer = PowerAnalyzer()

        n = analyzer.ttest_sample_size(
            effect_size=0.5,  # Medium effect
            power=0.8,
            alpha=0.05
        )

        assert n > 0
        assert isinstance(n, int)
        # Medium effect with power=0.8 should need ~60-70 per group
        assert 50 < n < 80

    def test_effect_size_interpretation(self):
        """Test effect size interpretation."""
        analyzer = PowerAnalyzer()

        assert analyzer.interpret_effect_size(0.2, "t_test") == "small"
        assert analyzer.interpret_effect_size(0.5, "t_test") == "medium"
        assert analyzer.interpret_effect_size(0.9, "t_test") == "large"

    def test_power_report_generation(self):
        """Test power analysis report generation."""
        analyzer = PowerAnalyzer()

        report = analyzer.generate_power_report(
            test_type="t_test",
            effect_size=0.5,
            power=0.8,
            alpha=0.05
        )

        assert "test_type" in report
        assert "effect_size" in report
        assert "required_sample_size_per_group" in report
        assert "recommendation" in report


class TestExperimentValidator:
    """Test experiment validation."""

    def test_valid_protocol(self):
        """Test validation of a valid protocol."""
        validator = ExperimentValidator(
            require_control_group=True,
            min_sample_size=20
        )

        protocol = ExperimentProtocol(
            name="Valid Experiment",
            hypothesis_id="hyp_123",
            experiment_type=ExperimentType.DATA_ANALYSIS,
            domain="test",
            description="A comprehensive experiment protocol with detailed description that meets requirements",
            objective="Test validation",
            steps=[
                ProtocolStep(step_number=i, title=f"Step {i}", description=f"Description for step {i}", action=f"Action {i}")
                for i in range(1, 6)
            ],
            variables={
                "iv": Variable(name="iv", type=VariableType.INDEPENDENT, description="Independent var"),
                "dv": Variable(name="dv", type=VariableType.DEPENDENT, description="Dependent var"),
            },
            control_groups=[
                ControlGroup(
                    name="control",
                    description="Control group description",
                    variables={"iv": "baseline"},
                    rationale="Standard control group for comparison with experimental condition"
                )
            ],
            statistical_tests=[
                StatisticalTestSpec(
                    test_type=StatisticalTest.T_TEST,
                    description="Independent samples t-test of dv between groups",
                    null_hypothesis="H0: mean dv is equal across groups",
                    variables=["iv", "dv"],
                )
            ],
            sample_size=60,
            power_analysis_performed=True,
            random_seed=42,
            reproducibility_notes="Complete reproducibility documentation",
            # Completeness checks fail without cost and duration estimates
            resource_requirements=ResourceRequirements(
                compute_hours=1.0,
                estimated_cost_usd=5.0,
                estimated_duration_days=1.0,
            )
        )

        report = validator.validate(protocol)

        assert report.validation_passed is True
        assert report.rigor_score > 0.5
        assert report.has_control_group is True
        assert report.sample_size_adequate is True

    def test_missing_control_group(self):
        """Test validation catches missing control group."""
        validator = ExperimentValidator(require_control_group=True)

        protocol = ExperimentProtocol(
            name="No Control",
            hypothesis_id="hyp_123",
            experiment_type=ExperimentType.DATA_ANALYSIS,
            domain="test",
            description="Experiment without control group that has enough text for validation",
            objective="Test the validator",
            steps=[ProtocolStep(step_number=1, title="Step", description="Description", action="Action")],
            variables={
                "iv": Variable(name="iv", type=VariableType.INDEPENDENT, description="Independent var"),
            },
            control_groups=[],  # No control group
            resource_requirements=ResourceRequirements()
        )

        report = validator.validate(protocol)

        assert report.validation_passed is False
        assert report.has_control_group is False

    def test_small_sample_size(self):
        """Test validation catches small sample size."""
        validator = ExperimentValidator(min_sample_size=30)

        protocol = ExperimentProtocol(
            name="Small Sample",
            hypothesis_id="hyp_123",
            experiment_type=ExperimentType.DATA_ANALYSIS,
            domain="test",
            description="Experiment with small sample size for validation testing purposes",
            objective="Test the validator",
            steps=[ProtocolStep(step_number=1, title="Step", description="Description", action="Action")],
            variables={},
            sample_size=10,  # Too small
            resource_requirements=ResourceRequirements()
        )

        report = validator.validate(protocol)

        assert report.validation_passed is False
        assert report.sample_size_adequate is False

    def test_rigor_score_calculation(self):
        """Test rigor score reflects protocol quality."""
        validator = ExperimentValidator()

        # High quality protocol
        good_protocol = ExperimentProtocol(
            name="High Quality",
            hypothesis_id="hyp_123",
            experiment_type=ExperimentType.DATA_ANALYSIS,
            domain="test",
            description="Comprehensive high-quality protocol with detailed experimental procedures",
            objective="Test rigor scoring",
            steps=[ProtocolStep(step_number=i, title=f"Step {i}", description=f"Description {i}", action=f"Act {i}") for i in range(1, 6)],
            variables={
                "iv": Variable(name="iv", type=VariableType.INDEPENDENT, description="Independent var"),
                "dv": Variable(name="dv", type=VariableType.DEPENDENT, description="Dependent var"),
            },
            control_groups=[ControlGroup(name="ctrl", description="Control", variables={}, rationale="Standard control comparison")],
            sample_size=100,
            power_analysis_performed=True,
            random_seed=42,
            reproducibility_notes="Full documentation",
            resource_requirements=ResourceRequirements()
        )

        # Low quality protocol
        poor_protocol = ExperimentProtocol(
            name="Low Quality",
            hypothesis_id="hyp_123",
            experiment_type=ExperimentType.DATA_ANALYSIS,
            domain="test",
            description="Basic protocol with minimal details for testing validation",
            objective="Test rigor scoring",
            steps=[ProtocolStep(step_number=1, title="Step", description="Description", action="Act")],
            variables={},
            control_groups=[],
            sample_size=10,
            resource_requirements=ResourceRequirements()
        )

        good_report = validator.validate(good_protocol)
        poor_report = validator.validate(poor_protocol)

        assert good_report.rigor_score > poor_report.rigor_score
