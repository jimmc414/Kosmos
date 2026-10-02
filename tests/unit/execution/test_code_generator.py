"""
Tests for code generation system.

Tests template matching, code generation, LLM fallback, and validation.
"""

import ast
import contextlib
import io

import pytest
from unittest.mock import Mock, patch, MagicMock
from kosmos.execution.code_generator import (
    ExperimentCodeGenerator,
    TTestComparisonCodeTemplate,
    CorrelationAnalysisCodeTemplate,
    LogLogScalingCodeTemplate,
    MLExperimentCodeTemplate,
    CodeTemplate
)
from kosmos.models.experiment import (
    ExperimentProtocol,
    ExperimentType,
    Variable,
    VariableType,
    StatisticalTestSpec,
    StatisticalTest,
    ProtocolStep,
    ResourceRequirements,
)


# Fixtures

@pytest.fixture
def ttest_protocol():
    """Create T-test experiment protocol."""
    return ExperimentProtocol(
        id="test-001",
        name="T-Test Experiment Protocol",
        hypothesis_id="hyp-001",
        domain="statistics",
        description="T-test comparison experiment for statistical analysis of treatment vs control groups",
        objective="Compare means between two groups using T-test",
        experiment_type=ExperimentType.DATA_ANALYSIS,
        statistical_tests=[
            StatisticalTestSpec(
                test_type=StatisticalTest.T_TEST,
                description="Two-sample T-test for group comparison",
                null_hypothesis="No difference between group means",
                variables=["group", "measurement"],
            )
        ],
        steps=[
            ProtocolStep(
                step_number=1,
                title="Execute T-test",
                description="Load data and run T-test analysis",
                action="run_ttest",
                expected_duration_minutes=5
            )
        ],
        variables={
            "group": Variable(name="group", type=VariableType.INDEPENDENT, description="Group variable"),
            "measurement": Variable(name="measurement", type=VariableType.DEPENDENT, description="Measurement")
        },
        resource_requirements=ResourceRequirements(
            estimated_runtime_seconds=300,
            cpu_cores=1,
            memory_gb=1,
            storage_gb=0.1
        ),
        data_requirements={"format": "csv", "columns": ["group", "measurement"]},
        expected_duration_minutes=10
    )


@pytest.fixture
def generic_protocol():
    """Create a COMPUTATIONAL protocol that only the generic template matches.

    Variable keys and the name contain spaces and an apostrophe, the shapes
    LLM-designed protocols produce, to exercise syntax safety.
    """
    return ExperimentProtocol(
        id="test-generic",
        name="CO2 vs temperature's trend",
        hypothesis_id="hyp-generic",
        domain="climate",
        description="Computational analysis of the relationship between two measured quantities",
        objective="Test whether CO2 concentration tracks the temperature anomaly",
        experiment_type=ExperimentType.COMPUTATIONAL,
        statistical_tests=[
            StatisticalTestSpec(
                test_type=StatisticalTest.T_TEST,
                description="Two-sample T-test for group comparison",
                null_hypothesis="No difference between group means",
                variables=["CO2 concentration", "temp anomaly"],
            )
        ],
        steps=[
            ProtocolStep(
                step_number=1,
                title="Execute analysis",
                description="Load data and run the analysis",
                action="run_analysis",
                expected_duration_minutes=5
            )
        ],
        variables={
            "CO2 concentration": Variable(name="CO2 concentration", type=VariableType.INDEPENDENT, description="Atmospheric CO2 concentration in ppm"),
            "temp anomaly": Variable(name="temp anomaly", type=VariableType.DEPENDENT, description="Temperature anomaly")
        },
        resource_requirements=ResourceRequirements(
            estimated_runtime_seconds=300,
            cpu_cores=1,
            memory_gb=1,
            storage_gb=0.1
        ),
        data_requirements={"format": "csv"},
        expected_duration_minutes=10
    )


@pytest.fixture
def correlation_protocol():
    """Create correlation analysis protocol."""
    return ExperimentProtocol(
        id="test-002",
        name="Correlation Analysis Protocol",
        hypothesis_id="hyp-002",
        domain="statistics",
        description="Correlation analysis experiment to test linear relationships between variables",
        objective="Calculate Pearson correlation between X and Y variables",
        experiment_type=ExperimentType.DATA_ANALYSIS,
        statistical_tests=[
            StatisticalTestSpec(
                test_type=StatisticalTest.CORRELATION,
                description="Pearson correlation analysis",
                null_hypothesis="No correlation between variables",
                variables=["x", "y"],
            )
        ],
        steps=[
            ProtocolStep(
                step_number=1,
                title="Run Correlation",
                description="Calculate correlation coefficient",
                action="run_correlation",
                expected_duration_minutes=5
            )
        ],
        variables={
            "x": Variable(name="x", type=VariableType.INDEPENDENT, description="X variable"),
            "y": Variable(name="y", type=VariableType.DEPENDENT, description="Y variable")
        },
        resource_requirements=ResourceRequirements(
            estimated_runtime_seconds=300,
            cpu_cores=1,
            memory_gb=1,
            storage_gb=0.1
        ),
        data_requirements={"format": "csv", "columns": ["x", "y"]},
        expected_duration_minutes=10
    )


@pytest.fixture
def loglog_protocol():
    """Create log-log scaling protocol."""
    return ExperimentProtocol(
        id="test-003",
        name="Log-Log Scaling Analysis Protocol",
        hypothesis_id="hyp-003",
        domain="statistics",
        description="Power law and log-log scaling analysis to identify scale-free relationships",
        objective="Detect power law scaling relationships in data",
        experiment_type=ExperimentType.DATA_ANALYSIS,
        statistical_tests=[],  # No statistical tests - matching is done via name/description
        steps=[
            ProtocolStep(
                step_number=1,
                title="Log-Log Analysis",
                description="Perform log-log scaling analysis",
                action="run_loglog",
                expected_duration_minutes=5
            )
        ],
        variables={
            "x": Variable(name="x", type=VariableType.INDEPENDENT, description="X variable input"),
            "y": Variable(name="y", type=VariableType.DEPENDENT, description="Y variable output")
        },
        resource_requirements=ResourceRequirements(
            estimated_runtime_seconds=300,
            cpu_cores=1,
            memory_gb=1,
            storage_gb=0.1
        ),
        data_requirements={"format": "csv", "columns": ["x", "y"]},
        expected_duration_minutes=10
    )


@pytest.fixture
def ml_protocol():
    """Create ML experiment protocol (using COMPUTATIONAL type for ML)."""
    return ExperimentProtocol(
        id="test-004",
        name="Machine Learning Classification Protocol",
        hypothesis_id="hyp-004",
        domain="machine_learning",
        description="Machine learning classification experiment to train and evaluate predictive models",
        objective="Train and evaluate ML model for classification task",
        experiment_type=ExperimentType.COMPUTATIONAL,  # ML uses COMPUTATIONAL type
        statistical_tests=[],  # ML doesn't use traditional statistical tests
        steps=[
            ProtocolStep(
                step_number=1,
                title="Train ML Model",
                description="Train and evaluate classification model",
                action="train_model",
                expected_duration_minutes=15
            )
        ],
        variables={
            "features": Variable(name="features", type=VariableType.INDEPENDENT, description="Input features for ML model training"),
            "target": Variable(name="target", type=VariableType.DEPENDENT, description="Target variable for prediction")
        },
        resource_requirements=ResourceRequirements(
            estimated_runtime_seconds=1800,
            cpu_cores=2,
            memory_gb=4,
            storage_gb=1
        ),
        data_requirements={"format": "csv"},
        expected_duration_minutes=30
    )


def make_valid_protocol(
    id: str = "test-001",
    hypothesis_id: str = "hyp-001",
    name: str = "Test Experiment Protocol",
    domain: str = "statistics",
    description: str = "Test experiment for validating code generation functionality",
    objective: str = "Validate code generation",
    experiment_type: ExperimentType = ExperimentType.DATA_ANALYSIS,
    statistical_tests: list = None,
    variables: dict = None,
    steps: list = None,
    data_requirements: dict = None,
    expected_duration_minutes: int = 5,
) -> ExperimentProtocol:
    """Helper to create valid ExperimentProtocol with all required fields."""
    if statistical_tests is None:
        statistical_tests = []
    if variables is None:
        variables = {}
    if steps is None:
        steps = [
            ProtocolStep(
                step_number=1,
                title="Execute Analysis",
                description="Run the experiment analysis",
                action="run_analysis",
                expected_duration_minutes=5
            )
        ]
    if data_requirements is None:
        data_requirements = {}

    return ExperimentProtocol(
        id=id,
        name=name,
        hypothesis_id=hypothesis_id,
        domain=domain,
        description=description,
        objective=objective,
        experiment_type=experiment_type,
        statistical_tests=statistical_tests,
        steps=steps,
        variables=variables,
        resource_requirements=ResourceRequirements(
            estimated_runtime_seconds=300,
            cpu_cores=1,
            memory_gb=1,
            storage_gb=0.1
        ),
        data_requirements=data_requirements,
        expected_duration_minutes=expected_duration_minutes,
    )


@pytest.fixture
def code_generator():
    """Create code generator without LLM."""
    return ExperimentCodeGenerator(use_templates=True, use_llm=False)


@pytest.fixture
def code_generator_with_llm():
    """Create code generator with LLM."""
    mock_llm = Mock()
    mock_llm.generate.return_value = "import numpy as np\nresults = {'value': 42}"
    return ExperimentCodeGenerator(use_templates=True, use_llm=True, llm_client=mock_llm)


# Template Matching Tests

class TestTemplateMatching:
    """Tests for template matching logic."""

    def test_ttest_template_matches_ttest_protocol(self, ttest_protocol):
        """Test T-test template matches T-test protocol."""
        template = TTestComparisonCodeTemplate()
        assert template.matches(ttest_protocol)

    def test_correlation_template_matches_correlation_protocol(self, correlation_protocol):
        """Test correlation template matches correlation protocol."""
        template = CorrelationAnalysisCodeTemplate()
        assert template.matches(correlation_protocol)

    def test_loglog_template_matches_scaling_protocol(self, loglog_protocol):
        """Test log-log template matches scaling protocol."""
        template = LogLogScalingCodeTemplate()
        assert template.matches(loglog_protocol)

    def test_ml_template_matches_ml_protocol(self, ml_protocol):
        """Test ML template matches ML protocol."""
        template = MLExperimentCodeTemplate()
        assert template.matches(ml_protocol)

    def test_ttest_template_does_not_match_correlation(self, correlation_protocol):
        """Test T-test template doesn't match correlation protocol."""
        template = TTestComparisonCodeTemplate()
        assert not template.matches(correlation_protocol)

    def test_generator_selects_correct_template_for_ttest(self, code_generator, ttest_protocol):
        """Test generator selects T-test template."""
        template = code_generator._match_template(ttest_protocol)
        assert isinstance(template, TTestComparisonCodeTemplate)

    def test_generator_selects_correct_template_for_correlation(self, code_generator, correlation_protocol):
        """Test generator selects correlation template."""
        template = code_generator._match_template(correlation_protocol)
        assert isinstance(template, CorrelationAnalysisCodeTemplate)


# Code Generation Tests

class TestCodeGeneration:
    """Tests for code generation from templates."""

    def test_ttest_code_generation(self, code_generator, ttest_protocol):
        """Test T-test code generation."""
        code = code_generator.generate(ttest_protocol)

        assert code is not None
        assert "import pandas as pd" in code
        assert "DataAnalyzer" in code
        assert "ttest_comparison" in code
        assert "results" in code

    def test_correlation_code_generation(self, code_generator, correlation_protocol):
        """Test correlation code generation."""
        code = code_generator.generate(correlation_protocol)

        assert code is not None
        assert "import pandas as pd" in code
        assert "DataAnalyzer" in code
        assert "correlation_analysis" in code

    def test_loglog_code_generation(self, code_generator, loglog_protocol):
        """Test log-log scaling code generation."""
        code = code_generator.generate(loglog_protocol)

        assert code is not None
        assert "import pandas as pd" in code
        assert "DataAnalyzer" in code
        assert "log_log_scaling_analysis" in code

    def test_ml_code_generation(self, code_generator, ml_protocol):
        """Test ML code generation."""
        code = code_generator.generate(ml_protocol)

        assert code is not None
        assert "import pandas as pd" in code
        assert "MLAnalyzer" in code
        assert "run_experiment" in code or "cross_validate" in code

    def test_generated_code_is_valid_python(self, code_generator, ttest_protocol):
        """Test generated code is valid Python syntax."""
        import ast

        code = code_generator.generate(ttest_protocol)

        try:
            ast.parse(code)
            syntax_valid = True
        except SyntaxError:
            syntax_valid = False

        assert syntax_valid, f"Generated code has syntax errors:\n{code}"

    def test_generated_code_contains_result_variable(self, code_generator, ttest_protocol):
        """Test generated code assigns to results variable."""
        code = code_generator.generate(ttest_protocol)
        assert "results" in code or "result" in code


# LLM Fallback Tests

class TestGenericTemplate:
    """The generic template must run without kosmos and survive LLM-chosen names (P0-1)."""

    def test_generic_code_is_self_contained_and_parses(self, code_generator, generic_protocol):
        code = code_generator.generate(generic_protocol)
        assert "kosmos" not in code
        ast.parse(code)

    def test_generic_code_reads_the_data_file(self, code_generator, generic_protocol, tmp_path):
        code = code_generator.generate(generic_protocol)
        csv = tmp_path / "data.csv"
        rows = ["year,co2_ppm"] + [f"{2000 + i},{370 + 2.1 * i + (i % 3) * 0.4}" for i in range(10)]
        csv.write_text("\n".join(rows) + "\n")

        ns = {"data_path": str(csv)}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, ns)

        assert ns["results"]["data_source"] == "file"
        assert isinstance(ns["results"]["p_value"], float)
        assert ns["results"]["n_samples"] == 10

    def test_llm_prompt_forbids_kosmos_imports(self, code_generator, generic_protocol):
        prompt = code_generator._create_code_generation_prompt(generic_protocol)
        assert "DataAnalyzer" not in prompt
        assert "kosmos.execution" not in prompt
        assert "Do NOT import kosmos" in prompt
        assert "top-level variable named results" in prompt


class TestLLMFallback:
    """Tests for LLM-based code generation fallback."""

    def test_llm_used_when_no_template_matches(self, code_generator_with_llm):
        """Test LLM used when no template matches."""
        # Create custom protocol that doesn't match any template
        # Use LITERATURE_SYNTHESIS which has no template
        custom_protocol = make_valid_protocol(
            id="custom-001",
            name="Custom Experiment Protocol",
            description="Novel experiment type that doesn't match any standard template",
            experiment_type=ExperimentType.LITERATURE_SYNTHESIS,
        )

        code = code_generator_with_llm.generate(custom_protocol)

        # Should have called LLM
        assert code_generator_with_llm.llm_client.generate.called
        assert code is not None

    def test_template_preferred_over_llm_when_available(self, code_generator_with_llm, ttest_protocol):
        """Test template used instead of LLM when available."""
        code = code_generator_with_llm.generate(ttest_protocol)

        # Should use template, not LLM
        assert "ttest_comparison" in code
        # LLM might still be called if enhance mode is on, but template should be primary

    def test_llm_can_enhance_template_code(self):
        """Test LLM enhancement of template code."""
        mock_llm = Mock()
        mock_llm.generate.return_value = "# Enhanced\nimport pandas as pd\nresults = {}"

        generator = ExperimentCodeGenerator(
            use_templates=True,
            use_llm=True,
            llm_enhance_templates=True,
            llm_client=mock_llm
        )

        protocol = make_valid_protocol(
            description="Test experiment for LLM enhancement validation",
            statistical_tests=[
                StatisticalTestSpec(
                    test_type=StatisticalTest.T_TEST,
                    description="T-test for enhancement test",
                    null_hypothesis="No difference",
                    variables=["x"],
                )
            ],
        )

        code = generator.generate(protocol)

        # LLM should have been called for enhancement
        assert mock_llm.generate.called


# Validation Tests

class TestCodeValidation:
    """Tests for code validation and syntax checking."""

    def test_validate_syntax_valid_code(self, code_generator):
        """Test validation accepts valid code."""
        valid_code = "import numpy as np\nx = np.array([1, 2, 3])\nresults = {'mean': np.mean(x)}"

        try:
            code_generator._validate_syntax(valid_code)
            is_valid = True
        except Exception:
            is_valid = False

        assert is_valid

    def test_validate_syntax_invalid_code(self, code_generator):
        """Test validation rejects invalid code."""
        invalid_code = "import numpy as np\nx = [1, 2, 3\nresults = {'mean': x}"

        # _validate_syntax raises ValueError (which wraps SyntaxError message)
        with pytest.raises((SyntaxError, ValueError)):
            code_generator._validate_syntax(invalid_code)

    def test_generated_code_passes_validation(self, code_generator, ttest_protocol):
        """Test all generated code passes validation."""
        code = code_generator.generate(ttest_protocol)

        try:
            code_generator._validate_syntax(code)
            is_valid = True
        except Exception:
            is_valid = False

        assert is_valid


# Variable Extraction Tests

class TestVariableExtraction:
    """Tests for extracting variables from protocols."""

    def test_extract_dependent_variable(self, ttest_protocol):
        """Test extraction of dependent variable."""
        template = TTestComparisonCodeTemplate()

        dependent_vars = [
            var for var in ttest_protocol.variables.values()
            if var.type == VariableType.DEPENDENT
        ]

        assert len(dependent_vars) > 0
        assert dependent_vars[0].name == "measurement"

    def test_extract_independent_variable(self, ttest_protocol):
        """Test extraction of independent variable."""
        template = TTestComparisonCodeTemplate()

        independent_vars = [
            var for var in ttest_protocol.variables.values()
            if var.type == VariableType.INDEPENDENT
        ]

        assert len(independent_vars) > 0
        assert independent_vars[0].name == "group"


# Integration Tests

class TestCodeGeneratorIntegration:
    """Integration tests for code generator."""

    def test_end_to_end_ttest_generation(self, code_generator, ttest_protocol):
        """Test complete T-test code generation pipeline."""
        code = code_generator.generate(ttest_protocol)

        # Verify code structure
        assert "import" in code
        assert "DataAnalyzer" in code
        assert "ttest_comparison" in code
        assert "results" in code

        # Verify valid syntax
        import ast
        ast.parse(code)

    def test_end_to_end_ml_generation(self, code_generator, ml_protocol):
        """Test complete ML code generation pipeline."""
        code = code_generator.generate(ml_protocol)

        assert "import" in code
        assert "MLAnalyzer" in code
        assert "results" in code

        # Verify valid syntax
        import ast
        ast.parse(code)

    def test_generator_handles_minimal_protocol(self, code_generator):
        """Test generator handles minimal protocol gracefully."""
        minimal_protocol = make_valid_protocol(
            id="minimal-001",
            name="Minimal Protocol Test",
            description="Minimal protocol for testing graceful handling",
        )

        code = code_generator.generate(minimal_protocol)

        # Should generate fallback code
        assert code is not None
        assert len(code) > 0


# Edge Cases and Error Handling

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_generator_with_no_templates_and_no_llm(self):
        """Test generator behavior when both templates and LLM disabled."""
        generator = ExperimentCodeGenerator(use_templates=False, use_llm=False)

        protocol = make_valid_protocol(
            description="Test protocol for no-template no-LLM scenario",
        )

        code = generator.generate(protocol)

        # Should generate basic fallback
        assert code is not None
        assert "import" in code

    def test_generator_handles_empty_variables(self, code_generator):
        """Test generator handles protocol with no variables."""
        protocol = make_valid_protocol(
            description="Test protocol with no variables defined",
            experiment_type=ExperimentType.LITERATURE_SYNTHESIS,  # Use valid enum value
            variables={},  # Empty
        )

        code = code_generator.generate(protocol)
        assert code is not None

    def test_generator_handles_missing_data_requirements(self, code_generator):
        """Test generator handles missing data requirements."""
        protocol = make_valid_protocol(
            description="Test protocol with missing data requirements scenario",
            statistical_tests=[
                StatisticalTestSpec(
                    test_type=StatisticalTest.T_TEST,
                    description="T-test for missing data requirements test",
                    null_hypothesis="No difference",
                    variables=["x"],
                )
            ],
            data_requirements={},  # Empty
        )

        code = code_generator.generate(protocol)
        assert code is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
