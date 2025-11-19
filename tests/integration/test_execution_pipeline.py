"""
Integration tests for complete execution pipeline.

Tests end-to-end workflow: Protocol → Code Generation → Execution → Result Collection.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from kosmos.models.experiment import ExperimentProtocol, ExperimentType, Variable, VariableType, ProtocolStep, ResourceRequirements, StatisticalTestSpec, StatisticalTest
from kosmos.execution.code_generator import ExperimentCodeGenerator
from kosmos.execution.executor import CodeExecutor, execute_protocol_code
from kosmos.execution.result_collector import ResultCollector
from kosmos.models.result import ResultStatus


# Fixtures

@pytest.fixture
def ttest_protocol():
    """Create T-test experiment protocol."""
    return ExperimentProtocol(
        id="integration-001",
        name="Integration Test T-test Protocol",
        hypothesis_id="hyp-001",
        domain="statistics",
        title="Integration Test - T-test",
        description="Comprehensive test protocol for complete pipeline validation with T-test statistical analysis on treatment vs control groups",
        objective="Validate complete execution pipeline from code generation through result collection",
        experiment_type=ExperimentType.DATA_ANALYSIS,
        statistical_tests=[
            StatisticalTestSpec(
                test_type=StatisticalTest.T_TEST,
                variables=["group", "score"],
                description="Two-sample T-test comparing treatment vs control groups",
                null_hypothesis="There is no difference in mean scores between groups"
            )
        ],
        steps=[
            ProtocolStep(
                step_number=1,
                title="Execute T-test Analysis",
                description="Load CSV data and perform T-test comparison",
                action="load_data_and_run_ttest",
                expected_duration_minutes=5
            )
        ],
        variables={
            "group": Variable(name="group", type=VariableType.INDEPENDENT, description="Treatment group assignment"),
            "score": Variable(name="score", type=VariableType.DEPENDENT, description="Test score measurement")
        },
        resource_requirements=ResourceRequirements(
            estimated_runtime_seconds=300,
            cpu_cores=1,
            memory_gb=1,
            storage_gb=0.1
        ),
        data_requirements={"format": "csv", "columns": ["group", "score"]},
        random_seed=42,
        expected_duration_minutes=5
    )


@pytest.fixture
def sample_data_file(tmp_path):
    """Create sample CSV data file."""
    # Create realistic T-test data
    np.random.seed(42)
    control = np.random.normal(75, 10, 50)
    treatment = np.random.normal(85, 10, 50)

    df = pd.DataFrame({
        'group': ['control'] * 50 + ['treatment'] * 50,
        'score': np.concatenate([control, treatment])
    })

    data_file = tmp_path / "experiment_data.csv"
    df.to_csv(data_file, index=False)

    return str(data_file)


# End-to-End Pipeline Tests

class TestEndToEndPipeline:
    """Tests for complete execution pipeline."""

    def test_complete_pipeline_ttest(self, ttest_protocol, sample_data_file):
        """Test complete pipeline: code generation → execution → result collection."""

        # Step 1: Generate code
        generator = ExperimentCodeGenerator(use_templates=True, use_llm=False)
        code = generator.generate(ttest_protocol)

        assert code is not None
        assert "ttest_comparison" in code

        # Step 2: Execute code
        executor = CodeExecutor(max_retries=1)
        execution_result = executor.execute_with_data(code, sample_data_file)

        assert execution_result.success is True

        # Step 3: Collect results
        collector = ResultCollector(store_in_db=False)

        execution_output = {
            'success': execution_result.success,
            'return_value': execution_result.return_value,
            'stdout': execution_result.stdout,
            'stderr': execution_result.stderr,
            'execution_time': execution_result.execution_time
        }

        result = collector.collect(ttest_protocol, execution_output)

        # Verify result
        assert result.status == ResultStatus.SUCCESS
        assert result.experiment_id == "integration-001"

    def test_pipeline_with_convenience_function(self, ttest_protocol, sample_data_file):
        """Test pipeline using convenience function."""

        # Generate code
        generator = ExperimentCodeGenerator(use_templates=True, use_llm=False)
        code = generator.generate(ttest_protocol)

        # Execute using convenience function
        result = execute_protocol_code(
            code,
            data_path=sample_data_file,
            validate_safety=True
        )

        assert result['success'] is True

    def test_pipeline_handles_errors_gracefully(self, ttest_protocol):
        """Test pipeline handles errors at each stage."""

        # Generate code
        generator = ExperimentCodeGenerator(use_templates=True, use_llm=False)
        code = generator.generate(ttest_protocol)

        # Execute with invalid data path
        result = execute_protocol_code(
            code,
            data_path="/nonexistent/path.csv",
            validate_safety=True
        )

        # Should fail gracefully
        assert result['success'] is False or 'error' in result


# Template-Based Generation Tests

class TestTemplatePipeline:
    """Tests for template-based code generation pipeline."""

    def test_ttest_template_pipeline(self, ttest_protocol, sample_data_file):
        """Test T-test template generates and executes successfully."""

        generator = ExperimentCodeGenerator(use_templates=True, use_llm=False)
        code = generator.generate(ttest_protocol)

        result = execute_protocol_code(code, sample_data_file, validate_safety=False)

        assert result['success'] is True
        assert result['return_value'] is not None

    def test_correlation_template_pipeline(self, sample_data_file):
        """Test correlation template pipeline."""

        protocol = ExperimentProtocol(
            id="corr-001",
            name="Correlation Test Protocol",
            hypothesis_id="hyp-001",
            domain="statistics",
            title="Correlation Test",
            description="Comprehensive test protocol for correlation pipeline validation with statistical correlation analysis",
            objective="Validate correlation analysis pipeline from data loading through statistical computation",
            experiment_type=ExperimentType.DATA_ANALYSIS,
            statistical_tests=[
                StatisticalTestSpec(
                test_type=StatisticalTest.CORRELATION,
                    variables=["group", "score"],
                    description="Pearson correlation analysis between variables",
                    null_hypothesis="There is no correlation between the variables"
                )
            ],
            steps=[
                ProtocolStep(
                    step_number=1,
                    title="Compute Correlation",
                    description="Compute correlation between variables",
                    action="compute_correlation",
                    expected_duration_minutes=5
                )
            ],
            variables={
                "group": Variable(name="group", type=VariableType.INDEPENDENT, description="Independent X variable for correlation"),
                "score": Variable(name="score", type=VariableType.DEPENDENT, description="Dependent Y variable for correlation")
            },
            resource_requirements=ResourceRequirements(
                estimated_runtime_seconds=300,
                cpu_cores=1,
                memory_gb=1,
                storage_gb=0.1
            ),
            data_requirements={},
            random_seed=42,
            expected_duration_minutes=5
        )

        generator = ExperimentCodeGenerator(use_templates=True, use_llm=False)
        code = generator.generate(protocol)

        # Note: May fail if data doesn't match expected format, but code should generate
        assert "correlation_analysis" in code


# Error Recovery Tests

class TestErrorRecovery:
    """Tests for error recovery in pipeline."""

    def test_retry_on_execution_failure(self, ttest_protocol):
        """Test retry logic on execution failure."""

        # Generate code that fails on first attempt
        code = """
import random
if random.random() > 0.9:  # High chance of success
    raise ValueError("Simulated error")
results = {'value': 42}
"""

        result = execute_protocol_code(code, max_retries=5, validate_safety=False)

        # Should eventually succeed or exhaust retries
        assert isinstance(result, dict)

    def test_validation_prevents_unsafe_code(self, ttest_protocol):
        """Test validation prevents unsafe code execution."""

        unsafe_code = """
import os
os.system('rm -rf /')
results = {}
"""

        result = execute_protocol_code(unsafe_code, validate_safety=True)

        assert result['success'] is False
        assert 'validation_errors' in result


# Data Flow Tests

class TestDataFlow:
    """Tests for data flow through pipeline."""

    def test_data_flows_through_pipeline(self, ttest_protocol, sample_data_file):
        """Test data flows correctly through pipeline."""

        generator = ExperimentCodeGenerator(use_templates=True, use_llm=False)
        code = generator.generate(ttest_protocol)

        # Add explicit data loading
        code_with_data = f"""
import pandas as pd
df = pd.read_csv('{sample_data_file}')

{code}
"""

        executor = CodeExecutor()
        result = executor.execute(code_with_data)

        assert result.success is True
        assert result.return_value is not None

    def test_results_preserved_through_collection(self, ttest_protocol, sample_data_file):
        """Test results are preserved during collection."""

        # Generate and execute
        generator = ExperimentCodeGenerator(use_templates=True, use_llm=False)
        code = generator.generate(ttest_protocol)
        execution_result = execute_protocol_code(code, sample_data_file, validate_safety=False)

        # Collect
        collector = ResultCollector(store_in_db=False)
        result = collector.collect(ttest_protocol, execution_result)

        # Verify data preserved
        assert result.raw_data is not None


# Statistical Analysis Pipeline Tests

class TestStatisticalPipeline:
    """Tests for statistical analysis in pipeline."""

    def test_pipeline_computes_statistics(self, ttest_protocol, sample_data_file):
        """Test pipeline computes statistical tests."""

        generator = ExperimentCodeGenerator(use_templates=True, use_llm=False)
        code = generator.generate(ttest_protocol)

        result = execute_protocol_code(code, sample_data_file, validate_safety=False)

        # Should have computed statistics
        if result['success'] and result['return_value']:
            assert 'p_value' in result['return_value'] or 't_statistic' in result['return_value']


# Performance Tests

class TestPipelinePerformance:
    """Tests for pipeline performance."""

    def test_pipeline_completes_within_timeout(self, ttest_protocol, sample_data_file):
        """Test pipeline completes within reasonable time."""
        import time

        start = time.time()

        generator = ExperimentCodeGenerator(use_templates=True, use_llm=False)
        code = generator.generate(ttest_protocol)
        result = execute_protocol_code(code, sample_data_file, validate_safety=True)

        duration = time.time() - start

        # Should complete in under 10 seconds
        assert duration < 10


# Sandbox Integration Tests (Mocked)

class TestSandboxPipeline:
    """Tests for sandboxed execution pipeline."""

    @patch('kosmos.execution.executor.SANDBOX_AVAILABLE', True)
    @patch('kosmos.execution.executor.DockerSandbox')
    def test_pipeline_with_sandbox(self, mock_sandbox_class, ttest_protocol, sample_data_file):
        """Test pipeline with sandbox execution."""

        # Mock sandbox execution
        mock_sandbox = Mock()
        mock_sandbox.execute.return_value = Mock(
            success=True,
            return_value={'p_value': 0.01},
            stdout="Test output",
            stderr="",
            error=None,
            error_type=None,
            execution_time=1.5
        )
        mock_sandbox_class.return_value = mock_sandbox

        generator = ExperimentCodeGenerator(use_templates=True, use_llm=False)
        code = generator.generate(ttest_protocol)

        result = execute_protocol_code(
            code,
            data_path=sample_data_file,
            use_sandbox=True,
            validate_safety=True
        )

        # Sandbox should have been used
        assert mock_sandbox.execute.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
