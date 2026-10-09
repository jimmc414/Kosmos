"""
Integration tests for complete execution pipeline.

Tests end-to-end workflow: Protocol → Code Generation → Execution → Result Collection.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from kosmos.models.experiment import ExperimentProtocol, ExperimentType, Variable, VariableType, ProtocolStep, ResourceRequirements, StatisticalTest, StatisticalTestSpec
from kosmos.execution.code_generator import ExperimentCodeGenerator
from kosmos.execution.executor import CodeExecutor, execute_protocol_code
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

    # Labels match the t-test template's default groups ('control' and
    # 'experimental'); with any other label the template raises a ValueError.
    df = pd.DataFrame({
        'group': ['control'] * 50 + ['experimental'] * 50,
        'score': np.concatenate([control, treatment])
    })

    data_file = tmp_path / "experiment_data.csv"
    df.to_csv(data_file, index=False)

    return str(data_file)


# End-to-End Pipeline Tests

class TestEndToEndPipeline:
    """Tests for complete execution pipeline."""

    def test_pipeline_with_convenience_function(self, ttest_protocol, sample_data_file):
        """Test pipeline using convenience function."""

        # Generate code
        generator = ExperimentCodeGenerator(use_templates=True, use_llm=False)
        code = generator.generate(ttest_protocol)

        # Execute using convenience function (always validates safety)
        result = execute_protocol_code(
            code,
            data_path=sample_data_file,
            use_sandbox=False
        )

        assert result['success'] is True
        # The real file was analysed (not the synthetic fallback, and not a
        # retry wrapper that turned an exception into a 'failed' payload)
        assert result['data_source'] == 'file'
        assert result['return_value'].get('status') != 'failed'
        assert result['return_value']['n'] == 100

    def test_pipeline_handles_errors_gracefully(self, ttest_protocol):
        """Test pipeline handles errors at each stage."""

        # Generate code
        generator = ExperimentCodeGenerator(use_templates=True, use_llm=False)
        code = generator.generate(ttest_protocol)

        # Execute with invalid data path
        result = execute_protocol_code(
            code,
            data_path="/nonexistent/path.csv",
            use_sandbox=False
        )

        # Should fail gracefully: either a reported failure, or a run that
        # truthfully labels its data as the synthetic fallback (never 'file')
        assert isinstance(result, dict)
        if result['success']:
            assert result['data_source'] == 'synthetic'
        else:
            assert result['error']


# Template-Based Generation Tests

class TestTemplatePipeline:
    """Tests for template-based code generation pipeline."""

    def test_ttest_template_pipeline(self, ttest_protocol, sample_data_file):
        """Test T-test template generates and executes successfully."""

        generator = ExperimentCodeGenerator(use_templates=True, use_llm=False)
        code = generator.generate(ttest_protocol)

        result = execute_protocol_code(code, sample_data_file, use_sandbox=False)

        assert result['success'] is True
        assert result['return_value'] is not None
        assert result['return_value']['test'] == 'welch_t_test'
        assert result['return_value']['n_group1'] == 50
        assert result['return_value']['n_group2'] == 50

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
                    test_type="correlation",
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
        assert "pearsonr" in code


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

        result = execute_protocol_code(code, max_retries=5, use_sandbox=False)

        # Should eventually succeed or exhaust retries
        assert isinstance(result, dict)

    def test_validation_prevents_unsafe_code(self, ttest_protocol):
        """Test validation prevents unsafe code execution (always-on validation)."""

        unsafe_code = """
import os
os.system('rm -rf /')
results = {}
"""

        result = execute_protocol_code(unsafe_code, use_sandbox=False)

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

        executor = CodeExecutor(use_sandbox=False)
        result = executor.execute(code_with_data)

        assert result.success is True
        assert result.return_value is not None


# Statistical Analysis Pipeline Tests

class TestStatisticalPipeline:
    """Tests for statistical analysis in pipeline."""

    def test_pipeline_computes_statistics(self, ttest_protocol, sample_data_file):
        """Test pipeline computes statistical tests."""

        generator = ExperimentCodeGenerator(use_templates=True, use_llm=False)
        code = generator.generate(ttest_protocol)

        result = execute_protocol_code(code, sample_data_file, use_sandbox=False)

        # Should have computed statistics on the file data
        assert result['success'] is True
        stats_out = result['return_value']
        assert 'p_value' in stats_out and 't_statistic' in stats_out
        assert 0.0 <= stats_out['p_value'] <= 1.0
        # Seeded data: experimental mean 85 vs control 75, so a clear effect
        assert stats_out['p_value'] < 0.05
        assert stats_out['group1_mean'] > stats_out['group2_mean']


# Performance Tests

class TestPipelinePerformance:
    """Tests for pipeline performance."""

    def test_pipeline_completes_within_timeout(self, ttest_protocol, sample_data_file):
        """Test pipeline completes within reasonable time."""
        import time

        start = time.time()

        generator = ExperimentCodeGenerator(use_templates=True, use_llm=False)
        code = generator.generate(ttest_protocol)
        result = execute_protocol_code(code, sample_data_file, use_sandbox=False)

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
            use_sandbox=True
        )

        # Sandbox should have been used
        assert mock_sandbox.execute.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
