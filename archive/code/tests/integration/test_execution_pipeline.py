"""
Integration tests for complete execution pipeline.

Tests end-to-end workflow: Protocol → Code Generation → Execution → Result Collection.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from kosmos.models.experiment import ExperimentProtocol, ExperimentType, Variable, VariableType, ProtocolStep, ResourceRequirements, StatisticalTestSpec
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
        assert "ttest_ind" in code

        # Step 2: Execute code
        executor = CodeExecutor(max_retries=1, use_sandbox=False)
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


# Template-Based Generation Tests


# Error Recovery Tests


# Data Flow Tests

class TestDataFlow:
    """Tests for data flow through pipeline."""

    def test_results_preserved_through_collection(self, ttest_protocol, sample_data_file):
        """Test results are preserved during collection."""

        # Generate and execute
        generator = ExperimentCodeGenerator(use_templates=True, use_llm=False)
        code = generator.generate(ttest_protocol)
        execution_result = execute_protocol_code(code, sample_data_file, use_sandbox=False)

        # Collect
        collector = ResultCollector(store_in_db=False)
        result = collector.collect(ttest_protocol, execution_result)

        # Verify data preserved
        assert result.raw_data is not None


# Statistical Analysis Pipeline Tests


# Performance Tests


# Sandbox Integration Tests (Mocked)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
