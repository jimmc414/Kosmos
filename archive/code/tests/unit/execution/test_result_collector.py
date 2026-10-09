"""
Tests for result collection and structuring.

Tests result extraction, metadata creation, statistical test parsing, and database storage.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from kosmos.execution.result_collector import ResultCollector
from kosmos.models.result import (
    ExperimentResult,
    ResultStatus,
    ExecutionMetadata,
    StatisticalTestResult,
    VariableResult
)
from kosmos.models.experiment import (
    ExperimentProtocol,
    ExperimentType,
    StatisticalTestSpec,
    StatisticalTest,
    ProtocolStep,
    ResourceRequirements,
)


# Fixtures

@pytest.fixture
def sample_protocol():
    """Create sample experiment protocol."""
    return ExperimentProtocol(
        id="exp-001",
        hypothesis_id="hyp-001",
        name="Test Experiment Protocol",
        domain="biology",
        title="Test Experiment",
        description="Test description for the sample protocol fixture",
        objective="Verify result collection functionality",
        experiment_type=ExperimentType.DATA_ANALYSIS,
        statistical_tests=[StatisticalTestSpec(
            test_type=StatisticalTest.T_TEST,
            description="Compare treatment and control groups using independent samples t-test",
            null_hypothesis="No difference between treatment and control group means",
            variables=["treatment", "control"]
        )],
        steps=[
            ProtocolStep(
                step_number=1,
                title="Load Data",
                description="Load treatment and control group data from CSV files",
                action="pd.read_csv('data.csv')",
                expected_duration_minutes=2
            ),
            ProtocolStep(
                step_number=2,
                title="Run Analysis",
                description="Perform statistical comparison between groups",
                action="scipy.stats.ttest_ind(treatment, control)",
                expected_duration_minutes=5
            ),
            ProtocolStep(
                step_number=3,
                title="Collect Results",
                description="Extract and format analysis results",
                action="return results dictionary",
                expected_duration_minutes=2
            )
        ],
        variables={},
        data_requirements={},
        resource_requirements=ResourceRequirements(
            cpu_cores=1,
            memory_gb=1.0,
            estimated_duration_minutes=10
        ),
        random_seed=42,
        expected_duration_minutes=10
    )


@pytest.fixture
def successful_execution_output():
    """Create successful execution output."""
    return {
        'success': True,
        'return_value': {'mean': 10.5, 'std': 2.3},
        'stdout': 'Analysis complete',
        'stderr': '',
        'execution_time': 2.5,
        'cpu_time': 2.3,
        'memory_peak_mb': 150.0
    }


@pytest.fixture
def failed_execution_output():
    """Create failed execution output."""
    return {
        'success': False,
        'error': 'Division by zero',
        'error_type': 'ZeroDivisionError',
        'stdout': '',
        'stderr': 'Traceback...',
        'execution_time': 0.5
    }


@pytest.fixture
def statistical_tests_output():
    """Create sample statistical test results."""
    return [
        {
            'test_type': 't-test',
            'test_name': 'Two-sample T-test',
            'statistic': 2.45,
            'p_value': 0.017,
            'effect_size': 0.65,
            'effect_size_type': "Cohen's d",
            'confidence_interval': {'lower': 0.2, 'upper': 1.5},
            'sample_size': 100,
            'degrees_of_freedom': 98,
            'is_primary': True
        },
        {
            'test_type': 'correlation',
            'test_name': 'Pearson Correlation',
            'statistic': 0.75,
            'p_value': 0.001,
            'sample_size': 100
        }
    ]


@pytest.fixture
def variable_data():
    """Create sample variable data."""
    return {
        'treatment': pd.Series([10.1, 10.5, 9.8, 10.2, 10.4] * 10),
        'control': pd.Series([8.5, 8.9, 8.7, 8.6, 8.8] * 10)
    }


@pytest.fixture
def result_collector():
    """Create result collector without database storage."""
    return ResultCollector(store_in_db=False)


# Result Collection Tests

class TestResultCollection:
    """Tests for result collection."""

    def test_collect_successful_result(self, result_collector, sample_protocol, successful_execution_output):
        """Test collecting successful result."""
        result = result_collector.collect(sample_protocol, successful_execution_output)

        assert isinstance(result, ExperimentResult)
        assert result.status == ResultStatus.SUCCESS
        assert result.experiment_id == "exp-001"
        assert result.hypothesis_id == "hyp-001"

    def test_collect_failed_result(self, result_collector, sample_protocol, failed_execution_output):
        """Test collecting failed result."""
        result = result_collector.collect(sample_protocol, failed_execution_output)

        assert result.status == ResultStatus.ERROR
        assert result.metadata.errors is not None

    def test_collect_with_statistical_tests(self, result_collector, sample_protocol, successful_execution_output,
                                          statistical_tests_output):
        """Test collecting result with statistical tests."""
        result = result_collector.collect(
            sample_protocol,
            successful_execution_output,
            statistical_tests=statistical_tests_output
        )

        assert len(result.statistical_tests) == 2
        assert result.primary_test == 'Two-sample T-test'
        assert result.primary_p_value == 0.017
        assert result.primary_effect_size == 0.65

    def test_collect_with_variable_data(self, result_collector, sample_protocol, successful_execution_output,
                                       variable_data):
        """Test collecting result with variable data."""
        result = result_collector.collect(
            sample_protocol,
            successful_execution_output,
            variable_data=variable_data
        )

        assert len(result.variable_results) == 2
        assert any(v.variable_name == 'treatment' for v in result.variable_results)
        assert any(v.variable_name == 'control' for v in result.variable_results)


# Status Determination Tests

class TestStatusDetermination:
    """Tests for determining result status."""

    def test_determine_status_success(self, result_collector):
        """Test status determination for success."""
        output = {'success': True}
        status = result_collector._determine_status(output)
        assert status == ResultStatus.SUCCESS

    def test_determine_status_error(self, result_collector):
        """Test status determination for error."""
        output = {'error': 'Some error'}
        status = result_collector._determine_status(output)
        assert status == ResultStatus.ERROR

    def test_determine_status_timeout(self, result_collector):
        """Test status determination for timeout."""
        output = {'timeout': True}
        status = result_collector._determine_status(output)
        assert status == ResultStatus.TIMEOUT

    def test_determine_status_partial(self, result_collector):
        """Test status determination for partial."""
        output = {'partial': True}
        status = result_collector._determine_status(output)
        assert status == ResultStatus.PARTIAL


# Metadata Creation Tests

class TestMetadataCreation:
    """Tests for execution metadata creation."""

    def test_create_metadata_basic(self, result_collector, sample_protocol, successful_execution_output):
        """Test metadata creation."""
        start_time = datetime.utcnow()
        end_time = datetime.utcnow()

        metadata = result_collector._create_metadata(
            sample_protocol,
            successful_execution_output,
            start_time,
            end_time
        )

        assert isinstance(metadata, ExecutionMetadata)
        assert metadata.experiment_id == "exp-001"
        assert metadata.random_seed == 42
        assert metadata.cpu_time_seconds == 2.3
        assert metadata.memory_peak_mb == 150.0

    def test_metadata_includes_library_versions(self, result_collector, sample_protocol, successful_execution_output):
        """Test metadata includes library versions."""
        metadata = result_collector._create_metadata(
            sample_protocol,
            successful_execution_output,
            datetime.utcnow(),
            datetime.utcnow()
        )

        assert 'numpy' in metadata.library_versions
        assert 'pandas' in metadata.library_versions


# Statistical Test Result Creation Tests

class TestStatisticalTestResultCreation:
    """Tests for creating statistical test results."""

    def test_create_statistical_test_results(self, result_collector, statistical_tests_output):
        """Test creating statistical test results."""
        test_results, primary_test, primary_p, primary_effect = \
            result_collector._create_statistical_test_results(statistical_tests_output)

        assert len(test_results) == 2
        assert all(isinstance(t, StatisticalTestResult) for t in test_results)
        assert primary_test == 'Two-sample T-test'
        assert primary_p == 0.017
        assert primary_effect == 0.65

    def test_statistical_test_significance_labels(self, result_collector):
        """Test significance labels are assigned correctly."""
        tests = [
            {'test_type': 'test1', 'test_name': 'Test 1', 'statistic': 1.0, 'p_value': 0.0005},
            {'test_type': 'test2', 'test_name': 'Test 2', 'statistic': 1.0, 'p_value': 0.005},
            {'test_type': 'test3', 'test_name': 'Test 3', 'statistic': 1.0, 'p_value': 0.03},
            {'test_type': 'test4', 'test_name': 'Test 4', 'statistic': 1.0, 'p_value': 0.5}
        ]

        test_results, _, _, _ = result_collector._create_statistical_test_results(tests)

        assert test_results[0].significance_label == '***'
        assert test_results[1].significance_label == '**'
        assert test_results[2].significance_label == '*'
        assert test_results[3].significance_label == 'ns'

    def test_statistical_test_handles_missing_fields(self, result_collector):
        """Test statistical test creation handles missing fields."""
        tests = [
            {'test_type': 'test1', 'test_name': 'Test 1', 'statistic': 1.0, 'p_value': 0.05}
            # Missing effect_size, CI, etc.
        ]

        test_results, _, _, _ = result_collector._create_statistical_test_results(tests)

        assert len(test_results) == 1
        assert test_results[0].effect_size is None


# Variable Result Creation Tests

class TestVariableResultCreation:
    """Tests for creating variable results."""

    def test_create_variable_results(self, result_collector, sample_protocol, variable_data):
        """Test creating variable results."""
        variable_results = result_collector._create_variable_results(variable_data, sample_protocol)

        assert len(variable_results) == 2
        assert all(isinstance(v, VariableResult) for v in variable_results)

    def test_variable_result_statistics(self, result_collector, sample_protocol, variable_data):
        """Test variable results include statistics."""
        variable_results = result_collector._create_variable_results(variable_data, sample_protocol)

        treatment_result = [v for v in variable_results if v.variable_name == 'treatment'][0]

        assert treatment_result.mean is not None
        assert treatment_result.median is not None
        assert treatment_result.std is not None
        assert treatment_result.min is not None
        assert treatment_result.max is not None
        assert treatment_result.n_samples == 50

    def test_variable_result_handles_missing_data(self, result_collector, sample_protocol):
        """Test variable results handle missing data."""
        data_with_nan = {
            'var1': pd.Series([1, 2, np.nan, 4, 5])
        }

        variable_results = result_collector._create_variable_results(data_with_nan, sample_protocol)

        assert variable_results[0].n_samples == 5
        assert variable_results[0].n_missing == 1


# Hypothesis Support Determination Tests

class TestHypothesisSupportDetermination:
    """Tests for determining hypothesis support."""

    def test_hypothesis_supported_significant_and_large_effect(self, result_collector, sample_protocol):
        """Test hypothesis supported when significant and large effect."""
        supports = result_collector._determine_hypothesis_support(0.01, 0.5, sample_protocol)
        assert supports is True

    def test_hypothesis_not_supported_not_significant(self, result_collector, sample_protocol):
        """Test hypothesis not supported when not significant."""
        supports = result_collector._determine_hypothesis_support(0.1, 0.5, sample_protocol)
        assert supports is False

    def test_hypothesis_not_supported_small_effect(self, result_collector, sample_protocol):
        """Test hypothesis not supported when effect too small."""
        supports = result_collector._determine_hypothesis_support(0.01, 0.1, sample_protocol)
        assert supports is False

    def test_hypothesis_support_none_when_no_pvalue(self, result_collector, sample_protocol):
        """Test hypothesis support is None when no p-value."""
        supports = result_collector._determine_hypothesis_support(None, 0.5, sample_protocol)
        assert supports is None


# Result Export Tests

class TestResultExport:
    """Tests for result export functionality."""

    def test_export_result_json(self, result_collector, sample_protocol, successful_execution_output):
        """Test exporting result to JSON."""
        result = result_collector.collect(sample_protocol, successful_execution_output)

        json_str = result_collector.export_result(result, format='json')

        assert isinstance(json_str, str)
        assert 'experiment_id' in json_str

    def test_export_result_csv(self, result_collector, sample_protocol, successful_execution_output, variable_data):
        """Test exporting result to CSV."""
        result = result_collector.collect(sample_protocol, successful_execution_output, variable_data=variable_data)

        csv_str = result_collector.export_result(result, format='csv')

        assert isinstance(csv_str, str)
        assert 'variable' in csv_str
        assert 'mean' in csv_str

    def test_export_result_markdown(self, result_collector, sample_protocol, successful_execution_output):
        """Test exporting result to Markdown."""
        result = result_collector.collect(sample_protocol, successful_execution_output)

        md_str = result_collector.export_result(result, format='markdown')

        assert isinstance(md_str, str)
        assert '# Experiment Result Report' in md_str
        assert '**Experiment ID:**' in md_str


# Result Versioning Tests

class TestResultVersioning:
    """Tests for result versioning."""

    def test_create_version(self, result_collector, sample_protocol, successful_execution_output):
        """Test creating new version of result."""
        original_result = result_collector.collect(sample_protocol, successful_execution_output)
        original_result.id = "result-001"

        new_execution_output = successful_execution_output.copy()
        new_execution_output['return_value'] = {'mean': 11.0, 'std': 2.5}

        new_result = result_collector.create_version(original_result, new_execution_output, sample_protocol)

        assert new_result.version == 2
        assert new_result.parent_result_id == "result-001"


# Database Storage Tests

class TestDatabaseStorage:
    """Tests for database storage integration."""

    @patch('kosmos.execution.result_collector.get_session')
    @patch('kosmos.execution.result_collector.db_ops')
    def test_store_result_called(self, mock_db_ops, mock_get_session, sample_protocol, successful_execution_output):
        """Test result stored in database when enabled."""
        # Mock the context manager properly
        mock_session = MagicMock()
        mock_get_session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_get_session.return_value.__exit__ = Mock(return_value=False)
        mock_db_ops.create_result.return_value = Mock(id="result-123")

        collector = ResultCollector(store_in_db=True)
        result = collector.collect(sample_protocol, successful_execution_output)

        # Should have called database
        assert mock_db_ops.create_result.called

    def test_store_result_not_called_when_disabled(self, result_collector, sample_protocol, successful_execution_output):
        """Test result not stored when store_in_db=False."""
        result = result_collector.collect(sample_protocol, successful_execution_output)

        # Result should be created but not stored
        assert isinstance(result, ExperimentResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
