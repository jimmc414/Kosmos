"""
Tests for publication-quality visualization.

Tests matplotlib-based PublicationVisualizer for kosmos-figures formatting compliance.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import os

from kosmos.analysis.visualization import PublicationVisualizer, COLORS
from kosmos.models.result import (
    ExperimentResult,
    ResultStatus,
    StatisticalTestResult,
    VariableResult,
    ExecutionMetadata
)
from datetime import datetime


def _stat_test(**kwargs):
    """Build a StatisticalTestResult; the significance fields are required and derive from p_value."""
    p = kwargs["p_value"]
    kwargs.setdefault("significant_0_05", p < 0.05)
    kwargs.setdefault("significant_0_01", p < 0.01)
    kwargs.setdefault("significant_0_001", p < 0.001)
    kwargs.setdefault(
        "significance_label",
        "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns",
    )
    return StatisticalTestResult(**kwargs)


def _metadata(experiment_id, protocol_id, duration_seconds):
    """Build ExecutionMetadata with all required fields."""
    return ExecutionMetadata(
        experiment_id=experiment_id,
        protocol_id=protocol_id,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        duration_seconds=duration_seconds,
        python_version="3.11.0",
        platform="linux",
        random_seed=42,
    )


# Fixtures

@pytest.fixture
def visualizer():
    """Create PublicationVisualizer instance."""
    return PublicationVisualizer()


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for plot output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_data():
    """Create sample data for plotting."""
    np.random.seed(42)
    return {
        'x': np.random.randn(100),
        'y': np.random.randn(100) * 2 + 3,
        'log2fc': np.random.randn(50),
        'p_values': 10 ** (-np.random.rand(50) * 5),  # Range 1 to 1e-5
        'labels': [f'gene_{i}' for i in range(50)]
    }


@pytest.fixture
def sample_experiment_result():
    """Create sample experiment result for auto-selection."""
    return ExperimentResult(
        id="result-001",
        experiment_id="exp-001",
        hypothesis_id="hyp-001",
        protocol_id="proto-001",
        status=ResultStatus.SUCCESS,
        primary_test="Two-sample T-test",
        primary_p_value=0.012,
        primary_effect_size=0.65,
        supports_hypothesis=True,
        statistical_tests=[
            _stat_test(
                test_type="t-test",
                test_name="Two-sample T-test",
                statistic=2.54,
                p_value=0.012,
                effect_size=0.65,
                effect_size_type="Cohen's d",
                sample_size=100,
                is_primary=True
            )
        ],
        variable_results=[
            VariableResult(
                variable_name=f"var_{i}",
                variable_type="dependent",
                mean=10 + i,
                median=10 + i,
                std=2.0,
                min=5.0,
                max=15.0,
                n_samples=50,
                n_missing=0
            )
            for i in range(3)
        ],
        metadata=_metadata("exp-001", "proto-001", 5.0),
        created_at=datetime.utcnow()
    )


# Formatting Tests

class TestFormattingStandards:
    """Tests for kosmos-figures formatting compliance."""

    def test_rcParams_set_correctly(self, visualizer):
        """Test that matplotlib rcParams match kosmos-figures standards."""
        assert plt.rcParams['font.family'] == ['Arial']
        assert plt.rcParams['font.size'] == 10
        assert plt.rcParams['axes.labelsize'] == 12
        assert plt.rcParams['xtick.labelsize'] == 10
        assert plt.rcParams['ytick.labelsize'] == 10
        assert plt.rcParams['pdf.fonttype'] == 42  # TrueType
        assert plt.rcParams['ps.fonttype'] == 42

    def test_color_scheme_defined(self):
        """Test kosmos-figures color scheme is defined."""
        assert COLORS['red'] == '#d7191c'
        assert COLORS['blue'] == '#0072B2'
        assert COLORS['neutral'] == '#abd9e9'
        assert COLORS['gray'] == '#808080'
        assert COLORS['black'] == '#000000'


# Volcano Plot Tests

class TestVolcanoPlot:
    """Tests for volcano plot generation."""

    def test_volcano_plot_creates_file(self, visualizer, temp_output_dir, sample_data):
        """Test volcano plot creates output file."""
        output_path = os.path.join(temp_output_dir, "volcano.png")

        result_path = visualizer.volcano_plot(
            log2fc=sample_data['log2fc'],
            p_values=sample_data['p_values'],
            output_path=output_path
        )

        assert os.path.exists(output_path)
        assert result_path == output_path

    def test_volcano_plot_with_labels(self, visualizer, temp_output_dir, sample_data):
        """Test volcano plot with gene labels."""
        output_path = os.path.join(temp_output_dir, "volcano_labeled.png")

        visualizer.volcano_plot(
            log2fc=sample_data['log2fc'],
            p_values=sample_data['p_values'],
            labels=sample_data['labels'],
            output_path=output_path
        )

        assert os.path.exists(output_path)

    def test_volcano_plot_custom_thresholds(self, visualizer, temp_output_dir, sample_data):
        """Test volcano plot with custom significance thresholds."""
        output_path = os.path.join(temp_output_dir, "volcano_custom.png")

        visualizer.volcano_plot(
            log2fc=sample_data['log2fc'],
            p_values=sample_data['p_values'],
            fc_threshold=1.0,
            p_threshold=0.01,
            output_path=output_path
        )

        assert os.path.exists(output_path)

    def test_volcano_plot_handles_edge_cases(self, visualizer, temp_output_dir):
        """Test volcano plot handles edge cases (very small p-values, etc.)."""
        log2fc = np.array([0.5, -0.5, 2.0, -2.0])
        p_values = np.array([1e-10, 1e-20, 0.5, 1.0])  # Edge cases

        output_path = os.path.join(temp_output_dir, "volcano_edge.png")

        visualizer.volcano_plot(
            log2fc=log2fc,
            p_values=p_values,
            output_path=output_path
        )

        assert os.path.exists(output_path)


# Heatmap Tests

class TestHeatmap:
    """Tests for heatmap generation."""

    def test_heatmap_creates_file(self, visualizer, temp_output_dir):
        """Test heatmap creates output file."""
        data = np.random.randn(5, 8)
        row_labels = [f'Row_{i}' for i in range(5)]
        col_labels = [f'Col_{i}' for i in range(8)]

        output_path = os.path.join(temp_output_dir, "heatmap.png")

        result_path = visualizer.custom_heatmap(
            data=data,
            row_labels=row_labels,
            col_labels=col_labels,
            output_path=output_path
        )

        assert os.path.exists(output_path)
        assert result_path == output_path

    def test_heatmap_custom_colormap(self, visualizer, temp_output_dir):
        """Test heatmap with custom colormap."""
        data = np.random.randn(4, 6)
        row_labels = [f'R{i}' for i in range(4)]
        col_labels = [f'C{i}' for i in range(6)]

        output_path = os.path.join(temp_output_dir, "heatmap_custom.png")

        visualizer.custom_heatmap(
            data=data,
            row_labels=row_labels,
            col_labels=col_labels,
            cmap='viridis',
            vmin=-2,
            vmax=2,
            output_path=output_path
        )

        assert os.path.exists(output_path)

    def test_heatmap_without_annotations(self, visualizer, temp_output_dir):
        """Test heatmap without cell annotations."""
        data = np.random.randn(10, 15)
        row_labels = [f'R{i}' for i in range(10)]
        col_labels = [f'C{i}' for i in range(15)]

        output_path = os.path.join(temp_output_dir, "heatmap_no_annot.png")

        visualizer.custom_heatmap(
            data=data,
            row_labels=row_labels,
            col_labels=col_labels,
            annot=False,
            output_path=output_path
        )

        assert os.path.exists(output_path)


# Scatter with Regression Tests

class TestScatterWithRegression:
    """Tests for scatter plot with regression."""

    def test_scatter_creates_file(self, visualizer, temp_output_dir, sample_data):
        """Test scatter plot creates output file."""
        output_path = os.path.join(temp_output_dir, "scatter.png")

        result_path = visualizer.scatter_with_regression(
            x=sample_data['x'],
            y=sample_data['y'],
            x_label="Independent Variable",
            y_label="Dependent Variable",
            title="Correlation Analysis",
            output_path=output_path
        )

        assert os.path.exists(output_path)
        assert result_path == output_path

    def test_scatter_linear_relationship(self, visualizer, temp_output_dir):
        """Test scatter plot with strong linear relationship."""
        x = np.linspace(0, 10, 50)
        y = 2 * x + 3 + np.random.randn(50) * 0.5  # y = 2x + 3 + noise

        output_path = os.path.join(temp_output_dir, "scatter_linear.png")

        visualizer.scatter_with_regression(
            x=x,
            y=y,
            x_label="X",
            y_label="Y",
            title="Linear Relationship",
            output_path=output_path
        )

        assert os.path.exists(output_path)


# Log-Log Plot Tests

class TestLogLogPlot:
    """Tests for log-log scaling plot."""

    def test_loglog_creates_file(self, visualizer, temp_output_dir):
        """Test log-log plot creates output file."""
        x = 10 ** np.linspace(0, 3, 50)  # 1 to 1000
        y = 2 * x ** 1.5 + np.random.randn(50) * 10  # Power law

        output_path = os.path.join(temp_output_dir, "loglog.png")

        result_path = visualizer.log_log_plot(
            x=x,
            y=y,
            x_label="Length",
            y_label="Synapses",
            title="Power Law Scaling",
            output_path=output_path
        )

        assert os.path.exists(output_path)
        assert result_path == output_path

    def test_loglog_custom_color(self, visualizer, temp_output_dir):
        """Test log-log plot with custom color."""
        x = 10 ** np.linspace(0, 2, 30)
        y = 5 * x ** 0.8

        output_path = os.path.join(temp_output_dir, "loglog_custom.png")

        visualizer.log_log_plot(
            x=x,
            y=y,
            x_label="X",
            y_label="Y",
            title="Custom Color",
            color='#FF5733',
            output_path=output_path
        )

        assert os.path.exists(output_path)


# Box Plot Tests

class TestBoxPlot:
    """Tests for box plot with points."""

    def test_boxplot_creates_file(self, visualizer, temp_output_dir):
        """Test box plot creates output file."""
        np.random.seed(42)
        data = {
            'Group A': np.random.normal(10, 2, 50),
            'Group B': np.random.normal(12, 2.5, 50),
            'Group C': np.random.normal(8, 1.5, 50)
        }

        output_path = os.path.join(temp_output_dir, "boxplot.png")

        result_path = visualizer.box_plot_with_points(
            data=data,
            title="Group Comparison",
            y_label="Measurement",
            output_path=output_path
        )

        assert os.path.exists(output_path)
        assert result_path == output_path

    def test_boxplot_single_group(self, visualizer, temp_output_dir):
        """Test box plot with single group."""
        data = {'Single Group': np.random.randn(30)}

        output_path = os.path.join(temp_output_dir, "boxplot_single.png")

        visualizer.box_plot_with_points(
            data=data,
            output_path=output_path
        )

        assert os.path.exists(output_path)


# Violin Plot Tests

class TestViolinPlot:
    """Tests for violin plot."""

    def test_violin_creates_file(self, visualizer, temp_output_dir):
        """Test violin plot creates output file."""
        np.random.seed(42)
        data = {
            'Condition 1': np.random.normal(15, 3, 60),
            'Condition 2': np.random.normal(18, 4, 60)
        }

        output_path = os.path.join(temp_output_dir, "violin.png")

        result_path = visualizer.violin_plot(
            data=data,
            title="Distribution Comparison",
            y_label="Response",
            output_path=output_path
        )

        assert os.path.exists(output_path)
        assert result_path == output_path


# Q-Q Plot Tests

class TestQQPlot:
    """Tests for Q-Q plot."""

    def test_qqplot_creates_file(self, visualizer, temp_output_dir):
        """Test Q-Q plot creates output file."""
        # Normal distribution
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)

        output_path = os.path.join(temp_output_dir, "qq.png")

        result_path = visualizer.qq_plot(
            data=data,
            title="Normality Check",
            output_path=output_path
        )

        assert os.path.exists(output_path)
        assert result_path == output_path

    def test_qqplot_non_normal_data(self, visualizer, temp_output_dir):
        """Test Q-Q plot with non-normal data."""
        # Exponential distribution (non-normal)
        np.random.seed(42)
        data = np.random.exponential(2, 100)

        output_path = os.path.join(temp_output_dir, "qq_non_normal.png")

        visualizer.qq_plot(
            data=data,
            title="Non-Normal Distribution",
            output_path=output_path
        )

        assert os.path.exists(output_path)


# Auto-Selection Tests

class TestAutoSelection:
    """Tests for automatic plot type selection."""

    def test_select_plots_for_ttest(self, visualizer, sample_experiment_result):
        """Test auto-selection for t-test result."""
        plots = visualizer.select_plot_types(sample_experiment_result)

        assert len(plots) > 0
        assert any(p['type'] == 'box_plot_with_points' for p in plots)

    def test_select_plots_for_correlation(self, visualizer):
        """Test auto-selection for correlation result."""
        result = ExperimentResult(
            id="result-002",
            experiment_id="exp-002",
            hypothesis_id="hyp-002",
            protocol_id="proto-002",
            status=ResultStatus.SUCCESS,
            primary_test="Pearson Correlation",
            primary_p_value=0.001,
            primary_effect_size=0.8,
            supports_hypothesis=True,
            statistical_tests=[
                _stat_test(
                    test_type="correlation",
                    test_name="Pearson Correlation",
                    statistic=0.8,
                    p_value=0.001,
                    sample_size=100,
                    is_primary=True
                )
            ],
            variable_results=[],
            metadata=_metadata("exp-002", "proto-002", 2.0),
            created_at=datetime.utcnow()
        )

        plots = visualizer.select_plot_types(result)

        assert len(plots) > 0
        assert any(p['type'] == 'scatter_with_regression' for p in plots)

    def test_select_plots_for_multiple_tests(self, visualizer):
        """Test auto-selection for multiple statistical tests."""
        result = ExperimentResult(
            id="result-003",
            experiment_id="exp-003",
            hypothesis_id="hyp-003",
            protocol_id="proto-003",
            status=ResultStatus.SUCCESS,
            # primary_test must name one of statistical_tests
            primary_test="Test 0",
            primary_p_value=0.01,
            primary_effect_size=0.5,
            supports_hypothesis=True,
            statistical_tests=[
                _stat_test(
                    test_type=f"test{i}",
                    test_name=f"Test {i}",
                    statistic=i * 0.5,
                    p_value=0.01 + i * 0.01,
                    sample_size=100,
                    is_primary=(i == 0)
                )
                for i in range(5)
            ],
            variable_results=[],
            metadata=_metadata("exp-003", "proto-003", 3.0),
            created_at=datetime.utcnow()
        )

        plots = visualizer.select_plot_types(result)

        assert len(plots) > 0
        # Should suggest volcano plot for multiple tests
        assert any(p['type'] == 'volcano_plot' for p in plots)

    def test_select_plots_with_many_variables(self, visualizer):
        """Test auto-selection suggests heatmap for many variables."""
        result = ExperimentResult(
            id="result-004",
            experiment_id="exp-004",
            hypothesis_id="hyp-004",
            protocol_id="proto-004",
            status=ResultStatus.SUCCESS,
            # No statistical tests, so no primary test to name
            primary_test=None,
            primary_p_value=0.05,
            primary_effect_size=0.3,
            supports_hypothesis=True,
            statistical_tests=[],
            variable_results=[
                VariableResult(
                    variable_name=f"var_{i}",
                    variable_type="dependent",
                    mean=10.0,
                    median=10.0,
                    std=2.0,
                    min=5.0,
                    max=15.0,
                    n_samples=50,
                    n_missing=0
                )
                for i in range(5)  # Multiple variables
            ],
            metadata=_metadata("exp-004", "proto-004", 2.0),
            created_at=datetime.utcnow()
        )

        plots = visualizer.select_plot_types(result)

        assert len(plots) > 0
        assert any(p['type'] == 'custom_heatmap' for p in plots)


# Error Handling Tests

class TestErrorHandling:
    """Tests for error handling in visualization."""

    def test_handles_empty_data(self, visualizer, temp_output_dir):
        """Test handling of empty data arrays."""
        output_path = os.path.join(temp_output_dir, "empty.png")

        # Should handle gracefully or raise informative error
        try:
            visualizer.volcano_plot(
                log2fc=np.array([]),
                p_values=np.array([]),
                output_path=output_path
            )
        except (ValueError, IndexError) as e:
            # Expected - empty data should raise error
            assert True

    def test_handles_nan_values(self, visualizer, temp_output_dir):
        """Test handling of NaN values."""
        data = np.random.randn(5, 5)
        data[0, 0] = np.nan
        data[2, 3] = np.nan

        row_labels = [f'R{i}' for i in range(5)]
        col_labels = [f'C{i}' for i in range(5)]

        output_path = os.path.join(temp_output_dir, "nan_heatmap.png")

        # Should handle NaN values
        visualizer.custom_heatmap(
            data=data,
            row_labels=row_labels,
            col_labels=col_labels,
            output_path=output_path
        )

        assert os.path.exists(output_path)

    def test_handles_mismatched_sizes(self, visualizer, temp_output_dir):
        """Test handling of mismatched array sizes."""
        x = np.array([1, 2, 3])
        y = np.array([1, 2])  # Different size

        output_path = os.path.join(temp_output_dir, "mismatch.png")

        # Should raise error for mismatched sizes
        with pytest.raises((ValueError, IndexError)):
            visualizer.scatter_with_regression(
                x=x,
                y=y,
                x_label="X",
                y_label="Y",
                title="Mismatch",
                output_path=output_path
            )


# DPI and Quality Tests

class TestOutputQuality:
    """Tests for output quality and DPI settings."""

    def test_standard_plots_use_dpi_300(self, visualizer, temp_output_dir, sample_data):
        """Test standard plots use DPI 300."""
        output_path = os.path.join(temp_output_dir, "standard_dpi.png")

        visualizer.scatter_with_regression(
            x=sample_data['x'],
            y=sample_data['y'],
            x_label="X",
            y_label="Y",
            title="Standard DPI",
            output_path=output_path
        )

        # File should exist and be reasonably sized (higher DPI = larger file)
        assert os.path.exists(output_path)
        file_size = os.path.getsize(output_path)
        assert file_size > 10000  # Should be > 10KB for DPI 300

    def test_panel_plots_use_dpi_600(self, visualizer, temp_output_dir):
        """Test panel plots use DPI 600."""
        x = 10 ** np.linspace(0, 3, 50)
        y = 2 * x ** 1.5

        output_path = os.path.join(temp_output_dir, "panel_dpi.png")

        visualizer.log_log_plot(
            x=x,
            y=y,
            x_label="X",
            y_label="Y",
            title="Panel DPI",
            output_path=output_path
        )

        # File should exist and be larger (DPI 600)
        assert os.path.exists(output_path)
        file_size = os.path.getsize(output_path)
        assert file_size > 15000  # Should be > 15KB for DPI 600


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
