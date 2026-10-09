"""
Unit tests for FigureManager (Issue #60).

Tests figure path generation, plot type selection, and metadata tracking.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from kosmos.execution.figure_manager import (
    FigureManager,
    FigureMetadata,
    ANALYSIS_PLOT_MAPPING,
    generate_code_figure_block
)


class TestFigureMetadata:
    """Test FigureMetadata dataclass."""

    def test_create_metadata(self):
        """Test creating metadata instance."""
        metadata = FigureMetadata(
            path="/path/to/figure.png",
            plot_type="scatter_with_regression",
            analysis_type="correlation",
            dpi=300,
            cycle=1,
            task_id=3
        )

        assert metadata.path == "/path/to/figure.png"
        assert metadata.plot_type == "scatter_with_regression"
        assert metadata.analysis_type == "correlation"
        assert metadata.dpi == 300
        assert metadata.cycle == 1
        assert metadata.task_id == 3

    def test_metadata_to_dict(self):
        """Test metadata serialization."""
        metadata = FigureMetadata(
            path="/path/to/figure.png",
            plot_type="box_plot_with_points",
            analysis_type="t_test",
            dpi=300,
            caption="Test caption"
        )

        data = metadata.to_dict()

        assert data['path'] == "/path/to/figure.png"
        assert data['plot_type'] == "box_plot_with_points"
        assert data['caption'] == "Test caption"
        assert 'cycle' in data  # Should include even if None


class TestFigureManagerInit:
    """Test FigureManager initialization."""

    def test_init_with_path(self, tmp_path):
        """Test initialization with path."""
        manager = FigureManager(artifacts_dir=tmp_path, use_visualizer=False)

        assert manager.artifacts_dir == tmp_path
        assert manager.default_dpi == 300
        assert manager.generated_figures == []

    def test_init_with_string_path(self, tmp_path):
        """Test initialization with string path."""
        manager = FigureManager(artifacts_dir=str(tmp_path), use_visualizer=False)

        assert manager.artifacts_dir == tmp_path

    def test_init_custom_dpi(self, tmp_path):
        """Test initialization with custom DPI."""
        manager = FigureManager(artifacts_dir=tmp_path, default_dpi=600, use_visualizer=False)

        assert manager.default_dpi == 600


class TestFigureManagerPaths:
    """Test path generation."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager for testing."""
        return FigureManager(artifacts_dir=tmp_path, use_visualizer=False)

    def test_get_figures_dir_creates_directory(self, manager, tmp_path):
        """Test that get_figures_dir creates the directory."""
        figures_dir = manager.get_figures_dir(cycle=1)

        assert figures_dir.exists()
        assert figures_dir == tmp_path / "cycle_1" / "figures"

    def test_get_figures_dir_multiple_cycles(self, manager, tmp_path):
        """Test figures dir for different cycles."""
        dir_1 = manager.get_figures_dir(cycle=1)
        dir_2 = manager.get_figures_dir(cycle=2)
        dir_5 = manager.get_figures_dir(cycle=5)

        assert dir_1 == tmp_path / "cycle_1" / "figures"
        assert dir_2 == tmp_path / "cycle_2" / "figures"
        assert dir_5 == tmp_path / "cycle_5" / "figures"

    def test_get_figure_path(self, manager, tmp_path):
        """Test figure path generation."""
        path = manager.get_figure_path(cycle=1, task_id=3, plot_type="scatter")

        assert path == tmp_path / "cycle_1" / "figures" / "task_3_scatter.png"

    def test_get_figure_path_custom_suffix(self, manager, tmp_path):
        """Test figure path with custom suffix."""
        path = manager.get_figure_path(cycle=2, task_id=1, plot_type="heatmap", suffix="pdf")

        assert path == tmp_path / "cycle_2" / "figures" / "task_1_heatmap.pdf"

    def test_get_figure_path_creates_parent_dir(self, manager, tmp_path):
        """Test that get_figure_path creates parent directory."""
        path = manager.get_figure_path(cycle=10, task_id=5, plot_type="box")

        assert path.parent.exists()


class TestPlotTypeSelection:
    """Test plot type selection logic."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager for testing."""
        return FigureManager(artifacts_dir=tmp_path, use_visualizer=False)

    def test_select_ttest(self, manager):
        """Test t-test maps to box plot."""
        assert manager.select_plot_type("t_test") == "box_plot_with_points"
        assert manager.select_plot_type("t-test") == "box_plot_with_points"
        assert manager.select_plot_type("ttest") == "box_plot_with_points"

    def test_select_correlation(self, manager):
        """Test correlation maps to scatter."""
        assert manager.select_plot_type("correlation") == "scatter_with_regression"
        assert manager.select_plot_type("regression") == "scatter_with_regression"
        assert manager.select_plot_type("pearson") == "scatter_with_regression"
        assert manager.select_plot_type("spearman") == "scatter_with_regression"

    def test_select_log_log(self, manager):
        """Test log-log maps to log_log_plot."""
        assert manager.select_plot_type("log_log") == "log_log_plot"
        assert manager.select_plot_type("log-log") == "log_log_plot"
        assert manager.select_plot_type("power_law") == "log_log_plot"
        assert manager.select_plot_type("scaling") == "log_log_plot"

    def test_select_anova(self, manager):
        """Test ANOVA maps to box plot."""
        assert manager.select_plot_type("anova") == "box_plot_with_points"

    def test_select_distribution(self, manager):
        """Test distribution maps to violin plot."""
        assert manager.select_plot_type("distribution") == "violin_plot"

    def test_select_normality(self, manager):
        """Test normality maps to QQ plot."""
        assert manager.select_plot_type("normality") == "qq_plot"
        assert manager.select_plot_type("qq") == "qq_plot"

    def test_select_volcano(self, manager):
        """Test volcano maps correctly."""
        assert manager.select_plot_type("volcano") == "volcano_plot"
        assert manager.select_plot_type("differential_expression") == "volcano_plot"

    def test_select_heatmap(self, manager):
        """Test heatmap maps correctly."""
        assert manager.select_plot_type("heatmap") == "custom_heatmap"
        assert manager.select_plot_type("correlation_matrix") == "custom_heatmap"

    def test_select_partial_match(self, manager):
        """Test partial matching works."""
        assert manager.select_plot_type("perform_t_test_analysis") == "box_plot_with_points"
        assert manager.select_plot_type("correlation_analysis") == "scatter_with_regression"

    def test_select_with_n_groups(self, manager):
        """Test selection with group count heuristic."""
        # Few groups -> box plot
        assert manager.select_plot_type("unknown", n_groups=2) == "box_plot_with_points"
        assert manager.select_plot_type("unknown", n_groups=4) == "box_plot_with_points"

        # Many groups -> violin plot
        assert manager.select_plot_type("unknown", n_groups=5) == "violin_plot"

    def test_select_with_n_variables(self, manager):
        """Test selection with variable count heuristic."""
        # Many variables -> heatmap
        assert manager.select_plot_type("unknown", n_variables=3) == "custom_heatmap"
        assert manager.select_plot_type("unknown", n_variables=10) == "custom_heatmap"

    def test_select_default(self, manager):
        """Test default when no match."""
        assert manager.select_plot_type("unknown_analysis") == "box_plot_with_points"


class TestFigureGeneration:
    """Test figure generation with mocked visualizer."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager with mocked visualizer."""
        manager = FigureManager(artifacts_dir=tmp_path, use_visualizer=False)
        # Create mock visualizer
        manager._visualizer = Mock()
        manager._use_visualizer = True
        return manager

    def test_generate_box_plot(self, manager, tmp_path):
        """Test generating box plot."""
        # Setup mock
        expected_path = tmp_path / "cycle_1" / "figures" / "task_1_box_plot_with_points.png"
        manager._visualizer.box_plot_with_points.return_value = str(expected_path)

        # Generate figure
        data = {'Group A': np.array([1, 2, 3]), 'Group B': np.array([4, 5, 6])}
        metadata = manager.generate_figure(
            data={'groups': data},
            analysis_type='t_test',
            cycle=1,
            task_id=1,
            title="Test Box Plot"
        )

        assert metadata is not None
        assert metadata.plot_type == "box_plot_with_points"
        assert manager._visualizer.box_plot_with_points.called

    def test_generate_scatter_plot(self, manager, tmp_path):
        """Test generating scatter plot."""
        expected_path = tmp_path / "cycle_1" / "figures" / "task_2_scatter_with_regression.png"
        manager._visualizer.scatter_with_regression.return_value = str(expected_path)

        metadata = manager.generate_figure(
            data={'x': [1, 2, 3], 'y': [2, 4, 6]},
            analysis_type='correlation',
            cycle=1,
            task_id=2
        )

        assert metadata is not None
        assert metadata.plot_type == "scatter_with_regression"

    def test_generate_log_log_plot_uses_600_dpi(self, manager, tmp_path):
        """Test log-log plot uses 600 DPI."""
        expected_path = tmp_path / "cycle_1" / "figures" / "task_3_log_log_plot.png"
        manager._visualizer.log_log_plot.return_value = str(expected_path)

        metadata = manager.generate_figure(
            data={'x': [1, 10, 100], 'y': [1, 100, 10000]},
            analysis_type='power_law',
            cycle=1,
            task_id=3
        )

        assert metadata is not None
        assert metadata.dpi == 600  # Panel DPI

    def test_generate_tracks_figures(self, manager, tmp_path):
        """Test that generated figures are tracked."""
        manager._visualizer.box_plot_with_points.return_value = "figure1.png"
        manager._visualizer.scatter_with_regression.return_value = "figure2.png"

        manager.generate_figure(data={'groups': {}}, analysis_type='t_test', cycle=1, task_id=1)
        manager.generate_figure(data={'x': [], 'y': []}, analysis_type='correlation', cycle=1, task_id=2)

        assert manager.get_figure_count() == 2
        assert len(manager.get_figure_paths()) == 2

    def test_generate_handles_error(self, manager):
        """Test error handling during generation."""
        manager._visualizer.box_plot_with_points.side_effect = Exception("Plot error")

        metadata = manager.generate_figure(
            data={'groups': {}},
            analysis_type='t_test',
            cycle=1,
            task_id=1
        )

        assert metadata is None  # Should return None on error


class TestFigureManagerSerialization:
    """Test serialization methods."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager with some figures."""
        manager = FigureManager(artifacts_dir=tmp_path, use_visualizer=False)
        manager.generated_figures = [
            FigureMetadata(path="fig1.png", plot_type="box", analysis_type="ttest", dpi=300, cycle=1, task_id=1),
            FigureMetadata(path="fig2.png", plot_type="scatter", analysis_type="corr", dpi=300, cycle=1, task_id=2),
        ]
        return manager

    def test_to_dict(self, manager, tmp_path):
        """Test serialization to dict."""
        data = manager.to_dict()

        assert data['artifacts_dir'] == str(tmp_path)
        assert data['default_dpi'] == 300
        assert data['figure_count'] == 2
        assert len(data['figures']) == 2

    def test_get_figures_for_cycle(self, manager):
        """Test filtering figures by cycle."""
        manager.generated_figures.append(
            FigureMetadata(path="fig3.png", plot_type="violin", analysis_type="dist", dpi=300, cycle=2, task_id=1)
        )

        cycle_1_figs = manager.get_figures_for_cycle(1)
        cycle_2_figs = manager.get_figures_for_cycle(2)

        assert len(cycle_1_figs) == 2
        assert len(cycle_2_figs) == 1


class TestAnalysisPlotMapping:
    """Test the analysis to plot type mapping."""

    def test_mapping_contains_common_types(self):
        """Test mapping has common analysis types."""
        assert 't_test' in ANALYSIS_PLOT_MAPPING
        assert 'correlation' in ANALYSIS_PLOT_MAPPING
        assert 'anova' in ANALYSIS_PLOT_MAPPING
        assert 'log_log' in ANALYSIS_PLOT_MAPPING

    def test_mapping_values_are_valid(self):
        """Test all mapping values are valid plot types."""
        valid_plot_types = {
            'box_plot_with_points', 'scatter_with_regression', 'log_log_plot',
            'violin_plot', 'qq_plot', 'custom_heatmap', 'volcano_plot'
        }

        for plot_type in ANALYSIS_PLOT_MAPPING.values():
            assert plot_type in valid_plot_types


class TestCodeFigureBlock:
    """Test generate_code_figure_block helper."""

    def test_box_plot_block(self):
        """Test generating box plot code block."""
        lines = generate_code_figure_block(
            plot_type='box_plot_with_points',
            group_var='treatment',
            measure_var='response',
            title='Treatment Effect'
        )

        code = '\n'.join(lines)
        assert 'PublicationVisualizer' in code
        assert 'box_plot_with_points' in code
        assert 'treatment' in code
        assert 'response' in code

    def test_scatter_plot_block(self):
        """Test generating scatter plot code block."""
        lines = generate_code_figure_block(
            plot_type='scatter_with_regression',
            x_var='x_feature',
            y_var='y_target',
            title='Correlation'
        )

        code = '\n'.join(lines)
        assert 'scatter_with_regression' in code
        assert 'x_feature' in code
        assert 'y_target' in code

    def test_log_log_block(self):
        """Test generating log-log plot code block."""
        lines = generate_code_figure_block(
            plot_type='log_log_plot',
            x_var='size',
            y_var='frequency',
            title='Scaling Law'
        )

        code = '\n'.join(lines)
        assert 'log_log_plot' in code
        assert 'size' in code
        assert 'frequency' in code

    def test_block_without_variables(self):
        """Test code block generation without variable names."""
        lines = generate_code_figure_block(
            plot_type='box_plot_with_points',
            title='Test'
        )

        code = '\n'.join(lines)
        # Should still generate valid code with comments
        assert 'PublicationVisualizer' in code
