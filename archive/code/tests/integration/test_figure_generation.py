"""
Integration tests for figure generation (Issue #60).

These tests create actual figures and verify they exist on disk.
Requires matplotlib to be installed.
"""

import pytest
from pathlib import Path
import numpy as np
from PIL import Image

from kosmos.execution.figure_manager import FigureManager, FigureMetadata
from kosmos.analysis.visualization import PublicationVisualizer, COLORS


class TestFigureManagerIntegration:
    """Test FigureManager with real figure generation."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create FigureManager for testing."""
        return FigureManager(artifacts_dir=tmp_path)

    def test_manager_generates_box_plot(self, manager, tmp_path):
        """Test FigureManager generates real box plot."""
        data = {
            'groups': {
                'Control': np.random.normal(10, 2, 20),
                'Treatment': np.random.normal(15, 2, 20)
            }
        }

        metadata = manager.generate_figure(
            data=data,
            analysis_type='t_test',
            cycle=1,
            task_id=1,
            title="T-Test Comparison"
        )

        assert metadata is not None
        assert Path(metadata.path).exists()
        assert metadata.plot_type == 'box_plot_with_points'
        assert metadata.cycle == 1
        assert metadata.task_id == 1

    def test_manager_generates_scatter_plot(self, manager, tmp_path):
        """Test FigureManager generates real scatter plot."""
        np.random.seed(42)
        x = np.random.uniform(0, 10, 30)
        y = 0.5 * x + np.random.normal(0, 1, 30)

        metadata = manager.generate_figure(
            data={'x': x, 'y': y},
            analysis_type='correlation',
            cycle=1,
            task_id=2,
            title="Correlation Analysis"
        )

        assert metadata is not None
        assert Path(metadata.path).exists()
        assert metadata.plot_type == 'scatter_with_regression'

    def test_manager_creates_figures_in_correct_directory(self, manager, tmp_path):
        """Test figures are created in correct directory structure."""
        data = {'groups': {'A': np.array([1, 2, 3]), 'B': np.array([4, 5, 6])}}

        metadata = manager.generate_figure(
            data=data,
            analysis_type='t_test',
            cycle=3,
            task_id=7
        )

        assert metadata is not None
        expected_dir = tmp_path / "cycle_3" / "figures"
        assert expected_dir.exists()
        assert Path(metadata.path).parent == expected_dir

    def test_manager_tracks_multiple_figures(self, manager):
        """Test manager tracks multiple generated figures."""
        # Generate several figures
        for i in range(3):
            data = {'groups': {'A': np.random.randn(10), 'B': np.random.randn(10)}}
            manager.generate_figure(
                data=data,
                analysis_type='t_test',
                cycle=1,
                task_id=i
            )

        assert manager.get_figure_count() == 3
        assert len(manager.get_figure_paths()) == 3

    def test_manager_filters_figures_by_cycle(self, manager):
        """Test filtering figures by cycle."""
        # Generate figures in different cycles
        data = {'groups': {'A': np.random.randn(10), 'B': np.random.randn(10)}}

        manager.generate_figure(data=data, analysis_type='t_test', cycle=1, task_id=1)
        manager.generate_figure(data=data, analysis_type='t_test', cycle=1, task_id=2)
        manager.generate_figure(data=data, analysis_type='t_test', cycle=2, task_id=1)

        cycle_1_figs = manager.get_figures_for_cycle(1)
        cycle_2_figs = manager.get_figures_for_cycle(2)

        assert len(cycle_1_figs) == 2
        assert len(cycle_2_figs) == 1
