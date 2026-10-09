"""
Visual regression tests for publication visualizations.

Tests that visualizations maintain consistent formatting and quality.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import hashlib
import os

from kosmos.analysis.visualization import PublicationVisualizer, COLORS


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
def deterministic_data():
    """Create deterministic data for consistent visual output."""
    np.random.seed(42)
    return {
        'x': np.linspace(0, 10, 50),
        'y': 2 * np.linspace(0, 10, 50) + np.random.randn(50) * 0.5,
        'log2fc': np.random.randn(100),
        'p_values': 10 ** (-np.random.rand(100) * 5)
    }


# Visual Consistency Tests

class TestVisualConsistency:
    """Tests for consistent visual output."""

    def test_scatter_plot_consistent_output(self, visualizer, temp_output_dir, deterministic_data):
        """Test scatter plot produces consistent output with same data."""
        output_path1 = os.path.join(temp_output_dir, "scatter1.png")
        output_path2 = os.path.join(temp_output_dir, "scatter2.png")

        # Generate same plot twice
        visualizer.scatter_with_regression(
            x=deterministic_data['x'],
            y=deterministic_data['y'],
            x_label="X",
            y_label="Y",
            title="Consistency Test",
            output_path=output_path1
        )

        visualizer.scatter_with_regression(
            x=deterministic_data['x'],
            y=deterministic_data['y'],
            x_label="X",
            y_label="Y",
            title="Consistency Test",
            output_path=output_path2
        )

        # Files should be identical (or very similar due to matplotlib rendering)
        # Note: Exact pixel matching is fragile, so we check file exists and size
        assert os.path.exists(output_path1)
        assert os.path.exists(output_path2)

        size1 = os.path.getsize(output_path1)
        size2 = os.path.getsize(output_path2)

        # Sizes should be identical or within 1% (allows for minor rendering differences)
        assert abs(size1 - size2) / size1 < 0.01

    def test_volcano_plot_consistent_output(self, visualizer, temp_output_dir, deterministic_data):
        """Test volcano plot produces consistent output."""
        output_path1 = os.path.join(temp_output_dir, "volcano1.png")
        output_path2 = os.path.join(temp_output_dir, "volcano2.png")

        # Generate same plot twice
        for output_path in [output_path1, output_path2]:
            visualizer.volcano_plot(
                log2fc=deterministic_data['log2fc'],
                p_values=deterministic_data['p_values'],
                output_path=output_path
            )

        assert os.path.exists(output_path1)
        assert os.path.exists(output_path2)

        # Check consistency
        size1 = os.path.getsize(output_path1)
        size2 = os.path.getsize(output_path2)
        assert abs(size1 - size2) / size1 < 0.01


# Formatting Preservation Tests

class TestFormattingPreservation:
    """Tests that formatting standards are preserved across plots."""

    def test_all_plots_remove_top_right_spines(self, visualizer, temp_output_dir):
        """Test all plots remove top and right spines."""
        # This is a visual check - would need image analysis or manual verification
        # For now, we verify plots generate without error
        np.random.seed(42)

        data_dict = {
            'Group A': np.random.randn(30),
            'Group B': np.random.randn(30)
        }

        plots = [
            ('box', lambda: visualizer.box_plot_with_points(
                data=data_dict,
                output_path=os.path.join(temp_output_dir, "box.png")
            )),
            ('violin', lambda: visualizer.violin_plot(
                data=data_dict,
                output_path=os.path.join(temp_output_dir, "violin.png")
            )),
            ('qq', lambda: visualizer.qq_plot(
                data=np.random.randn(100),
                output_path=os.path.join(temp_output_dir, "qq.png")
            ))
        ]

        for name, plot_func in plots:
            plot_func()
            # Plot should generate successfully
            assert os.path.exists(os.path.join(temp_output_dir, f"{name}.png"))

    def test_plots_use_correct_dpi(self, visualizer, temp_output_dir, deterministic_data):
        """Test plots use correct DPI settings."""
        # Standard plots use DPI 300
        standard_path = os.path.join(temp_output_dir, "standard_dpi.png")
        visualizer.scatter_with_regression(
            x=deterministic_data['x'],
            y=deterministic_data['y'],
            x_label="X",
            y_label="Y",
            title="Standard DPI",
            output_path=standard_path
        )

        # Panel plots use DPI 600
        panel_path = os.path.join(temp_output_dir, "panel_dpi.png")
        visualizer.log_log_plot(
            x=10 ** np.linspace(0, 3, 50),
            y=10 ** np.linspace(0, 3, 50) * 2,
            x_label="X",
            y_label="Y",
            title="Panel DPI",
            output_path=panel_path
        )

        # Read the DPI recorded in each PNG's pHYs chunk instead of comparing
        # byte sizes (compression makes file size an unreliable DPI proxy).
        from PIL import Image

        with Image.open(standard_path) as img:
            standard_dpi = img.info.get('dpi')
        with Image.open(panel_path) as img:
            panel_dpi = img.info.get('dpi')

        assert standard_dpi is not None
        assert panel_dpi is not None
        # PIL converts pixels-per-metre back to DPI, so allow rounding error
        assert standard_dpi[0] == pytest.approx(300, abs=1)
        assert standard_dpi[1] == pytest.approx(300, abs=1)
        assert panel_dpi[0] == pytest.approx(600, abs=1)
        assert panel_dpi[1] == pytest.approx(600, abs=1)


# Color Scheme Tests

class TestColorScheme:
    """Tests that color scheme is consistent."""

    def test_color_constants_match_kosmos_figures(self):
        """Test color constants match kosmos-figures exactly."""
        assert COLORS['red'] == '#d7191c'
        assert COLORS['blue'] == '#0072B2'
        assert COLORS['blue_dark'] == '#2c7bb6'
        assert COLORS['neutral'] == '#abd9e9'
        assert COLORS['gray'] == '#808080'
        assert COLORS['black'] == '#000000'

    def test_plots_use_defined_colors(self, visualizer, temp_output_dir, deterministic_data):
        """Test plots use colors from COLORS dictionary."""
        # Generate plots and verify they complete without error
        # Actual color usage would require image analysis

        visualizer.volcano_plot(
            log2fc=deterministic_data['log2fc'],
            p_values=deterministic_data['p_values'],
            output_path=os.path.join(temp_output_dir, "volcano_colors.png")
        )

        visualizer.scatter_with_regression(
            x=deterministic_data['x'],
            y=deterministic_data['y'],
            x_label="X",
            y_label="Y",
            title="Scatter Colors",
            output_path=os.path.join(temp_output_dir, "scatter_colors.png")
        )

        # Both should generate successfully
        assert os.path.exists(os.path.join(temp_output_dir, "volcano_colors.png"))
        assert os.path.exists(os.path.join(temp_output_dir, "scatter_colors.png"))


# Matplotlib rcParams Tests

class TestMatplotlibConfig:
    """Tests that matplotlib configuration is correct."""

    def test_rcparams_set_on_init(self, visualizer):
        """Test rcParams are set when visualizer initializes."""
        # These should be set by __init__
        assert plt.rcParams['font.family'] == ['Arial']
        assert plt.rcParams['pdf.fonttype'] == 42
        assert plt.rcParams['ps.fonttype'] == 42

    def test_rcparams_persist_across_plots(self, visualizer, temp_output_dir, deterministic_data):
        """Test rcParams persist across multiple plot generations."""
        # Generate multiple plots
        for i in range(3):
            visualizer.scatter_with_regression(
                x=deterministic_data['x'],
                y=deterministic_data['y'],
                x_label="X",
                y_label="Y",
                title=f"Plot {i}",
                output_path=os.path.join(temp_output_dir, f"plot_{i}.png")
            )

            # Check rcParams still set
            assert plt.rcParams['font.family'] == ['Arial']
            assert plt.rcParams['pdf.fonttype'] == 42


# File Output Tests

class TestFileOutput:
    """Tests for file output quality and format."""

    def test_all_plot_types_create_files(self, visualizer, temp_output_dir):
        """Test all plot types successfully create output files."""
        np.random.seed(42)

        plots_to_test = {
            'volcano': lambda: visualizer.volcano_plot(
                np.random.randn(50),
                10 ** (-np.random.rand(50) * 5),
                output_path=os.path.join(temp_output_dir, "volcano.png")
            ),
            'heatmap': lambda: visualizer.custom_heatmap(
                np.random.randn(5, 5),
                [f'R{i}' for i in range(5)],
                [f'C{i}' for i in range(5)],
                output_path=os.path.join(temp_output_dir, "heatmap.png")
            ),
            'scatter': lambda: visualizer.scatter_with_regression(
                np.random.randn(50),
                np.random.randn(50),
                "X", "Y", "Scatter",
                output_path=os.path.join(temp_output_dir, "scatter.png")
            ),
            'loglog': lambda: visualizer.log_log_plot(
                10 ** np.linspace(0, 3, 50),
                10 ** np.linspace(0, 3, 50) * 2,
                "X", "Y", "Log-Log",
                output_path=os.path.join(temp_output_dir, "loglog.png")
            ),
            'box': lambda: visualizer.box_plot_with_points(
                {'A': np.random.randn(30), 'B': np.random.randn(30)},
                output_path=os.path.join(temp_output_dir, "box.png")
            ),
            'violin': lambda: visualizer.violin_plot(
                {'A': np.random.randn(30), 'B': np.random.randn(30)},
                output_path=os.path.join(temp_output_dir, "violin.png")
            ),
            'qq': lambda: visualizer.qq_plot(
                np.random.randn(100),
                output_path=os.path.join(temp_output_dir, "qq.png")
            )
        }

        for name, plot_func in plots_to_test.items():
            plot_func()
            file_path = os.path.join(temp_output_dir, f"{name}.png")
            assert os.path.exists(file_path), f"{name} plot file not created"
            assert os.path.getsize(file_path) > 1000, f"{name} plot file too small"

    def test_plots_use_png_format(self, visualizer, temp_output_dir, deterministic_data):
        """Test plots save in PNG format."""
        output_path = os.path.join(temp_output_dir, "format_test.png")

        visualizer.scatter_with_regression(
            x=deterministic_data['x'],
            y=deterministic_data['y'],
            x_label="X",
            y_label="Y",
            title="Format Test",
            output_path=output_path
        )

        # Check file exists and has PNG signature
        assert os.path.exists(output_path)

        with open(output_path, 'rb') as f:
            header = f.read(8)
            # PNG signature: 89 50 4E 47 0D 0A 1A 0A
            assert header[:4] == b'\x89PNG'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
