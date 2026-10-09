"""
Integration tests for figure generation (Issue #60).

These tests create actual figures and verify they exist on disk.
Requires matplotlib to be installed.
"""

import pytest
from pathlib import Path
import numpy as np
from PIL import Image

from kosmos.analysis.visualization import PublicationVisualizer, COLORS


class TestRealFigureGeneration:
    """Test actual figure generation with PublicationVisualizer."""

    @pytest.fixture
    def visualizer(self):
        """Create visualizer for testing."""
        return PublicationVisualizer()

    def test_generate_real_box_plot(self, visualizer, tmp_path):
        """Test generating actual box plot PNG."""
        output_path = tmp_path / "box_plot.png"

        # Create sample data
        data = {
            'Control': np.random.normal(10, 2, 30),
            'Treatment': np.random.normal(12, 2, 30)
        }

        # Generate figure
        result_path = visualizer.box_plot_with_points(
            data=data,
            title="Treatment vs Control",
            y_label="Response",
            output_path=str(output_path)
        )

        # Verify file exists
        assert Path(result_path).exists()
        assert output_path.stat().st_size > 0

    def test_generate_real_scatter_plot(self, visualizer, tmp_path):
        """Test generating actual scatter plot with regression."""
        output_path = tmp_path / "scatter.png"

        # Create correlated data
        np.random.seed(42)
        x = np.random.uniform(0, 10, 50)
        y = 2 * x + 1 + np.random.normal(0, 1, 50)

        # Generate figure
        result_path = visualizer.scatter_with_regression(
            x=x,
            y=y,
            x_label="Independent Variable",
            y_label="Dependent Variable",
            title="Correlation Analysis",
            output_path=str(output_path)
        )

        # Verify file exists
        assert Path(result_path).exists()
        assert output_path.stat().st_size > 0

    def test_generate_real_log_log_plot(self, visualizer, tmp_path):
        """Test generating actual log-log plot."""
        output_path = tmp_path / "log_log.png"

        # Create power law data
        np.random.seed(42)
        x = np.logspace(0, 3, 100)
        y = x ** 2 * np.random.uniform(0.8, 1.2, 100)

        # Generate figure
        result_path = visualizer.log_log_plot(
            x=x,
            y=y,
            x_label="Size",
            y_label="Frequency",
            title="Scaling Law",
            output_path=str(output_path)
        )

        # Verify file exists
        assert Path(result_path).exists()
        assert output_path.stat().st_size > 0

    def test_generate_real_violin_plot(self, visualizer, tmp_path):
        """Test generating actual violin plot."""
        output_path = tmp_path / "violin.png"

        data = {
            'Group A': np.random.normal(0, 1, 100),
            'Group B': np.random.normal(0.5, 1.5, 100),
            'Group C': np.random.normal(-0.5, 0.8, 100)
        }

        result_path = visualizer.violin_plot(
            data=data,
            title="Distribution Comparison",
            y_label="Value",
            output_path=str(output_path)
        )

        assert Path(result_path).exists()

    def test_generate_real_volcano_plot(self, visualizer, tmp_path):
        """Test generating actual volcano plot."""
        output_path = tmp_path / "volcano.png"

        # Simulated differential expression data
        np.random.seed(42)
        n_genes = 1000
        log2fc = np.random.normal(0, 1, n_genes)
        p_values = np.random.uniform(0, 1, n_genes)

        # Make some significant
        p_values[:50] = np.random.uniform(0.001, 0.01, 50)
        log2fc[:25] = np.random.uniform(1, 3, 25)
        log2fc[25:50] = np.random.uniform(-3, -1, 25)

        result_path = visualizer.volcano_plot(
            log2fc=log2fc,
            p_values=p_values,
            title="Differential Expression",
            output_path=str(output_path)
        )

        assert Path(result_path).exists()

    def test_generate_real_heatmap(self, visualizer, tmp_path):
        """Test generating actual heatmap."""
        output_path = tmp_path / "heatmap.png"

        # Create correlation matrix-like data
        np.random.seed(42)
        data = np.random.uniform(-1, 1, (5, 5))
        np.fill_diagonal(data, 1)

        result_path = visualizer.custom_heatmap(
            data=data,
            row_labels=['Gene A', 'Gene B', 'Gene C', 'Gene D', 'Gene E'],
            col_labels=['Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Sample 5'],
            title="Correlation Matrix",
            output_path=str(output_path)
        )

        assert Path(result_path).exists()


class TestFigureDPI:
    """Test that figures are saved at correct DPI."""

    @pytest.fixture
    def visualizer(self):
        """Create visualizer for testing."""
        return PublicationVisualizer()

    def test_standard_figure_dpi_is_300(self, visualizer, tmp_path):
        """Test standard figures are at least 300 DPI."""
        output_path = tmp_path / "standard.png"

        data = {'A': np.array([1, 2, 3]), 'B': np.array([4, 5, 6])}
        visualizer.box_plot_with_points(data=data, output_path=str(output_path))

        # Open image and check resolution
        img = Image.open(output_path)
        dpi = img.info.get('dpi', (72, 72))

        # DPI should be at least 300 (may be slightly different due to rounding)
        assert dpi[0] >= 250, f"DPI {dpi[0]} is less than expected 300"

    def test_log_log_figure_dpi_is_600(self, visualizer, tmp_path):
        """Test log-log figures are at least 600 DPI."""
        output_path = tmp_path / "log_log.png"

        x = np.logspace(0, 2, 50)
        y = x ** 1.5
        visualizer.log_log_plot(x=x, y=y, x_label="X", y_label="Y", title="Test", output_path=str(output_path))

        img = Image.open(output_path)
        dpi = img.info.get('dpi', (72, 72))

        # DPI should be at least 600
        assert dpi[0] >= 500, f"DPI {dpi[0]} is less than expected 600"


class TestFindingFigureIntegration:
    """Test Finding dataclass figure fields."""

    def test_finding_with_figure_paths(self):
        """Test Finding includes figure paths."""
        from kosmos.world_model.artifacts import Finding

        finding = Finding(
            finding_id="find_001",
            cycle=1,
            task_id=1,
            summary="Test finding",
            statistics={'p_value': 0.01},
            figure_paths=["artifacts/cycle_1/figures/task_1_box.png"],
            figure_metadata={'plot_type': 'box_plot_with_points', 'dpi': 300}
        )

        assert finding.figure_paths is not None
        assert len(finding.figure_paths) == 1
        assert finding.figure_metadata['plot_type'] == 'box_plot_with_points'

    def test_finding_serializes_with_figures(self):
        """Test Finding serialization includes figure data."""
        from kosmos.world_model.artifacts import Finding

        finding = Finding(
            finding_id="find_002",
            cycle=1,
            task_id=2,
            summary="Test finding with figures",
            statistics={'r_squared': 0.85},
            figure_paths=["path/to/fig1.png", "path/to/fig2.png"],
            figure_metadata={'caption': 'Test figures'}
        )

        data = finding.to_dict()

        assert 'figure_paths' in data
        assert len(data['figure_paths']) == 2
        assert 'figure_metadata' in data


class TestKosmosFiguresColorScheme:
    """Test that figures use correct kosmos-figures color scheme."""

    def test_colors_match_kosmos_figures(self):
        """Test color constants match kosmos-figures specification."""
        assert COLORS['red'] == '#d7191c'
        assert COLORS['blue'] == '#0072B2'
        assert COLORS['neutral'] == '#abd9e9'
        assert COLORS['blue_dark'] == '#2c7bb6'
        assert COLORS['gray'] == '#808080'
        assert COLORS['black'] == '#000000'


def _protocol(name, description, variables, control_groups=None, statistical_tests=None):
    """Build a real ExperimentProtocol for the code templates.

    The templates read typed protocol fields (random_seed, variable roles and
    bound columns, test specs), so Mock protocols no longer work.
    """
    from kosmos.models.experiment import ExperimentProtocol, ProtocolStep, ResourceRequirements
    from kosmos.models.hypothesis import ExperimentType

    return ExperimentProtocol(
        name=name,
        hypothesis_id="hyp_fig_001",
        experiment_type=ExperimentType.DATA_ANALYSIS,
        domain="test_domain",
        description=description,
        objective="Verify that figure generation code is emitted",
        steps=[
            ProtocolStep(
                step_number=1,
                title="Analyze data",
                description="Run the analysis and save the figure",
                action="Run the analysis",
            )
        ],
        variables=variables,
        control_groups=control_groups or [],
        statistical_tests=statistical_tests or [],
        random_seed=42,
        resource_requirements=ResourceRequirements(),
    )


def _var(name, var_type):
    from kosmos.models.experiment import Variable, VariableType

    return Variable(
        name=name,
        type=VariableType(var_type),
        description=f"Test variable {name} for figure generation",
    )


class TestFigureInCodeTemplates:
    """Code templates no longer embed figure generation.

    Plan items P0-5 (commit 7c832a6) and P0-6 (commit 3be9bc2) deleted the dead
    PublicationVisualizer blocks from the TTest, Correlation and LogLog templates:
    generated code runs in the sandbox, where the kosmos package is not
    installed, so it must be self-contained (numpy/scipy/pandas only). Figures
    are drawn on the host with PublicationVisualizer (tested above). These tests
    pin that contract: no kosmos import, the inline statistics are present, and
    the code compiles.
    """

    def test_ttest_template_is_self_contained(self):
        """T-test template emits inline scipy code and no kosmos/figure import."""
        from kosmos.execution.code_generator import TTestComparisonCodeTemplate
        from kosmos.models.experiment import ControlGroup

        protocol = _protocol(
            name="Test Protocol",
            description="Two-group comparison protocol used to test figure generation code",
            variables={'group': _var('group', 'independent'), 'value': _var('value', 'dependent')},
            control_groups=[
                ControlGroup(
                    name='control',
                    description="Control group for the comparison",
                    variables={'group': 'control'},
                    rationale="Baseline condition for the two-group comparison",
                )
            ],
        )

        template = TTestComparisonCodeTemplate()
        code = template.generate(protocol)

        assert 'kosmos' not in code
        assert 'PublicationVisualizer' not in code
        assert 'ttest_ind' in code
        compile(code, '<ttest_template>', 'exec')

    def test_correlation_template_is_self_contained(self):
        """Correlation template emits inline scipy code and no kosmos/figure import."""
        from kosmos.execution.code_generator import CorrelationAnalysisCodeTemplate
        from kosmos.models.experiment import StatisticalTest, StatisticalTestSpec

        protocol = _protocol(
            name="Correlation Analysis",
            description="Correlation protocol between x and y used to test figure generation",
            variables={'x': _var('x', 'independent'), 'y': _var('y', 'dependent')},
            statistical_tests=[
                StatisticalTestSpec(
                    test_type=StatisticalTest.CORRELATION,
                    description="Pearson correlation between x and y",
                    null_hypothesis="H0: x and y are uncorrelated",
                    variables=['x', 'y'],
                )
            ],
        )

        template = CorrelationAnalysisCodeTemplate()
        code = template.generate(protocol)

        assert 'kosmos' not in code
        assert 'PublicationVisualizer' not in code
        assert 'pearsonr' in code
        compile(code, '<correlation_template>', 'exec')

    def test_log_log_template_is_self_contained(self):
        """Log-log template emits inline numpy/scipy code and no kosmos/figure import."""
        from kosmos.execution.code_generator import LogLogScalingCodeTemplate

        protocol = _protocol(
            name="Power Law Scaling",
            description="Power law scaling analysis of y against x on log-log axes",
            variables={'x': _var('x', 'independent'), 'y': _var('y', 'dependent')},
        )

        template = LogLogScalingCodeTemplate()
        code = template.generate(protocol)

        assert 'kosmos' not in code
        assert 'PublicationVisualizer' not in code
        assert 'log10' in code
        compile(code, '<log_log_template>', 'exec')
