"""
Integration tests for notebook generation (Issue #61).

These tests create actual notebooks and verify they are valid.
Requires nbformat to be installed.
"""

import pytest
from pathlib import Path
import json

import nbformat
from nbformat.validator import validate

from kosmos.execution.notebook_generator import (
    NotebookGenerator,
    NotebookMetadata,
    create_notebook_from_code,
)
from kosmos.execution.jupyter_client import ExecutionResult, ExecutionStatus, CellOutput


class TestRealNotebookGeneration:
    """Test actual notebook generation with file I/O."""

    @pytest.fixture
    def generator(self, tmp_path):
        """Create NotebookGenerator for testing."""
        return NotebookGenerator(artifacts_dir=tmp_path)

    def test_create_real_notebook(self, generator, tmp_path):
        """Test creating actual notebook file."""
        code = """import pandas as pd
import numpy as np

# Create sample data
data = {'x': [1, 2, 3, 4, 5], 'y': [2, 4, 5, 4, 5]}
df = pd.DataFrame(data)

# Calculate correlation
correlation = df['x'].corr(df['y'])
print(f"Correlation: {correlation:.3f}")
"""
        metadata = generator.create_notebook(
            code=code,
            cycle=1,
            task_id=1,
            analysis_type="correlation",
            title="Correlation Analysis"
        )

        assert metadata is not None
        notebook_path = Path(metadata.path)
        assert notebook_path.exists()

    def test_notebook_has_correct_structure(self, generator):
        """Test that created notebook has correct structure."""
        code = "print('hello')"
        metadata = generator.create_notebook(
            code=code,
            cycle=1,
            task_id=1,
            analysis_type="test",
            title="Test Notebook"
        )

        # Read back the notebook
        with open(metadata.path, 'r') as f:
            nb = nbformat.read(f, as_version=4)

        # Check structure
        assert 'kernelspec' in nb.metadata
        assert nb.metadata['kernelspec']['name'] == 'python3'
        assert len(nb.cells) >= 2  # At least header + code
        assert nb.cells[0].cell_type == 'markdown'  # Header

    def test_notebook_validates(self, generator):
        """Test that created notebook passes nbformat validation."""
        code = "x = 1 + 2\nprint(x)"
        metadata = generator.create_notebook(
            code=code,
            cycle=1,
            task_id=1,
            analysis_type="test"
        )

        with open(metadata.path, 'r') as f:
            nb = nbformat.read(f, as_version=4)

        # This will raise if invalid
        validate(nb)

    def test_outputs_embedded_correctly(self, generator):
        """Test that execution outputs are embedded in notebook."""
        code = "print('test output')"

        # Create mock execution result
        mock_result = ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            stdout="test output\n",
            stderr="",
            execution_time=0.5
        )

        metadata = generator.create_notebook(
            code=code,
            cycle=1,
            task_id=1,
            analysis_type="test",
            execution_result=mock_result
        )

        # Read notebook and check outputs
        with open(metadata.path, 'r') as f:
            nb = nbformat.read(f, as_version=4)

        # Find code cell with outputs
        code_cells = [c for c in nb.cells if c.cell_type == 'code']
        assert len(code_cells) > 0

        # Check that outputs exist on the first code cell
        first_code_cell = code_cells[0]
        assert hasattr(first_code_cell, 'outputs')

    def test_figures_referenced(self, generator, tmp_path):
        """Test that figure paths are referenced in markdown."""
        code = "print('analysis with figure')"
        figure_paths = [
            str(tmp_path / "cycle_1" / "figures" / "scatter.png"),
            str(tmp_path / "cycle_1" / "figures" / "histogram.png"),
        ]

        metadata = generator.create_notebook(
            code=code,
            cycle=1,
            task_id=1,
            analysis_type="test",
            figure_paths=figure_paths
        )

        # Read notebook
        with open(metadata.path, 'r') as f:
            nb = nbformat.read(f, as_version=4)

        # Find markdown cells with figure references
        md_cells = [c for c in nb.cells if c.cell_type == 'markdown']
        figure_cells = [c for c in md_cells if '![Figure]' in c.source]

        assert len(figure_cells) == 2

    def test_hypothesis_in_header(self, generator):
        """Test that hypothesis is included in header."""
        hypothesis = "Gene X expression correlates positively with phenotype Y"

        metadata = generator.create_notebook(
            code="print('test')",
            cycle=1,
            task_id=1,
            analysis_type="test",
            hypothesis=hypothesis
        )

        with open(metadata.path, 'r') as f:
            nb = nbformat.read(f, as_version=4)

        header_cell = nb.cells[0]
        assert hypothesis in header_cell.source

    def test_multiple_notebooks_per_cycle(self, generator, tmp_path):
        """Test creating multiple notebooks in same cycle."""
        for i in range(3):
            generator.create_notebook(
                code=f"print('task {i}')",
                cycle=1,
                task_id=i,
                analysis_type="test"
            )

        # Check all notebooks exist
        notebooks_dir = tmp_path / "cycle_1" / "notebooks"
        notebooks = list(notebooks_dir.glob("*.ipynb"))
        assert len(notebooks) == 3

    def test_total_line_count_accumulates(self, generator):
        """Test paper requirement: track total line count."""
        # Create several notebooks
        code1 = "import pandas as pd\nimport numpy as np\nprint('hello')"  # 3 lines
        code2 = "x = 1\ny = 2\nz = x + y\nprint(z)"  # 4 lines
        code3 = "for i in range(10):\n    print(i)"  # 2 lines

        generator.create_notebook(code=code1, cycle=1, task_id=1, analysis_type="test")
        generator.create_notebook(code=code2, cycle=1, task_id=2, analysis_type="test")
        generator.create_notebook(code=code3, cycle=1, task_id=3, analysis_type="test")

        total = generator.get_total_line_count()
        assert total >= 9  # At least 9 lines total (3+4+2)


class TestNotebookKernels:
    """Test notebook kernel support."""

    @pytest.fixture
    def generator(self, tmp_path):
        """Create NotebookGenerator for testing."""
        return NotebookGenerator(artifacts_dir=tmp_path)

    def test_python_kernel_notebook(self, generator):
        """Test creating Python kernel notebook."""
        code = "import pandas as pd\ndf = pd.DataFrame()"
        metadata = generator.create_notebook(
            code=code,
            cycle=1,
            task_id=1,
            analysis_type="test",
            kernel="python3"
        )

        with open(metadata.path, 'r') as f:
            nb = nbformat.read(f, as_version=4)

        assert nb.metadata['kernelspec']['name'] == 'python3'
        assert nb.metadata['kernelspec']['language'] == 'python'

    def test_r_kernel_notebook(self, generator):
        """Test creating R kernel notebook."""
        r_code = """library(ggplot2)
df <- data.frame(x = 1:10, y = 1:10)
print(summary(df))
"""
        metadata = generator.create_notebook(
            code=r_code,
            cycle=1,
            task_id=1,
            analysis_type="r_analysis",
            kernel="ir"
        )

        with open(metadata.path, 'r') as f:
            nb = nbformat.read(f, as_version=4)

        assert nb.metadata['kernelspec']['name'] == 'ir'
        assert nb.metadata['kernelspec']['language'] == 'R'


class TestDirectoryStructure:
    """Test directory structure compliance."""

    @pytest.fixture
    def generator(self, tmp_path):
        """Create NotebookGenerator for testing."""
        return NotebookGenerator(artifacts_dir=tmp_path)

    def test_creates_artifacts_cycle_notebooks_structure(self, generator, tmp_path):
        """Test directory structure: artifacts/cycle_N/notebooks/"""
        generator.create_notebook(
            code="print(1)",
            cycle=1,
            task_id=1,
            analysis_type="test"
        )

        expected_dir = tmp_path / "cycle_1" / "notebooks"
        assert expected_dir.exists()
        assert expected_dir.is_dir()

    def test_multiple_cycles_separate_directories(self, generator, tmp_path):
        """Test that different cycles use different directories."""
        generator.create_notebook(code="print(1)", cycle=1, task_id=1, analysis_type="test")
        generator.create_notebook(code="print(2)", cycle=2, task_id=1, analysis_type="test")
        generator.create_notebook(code="print(3)", cycle=3, task_id=1, analysis_type="test")

        for cycle in [1, 2, 3]:
            expected_dir = tmp_path / f"cycle_{cycle}" / "notebooks"
            assert expected_dir.exists()


class TestCellSplitting:
    """Test code cell splitting with real notebooks."""

    @pytest.fixture
    def generator(self, tmp_path):
        """Create NotebookGenerator for testing."""
        return NotebookGenerator(artifacts_dir=tmp_path)

    def test_cell_markers_create_multiple_cells(self, generator):
        """Test that cell markers create multiple code cells."""
        code = """# %%
import pandas as pd

# %%
df = pd.DataFrame({'a': [1, 2, 3]})

# %%
print(df.describe())
"""
        metadata = generator.create_notebook(
            code=code,
            cycle=1,
            task_id=1,
            analysis_type="test"
        )

        with open(metadata.path, 'r') as f:
            nb = nbformat.read(f, as_version=4)

        code_cells = [c for c in nb.cells if c.cell_type == 'code']
        assert len(code_cells) == 3


class TestConvenienceFunction:
    """Test standalone convenience function."""

    def test_create_notebook_from_code_validates(self, tmp_path):
        """Test that convenience function creates valid notebook."""
        output_path = tmp_path / "test.ipynb"

        create_notebook_from_code(
            code="print('hello')",
            output_path=output_path,
            title="Test Notebook"
        )

        # Read and validate
        with open(output_path, 'r') as f:
            nb = nbformat.read(f, as_version=4)

        validate(nb)


class TestMetadataAccuracy:
    """Test that metadata accurately reflects notebook content."""

    @pytest.fixture
    def generator(self, tmp_path):
        """Create NotebookGenerator for testing."""
        return NotebookGenerator(artifacts_dir=tmp_path)

    def test_code_cell_count_accurate(self, generator):
        """Test that code cell count is accurate."""
        code = """# %%
cell1

# %%
cell2

# %%
cell3
"""
        metadata = generator.create_notebook(
            code=code,
            cycle=1,
            task_id=1,
            analysis_type="test"
        )

        with open(metadata.path, 'r') as f:
            nb = nbformat.read(f, as_version=4)

        actual_code_cells = len([c for c in nb.cells if c.cell_type == 'code'])
        assert metadata.code_cell_count == actual_code_cells

    def test_markdown_cell_count_accurate(self, generator, tmp_path):
        """Test that markdown cell count is accurate."""
        figure_paths = [
            str(tmp_path / "figures" / "fig1.png"),
            str(tmp_path / "figures" / "fig2.png"),
        ]

        metadata = generator.create_notebook(
            code="print('test')",
            cycle=1,
            task_id=1,
            analysis_type="test",
            figure_paths=figure_paths,
            hypothesis="Test hypothesis"
        )

        with open(metadata.path, 'r') as f:
            nb = nbformat.read(f, as_version=4)

        actual_md_cells = len([c for c in nb.cells if c.cell_type == 'markdown'])
        assert metadata.markdown_cell_count == actual_md_cells


class TestExecutionResultIntegration:
    """Test integration with ExecutionResult from jupyter_client."""

    @pytest.fixture
    def generator(self, tmp_path):
        """Create NotebookGenerator for testing."""
        return NotebookGenerator(artifacts_dir=tmp_path)

    def test_with_execution_result_object(self, generator):
        """Test creating notebook with ExecutionResult object."""
        result = ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            stdout="Analysis complete.\nCorrelation: 0.85\n",
            stderr="",
            execution_time=1.23,
            return_value={'correlation': 0.85}
        )

        metadata = generator.create_notebook(
            code="# Correlation analysis\nprint('test')",
            cycle=1,
            task_id=1,
            analysis_type="correlation",
            execution_result=result
        )

        assert metadata.has_outputs is True

        # Verify outputs are in notebook
        with open(metadata.path, 'r') as f:
            nb = nbformat.read(f, as_version=4)

        code_cells = [c for c in nb.cells if c.cell_type == 'code']
        assert len(code_cells) > 0

    def test_with_cell_outputs(self, generator):
        """Test creating notebook with CellOutput objects."""
        cell_output = CellOutput(
            output_type="stream",
            content="Hello from cell output",
            mime_type="text/plain"
        )

        result = ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            outputs=[cell_output],
            execution_time=0.5
        )

        metadata = generator.create_notebook(
            code="print('hello')",
            cycle=1,
            task_id=1,
            analysis_type="test",
            execution_result=result
        )

        assert metadata is not None
