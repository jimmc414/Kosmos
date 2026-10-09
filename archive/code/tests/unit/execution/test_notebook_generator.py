"""
Unit tests for NotebookGenerator (Issue #61).

Tests notebook creation, cell splitting, output conversion, and metadata tracking.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from kosmos.execution.notebook_generator import (
    NotebookGenerator,
    NotebookMetadata,
    create_notebook_from_code,
    CELL_PATTERN,
)


class TestNotebookMetadata:
    """Test NotebookMetadata dataclass."""

    def test_create_metadata(self):
        """Test creating metadata with required fields."""
        metadata = NotebookMetadata(
            path="/path/to/notebook.ipynb",
            title="Test Analysis",
            cycle=1,
            task_id=5,
            analysis_type="correlation",
            kernel="python3",
            code_cell_count=3,
            markdown_cell_count=2,
            total_line_count=50
        )

        assert metadata.path == "/path/to/notebook.ipynb"
        assert metadata.title == "Test Analysis"
        assert metadata.cycle == 1
        assert metadata.task_id == 5
        assert metadata.analysis_type == "correlation"
        assert metadata.kernel == "python3"
        assert metadata.code_cell_count == 3
        assert metadata.total_line_count == 50

    def test_metadata_defaults(self):
        """Test optional fields have correct defaults."""
        metadata = NotebookMetadata(
            path="/path/test.ipynb",
            title="Test",
            cycle=1,
            task_id=1,
            analysis_type="test",
            kernel="python3",
            code_cell_count=1,
            markdown_cell_count=1,
            total_line_count=10
        )

        assert metadata.execution_time is None
        assert metadata.has_figures is False
        assert metadata.has_outputs is False
        assert metadata.timestamp is None
        assert metadata.hypothesis is None

    def test_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = NotebookMetadata(
            path="/path/test.ipynb",
            title="Test Analysis",
            cycle=2,
            task_id=3,
            analysis_type="t_test",
            kernel="python3",
            code_cell_count=4,
            markdown_cell_count=2,
            total_line_count=100,
            has_figures=True,
            has_outputs=True
        )

        data = metadata.to_dict()

        assert data['path'] == "/path/test.ipynb"
        assert data['title'] == "Test Analysis"
        assert data['cycle'] == 2
        assert data['task_id'] == 3
        assert data['analysis_type'] == "t_test"
        assert data['kernel'] == "python3"
        assert data['code_cell_count'] == 4
        assert data['has_figures'] is True
        assert data['has_outputs'] is True
        # timestamp should be set if None
        assert 'timestamp' in data

    def test_to_dict_preserves_timestamp(self):
        """Test that existing timestamp is preserved."""
        timestamp = "2025-01-01T12:00:00"
        metadata = NotebookMetadata(
            path="/path/test.ipynb",
            title="Test",
            cycle=1,
            task_id=1,
            analysis_type="test",
            kernel="python3",
            code_cell_count=1,
            markdown_cell_count=1,
            total_line_count=10,
            timestamp=timestamp
        )

        data = metadata.to_dict()
        assert data['timestamp'] == timestamp


class TestNotebookGeneratorInit:
    """Test NotebookGenerator initialization."""

    def test_init_with_string_path(self, tmp_path):
        """Test initialization with string path."""
        generator = NotebookGenerator(artifacts_dir=str(tmp_path))

        assert generator.artifacts_dir == tmp_path
        assert generator.default_kernel == "python3"
        assert generator.generated_notebooks == []
        assert generator.total_line_count == 0

    def test_init_with_path_object(self, tmp_path):
        """Test initialization with Path object."""
        generator = NotebookGenerator(artifacts_dir=tmp_path)

        assert generator.artifacts_dir == tmp_path

    def test_init_with_custom_kernel(self, tmp_path):
        """Test initialization with R kernel."""
        generator = NotebookGenerator(artifacts_dir=tmp_path, default_kernel="ir")

        assert generator.default_kernel == "ir"


class TestNotebookGeneratorPathGeneration:
    """Test path generation methods."""

    def test_get_notebooks_dir_creates_directory(self, tmp_path):
        """Test that notebooks directory is created."""
        generator = NotebookGenerator(artifacts_dir=tmp_path)

        notebooks_dir = generator.get_notebooks_dir(cycle=1)

        assert notebooks_dir.exists()
        assert notebooks_dir == tmp_path / "cycle_1" / "notebooks"

    def test_get_notebooks_dir_multiple_cycles(self, tmp_path):
        """Test notebooks directories for different cycles."""
        generator = NotebookGenerator(artifacts_dir=tmp_path)

        dir1 = generator.get_notebooks_dir(cycle=1)
        dir2 = generator.get_notebooks_dir(cycle=2)
        dir3 = generator.get_notebooks_dir(cycle=10)

        assert dir1 == tmp_path / "cycle_1" / "notebooks"
        assert dir2 == tmp_path / "cycle_2" / "notebooks"
        assert dir3 == tmp_path / "cycle_10" / "notebooks"

    def test_get_notebook_path_format(self, tmp_path):
        """Test notebook path format."""
        generator = NotebookGenerator(artifacts_dir=tmp_path)

        path = generator.get_notebook_path(cycle=1, task_id=5, analysis_type="correlation")

        assert path == tmp_path / "cycle_1" / "notebooks" / "task_5_correlation.ipynb"

    def test_get_notebook_path_sanitizes_analysis_type(self, tmp_path):
        """Test that analysis type is sanitized for filename."""
        generator = NotebookGenerator(artifacts_dir=tmp_path)

        path = generator.get_notebook_path(cycle=1, task_id=1, analysis_type="t-test analysis")

        assert "t_test_analysis" in str(path)
        assert ".ipynb" in str(path)


class TestNotebookGeneratorCellSplitting:
    """Test code cell splitting logic."""

    def test_split_code_with_cell_markers(self, tmp_path):
        """Test splitting code with # %% markers."""
        generator = NotebookGenerator(artifacts_dir=tmp_path)

        code = """# %%
import pandas as pd
import numpy as np

# %%
# Load data
df = pd.read_csv('data.csv')

# %%
# Analysis
print(df.describe())
"""
        cells = generator._split_code_into_cells(code)

        assert len(cells) == 3

    def test_split_code_with_jupyter_markers(self, tmp_path):
        """Test splitting code with # In[N]: markers."""
        generator = NotebookGenerator(artifacts_dir=tmp_path)

        code = """# In[1]:
import pandas as pd

# In[2]:
df = pd.read_csv('data.csv')
"""
        cells = generator._split_code_into_cells(code)

        assert len(cells) == 2

    def test_split_code_no_markers_imports_separate(self, tmp_path):
        """Test code without markers separates imports."""
        generator = NotebookGenerator(artifacts_dir=tmp_path)

        code = """import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')
print(df.head())
"""
        cells = generator._split_code_into_cells(code)

        # Should split imports from main code
        assert len(cells) >= 1

    def test_split_code_empty_returns_original(self, tmp_path):
        """Test that empty splits return original code."""
        generator = NotebookGenerator(artifacts_dir=tmp_path)

        code = "print('hello')"
        cells = generator._split_code_into_cells(code)

        assert len(cells) == 1
        assert cells[0] == code

    def test_cell_pattern_matches_markers(self):
        """Test regex pattern matches cell markers."""
        assert CELL_PATTERN.search("# %% cell")
        assert CELL_PATTERN.search("# In[1]:")
        assert CELL_PATTERN.search("# CELL: Analysis")
        assert CELL_PATTERN.search("# ---")
        assert not CELL_PATTERN.search("# Regular comment")


class TestNotebookGeneratorOutputConversion:
    """Test execution result to output conversion."""

    def test_convert_execution_result_with_stdout(self, tmp_path):
        """Test converting execution result with stdout."""
        generator = NotebookGenerator(artifacts_dir=tmp_path)

        mock_result = Mock()
        mock_result.stdout = "Hello, World!\n"
        mock_result.stderr = ""
        mock_result.return_value = None
        mock_result.error_message = None
        mock_result.outputs = []

        outputs = generator._convert_execution_result(mock_result)

        assert len(outputs) == 1
        assert outputs[0]['output_type'] == 'stream'
        assert outputs[0]['name'] == 'stdout'
        assert outputs[0]['text'] == "Hello, World!\n"

    def test_convert_execution_result_with_stderr(self, tmp_path):
        """Test converting execution result with stderr."""
        generator = NotebookGenerator(artifacts_dir=tmp_path)

        mock_result = Mock()
        mock_result.stdout = ""
        mock_result.stderr = "Warning: deprecated\n"
        mock_result.return_value = None
        mock_result.error_message = None
        mock_result.outputs = []

        outputs = generator._convert_execution_result(mock_result)

        assert len(outputs) == 1
        assert outputs[0]['output_type'] == 'stream'
        assert outputs[0]['name'] == 'stderr'

    def test_convert_execution_result_with_return_value(self, tmp_path):
        """Test converting execution result with return value."""
        generator = NotebookGenerator(artifacts_dir=tmp_path)

        mock_result = Mock()
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_result.return_value = {"correlation": 0.85}
        mock_result.error_message = None
        mock_result.outputs = []

        outputs = generator._convert_execution_result(mock_result)

        assert len(outputs) == 1
        assert outputs[0]['output_type'] == 'execute_result'
        assert "correlation" in outputs[0]['data']['text/plain']

    def test_convert_execution_result_with_error(self, tmp_path):
        """Test converting execution result with error."""
        generator = NotebookGenerator(artifacts_dir=tmp_path)

        mock_result = Mock()
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_result.return_value = None
        mock_result.error_message = "ZeroDivisionError"
        mock_result.error_traceback = "Traceback..."
        mock_result.outputs = []

        outputs = generator._convert_execution_result(mock_result)

        assert len(outputs) == 1
        assert outputs[0]['output_type'] == 'error'
        assert outputs[0]['evalue'] == "ZeroDivisionError"

    def test_convert_cell_output_stream(self, tmp_path):
        """Test converting CellOutput stream type."""
        generator = NotebookGenerator(artifacts_dir=tmp_path)

        mock_output = Mock()
        mock_output.output_type = 'stream'
        mock_output.content = 'Output text'

        result = generator._convert_cell_output(mock_output)

        assert result['output_type'] == 'stream'
        assert result['text'] == 'Output text'

    def test_convert_cell_output_execute_result(self, tmp_path):
        """Test converting CellOutput execute_result type."""
        generator = NotebookGenerator(artifacts_dir=tmp_path)

        mock_output = Mock()
        mock_output.output_type = 'execute_result'
        mock_output.content = '42'

        result = generator._convert_cell_output(mock_output)

        assert result['output_type'] == 'execute_result'
        assert result['data']['text/plain'] == '42'


class TestNotebookGeneratorCreation:
    """Test notebook creation."""

    def test_create_notebook_basic(self, tmp_path):
        """Test creating a basic notebook."""
        generator = NotebookGenerator(artifacts_dir=tmp_path)

        code = "print('Hello, World!')"
        metadata = generator.create_notebook(
            code=code,
            cycle=1,
            task_id=1,
            analysis_type="test",
            title="Test Notebook"
        )

        assert metadata is not None
        assert metadata.cycle == 1
        assert metadata.task_id == 1
        assert Path(metadata.path).exists()

    def test_create_notebook_tracks_line_count(self, tmp_path):
        """Test that line count is tracked."""
        generator = NotebookGenerator(artifacts_dir=tmp_path)

        code = """import pandas as pd
import numpy as np

df = pd.DataFrame()
print(df)"""

        metadata = generator.create_notebook(
            code=code,
            cycle=1,
            task_id=1,
            analysis_type="test"
        )

        assert metadata.total_line_count > 0
        assert generator.get_total_line_count() > 0

    def test_create_notebook_with_execution_result(self, tmp_path):
        """Test creating notebook with execution result."""
        from types import SimpleNamespace

        generator = NotebookGenerator(artifacts_dir=tmp_path)

        # Use SimpleNamespace instead of Mock for cleaner attribute access
        mock_result = SimpleNamespace(
            stdout="Output here",
            stderr="",
            return_value=None,
            error_message=None,
            error_traceback=None,
            outputs=[]
        )

        metadata = generator.create_notebook(
            code="print('test')",
            cycle=1,
            task_id=1,
            analysis_type="test",
            execution_result=mock_result
        )

        assert metadata is not None
        assert metadata.has_outputs is True

    def test_create_notebook_with_figures(self, tmp_path):
        """Test creating notebook with figure references."""
        generator = NotebookGenerator(artifacts_dir=tmp_path)

        figure_paths = [
            str(tmp_path / "cycle_1" / "figures" / "fig1.png"),
            str(tmp_path / "cycle_1" / "figures" / "fig2.png"),
        ]

        metadata = generator.create_notebook(
            code="print('analysis')",
            cycle=1,
            task_id=1,
            analysis_type="test",
            figure_paths=figure_paths
        )

        assert metadata is not None
        assert metadata.has_figures is True
        assert metadata.markdown_cell_count >= 3  # Header + 2 figure cells

    def test_create_notebook_with_hypothesis(self, tmp_path):
        """Test creating notebook with hypothesis."""
        generator = NotebookGenerator(artifacts_dir=tmp_path)

        hypothesis = "Gene X expression correlates with phenotype Y"

        metadata = generator.create_notebook(
            code="print('test')",
            cycle=1,
            task_id=1,
            analysis_type="correlation",
            title="Correlation Analysis",
            hypothesis=hypothesis
        )

        assert metadata is not None
        assert metadata.hypothesis == hypothesis

    def test_create_notebook_empty_code_returns_none(self, tmp_path):
        """Test that empty code returns None."""
        generator = NotebookGenerator(artifacts_dir=tmp_path)

        metadata = generator.create_notebook(
            code="",
            cycle=1,
            task_id=1,
            analysis_type="test"
        )

        assert metadata is None

    def test_create_notebook_whitespace_only_returns_none(self, tmp_path):
        """Test that whitespace-only code returns None."""
        generator = NotebookGenerator(artifacts_dir=tmp_path)

        metadata = generator.create_notebook(
            code="   \n\t  \n  ",
            cycle=1,
            task_id=1,
            analysis_type="test"
        )

        assert metadata is None

    def test_create_notebook_with_r_kernel(self, tmp_path):
        """Test creating notebook with R kernel."""
        generator = NotebookGenerator(artifacts_dir=tmp_path)

        r_code = """library(ggplot2)
df <- data.frame(x=1:10, y=1:10)
print(summary(df))
"""
        metadata = generator.create_notebook(
            code=r_code,
            cycle=1,
            task_id=1,
            analysis_type="r_analysis",
            kernel="ir"
        )

        assert metadata is not None
        assert metadata.kernel == "ir"


class TestNotebookGeneratorTracking:
    """Test notebook tracking methods."""

    def test_get_notebook_count(self, tmp_path):
        """Test getting notebook count."""
        generator = NotebookGenerator(artifacts_dir=tmp_path)

        assert generator.get_notebook_count() == 0

        generator.create_notebook(code="print(1)", cycle=1, task_id=1, analysis_type="test")
        generator.create_notebook(code="print(2)", cycle=1, task_id=2, analysis_type="test")
        generator.create_notebook(code="print(3)", cycle=2, task_id=1, analysis_type="test")

        assert generator.get_notebook_count() == 3

    def test_get_notebooks_for_cycle(self, tmp_path):
        """Test getting notebooks for specific cycle."""
        generator = NotebookGenerator(artifacts_dir=tmp_path)

        generator.create_notebook(code="print(1)", cycle=1, task_id=1, analysis_type="test")
        generator.create_notebook(code="print(2)", cycle=1, task_id=2, analysis_type="test")
        generator.create_notebook(code="print(3)", cycle=2, task_id=1, analysis_type="test")

        cycle1_notebooks = generator.get_notebooks_for_cycle(1)
        cycle2_notebooks = generator.get_notebooks_for_cycle(2)
        cycle3_notebooks = generator.get_notebooks_for_cycle(3)

        assert len(cycle1_notebooks) == 2
        assert len(cycle2_notebooks) == 1
        assert len(cycle3_notebooks) == 0

    def test_get_notebook_paths(self, tmp_path):
        """Test getting all notebook paths."""
        generator = NotebookGenerator(artifacts_dir=tmp_path)

        generator.create_notebook(code="print(1)", cycle=1, task_id=1, analysis_type="test")
        generator.create_notebook(code="print(2)", cycle=1, task_id=2, analysis_type="test")

        paths = generator.get_notebook_paths()

        assert len(paths) == 2
        assert all(path.endswith('.ipynb') for path in paths)

    def test_get_total_line_count_accumulates(self, tmp_path):
        """Test that line count accumulates across notebooks."""
        generator = NotebookGenerator(artifacts_dir=tmp_path)

        assert generator.get_total_line_count() == 0

        # First notebook
        code1 = "line1\nline2\nline3"  # 3 lines
        generator.create_notebook(code=code1, cycle=1, task_id=1, analysis_type="test")
        count1 = generator.get_total_line_count()

        # Second notebook
        code2 = "a\nb\nc\nd\ne"  # 5 lines
        generator.create_notebook(code=code2, cycle=1, task_id=2, analysis_type="test")
        count2 = generator.get_total_line_count()

        assert count2 > count1
        assert count2 == count1 + 5


class TestNotebookGeneratorSerialization:
    """Test serialization methods."""

    def test_to_dict(self, tmp_path):
        """Test serializing generator state."""
        generator = NotebookGenerator(artifacts_dir=tmp_path)

        generator.create_notebook(code="print(1)", cycle=1, task_id=1, analysis_type="test")
        generator.create_notebook(code="print(2)", cycle=1, task_id=2, analysis_type="analysis")

        data = generator.to_dict()

        assert data['artifacts_dir'] == str(tmp_path)
        assert data['default_kernel'] == 'python3'
        assert data['notebook_count'] == 2
        assert data['total_line_count'] > 0
        assert len(data['notebooks']) == 2

    def test_to_dict_empty(self, tmp_path):
        """Test serializing empty generator."""
        generator = NotebookGenerator(artifacts_dir=tmp_path)

        data = generator.to_dict()

        assert data['notebook_count'] == 0
        assert data['total_line_count'] == 0
        assert data['notebooks'] == []


class TestNotebookGeneratorKernelSpecs:
    """Test kernel specification methods."""

    def test_get_kernel_spec_python(self, tmp_path):
        """Test Python kernel spec."""
        generator = NotebookGenerator(artifacts_dir=tmp_path)

        spec = generator._get_kernel_spec("python3")

        assert spec['name'] == 'python3'
        assert spec['display_name'] == 'Python 3'
        assert spec['language'] == 'python'

    def test_get_kernel_spec_r(self, tmp_path):
        """Test R kernel spec."""
        generator = NotebookGenerator(artifacts_dir=tmp_path)

        spec = generator._get_kernel_spec("ir")

        assert spec['name'] == 'ir'
        assert spec['display_name'] == 'R'
        assert spec['language'] == 'R'

    def test_get_language_info_python(self, tmp_path):
        """Test Python language info."""
        generator = NotebookGenerator(artifacts_dir=tmp_path)

        info = generator._get_language_info("python3")

        assert info['name'] == 'python'
        assert info['file_extension'] == '.py'

    def test_get_language_info_r(self, tmp_path):
        """Test R language info."""
        generator = NotebookGenerator(artifacts_dir=tmp_path)

        info = generator._get_language_info("ir")

        assert info['name'] == 'R'
        assert info['file_extension'] == '.r'


class TestConvenienceFunction:
    """Test standalone convenience function."""

    def test_create_notebook_from_code(self, tmp_path):
        """Test creating notebook with convenience function."""
        output_path = tmp_path / "test.ipynb"

        result = create_notebook_from_code(
            code="print('hello')",
            output_path=output_path,
            title="Test"
        )

        assert result == output_path
        assert output_path.exists()

    def test_create_notebook_from_code_creates_parent_dir(self, tmp_path):
        """Test that parent directories are created."""
        output_path = tmp_path / "nested" / "dir" / "test.ipynb"

        result = create_notebook_from_code(
            code="print('hello')",
            output_path=output_path,
            title="Test"
        )

        assert result.exists()


class TestRelativeFigurePath:
    """Test figure path resolution."""

    def test_relative_figure_path_from_figures_dir(self, tmp_path):
        """Test relative path calculation from figures directory."""
        generator = NotebookGenerator(artifacts_dir=tmp_path)

        fig_path = str(tmp_path / "cycle_1" / "figures" / "plot.png")
        rel_path = generator._get_relative_figure_path(fig_path, cycle=1)

        assert rel_path == "../figures/plot.png"

    def test_relative_figure_path_absolute(self, tmp_path):
        """Test that non-figures paths are returned as-is."""
        generator = NotebookGenerator(artifacts_dir=tmp_path)

        fig_path = "/some/other/path/image.png"
        rel_path = generator._get_relative_figure_path(fig_path, cycle=1)

        assert rel_path == fig_path
