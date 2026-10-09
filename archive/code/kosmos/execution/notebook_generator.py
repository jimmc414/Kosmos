"""
Notebook Generator for Kosmos research artifacts.

Creates Jupyter notebooks from executed code with embedded outputs.
Implements Issue #61: Jupyter Notebook Generation from paper requirements.

Paper Claim (Section 5):
> "Code Repository: All ~42,000 lines of executable Python code generated
> during the run (Jupyter notebooks)"

Features:
- Creates .ipynb files from executed code
- Embeds execution outputs (stdout, results, errors)
- References generated figures
- Tracks total line count across all notebooks
- Supports Python and R kernels
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime

import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell, new_output

logger = logging.getLogger(__name__)


# Cell marker patterns for splitting code into cells
CELL_MARKERS = [
    r'^# %%',           # VS Code / Spyder cell marker
    r'^# In\[\d*\]:',   # Jupyter cell marker
    r'^# CELL:',        # Explicit cell marker
    r'^# ---',          # Alternate marker
]

# Compiled pattern for cell splitting
CELL_PATTERN = re.compile('|'.join(CELL_MARKERS), re.MULTILINE)


@dataclass
class NotebookMetadata:
    """
    Metadata for a generated Jupyter notebook.

    Tracks information about the notebook for reporting and provenance.
    """
    path: str
    title: str
    cycle: int
    task_id: int
    analysis_type: str
    kernel: str  # 'python3' or 'ir' (R)
    code_cell_count: int
    markdown_cell_count: int
    total_line_count: int
    execution_time: Optional[float] = None
    has_figures: bool = False
    has_outputs: bool = False
    timestamp: Optional[str] = None
    hypothesis: Optional[str] = None
    # Issue #62: Cell-to-line mapping for provenance
    cell_line_mappings: Optional[List[Dict]] = None  # [{cell_index, start_line, end_line, code_hash}]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        if self.timestamp is None:
            data['timestamp'] = datetime.now().isoformat()
        return data


class NotebookGenerator:
    """
    Manages Jupyter notebook creation and storage for research artifacts.

    Creates complete notebooks with code, outputs, and figure references.
    Follows FigureManager pattern from Issue #60.

    Example:
        ```python
        from kosmos.execution.notebook_generator import NotebookGenerator

        generator = NotebookGenerator(artifacts_dir=Path("./artifacts"))

        # Create notebook from executed code
        metadata = generator.create_notebook(
            code="import pandas as pd\\nprint('Hello')",
            execution_result=result,
            cycle=1,
            task_id=3,
            analysis_type="correlation",
            title="Correlation Analysis"
        )

        # Get total line count (paper claims ~42,000)
        print(f"Total lines: {generator.get_total_line_count()}")
        ```
    """

    def __init__(
        self,
        artifacts_dir: Union[str, Path],
        default_kernel: str = "python3"
    ):
        """
        Initialize NotebookGenerator.

        Args:
            artifacts_dir: Base directory for artifacts
            default_kernel: Default kernel for notebooks ('python3' or 'ir')
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.default_kernel = default_kernel
        self.generated_notebooks: List[NotebookMetadata] = []
        self.total_line_count: int = 0

        logger.info(f"NotebookGenerator initialized with artifacts_dir={artifacts_dir}")

    def get_notebooks_dir(self, cycle: int) -> Path:
        """
        Get notebooks directory for a specific cycle.

        Creates the directory if it doesn't exist.

        Args:
            cycle: Research cycle number

        Returns:
            Path to notebooks directory
        """
        notebooks_dir = self.artifacts_dir / f"cycle_{cycle}" / "notebooks"
        notebooks_dir.mkdir(parents=True, exist_ok=True)
        return notebooks_dir

    def get_notebook_path(
        self,
        cycle: int,
        task_id: int,
        analysis_type: str,
        suffix: str = "ipynb"
    ) -> Path:
        """
        Generate unique notebook path.

        Format: artifacts/cycle_N/notebooks/task_M_analysistype.ipynb

        Args:
            cycle: Research cycle number
            task_id: Task identifier
            analysis_type: Type of analysis
            suffix: File extension (default: "ipynb")

        Returns:
            Path to notebook file
        """
        notebooks_dir = self.get_notebooks_dir(cycle)
        # Sanitize analysis type for filename (replace non-alphanumeric with underscore)
        safe_type = re.sub(r'[^\w]', '_', analysis_type.lower())
        filename = f"task_{task_id}_{safe_type}.{suffix}"
        return notebooks_dir / filename

    def create_notebook(
        self,
        code: str,
        cycle: int,
        task_id: int,
        analysis_type: str,
        execution_result: Optional[Any] = None,
        title: Optional[str] = None,
        hypothesis: Optional[str] = None,
        figure_paths: Optional[List[str]] = None,
        kernel: Optional[str] = None,
        execution_time: Optional[float] = None
    ) -> Optional[NotebookMetadata]:
        """
        Create a complete Jupyter notebook with code and outputs.

        Args:
            code: Python/R code to include in notebook
            cycle: Research cycle number
            task_id: Task identifier
            analysis_type: Type of analysis (e.g., "correlation", "t_test")
            execution_result: Optional ExecutionResult with outputs
            title: Optional notebook title
            hypothesis: Optional hypothesis being tested
            figure_paths: Optional list of figure paths to reference
            kernel: Kernel to use ('python3' or 'ir'), defaults to default_kernel
            execution_time: Optional execution time in seconds

        Returns:
            NotebookMetadata if successful, None otherwise
        """
        if not code or not code.strip():
            logger.warning("Cannot create notebook with empty code")
            return None

        kernel = kernel or self.default_kernel

        try:
            # Create new notebook
            nb = new_notebook()
            nb.metadata['kernelspec'] = self._get_kernel_spec(kernel)
            nb.metadata['language_info'] = self._get_language_info(kernel)

            markdown_count = 0
            code_cell_count = 0
            line_count = 0
            # Issue #62: Track cell-to-line mappings for provenance
            cell_line_mappings = []
            cumulative_line = 1  # 1-based line numbering

            # Add header markdown cell
            header_content = self._create_header(
                title=title or f"Analysis {task_id}",
                hypothesis=hypothesis,
                analysis_type=analysis_type,
                cycle=cycle,
                task_id=task_id
            )
            nb.cells.append(new_markdown_cell(header_content))
            markdown_count += 1

            # Split code into cells and add
            code_cells = self._split_code_into_cells(code)

            for i, cell_code in enumerate(code_cells):
                if not cell_code.strip():
                    continue

                cell = new_code_cell(cell_code)

                # Add outputs to the first code cell if execution_result provided
                if i == 0 and execution_result is not None:
                    cell.outputs = self._convert_execution_result(execution_result)

                nb.cells.append(cell)
                cell_lines = len(cell_code.splitlines())

                # Issue #62: Build cell-to-line mapping for provenance
                import hashlib
                code_hash = hashlib.sha256(cell_code.encode('utf-8')).hexdigest()[:16]
                cell_line_mappings.append({
                    'cell_index': code_cell_count,  # 0-based index within code cells
                    'notebook_cell_index': len(nb.cells) - 1,  # Actual cell index in notebook
                    'start_line': cumulative_line,
                    'end_line': cumulative_line + cell_lines - 1,
                    'line_count': cell_lines,
                    'cell_type': 'code',
                    'code_hash': code_hash,
                })

                code_cell_count += 1
                line_count += cell_lines
                cumulative_line += cell_lines

            # Add figure references as markdown cells
            if figure_paths:
                for fig_path in figure_paths:
                    # Make path relative to notebook location
                    rel_path = self._get_relative_figure_path(fig_path, cycle)
                    fig_cell = new_markdown_cell(
                        f"## Generated Figure\n\n![Figure]({rel_path})"
                    )
                    nb.cells.append(fig_cell)
                    markdown_count += 1

            # Write notebook
            output_path = self.get_notebook_path(cycle, task_id, analysis_type)
            with open(output_path, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)

            # Update total line count
            self.total_line_count += line_count

            # Create metadata
            metadata = NotebookMetadata(
                path=str(output_path),
                title=title or f"Analysis {task_id}",
                cycle=cycle,
                task_id=task_id,
                analysis_type=analysis_type,
                kernel=kernel,
                code_cell_count=code_cell_count,
                markdown_cell_count=markdown_count,
                total_line_count=line_count,
                execution_time=execution_time,
                has_figures=bool(figure_paths),
                has_outputs=execution_result is not None,
                timestamp=datetime.now().isoformat(),
                hypothesis=hypothesis,
                # Issue #62: Include cell-to-line mappings for provenance
                cell_line_mappings=cell_line_mappings if cell_line_mappings else None,
            )

            self.generated_notebooks.append(metadata)
            logger.info(f"Generated notebook: {output_path} ({line_count} lines, {code_cell_count} code cells)")

            return metadata

        except Exception as e:
            logger.error(f"Failed to create notebook: {e}")
            return None

    def _get_kernel_spec(self, kernel: str) -> Dict[str, str]:
        """Get Jupyter kernel specification."""
        if kernel == 'ir':
            return {
                'name': 'ir',
                'display_name': 'R',
                'language': 'R'
            }
        else:
            return {
                'name': 'python3',
                'display_name': 'Python 3',
                'language': 'python'
            }

    def _get_language_info(self, kernel: str) -> Dict[str, Any]:
        """Get language info for notebook metadata."""
        if kernel == 'ir':
            return {
                'name': 'R',
                'file_extension': '.r',
                'mimetype': 'text/x-r-source'
            }
        else:
            return {
                'name': 'python',
                'version': '3.11',
                'file_extension': '.py',
                'mimetype': 'text/x-python',
                'codemirror_mode': {'name': 'ipython', 'version': 3}
            }

    def _create_header(
        self,
        title: str,
        hypothesis: Optional[str],
        analysis_type: str,
        cycle: int,
        task_id: int
    ) -> str:
        """Create header markdown for notebook."""
        lines = [f"# {title}", ""]

        if hypothesis:
            lines.extend([f"**Hypothesis**: {hypothesis}", ""])

        lines.extend([
            f"**Analysis Type**: {analysis_type.replace('_', ' ').title()}",
            f"**Cycle**: {cycle}",
            f"**Task ID**: {task_id}",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            ""
        ])

        return '\n'.join(lines)

    def _split_code_into_cells(self, code: str) -> List[str]:
        """
        Split code into logical cells.

        Splits on cell markers (# %%, # In[N]:, # CELL:) or keeps
        as single cell if no markers found.

        Args:
            code: Code string to split

        Returns:
            List of code cell contents
        """
        # Check if code has cell markers
        if CELL_PATTERN.search(code):
            # Split on cell markers, keeping non-empty cells
            parts = CELL_PATTERN.split(code)
            cells = [p.strip() for p in parts if p.strip()]
            return cells if cells else [code]

        # No markers - try to split on major sections
        # Look for import blocks, function definitions, main code
        cells = []
        current_cell = []
        in_imports = True

        for line in code.splitlines():
            stripped = line.strip()

            # Check for transition from imports to main code
            if in_imports and stripped and not stripped.startswith(('import ', 'from ', '#')):
                if current_cell:
                    cells.append('\n'.join(current_cell))
                    current_cell = []
                in_imports = False

            # Check for function/class definitions (start new cell)
            if stripped.startswith(('def ', 'class ', 'async def ')):
                if current_cell:
                    cells.append('\n'.join(current_cell))
                    current_cell = []

            current_cell.append(line)

        # Add remaining code
        if current_cell:
            cells.append('\n'.join(current_cell))

        # If no splitting happened, return original as single cell
        return cells if cells else [code]

    def _convert_execution_result(self, execution_result: Any) -> List:
        """
        Convert ExecutionResult to nbformat cell outputs.

        Args:
            execution_result: ExecutionResult object with outputs

        Returns:
            List of nbformat NotebookNode outputs for notebook cell
        """
        outputs = []

        # Handle ExecutionResult object
        if hasattr(execution_result, 'stdout') and execution_result.stdout:
            outputs.append(new_output(
                output_type='stream',
                name='stdout',
                text=execution_result.stdout
            ))

        if hasattr(execution_result, 'stderr') and execution_result.stderr:
            outputs.append(new_output(
                output_type='stream',
                name='stderr',
                text=execution_result.stderr
            ))

        if hasattr(execution_result, 'return_value') and execution_result.return_value is not None:
            outputs.append(new_output(
                output_type='execute_result',
                data={'text/plain': str(execution_result.return_value)},
                metadata={},
                execution_count=1
            ))

        if hasattr(execution_result, 'error_message') and execution_result.error_message:
            outputs.append(new_output(
                output_type='error',
                ename='ExecutionError',
                evalue=execution_result.error_message,
                traceback=[execution_result.error_traceback or '']
            ))

        # Handle CellOutput objects
        if hasattr(execution_result, 'outputs'):
            for cell_output in execution_result.outputs:
                if hasattr(cell_output, 'output_type'):
                    output = self._convert_cell_output(cell_output)
                    if output:
                        outputs.append(output)

        return outputs

    def _convert_cell_output(self, cell_output: Any) -> Optional[Any]:
        """
        Convert a CellOutput object to nbformat NotebookNode output.

        Args:
            cell_output: CellOutput object

        Returns:
            nbformat NotebookNode output or None
        """
        output_type = getattr(cell_output, 'output_type', 'stream')
        content = getattr(cell_output, 'content', '')

        if output_type == 'stream':
            return new_output(
                output_type='stream',
                name='stdout',
                text=content
            )
        elif output_type == 'execute_result':
            return new_output(
                output_type='execute_result',
                data={'text/plain': content},
                metadata={},
                execution_count=1
            )
        elif output_type == 'error':
            return new_output(
                output_type='error',
                ename='Error',
                evalue=content,
                traceback=[]
            )
        elif output_type == 'display_data':
            mime_type = getattr(cell_output, 'mime_type', 'text/plain')
            return new_output(
                output_type='display_data',
                data={mime_type: content},
                metadata={}
            )

        return None

    def _get_relative_figure_path(self, figure_path: str, cycle: int) -> str:
        """
        Get relative path to figure from notebook location.

        Args:
            figure_path: Absolute or artifact-relative figure path
            cycle: Current cycle number

        Returns:
            Relative path string
        """
        fig_path = Path(figure_path)

        # If path contains 'figures', construct relative path
        if 'figures' in fig_path.parts:
            # Path from notebooks/ to figures/ is ../figures/
            return f"../figures/{fig_path.name}"

        # Otherwise return as-is
        return str(figure_path)

    def get_notebook_count(self) -> int:
        """Get total number of generated notebooks."""
        return len(self.generated_notebooks)

    def get_notebooks_for_cycle(self, cycle: int) -> List[NotebookMetadata]:
        """Get all notebooks generated for a specific cycle."""
        return [nb for nb in self.generated_notebooks if nb.cycle == cycle]

    def get_notebook_paths(self) -> List[str]:
        """Get list of all generated notebook paths."""
        return [nb.path for nb in self.generated_notebooks]

    def get_total_line_count(self) -> int:
        """
        Get total lines of code across all notebooks.

        Paper claims ~42,000 lines generated per run.

        Returns:
            Total line count
        """
        return self.total_line_count

    def to_dict(self) -> Dict[str, Any]:
        """Serialize notebook generator state to dictionary."""
        return {
            'artifacts_dir': str(self.artifacts_dir),
            'default_kernel': self.default_kernel,
            'notebook_count': self.get_notebook_count(),
            'total_line_count': self.total_line_count,
            'notebooks': [nb.to_dict() for nb in self.generated_notebooks]
        }


def create_notebook_from_code(
    code: str,
    output_path: Union[str, Path],
    title: str = "Analysis",
    kernel: str = "python3",
    execution_result: Optional[Any] = None
) -> Path:
    """
    Convenience function to create a standalone notebook.

    Args:
        code: Code to include in notebook
        output_path: Path to save notebook
        title: Notebook title
        kernel: Kernel to use
        execution_result: Optional execution result with outputs

    Returns:
        Path to created notebook
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nb = new_notebook()
    nb.metadata['kernelspec'] = {
        'name': kernel,
        'display_name': 'Python 3' if kernel == 'python3' else 'R'
    }

    # Add title cell
    nb.cells.append(new_markdown_cell(f"# {title}"))

    # Add code cell
    cell = new_code_cell(code)
    nb.cells.append(cell)

    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    return output_path
