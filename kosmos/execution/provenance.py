"""
Code Line Provenance for Kosmos research artifacts.

Implements Issue #62: Code Line Provenance from paper requirements.

Paper Claim (Section 5):
> "Code Citation: Hyperlink to the exact Jupyter notebook and line of code
> that produced the claim"

Features:
- Links findings to specific notebook cells and line numbers
- Generates hyperlinks for reports (notebook.ipynb#cell=N&line=M)
- Tracks code snippets that produced findings
- Supports provenance chain: finding → code → hypothesis
"""

import hashlib
import logging
import platform
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


def get_git_sha() -> Optional[str]:
    """Get current git commit SHA."""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        logger.debug(f"Git commit hash retrieval failed: {e}")
    return None


def _kosmos_version() -> Optional[str]:
    """Installed distribution version, else the package's __version__."""
    import importlib.metadata
    for dist in ("kosmos-ai-scientist", "kosmos"):
        try:
            return importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
    try:
        from kosmos import __version__
        return __version__
    except ImportError:
        return None


def _file_sha256_and_rows(data_path: str) -> tuple:
    """SHA-256 of the file bytes and its line count less a header, or (None, None)."""
    try:
        digest = hashlib.sha256()
        lines = 0
        last = b""
        with open(data_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
                lines += chunk.count(b"\n")
                last = chunk
        if last and not last.endswith(b"\n"):
            lines += 1
        return digest.hexdigest(), max(lines - 1, 0)
    except OSError as e:
        logger.warning(f"Could not hash data file {data_path}: {e}")
        return None, None


def _str_or_none(value: Any) -> Optional[str]:
    return value if isinstance(value, str) else None


def build_run_provenance(
    code: str,
    data_path: Optional[str],
    llm_client: Any,
    seed: Optional[int],
    protocol: Any,
    sandbox_used: bool,
    *,
    code_path: Optional[str] = None,
    template: Optional[str] = None,
) -> Dict[str, Any]:
    """Provenance record stored on every result row.

    Captures what is needed to re-run a result: the code (hash and saved path),
    the data file (hash and row count), the seed, the LLM and the environment.

    Args:
        code: The code that was executed
        data_path: The dataset file, or None for runs without data
        llm_client: The run's LLM client (model, provider_name, temperature_default)
        seed: The run seed
        protocol: The experiment protocol the code was generated from
        sandbox_used: Whether the code ran in the Docker sandbox
        code_path: Where the code was saved
        template: The code template that produced the code; defaults to
            protocol.template_name, else 'llm'
    """
    data_sha256, data_rows = _file_sha256_and_rows(data_path) if data_path else (None, None)
    temperature = getattr(llm_client, "temperature_default", None)
    # Only plain values: the record is stored as JSON
    return {
        "git_sha": get_git_sha(),
        "model": _str_or_none(getattr(llm_client, "model", None)),
        "provider": _str_or_none(getattr(llm_client, "provider_name", None)),
        "temperature": float(temperature) if isinstance(temperature, (int, float)) else None,
        "data_path": str(data_path) if data_path else None,
        "data_sha256": data_sha256,
        "data_rows": data_rows,
        "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        "code_path": str(code_path) if code_path else None,
        "template": _str_or_none(template) or _str_or_none(getattr(protocol, "template_name", None)) or "llm",
        "seed": seed,
        "sandbox_used": bool(sandbox_used),
        "python_version": platform.python_version(),
        "kosmos_version": _kosmos_version(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@dataclass
class CellLineMapping:
    """
    Maps a notebook cell to its line number range.

    Used for tracking cumulative line counts in notebooks to enable
    precise code citations.
    """
    cell_index: int      # 0-based cell index in notebook
    start_line: int      # 1-based start line (cumulative in notebook)
    end_line: int        # 1-based end line (cumulative in notebook)
    cell_type: str       # 'code' or 'markdown'
    line_count: int      # Number of lines in this cell
    code_hash: Optional[str] = None  # SHA256 hash of code content for validation

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CellLineMapping':
        """Create CellLineMapping from dictionary."""
        return cls(**data)

    @staticmethod
    def compute_hash(code: str) -> str:
        """Compute SHA256 hash of code content."""
        return hashlib.sha256(code.encode('utf-8')).hexdigest()[:16]


@dataclass
class CodeProvenance:
    """
    Code provenance for linking findings to source code.

    Enables "Code Citation: Hyperlink to the exact Jupyter notebook and
    line of code that produced the claim" as specified in the paper.

    Example:
        ```python
        from kosmos.execution.provenance import CodeProvenance

        # Create provenance for a finding
        provenance = CodeProvenance(
            notebook_path="artifacts/cycle_1/notebooks/task_5_correlation.ipynb",
            cell_index=3,
            start_line=1,
            end_line=15,
            code_snippet="import pandas as pd\\ndf = pd.read_csv('data.csv')",
            hypothesis_id="hyp_001"
        )

        # Generate hyperlink for reports
        link = provenance.to_hyperlink()
        # "artifacts/cycle_1/notebooks/task_5_correlation.ipynb#cell=3&line=1"

        # Serialize for storage
        data = provenance.to_dict()

        # Reconstruct from storage
        restored = CodeProvenance.from_dict(data)
        ```
    """
    # Required fields
    notebook_path: str           # Path to Jupyter notebook
    cell_index: int              # 0-based cell index in notebook
    start_line: int              # 1-based start line in cell
    end_line: Optional[int]      # 1-based end line (None = single line)
    code_snippet: str            # Relevant code (max ~500 chars)

    # Optional tracking fields
    hypothesis_id: Optional[str] = None   # Link to hypothesis being tested
    execution_id: Optional[str] = None    # Link to execution run
    timestamp: Optional[str] = None       # When code was executed
    code_hash: Optional[str] = None       # Hash of code for validation

    # Additional metadata
    cycle: Optional[int] = None           # Research cycle number
    task_id: Optional[int] = None         # Task ID within cycle
    analysis_type: Optional[str] = None   # Type of analysis (correlation, t-test, etc.)

    # Provenance & reproducibility metadata (WP8)
    seed: Optional[int] = None            # Random seed used
    git_sha: Optional[str] = None         # Git commit SHA
    model: Optional[str] = None           # LLM model used for generation
    temperature: Optional[float] = None   # LLM temperature setting
    data_hash: Optional[str] = None       # SHA256 hash of input data

    def __post_init__(self):
        """Validate and set defaults after initialization."""
        # Truncate code snippet if too long
        max_snippet_length = 500
        if len(self.code_snippet) > max_snippet_length:
            self.code_snippet = self.code_snippet[:max_snippet_length] + "..."

        # Set timestamp if not provided
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

        # Compute code hash if not provided
        if self.code_hash is None and self.code_snippet:
            self.code_hash = hashlib.sha256(
                self.code_snippet.encode('utf-8')
            ).hexdigest()[:16]

        # Auto-populate git SHA if not set
        if self.git_sha is None:
            self.git_sha = get_git_sha()

    def to_hyperlink(self, include_line: bool = True) -> str:
        """
        Generate hyperlink string for reports.

        Format: notebook.ipynb#cell=N&line=M

        This format is compatible with:
        - GitHub/GitLab notebook rendering
        - VS Code Jupyter extension
        - JupyterLab

        Args:
            include_line: Whether to include line number in hyperlink

        Returns:
            Hyperlink string for the code location
        """
        if include_line and self.start_line is not None:
            return f"{self.notebook_path}#cell={self.cell_index}&line={self.start_line}"
        return f"{self.notebook_path}#cell={self.cell_index}"

    def to_markdown_link(self, text: Optional[str] = None) -> str:
        """
        Generate Markdown-formatted hyperlink.

        Args:
            text: Link text (defaults to notebook filename)

        Returns:
            Markdown link string
        """
        if text is None:
            # Extract filename from path
            text = self.notebook_path.split('/')[-1]
        return f"[{text}]({self.to_hyperlink()})"

    def get_line_range_str(self) -> str:
        """
        Get human-readable line range string.

        Returns:
            String like "lines 1-15" or "line 1"
        """
        if self.end_line is None or self.end_line == self.start_line:
            return f"line {self.start_line}"
        return f"lines {self.start_line}-{self.end_line}"

    def get_citation_string(self) -> str:
        """
        Get full citation string for reports.

        Returns:
            String like "task_5_correlation.ipynb, cell 3, lines 1-15"
        """
        filename = self.notebook_path.split('/')[-1]
        line_info = self.get_line_range_str()
        return f"{filename}, cell {self.cell_index}, {line_info}"

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary for JSON storage.

        Returns:
            Dictionary representation of provenance
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeProvenance':
        """
        Deserialize from dictionary.

        Args:
            data: Dictionary with provenance fields

        Returns:
            CodeProvenance instance
        """
        # Handle missing optional fields
        return cls(
            notebook_path=data['notebook_path'],
            cell_index=data['cell_index'],
            start_line=data['start_line'],
            end_line=data.get('end_line'),
            code_snippet=data['code_snippet'],
            hypothesis_id=data.get('hypothesis_id'),
            execution_id=data.get('execution_id'),
            timestamp=data.get('timestamp'),
            code_hash=data.get('code_hash'),
            cycle=data.get('cycle'),
            task_id=data.get('task_id'),
            analysis_type=data.get('analysis_type'),
            seed=data.get('seed'),
            git_sha=data.get('git_sha'),
            model=data.get('model'),
            temperature=data.get('temperature'),
            data_hash=data.get('data_hash'),
        )

    @classmethod
    def create_from_execution(
        cls,
        notebook_path: str,
        code: str,
        cell_index: int = 0,
        hypothesis_id: Optional[str] = None,
        cycle: Optional[int] = None,
        task_id: Optional[int] = None,
        analysis_type: Optional[str] = None,
    ) -> 'CodeProvenance':
        """
        Create provenance from execution context.

        Convenience factory method for creating provenance during
        code execution.

        Args:
            notebook_path: Path to generated notebook
            code: Executed code
            cell_index: Cell index in notebook (default 0)
            hypothesis_id: Optional hypothesis being tested
            cycle: Research cycle number
            task_id: Task ID within cycle
            analysis_type: Type of analysis

        Returns:
            CodeProvenance instance
        """
        lines = code.split('\n')
        line_count = len(lines)

        return cls(
            notebook_path=notebook_path,
            cell_index=cell_index,
            start_line=1,
            end_line=line_count if line_count > 1 else None,
            code_snippet=code[:500],  # Truncated in __post_init__
            hypothesis_id=hypothesis_id,
            execution_id=f"exec_{cycle}_{task_id}" if cycle and task_id else None,
            cycle=cycle,
            task_id=task_id,
            analysis_type=analysis_type,
        )


def create_provenance_from_notebook(
    notebook_path: str,
    code: str,
    cell_index: int,
    hypothesis_id: Optional[str] = None,
    cycle: Optional[int] = None,
    task_id: Optional[int] = None,
    analysis_type: Optional[str] = None,
) -> CodeProvenance:
    """
    Convenience function to create provenance from notebook execution.

    Args:
        notebook_path: Path to Jupyter notebook
        code: Code that was executed
        cell_index: Cell index in notebook
        hypothesis_id: Optional hypothesis ID
        cycle: Research cycle number
        task_id: Task ID
        analysis_type: Type of analysis

    Returns:
        CodeProvenance instance
    """
    return CodeProvenance.create_from_execution(
        notebook_path=notebook_path,
        code=code,
        cell_index=cell_index,
        hypothesis_id=hypothesis_id,
        cycle=cycle,
        task_id=task_id,
        analysis_type=analysis_type,
    )


def build_cell_line_mappings(
    code_cells: List[str],
    start_offset: int = 1
) -> List[CellLineMapping]:
    """
    Build cell-to-line mappings from a list of code cells.

    Creates cumulative line number tracking for precise code citations.

    Args:
        code_cells: List of code strings (one per cell)
        start_offset: Starting line number (default 1 for 1-based)

    Returns:
        List of CellLineMapping objects
    """
    mappings = []
    current_line = start_offset

    for idx, code in enumerate(code_cells):
        lines = code.split('\n')
        line_count = len(lines)
        end_line = current_line + line_count - 1

        mapping = CellLineMapping(
            cell_index=idx,
            start_line=current_line,
            end_line=end_line,
            cell_type='code',
            line_count=line_count,
            code_hash=CellLineMapping.compute_hash(code),
        )
        mappings.append(mapping)

        # Move to next cell's starting line
        current_line = end_line + 1

    return mappings


def get_cell_for_line(
    mappings: List[CellLineMapping],
    line_number: int
) -> Optional[CellLineMapping]:
    """
    Find the cell containing a specific line number.

    Args:
        mappings: List of cell-to-line mappings
        line_number: 1-based line number to find

    Returns:
        CellLineMapping containing the line, or None if not found
    """
    for mapping in mappings:
        if mapping.start_line <= line_number <= mapping.end_line:
            return mapping
    return None
