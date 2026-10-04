"""
Dataset schema for binding protocol variables to real columns.

describe_dataset() reads the user's dataset once and records its columns,
types, categorical levels and a content hash. The experiment designer shows
the schema to the LLM and resolves every variable it returns against
DatasetSchema.columns; the code generator then reads the bound columns.
"""

import difflib
import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Non-numeric columns with at most this many distinct values count as categorical
MAX_CATEGORICAL_LEVELS = 10

_ALLOWED_EXTENSIONS = {'.csv', '.tsv', '.parquet', '.json', '.jsonl', '.xlsx'}
_N_EXAMPLES = 3


@dataclass
class DatasetSchema:
    """Columns, types and levels of one dataset file."""

    path: str
    sha256: str
    n_rows: int
    columns: List[str]
    numeric_columns: List[str]
    categorical_columns: Dict[str, List[str]]  # column -> levels (at most MAX_CATEGORICAL_LEVELS)
    dtypes: Dict[str, str]
    examples: Dict[str, List[Any]] = field(default_factory=dict)

    def is_numeric(self, column: str) -> bool:
        return column in self.numeric_columns

    def resolve_column(self, name: Optional[str]) -> Optional[str]:
        """Map a column name proposed by the LLM to an actual column, or None."""
        return resolve_column(name, self.columns)

    def to_prompt_block(self) -> str:
        """Describe the dataset for an LLM prompt: one line per column."""
        lines = [
            f"Dataset: {Path(self.path).name} ({self.n_rows} rows, {len(self.columns)} columns)",
            "Columns (use these exact names):",
        ]
        for col in self.columns:
            examples = ", ".join(str(v) for v in self.examples.get(col, []))
            if col in self.categorical_columns:
                levels = self.categorical_columns[col]
                kind = f"categorical, {len(levels)} levels: {', '.join(levels)}"
            elif col in self.numeric_columns:
                kind = "numeric"
            else:
                kind = "text"
            line = f"- {col} ({self.dtypes.get(col, 'unknown')}, {kind})"
            if examples:
                line += f"; examples: {examples}"
            lines.append(line)
        return "\n".join(lines)


def _normalize(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', name.lower())


def resolve_column(name: Optional[str], columns: List[str]) -> Optional[str]:
    """
    Resolve a proposed column name against the dataset columns.

    Tries, in order: exact match, case-insensitive match, match after removing
    everything but letters and digits, then difflib with cutoff 0.8.
    """
    if not name or not isinstance(name, str):
        return None
    name = name.strip()
    if name in columns:
        return name
    lower = {c.lower(): c for c in columns}
    if name.lower() in lower:
        return lower[name.lower()]
    normalized = {_normalize(c): c for c in columns}
    if _normalize(name) and _normalize(name) in normalized:
        return normalized[_normalize(name)]
    close = difflib.get_close_matches(name.lower(), list(lower), n=1, cutoff=0.8)
    if close:
        return lower[close[0]]
    return None


def _read_table(path: Path) -> pd.DataFrame:
    """Read a dataset with the same suffix dispatch as DataProvider.get_data."""
    suffix = path.suffix.lower()
    if suffix == '.tsv':
        return pd.read_csv(path, sep='\t')
    if suffix == '.parquet':
        return pd.read_parquet(path)
    if suffix == '.json':
        return pd.read_json(path)
    if suffix == '.jsonl':
        return pd.read_json(path, lines=True)
    if suffix == '.xlsx':
        return pd.read_excel(path)
    return pd.read_csv(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json_scalar(value: Any) -> Any:
    """Convert numpy scalars to plain Python for prompts and JSON."""
    return value.item() if hasattr(value, 'item') else value


def describe_dataset(path: str) -> DatasetSchema:
    """
    Read a dataset file and describe its columns.

    Raises:
        FileNotFoundError: if the file does not exist
        ValueError: if the extension is not a supported table format
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if file_path.suffix.lower() not in _ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported dataset extension '{file_path.suffix}'. "
            f"Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
        )

    df = _read_table(file_path)
    columns = [str(c) for c in df.columns]
    df.columns = columns

    numeric_columns: List[str] = []
    categorical_columns: Dict[str, List[str]] = {}
    dtypes: Dict[str, str] = {}
    examples: Dict[str, List[Any]] = {}
    for col in columns:
        series = df[col]
        dtypes[col] = str(series.dtype)
        non_null = series.dropna()
        examples[col] = [_json_scalar(v) for v in non_null.head(_N_EXAMPLES).tolist()]
        if pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(col)
            continue
        levels = sorted({str(v) for v in non_null.tolist()})
        if len(levels) <= MAX_CATEGORICAL_LEVELS:
            categorical_columns[col] = levels

    schema = DatasetSchema(
        path=str(file_path),
        sha256=_sha256(file_path),
        n_rows=int(len(df)),
        columns=columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        dtypes=dtypes,
        examples=examples,
    )
    logger.info(
        f"Described dataset {file_path.name}: {schema.n_rows} rows, {len(columns)} columns "
        f"({len(numeric_columns)} numeric, {len(categorical_columns)} categorical)"
    )
    return schema
