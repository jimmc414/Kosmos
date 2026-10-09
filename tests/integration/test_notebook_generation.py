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

from kosmos.execution.jupyter_client import ExecutionResult, ExecutionStatus, CellOutput


class TestFindingIntegration:
    """Test integration with Finding dataclass."""

    def test_finding_with_notebook_path(self):
        """Test Finding includes notebook_path."""
        from kosmos.world_model.artifacts import Finding

        finding = Finding(
            finding_id="find_001",
            cycle=1,
            task_id=1,
            summary="Test finding",
            statistics={'p_value': 0.01},
            notebook_path="artifacts/cycle_1/notebooks/task_1_test.ipynb"
        )

        assert finding.notebook_path is not None
        assert "task_1_test.ipynb" in finding.notebook_path

    def test_finding_with_notebook_metadata(self):
        """Test Finding includes notebook_metadata."""
        from kosmos.world_model.artifacts import Finding

        finding = Finding(
            finding_id="find_002",
            cycle=1,
            task_id=2,
            summary="Test finding with notebook",
            statistics={'correlation': 0.85},
            notebook_path="artifacts/cycle_1/notebooks/task_2_correlation.ipynb",
            notebook_metadata={
                'kernel': 'python3',
                'line_count': 50,
                'cell_count': 5
            }
        )

        assert finding.notebook_metadata is not None
        assert finding.notebook_metadata['kernel'] == 'python3'
        assert finding.notebook_metadata['line_count'] == 50

    def test_finding_serializes_with_notebook(self):
        """Test Finding serialization includes notebook data."""
        from kosmos.world_model.artifacts import Finding

        finding = Finding(
            finding_id="find_003",
            cycle=1,
            task_id=3,
            summary="Test serialization",
            statistics={},
            notebook_path="path/to/notebook.ipynb",
            notebook_metadata={'kernel': 'python3'}
        )

        data = finding.to_dict()

        assert 'notebook_path' in data
        assert 'notebook_metadata' in data
        assert data['notebook_path'] == "path/to/notebook.ipynb"
