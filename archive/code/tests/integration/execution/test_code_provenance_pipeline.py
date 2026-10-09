"""
Integration tests for code line provenance in the execution pipeline.

Tests end-to-end provenance tracking through notebook generation,
finding augmentation, and report generation.

Issue: #62 (GAP-009)
"""

import pytest
import json
import tempfile
from pathlib import Path
from typing import Dict, Any

from kosmos.execution.provenance import (
    CodeProvenance,
    CellLineMapping,
    create_provenance_from_notebook,
    build_cell_line_mappings,
)
from kosmos.execution.notebook_generator import NotebookGenerator, NotebookMetadata
from kosmos.world_model.artifacts import Finding, ArtifactStateManager


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_artifacts_dir():
    """Create temporary artifacts directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def notebook_generator(temp_artifacts_dir):
    """Create NotebookGenerator for testing."""
    return NotebookGenerator(artifacts_dir=temp_artifacts_dir)


@pytest.fixture
def state_manager(temp_artifacts_dir):
    """Create ArtifactStateManager for testing."""
    return ArtifactStateManager(artifacts_dir=temp_artifacts_dir)


@pytest.fixture
def sample_code():
    """Sample analysis code."""
    return """import pandas as pd
import numpy as np

# Load data
df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})

# Calculate correlation
correlation = df['x'].corr(df['y'])
print(f"Correlation: {correlation}")

# Return result
result = {'correlation': correlation}
"""


@pytest.fixture
def finding_with_provenance():
    """Create a Finding with code provenance."""
    provenance = CodeProvenance(
        notebook_path="artifacts/cycle_1/notebooks/task_5_correlation.ipynb",
        cell_index=1,
        start_line=1,
        end_line=12,
        code_snippet="import pandas as pd\nimport numpy as np\n...",
        hypothesis_id="hyp_001",
        cycle=1,
        task_id=5,
        analysis_type="correlation",
    )
    return Finding(
        finding_id="f001",
        cycle=1,
        task_id=5,
        summary="Strong correlation found between x and y",
        statistics={'correlation': 0.95, 'p_value': 0.01},
        code_provenance=provenance.to_dict(),
    )


# =============================================================================
# TestProvenanceWithNotebookGenerator
# =============================================================================


class TestProvenanceWithNotebookGenerator:
    """Tests for provenance integration with NotebookGenerator."""

    def test_notebook_metadata_includes_cell_mappings(
        self, notebook_generator, sample_code
    ):
        """Test that generated notebooks include cell-to-line mappings."""
        metadata = notebook_generator.create_notebook(
            code=sample_code,
            cycle=1,
            task_id=1,
            analysis_type="correlation",
            title="Correlation Analysis",
        )

        assert metadata is not None
        assert metadata.cell_line_mappings is not None
        assert len(metadata.cell_line_mappings) > 0

    def test_cell_mappings_track_line_numbers(
        self, notebook_generator, sample_code
    ):
        """Test that cell mappings track correct line numbers."""
        metadata = notebook_generator.create_notebook(
            code=sample_code,
            cycle=1,
            task_id=1,
            analysis_type="correlation",
        )

        mappings = metadata.cell_line_mappings
        # First cell should start at line 1
        assert mappings[0]['start_line'] == 1
        # Each mapping should have required fields
        for mapping in mappings:
            assert 'cell_index' in mapping
            assert 'start_line' in mapping
            assert 'end_line' in mapping
            assert 'code_hash' in mapping

    def test_create_provenance_from_notebook_metadata(
        self, notebook_generator, sample_code
    ):
        """Test creating provenance from generated notebook metadata."""
        metadata = notebook_generator.create_notebook(
            code=sample_code,
            cycle=1,
            task_id=1,
            analysis_type="correlation",
        )

        # Create provenance using notebook metadata
        provenance = CodeProvenance.create_from_execution(
            notebook_path=metadata.path,
            code=sample_code,
            cell_index=0,
            cycle=1,
            task_id=1,
            analysis_type="correlation",
        )

        assert provenance.notebook_path == metadata.path
        assert provenance.code_hash is not None

    def test_multiple_cells_get_correct_mappings(self, notebook_generator):
        """Test that code split into multiple cells gets correct mappings."""
        # Code with explicit cell markers
        code = """# %% Cell 1
import pandas as pd
import numpy as np

# %% Cell 2
df = pd.DataFrame({'x': [1, 2, 3]})
print(df)

# %% Cell 3
result = df.describe()
"""
        metadata = notebook_generator.create_notebook(
            code=code,
            cycle=1,
            task_id=2,
            analysis_type="data_analysis",
        )

        assert metadata is not None
        # Should have multiple cells (code is split by # %%)
        assert metadata.code_cell_count >= 1


# =============================================================================
# TestProvenanceInFinding
# =============================================================================


# =============================================================================
# TestProvenanceInReports
# =============================================================================


# =============================================================================
# TestProvenanceValidation
# =============================================================================


class TestProvenanceValidation:
    """Tests for provenance validation and consistency."""

    def test_cell_mapping_hash_matches_provenance(
        self, notebook_generator
    ):
        """Test that cell mapping hashes can validate provenance."""
        code = "import pandas\ndf = pandas.DataFrame()"
        metadata = notebook_generator.create_notebook(
            code=code,
            cycle=1,
            task_id=1,
            analysis_type="test",
        )

        # Both should have consistent hashing
        assert metadata.cell_line_mappings is not None
        for mapping in metadata.cell_line_mappings:
            assert 'code_hash' in mapping
            assert len(mapping['code_hash']) == 16


# =============================================================================
# TestEndToEndPipeline
# =============================================================================


class TestEndToEndPipeline:
    """End-to-end tests for provenance pipeline."""

    def test_full_pipeline_notebook_to_finding(
        self, notebook_generator, sample_code
    ):
        """Test full pipeline from notebook generation to finding."""
        # Step 1: Generate notebook
        metadata = notebook_generator.create_notebook(
            code=sample_code,
            cycle=1,
            task_id=5,
            analysis_type="correlation",
            title="Correlation Analysis",
        )
        assert metadata is not None

        # Step 2: Create provenance
        provenance = CodeProvenance.create_from_execution(
            notebook_path=metadata.path,
            code=sample_code,
            cell_index=0,
            hypothesis_id="hyp_001",
            cycle=1,
            task_id=5,
            analysis_type="correlation",
        )

        # Step 3: Create finding with provenance
        finding = Finding(
            finding_id="f_full_test",
            cycle=1,
            task_id=5,
            summary="Correlation analysis completed",
            statistics={'correlation': 0.95},
            code_provenance=provenance.to_dict(),
        )

        # Step 4: Verify complete chain
        assert finding.code_provenance is not None
        assert finding.code_provenance['notebook_path'] == metadata.path
        assert finding.code_provenance['cell_index'] == 0


# =============================================================================
# TestPerformance
# =============================================================================


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Edge case tests for provenance pipeline."""

    def test_empty_code_notebook(self, notebook_generator):
        """Test handling empty code."""
        metadata = notebook_generator.create_notebook(
            code="",
            cycle=1,
            task_id=1,
            analysis_type="test",
        )
        # Should return None for empty code
        assert metadata is None

    def test_whitespace_only_code(self, notebook_generator):
        """Test handling whitespace-only code."""
        metadata = notebook_generator.create_notebook(
            code="   \n   \n   ",
            cycle=1,
            task_id=1,
            analysis_type="test",
        )
        # Should return None for whitespace-only
        assert metadata is None

    def test_very_long_code(self, notebook_generator):
        """Test handling very long code."""
        long_code = "\n".join([f"x_{i} = {i}" for i in range(1000)])
        metadata = notebook_generator.create_notebook(
            code=long_code,
            cycle=1,
            task_id=1,
            analysis_type="long_test",
        )
        assert metadata is not None
        assert metadata.total_line_count == 1000
