"""
Unit tests for kosmos.world_model.artifacts module.

Tests:
- Finding dataclass: creation, serialization
- Hypothesis dataclass: creation, serialization
- ArtifactStateManager: save/load findings, cycle context, validation
"""

import json
import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from kosmos.world_model.artifacts import (
    Finding,
    Hypothesis,
    ArtifactStateManager
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_finding_dict():
    """Sample finding as dictionary."""
    return {
        'finding_id': 'cycle1_task1',
        'cycle': 1,
        'task_id': 1,
        'summary': 'Found 42 differentially expressed genes',
        'statistics': {'p_value': 0.001, 'n_genes': 42},
        'methods': 'DESeq2 differential expression analysis',
        'interpretation': 'Significant gene expression changes',
        'evidence_type': 'data_analysis',
        'notebook_path': '/path/to/notebook.ipynb',
        'citations': [{'paper_id': 'paper_001'}],
        'scholar_eval': {'overall_score': 0.82, 'passes_threshold': True},
        'metadata': {'domain': 'genomics'}
    }


@pytest.fixture
def sample_hypothesis_dict():
    """Sample hypothesis as dictionary."""
    return {
        'hypothesis_id': 'hyp_001',
        'statement': 'KRAS mutations are associated with poor prognosis',
        'status': 'supported',
        'domain': 'oncology',
        'confidence': 0.85,
        'supporting_evidence': ['cycle1_task1', 'cycle2_task3'],
        'refuting_evidence': [],
        'metadata': {'source': 'literature_review'}
    }


@pytest.fixture
def artifacts_dir(temp_dir):
    """Create artifacts directory for tests."""
    artifacts = temp_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    return artifacts


@pytest.fixture
def state_manager(artifacts_dir):
    """Create ArtifactStateManager instance."""
    return ArtifactStateManager(artifacts_dir=str(artifacts_dir))


@pytest.fixture
def populated_state_manager(state_manager, sample_finding_dict):
    """State manager with some findings already saved."""
    import asyncio

    async def populate():
        # Save findings in multiple cycles
        for cycle in range(1, 4):
            for task_id in range(1, 4):
                finding = sample_finding_dict.copy()
                finding['finding_id'] = f'cycle{cycle}_task{task_id}'
                finding['cycle'] = cycle
                finding['task_id'] = task_id
                await state_manager.save_finding_artifact(cycle, task_id, finding)

    asyncio.get_event_loop().run_until_complete(populate())
    return state_manager


# ============================================================================
# Finding Dataclass Tests
# ============================================================================

class TestFinding:
    """Tests for Finding dataclass."""

    def test_create_from_dict(self, sample_finding_dict):
        """Test Creating Finding from dictionary."""
        finding = Finding.from_dict(sample_finding_dict)

        assert finding.finding_id == 'cycle1_task1'
        assert finding.cycle == 1
        assert finding.summary == 'Found 42 differentially expressed genes'
        assert finding.statistics['p_value'] == 0.001

    def test_to_dict(self, sample_finding_dict):
        """Test converting Finding to dictionary."""
        finding = Finding.from_dict(sample_finding_dict)
        result = finding.to_dict()

        assert result['finding_id'] == 'cycle1_task1'
        assert result['statistics']['n_genes'] == 42
        # timestamp should be added
        assert 'timestamp' in result

    def test_minimal_finding(self):
        """Test creating minimal Finding."""
        finding = Finding(
            finding_id='min_001',
            cycle=1,
            task_id=1,
            summary='Minimal finding',
            statistics={}
        )

        assert finding.finding_id == 'min_001'
        assert finding.methods is None
        assert finding.evidence_type == 'data_analysis'

    def test_finding_timestamp_auto_generated(self):
        """Test that timestamp is auto-generated if not provided."""
        finding = Finding(
            finding_id='test',
            cycle=1,
            task_id=1,
            summary='Test',
            statistics={}
        )
        result = finding.to_dict()

        assert 'timestamp' in result
        # Should be valid ISO format
        datetime.fromisoformat(result['timestamp'])


# ============================================================================
# Hypothesis Dataclass Tests
# ============================================================================

class TestHypothesis:
    """Tests for Hypothesis dataclass."""

    def test_create_from_dict(self, sample_hypothesis_dict):
        """Test creating Hypothesis from dictionary."""
        hypothesis = Hypothesis.from_dict(sample_hypothesis_dict)

        assert hypothesis.hypothesis_id == 'hyp_001'
        assert hypothesis.statement == 'KRAS mutations are associated with poor prognosis'
        assert hypothesis.status == 'supported'
        assert hypothesis.confidence == 0.85

    def test_to_dict(self, sample_hypothesis_dict):
        """Test converting Hypothesis to dictionary."""
        hypothesis = Hypothesis.from_dict(sample_hypothesis_dict)
        result = hypothesis.to_dict()

        assert result['hypothesis_id'] == 'hyp_001'
        assert 'supporting_evidence' in result
        assert len(result['supporting_evidence']) == 2

    def test_minimal_hypothesis(self):
        """Test creating minimal Hypothesis."""
        hypothesis = Hypothesis(
            hypothesis_id='hyp_min',
            statement='Test hypothesis',
            status='unknown'
        )

        assert hypothesis.hypothesis_id == 'hyp_min'
        assert hypothesis.confidence == 0.0
        assert hypothesis.supporting_evidence is None


# ============================================================================
# ArtifactStateManager Tests
# ============================================================================

class TestArtifactStateManager:
    """Tests for ArtifactStateManager class."""

    def test_init_creates_directory(self, temp_dir):
        """Test that initialization creates artifacts directory."""
        artifacts_path = temp_dir / "new_artifacts"
        manager = ArtifactStateManager(artifacts_dir=str(artifacts_path))

        assert artifacts_path.exists()
        assert manager.world_model is None
        assert manager.vector_store is None

    def test_init_with_optional_layers(self, temp_dir):
        """Test initialization with world_model and vector_store."""
        mock_world_model = Mock()
        mock_vector_store = Mock()

        manager = ArtifactStateManager(
            artifacts_dir=str(temp_dir / "artifacts"),
            world_model=mock_world_model,
            vector_store=mock_vector_store
        )

        assert manager.world_model == mock_world_model
        assert manager.vector_store == mock_vector_store


class TestArtifactStateManagerSaveLoad:
    """Tests for save/load operations."""

    @pytest.mark.asyncio
    async def test_save_finding_artifact(self, state_manager, sample_finding_dict, artifacts_dir):
        """Test saving finding artifact."""
        path = await state_manager.save_finding_artifact(
            cycle=1,
            task_id=1,
            finding=sample_finding_dict
        )

        assert path.exists()
        assert path.name == 'task_1_finding.json'

        # Verify content
        with open(path) as f:
            saved = json.load(f)
        assert saved['finding_id'] == 'cycle1_task1'

    @pytest.mark.asyncio
    async def test_save_finding_auto_generates_id(self, state_manager):
        """Test that finding_id is auto-generated if not provided."""
        finding = {
            'summary': 'Test finding',
            'statistics': {}
        }

        await state_manager.save_finding_artifact(cycle=5, task_id=3, finding=finding)

        # Check cache
        assert 'cycle5_task3' in state_manager._findings_cache

    @pytest.mark.asyncio
    async def test_save_finding_caches(self, state_manager, sample_finding_dict):
        """Test that saved findings are cached."""
        await state_manager.save_finding_artifact(1, 1, sample_finding_dict)

        assert 'cycle1_task1' in state_manager._findings_cache
        cached = state_manager._findings_cache['cycle1_task1']
        assert cached.summary == 'Found 42 differentially expressed genes'

    @pytest.mark.asyncio
    async def test_save_hypothesis(self, state_manager, sample_hypothesis_dict, artifacts_dir):
        """Test saving hypothesis."""
        hyp_id = await state_manager.save_hypothesis(sample_hypothesis_dict)

        assert hyp_id == 'hyp_001'

        # Check file exists
        hyp_path = artifacts_dir / "hypotheses" / "hyp_001.json"
        assert hyp_path.exists()

    def test_get_finding_from_cache(self, state_manager, sample_finding_dict):
        """Test retrieving finding from cache."""
        # Add to cache manually
        finding = Finding.from_dict(sample_finding_dict)
        state_manager._findings_cache['cycle1_task1'] = finding

        result = state_manager.get_finding('cycle1_task1')

        assert result is not None
        assert result.finding_id == 'cycle1_task1'

    @pytest.mark.asyncio
    async def test_get_finding_from_file(self, state_manager, sample_finding_dict):
        """Test retrieving finding from file when not in cache."""
        # Save finding
        await state_manager.save_finding_artifact(1, 1, sample_finding_dict)

        # Clear cache
        state_manager._findings_cache.clear()

        # Should load from file
        result = state_manager.get_finding('cycle1_task1')

        assert result is not None
        assert result.summary == 'Found 42 differentially expressed genes'

    def test_get_finding_not_found(self, state_manager):
        """Test get_finding returns None for non-existent finding."""
        result = state_manager.get_finding('nonexistent')

        assert result is None


class TestArtifactStateManagerCycleOperations:
    """Tests for cycle-related operations."""

    @pytest.mark.asyncio
    async def test_get_all_cycle_findings(self, state_manager, sample_finding_dict):
        """Test retrieving all findings from a cycle."""
        # Save multiple findings in cycle 2
        for task_id in range(1, 4):
            finding = sample_finding_dict.copy()
            finding['finding_id'] = f'cycle2_task{task_id}'
            finding['task_id'] = task_id
            await state_manager.save_finding_artifact(2, task_id, finding)

        findings = state_manager.get_all_cycle_findings(2)

        assert len(findings) == 3
        assert all(f.cycle == 2 for f in findings)

    def test_get_all_cycle_findings_empty(self, state_manager):
        """Test get_all_cycle_findings for non-existent cycle."""
        findings = state_manager.get_all_cycle_findings(99)

        assert findings == []

    @pytest.mark.asyncio
    async def test_get_all_findings(self, state_manager, sample_finding_dict):
        """Test retrieving all findings across cycles."""
        # Save findings in multiple cycles
        for cycle in range(1, 4):
            finding = sample_finding_dict.copy()
            finding['finding_id'] = f'cycle{cycle}_task1'
            finding['cycle'] = cycle
            await state_manager.save_finding_artifact(cycle, 1, finding)

        all_findings = state_manager.get_all_findings()

        assert len(all_findings) == 3

    @pytest.mark.asyncio
    async def test_get_validated_findings(self, state_manager):
        """Test retrieving only validated findings."""
        # Save validated finding
        validated = {
            'finding_id': 'validated_001',
            'cycle': 1,
            'task_id': 1,
            'summary': 'Validated finding',
            'statistics': {},
            'scholar_eval': {'overall_score': 0.85, 'passes_threshold': True}
        }
        await state_manager.save_finding_artifact(1, 1, validated)

        # Save non-validated finding
        unvalidated = {
            'finding_id': 'unvalidated_001',
            'cycle': 1,
            'task_id': 2,
            'summary': 'Unvalidated finding',
            'statistics': {},
            'scholar_eval': {'overall_score': 0.5, 'passes_threshold': False}
        }
        await state_manager.save_finding_artifact(1, 2, unvalidated)

        validated_findings = state_manager.get_validated_findings()

        assert len(validated_findings) == 1
        assert validated_findings[0].finding_id == 'validated_001'


class TestArtifactStateManagerContext:
    """Tests for context retrieval."""

    @pytest.mark.asyncio
    async def test_get_cycle_context(self, state_manager, sample_finding_dict):
        """Test getting cycle context."""
        # Save some findings
        for cycle in range(1, 4):
            finding = sample_finding_dict.copy()
            finding['finding_id'] = f'cycle{cycle}_task1'
            finding['cycle'] = cycle
            finding['scholar_eval'] = {'passes_threshold': True}
            await state_manager.save_finding_artifact(cycle, 1, finding)

        context = state_manager.get_cycle_context(cycle=3, lookback=2)

        assert context['cycle'] == 3
        assert 'findings_count' in context
        assert 'recent_findings' in context
        assert 'validated_discoveries' in context
        assert 'statistics' in context

    @pytest.mark.asyncio
    async def test_get_cycle_context_statistics(self, state_manager, sample_finding_dict):
        """Test that context includes correct statistics."""
        # Save findings
        for i in range(5):
            finding = sample_finding_dict.copy()
            finding['finding_id'] = f'cycle1_task{i}'
            finding['task_id'] = i
            finding['scholar_eval'] = {'passes_threshold': i % 2 == 0}  # Half validated
            await state_manager.save_finding_artifact(1, i, finding)

        context = state_manager.get_cycle_context(cycle=2, lookback=2)

        assert context['statistics']['total_findings'] == 5
        # Should have 3 validated (0, 2, 4)
        assert context['statistics']['validated_findings'] == 3


class TestArtifactStateManagerSummary:
    """Tests for summary generation."""

    @pytest.mark.asyncio
    async def test_generate_cycle_summary(self, state_manager, sample_finding_dict, artifacts_dir):
        """Test generating cycle summary."""
        # Save findings with validation
        for task_id in range(1, 4):
            finding = sample_finding_dict.copy()
            finding['finding_id'] = f'cycle1_task{task_id}'
            finding['task_id'] = task_id
            finding['scholar_eval'] = {'passes_threshold': True}
            await state_manager.save_finding_artifact(1, task_id, finding)

        summary = await state_manager.generate_cycle_summary(1)

        assert '# Cycle 1 Summary' in summary
        assert 'Total Findings' in summary
        assert 'Validated Findings' in summary

        # Check file was saved
        summary_path = artifacts_dir / "cycle_1" / "summary.md"
        assert summary_path.exists()

    @pytest.mark.asyncio
    async def test_generate_cycle_summary_empty(self, state_manager):
        """Test generating summary for empty cycle."""
        summary = await state_manager.generate_cycle_summary(99)

        assert '# Cycle 99 Summary' in summary
        assert 'Total Findings**: 0' in summary


class TestArtifactStateManagerExport:
    """Tests for export/import operations."""

    @pytest.mark.asyncio
    async def test_export_artifacts(self, state_manager, sample_finding_dict, temp_dir):
        """Test exporting all artifacts."""
        # Save some findings
        for i in range(3):
            finding = sample_finding_dict.copy()
            finding['finding_id'] = f'finding_{i}'
            await state_manager.save_finding_artifact(1, i, finding)

        export_path = temp_dir / "export.json"
        state_manager.export_artifacts(str(export_path))

        assert export_path.exists()

        with open(export_path) as f:
            exported = json.load(f)

        assert 'findings' in exported
        assert len(exported['findings']) == 3
        assert 'export_timestamp' in exported

    @pytest.mark.asyncio
    async def test_get_statistics(self, state_manager, sample_finding_dict):
        """Test getting statistics."""
        # Save some findings
        for cycle in range(1, 3):
            finding = sample_finding_dict.copy()
            finding['finding_id'] = f'cycle{cycle}_task1'
            finding['cycle'] = cycle
            finding['scholar_eval'] = {'passes_threshold': True}
            await state_manager.save_finding_artifact(cycle, 1, finding)

        stats = state_manager.get_statistics()

        assert stats['total_findings'] == 2
        assert stats['validated_findings'] == 2
        assert stats['validation_rate'] == 1.0
        assert stats['cycles_completed'] == 2


class TestArtifactStateManagerConflictDetection:
    """Tests for conflict detection (Issue #55)."""

    @pytest.mark.asyncio
    async def test_add_finding_with_conflict_check(self, state_manager, sample_finding_dict):
        """Test adding finding with conflict detection."""
        from kosmos.world_model.artifacts import UpdateType, FindingIntegrationResult

        result = await state_manager.add_finding_with_conflict_check(sample_finding_dict)

        # Now returns FindingIntegrationResult (Issue #55)
        assert isinstance(result, FindingIntegrationResult)
        assert result.success is True
        assert result.update_type == UpdateType.CONFIRMATION

        # Finding should be saved
        assert 'cycle1_task1' in state_manager._findings_cache

    @pytest.mark.asyncio
    async def test_conflict_detection_opposite_effects(self, state_manager):
        """Test conflict detection with opposite effect sizes."""
        from kosmos.world_model.artifacts import UpdateType

        # First finding with positive effect
        finding1 = {
            'finding_id': 'conflict_test_1',
            'cycle': 1,
            'task_id': 1,
            'summary': 'Positive effect observed',
            'statistics': {'effect_size': 0.5, 'p_value': 0.01},
            'hypothesis_id': 'hyp-001'
        }
        result1 = await state_manager.add_finding_with_conflict_check(finding1)
        assert result1.update_type == UpdateType.CONFIRMATION

        # Second finding with negative effect (same hypothesis)
        finding2 = {
            'finding_id': 'conflict_test_2',
            'cycle': 1,
            'task_id': 2,
            'summary': 'Negative effect observed',
            'statistics': {'effect_size': -0.3, 'p_value': 0.02},
            'hypothesis_id': 'hyp-001'
        }
        result2 = await state_manager.add_finding_with_conflict_check(finding2)
        assert result2.update_type == UpdateType.CONFLICT
        assert 'contradiction' in result2.reasoning.lower()

    @pytest.mark.asyncio
    async def test_pruning_on_hypothesis_refutation(self, state_manager):
        """Test pruning when hypothesis is refuted."""
        from kosmos.world_model.artifacts import UpdateType

        finding = {
            'finding_id': 'prune_test',
            'cycle': 1,
            'task_id': 1,
            'summary': 'Hypothesis refuted',
            'statistics': {'effect_size': 0.0, 'p_value': 0.95},
            'hypothesis_id': 'hyp-002',
            'refutes_hypothesis': True
        }
        result = await state_manager.add_finding_with_conflict_check(finding)
        assert result.update_type == UpdateType.PRUNING
        assert 'hyp-002' in result.affected_hypotheses


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestArtifactEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_save_finding_with_none_values(self, state_manager):
        """Test saving finding with None values."""
        finding = {
            'finding_id': 'null_test',
            'cycle': 1,
            'task_id': 1,
            'summary': 'Test',
            'statistics': {},
            'methods': None,
            'interpretation': None
        }

        path = await state_manager.save_finding_artifact(1, 1, finding)

        assert path.exists()

    def test_get_unsupported_hypotheses_no_dir(self, state_manager):
        """Test getting unsupported hypotheses when no directory exists."""
        result = state_manager._get_unsupported_hypotheses()

        assert result == []

    @pytest.mark.asyncio
    async def test_save_finding_creates_cycle_directory(self, state_manager, sample_finding_dict, artifacts_dir):
        """Test that saving creates cycle directory."""
        await state_manager.save_finding_artifact(10, 1, sample_finding_dict)

        cycle_dir = artifacts_dir / "cycle_10"
        assert cycle_dir.exists()

    @pytest.mark.asyncio
    async def test_concurrent_saves(self, state_manager, sample_finding_dict):
        """Test concurrent saves to same cycle."""
        import asyncio

        async def save_finding(task_id):
            finding = sample_finding_dict.copy()
            finding['finding_id'] = f'concurrent_{task_id}'
            finding['task_id'] = task_id
            await state_manager.save_finding_artifact(1, task_id, finding)

        await asyncio.gather(*[save_finding(i) for i in range(5)])

        findings = state_manager.get_all_cycle_findings(1)
        assert len(findings) == 5
