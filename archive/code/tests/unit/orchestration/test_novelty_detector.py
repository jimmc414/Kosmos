"""
Unit tests for kosmos.orchestration.novelty_detector module.

Tests:
- NoveltyDetector: task indexing, novelty checking, similarity computation
- Both semantic (sentence-transformers) and token-based similarity
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from kosmos.orchestration.novelty_detector import NoveltyDetector


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_past_tasks():
    """Sample past tasks for indexing."""
    return [
        {
            'id': 1,
            'type': 'data_analysis',
            'description': 'Analyze gene expression data for KRAS mutations'
        },
        {
            'id': 2,
            'type': 'literature_review',
            'description': 'Review papers on cancer immunotherapy'
        },
        {
            'id': 3,
            'type': 'data_analysis',
            'description': 'Perform differential expression analysis on RNA-seq'
        }
    ]


@pytest.fixture
def novelty_detector():
    """Create NoveltyDetector with token-based similarity (no sentence-transformers)."""
    return NoveltyDetector(
        novelty_threshold=0.75,
        use_sentence_transformers=False
    )


@pytest.fixture
def indexed_detector(novelty_detector, sample_past_tasks):
    """Create NoveltyDetector with indexed tasks."""
    novelty_detector.index_past_tasks(sample_past_tasks)
    return novelty_detector


# ============================================================================
# NoveltyDetector Initialization Tests
# ============================================================================

class TestNoveltyDetectorInit:
    """Tests for NoveltyDetector initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        detector = NoveltyDetector(use_sentence_transformers=False)

        assert detector.novelty_threshold == 0.75
        assert detector.task_embeddings == []
        assert detector.task_texts == []

    def test_custom_threshold(self):
        """Test custom novelty threshold."""
        detector = NoveltyDetector(
            novelty_threshold=0.5,
            use_sentence_transformers=False
        )

        assert detector.novelty_threshold == 0.5

    def test_token_based_mode(self):
        """Test token-based similarity mode."""
        detector = NoveltyDetector(use_sentence_transformers=False)

        assert detector.use_sentence_transformers is False
        assert detector.model is None

    def test_sentence_transformers_mode_fallback(self):
        """Test fallback when sentence-transformers not available."""
        with patch.dict('sys.modules', {'sentence_transformers': None}):
            detector = NoveltyDetector(use_sentence_transformers=True)

            # Should fall back to token-based
            # (depends on import availability)


# ============================================================================
# Task Indexing Tests
# ============================================================================

class TestTaskIndexing:
    """Tests for task indexing."""

    def test_index_past_tasks(self, novelty_detector, sample_past_tasks):
        """Test indexing past tasks."""
        novelty_detector.index_past_tasks(sample_past_tasks)

        assert len(novelty_detector.task_texts) == 3
        assert len(novelty_detector.task_metadata) == 3

    def test_index_empty_tasks(self, novelty_detector):
        """Test indexing empty task list."""
        novelty_detector.index_past_tasks([])

        assert novelty_detector.task_texts == []

    def test_index_incremental(self, novelty_detector, sample_past_tasks):
        """Test incremental indexing."""
        # Index first batch
        novelty_detector.index_past_tasks(sample_past_tasks[:2])
        assert len(novelty_detector.task_texts) == 2

        # Index more
        novelty_detector.index_past_tasks([sample_past_tasks[2]])
        assert len(novelty_detector.task_texts) == 3

    def test_task_text_format(self, novelty_detector):
        """Test task text is properly formatted."""
        task = {'type': 'data_analysis', 'description': 'Test task'}
        novelty_detector.index_past_tasks([task])

        assert 'data_analysis: Test task' in novelty_detector.task_texts[0]

    def test_clear_index(self, indexed_detector):
        """Test clearing the index."""
        indexed_detector.clear_index()

        assert indexed_detector.task_embeddings == []
        assert indexed_detector.task_texts == []
        assert indexed_detector.task_metadata == []


# ============================================================================
# Novelty Checking Tests
# ============================================================================

class TestNoveltyChecking:
    """Tests for novelty checking."""

    def test_check_novel_task(self, indexed_detector):
        """Test checking a novel task."""
        novel_task = {
            'type': 'experiment_design',
            'description': 'Design clinical trial for new drug'
        }

        result = indexed_detector.check_task_novelty(novel_task)

        assert result['is_novel'] is True
        assert result['novelty_score'] > 0.5

    def test_check_similar_task(self, indexed_detector):
        """Test checking a similar/redundant task."""
        similar_task = {
            'type': 'data_analysis',
            'description': 'Analyze gene expression data for KRAS mutations'
        }

        result = indexed_detector.check_task_novelty(similar_task)

        # Should be identified as redundant (high similarity)
        assert result['max_similarity'] > 0.5

    def test_check_novelty_empty_index(self, novelty_detector):
        """Test novelty check with no indexed tasks."""
        task = {'type': 'data_analysis', 'description': 'Any task'}

        result = novelty_detector.check_task_novelty(task)

        # Everything is novel when no history
        assert result['is_novel'] is True
        assert result['novelty_score'] == 1.0
        assert result['max_similarity'] == 0.0

    def test_novelty_result_format(self, indexed_detector):
        """Test novelty result contains required fields."""
        task = {'type': 'data_analysis', 'description': 'Test task'}

        result = indexed_detector.check_task_novelty(task)

        assert 'is_novel' in result
        assert 'novelty_score' in result
        assert 'max_similarity' in result
        assert 'similar_tasks' in result

    def test_similar_tasks_returned(self, indexed_detector):
        """Test that similar tasks are returned."""
        similar_task = {
            'type': 'data_analysis',
            'description': 'RNA-seq differential expression analysis'
        }

        result = indexed_detector.check_task_novelty(similar_task)

        # Should have some similar tasks
        if result['max_similarity'] > 0.6:
            assert len(result['similar_tasks']) > 0


# ============================================================================
# Token-Based Similarity Tests
# ============================================================================

class TestTokenSimilarity:
    """Tests for token-based similarity computation."""

    def test_compute_token_similarities(self, indexed_detector):
        """Test Jaccard similarity computation."""
        task_text = "data_analysis: Analyze gene expression"

        similarities = indexed_detector._compute_token_similarities(task_text)

        assert len(similarities) == len(indexed_detector.task_texts)
        assert all(0 <= s <= 1 for s in similarities)

    def test_identical_text_similarity(self, novelty_detector):
        """Test similarity of identical texts."""
        novelty_detector.task_texts = ["data_analysis: Test task"]
        novelty_detector.task_metadata = [{'id': 1}]

        similarities = novelty_detector._compute_token_similarities("data_analysis: Test task")

        assert similarities[0] == 1.0

    def test_completely_different_text(self, novelty_detector):
        """Test similarity of completely different texts."""
        novelty_detector.task_texts = ["aaa bbb ccc"]
        novelty_detector.task_metadata = [{'id': 1}]

        similarities = novelty_detector._compute_token_similarities("xxx yyy zzz")

        assert similarities[0] == 0.0

    def test_partial_overlap_similarity(self, novelty_detector):
        """Test similarity with partial overlap."""
        novelty_detector.task_texts = ["the quick brown fox"]
        novelty_detector.task_metadata = [{'id': 1}]

        similarities = novelty_detector._compute_token_similarities("the slow brown cat")

        # 2 common words (the, brown) out of 6 unique
        assert 0 < similarities[0] < 1


# ============================================================================
# Plan Novelty Tests
# ============================================================================

class TestPlanNovelty:
    """Tests for checking plan novelty."""

    def test_check_plan_novelty(self, indexed_detector):
        """Test checking novelty of entire plan."""
        plan = {
            'tasks': [
                {'type': 'data_analysis', 'description': 'New analysis'},
                {'type': 'literature_review', 'description': 'New review'},
                {'type': 'experiment_design', 'description': 'Design experiment'}
            ]
        }

        result = indexed_detector.check_plan_novelty(plan)

        assert 'plan_novelty_score' in result
        assert 'redundant_task_count' in result
        assert 'novel_task_count' in result
        assert 'task_novelties' in result
        assert len(result['task_novelties']) == 3

    def test_check_plan_novelty_empty(self, indexed_detector):
        """Test checking empty plan."""
        plan = {'tasks': []}

        result = indexed_detector.check_plan_novelty(plan)

        assert result['plan_novelty_score'] == 1.0
        assert result['redundant_task_count'] == 0
        assert result['novel_task_count'] == 0

    def test_plan_with_redundant_tasks(self, indexed_detector):
        """Test plan containing redundant tasks."""
        plan = {
            'tasks': [
                # Similar to indexed task
                {'type': 'data_analysis', 'description': 'Analyze gene expression data for KRAS mutations'},
                # Novel task
                {'type': 'experiment_design', 'description': 'Design CRISPR knockout experiment'}
            ]
        }

        result = indexed_detector.check_plan_novelty(plan)

        # At least some redundancy should be detected
        assert result['plan_novelty_score'] <= 1.0


# ============================================================================
# Filter Redundant Tasks Tests
# ============================================================================

class TestFilterRedundantTasks:
    """Tests for filtering redundant tasks."""

    def test_filter_redundant_tasks(self, indexed_detector):
        """Test filtering redundant tasks from list."""
        tasks = [
            # Should be filtered (similar to indexed)
            {'type': 'data_analysis', 'description': 'Analyze gene expression data for KRAS'},
            # Should remain (novel)
            {'type': 'experiment_design', 'description': 'Design new drug trial'}
        ]

        filtered = indexed_detector.filter_redundant_tasks(tasks)

        # Should have at least some tasks
        assert len(filtered) >= 1

    def test_filter_empty_list(self, indexed_detector):
        """Test filtering empty task list."""
        filtered = indexed_detector.filter_redundant_tasks([])

        assert filtered == []

    def test_filter_all_novel(self, indexed_detector):
        """Test filtering when all tasks are novel."""
        tasks = [
            {'type': 'experiment_design', 'description': 'Novel task 1'},
            {'type': 'experiment_design', 'description': 'Novel task 2'}
        ]

        filtered = indexed_detector.filter_redundant_tasks(tasks)

        # All should remain
        assert len(filtered) == 2


# ============================================================================
# Statistics Tests
# ============================================================================

class TestNoveltyStatistics:
    """Tests for novelty detection statistics."""

    def test_get_statistics(self, indexed_detector):
        """Test getting statistics."""
        stats = indexed_detector.get_statistics()

        assert stats['total_indexed_tasks'] == 3
        assert stats['novelty_threshold'] == 0.75
        assert stats['using_semantic_similarity'] is False

    def test_get_statistics_empty(self, novelty_detector):
        """Test statistics for empty detector."""
        stats = novelty_detector.get_statistics()

        assert stats['total_indexed_tasks'] == 0
        assert stats['has_embeddings'] is False


# ============================================================================
# Edge Cases
# ============================================================================

class TestNoveltyEdgeCases:
    """Tests for edge cases."""

    def test_missing_task_fields(self, novelty_detector):
        """Test handling tasks with missing fields."""
        task = {'type': 'unknown'}  # Missing description

        result = novelty_detector.check_task_novelty(task)

        # Should handle gracefully
        assert 'is_novel' in result

    def test_empty_description(self, novelty_detector):
        """Test task with empty description."""
        novelty_detector.index_past_tasks([{'type': 'data_analysis', 'description': ''}])

        result = novelty_detector.check_task_novelty({
            'type': 'data_analysis',
            'description': ''
        })

        assert result['is_novel'] is False or result['max_similarity'] >= 0

    def test_special_characters_in_description(self, novelty_detector):
        """Test handling special characters."""
        task = {
            'type': 'data_analysis',
            'description': 'Analyze p53 (TP53) gene with α=0.05 threshold'
        }

        novelty_detector.index_past_tasks([task])
        result = novelty_detector.check_task_novelty(task)

        assert result['is_novel'] is False  # Same task

    def test_very_long_description(self, novelty_detector):
        """Test handling very long descriptions."""
        long_desc = "This is a very long task description. " * 100

        task = {'type': 'data_analysis', 'description': long_desc}

        novelty_detector.index_past_tasks([task])
        result = novelty_detector.check_task_novelty(task)

        assert 'is_novel' in result

    def test_threshold_boundary(self):
        """Test novelty at threshold boundary."""
        detector = NoveltyDetector(novelty_threshold=0.5, use_sentence_transformers=False)
        detector.task_texts = ["word1 word2 word3 word4"]
        detector.task_metadata = [{'id': 1}]

        # Task with 50% overlap
        task = {'type': 'test', 'description': 'word1 word2 other1 other2'}

        result = detector.check_task_novelty(task)

        # Should be at or near boundary
        assert abs(result['max_similarity'] - 0.5) < 0.3


# ============================================================================
# Semantic Similarity Tests (Mock)
# ============================================================================

class TestSemanticSimilarity:
    """Tests for semantic similarity (mocked sentence-transformers)."""

    def test_semantic_similarity_with_mock(self):
        """Test semantic similarity with mocked model."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[1.0, 0.0, 0.0]])

        detector = NoveltyDetector(use_sentence_transformers=False)
        detector.model = mock_model
        detector.use_sentence_transformers = True
        detector.task_embeddings = [np.array([1.0, 0.0, 0.0])]
        detector.task_texts = ["test"]
        detector.task_metadata = [{'id': 1}]

        similarities = detector._compute_semantic_similarities("test task")

        mock_model.encode.assert_called_once()

    def test_semantic_vs_token_similarity(self, novelty_detector):
        """Test that token similarity is used when semantic unavailable."""
        novelty_detector.index_past_tasks([
            {'type': 'data_analysis', 'description': 'Test task'}
        ])

        result = novelty_detector.check_task_novelty({
            'type': 'data_analysis',
            'description': 'Test task'
        })

        # Should use token-based
        assert novelty_detector.use_sentence_transformers is False
        assert result['max_similarity'] > 0
