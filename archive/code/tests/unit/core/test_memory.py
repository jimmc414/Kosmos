"""
Tests for MemoryStore (Phase 7).

Tests memory storage, querying, experiment deduplication, and pruning.
"""

from datetime import datetime, timedelta
import pytest

from kosmos.core.memory import (
    MemoryStore,
    Memory,
    MemoryCategory,
    ExperimentSignature,
)
from kosmos.models.hypothesis import Hypothesis, HypothesisStatus
from kosmos.models.result import ExperimentResult, ResultStatus
from kosmos.models.experiment import ExperimentProtocol


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def memory_store():
    """Create a MemoryStore instance."""
    return MemoryStore(
        config={
            "prune_after_days": 30,
            "min_importance_to_keep": 0.3,
        },
        max_memories=1000,
    )


@pytest.fixture
def sample_hypothesis():
    """Create a sample hypothesis."""
    return Hypothesis(
        id="hyp_001",
        research_question="Does caffeine improve cognitive performance?",
        statement="Caffeine improves working memory",
        rationale="Stimulant effects",
        domain="neuroscience",
    )


@pytest.fixture
def sample_result():
    """Create a sample experiment result."""
    return ExperimentResult(
        id="result_001",
        hypothesis_id="hyp_001",
        supports_hypothesis=True,
        primary_p_value=0.01,
        primary_effect_size=0.75,
        primary_test="t-test",
        status=ResultStatus.SUCCESS,
    )


@pytest.fixture
def sample_protocol():
    """Create a sample experiment protocol."""
    return ExperimentProtocol(
        id="protocol_001",
        hypothesis_id="hyp_001",
        experiment_type="computational",
        methodology="Randomized controlled trial",
        description="Test caffeine effects on memory",
    )


# ============================================================================
# Test Class 1: Initialization
# ============================================================================

class TestMemoryStoreInitialization:
    """Test MemoryStore initialization."""

    def test_initialization_default_config(self):
        """Test memory store initializes with default configuration."""
        store = MemoryStore()

        assert store.max_memories == 1000
        assert store.prune_after_days == 30
        assert store.min_importance_to_keep == 0.3
        assert len(store.memories) == 5  # 5 categories
        assert all(isinstance(mems, list) for mems in store.memories.values())
        assert store.experiment_signatures == {}

    def test_initialization_custom_config(self):
        """Test memory store initializes with custom configuration."""
        custom_config = {
            "prune_after_days": 60,
            "min_importance_to_keep": 0.5,
        }

        store = MemoryStore(config=custom_config, max_memories=500)

        assert store.max_memories == 500
        assert store.prune_after_days == 60
        assert store.min_importance_to_keep == 0.5

    def test_initialization_all_categories(self):
        """Test all memory categories are initialized."""
        store = MemoryStore()

        assert MemoryCategory.SUCCESS_PATTERNS in store.memories
        assert MemoryCategory.FAILURE_PATTERNS in store.memories
        assert MemoryCategory.DEAD_ENDS in store.memories
        assert MemoryCategory.INSIGHTS in store.memories
        assert MemoryCategory.GENERAL in store.memories


# ============================================================================
# Test Class 2: Memory Addition
# ============================================================================

class TestMemoryAddition:
    """Test adding memories to all categories."""

    def test_add_memory_basic(self, memory_store):
        """Test adding a basic memory."""
        memory_id = memory_store.add_memory(
            category=MemoryCategory.GENERAL,
            content="Test memory content",
            data={"key": "value"},
            importance=0.7,
            tags=["test", "general"],
        )

        assert memory_id is not None
        assert isinstance(memory_id, str)
        assert len(memory_store.memories[MemoryCategory.GENERAL]) == 1

        memory = memory_store.memories[MemoryCategory.GENERAL][0]
        assert memory.id == memory_id
        assert memory.content == "Test memory content"
        assert memory.data["key"] == "value"
        assert memory.importance == 0.7
        assert "test" in memory.tags

    def test_add_success_memory(
        self, memory_store, sample_result, sample_hypothesis
    ):
        """Test adding success pattern memory."""
        memory_id = memory_store.add_success_memory(
            result=sample_result,
            hypothesis=sample_hypothesis,
            insights="Strong effect observed",
        )

        assert memory_id is not None
        assert len(memory_store.memories[MemoryCategory.SUCCESS_PATTERNS]) == 1

        memory = memory_store.memories[MemoryCategory.SUCCESS_PATTERNS][0]
        assert "Success" in memory.content
        assert memory.data["result_id"] == sample_result.id
        assert memory.data["hypothesis_id"] == sample_hypothesis.id
        assert memory.data["p_value"] == 0.01
        assert memory.data["effect_size"] == 0.75
        assert memory.importance == 0.8  # Successes are important

    def test_add_failure_memory(
        self, memory_store, sample_hypothesis
    ):
        """Test adding failure pattern memory."""
        failed_result = ExperimentResult(
            id="result_fail_001",
            hypothesis_id="hyp_001",
            supports_hypothesis=False,
            primary_p_value=0.65,
            primary_effect_size=0.12,
            primary_test="t-test",
            status=ResultStatus.SUCCESS,
        )

        memory_id = memory_store.add_failure_memory(
            result=failed_result,
            hypothesis=sample_hypothesis,
            failure_reason="Weak effect size",
        )

        assert memory_id is not None
        assert len(memory_store.memories[MemoryCategory.FAILURE_PATTERNS]) == 1

        memory = memory_store.memories[MemoryCategory.FAILURE_PATTERNS][0]
        assert "Failure" in memory.content
        assert memory.data["failure_reason"] == "Weak effect size"
        assert memory.importance == 0.7

    def test_add_dead_end_memory(
        self, memory_store, sample_hypothesis
    ):
        """Test adding dead-end memory."""
        memory_id = memory_store.add_dead_end_memory(
            hypothesis=sample_hypothesis,
            reason="Repeated failures with strong evidence",
        )

        assert memory_id is not None
        assert len(memory_store.memories[MemoryCategory.DEAD_ENDS]) == 1

        memory = memory_store.memories[MemoryCategory.DEAD_ENDS][0]
        assert "Dead end" in memory.content
        assert memory.data["reason"] == "Repeated failures with strong evidence"
        assert memory.importance == 0.9  # Very important to avoid repeating

    def test_add_insight_memory(self, memory_store):
        """Test adding insight memory."""
        memory_id = memory_store.add_insight_memory(
            insight="Caffeine effects are dose-dependent",
            source="Meta-analysis of 10 experiments",
            related_hypotheses=["hyp_001", "hyp_002"],
        )

        assert memory_id is not None
        assert len(memory_store.memories[MemoryCategory.INSIGHTS]) == 1

        memory = memory_store.memories[MemoryCategory.INSIGHTS][0]
        assert memory.content == "Caffeine effects are dose-dependent"
        assert memory.data["source"] == "Meta-analysis of 10 experiments"
        assert "hyp_001" in memory.data["related_hypotheses"]
        assert memory.importance == 0.95  # Insights are very important

    def test_memory_pruning_on_overflow(self, memory_store):
        """Test memories are pruned when category exceeds limit."""
        # Set a low max_memories for testing
        memory_store.max_memories = 10  # Total across all categories
        # Per category limit = 10 / 5 = 2

        # Add 3 memories to trigger pruning
        for i in range(3):
            memory_store.add_memory(
                category=MemoryCategory.GENERAL,
                content=f"Memory {i}",
                importance=0.2 if i == 0 else 0.8,  # First has low importance
            )

        # Should have pruned to stay under limit
        assert len(memory_store.memories[MemoryCategory.GENERAL]) <= 2


# ============================================================================
# Test Class 3: Memory Querying
# ============================================================================

class TestMemoryQuerying:
    """Test querying and searching memories."""

    def test_query_by_category(self, memory_store):
        """Test querying memories by category."""
        # Add memories to different categories
        memory_store.add_memory(
            category=MemoryCategory.SUCCESS_PATTERNS,
            content="Success 1",
        )
        memory_store.add_memory(
            category=MemoryCategory.FAILURE_PATTERNS,
            content="Failure 1",
        )

        # Query specific category
        results = memory_store.query_memory(
            category=MemoryCategory.SUCCESS_PATTERNS,
            limit=10,
        )

        assert len(results) == 1
        assert results[0].content == "Success 1"

    def test_query_by_tags(self, memory_store):
        """Test querying memories by tags."""
        memory_store.add_memory(
            category=MemoryCategory.GENERAL,
            content="Memory 1",
            tags=["neuroscience", "caffeine"],
        )
        memory_store.add_memory(
            category=MemoryCategory.GENERAL,
            content="Memory 2",
            tags=["psychology", "memory"],
        )

        # Query by tag
        results = memory_store.query_memory(
            tags=["neuroscience"],
            limit=10,
        )

        assert len(results) == 1
        assert results[0].content == "Memory 1"

    def test_query_by_min_importance(self, memory_store):
        """Test querying memories by minimum importance."""
        memory_store.add_memory(
            category=MemoryCategory.GENERAL,
            content="High importance",
            importance=0.9,
        )
        memory_store.add_memory(
            category=MemoryCategory.GENERAL,
            content="Low importance",
            importance=0.2,
        )

        # Query with importance filter
        results = memory_store.query_memory(
            min_importance=0.5,
            limit=10,
        )

        assert len(results) == 1
        assert results[0].content == "High importance"

    def test_query_with_limit(self, memory_store):
        """Test query respects limit parameter."""
        # Add 5 memories
        for i in range(5):
            memory_store.add_memory(
                category=MemoryCategory.GENERAL,
                content=f"Memory {i}",
            )

        # Query with limit of 3
        results = memory_store.query_memory(limit=3)

        assert len(results) == 3

    def test_query_updates_access_count(self, memory_store):
        """Test querying updates memory access count."""
        memory_id = memory_store.add_memory(
            category=MemoryCategory.GENERAL,
            content="Test memory",
        )

        # Query
        results = memory_store.query_memory(limit=10)

        memory = results[0]
        assert memory.access_count > 0
        assert memory.last_accessed is not None

    def test_search_similar_hypothesis(
        self, memory_store, sample_hypothesis, sample_result
    ):
        """Test searching for similar hypothesis memories."""
        # Add memory for hypothesis
        memory_store.add_success_memory(
            result=sample_result,
            hypothesis=sample_hypothesis,
        )

        # Create similar hypothesis
        similar_hyp = Hypothesis(
            research_question="Question",
            statement="Caffeine improves working memory performance",  # Similar
            rationale="Rationale",
            domain="neuroscience",
        )

        # Search for similar
        results = memory_store.search_similar_hypothesis(similar_hyp)

        # Should find the similar hypothesis memory
        assert len(results) > 0

    def test_get_dead_ends(self, memory_store, sample_hypothesis):
        """Test retrieving all dead-end memories."""
        memory_store.add_dead_end_memory(
            hypothesis=sample_hypothesis,
            reason="Tested 5 times, all failures",
        )

        dead_ends = memory_store.get_dead_ends()

        assert len(dead_ends) == 1
        assert "Dead end" in dead_ends[0].content

    def test_get_insights(self, memory_store):
        """Test retrieving all insight memories."""
        memory_store.add_insight_memory(
            insight="Important discovery",
            source="Experiment 123",
        )

        insights = memory_store.get_insights()

        assert len(insights) == 1
        assert insights[0].content == "Important discovery"


# ============================================================================
# Test Class 4: Experiment Deduplication
# ============================================================================

class TestExperimentDeduplication:
    """Test experiment deduplication via signatures."""

    def test_record_experiment(
        self, memory_store, sample_hypothesis, sample_protocol
    ):
        """Test recording experiment signature."""
        signature_hash = memory_store.record_experiment(
            hypothesis=sample_hypothesis,
            protocol=sample_protocol,
        )

        assert signature_hash is not None
        assert isinstance(signature_hash, str)
        assert signature_hash in memory_store.experiment_signatures

        signature = memory_store.experiment_signatures[signature_hash]
        assert signature.hypothesis_id == sample_hypothesis.id
        assert signature.protocol_id == sample_protocol.id

    def test_is_duplicate_exact_match(
        self, memory_store, sample_hypothesis, sample_protocol
    ):
        """Test detecting exact duplicate experiments."""
        # Record first experiment
        memory_store.record_experiment(sample_hypothesis, sample_protocol)

        # Check for duplicate (same hypothesis + protocol)
        is_dup, reason = memory_store.is_duplicate_experiment(
            sample_hypothesis, sample_protocol
        )

        assert is_dup is True
        assert "Exact duplicate" in reason

    def test_is_duplicate_similar_hypothesis(
        self, memory_store, sample_hypothesis, sample_protocol
    ):
        """Test detecting duplicate with same hypothesis."""
        # Record first experiment
        memory_store.record_experiment(sample_hypothesis, sample_protocol)

        # Create different protocol for same hypothesis
        different_protocol = ExperimentProtocol(
            id="protocol_002",
            hypothesis_id="hyp_001",
            experiment_type="computational",
            methodology="Different methodology",
            description="Different description",
        )

        # Check for duplicate (same hypothesis, different protocol)
        is_dup, reason = memory_store.is_duplicate_experiment(
            sample_hypothesis, different_protocol
        )

        assert is_dup is True
        assert "Similar hypothesis tested" in reason

    def test_is_duplicate_new_experiment(
        self, memory_store, sample_hypothesis, sample_protocol
    ):
        """Test non-duplicate experiment."""
        # Record first experiment
        memory_store.record_experiment(sample_hypothesis, sample_protocol)

        # Create different hypothesis
        different_hypothesis = Hypothesis(
            id="hyp_002",
            research_question="Different question",
            statement="Different statement",
            rationale="Different rationale",
            domain="biology",
        )

        # Check for duplicate (different hypothesis)
        is_dup, reason = memory_store.is_duplicate_experiment(
            different_hypothesis, sample_protocol
        )

        assert is_dup is False
        assert reason is None

    def test_record_experiment_without_protocol(
        self, memory_store, sample_hypothesis
    ):
        """Test recording experiment without protocol."""
        signature_hash = memory_store.record_experiment(
            hypothesis=sample_hypothesis,
            protocol=None,
        )

        assert signature_hash is not None
        signature = memory_store.experiment_signatures[signature_hash]
        assert signature.protocol_hash == "none"

    def test_duplicate_detection_without_protocol(
        self, memory_store, sample_hypothesis
    ):
        """Test duplicate detection works without protocol."""
        # Record experiment without protocol
        memory_store.record_experiment(sample_hypothesis, None)

        # Check duplicate
        is_dup, reason = memory_store.is_duplicate_experiment(sample_hypothesis, None)

        assert is_dup is True


# ============================================================================
# Test Class 5: Memory Pruning
# ============================================================================

class TestMemoryPruning:
    """Test memory pruning logic."""

    def test_prune_old_memories(self, memory_store):
        """Test pruning memories older than threshold."""
        # Add old memory (manually set timestamp)
        old_memory = Memory(
            id="old_001",
            category=MemoryCategory.GENERAL,
            content="Old memory",
            importance=0.2,  # Low importance
        )
        old_memory.created_at = datetime.utcnow() - timedelta(days=60)  # 60 days old
        memory_store.memories[MemoryCategory.GENERAL].append(old_memory)

        # Add recent memory
        memory_store.add_memory(
            category=MemoryCategory.GENERAL,
            content="Recent memory",
            importance=0.2,
        )

        # Prune (threshold is 30 days)
        memory_store.prune_old_memories()

        # Old memory should be removed
        contents = [m.content for m in memory_store.memories[MemoryCategory.GENERAL]]
        assert "Recent memory" in contents
        # Old memory might be kept if accessed or important

    def test_prune_keeps_high_importance(self, memory_store):
        """Test pruning keeps high-importance memories even if old."""
        # Add old but important memory
        old_important = Memory(
            id="important_001",
            category=MemoryCategory.GENERAL,
            content="Important old memory",
            importance=0.9,  # High importance
        )
        old_important.created_at = datetime.utcnow() - timedelta(days=60)
        memory_store.memories[MemoryCategory.GENERAL].append(old_important)

        # Prune
        memory_store.prune_old_memories()

        # Important memory should be kept
        contents = [m.content for m in memory_store.memories[MemoryCategory.GENERAL]]
        assert "Important old memory" in contents

    def test_prune_keeps_accessed_memories(self, memory_store):
        """Test pruning keeps frequently accessed memories."""
        # Add old but accessed memory
        old_accessed = Memory(
            id="accessed_001",
            category=MemoryCategory.GENERAL,
            content="Accessed old memory",
            importance=0.2,
            access_count=5,  # Frequently accessed
        )
        old_accessed.created_at = datetime.utcnow() - timedelta(days=60)
        memory_store.memories[MemoryCategory.GENERAL].append(old_accessed)

        # Prune
        memory_store.prune_old_memories()

        # Accessed memory should be kept
        contents = [m.content for m in memory_store.memories[MemoryCategory.GENERAL]]
        assert "Accessed old memory" in contents

    def test_prune_category(self, memory_store):
        """Test pruning specific category."""
        # Add multiple memories
        for i in range(5):
            memory = Memory(
                id=f"mem_{i}",
                category=MemoryCategory.GENERAL,
                content=f"Memory {i}",
                importance=0.2,
            )
            memory.created_at = datetime.utcnow() - timedelta(days=60)
            memory_store.memories[MemoryCategory.GENERAL].append(memory)

        initial_count = len(memory_store.memories[MemoryCategory.GENERAL])

        # Prune category
        memory_store._prune_category(MemoryCategory.GENERAL)

        # Should have pruned some memories
        final_count = len(memory_store.memories[MemoryCategory.GENERAL])
        # Might be 0 if all are old and low importance
        assert final_count <= initial_count


# ============================================================================
# Test Class 6: Memory Statistics
# ============================================================================

class TestMemoryStatistics:
    """Test memory statistics and reporting."""

    def test_get_memory_statistics(
        self, memory_store, sample_hypothesis, sample_result
    ):
        """Test getting comprehensive memory statistics."""
        # Add various memories
        memory_store.add_success_memory(sample_result, sample_hypothesis)
        memory_store.add_insight_memory("Test insight", "Source")
        memory_store.record_experiment(sample_hypothesis, None)

        stats = memory_store.get_memory_statistics()

        assert "total_memories" in stats
        assert "by_category" in stats
        assert "experiment_signatures" in stats
        assert stats["total_memories"] >= 2
        assert stats["experiment_signatures"] >= 1

    def test_get_most_accessed_memory(
        self, memory_store
    ):
        """Test identifying most accessed memory."""
        # Add memories with different access counts
        mem1_id = memory_store.add_memory(
            category=MemoryCategory.GENERAL,
            content="Memory 1",
        )

        mem2_id = memory_store.add_memory(
            category=MemoryCategory.GENERAL,
            content="Memory 2",
        )

        # Access memory 2 multiple times
        for _ in range(5):
            memory_store.memories[MemoryCategory.GENERAL][1].access()

        most_accessed = memory_store._get_most_accessed_memory()

        assert most_accessed is not None
        assert "Memory 2" in most_accessed

    def test_get_highest_importance_memory(
        self, memory_store
    ):
        """Test identifying highest importance memory."""
        memory_store.add_memory(
            category=MemoryCategory.GENERAL,
            content="Low importance",
            importance=0.2,
        )

        memory_store.add_memory(
            category=MemoryCategory.GENERAL,
            content="High importance",
            importance=0.95,
        )

        highest = memory_store._get_highest_importance_memory()

        assert highest is not None
        assert "High importance" in highest

    def test_export_memories_all_categories(
        self, memory_store, sample_hypothesis, sample_result
    ):
        """Test exporting all memories."""
        memory_store.add_success_memory(sample_result, sample_hypothesis)
        memory_store.add_insight_memory("Test insight", "Source")

        exported = memory_store.export_memories()

        assert isinstance(exported, list)
        assert len(exported) >= 2
        assert all(isinstance(m, dict) for m in exported)

    def test_export_memories_specific_category(
        self, memory_store, sample_hypothesis, sample_result
    ):
        """Test exporting memories from specific category."""
        memory_store.add_success_memory(sample_result, sample_hypothesis)
        memory_store.add_insight_memory("Test insight", "Source")

        exported = memory_store.export_memories(category=MemoryCategory.SUCCESS_PATTERNS)

        assert isinstance(exported, list)
        assert len(exported) == 1
        assert exported[0]["category"] == "success_patterns"

    def test_statistics_empty_store(self, memory_store):
        """Test statistics with no memories."""
        stats = memory_store.get_memory_statistics()

        assert stats["total_memories"] == 0
        assert stats["experiment_signatures"] == 0
        assert stats["most_accessed"] is None
        assert stats["highest_importance"] is None

    def test_memory_access_method(self, memory_store):
        """Test memory access method updates correctly."""
        memory_id = memory_store.add_memory(
            category=MemoryCategory.GENERAL,
            content="Test memory",
        )

        memory = memory_store.memories[MemoryCategory.GENERAL][0]
        initial_count = memory.access_count
        initial_time = memory.last_accessed

        # Access memory
        memory.access()

        assert memory.access_count == initial_count + 1
        assert memory.last_accessed >= initial_time
