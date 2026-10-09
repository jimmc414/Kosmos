"""
Memory System - Stores and retrieves research history to avoid duplication (Phase 7).

Categories:
- SUCCESS_PATTERNS: What worked
- FAILURE_PATTERNS: What didn't work
- DEAD_ENDS: Hypotheses/approaches to avoid
- INSIGHTS: Key discoveries
"""

from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, Field
from enum import Enum
import logging
import hashlib

from kosmos.models.hypothesis import Hypothesis
from kosmos.models.result import ExperimentResult
from kosmos.models.experiment import ExperimentProtocol
from kosmos.utils.compat import model_to_dict

logger = logging.getLogger(__name__)


class MemoryCategory(str, Enum):
    """Categories of memories."""

    SUCCESS_PATTERNS = "success_patterns"
    FAILURE_PATTERNS = "failure_patterns"
    DEAD_ENDS = "dead_ends"
    INSIGHTS = "insights"
    GENERAL = "general"


class Memory(BaseModel):
    """A single memory entry."""

    id: str
    category: MemoryCategory
    content: str
    data: Dict[str, Any] = Field(default_factory=dict)
    importance: float = Field(0.5, ge=0.0, le=1.0, description="Importance score")
    access_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: List[str] = Field(default_factory=list)

    def access(self):
        """Record memory access."""
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)


class ExperimentSignature(BaseModel):
    """Signature for experiment deduplication."""

    hypothesis_hash: str
    protocol_hash: str
    combined_hash: str
    hypothesis_id: Optional[str] = None
    protocol_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MemoryStore:
    """
    Memory system for research history.

    Stores:
    - Success patterns (what worked)
    - Failure patterns (what didn't work)
    - Dead ends (avoid repeating)
    - Insights (key discoveries)
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        max_memories: int = 1000
    ):
        """
        Initialize memory store.

        Args:
            config: Configuration dict
            max_memories: Maximum memories to store (oldest pruned)
        """
        self.config = config or {}
        self.max_memories = max_memories

        # Memory storage by category
        self.memories: Dict[MemoryCategory, List[Memory]] = {
            category: [] for category in MemoryCategory
        }

        # Experiment signatures for deduplication
        self.experiment_signatures: Dict[str, ExperimentSignature] = {}

        # Pruning configuration
        self.prune_after_days = self.config.get("prune_after_days", 30)
        self.min_importance_to_keep = self.config.get("min_importance_to_keep", 0.3)

        logger.info(f"MemoryStore initialized (max: {max_memories}, prune_after: {self.prune_after_days} days)")

    # ========================================================================
    # ADD MEMORIES
    # ========================================================================

    def add_memory(
        self,
        category: MemoryCategory,
        content: str,
        data: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Add a memory to the store.

        Args:
            category: Memory category
            content: Memory content description
            data: Associated data
            importance: Importance score (0.0-1.0)
            tags: Tags for retrieval

        Returns:
            str: Memory ID
        """
        # Generate memory ID
        memory_id = hashlib.md5(
            f"{category}:{content}:{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]

        memory = Memory(
            id=memory_id,
            category=category,
            content=content,
            data=data or {},
            importance=importance,
            tags=tags or []
        )

        self.memories[category].append(memory)

        # Prune if needed
        if len(self.memories[category]) > self.max_memories // len(MemoryCategory):
            self._prune_category(category)

        logger.debug(f"Added memory to {category}: {content[:50]}...")
        return memory_id

    def add_success_memory(
        self,
        result: ExperimentResult,
        hypothesis: Hypothesis,
        insights: Optional[str] = None
    ) -> str:
        """
        Add success pattern memory.

        Args:
            result: Successful result
            hypothesis: Hypothesis that was supported
            insights: Optional insights

        Returns:
            str: Memory ID
        """
        content = f"Success: {hypothesis.statement[:100]} (p={result.primary_p_value:.4f}, effect={result.primary_effect_size:.2f})"

        data = {
            "result_id": result.id,
            "hypothesis_id": hypothesis.id,
            "p_value": result.primary_p_value,
            "effect_size": result.primary_effect_size,
            "test_type": result.primary_test,
            "insights": insights
        }

        tags = ["success", hypothesis.domain, result.primary_test]

        return self.add_memory(
            category=MemoryCategory.SUCCESS_PATTERNS,
            content=content,
            data=data,
            importance=0.8,  # Successes are important
            tags=tags
        )

    def add_failure_memory(
        self,
        result: ExperimentResult,
        hypothesis: Hypothesis,
        failure_reason: str
    ) -> str:
        """
        Add failure pattern memory.

        Args:
            result: Failed result
            hypothesis: Hypothesis that was rejected
            failure_reason: Why it failed

        Returns:
            str: Memory ID
        """
        content = f"Failure: {hypothesis.statement[:100]} - {failure_reason}"

        data = {
            "result_id": result.id,
            "hypothesis_id": hypothesis.id,
            "failure_reason": failure_reason,
            "test_type": result.primary_test
        }

        tags = ["failure", hypothesis.domain, failure_reason]

        return self.add_memory(
            category=MemoryCategory.FAILURE_PATTERNS,
            content=content,
            data=data,
            importance=0.7,  # Failures are also important to learn from
            tags=tags
        )

    def add_dead_end_memory(
        self,
        hypothesis: Hypothesis,
        reason: str
    ) -> str:
        """
        Add dead-end memory to avoid repeating.

        Args:
            hypothesis: Hypothesis that is a dead end
            reason: Why it's a dead end

        Returns:
            str: Memory ID
        """
        content = f"Dead end: {hypothesis.statement[:100]} - {reason}"

        data = {
            "hypothesis_id": hypothesis.id,
            "reason": reason
        }

        tags = ["dead_end", hypothesis.domain]

        return self.add_memory(
            category=MemoryCategory.DEAD_ENDS,
            content=content,
            data=data,
            importance=0.9,  # Very important to avoid repeating
            tags=tags
        )

    def add_insight_memory(
        self,
        insight: str,
        source: str,
        related_hypotheses: Optional[List[str]] = None
    ) -> str:
        """
        Add key insight memory.

        Args:
            insight: Insight description
            source: Where insight came from
            related_hypotheses: Related hypothesis IDs

        Returns:
            str: Memory ID
        """
        content = insight

        data = {
            "source": source,
            "related_hypotheses": related_hypotheses or []
        }

        tags = ["insight"]

        return self.add_memory(
            category=MemoryCategory.INSIGHTS,
            content=content,
            data=data,
            importance=0.95,  # Insights are very important
            tags=tags
        )

    # ========================================================================
    # QUERY MEMORIES
    # ========================================================================

    def query_memory(
        self,
        category: Optional[MemoryCategory] = None,
        tags: Optional[List[str]] = None,
        min_importance: float = 0.0,
        limit: int = 10
    ) -> List[Memory]:
        """
        Query memories.

        Args:
            category: Filter by category
            tags: Filter by tags
            min_importance: Minimum importance
            limit: Maximum results

        Returns:
            List[Memory]: Matching memories
        """
        results = []

        # Select categories to search
        if category:
            categories = [category]
        else:
            categories = list(MemoryCategory)

        # Search
        for cat in categories:
            for memory in self.memories[cat]:
                # Filter by importance
                if memory.importance < min_importance:
                    continue

                # Filter by tags
                if tags and not any(tag in memory.tags for tag in tags):
                    continue

                results.append(memory)

        # Sort by importance * recency
        results.sort(
            key=lambda m: m.importance * (1.0 / max(1, (datetime.now(timezone.utc) - m.created_at).days + 1)),
            reverse=True
        )

        # Record access
        for memory in results[:limit]:
            memory.access()

        return results[:limit]

    def search_similar_hypothesis(self, hypothesis: Hypothesis) -> List[Memory]:
        """
        Search for memories of similar hypotheses.

        Args:
            hypothesis: Hypothesis to search for

        Returns:
            List[Memory]: Similar hypothesis memories
        """
        # Simple keyword matching (could use semantic search with vector DB)
        keywords = set(hypothesis.statement.lower().split())

        similar = []

        for category in [MemoryCategory.SUCCESS_PATTERNS, MemoryCategory.FAILURE_PATTERNS, MemoryCategory.DEAD_ENDS]:
            for memory in self.memories[category]:
                memory_keywords = set(memory.content.lower().split())
                overlap = len(keywords & memory_keywords)

                if overlap >= 3:  # At least 3 common words
                    similar.append(memory)

        similar.sort(key=lambda m: m.importance, reverse=True)
        return similar[:5]

    def get_dead_ends(self) -> List[Memory]:
        """Get all dead-end memories."""
        return self.memories[MemoryCategory.DEAD_ENDS]

    def get_insights(self) -> List[Memory]:
        """Get all insights."""
        return self.memories[MemoryCategory.INSIGHTS]

    # ========================================================================
    # EXPERIMENT DEDUPLICATION
    # ========================================================================

    def record_experiment(
        self,
        hypothesis: Hypothesis,
        protocol: Optional[ExperimentProtocol] = None
    ) -> str:
        """
        Record experiment signature for deduplication.

        Args:
            hypothesis: Hypothesis being tested
            protocol: Experiment protocol

        Returns:
            str: Signature hash
        """
        # Create hypothesis hash
        hypothesis_hash = hashlib.md5(
            hypothesis.statement.encode()
        ).hexdigest()[:16]

        # Create protocol hash (if available)
        if protocol:
            protocol_str = f"{protocol.experiment_type}:{protocol.methodology if hasattr(protocol, 'methodology') else ''}"
            protocol_hash = hashlib.md5(protocol_str.encode()).hexdigest()[:16]
        else:
            protocol_hash = "none"

        # Combined hash
        combined_hash = hashlib.md5(
            f"{hypothesis_hash}:{protocol_hash}".encode()
        ).hexdigest()

        # Store signature
        signature = ExperimentSignature(
            hypothesis_hash=hypothesis_hash,
            protocol_hash=protocol_hash,
            combined_hash=combined_hash,
            hypothesis_id=hypothesis.id,
            protocol_id=protocol.id if protocol and hasattr(protocol, 'id') else None
        )

        self.experiment_signatures[combined_hash] = signature

        logger.debug(f"Recorded experiment signature: {combined_hash}")
        return combined_hash

    def is_duplicate_experiment(
        self,
        hypothesis: Hypothesis,
        protocol: Optional[ExperimentProtocol] = None,
        similarity_threshold: float = 0.9
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if experiment is a duplicate.

        Args:
            hypothesis: Hypothesis to test
            protocol: Experiment protocol
            similarity_threshold: Threshold for considering duplicate

        Returns:
            Tuple of (is_duplicate: bool, reason: Optional[str])
        """
        # Create hash for this experiment
        hypothesis_hash = hashlib.md5(
            hypothesis.statement.encode()
        ).hexdigest()[:16]

        if protocol:
            protocol_str = f"{protocol.experiment_type}:{protocol.methodology if hasattr(protocol, 'methodology') else ''}"
            protocol_hash = hashlib.md5(protocol_str.encode()).hexdigest()[:16]
        else:
            protocol_hash = "none"

        combined_hash = hashlib.md5(
            f"{hypothesis_hash}:{protocol_hash}".encode()
        ).hexdigest()

        # Check for exact match
        if combined_hash in self.experiment_signatures:
            return True, "Exact duplicate (same hypothesis + same protocol)"

        # Check for similar hypothesis
        similar_hyp = sum(
            1 for sig in self.experiment_signatures.values()
            if sig.hypothesis_hash == hypothesis_hash
        )

        if similar_hyp > 0:
            return True, f"Similar hypothesis tested {similar_hyp} time(s)"

        return False, None

    # ========================================================================
    # MEMORY PRUNING
    # ========================================================================

    def _prune_category(self, category: MemoryCategory):
        """Prune old or low-importance memories from category."""
        memories = self.memories[category]

        if not memories:
            return

        # Keep memories that are:
        # 1. Important (>= min_importance)
        # 2. Recent (< prune_after_days)
        # 3. Frequently accessed (access_count > 0)

        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.prune_after_days)

        kept = []
        pruned_count = 0

        for memory in memories:
            should_keep = (
                memory.importance >= self.min_importance_to_keep or
                memory.created_at > cutoff_date or
                memory.access_count > 0
            )

            if should_keep:
                kept.append(memory)
            else:
                pruned_count += 1

        self.memories[category] = kept

        if pruned_count > 0:
            logger.info(f"Pruned {pruned_count} memories from {category}")

    def prune_old_memories(self):
        """Prune old memories from all categories."""
        for category in MemoryCategory:
            self._prune_category(category)

    # ========================================================================
    # STATISTICS & REPORTING
    # ========================================================================

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory store statistics."""
        stats = {
            "total_memories": sum(len(memories) for memories in self.memories.values()),
            "by_category": {
                category.value: len(self.memories[category])
                for category in MemoryCategory
            },
            "experiment_signatures": len(self.experiment_signatures),
            "most_accessed": self._get_most_accessed_memory(),
            "highest_importance": self._get_highest_importance_memory()
        }

        return stats

    def _get_most_accessed_memory(self) -> Optional[str]:
        """Get most accessed memory."""
        all_memories = [
            memory
            for memories in self.memories.values()
            for memory in memories
        ]

        if not all_memories:
            return None

        most_accessed = max(all_memories, key=lambda m: m.access_count)
        return most_accessed.content

    def _get_highest_importance_memory(self) -> Optional[str]:
        """Get highest importance memory."""
        all_memories = [
            memory
            for memories in self.memories.values()
            for memory in memories
        ]

        if not all_memories:
            return None

        highest = max(all_memories, key=lambda m: m.importance)
        return highest.content

    def export_memories(self, category: Optional[MemoryCategory] = None) -> List[Dict[str, Any]]:
        """
        Export memories as dictionaries.

        Args:
            category: Optional category filter

        Returns:
            List of memory dicts
        """
        if category:
            memories = self.memories[category]
        else:
            memories = [
                memory
                for memories in self.memories.values()
                for memory in memories
            ]

        return [model_to_dict(memory) for memory in memories]
