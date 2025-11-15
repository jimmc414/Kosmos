"""
Abstract interfaces for world model storage backends.

This module defines the abstract base classes (ABCs) that all world model
storage implementations must follow. This enables:

1. **Mode Switching**: Swap between Simple Mode (Neo4j) and Production Mode (polyglot)
2. **Testability**: Create mock implementations for fast unit tests
3. **Extensibility**: Add new storage backends without changing client code
4. **API Consistency**: Same interface across all modes

DESIGN PATTERN: Abstract Base Class (ABC)
- Enforces interface contract
- Documents expected behavior
- Enables polymorphism

IMPLEMENTATIONS:
- Neo4jWorldModel: Simple Mode (Phase 1) - Single Neo4j database
- PolyglotWorldModel: Production Mode (Phase 4) - PostgreSQL + Neo4j + ES + Vector DB

EDUCATIONAL NOTE:
This is a classic example of the Strategy Pattern:
- WorldModelStorage is the strategy interface
- Neo4jWorldModel and PolyglotWorldModel are concrete strategies
- Factory pattern selects which strategy to use
See: https://refactoring.guru/design-patterns/strategy
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from kosmos.world_model.models import Annotation, Entity, Relationship


class WorldModelStorage(ABC):
    """
    Abstract storage interface for world model.

    This is the primary interface that all storage backends must implement.
    It provides CRUD operations for entities and relationships, plus
    import/export and statistics.

    IMPLEMENTATION REQUIREMENTS:
    - Thread-safe operations (use locks where needed)
    - Handle duplicate entities gracefully (merge parameter)
    - Validate entity/relationship data before storing
    - Clean up resources in close()

    PERFORMANCE TARGETS (Simple Mode):
    - add_entity: <100ms p95
    - get_entity: <50ms p95
    - add_relationship: <100ms p95
    - export_graph (10K entities): <30s
    - import_graph (10K entities): <60s

    Example:
        storage = get_world_model()  # Returns implementation

        # Add entity
        entity = Entity(type="Paper", properties={"title": "..."})
        entity_id = storage.add_entity(entity)

        # Query entity
        retrieved = storage.get_entity(entity_id)

        # Export graph
        storage.export_graph("backup.json")
    """

    @abstractmethod
    def add_entity(self, entity: Entity, merge: bool = True) -> str:
        """
        Add entity to the knowledge graph.

        Args:
            entity: Entity to add
            merge: If True, merge with existing entity if duplicate found
                   If False, create new entity (may create duplicates)

        Returns:
            Entity ID (the entity's id field)

        Raises:
            ValueError: If entity validation fails
            DuplicateEntityError: If merge=False and duplicate exists

        Implementation Notes:
            - If merge=True and duplicate found:
                - Update properties (merge dicts, newer values win)
                - Increase confidence if new confidence higher
                - Add to annotations list
            - Generate new ID if entity.id is None
            - Set created_at and updated_at timestamps
            - Validate entity type and properties

        Example:
            paper = Entity(
                type="Paper",
                properties={"title": "Test", "year": 2024}
            )
            entity_id = storage.add_entity(paper, merge=True)
        """
        pass

    @abstractmethod
    def get_entity(self, entity_id: str, project: Optional[str] = None) -> Optional[Entity]:
        """
        Retrieve entity by ID.

        Args:
            entity_id: Unique entity identifier
            project: Optional project filter (None = any project)

        Returns:
            Entity if found, None otherwise

        Example:
            entity = storage.get_entity("paper-id-123")
            if entity:
                print(entity.properties["title"])
        """
        pass

    @abstractmethod
    def update_entity(self, entity_id: str, updates: Dict[str, Any]) -> None:
        """
        Update entity properties.

        Args:
            entity_id: Entity to update
            updates: Dict of property updates (merged with existing)

        Raises:
            EntityNotFoundError: If entity doesn't exist

        Implementation Notes:
            - Merge updates with existing properties
            - Update updated_at timestamp
            - Don't update created_at or created_by

        Example:
            storage.update_entity(
                "paper-id",
                {"verified": True, "properties.impact_score": 9.5}
            )
        """
        pass

    @abstractmethod
    def delete_entity(self, entity_id: str) -> None:
        """
        Delete entity and all its relationships.

        Args:
            entity_id: Entity to delete

        Raises:
            EntityNotFoundError: If entity doesn't exist

        Implementation Notes:
            - Delete all relationships where entity is source or target
            - Delete entity node
            - This is a destructive operation (should be logged)
        """
        pass

    @abstractmethod
    def add_relationship(self, relationship: Relationship) -> str:
        """
        Add relationship between two entities.

        Args:
            relationship: Relationship to add

        Returns:
            Relationship ID

        Raises:
            ValueError: If relationship validation fails
            EntityNotFoundError: If source or target entity doesn't exist

        Implementation Notes:
            - Validate source_id and target_id exist
            - Check for duplicate relationships
            - Generate ID if not provided

        Example:
            rel = Relationship(
                source_id="paper1",
                target_id="paper2",
                type="CITES"
            )
            rel_id = storage.add_relationship(rel)
        """
        pass

    @abstractmethod
    def get_relationship(self, relationship_id: str) -> Optional[Relationship]:
        """
        Retrieve relationship by ID.

        Args:
            relationship_id: Relationship identifier

        Returns:
            Relationship if found, None otherwise
        """
        pass

    @abstractmethod
    def query_related_entities(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None,
        direction: str = "outgoing",
        max_depth: int = 1,
    ) -> List[Entity]:
        """
        Query entities related to a given entity.

        Args:
            entity_id: Starting entity
            relationship_type: Optional filter by relationship type
            direction: "outgoing", "incoming", or "both"
            max_depth: Maximum traversal depth (1 = direct neighbors)

        Returns:
            List of related entities

        Example:
            # Get all papers cited by this paper
            cited_papers = storage.query_related_entities(
                "paper1",
                relationship_type="CITES",
                direction="outgoing"
            )
        """
        pass

    @abstractmethod
    def export_graph(self, filepath: str, project: Optional[str] = None) -> None:
        """
        Export knowledge graph to file.

        Args:
            filepath: Output file path (.json or .graphml)
            project: Optional project filter (None = all projects)

        Format:
            {
                "version": "1.0",
                "exported_at": "2024-01-15T10:30:00",
                "source": "kosmos",
                "mode": "simple",
                "project": "my_project",
                "statistics": {...},
                "entities": [...],
                "relationships": [...]
            }

        Implementation Notes:
            - Create parent directories if needed
            - Use JSON for human readability
            - Include statistics for verification
            - Handle large graphs (streaming if needed)

        Example:
            storage.export_graph("backup.json")
            storage.export_graph("project1.json", project="project1")
        """
        pass

    @abstractmethod
    def import_graph(self, filepath: str, clear: bool = False, project: Optional[str] = None) -> None:
        """
        Import knowledge graph from file.

        Args:
            filepath: Input file path
            clear: If True, clear existing graph before import
            project: Optional project to import into

        Implementation Notes:
            - Validate file format and version
            - Handle ID conflicts (merge or error)
            - Import entities first, then relationships
            - Log import statistics

        Example:
            storage.import_graph("backup.json")
            storage.import_graph("data.json", clear=True)
        """
        pass

    @abstractmethod
    def get_statistics(self, project: Optional[str] = None) -> Dict[str, Any]:
        """
        Get knowledge graph statistics.

        Args:
            project: Optional project filter

        Returns:
            Dictionary with statistics:
            {
                "entity_count": 1234,
                "relationship_count": 5678,
                "entity_types": {
                    "Paper": 800,
                    "Concept": 300,
                    "Author": 134
                },
                "relationship_types": {
                    "CITES": 3000,
                    "MENTIONS": 2000
                },
                "projects": ["project1", "project2"],
                "storage_size_mb": 12.5
            }

        Example:
            stats = storage.get_statistics()
            print(f"Total entities: {stats['entity_count']}")
        """
        pass

    @abstractmethod
    def reset(self, project: Optional[str] = None) -> None:
        """
        Clear all knowledge graph data.

        WARNING: This is a destructive operation!

        Args:
            project: Optional project to reset (None = reset ALL data)

        Implementation Notes:
            - If project specified, delete only that project's data
            - If project is None, delete ALL data
            - This should be logged and require confirmation in CLI

        Example:
            storage.reset(project="test_project")  # Safe
            storage.reset()  # DELETES EVERYTHING!
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close connections and cleanup resources.

        Implementation Notes:
            - Close database connections
            - Flush any pending writes
            - Release locks
            - Safe to call multiple times

        Example:
            storage = get_world_model()
            try:
                # ... use storage
            finally:
                storage.close()
        """
        pass


class EntityManager(ABC):
    """
    Abstract interface for entity curation operations.

    This interface provides entity management features beyond basic CRUD:
    - Verification (mark entities as manually verified)
    - Annotations (add notes/corrections)
    - Quality scoring

    PHASE: 2 (Curation Features)
    - Simple Mode: Store in Neo4j properties
    - Production Mode: Store in PostgreSQL with audit trail

    Example:
        manager = get_world_model()  # Also implements EntityManager

        # Verify entity
        manager.verify_entity("entity-id", verified_by="researcher@uni.edu")

        # Add annotation
        ann = Annotation(text="Check this reference", created_by="reviewer")
        manager.add_annotation("entity-id", ann)
    """

    @abstractmethod
    def verify_entity(self, entity_id: str, verified_by: str) -> None:
        """
        Mark entity as manually verified.

        Args:
            entity_id: Entity to verify
            verified_by: Who verified (email/username)

        Raises:
            EntityNotFoundError: If entity doesn't exist
        """
        pass

    @abstractmethod
    def add_annotation(self, entity_id: str, annotation: Annotation) -> None:
        """
        Add annotation to entity.

        Args:
            entity_id: Entity to annotate
            annotation: Annotation to add

        Raises:
            EntityNotFoundError: If entity doesn't exist
        """
        pass

    @abstractmethod
    def get_annotations(self, entity_id: str) -> List[Annotation]:
        """
        Get all annotations for an entity.

        Args:
            entity_id: Entity to query

        Returns:
            List of annotations (empty if none)
        """
        pass


class ProvenanceTracker(ABC):
    """
    Abstract interface for provenance tracking.

    Provenance tracks the derivation history of entities:
    - What sources were used?
    - What agent created it?
    - What transformations were applied?

    PHASE: 4 (Production Mode)
    - Implements W3C PROV-O standard
    - Stored in Elasticsearch for querying

    STANDARD: W3C PROV-O (https://www.w3.org/TR/prov-o/)
    - Entity: Thing produced/used
    - Activity: What happened
    - Agent: Who/what did it

    Example:
        tracker = get_world_model()  # Production Mode

        # Record derivation
        tracker.record_derivation(
            entity_id="hypothesis-123",
            sources=["paper-1", "paper-2"],
            agent="hypothesis_generator",
            activity="llm_generation"
        )
    """

    @abstractmethod
    def record_derivation(
        self,
        entity_id: str,
        sources: List[str],
        agent: str,
        activity: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record entity derivation (PROV-O wasDerivedFrom).

        Args:
            entity_id: Entity that was created
            sources: Source entity IDs
            agent: Agent that created it
            activity: Optional activity description
            metadata: Optional additional context

        Example:
            tracker.record_derivation(
                entity_id="hypothesis-new",
                sources=["paper-123", "paper-456"],
                agent="hypothesis_generator",
                activity="llm_synthesis",
                metadata={"model": "claude-3.5", "temperature": 0.7}
            )
        """
        pass

    @abstractmethod
    def get_provenance(self, entity_id: str) -> Dict[str, Any]:
        """
        Get provenance chain for an entity.

        Args:
            entity_id: Entity to trace

        Returns:
            PROV-O compatible provenance graph

        Example:
            prov = tracker.get_provenance("hypothesis-123")
            print(f"Derived from: {prov['sources']}")
            print(f"Created by: {prov['agent']}")
        """
        pass
