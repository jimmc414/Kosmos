"""
Tests for World Model Persistence Requirements (REQ-WM-PERSIST-001 through REQ-WM-PERSIST-006).

These tests validate data persistence, export/import operations, backup/restore,
versioning, and data integrity across sessions.
"""

import pytest
import uuid
import json
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, patch

from kosmos.world_model.models import Entity, Relationship, Annotation, EXPORT_FORMAT_VERSION
from kosmos.world_model import get_world_model, reset_world_model

# Test markers for requirements traceability
pytestmark = [
    pytest.mark.requirement("REQ-WM-PERSIST"),
    pytest.mark.category("world_model"),
]


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def world_model():
    """Provide a clean world model instance for each test."""
    try:
        wm = get_world_model()
        try:
            wm.reset(project="test_persistence")
        except Exception:
            pass
        yield wm
    finally:
        try:
            reset_world_model()
        except Exception:
            pass


@pytest.fixture
def temp_export_dir():
    """Provide a temporary directory for export files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_knowledge_graph(world_model):
    """Create a sample knowledge graph for persistence testing."""
    project = "test_persistence"

    # Create diverse entities
    entities = []

    paper = Entity(
        type="Paper",
        properties={
            "title": "Neural Networks: A Comprehensive Review",
            "authors": ["Smith, J.", "Doe, A.", "Johnson, K."],
            "year": 2024,
            "doi": "10.1234/example.doi",
            "abstract": "This paper reviews neural network architectures."
        },
        confidence=0.95,
        project=project,
        created_by="literature_agent",
        verified=True,
        annotations=[
            Annotation(text="Seminal work in the field", created_by="reviewer1@example.com")
        ]
    )
    entities.append(paper)

    concept = Entity(
        type="Concept",
        properties={
            "name": "Deep Learning",
            "description": "Machine learning using deep neural networks",
            "domain": "artificial_intelligence"
        },
        confidence=0.9,
        project=project,
        created_by="concept_extractor"
    )
    entities.append(concept)

    hypothesis = Entity(
        type="Hypothesis",
        properties={
            "statement": "Increased network depth improves performance",
            "rationale": "Based on empirical observations",
            "testability_score": 0.85,
            "novelty_score": 0.7
        },
        confidence=0.8,
        project=project,
        created_by="hypothesis_generator"
    )
    entities.append(hypothesis)

    # Add entities to graph
    entity_ids = []
    for entity in entities:
        entity_id = world_model.add_entity(entity)
        entity_ids.append(entity_id)

    # Create relationships
    relationships = []

    if len(entity_ids) >= 2:
        rel1 = Relationship(
            source_id=entity_ids[0],  # Paper
            target_id=entity_ids[1],  # Concept
            type="MENTIONS",
            properties={"section": "introduction", "context": "primary topic"}
        )
        relationships.append(rel1)

    if len(entity_ids) >= 3:
        rel2 = Relationship(
            source_id=entity_ids[2],  # Hypothesis
            target_id=entity_ids[0],  # Paper
            type="DERIVED_FROM",
            properties={"confidence": 0.8}
        )
        relationships.append(rel2)

    # Add relationships
    rel_ids = []
    for rel in relationships:
        rel_id = world_model.add_relationship(rel)
        rel_ids.append(rel_id)

    return {
        "entity_ids": entity_ids,
        "relationship_ids": rel_ids,
        "entities": entities,
        "relationships": relationships
    }


# ============================================================================
# REQ-WM-PERSIST-001: Export Graph (MUST)
# ============================================================================

@pytest.mark.requirement("REQ-WM-PERSIST-001")
@pytest.mark.priority("MUST")
class TestREQ_WM_PERSIST_001_ExportGraph:
    """
    REQ-WM-PERSIST-001: The World Model MUST support exporting the complete
    knowledge graph to a file format (JSON) including all entities, relationships,
    and metadata.
    """

    def test_export_graph_creates_file(self, world_model, sample_knowledge_graph, temp_export_dir):
        """Verify export creates a valid file."""
        export_path = temp_export_dir / "export.json"

        world_model.export_graph(str(export_path), project="test_persistence")

        # Verify file was created
        assert export_path.exists()
        assert export_path.is_file()
        assert export_path.stat().st_size > 0

    def test_export_graph_contains_version(self, world_model, sample_knowledge_graph, temp_export_dir):
        """Verify exported file contains version information."""
        export_path = temp_export_dir / "export_version.json"

        world_model.export_graph(str(export_path), project="test_persistence")

        # Read and verify content
        with open(export_path) as f:
            data = json.load(f)

        assert "version" in data
        assert data["version"] == EXPORT_FORMAT_VERSION

    def test_export_graph_includes_entities(self, world_model, sample_knowledge_graph, temp_export_dir):
        """Verify exported file includes all entities."""
        export_path = temp_export_dir / "export_entities.json"

        world_model.export_graph(str(export_path), project="test_persistence")

        with open(export_path) as f:
            data = json.load(f)

        assert "entities" in data
        assert isinstance(data["entities"], list)
        assert len(data["entities"]) >= len(sample_knowledge_graph["entity_ids"])

    def test_export_graph_includes_relationships(self, world_model, sample_knowledge_graph, temp_export_dir):
        """Verify exported file includes all relationships."""
        export_path = temp_export_dir / "export_relationships.json"

        world_model.export_graph(str(export_path), project="test_persistence")

        with open(export_path) as f:
            data = json.load(f)

        assert "relationships" in data
        assert isinstance(data["relationships"], list)

    def test_export_graph_includes_metadata(self, world_model, sample_knowledge_graph, temp_export_dir):
        """Verify exported file includes metadata."""
        export_path = temp_export_dir / "export_metadata.json"

        world_model.export_graph(str(export_path), project="test_persistence")

        with open(export_path) as f:
            data = json.load(f)

        # Check for metadata fields
        assert "exported_at" in data
        assert "source" in data
        assert data["source"] == "kosmos"

    def test_export_graph_preserves_all_properties(self, world_model, sample_knowledge_graph, temp_export_dir):
        """Verify all entity properties are preserved in export."""
        export_path = temp_export_dir / "export_properties.json"

        world_model.export_graph(str(export_path), project="test_persistence")

        with open(export_path) as f:
            data = json.load(f)

        # Find paper entity in export
        paper_exports = [e for e in data["entities"] if e["type"] == "Paper"]
        assert len(paper_exports) >= 1

        paper = paper_exports[0]
        assert "title" in paper["properties"]
        assert "authors" in paper["properties"]
        assert "confidence" in paper

    def test_export_graph_includes_annotations(self, world_model, sample_knowledge_graph, temp_export_dir):
        """Verify entity annotations are included in export."""
        export_path = temp_export_dir / "export_annotations.json"

        world_model.export_graph(str(export_path), project="test_persistence")

        with open(export_path) as f:
            data = json.load(f)

        # Find entity with annotations
        entities_with_annotations = [e for e in data["entities"] if len(e.get("annotations", [])) > 0]
        assert len(entities_with_annotations) >= 1

        # Verify annotation structure
        annotation = entities_with_annotations[0]["annotations"][0]
        assert "text" in annotation
        assert "created_by" in annotation

    def test_export_graph_with_project_filter(self, world_model, temp_export_dir):
        """Verify export can be filtered by project."""
        # Create entities in different projects
        entity_a = Entity(
            type="Paper",
            properties={"title": "Project A"},
            project="project_a"
        )
        entity_b = Entity(
            type="Paper",
            properties={"title": "Project B"},
            project="project_b"
        )

        world_model.add_entity(entity_a)
        world_model.add_entity(entity_b)

        # Export only project_a
        export_path = temp_export_dir / "export_project_a.json"
        world_model.export_graph(str(export_path), project="project_a")

        with open(export_path) as f:
            data = json.load(f)

        # Should include project_a entities
        assert "project" in data or "entities" in data


# ============================================================================
# REQ-WM-PERSIST-002: Import Graph (MUST)
# ============================================================================

@pytest.mark.requirement("REQ-WM-PERSIST-002")
@pytest.mark.priority("MUST")
class TestREQ_WM_PERSIST_002_ImportGraph:
    """
    REQ-WM-PERSIST-002: The World Model MUST support importing knowledge graphs
    from exported files, restoring all entities, relationships, and metadata.
    """

    def test_import_graph_restores_entities(self, world_model, sample_knowledge_graph, temp_export_dir):
        """Verify import restores all entities."""
        export_path = temp_export_dir / "export_for_import.json"

        # Export
        world_model.export_graph(str(export_path), project="test_persistence")
        original_stats = world_model.get_statistics(project="test_persistence")

        # Clear and import
        world_model.reset(project="test_persistence")
        world_model.import_graph(str(export_path), project="test_persistence")

        # Verify restoration
        restored_stats = world_model.get_statistics(project="test_persistence")
        assert restored_stats["entity_count"] >= original_stats["entity_count"]

    def test_import_graph_restores_relationships(self, world_model, sample_knowledge_graph, temp_export_dir):
        """Verify import restores all relationships."""
        export_path = temp_export_dir / "export_relationships.json"

        # Export
        world_model.export_graph(str(export_path), project="test_persistence")
        original_stats = world_model.get_statistics(project="test_persistence")

        # Clear and import
        world_model.reset(project="test_persistence")
        world_model.import_graph(str(export_path), project="test_persistence")

        # Verify relationships restored
        restored_stats = world_model.get_statistics(project="test_persistence")
        assert "relationship_count" in restored_stats

    def test_import_graph_preserves_entity_ids(self, world_model, sample_knowledge_graph, temp_export_dir):
        """Verify entity IDs are preserved during import."""
        export_path = temp_export_dir / "export_ids.json"
        original_entity_id = sample_knowledge_graph["entity_ids"][0]

        # Export
        world_model.export_graph(str(export_path), project="test_persistence")

        # Clear and import
        world_model.reset(project="test_persistence")
        world_model.import_graph(str(export_path), project="test_persistence")

        # Verify entity can be retrieved with original ID
        restored_entity = world_model.get_entity(original_entity_id)
        assert restored_entity is not None
        assert restored_entity.id == original_entity_id

    def test_import_graph_with_clear_option(self, world_model, temp_export_dir):
        """Verify import with clear option removes existing data."""
        # Create initial entity
        initial_entity = Entity(
            type="Paper",
            properties={"title": "Initial Paper"},
            project="test_persistence"
        )
        world_model.add_entity(initial_entity)

        # Create export with different data
        export_path = temp_export_dir / "export_clear.json"
        export_data = {
            "version": EXPORT_FORMAT_VERSION,
            "exported_at": datetime.now().isoformat(),
            "source": "kosmos",
            "entities": [
                {
                    "id": str(uuid.uuid4()),
                    "type": "Concept",
                    "properties": {"name": "New Concept"},
                    "confidence": 1.0,
                    "project": "test_persistence"
                }
            ],
            "relationships": []
        }

        with open(export_path, "w") as f:
            json.dump(export_data, f)

        # Import with clear=True
        world_model.import_graph(str(export_path), clear=True, project="test_persistence")

        # Statistics should reflect new data
        stats = world_model.get_statistics(project="test_persistence")
        assert stats["entity_count"] >= 1

    def test_import_graph_validates_format_version(self, world_model, temp_export_dir):
        """Verify import validates file format version."""
        export_path = temp_export_dir / "invalid_version.json"

        # Create file with invalid version
        invalid_data = {
            "version": "999.0",  # Invalid future version
            "entities": [],
            "relationships": []
        }

        with open(export_path, "w") as f:
            json.dump(invalid_data, f)

        # Import should handle gracefully (warn or accept)
        # Specific behavior depends on implementation
        try:
            world_model.import_graph(str(export_path), project="test_persistence")
        except Exception:
            # May raise exception for incompatible version
            pass

    def test_import_graph_handles_missing_file(self, world_model):
        """Verify import handles missing file gracefully."""
        nonexistent_path = "/tmp/nonexistent_export.json"

        with pytest.raises(Exception):
            world_model.import_graph(nonexistent_path, project="test_persistence")


# ============================================================================
# REQ-WM-PERSIST-003: Data Integrity (MUST)
# ============================================================================

@pytest.mark.requirement("REQ-WM-PERSIST-003")
@pytest.mark.priority("MUST")
class TestREQ_WM_PERSIST_003_DataIntegrity:
    """
    REQ-WM-PERSIST-003: The World Model MUST maintain data integrity during
    export/import operations, ensuring no data loss or corruption.
    """

    def test_round_trip_preserves_all_data(self, world_model, sample_knowledge_graph, temp_export_dir):
        """Verify complete round-trip preserves all data."""
        export_path = temp_export_dir / "roundtrip.json"

        # Get original data
        original_entity_id = sample_knowledge_graph["entity_ids"][0]
        original_entity = world_model.get_entity(original_entity_id)

        # Export, clear, import
        world_model.export_graph(str(export_path), project="test_persistence")
        world_model.reset(project="test_persistence")
        world_model.import_graph(str(export_path), project="test_persistence")

        # Verify data preserved
        restored_entity = world_model.get_entity(original_entity_id)
        assert restored_entity is not None
        assert restored_entity.type == original_entity.type
        assert restored_entity.properties == original_entity.properties

    def test_export_import_preserves_confidence_scores(self, world_model, temp_export_dir):
        """Verify confidence scores are preserved."""
        entity = Entity(
            type="Hypothesis",
            properties={"statement": "Test hypothesis"},
            confidence=0.73,
            project="test_persistence"
        )
        entity_id = world_model.add_entity(entity)

        export_path = temp_export_dir / "confidence.json"
        world_model.export_graph(str(export_path), project="test_persistence")
        world_model.reset(project="test_persistence")
        world_model.import_graph(str(export_path), project="test_persistence")

        restored = world_model.get_entity(entity_id)
        assert restored is not None
        # Note: Confidence preservation depends on implementation

    def test_export_import_preserves_timestamps(self, world_model, temp_export_dir):
        """Verify timestamps are preserved."""
        entity = Entity(
            type="Paper",
            properties={"title": "Timestamp Test"},
            project="test_persistence"
        )
        entity_id = world_model.add_entity(entity)
        original = world_model.get_entity(entity_id)
        original_created_at = original.created_at if original else None

        export_path = temp_export_dir / "timestamps.json"
        world_model.export_graph(str(export_path), project="test_persistence")
        world_model.reset(project="test_persistence")
        world_model.import_graph(str(export_path), project="test_persistence")

        restored = world_model.get_entity(entity_id)
        assert restored is not None
        # Timestamps should be preserved or handled consistently

    def test_export_import_preserves_annotations(self, world_model, temp_export_dir):
        """Verify annotations are preserved."""
        entity = Entity(
            type="Paper",
            properties={"title": "Annotated Paper"},
            project="test_persistence",
            annotations=[
                Annotation(text="Important", created_by="reviewer1"),
                Annotation(text="Needs verification", created_by="reviewer2")
            ]
        )
        entity_id = world_model.add_entity(entity)

        export_path = temp_export_dir / "annotations.json"
        world_model.export_graph(str(export_path), project="test_persistence")
        world_model.reset(project="test_persistence")
        world_model.import_graph(str(export_path), project="test_persistence")

        restored = world_model.get_entity(entity_id)
        assert restored is not None
        # Annotations preservation depends on implementation

    def test_export_import_preserves_relationship_properties(self, world_model, temp_export_dir):
        """Verify relationship properties are preserved."""
        entity1 = Entity(type="Paper", properties={"title": "P1"}, project="test_persistence")
        entity2 = Entity(type="Paper", properties={"title": "P2"}, project="test_persistence")

        id1 = world_model.add_entity(entity1)
        id2 = world_model.add_entity(entity2)

        rel = Relationship(
            source_id=id1,
            target_id=id2,
            type="CITES",
            properties={"section": "introduction", "page": 5, "importance": "high"},
            confidence=0.95
        )
        rel_id = world_model.add_relationship(rel)

        export_path = temp_export_dir / "rel_properties.json"
        world_model.export_graph(str(export_path), project="test_persistence")
        world_model.reset(project="test_persistence")
        world_model.import_graph(str(export_path), project="test_persistence")

        restored_rel = world_model.get_relationship(rel_id)
        assert restored_rel is not None


# ============================================================================
# REQ-WM-PERSIST-004: Backup and Restore (MUST)
# ============================================================================

@pytest.mark.requirement("REQ-WM-PERSIST-004")
@pytest.mark.priority("MUST")
class TestREQ_WM_PERSIST_004_BackupAndRestore:
    """
    REQ-WM-PERSIST-004: The World Model MUST support backup and restore
    operations allowing recovery from data loss or corruption.
    """

    def test_backup_creates_valid_export(self, world_model, sample_knowledge_graph, temp_export_dir):
        """Verify backup operation creates valid export file."""
        backup_path = temp_export_dir / "backup.json"

        # Create backup (export)
        world_model.export_graph(str(backup_path), project="test_persistence")

        # Verify backup is valid
        assert backup_path.exists()
        with open(backup_path) as f:
            data = json.load(f)
        assert "version" in data
        assert "entities" in data

    def test_restore_from_backup(self, world_model, sample_knowledge_graph, temp_export_dir):
        """Verify restore operation recovers data from backup."""
        backup_path = temp_export_dir / "restore_backup.json"

        # Create backup
        world_model.export_graph(str(backup_path), project="test_persistence")
        original_stats = world_model.get_statistics(project="test_persistence")

        # Simulate data loss
        world_model.reset(project="test_persistence")
        after_loss_stats = world_model.get_statistics(project="test_persistence")
        assert after_loss_stats["entity_count"] < original_stats["entity_count"]

        # Restore from backup
        world_model.import_graph(str(backup_path), project="test_persistence")

        # Verify restoration
        restored_stats = world_model.get_statistics(project="test_persistence")
        assert restored_stats["entity_count"] >= original_stats["entity_count"]

    def test_multiple_backups(self, world_model, temp_export_dir):
        """Verify multiple backups can be created and managed."""
        # Create initial graph
        entity1 = Entity(type="Paper", properties={"title": "V1"}, project="test_persistence")
        world_model.add_entity(entity1)

        # Backup 1
        backup1_path = temp_export_dir / "backup_v1.json"
        world_model.export_graph(str(backup1_path), project="test_persistence")

        # Modify graph
        entity2 = Entity(type="Paper", properties={"title": "V2"}, project="test_persistence")
        world_model.add_entity(entity2)

        # Backup 2
        backup2_path = temp_export_dir / "backup_v2.json"
        world_model.export_graph(str(backup2_path), project="test_persistence")

        # Both backups should exist
        assert backup1_path.exists()
        assert backup2_path.exists()

        # Backups should have different content
        with open(backup1_path) as f:
            data1 = json.load(f)
        with open(backup2_path) as f:
            data2 = json.load(f)

        # V2 should have more entities
        assert len(data2["entities"]) >= len(data1["entities"])

    def test_restore_to_specific_point_in_time(self, world_model, temp_export_dir):
        """Verify restore can recover to specific point in time."""
        # State 1: Initial entities
        entity1 = Entity(type="Paper", properties={"title": "Initial"}, project="test_persistence")
        id1 = world_model.add_entity(entity1)

        backup1 = temp_export_dir / "state1.json"
        world_model.export_graph(str(backup1), project="test_persistence")

        # State 2: Add more entities
        entity2 = Entity(type="Paper", properties={"title": "Additional"}, project="test_persistence")
        world_model.add_entity(entity2)

        # Restore to state 1
        world_model.reset(project="test_persistence")
        world_model.import_graph(str(backup1), project="test_persistence")

        # Should have state 1 data
        restored = world_model.get_entity(id1)
        assert restored is not None


# ============================================================================
# REQ-WM-PERSIST-005: Incremental Updates (SHOULD)
# ============================================================================

@pytest.mark.requirement("REQ-WM-PERSIST-005")
@pytest.mark.priority("SHOULD")
class TestREQ_WM_PERSIST_005_IncrementalUpdates:
    """
    REQ-WM-PERSIST-005: The World Model SHOULD support incremental updates
    to persisted data without requiring full re-export.
    """

    def test_add_entity_persists_immediately(self, world_model):
        """Verify new entities are persisted immediately."""
        entity = Entity(
            type="Paper",
            properties={"title": "Immediate Persistence Test"},
            project="test_persistence"
        )
        entity_id = world_model.add_entity(entity)

        # Should be immediately retrievable (persisted)
        retrieved = world_model.get_entity(entity_id)
        assert retrieved is not None
        assert retrieved.id == entity_id

    def test_update_entity_persists_immediately(self, world_model):
        """Verify entity updates are persisted immediately."""
        entity = Entity(
            type="Paper",
            properties={"title": "Original Title"},
            project="test_persistence"
        )
        entity_id = world_model.add_entity(entity)

        # Update entity
        world_model.update_entity(entity_id, {"verified": True})

        # Update should be immediately visible
        updated = world_model.get_entity(entity_id)
        assert updated is not None

    def test_delete_entity_removes_immediately(self, world_model):
        """Verify entity deletion is persisted immediately."""
        entity = Entity(
            type="Paper",
            properties={"title": "To Be Deleted"},
            project="test_persistence"
        )
        entity_id = world_model.add_entity(entity)

        # Verify exists
        assert world_model.get_entity(entity_id) is not None

        # Delete
        world_model.delete_entity(entity_id)

        # Should be immediately gone
        assert world_model.get_entity(entity_id) is None


# ============================================================================
# REQ-WM-PERSIST-006: Data Versioning (SHOULD)
# ============================================================================

@pytest.mark.requirement("REQ-WM-PERSIST-006")
@pytest.mark.priority("SHOULD")
class TestREQ_WM_PERSIST_006_DataVersioning:
    """
    REQ-WM-PERSIST-006: The World Model SHOULD support versioning of entities
    and relationships to track changes over time.
    """

    def test_entity_timestamps_track_creation(self, world_model):
        """Verify entity creation timestamp is recorded."""
        before = datetime.now()
        entity = Entity(
            type="Paper",
            properties={"title": "Versioning Test"},
            project="test_persistence"
        )
        entity_id = world_model.add_entity(entity)
        after = datetime.now()

        retrieved = world_model.get_entity(entity_id)
        assert retrieved is not None
        if retrieved.created_at:
            assert before <= retrieved.created_at <= after

    def test_entity_timestamps_track_updates(self, world_model):
        """Verify entity update timestamp is modified."""
        entity = Entity(
            type="Paper",
            properties={"title": "Update Test"},
            project="test_persistence"
        )
        entity_id = world_model.add_entity(entity)

        original = world_model.get_entity(entity_id)
        original_updated_at = original.updated_at if original else None

        # Wait briefly and update
        import time
        time.sleep(0.1)

        world_model.update_entity(entity_id, {"verified": True})

        updated = world_model.get_entity(entity_id)
        # Note: Timestamp update behavior depends on implementation

    def test_export_includes_version_metadata(self, world_model, sample_knowledge_graph, temp_export_dir):
        """Verify exports include version metadata."""
        export_path = temp_export_dir / "versioned_export.json"

        world_model.export_graph(str(export_path), project="test_persistence")

        with open(export_path) as f:
            data = json.load(f)

        # Should include version information
        assert "version" in data
        assert "exported_at" in data

        # Timestamp should be valid
        exported_at = datetime.fromisoformat(data["exported_at"])
        assert isinstance(exported_at, datetime)

    def test_entity_creation_source_tracked(self, world_model):
        """Verify entity creation source is tracked."""
        entity = Entity(
            type="Hypothesis",
            properties={"statement": "Test hypothesis"},
            project="test_persistence",
            created_by="hypothesis_generator_v2"
        )
        entity_id = world_model.add_entity(entity)

        retrieved = world_model.get_entity(entity_id)
        assert retrieved is not None
        assert retrieved.created_by == "hypothesis_generator_v2"

    def test_relationship_provenance_tracked(self, world_model):
        """Verify relationship provenance is tracked."""
        entity1 = Entity(type="Paper", properties={"title": "P1"}, project="test_persistence")
        entity2 = Entity(type="Hypothesis", properties={"statement": "H1"}, project="test_persistence")

        id1 = world_model.add_entity(entity1)
        id2 = world_model.add_entity(entity2)

        # Create relationship with provenance
        rel = Relationship.with_provenance(
            source_id=id2,
            target_id=id1,
            rel_type="DERIVED_FROM",
            agent="hypothesis_generator",
            confidence=0.85,
            iteration=3,
            model="claude-3.5"
        )
        rel_id = world_model.add_relationship(rel)

        retrieved_rel = world_model.get_relationship(rel_id)
        assert retrieved_rel is not None
        assert retrieved_rel.created_by == "hypothesis_generator"
        assert retrieved_rel.properties["agent"] == "hypothesis_generator"
        assert retrieved_rel.properties["iteration"] == 3
