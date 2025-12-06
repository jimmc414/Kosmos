"""
E2E Tests for World Model Persistence.

Tests the world model entity and relationship persistence:
- Entity persistence across cycles
- Relationship creation (SPAWNED_BY, TESTS, SUPPORTS)
- Hypothesis → Entity conversion
- Result → Entity conversion
- Export/Import roundtrip
- Statistics accuracy
- Multi-project isolation
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
from kosmos.world_model.models import Entity, Relationship, Annotation
from tests.e2e.factories import EntityFactory, RelationshipFactory, ResearchScenarioFactory
from tests.e2e.conftest import requires_neo4j


pytestmark = [pytest.mark.e2e]


class TestEntityPersistence:
    """Tests for entity persistence across research cycles."""

    def test_entity_persistence_across_cycles(self, in_memory_world_model):
        """Verify entities from cycle 1 are available in cycle 2."""
        wm = in_memory_world_model

        # Cycle 1: Create hypothesis entity
        hypothesis_entity = EntityFactory.create_hypothesis_entity(
            hypothesis_id="hyp_001",
            statement="Temperature affects enzyme activity",
            domain="biology"
        )
        wm.add_entity(hypothesis_entity)

        # Cycle 2: Create experiment testing hypothesis
        protocol_entity = EntityFactory.create_protocol_entity(
            protocol_id="proto_001",
            hypothesis_id="hyp_001",
            name="Temperature experiment"
        )
        wm.add_entity(protocol_entity)

        # Create TESTS relationship
        relationship = RelationshipFactory.create_tests_relationship(
            source_id=protocol_entity.id,
            target_id=hypothesis_entity.id,
            iteration=1
        )
        wm.add_relationship(relationship)

        # Verify: Hypothesis is still available
        retrieved = wm.get_entity("hyp_001")
        assert retrieved is not None
        assert retrieved.id == "hyp_001"
        assert retrieved.properties["statement"] == "Temperature affects enzyme activity"

        # Verify: Protocol exists
        protocol = wm.get_entity("proto_001")
        assert protocol is not None

        # Verify: Statistics
        stats = wm.get_statistics()
        assert stats['entity_count'] == 2
        assert stats['relationship_count'] == 1

    def test_entity_merge_on_duplicate(self, in_memory_world_model):
        """Verify duplicate entities are merged correctly."""
        wm = in_memory_world_model

        # Add entity
        entity1 = Entity(
            id="entity_001",
            type="Hypothesis",
            properties={"statement": "Initial statement"},
            confidence=0.5
        )
        wm.add_entity(entity1, merge=True)

        # Add duplicate with updated properties
        entity2 = Entity(
            id="entity_001",
            type="Hypothesis",
            properties={"statement": "Updated statement", "extra": "value"},
            confidence=0.9
        )
        wm.add_entity(entity2, merge=True)

        # Verify merge
        result = wm.get_entity("entity_001")
        assert result.properties["statement"] == "Updated statement"
        assert result.properties.get("extra") == "value"
        assert result.confidence == 0.9


class TestRelationshipCreation:
    """Tests for relationship creation."""

    def test_spawned_by_relationship(self):
        """Verify SPAWNED_BY relationship creation."""
        relationship = RelationshipFactory.create_spawned_by_relationship(
            source_id="hyp_001",
            target_id="question_001",
            generation=1
        )

        assert relationship.type == "SPAWNED_BY"
        assert relationship.source_id == "hyp_001"
        assert relationship.target_id == "question_001"
        assert relationship.properties["generation"] == 1

    def test_tests_relationship(self):
        """Verify TESTS relationship creation."""
        relationship = RelationshipFactory.create_tests_relationship(
            source_id="proto_001",
            target_id="hyp_001",
            iteration=2
        )

        assert relationship.type == "TESTS"
        assert relationship.source_id == "proto_001"
        assert relationship.target_id == "hyp_001"
        assert relationship.properties["iteration"] == 2

    def test_supports_relationship(self):
        """Verify SUPPORTS relationship with provenance."""
        relationship = RelationshipFactory.create_supports_relationship(
            source_id="result_001",
            target_id="hyp_001",
            p_value=0.001,
            effect_size=0.85
        )

        assert relationship.type == "SUPPORTS"
        assert relationship.properties["p_value"] == 0.001
        assert relationship.properties["effect_size"] == 0.85
        assert relationship.confidence == 0.95

    def test_relationship_with_provenance(self):
        """Verify Relationship.with_provenance factory method."""
        rel = Relationship.with_provenance(
            source_id="result_001",
            target_id="hyp_001",
            rel_type="SUPPORTS",
            agent="DataAnalystAgent",
            confidence=0.95,
            p_value=0.001,
            effect_size=0.78,
            iteration=3
        )

        assert rel.type == "SUPPORTS"
        assert rel.properties["agent"] == "DataAnalystAgent"
        assert rel.properties["p_value"] == 0.001
        assert rel.properties["iteration"] == 3
        assert rel.confidence == 0.95


class TestEntityConversion:
    """Tests for entity conversion from domain models."""

    def test_hypothesis_to_entity_flow(self, sample_hypothesis):
        """Verify Hypothesis → Entity conversion."""
        entity = Entity.from_hypothesis(
            sample_hypothesis,
            created_by="HypothesisGeneratorAgent"
        )

        assert entity.id == sample_hypothesis.id
        assert entity.type == "Hypothesis"
        assert entity.properties["statement"] == sample_hypothesis.statement
        assert entity.properties["domain"] == sample_hypothesis.domain
        assert entity.properties["research_question"] == sample_hypothesis.research_question
        assert entity.created_by == "HypothesisGeneratorAgent"

        # Verify scores are included
        assert entity.properties.get("testability_score") == sample_hypothesis.testability_score
        assert entity.properties.get("novelty_score") == sample_hypothesis.novelty_score

    def test_result_to_entity_flow(self, sample_experiment_result):
        """Verify ExperimentResult → Entity conversion."""
        entity = Entity.from_result(
            sample_experiment_result,
            created_by="Executor"
        )

        assert entity.id == sample_experiment_result.id
        assert entity.type == "ExperimentResult"
        assert entity.properties["experiment_id"] == sample_experiment_result.experiment_id
        assert entity.properties["supports_hypothesis"] == sample_experiment_result.supports_hypothesis
        assert entity.created_by == "Executor"

    def test_research_question_to_entity(self):
        """Verify research question → Entity conversion."""
        entity = Entity.from_research_question(
            question_text="How do transformers learn long-range dependencies?",
            domain="machine_learning",
            created_by="ResearchDirectorAgent"
        )

        assert entity.type == "ResearchQuestion"
        assert entity.properties["text"] == "How do transformers learn long-range dependencies?"
        assert entity.properties["domain"] == "machine_learning"
        assert entity.created_by == "ResearchDirectorAgent"


class TestWorldModelExportImport:
    """Tests for export/import roundtrip."""

    def test_world_model_export_import_roundtrip(self, e2e_artifacts_dir):
        """Verify full graph export/import roundtrip."""
        # Create test entities
        entities = []
        relationships = []

        # Add entities
        hypothesis = EntityFactory.create_hypothesis_entity(
            hypothesis_id="hyp_export_001",
            statement="Export test hypothesis"
        )
        entities.append(hypothesis)

        result = EntityFactory.create_result_entity(
            result_id="result_export_001",
            hypothesis_id="hyp_export_001"
        )
        entities.append(result)

        # Add relationship
        rel = RelationshipFactory.create_supports_relationship(
            source_id="result_export_001",
            target_id="hyp_export_001"
        )
        relationships.append(rel)

        # Create export data
        export_data = {
            "version": "1.0",
            "exported_at": datetime.utcnow().isoformat(),
            "source": "kosmos",
            "mode": "simple",
            "project": None,
            "statistics": {
                "entity_count": len(entities),
                "relationship_count": len(relationships)
            },
            "entities": [e.to_dict() for e in entities],
            "relationships": [r.to_dict() for r in relationships]
        }

        # Export to file
        export_file = e2e_artifacts_dir / "test_export.json"
        with open(export_file, "w") as f:
            json.dump(export_data, f, indent=2)

        # Import from file
        with open(export_file, "r") as f:
            imported_data = json.load(f)

        # Verify data integrity
        assert imported_data["version"] == "1.0"
        assert len(imported_data["entities"]) == 2
        assert len(imported_data["relationships"]) == 1

        # Reconstruct entities
        imported_entities = [Entity.from_dict(e) for e in imported_data["entities"]]
        assert len(imported_entities) == 2

        # Find hypothesis
        imported_hyp = next(e for e in imported_entities if e.type == "Hypothesis")
        assert imported_hyp.id == "hyp_export_001"
        assert imported_hyp.properties["statement"] == "Export test hypothesis"

        # Reconstruct relationships
        imported_rels = [Relationship.from_dict(r) for r in imported_data["relationships"]]
        assert len(imported_rels) == 1
        assert imported_rels[0].type == "SUPPORTS"


class TestWorldModelStatistics:
    """Tests for statistics accuracy."""

    def test_world_model_statistics_accuracy(self, in_memory_world_model):
        """Verify statistics are accurate after operations."""
        wm = in_memory_world_model

        # Add various entities
        for i in range(3):
            wm.add_entity(EntityFactory.create_hypothesis_entity(
                hypothesis_id=f"hyp_{i:03d}"
            ))

        for i in range(2):
            wm.add_entity(EntityFactory.create_result_entity(
                result_id=f"result_{i:03d}"
            ))

        # Add relationships
        wm.add_relationship(RelationshipFactory.create_tests_relationship(
            source_id="proto_001",
            target_id="hyp_001"
        ))
        wm.add_relationship(RelationshipFactory.create_supports_relationship(
            source_id="result_001",
            target_id="hyp_001"
        ))

        # Verify statistics
        stats = wm.get_statistics()

        assert stats["entity_count"] == 5
        assert stats["relationship_count"] == 2
        assert stats["entity_types"]["Hypothesis"] == 3
        assert stats["entity_types"]["ExperimentResult"] == 2
        assert stats["relationship_types"]["TESTS"] == 1
        assert stats["relationship_types"]["SUPPORTS"] == 1


class TestMultiProjectIsolation:
    """Tests for multi-project data isolation."""

    def test_multi_project_isolation(self, in_memory_world_model):
        """Verify projects don't leak data."""
        wm = in_memory_world_model

        # Add entities to project A
        entity_a = Entity(
            id="entity_project_a",
            type="Hypothesis",
            properties={"statement": "Project A hypothesis"},
            project="project_a"
        )
        wm.add_entity(entity_a)

        # Add entities to project B
        entity_b = Entity(
            id="entity_project_b",
            type="Hypothesis",
            properties={"statement": "Project B hypothesis"},
            project="project_b"
        )
        wm.add_entity(entity_b)

        # Verify isolation
        # Project A should only see its entities
        stats_a = wm.get_statistics(project="project_a")
        assert stats_a["entity_count"] == 1

        # Project B should only see its entities
        stats_b = wm.get_statistics(project="project_b")
        assert stats_b["entity_count"] == 1

        # Get entity with wrong project filter
        result = wm.get_entity("entity_project_a", project="project_b")
        assert result is None

        # Get entity with correct project filter
        result = wm.get_entity("entity_project_a", project="project_a")
        assert result is not None
        assert result.id == "entity_project_a"


@requires_neo4j
class TestNeo4jWorldModelIntegration:
    """Integration tests requiring Neo4j."""

    def test_neo4j_entity_persistence(self):
        """Test entity persistence with real Neo4j."""
        from kosmos.world_model.factory import get_world_model, reset_world_model

        try:
            wm = get_world_model()

            # Create test entity
            entity = EntityFactory.create_hypothesis_entity(
                hypothesis_id=f"hyp_neo4j_test_{datetime.utcnow().timestamp()}"
            )

            # Add entity
            entity_id = wm.add_entity(entity)

            # Retrieve entity
            retrieved = wm.get_entity(entity_id)

            # Verify
            assert retrieved is not None
            assert retrieved.id == entity_id

            # Cleanup
            wm.delete_entity(entity_id)

        finally:
            reset_world_model()

    def test_neo4j_relationship_traversal(self):
        """Test relationship traversal with real Neo4j."""
        from kosmos.world_model.factory import get_world_model, reset_world_model

        try:
            wm = get_world_model()

            timestamp = datetime.utcnow().timestamp()

            # Create entities
            hyp_entity = EntityFactory.create_hypothesis_entity(
                hypothesis_id=f"hyp_traversal_{timestamp}"
            )
            proto_entity = EntityFactory.create_protocol_entity(
                protocol_id=f"proto_traversal_{timestamp}",
                hypothesis_id=hyp_entity.id
            )

            wm.add_entity(hyp_entity)
            wm.add_entity(proto_entity)

            # Create relationship
            rel = RelationshipFactory.create_tests_relationship(
                source_id=proto_entity.id,
                target_id=hyp_entity.id
            )
            wm.add_relationship(rel)

            # Query related entities
            related = wm.query_related_entities(
                entity_id=hyp_entity.id,
                relationship_type="TESTS",
                direction="incoming"
            )

            # Verify
            assert len(related) >= 1

            # Cleanup
            wm.delete_entity(hyp_entity.id)
            wm.delete_entity(proto_entity.id)

        finally:
            reset_world_model()
