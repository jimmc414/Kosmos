"""
Tests for World Model Schema Requirements (REQ-WM-SCHEMA-001 through REQ-WM-SCHEMA-006).

These tests validate the schema definition, validation, and enforcement for the
knowledge graph including entity types, relationship types, properties, and constraints.
"""

import pytest
import uuid
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock

from kosmos.world_model.models import Entity, Relationship, Annotation, EXPORT_FORMAT_VERSION
from kosmos.world_model import get_world_model, reset_world_model

# Test markers for requirements traceability
pytestmark = [
    pytest.mark.requirement("REQ-WM-SCHEMA"),
    pytest.mark.category("world_model"),
]


# ============================================================================
# REQ-WM-SCHEMA-001: Entity Type Definition (MUST)
# ============================================================================

@pytest.mark.requirement("REQ-WM-SCHEMA-001")
@pytest.mark.priority("MUST")
class TestREQ_WM_SCHEMA_001_EntityTypeDefinition:
    """
    REQ-WM-SCHEMA-001: The World Model MUST support well-defined entity types
    including Paper, Concept, Author, Method, Experiment, Hypothesis, Finding,
    Dataset, ResearchQuestion, ExperimentProtocol, and ExperimentResult.
    """

    def test_standard_entity_types_defined(self):
        """Verify all standard entity types are defined in Entity model."""
        expected_types = {
            "Paper", "Concept", "Author", "Method",
            "Experiment", "Hypothesis", "Finding", "Dataset",
            "ResearchQuestion", "ExperimentProtocol", "ExperimentResult"
        }

        # Check Entity.VALID_TYPES includes all expected types
        assert expected_types.issubset(Entity.VALID_TYPES), \
            f"Missing entity types: {expected_types - Entity.VALID_TYPES}"

    def test_entity_creation_with_valid_types(self):
        """Verify entities can be created with all standard types."""
        standard_types = ["Paper", "Concept", "Author", "Method",
                         "Hypothesis", "Finding", "Dataset"]

        for entity_type in standard_types:
            entity = Entity(
                type=entity_type,
                properties={"name": f"Test {entity_type}"}
            )
            assert entity.type == entity_type
            assert entity.id is not None
            assert isinstance(entity.properties, dict)

    def test_entity_creation_with_custom_type_warning(self):
        """Verify custom entity types generate warning but are allowed."""
        with pytest.warns(UserWarning, match="not a standard type"):
            entity = Entity(
                type="CustomType",
                properties={"custom_prop": "value"}
            )
            assert entity.type == "CustomType"
            assert entity.id is not None

    def test_entity_type_required(self):
        """Verify entity type is required and cannot be empty."""
        with pytest.raises(ValueError, match="Entity type is required"):
            Entity(type="", properties={})

    def test_entity_properties_must_be_dict(self):
        """Verify entity properties must be a dictionary."""
        with pytest.raises(ValueError, match="Properties must be a dictionary"):
            Entity(type="Paper", properties="not a dict")  # type: ignore


# ============================================================================
# REQ-WM-SCHEMA-002: Relationship Type Definition (MUST)
# ============================================================================

@pytest.mark.requirement("REQ-WM-SCHEMA-002")
@pytest.mark.priority("MUST")
class TestREQ_WM_SCHEMA_002_RelationshipTypeDefinition:
    """
    REQ-WM-SCHEMA-002: The World Model MUST support well-defined relationship types
    including CITES, AUTHOR_OF, MENTIONS, RELATES_TO, SUPPORTS, REFUTES, USES_METHOD,
    PRODUCED_BY, DERIVED_FROM, SPAWNED_BY, TESTS, and REFINED_FROM.
    """

    def test_standard_relationship_types_defined(self):
        """Verify all standard relationship types are defined."""
        expected_types = {
            "CITES", "AUTHOR_OF", "MENTIONS", "RELATES_TO",
            "SUPPORTS", "REFUTES", "USES_METHOD", "PRODUCED_BY",
            "DERIVED_FROM", "SPAWNED_BY", "TESTS", "REFINED_FROM"
        }

        assert expected_types.issubset(Relationship.VALID_TYPES), \
            f"Missing relationship types: {expected_types - Relationship.VALID_TYPES}"

    def test_relationship_creation_with_valid_types(self):
        """Verify relationships can be created with all standard types."""
        source_id = str(uuid.uuid4())
        target_id = str(uuid.uuid4())

        standard_types = ["CITES", "MENTIONS", "SUPPORTS", "REFUTES"]

        for rel_type in standard_types:
            rel = Relationship(
                source_id=source_id,
                target_id=target_id,
                type=rel_type
            )
            assert rel.type == rel_type
            assert rel.source_id == source_id
            assert rel.target_id == target_id
            assert rel.id is not None

    def test_relationship_custom_type_warning(self):
        """Verify custom relationship types generate warning."""
        with pytest.warns(UserWarning, match="not standard"):
            rel = Relationship(
                source_id=str(uuid.uuid4()),
                target_id=str(uuid.uuid4()),
                type="CUSTOM_RELATION"
            )
            assert rel.type == "CUSTOM_RELATION"

    def test_relationship_requires_source_and_target(self):
        """Verify relationships require both source and target IDs."""
        with pytest.raises(ValueError, match="Source and target IDs are required"):
            Relationship(source_id="", target_id="test", type="CITES")

        with pytest.raises(ValueError, match="Source and target IDs are required"):
            Relationship(source_id="test", target_id="", type="CITES")

    def test_relationship_type_required(self):
        """Verify relationship type is required."""
        with pytest.raises(ValueError, match="Relationship type is required"):
            Relationship(
                source_id=str(uuid.uuid4()),
                target_id=str(uuid.uuid4()),
                type=""
            )


# ============================================================================
# REQ-WM-SCHEMA-003: Property Validation (MUST)
# ============================================================================

@pytest.mark.requirement("REQ-WM-SCHEMA-003")
@pytest.mark.priority("MUST")
class TestREQ_WM_SCHEMA_003_PropertyValidation:
    """
    REQ-WM-SCHEMA-003: The World Model MUST validate entity and relationship
    properties including type checking, required fields, and value constraints.
    """

    def test_entity_confidence_score_validation(self):
        """Verify confidence scores are validated between 0.0 and 1.0."""
        # Valid confidence scores
        for score in [0.0, 0.5, 1.0]:
            entity = Entity(type="Paper", properties={}, confidence=score)
            assert entity.confidence == score

        # Invalid confidence scores
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            Entity(type="Paper", properties={}, confidence=1.5)

        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            Entity(type="Paper", properties={}, confidence=-0.1)

    def test_relationship_confidence_score_validation(self):
        """Verify relationship confidence scores are validated."""
        source_id = str(uuid.uuid4())
        target_id = str(uuid.uuid4())

        # Valid confidence
        rel = Relationship(
            source_id=source_id,
            target_id=target_id,
            type="CITES",
            confidence=0.95
        )
        assert rel.confidence == 0.95

        # Invalid confidence
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            Relationship(
                source_id=source_id,
                target_id=target_id,
                type="CITES",
                confidence=2.0
            )

    def test_entity_timestamps_auto_generated(self):
        """Verify timestamps are automatically generated."""
        before = datetime.now()
        entity = Entity(type="Paper", properties={})
        after = datetime.now()

        assert entity.created_at is not None
        assert entity.updated_at is not None
        assert before <= entity.created_at <= after
        assert entity.created_at <= entity.updated_at <= after

    def test_entity_id_auto_generated(self):
        """Verify entity IDs are automatically generated if not provided."""
        entity = Entity(type="Paper", properties={})
        assert entity.id is not None
        assert isinstance(entity.id, str)
        assert len(entity.id) > 0

        # Verify it's a valid UUID format
        try:
            uuid.UUID(entity.id)
        except ValueError:
            pytest.fail("Entity ID is not a valid UUID")

    def test_annotation_validation(self):
        """Verify annotation validation."""
        # Valid annotation
        ann = Annotation(text="This is a note", created_by="researcher@example.com")
        assert ann.text == "This is a note"
        assert ann.created_by == "researcher@example.com"
        assert ann.created_at is not None

        # Empty text not allowed
        with pytest.raises(ValueError, match="Annotation text cannot be empty"):
            Annotation(text="", created_by="researcher@example.com")

        # Empty creator not allowed
        with pytest.raises(ValueError, match="Annotation must have a creator"):
            Annotation(text="Valid text", created_by="")


# ============================================================================
# REQ-WM-SCHEMA-004: Data Model Serialization (MUST)
# ============================================================================

@pytest.mark.requirement("REQ-WM-SCHEMA-004")
@pytest.mark.priority("MUST")
class TestREQ_WM_SCHEMA_004_DataModelSerialization:
    """
    REQ-WM-SCHEMA-004: The World Model MUST support serialization and
    deserialization of entities and relationships for export/import operations.
    """

    def test_entity_to_dict_serialization(self):
        """Verify entity can be serialized to dictionary."""
        entity = Entity(
            type="Paper",
            properties={"title": "Test Paper", "year": 2024},
            confidence=0.95,
            project="test_project",
            created_by="test_agent"
        )

        entity_dict = entity.to_dict()

        assert isinstance(entity_dict, dict)
        assert entity_dict["type"] == "Paper"
        assert entity_dict["properties"] == {"title": "Test Paper", "year": 2024}
        assert entity_dict["confidence"] == 0.95
        assert entity_dict["project"] == "test_project"
        assert entity_dict["created_by"] == "test_agent"
        assert "created_at" in entity_dict
        assert "id" in entity_dict

    def test_entity_from_dict_deserialization(self):
        """Verify entity can be deserialized from dictionary."""
        entity_dict = {
            "id": str(uuid.uuid4()),
            "type": "Concept",
            "properties": {"name": "Machine Learning"},
            "confidence": 0.9,
            "project": "ml_research",
            "created_at": "2024-01-15T10:30:00",
            "updated_at": "2024-01-15T10:30:00",
            "created_by": "concept_extractor",
            "verified": True,
            "annotations": []
        }

        entity = Entity.from_dict(entity_dict)

        assert entity.id == entity_dict["id"]
        assert entity.type == "Concept"
        assert entity.properties["name"] == "Machine Learning"
        assert entity.confidence == 0.9
        assert entity.project == "ml_research"
        assert entity.verified is True

    def test_relationship_to_dict_serialization(self):
        """Verify relationship can be serialized to dictionary."""
        rel = Relationship(
            source_id="entity1",
            target_id="entity2",
            type="CITES",
            properties={"context": "introduction"},
            confidence=1.0,
            created_by="citation_extractor"
        )

        rel_dict = rel.to_dict()

        assert isinstance(rel_dict, dict)
        assert rel_dict["source_id"] == "entity1"
        assert rel_dict["target_id"] == "entity2"
        assert rel_dict["type"] == "CITES"
        assert rel_dict["properties"]["context"] == "introduction"
        assert rel_dict["confidence"] == 1.0

    def test_relationship_from_dict_deserialization(self):
        """Verify relationship can be deserialized from dictionary."""
        rel_dict = {
            "id": str(uuid.uuid4()),
            "source_id": "paper1",
            "target_id": "paper2",
            "type": "MENTIONS",
            "properties": {"section": "methods"},
            "confidence": 0.85,
            "created_at": "2024-01-15T11:00:00",
            "created_by": "text_analyzer"
        }

        rel = Relationship.from_dict(rel_dict)

        assert rel.id == rel_dict["id"]
        assert rel.source_id == "paper1"
        assert rel.target_id == "paper2"
        assert rel.type == "MENTIONS"
        assert rel.properties["section"] == "methods"

    def test_entity_with_annotations_serialization(self):
        """Verify entity with annotations can be serialized."""
        entity = Entity(
            type="Paper",
            properties={"title": "Annotated Paper"},
            annotations=[
                Annotation(text="Important paper", created_by="reviewer1"),
                Annotation(text="Check references", created_by="reviewer2")
            ]
        )

        entity_dict = entity.to_dict()

        assert len(entity_dict["annotations"]) == 2
        assert entity_dict["annotations"][0]["text"] == "Important paper"
        assert entity_dict["annotations"][1]["created_by"] == "reviewer2"


# ============================================================================
# REQ-WM-SCHEMA-005: Version Compatibility (MUST)
# ============================================================================

@pytest.mark.requirement("REQ-WM-SCHEMA-005")
@pytest.mark.priority("MUST")
class TestREQ_WM_SCHEMA_005_VersionCompatibility:
    """
    REQ-WM-SCHEMA-005: The World Model MUST maintain version compatibility
    for schema changes to support backward compatibility of exported graphs.
    """

    def test_export_format_version_defined(self):
        """Verify export format version is defined."""
        assert EXPORT_FORMAT_VERSION is not None
        assert isinstance(EXPORT_FORMAT_VERSION, str)
        assert len(EXPORT_FORMAT_VERSION) > 0

    def test_export_format_version_semantic(self):
        """Verify export format uses semantic versioning."""
        # Should be in format "X.Y" or "X.Y.Z"
        parts = EXPORT_FORMAT_VERSION.split(".")
        assert len(parts) in [2, 3], \
            f"Version should be semantic (e.g., '1.0' or '1.0.0'), got: {EXPORT_FORMAT_VERSION}"

        # Each part should be numeric
        for part in parts:
            assert part.isdigit(), \
                f"Version parts should be numeric, got: {EXPORT_FORMAT_VERSION}"

    def test_entity_serialization_includes_schema_metadata(self):
        """Verify serialized entities include necessary metadata for versioning."""
        entity = Entity(
            type="Paper",
            properties={"title": "Version Test"}
        )

        entity_dict = entity.to_dict()

        # Should include type information for schema validation
        assert "type" in entity_dict
        assert "properties" in entity_dict
        assert "confidence" in entity_dict
        assert "created_at" in entity_dict

    def test_backward_compatible_deserialization(self):
        """Verify entities can be deserialized from older format."""
        # Simulate old format (minimal fields)
        old_format = {
            "id": str(uuid.uuid4()),
            "type": "Paper",
            "properties": {"title": "Old Format Paper"}
            # Missing: confidence, project, timestamps, verified, annotations
        }

        # Should deserialize successfully with defaults
        entity = Entity.from_dict(old_format)

        assert entity.id == old_format["id"]
        assert entity.type == "Paper"
        assert entity.properties["title"] == "Old Format Paper"
        assert entity.confidence == 1.0  # Default
        assert entity.project is None  # Default
        assert entity.verified is False  # Default
        assert len(entity.annotations) == 0  # Default


# ============================================================================
# REQ-WM-SCHEMA-006: Referential Integrity (MUST)
# ============================================================================

@pytest.mark.requirement("REQ-WM-SCHEMA-006")
@pytest.mark.priority("MUST")
class TestREQ_WM_SCHEMA_006_ReferentialIntegrity:
    """
    REQ-WM-SCHEMA-006: The World Model MUST enforce referential integrity
    ensuring relationships reference valid entities.
    """

    def test_relationship_requires_valid_entity_ids(self):
        """Verify relationships require non-empty entity IDs."""
        # Valid relationship
        rel = Relationship(
            source_id="valid_id_1",
            target_id="valid_id_2",
            type="CITES"
        )
        assert rel.source_id == "valid_id_1"
        assert rel.target_id == "valid_id_2"

        # Invalid: empty source
        with pytest.raises(ValueError):
            Relationship(source_id="", target_id="valid", type="CITES")

        # Invalid: empty target
        with pytest.raises(ValueError):
            Relationship(source_id="valid", target_id="", type="CITES")

    def test_relationship_id_validation_at_schema_level(self):
        """Verify relationship ID validation at model level."""
        # Valid UUID IDs
        entity1_id = str(uuid.uuid4())
        entity2_id = str(uuid.uuid4())

        rel = Relationship(
            source_id=entity1_id,
            target_id=entity2_id,
            type="MENTIONS"
        )

        # IDs should be stored as provided
        assert rel.source_id == entity1_id
        assert rel.target_id == entity2_id
        assert rel.id is not None
        assert isinstance(rel.id, str)

    def test_entity_from_hypothesis_preserves_relationships(self):
        """Verify Entity.from_hypothesis preserves relationship data."""
        from unittest.mock import Mock

        # Create mock hypothesis
        hypothesis = Mock()
        hypothesis.id = str(uuid.uuid4())
        hypothesis.research_question = "Test question"
        hypothesis.statement = "Test hypothesis"
        hypothesis.rationale = "Test rationale"
        hypothesis.domain = "test_domain"
        hypothesis.status = Mock()
        hypothesis.status.value = "active"
        hypothesis.testability_score = 0.8
        hypothesis.novelty_score = 0.7
        hypothesis.confidence_score = 0.9
        hypothesis.priority_score = 0.85
        hypothesis.parent_hypothesis_id = None
        hypothesis.generation = 1
        hypothesis.refinement_count = 0
        hypothesis.related_papers = ["paper1", "paper2"]
        hypothesis.created_at = datetime.now()
        hypothesis.updated_at = datetime.now()

        entity = Entity.from_hypothesis(hypothesis, created_by="test_agent")

        # Verify entity structure
        assert entity.id == hypothesis.id
        assert entity.type == "Hypothesis"
        assert entity.properties["statement"] == "Test hypothesis"
        assert entity.properties["related_papers"] == ["paper1", "paper2"]
        # This data will be used to create relationships
        assert "related_papers" in entity.properties

    def test_relationship_with_provenance_includes_metadata(self):
        """Verify provenance relationships include necessary metadata."""
        entity1_id = str(uuid.uuid4())
        entity2_id = str(uuid.uuid4())

        rel = Relationship.with_provenance(
            source_id=entity1_id,
            target_id=entity2_id,
            rel_type="SUPPORTS",
            agent="DataAnalystAgent",
            confidence=0.95,
            p_value=0.001,
            effect_size=0.78
        )

        assert rel.source_id == entity1_id
        assert rel.target_id == entity2_id
        assert rel.type == "SUPPORTS"
        assert rel.created_by == "DataAnalystAgent"
        assert rel.confidence == 0.95
        assert rel.properties["agent"] == "DataAnalystAgent"
        assert rel.properties["p_value"] == 0.001
        assert rel.properties["effect_size"] == 0.78
        assert "timestamp" in rel.properties
