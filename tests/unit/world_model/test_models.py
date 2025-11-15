"""Tests for world model data models."""

import json
import pytest
from datetime import datetime

from kosmos.world_model.models import Annotation, Entity, Relationship


class TestAnnotation:
    """Test Annotation model."""

    def test_create_annotation(self):
        """Test creating valid annotation."""
        ann = Annotation(text="Test annotation", created_by="test@example.com")

        assert ann.text == "Test annotation"
        assert ann.created_by == "test@example.com"
        assert isinstance(ann.created_at, datetime)

    def test_empty_text_raises_error(self):
        """Test that empty text raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Annotation(text="", created_by="test@example.com")

        with pytest.raises(ValueError, match="cannot be empty"):
            Annotation(text="   ", created_by="test@example.com")

    def test_missing_creator_raises_error(self):
        """Test that missing creator raises error."""
        with pytest.raises(ValueError, match="must have a creator"):
            Annotation(text="Test", created_by="")

        with pytest.raises(ValueError, match="must have a creator"):
            Annotation(text="Test", created_by="   ")

    def test_annotation_with_timestamp(self):
        """Test annotation with explicit timestamp."""
        now = datetime.now()
        ann = Annotation(
            text="Test",
            created_by="user",
            created_at=now
        )
        assert ann.created_at == now


class TestEntity:
    """Test Entity model."""

    def test_create_entity_minimal(self):
        """Test creating entity with minimal fields."""
        entity = Entity(type="Paper", properties={"title": "Test Paper"})

        assert entity.type == "Paper"
        assert entity.properties["title"] == "Test Paper"
        assert entity.id is not None  # Auto-generated
        assert entity.confidence == 1.0  # Default
        assert isinstance(entity.created_at, datetime)
        assert entity.created_at == entity.updated_at
        assert entity.verified is False
        assert entity.annotations == []

    def test_create_entity_full(self):
        """Test creating entity with all fields."""
        entity = Entity(
            type="Paper",
            properties={"title": "Test", "year": 2024},
            confidence=0.95,
            project="test_project",
            created_by="test_agent",
            verified=True,
        )

        assert entity.confidence == 0.95
        assert entity.project == "test_project"
        assert entity.created_by == "test_agent"
        assert entity.verified is True

    def test_auto_generate_id(self):
        """Test that ID is auto-generated."""
        entity1 = Entity(type="Paper", properties={})
        entity2 = Entity(type="Paper", properties={})

        assert entity1.id is not None
        assert entity2.id is not None
        assert entity1.id != entity2.id  # Different IDs

    def test_preserve_provided_id(self):
        """Test that provided ID is preserved."""
        entity = Entity(
            id="custom-id",
            type="Paper",
            properties={}
        )
        assert entity.id == "custom-id"

    def test_invalid_confidence_raises_error(self):
        """Test that invalid confidence raises error."""
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            Entity(type="Paper", properties={}, confidence=1.5)

        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            Entity(type="Paper", properties={}, confidence=-0.1)

    def test_missing_type_raises_error(self):
        """Test that missing type raises error."""
        with pytest.raises(ValueError, match="type is required"):
            Entity(type="", properties={})

    def test_invalid_properties_raises_error(self):
        """Test that non-dict properties raise error."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            Entity(type="Paper", properties="not a dict")  # type: ignore

    def test_standard_entity_types(self):
        """Test that standard types don't warn."""
        for entity_type in Entity.VALID_TYPES:
            # Should not raise warnings
            entity = Entity(type=entity_type, properties={})
            assert entity.type == entity_type

    def test_non_standard_type_warns(self):
        """Test that non-standard type issues warning."""
        with pytest.warns(UserWarning, match="not a standard type"):
            Entity(type="CustomType", properties={})

    def test_entity_with_annotations(self):
        """Test entity with annotations."""
        ann1 = Annotation(text="Note 1", created_by="user1")
        ann2 = Annotation(text="Note 2", created_by="user2")

        entity = Entity(
            type="Paper",
            properties={},
            annotations=[ann1, ann2]
        )

        assert len(entity.annotations) == 2
        assert entity.annotations[0].text == "Note 1"
        assert entity.annotations[1].text == "Note 2"

    def test_to_dict(self):
        """Test entity serialization."""
        entity = Entity(
            id="test_id",
            type="Paper",
            properties={"title": "Test", "year": 2024},
            confidence=0.9,
            project="proj1",
            created_by="agent",
            verified=True,
        )

        data = entity.to_dict()

        assert data["id"] == "test_id"
        assert data["type"] == "Paper"
        assert data["properties"]["title"] == "Test"
        assert data["properties"]["year"] == 2024
        assert data["confidence"] == 0.9
        assert data["project"] == "proj1"
        assert data["created_by"] == "agent"
        assert data["verified"] is True
        assert "created_at" in data
        assert "updated_at" in data
        assert data["annotations"] == []

    def test_to_dict_with_annotations(self):
        """Test serialization with annotations."""
        ann = Annotation(text="Test note", created_by="user")
        entity = Entity(
            type="Paper",
            properties={},
            annotations=[ann]
        )

        data = entity.to_dict()

        assert len(data["annotations"]) == 1
        assert data["annotations"][0]["text"] == "Test note"
        assert data["annotations"][0]["created_by"] == "user"
        assert "created_at" in data["annotations"][0]

    def test_from_dict(self):
        """Test entity deserialization."""
        data = {
            "id": "test_id",
            "type": "Paper",
            "properties": {"title": "Test"},
            "confidence": 0.9,
            "project": "proj1",
            "created_at": "2024-01-15T10:30:00",
            "updated_at": "2024-01-15T11:00:00",
            "created_by": "agent",
            "verified": True,
            "annotations": [],
        }

        entity = Entity.from_dict(data)

        assert entity.id == "test_id"
        assert entity.type == "Paper"
        assert entity.properties["title"] == "Test"
        assert entity.confidence == 0.9
        assert entity.project == "proj1"
        assert entity.created_by == "agent"
        assert entity.verified is True
        assert isinstance(entity.created_at, datetime)
        assert isinstance(entity.updated_at, datetime)

    def test_from_dict_with_annotations(self):
        """Test deserialization with annotations."""
        data = {
            "id": "test_id",
            "type": "Paper",
            "properties": {},
            "annotations": [
                {
                    "text": "Test note",
                    "created_by": "user",
                    "created_at": "2024-01-15T10:30:00"
                }
            ],
        }

        entity = Entity.from_dict(data)

        assert len(entity.annotations) == 1
        assert entity.annotations[0].text == "Test note"
        assert entity.annotations[0].created_by == "user"
        assert isinstance(entity.annotations[0].created_at, datetime)

    def test_roundtrip_serialization(self):
        """Test that to_dict → from_dict preserves data."""
        original = Entity(
            id="test_id",
            type="Concept",
            properties={"name": "Machine Learning", "field": "AI"},
            confidence=0.95,
            project="ai_research",
            created_by="extractor",
            verified=True,
        )

        # Serialize
        data = original.to_dict()
        json_str = json.dumps(data)  # Ensure JSON-serializable

        # Deserialize
        restored = Entity.from_dict(json.loads(json_str))

        assert restored.id == original.id
        assert restored.type == original.type
        assert restored.properties == original.properties
        assert restored.confidence == original.confidence
        assert restored.project == original.project
        assert restored.created_by == original.created_by
        assert restored.verified == original.verified


class TestRelationship:
    """Test Relationship model."""

    def test_create_relationship_minimal(self):
        """Test creating relationship with minimal fields."""
        rel = Relationship(
            source_id="entity1",
            target_id="entity2",
            type="CITES"
        )

        assert rel.source_id == "entity1"
        assert rel.target_id == "entity2"
        assert rel.type == "CITES"
        assert rel.id is not None  # Auto-generated
        assert rel.confidence == 1.0  # Default
        assert rel.properties == {}
        assert isinstance(rel.created_at, datetime)

    def test_create_relationship_full(self):
        """Test creating relationship with all fields."""
        rel = Relationship(
            source_id="paper1",
            target_id="paper2",
            type="CITES",
            properties={"section": "introduction", "context": "builds on"},
            confidence=0.95,
            created_by="citation_extractor"
        )

        assert rel.properties["section"] == "introduction"
        assert rel.confidence == 0.95
        assert rel.created_by == "citation_extractor"

    def test_auto_generate_id(self):
        """Test that ID is auto-generated."""
        rel1 = Relationship(source_id="a", target_id="b", type="RELATES_TO")
        rel2 = Relationship(source_id="a", target_id="b", type="RELATES_TO")

        assert rel1.id is not None
        assert rel2.id is not None
        assert rel1.id != rel2.id

    def test_missing_source_raises_error(self):
        """Test that missing source ID raises error."""
        with pytest.raises(ValueError, match="Source and target"):
            Relationship(source_id="", target_id="target", type="CITES")

    def test_missing_target_raises_error(self):
        """Test that missing target ID raises error."""
        with pytest.raises(ValueError, match="Source and target"):
            Relationship(source_id="source", target_id="", type="CITES")

    def test_missing_type_raises_error(self):
        """Test that missing type raises error."""
        with pytest.raises(ValueError, match="type is required"):
            Relationship(source_id="a", target_id="b", type="")

    def test_invalid_confidence_raises_error(self):
        """Test that invalid confidence raises error."""
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            Relationship(
                source_id="a",
                target_id="b",
                type="CITES",
                confidence=1.5
            )

    def test_standard_relationship_types(self):
        """Test that standard types don't warn."""
        for rel_type in Relationship.VALID_TYPES:
            rel = Relationship(
                source_id="a",
                target_id="b",
                type=rel_type
            )
            assert rel.type == rel_type

    def test_non_standard_type_warns(self):
        """Test that non-standard type issues warning."""
        with pytest.warns(UserWarning, match="not standard"):
            Relationship(
                source_id="a",
                target_id="b",
                type="CUSTOM_RELATION"
            )

    def test_to_dict(self):
        """Test relationship serialization."""
        rel = Relationship(
            id="rel_id",
            source_id="paper1",
            target_id="paper2",
            type="CITES",
            properties={"section": "intro"},
            confidence=0.9,
            created_by="extractor"
        )

        data = rel.to_dict()

        assert data["id"] == "rel_id"
        assert data["source_id"] == "paper1"
        assert data["target_id"] == "paper2"
        assert data["type"] == "CITES"
        assert data["properties"]["section"] == "intro"
        assert data["confidence"] == 0.9
        assert data["created_by"] == "extractor"
        assert "created_at" in data

    def test_from_dict(self):
        """Test relationship deserialization."""
        data = {
            "id": "rel_id",
            "source_id": "paper1",
            "target_id": "paper2",
            "type": "CITES",
            "properties": {"section": "intro"},
            "confidence": 0.9,
            "created_at": "2024-01-15T10:30:00",
            "created_by": "extractor"
        }

        rel = Relationship.from_dict(data)

        assert rel.id == "rel_id"
        assert rel.source_id == "paper1"
        assert rel.target_id == "paper2"
        assert rel.type == "CITES"
        assert rel.properties["section"] == "intro"
        assert rel.confidence == 0.9
        assert rel.created_by == "extractor"
        assert isinstance(rel.created_at, datetime)

    def test_roundtrip_serialization(self):
        """Test that to_dict → from_dict preserves data."""
        original = Relationship(
            id="rel_id",
            source_id="hyp1",
            target_id="finding1",
            type="SUPPORTS",
            properties={"strength": "strong", "p_value": 0.001},
            confidence=0.99,
            created_by="analyzer"
        )

        # Serialize
        data = original.to_dict()
        json_str = json.dumps(data)

        # Deserialize
        restored = Relationship.from_dict(json.loads(json_str))

        assert restored.id == original.id
        assert restored.source_id == original.source_id
        assert restored.target_id == original.target_id
        assert restored.type == original.type
        assert restored.properties == original.properties
        assert restored.confidence == original.confidence
        assert restored.created_by == original.created_by
