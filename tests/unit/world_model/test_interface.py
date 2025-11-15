"""Tests for world model abstract interfaces."""

import pytest
from abc import ABC

from kosmos.world_model.interface import (
    EntityManager,
    ProvenanceTracker,
    WorldModelStorage,
)
from kosmos.world_model.models import Annotation, Entity, Relationship


class TestWorldModelStorage:
    """Test WorldModelStorage abstract interface."""

    def test_is_abstract_base_class(self):
        """Test that WorldModelStorage is an ABC."""
        assert issubclass(WorldModelStorage, ABC)

    def test_cannot_instantiate_directly(self):
        """Test that abstract class cannot be instantiated."""
        with pytest.raises(TypeError, match="abstract"):
            WorldModelStorage()  # type: ignore

    def test_has_required_abstract_methods(self):
        """Test that all required methods are abstract."""
        abstract_methods = WorldModelStorage.__abstractmethods__

        required_methods = {
            "add_entity",
            "get_entity",
            "update_entity",
            "delete_entity",
            "add_relationship",
            "get_relationship",
            "query_related_entities",
            "export_graph",
            "import_graph",
            "get_statistics",
            "reset",
            "close",
        }

        assert required_methods.issubset(abstract_methods)

    def test_concrete_implementation_required(self):
        """Test that concrete class must implement all abstract methods."""

        # Incomplete implementation (missing some methods)
        class IncompleteStorage(WorldModelStorage):
            def add_entity(self, entity, merge=True):
                return entity.id

            # Missing other required methods...

        # Should not be able to instantiate
        with pytest.raises(TypeError, match="abstract"):
            IncompleteStorage()  # type: ignore


class TestEntityManager:
    """Test EntityManager abstract interface."""

    def test_is_abstract_base_class(self):
        """Test that EntityManager is an ABC."""
        assert issubclass(EntityManager, ABC)

    def test_cannot_instantiate_directly(self):
        """Test that abstract class cannot be instantiated."""
        with pytest.raises(TypeError, match="abstract"):
            EntityManager()  # type: ignore

    def test_has_required_abstract_methods(self):
        """Test that all required methods are abstract."""
        abstract_methods = EntityManager.__abstractmethods__

        required_methods = {
            "verify_entity",
            "add_annotation",
            "get_annotations",
        }

        assert required_methods.issubset(abstract_methods)


class TestProvenanceTracker:
    """Test ProvenanceTracker abstract interface."""

    def test_is_abstract_base_class(self):
        """Test that ProvenanceTracker is an ABC."""
        assert issubclass(ProvenanceTracker, ABC)

    def test_cannot_instantiate_directly(self):
        """Test that abstract class cannot be instantiated."""
        with pytest.raises(TypeError, match="abstract"):
            ProvenanceTracker()  # type: ignore

    def test_has_required_abstract_methods(self):
        """Test that all required methods are abstract."""
        abstract_methods = ProvenanceTracker.__abstractmethods__

        required_methods = {
            "record_derivation",
            "get_provenance",
        }

        assert required_methods.issubset(abstract_methods)


class TestInterfaceSignatures:
    """Test method signatures match expected patterns."""

    def test_add_entity_signature(self):
        """Test add_entity method signature."""
        method = WorldModelStorage.add_entity

        # Check annotations (type hints)
        annotations = method.__annotations__
        assert "entity" in annotations
        assert "merge" in annotations
        assert "return" in annotations

    def test_export_graph_signature(self):
        """Test export_graph method signature."""
        method = WorldModelStorage.export_graph

        annotations = method.__annotations__
        assert "filepath" in annotations
        assert "project" in annotations

    def test_get_statistics_signature(self):
        """Test get_statistics method signature."""
        method = WorldModelStorage.get_statistics

        annotations = method.__annotations__
        assert "project" in annotations
        assert "return" in annotations


class MockWorldModel(WorldModelStorage, EntityManager):
    """
    Mock implementation for testing.

    This demonstrates what a complete implementation looks like.
    Used by other tests to verify interface compliance.
    """

    def __init__(self):
        self.entities = {}
        self.relationships = {}

    def add_entity(self, entity: Entity, merge: bool = True) -> str:
        self.entities[entity.id] = entity
        return entity.id

    def get_entity(self, entity_id: str, project=None):
        return self.entities.get(entity_id)

    def update_entity(self, entity_id: str, updates):
        if entity_id in self.entities:
            entity = self.entities[entity_id]
            entity.properties.update(updates)

    def delete_entity(self, entity_id: str):
        if entity_id in self.entities:
            del self.entities[entity_id]

    def add_relationship(self, relationship: Relationship) -> str:
        self.relationships[relationship.id] = relationship
        return relationship.id

    def get_relationship(self, relationship_id: str):
        return self.relationships.get(relationship_id)

    def query_related_entities(self, entity_id, relationship_type=None, direction="outgoing", max_depth=1):
        return []

    def export_graph(self, filepath: str, project=None):
        pass

    def import_graph(self, filepath: str, clear: bool = False, project=None):
        pass

    def get_statistics(self, project=None):
        return {
            "entity_count": len(self.entities),
            "relationship_count": len(self.relationships),
        }

    def reset(self, project=None):
        self.entities.clear()
        self.relationships.clear()

    def close(self):
        pass

    # EntityManager methods
    def verify_entity(self, entity_id: str, verified_by: str):
        if entity_id in self.entities:
            self.entities[entity_id].verified = True

    def add_annotation(self, entity_id: str, annotation: Annotation):
        if entity_id in self.entities:
            self.entities[entity_id].annotations.append(annotation)

    def get_annotations(self, entity_id: str):
        if entity_id in self.entities:
            return self.entities[entity_id].annotations
        return []


class TestMockImplementation:
    """Test that mock implementation works correctly."""

    def test_can_instantiate_mock(self):
        """Test that complete implementation can be instantiated."""
        mock = MockWorldModel()
        assert isinstance(mock, WorldModelStorage)
        assert isinstance(mock, EntityManager)

    def test_mock_add_and_get_entity(self):
        """Test basic entity operations on mock."""
        mock = MockWorldModel()

        entity = Entity(type="Paper", properties={"title": "Test"})
        entity_id = mock.add_entity(entity)

        retrieved = mock.get_entity(entity_id)
        assert retrieved is not None
        assert retrieved.id == entity_id
        assert retrieved.properties["title"] == "Test"

    def test_mock_add_and_get_relationship(self):
        """Test basic relationship operations on mock."""
        mock = MockWorldModel()

        # Add entities first
        e1 = Entity(type="Paper", properties={})
        e2 = Entity(type="Paper", properties={})
        mock.add_entity(e1)
        mock.add_entity(e2)

        # Add relationship
        rel = Relationship(source_id=e1.id, target_id=e2.id, type="CITES")
        rel_id = mock.add_relationship(rel)

        retrieved = mock.get_relationship(rel_id)
        assert retrieved is not None
        assert retrieved.type == "CITES"

    def test_mock_statistics(self):
        """Test statistics on mock."""
        mock = MockWorldModel()

        # Add some entities
        for i in range(5):
            mock.add_entity(Entity(type="Paper", properties={}))

        stats = mock.get_statistics()
        assert stats["entity_count"] == 5
        assert stats["relationship_count"] == 0

    def test_mock_reset(self):
        """Test reset on mock."""
        mock = MockWorldModel()

        # Add data
        mock.add_entity(Entity(type="Paper", properties={}))
        assert mock.get_statistics()["entity_count"] == 1

        # Reset
        mock.reset()
        assert mock.get_statistics()["entity_count"] == 0

    def test_mock_verify_entity(self):
        """Test entity verification on mock."""
        mock = MockWorldModel()

        entity = Entity(type="Paper", properties={})
        entity_id = mock.add_entity(entity)

        # Initially not verified
        assert mock.get_entity(entity_id).verified is False

        # Verify
        mock.verify_entity(entity_id, verified_by="user@example.com")
        assert mock.get_entity(entity_id).verified is True

    def test_mock_add_annotation(self):
        """Test annotations on mock."""
        mock = MockWorldModel()

        entity = Entity(type="Paper", properties={})
        entity_id = mock.add_entity(entity)

        # Add annotation
        ann = Annotation(text="Test note", created_by="user")
        mock.add_annotation(entity_id, ann)

        # Retrieve annotations
        annotations = mock.get_annotations(entity_id)
        assert len(annotations) == 1
        assert annotations[0].text == "Test note"
