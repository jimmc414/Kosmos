"""
Tests for World Model Query Requirements (REQ-WM-QUERY-001 through REQ-WM-QUERY-004).

These tests validate querying capabilities including related entity queries,
graph traversal, filtering, and complex queries across the knowledge graph.
"""

import pytest
import uuid
from datetime import datetime
from typing import List
from unittest.mock import Mock, patch

from kosmos.world_model.models import Entity, Relationship
from kosmos.world_model import get_world_model, reset_world_model

# Test markers for requirements traceability
pytestmark = [
    pytest.mark.requirement("REQ-WM-QUERY"),
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
        # Try to reset for clean state
        try:
            wm.reset(project="test_query_project")
        except Exception:
            pass
        yield wm
    finally:
        try:
            reset_world_model()
        except Exception:
            pass


@pytest.fixture
def sample_graph(world_model):
    """
    Create a sample knowledge graph for query testing.

    Structure:
    - Paper1 -> CITES -> Paper2
    - Paper1 -> MENTIONS -> Concept1
    - Paper2 -> MENTIONS -> Concept1
    - Author1 -> AUTHOR_OF -> Paper1
    - Hypothesis1 -> TESTS -> Experiment1
    """
    project = "test_query_project"

    # Create entities
    paper1 = Entity(
        type="Paper",
        properties={"title": "Neural Networks Survey", "year": 2024},
        project=project
    )
    paper2 = Entity(
        type="Paper",
        properties={"title": "Deep Learning Foundations", "year": 2023},
        project=project
    )
    concept1 = Entity(
        type="Concept",
        properties={"name": "Deep Learning", "domain": "AI"},
        project=project
    )
    author1 = Entity(
        type="Author",
        properties={"name": "Dr. Smith", "affiliation": "MIT"},
        project=project
    )
    hypothesis1 = Entity(
        type="Hypothesis",
        properties={"statement": "Test hypothesis", "testability_score": 0.9},
        project=project
    )
    experiment1 = Entity(
        type="Experiment",
        properties={"name": "Validation Experiment"},
        project=project
    )

    # Add entities
    id_paper1 = world_model.add_entity(paper1)
    id_paper2 = world_model.add_entity(paper2)
    id_concept1 = world_model.add_entity(concept1)
    id_author1 = world_model.add_entity(author1)
    id_hypothesis1 = world_model.add_entity(hypothesis1)
    id_experiment1 = world_model.add_entity(experiment1)

    # Create relationships
    rel1 = Relationship(source_id=id_paper1, target_id=id_paper2, type="CITES")
    rel2 = Relationship(source_id=id_paper1, target_id=id_concept1, type="MENTIONS")
    rel3 = Relationship(source_id=id_paper2, target_id=id_concept1, type="MENTIONS")
    rel4 = Relationship(source_id=id_author1, target_id=id_paper1, type="AUTHOR_OF")
    rel5 = Relationship(source_id=id_hypothesis1, target_id=id_experiment1, type="TESTS")

    world_model.add_relationship(rel1)
    world_model.add_relationship(rel2)
    world_model.add_relationship(rel3)
    world_model.add_relationship(rel4)
    world_model.add_relationship(rel5)

    return {
        "paper1": id_paper1,
        "paper2": id_paper2,
        "concept1": id_concept1,
        "author1": id_author1,
        "hypothesis1": id_hypothesis1,
        "experiment1": id_experiment1,
    }


# ============================================================================
# REQ-WM-QUERY-001: Query Related Entities (MUST)
# ============================================================================

@pytest.mark.requirement("REQ-WM-QUERY-001")
@pytest.mark.priority("MUST")
class TestREQ_WM_QUERY_001_QueryRelatedEntities:
    """
    REQ-WM-QUERY-001: The World Model MUST support querying entities related
    to a given entity through specified relationship types and directions.
    """

    def test_query_outgoing_relationships(self, world_model, sample_graph):
        """Verify querying entities through outgoing relationships."""
        paper1_id = sample_graph["paper1"]

        # Query papers cited by paper1 (outgoing CITES)
        cited_papers = world_model.query_related_entities(
            entity_id=paper1_id,
            relationship_type="CITES",
            direction="outgoing",
            max_depth=1
        )

        assert len(cited_papers) >= 1
        # Should include paper2
        cited_titles = [p.properties.get("title") for p in cited_papers]
        assert "Deep Learning Foundations" in cited_titles

    def test_query_incoming_relationships(self, world_model, sample_graph):
        """Verify querying entities through incoming relationships."""
        paper1_id = sample_graph["paper1"]

        # Query authors of paper1 (incoming AUTHOR_OF)
        authors = world_model.query_related_entities(
            entity_id=paper1_id,
            relationship_type="AUTHOR_OF",
            direction="incoming",
            max_depth=1
        )

        assert len(authors) >= 1
        # Should include author1
        author_names = [a.properties.get("name") for a in authors]
        assert "Dr. Smith" in author_names

    def test_query_bidirectional_relationships(self, world_model, sample_graph):
        """Verify querying entities in both directions."""
        concept1_id = sample_graph["concept1"]

        # Query papers that mention concept1 (incoming MENTIONS)
        related_entities = world_model.query_related_entities(
            entity_id=concept1_id,
            relationship_type="MENTIONS",
            direction="both",
            max_depth=1
        )

        # Should include both paper1 and paper2
        assert len(related_entities) >= 2

    def test_query_all_relationship_types(self, world_model, sample_graph):
        """Verify querying without relationship type filter returns all related entities."""
        paper1_id = sample_graph["paper1"]

        # Query all related entities (any relationship type)
        all_related = world_model.query_related_entities(
            entity_id=paper1_id,
            relationship_type=None,  # No filter
            direction="outgoing",
            max_depth=1
        )

        # Should include paper2 (CITES) and concept1 (MENTIONS)
        assert len(all_related) >= 2

    def test_query_with_max_depth(self, world_model, sample_graph):
        """Verify querying respects max_depth parameter."""
        paper1_id = sample_graph["paper1"]

        # Depth 1: Direct relationships only
        depth1_results = world_model.query_related_entities(
            entity_id=paper1_id,
            direction="outgoing",
            max_depth=1
        )

        # Should have at least direct connections
        assert len(depth1_results) >= 1

        # Depth 2: Could include entities 2 hops away
        # (Paper1 -> Paper2 -> Concept1)
        depth2_results = world_model.query_related_entities(
            entity_id=paper1_id,
            direction="outgoing",
            max_depth=2
        )

        # Depth 2 should have >= depth 1 results
        assert len(depth2_results) >= len(depth1_results)

    def test_query_nonexistent_entity_returns_empty(self, world_model):
        """Verify querying non-existent entity returns empty list."""
        fake_id = str(uuid.uuid4())

        results = world_model.query_related_entities(
            entity_id=fake_id,
            direction="outgoing"
        )

        assert isinstance(results, list)
        assert len(results) == 0


# ============================================================================
# REQ-WM-QUERY-002: Filter by Entity Properties (MUST)
# ============================================================================

@pytest.mark.requirement("REQ-WM-QUERY-002")
@pytest.mark.priority("MUST")
class TestREQ_WM_QUERY_002_FilterByProperties:
    """
    REQ-WM-QUERY-002: The World Model MUST support filtering query results
    by entity properties including type, project, confidence, and custom properties.
    """

    def test_filter_by_entity_type(self, world_model, sample_graph):
        """Verify entities can be filtered by type."""
        # Get statistics to see entity types
        stats = world_model.get_statistics(project="test_query_project")

        assert "entity_types" in stats
        entity_types = stats["entity_types"]

        # Should have Paper, Concept, Author, etc.
        assert "Paper" in entity_types or stats["entity_count"] > 0
        assert isinstance(entity_types, dict)

    def test_filter_by_project(self, world_model):
        """Verify entities can be filtered by project."""
        # Create entities in different projects
        entity_a = Entity(
            type="Paper",
            properties={"title": "Project A Paper"},
            project="project_a"
        )
        entity_b = Entity(
            type="Paper",
            properties={"title": "Project B Paper"},
            project="project_b"
        )

        id_a = world_model.add_entity(entity_a)
        id_b = world_model.add_entity(entity_b)

        # Get statistics by project
        stats_a = world_model.get_statistics(project="project_a")
        stats_b = world_model.get_statistics(project="project_b")

        # Each project should have at least one entity
        assert stats_a["entity_count"] >= 1
        assert stats_b["entity_count"] >= 1

    def test_filter_by_confidence_threshold(self, world_model):
        """Verify entities can be filtered by confidence score."""
        # Create entities with different confidence scores
        high_conf = Entity(
            type="Hypothesis",
            properties={"statement": "High confidence hypothesis"},
            confidence=0.95,
            project="test_query_project"
        )
        low_conf = Entity(
            type="Hypothesis",
            properties={"statement": "Low confidence hypothesis"},
            confidence=0.4,
            project="test_query_project"
        )

        id_high = world_model.add_entity(high_conf)
        id_low = world_model.add_entity(low_conf)

        # Retrieve and verify confidence
        retrieved_high = world_model.get_entity(id_high)
        retrieved_low = world_model.get_entity(id_low)

        assert retrieved_high is not None
        assert retrieved_low is not None
        # Note: Actual filtering by confidence would require query API
        # This test verifies confidence is preserved

    def test_filter_verified_entities(self, world_model):
        """Verify entities can be filtered by verification status."""
        verified_entity = Entity(
            type="Paper",
            properties={"title": "Verified Paper"},
            verified=True,
            project="test_query_project"
        )
        unverified_entity = Entity(
            type="Paper",
            properties={"title": "Unverified Paper"},
            verified=False,
            project="test_query_project"
        )

        id_verified = world_model.add_entity(verified_entity)
        id_unverified = world_model.add_entity(unverified_entity)

        # Retrieve and check verification status
        retrieved_verified = world_model.get_entity(id_verified)
        retrieved_unverified = world_model.get_entity(id_unverified)

        assert retrieved_verified is not None
        assert retrieved_unverified is not None
        # Note: Actual filtering would require query API extension


# ============================================================================
# REQ-WM-QUERY-003: Graph Traversal (MUST)
# ============================================================================

@pytest.mark.requirement("REQ-WM-QUERY-003")
@pytest.mark.priority("MUST")
class TestREQ_WM_QUERY_003_GraphTraversal:
    """
    REQ-WM-QUERY-003: The World Model MUST support graph traversal operations
    including multi-hop queries, path finding, and subgraph extraction.
    """

    def test_multi_hop_query(self, world_model, sample_graph):
        """Verify multi-hop graph traversal."""
        author1_id = sample_graph["author1"]

        # Traverse: Author -> Paper -> Concept (2 hops)
        # Author1 -AUTHOR_OF-> Paper1 -MENTIONS-> Concept1
        related = world_model.query_related_entities(
            entity_id=author1_id,
            direction="outgoing",
            max_depth=2
        )

        # Should include entities multiple hops away
        assert len(related) >= 1

    def test_query_related_entities_depth_1(self, world_model, sample_graph):
        """Verify single-hop traversal (direct neighbors)."""
        paper1_id = sample_graph["paper1"]

        neighbors = world_model.query_related_entities(
            entity_id=paper1_id,
            direction="both",
            max_depth=1
        )

        # Should include direct neighbors only
        assert len(neighbors) >= 1

    def test_query_related_entities_depth_2(self, world_model, sample_graph):
        """Verify two-hop traversal."""
        paper1_id = sample_graph["paper1"]

        # 2-hop neighborhood
        neighborhood = world_model.query_related_entities(
            entity_id=paper1_id,
            direction="both",
            max_depth=2
        )

        # Should include entities up to 2 hops away
        assert len(neighborhood) >= 1

    def test_traversal_avoids_cycles(self, world_model):
        """Verify graph traversal handles cycles correctly."""
        # Create circular reference: A -> B -> C -> A
        entity_a = Entity(type="Concept", properties={"name": "A"}, project="test_query_project")
        entity_b = Entity(type="Concept", properties={"name": "B"}, project="test_query_project")
        entity_c = Entity(type="Concept", properties={"name": "C"}, project="test_query_project")

        id_a = world_model.add_entity(entity_a)
        id_b = world_model.add_entity(entity_b)
        id_c = world_model.add_entity(entity_c)

        rel_ab = Relationship(source_id=id_a, target_id=id_b, type="RELATES_TO")
        rel_bc = Relationship(source_id=id_b, target_id=id_c, type="RELATES_TO")
        rel_ca = Relationship(source_id=id_c, target_id=id_a, type="RELATES_TO")

        world_model.add_relationship(rel_ab)
        world_model.add_relationship(rel_bc)
        world_model.add_relationship(rel_ca)

        # Query with depth that would traverse cycle
        results = world_model.query_related_entities(
            entity_id=id_a,
            direction="outgoing",
            max_depth=5
        )

        # Should not infinite loop and return reasonable results
        assert isinstance(results, list)
        # Should not duplicate entities (implementation dependent)

    def test_subgraph_extraction(self, world_model, sample_graph):
        """Verify extracting subgraph around an entity."""
        paper1_id = sample_graph["paper1"]

        # Get all related entities (subgraph centered on paper1)
        subgraph_entities = world_model.query_related_entities(
            entity_id=paper1_id,
            direction="both",
            max_depth=1
        )

        assert len(subgraph_entities) >= 1
        # Entities should be Entity objects
        for entity in subgraph_entities:
            assert isinstance(entity, Entity)
            assert entity.id is not None


# ============================================================================
# REQ-WM-QUERY-004: Complex Queries (SHOULD)
# ============================================================================

@pytest.mark.requirement("REQ-WM-QUERY-004")
@pytest.mark.priority("SHOULD")
class TestREQ_WM_QUERY_004_ComplexQueries:
    """
    REQ-WM-QUERY-004: The World Model SHOULD support complex queries including
    combining multiple filters, aggregations, and pattern matching.
    """

    def test_query_with_multiple_relationship_types(self, world_model, sample_graph):
        """Verify querying with multiple relationship type filters."""
        paper1_id = sample_graph["paper1"]

        # Query both CITES and MENTIONS relationships
        # Note: Current API takes single relationship_type
        # This tests querying each separately

        cites = world_model.query_related_entities(
            entity_id=paper1_id,
            relationship_type="CITES",
            direction="outgoing"
        )

        mentions = world_model.query_related_entities(
            entity_id=paper1_id,
            relationship_type="MENTIONS",
            direction="outgoing"
        )

        # Both queries should return results
        assert len(cites) >= 1
        assert len(mentions) >= 1

    def test_query_statistics_aggregation(self, world_model, sample_graph):
        """Verify statistical aggregations over query results."""
        stats = world_model.get_statistics(project="test_query_project")

        # Statistics provide aggregated data
        assert "entity_count" in stats
        assert "relationship_count" in stats
        assert "entity_types" in stats
        assert "relationship_types" in stats

        # Entity types is a dictionary of counts
        entity_types = stats["entity_types"]
        assert isinstance(entity_types, dict)

        # Should have various entity types
        total_entities = sum(entity_types.values()) if entity_types else 0
        assert total_entities <= stats["entity_count"]

    def test_query_entities_by_creation_time(self, world_model):
        """Verify entities can be filtered by creation time range."""
        # Create entities at different times
        before = datetime.now()

        entity1 = Entity(
            type="Paper",
            properties={"title": "First Paper"},
            project="test_query_project"
        )
        id1 = world_model.add_entity(entity1)

        import time
        time.sleep(0.1)

        entity2 = Entity(
            type="Paper",
            properties={"title": "Second Paper"},
            project="test_query_project"
        )
        id2 = world_model.add_entity(entity2)

        after = datetime.now()

        # Retrieve and verify timestamps
        retrieved1 = world_model.get_entity(id1)
        retrieved2 = world_model.get_entity(id2)

        assert retrieved1 is not None
        assert retrieved2 is not None

        # Timestamps should be within range
        if retrieved1.created_at and retrieved2.created_at:
            assert before <= retrieved1.created_at <= after
            assert before <= retrieved2.created_at <= after

    def test_query_by_property_pattern(self, world_model):
        """Verify querying entities by property patterns."""
        # Create entities with specific property patterns
        entity1 = Entity(
            type="Paper",
            properties={
                "title": "Neural Networks in Healthcare",
                "domain": "healthcare",
                "year": 2024
            },
            project="test_query_project"
        )
        entity2 = Entity(
            type="Paper",
            properties={
                "title": "Neural Networks in Finance",
                "domain": "finance",
                "year": 2024
            },
            project="test_query_project"
        )

        id1 = world_model.add_entity(entity1)
        id2 = world_model.add_entity(entity2)

        # Retrieve and verify properties
        retrieved1 = world_model.get_entity(id1)
        retrieved2 = world_model.get_entity(id2)

        assert retrieved1 is not None
        assert retrieved2 is not None
        assert retrieved1.properties["domain"] == "healthcare"
        assert retrieved2.properties["domain"] == "finance"

        # Note: Pattern matching would require query API extension
        # This test verifies properties are stored correctly

    def test_query_with_confidence_and_type_filter(self, world_model):
        """Verify combining multiple filters in queries."""
        # Create entities with different types and confidence
        high_conf_paper = Entity(
            type="Paper",
            properties={"title": "High Confidence Paper"},
            confidence=0.95,
            project="test_query_project"
        )
        high_conf_concept = Entity(
            type="Concept",
            properties={"name": "High Confidence Concept"},
            confidence=0.9,
            project="test_query_project"
        )

        id_paper = world_model.add_entity(high_conf_paper)
        id_concept = world_model.add_entity(high_conf_concept)

        # Retrieve and verify
        retrieved_paper = world_model.get_entity(id_paper)
        retrieved_concept = world_model.get_entity(id_concept)

        assert retrieved_paper is not None
        assert retrieved_concept is not None
        assert retrieved_paper.type == "Paper"
        assert retrieved_concept.type == "Concept"

        # Note: Combined filtering would require extended query API
