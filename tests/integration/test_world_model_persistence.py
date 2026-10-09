"""
Integration tests for world model persistence in research workflows.

Tests that research artifacts (questions, hypotheses, protocols, results)
are automatically persisted to the knowledge graph during research workflows.
"""

import pytest
from unittest.mock import ANY, MagicMock, patch
from datetime import datetime

from kosmos.agents.research_director import ResearchDirectorAgent
from kosmos.agents.base import AgentMessage, MessageType
from kosmos.world_model import get_world_model, reset_world_model
from kosmos.world_model import factory as world_model_factory
from kosmos.world_model.in_memory import InMemoryWorldModel
from kosmos.models.hypothesis import Hypothesis, HypothesisStatus

# The director reads these through names it imported at module load, so they are
# patched where they are looked up (kosmos.agents.research_director), not at source.
GET_HYPOTHESIS = 'kosmos.agents.research_director.get_hypothesis'
GET_EXPERIMENT = 'kosmos.agents.research_director.get_experiment'


@pytest.fixture
def reset_graph():
    """Give each test a fresh in-memory knowledge graph.

    The singleton is pre-set to InMemoryWorldModel so the director never probes
    or connects to a live Neo4j (the default "simple" mode would try one on
    bolt://localhost:7687 and fall back only when the connection fails).
    """
    reset_world_model()
    world_model_factory._world_model = InMemoryWorldModel()
    yield world_model_factory._world_model
    # Cleanup after test
    reset_world_model()


def _no_follow_up(director):
    """Patch the handler's follow-up step so only persistence runs.

    The handlers call the async _execute_next_action without awaiting it; a plain
    MagicMock records the call without leaving an un-awaited coroutine behind.
    """
    return (
        patch.object(director, 'decide_next_action', return_value=None),
        patch.object(director, '_execute_next_action', new=MagicMock()),
    )


@pytest.fixture
def mock_hypothesis():
    """Create a mock hypothesis for testing."""
    return Hypothesis(
        id="hyp_test_001",
        research_question="How do transformers learn?",
        statement="Transformers learn through attention mechanisms",
        rationale="Attention is the key mechanism in transformers",
        domain="machine_learning",
        status=HypothesisStatus.GENERATED,
        testability_score=0.85,
        novelty_score=0.72,
        confidence_score=0.80,
        generated_by="HypothesisGeneratorAgent",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        generation=1,
        refinement_count=0,
        parent_hypothesis_id=None
    )


class TestResearchQuestionPersistence:
    """Test research question entity persistence."""

    def test_research_question_created_on_init(self, reset_graph):
        """Test that research question entity is created when director initializes."""
        # Create director
        director = ResearchDirectorAgent(
            research_question="How do neural networks generalize?",
            domain="machine_learning"
        )

        # Verify world model initialized
        assert director.wm is not None
        assert director.question_entity_id is not None

        # Verify entity in graph
        wm = get_world_model()
        stats = wm.get_statistics()

        assert stats['entity_count'] == 1, "Should have 1 entity (research question)"
        assert stats['relationship_count'] == 0, "Should have no relationships yet"

    def test_research_question_contains_text(self, reset_graph):
        """Test that research question entity contains the question text."""
        question_text = "Can we improve transformer efficiency?"
        director = ResearchDirectorAgent(
            research_question=question_text,
            domain="machine_learning"
        )

        # Get the entity from graph
        wm = get_world_model()
        # Note: We can't directly query entity by ID in the simple interface,
        # but we verify it was created (tested above)

        assert director.question_entity_id is not None


class TestHypothesisPersistence:
    """Test hypothesis persistence and relationships."""

    @patch(GET_HYPOTHESIS)
    async def test_hypothesis_persisted_to_graph(self, mock_get_hyp, reset_graph, mock_hypothesis):
        """Test that hypothesis is persisted when hypothesis_generator responds."""
        mock_get_hyp.return_value = mock_hypothesis

        # Create director
        director = ResearchDirectorAgent(
            research_question="Test question",
            domain="test"
        )

        initial_entity_count = get_world_model().get_statistics()['entity_count']

        # Simulate hypothesis generator response
        message = AgentMessage(
            type=MessageType.RESPONSE,
            from_agent="hyp_gen_001",
            to_agent=director.agent_id,
            content={
                "hypothesis_ids": ["hyp_test_001"],
                "count": 1
            },
            metadata={"agent_type": "HypothesisGeneratorAgent"}
        )

        # Stop the handler's follow-up so no workflow step runs
        decide_patch, execute_patch = _no_follow_up(director)
        with decide_patch, execute_patch as mock_execute:
            await director.process_message(message)
        mock_execute.assert_called_once_with(None)

        # Verify hypothesis was persisted
        stats = get_world_model().get_statistics()
        assert stats['entity_count'] == initial_entity_count + 1, "Should have added 1 hypothesis entity"
        assert stats['relationship_count'] == 1, "Should have 1 SPAWNED_BY relationship"

    @patch(GET_HYPOTHESIS)
    async def test_hypothesis_spawned_by_relationship(self, mock_get_hyp, reset_graph, mock_hypothesis):
        """Test that SPAWNED_BY relationship is created."""
        mock_get_hyp.return_value = mock_hypothesis

        director = ResearchDirectorAgent(
            research_question="Test question",
            domain="test"
        )

        message = AgentMessage(
            type=MessageType.RESPONSE,
            from_agent="hyp_gen_001",
            to_agent=director.agent_id,
            content={
                "hypothesis_ids": ["hyp_test_001"],
                "count": 1
            },
            metadata={"agent_type": "HypothesisGeneratorAgent"}
        )

        # Stop the handler's follow-up so no workflow step runs
        decide_patch, execute_patch = _no_follow_up(director)
        with decide_patch, execute_patch as mock_execute:
            await director.process_message(message)
        mock_execute.assert_called_once_with(None)

        # Verify relationship created
        stats = get_world_model().get_statistics()
        assert stats['relationship_count'] == 1, "Should have SPAWNED_BY relationship"
        assert stats['relationship_types'] == {"SPAWNED_BY": 1}
        rel = next(iter(get_world_model()._relationships.values()))
        assert rel.source_id == "hyp_test_001"
        assert rel.target_id == director.question_entity_id


class TestRefinedHypothesisPersistence:
    """Test refined hypothesis persistence with REFINED_FROM relationship."""

    @patch(GET_HYPOTHESIS)
    async def test_refined_hypothesis_has_parent_relationship(self, mock_get_hyp, reset_graph):
        """Test that refined hypothesis has REFINED_FROM relationship to parent."""
        # Create parent hypothesis
        parent_hyp = Hypothesis(
            id="hyp_parent_001",
            research_question="Test question",
            statement="Parent hypothesis",
            rationale="Parent rationale for the original hypothesis",
            domain="test",
            status=HypothesisStatus.GENERATED,
            generation=1,
            refinement_count=0,
            parent_hypothesis_id=None,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        # Create refined hypothesis
        refined_hyp = Hypothesis(
            id="hyp_refined_001",
            research_question="Test question",
            statement="Refined hypothesis",
            rationale="Refined rationale after the first test result",
            domain="test",
            status=HypothesisStatus.GENERATED,
            generation=2,
            refinement_count=1,
            parent_hypothesis_id="hyp_parent_001",  # Link to parent
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        mock_get_hyp.return_value = refined_hyp

        director = ResearchDirectorAgent(
            research_question="Test question",
            domain="test"
        )

        message = AgentMessage(
            type=MessageType.RESPONSE,
            from_agent="refiner_001",
            to_agent=director.agent_id,
            content={
                "refined_hypothesis_ids": ["hyp_refined_001"],
                "retired_hypothesis_ids": []
            },
            metadata={"agent_type": "HypothesisRefiner"}
        )

        # Stop the handler's follow-up so no workflow step runs
        decide_patch, execute_patch = _no_follow_up(director)
        with decide_patch, execute_patch as mock_execute:
            await director.process_message(message)
        mock_execute.assert_called_once_with(None)

        # Verify relationships: SPAWNED_BY (to question) + REFINED_FROM (to parent)
        stats = get_world_model().get_statistics()
        assert stats['relationship_count'] == 2, "Should have SPAWNED_BY + REFINED_FROM relationships"
        assert stats['relationship_types'] == {"SPAWNED_BY": 1, "REFINED_FROM": 1}
        refined_from = [
            r for r in get_world_model()._relationships.values() if r.type == "REFINED_FROM"
        ][0]
        assert refined_from.source_id == "hyp_refined_001"
        assert refined_from.target_id == "hyp_parent_001"


class TestProtocolPersistence:
    """Test experiment protocol persistence."""

    @patch(GET_EXPERIMENT)
    async def test_protocol_persisted_with_tests_relationship(self, mock_get_exp, reset_graph):
        """Test that protocol is persisted with TESTS relationship to hypothesis."""
        from kosmos.models.experiment import ExperimentProtocol, ExperimentType

        protocol = MagicMock()
        protocol.id = "proto_001"
        protocol.name = "Test Protocol"
        protocol.hypothesis_id = "hyp_001"
        protocol.experiment_type = ExperimentType.COMPUTATIONAL
        protocol.domain = "test"
        protocol.description = "Test description"
        protocol.objective = "Test objective"
        protocol.rigor_score = 0.8
        protocol.template_name = None
        protocol.created_at = datetime.now()
        protocol.updated_at = datetime.now()

        mock_get_exp.return_value = protocol

        director = ResearchDirectorAgent(
            research_question="Test question",
            domain="test"
        )

        message = AgentMessage(
            type=MessageType.RESPONSE,
            from_agent="designer_001",
            to_agent=director.agent_id,
            content={
                "protocol_id": "proto_001",
                "hypothesis_id": "hyp_001"
            },
            metadata={"agent_type": "ExperimentDesignerAgent"}
        )

        # Stop the handler's follow-up so no workflow step runs
        decide_patch, execute_patch = _no_follow_up(director)
        with decide_patch, execute_patch as mock_execute:
            await director.process_message(message)
        mock_execute.assert_called_once_with(None)

        # Verify protocol and relationship persisted
        stats = get_world_model().get_statistics()
        assert stats['entity_count'] == 2, "Should have question + protocol"
        assert stats['relationship_count'] == 1, "Should have TESTS relationship"
        assert stats['relationship_types'] == {"TESTS": 1}
        assert stats['entity_types'].get("ExperimentProtocol") == 1


class TestDualPersistence:
    """Test that dual persistence (SQL + Graph) works correctly."""

    @patch(GET_HYPOTHESIS)
    async def test_sql_persistence_unaffected(self, mock_get_hyp, reset_graph, mock_hypothesis):
        """Test that SQL persistence continues to work alongside graph persistence."""
        mock_get_hyp.return_value = mock_hypothesis

        director = ResearchDirectorAgent(
            research_question="Test question",
            domain="test"
        )

        message = AgentMessage(
            type=MessageType.RESPONSE,
            from_agent="hyp_gen_001",
            to_agent=director.agent_id,
            content={
                "hypothesis_ids": ["hyp_test_001"],
                "count": 1
            },
            metadata={"agent_type": "HypothesisGeneratorAgent"}
        )

        # Stop the handler's follow-up so no workflow step runs
        decide_patch, execute_patch = _no_follow_up(director)
        with decide_patch, execute_patch as mock_execute:
            await director.process_message(message)
        mock_execute.assert_called_once_with(None)

        # Verify graph has the data
        stats = get_world_model().get_statistics()
        assert stats['entity_count'] > 0

        # Verify SQL was called (hypothesis was fetched from DB)
        mock_get_hyp.assert_called_once_with(ANY, "hyp_test_001")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
