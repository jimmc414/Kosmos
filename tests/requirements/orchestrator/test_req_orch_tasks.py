"""
Tests for Orchestrator Task Dispatch Requirements (REQ-ORCH-TASK-001 through REQ-ORCH-TASK-007).

These tests validate task dispatch, agent coordination, message passing,
and action execution for the Research Director orchestrator.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, Any

from kosmos.core.workflow import WorkflowState, NextAction
from kosmos.agents.research_director import ResearchDirectorAgent
from kosmos.agents.base import AgentMessage, MessageType

# Test markers for requirements traceability
pytestmark = [
    pytest.mark.requirement("REQ-ORCH-TASK"),
    pytest.mark.category("orchestrator"),
]


@pytest.mark.requirement("REQ-ORCH-TASK-001")
@pytest.mark.priority("MUST")
class TestREQ_ORCH_TASK_001_DecisionMaking:
    """
    REQ-ORCH-TASK-001: The Research Director MUST implement decide_next_action()
    to determine the next action based on workflow state and research plan.
    """

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_decide_next_action_exists(self, mock_wm, mock_llm):
        """Verify decide_next_action method exists and is callable."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        assert hasattr(director, 'decide_next_action')
        assert callable(director.decide_next_action)

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_decide_generate_hypothesis_when_no_hypotheses(self, mock_wm, mock_llm):
        """Verify decision to generate hypotheses when pool is empty."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")
        director.workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES)

        action = director.decide_next_action()

        assert action == NextAction.GENERATE_HYPOTHESIS

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_decide_design_experiment_when_untested_hypotheses(self, mock_wm, mock_llm):
        """Verify decision to design experiments when hypotheses are untested."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")
        director.workflow.transition_to(WorkflowState.DESIGNING_EXPERIMENTS)

        # Add untested hypothesis
        director.research_plan.add_hypothesis("hyp1")

        action = director.decide_next_action()

        assert action == NextAction.DESIGN_EXPERIMENT

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_decide_execute_when_experiments_queued(self, mock_wm, mock_llm):
        """Verify decision to execute experiments when queue is not empty."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")
        director.workflow.transition_to(WorkflowState.EXECUTING)

        # Add experiment to queue
        director.research_plan.add_experiment("exp1")

        action = director.decide_next_action()

        assert action == NextAction.EXECUTE_EXPERIMENT

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_decide_analyze_when_in_analyzing_state(self, mock_wm, mock_llm):
        """Verify decision to analyze results when in ANALYZING state."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")
        director.workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES)
        director.workflow.transition_to(WorkflowState.DESIGNING_EXPERIMENTS)
        director.workflow.transition_to(WorkflowState.EXECUTING)
        director.workflow.transition_to(WorkflowState.ANALYZING)

        action = director.decide_next_action()

        assert action == NextAction.ANALYZE_RESULT

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_decide_refine_when_in_refining_state(self, mock_wm, mock_llm):
        """Verify decision to refine hypotheses when in REFINING state."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")
        director.workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES)
        director.workflow.transition_to(WorkflowState.DESIGNING_EXPERIMENTS)
        director.workflow.transition_to(WorkflowState.EXECUTING)
        director.workflow.transition_to(WorkflowState.ANALYZING)
        director.workflow.transition_to(WorkflowState.REFINING)

        # Add tested hypothesis
        director.research_plan.add_hypothesis("hyp1")
        director.research_plan.mark_tested("hyp1")

        action = director.decide_next_action()

        assert action == NextAction.REFINE_HYPOTHESIS

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_decide_converge_when_iteration_limit_reached(self, mock_wm, mock_llm):
        """Verify decision to converge when iteration limit is reached."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test", config={"max_iterations": 5})

        # Set iteration at limit
        director.research_plan.iteration_count = 5

        action = director.decide_next_action()

        assert action == NextAction.CONVERGE


@pytest.mark.requirement("REQ-ORCH-TASK-002")
@pytest.mark.priority("MUST")
class TestREQ_ORCH_TASK_002_AgentMessageDispatch:
    """
    REQ-ORCH-TASK-002: The Research Director MUST dispatch messages to
    specialized agents (HypothesisGenerator, ExperimentDesigner, etc.)
    using message-based coordination.
    """

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_send_to_hypothesis_generator(self, mock_wm, mock_llm):
        """Verify director can send messages to HypothesisGeneratorAgent."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")
        director.register_agent("HypothesisGeneratorAgent", "hyp_gen_001")

        message = director._send_to_hypothesis_generator(action="generate")

        assert isinstance(message, AgentMessage)
        assert message.type == MessageType.REQUEST
        assert message.content["action"] == "generate"
        assert message.content["research_question"] == "Test"

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_send_to_experiment_designer(self, mock_wm, mock_llm):
        """Verify director can send messages to ExperimentDesignerAgent."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")
        director.register_agent("ExperimentDesignerAgent", "exp_designer_001")

        message = director._send_to_experiment_designer(hypothesis_id="hyp1")

        assert isinstance(message, AgentMessage)
        assert message.type == MessageType.REQUEST
        assert message.content["action"] == "design_experiment"
        assert message.content["hypothesis_id"] == "hyp1"

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_send_to_executor(self, mock_wm, mock_llm):
        """Verify director can send messages to Executor."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")
        director.register_agent("Executor", "executor_001")

        message = director._send_to_executor(protocol_id="exp1")

        assert isinstance(message, AgentMessage)
        assert message.type == MessageType.REQUEST
        assert message.content["action"] == "execute_experiment"
        assert message.content["protocol_id"] == "exp1"

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_send_to_data_analyst(self, mock_wm, mock_llm):
        """Verify director can send messages to DataAnalystAgent."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")
        director.register_agent("DataAnalystAgent", "analyst_001")

        message = director._send_to_data_analyst(result_id="res1", hypothesis_id="hyp1")

        assert isinstance(message, AgentMessage)
        assert message.type == MessageType.REQUEST
        assert message.content["action"] == "interpret_results"
        assert message.content["result_id"] == "res1"
        assert message.content["hypothesis_id"] == "hyp1"

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_send_to_hypothesis_refiner(self, mock_wm, mock_llm):
        """Verify director can send messages to HypothesisRefiner."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")
        director.register_agent("HypothesisRefiner", "refiner_001")

        message = director._send_to_hypothesis_refiner(hypothesis_id="hyp1", action="evaluate")

        assert isinstance(message, AgentMessage)
        assert message.type == MessageType.REQUEST
        assert message.content["action"] == "evaluate"
        assert message.content["hypothesis_id"] == "hyp1"

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_send_to_convergence_detector(self, mock_wm, mock_llm):
        """Verify director can send messages to ConvergenceDetector."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")
        director.register_agent("ConvergenceDetector", "convergence_001")

        message = director._send_to_convergence_detector()

        assert isinstance(message, AgentMessage)
        assert message.type == MessageType.REQUEST
        assert message.content["action"] == "check_convergence"
        assert "research_plan" in message.content

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_pending_requests_tracked(self, mock_wm, mock_llm):
        """Verify pending requests are tracked with correlation IDs."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")
        director.register_agent("HypothesisGeneratorAgent", "hyp_gen_001")

        initial_pending = len(director.pending_requests)

        message = director._send_to_hypothesis_generator(action="generate")

        assert len(director.pending_requests) == initial_pending + 1
        assert message.id in director.pending_requests
        assert director.pending_requests[message.id]["agent"] == "HypothesisGeneratorAgent"


@pytest.mark.requirement("REQ-ORCH-TASK-003")
@pytest.mark.priority("MUST")
class TestREQ_ORCH_TASK_003_MessageHandling:
    """
    REQ-ORCH-TASK-003: The Research Director MUST process incoming messages
    from agents and route them to appropriate handlers.
    """

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_process_message_routes_to_handlers(self, mock_wm, mock_llm):
        """Verify messages are routed to correct handlers based on agent type."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        # Create test message from HypothesisGeneratorAgent
        message = AgentMessage(
            from_agent="hyp_gen_001",
            to_agent=director.agent_id,
            type=MessageType.RESPONSE,
            content={"hypothesis_ids": ["hyp1"], "count": 1},
            metadata={"agent_type": "HypothesisGeneratorAgent"}
        )

        with patch.object(director, '_handle_hypothesis_generator_response') as mock_handler:
            director.process_message(message)
            mock_handler.assert_called_once_with(message)

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_handle_hypothesis_generator_response(self, mock_wm, mock_llm):
        """Verify hypothesis generator responses are handled correctly."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        message = AgentMessage(
            from_agent="hyp_gen_001",
            to_agent=director.agent_id,
            type=MessageType.RESPONSE,
            content={"hypothesis_ids": ["hyp1", "hyp2"], "count": 2},
            metadata={"agent_type": "HypothesisGeneratorAgent"}
        )

        with patch.object(director, '_persist_hypothesis_to_graph'):
            director._handle_hypothesis_generator_response(message)

        # Verify hypotheses added to research plan
        assert len(director.research_plan.hypothesis_pool) == 2
        assert "hyp1" in director.research_plan.hypothesis_pool
        assert "hyp2" in director.research_plan.hypothesis_pool

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_handle_experiment_designer_response(self, mock_wm, mock_llm):
        """Verify experiment designer responses are handled correctly."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        message = AgentMessage(
            from_agent="exp_designer_001",
            to_agent=director.agent_id,
            type=MessageType.RESPONSE,
            content={"protocol_id": "exp1", "hypothesis_id": "hyp1"},
            metadata={"agent_type": "ExperimentDesignerAgent"}
        )

        with patch.object(director, '_persist_protocol_to_graph'):
            director._handle_experiment_designer_response(message)

        # Verify experiment added to queue
        assert "exp1" in director.research_plan.experiment_queue

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_handle_executor_response(self, mock_wm, mock_llm):
        """Verify executor responses are handled correctly."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")
        director.research_plan.add_experiment("exp1")

        message = AgentMessage(
            from_agent="executor_001",
            to_agent=director.agent_id,
            type=MessageType.RESPONSE,
            content={
                "result_id": "res1",
                "protocol_id": "exp1",
                "hypothesis_id": "hyp1",
                "status": "SUCCESS"
            },
            metadata={"agent_type": "Executor"}
        )

        with patch.object(director, '_persist_result_to_graph'):
            director._handle_executor_response(message)

        # Verify result added and experiment completed
        assert "res1" in director.research_plan.results
        assert "exp1" in director.research_plan.completed_experiments
        assert "exp1" not in director.research_plan.experiment_queue

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_handle_data_analyst_response(self, mock_wm, mock_llm):
        """Verify data analyst responses are handled correctly."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")
        director.research_plan.add_hypothesis("hyp1")

        message = AgentMessage(
            from_agent="analyst_001",
            to_agent=director.agent_id,
            type=MessageType.RESPONSE,
            content={
                "result_id": "res1",
                "hypothesis_id": "hyp1",
                "hypothesis_supported": True,
                "confidence": 0.85
            },
            metadata={"agent_type": "DataAnalystAgent"}
        )

        with patch.object(director, '_add_support_relationship'):
            director._handle_data_analyst_response(message)

        # Verify hypothesis marked as supported
        assert "hyp1" in director.research_plan.supported_hypotheses

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_handle_error_messages(self, mock_wm, mock_llm):
        """Verify error messages are handled without crashing."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        error_message = AgentMessage(
            from_agent="hyp_gen_001",
            to_agent=director.agent_id,
            type=MessageType.ERROR,
            content={"error": "Generation failed"},
            metadata={"agent_type": "HypothesisGeneratorAgent"}
        )

        # Should not raise exception
        director._handle_hypothesis_generator_response(error_message)

        # Verify error tracked
        assert director.errors_encountered > 0


@pytest.mark.requirement("REQ-ORCH-TASK-004")
@pytest.mark.priority("MUST")
class TestREQ_ORCH_TASK_004_AgentRegistry:
    """
    REQ-ORCH-TASK-004: The Research Director MUST maintain an agent registry
    for tracking and routing messages to specialized agents.
    """

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_agent_registry_initialization(self, mock_wm, mock_llm):
        """Verify agent registry is initialized empty."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        assert hasattr(director, 'agent_registry')
        assert isinstance(director.agent_registry, dict)
        assert len(director.agent_registry) == 0

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_register_agent(self, mock_wm, mock_llm):
        """Verify agents can be registered."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        director.register_agent("HypothesisGeneratorAgent", "hyp_gen_001")

        assert "HypothesisGeneratorAgent" in director.agent_registry
        assert director.agent_registry["HypothesisGeneratorAgent"] == "hyp_gen_001"

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_register_multiple_agents(self, mock_wm, mock_llm):
        """Verify multiple agents can be registered."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        agents = {
            "HypothesisGeneratorAgent": "hyp_gen_001",
            "ExperimentDesignerAgent": "exp_designer_001",
            "DataAnalystAgent": "analyst_001",
            "Executor": "executor_001"
        }

        for agent_type, agent_id in agents.items():
            director.register_agent(agent_type, agent_id)

        assert len(director.agent_registry) == 4
        for agent_type, agent_id in agents.items():
            assert director.agent_registry[agent_type] == agent_id

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_get_agent_id(self, mock_wm, mock_llm):
        """Verify agent IDs can be retrieved by type."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")
        director.register_agent("HypothesisGeneratorAgent", "hyp_gen_001")

        agent_id = director.get_agent_id("HypothesisGeneratorAgent")

        assert agent_id == "hyp_gen_001"

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_get_nonexistent_agent_returns_none(self, mock_wm, mock_llm):
        """Verify getting nonexistent agent returns None."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        agent_id = director.get_agent_id("NonexistentAgent")

        assert agent_id is None


@pytest.mark.requirement("REQ-ORCH-TASK-005")
@pytest.mark.priority("MUST")
class TestREQ_ORCH_TASK_005_ActionExecution:
    """
    REQ-ORCH-TASK-005: The Research Director MUST execute decided actions by
    invoking _execute_next_action() with appropriate agent coordination.
    """

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_execute_next_action_generate_hypothesis(self, mock_wm, mock_llm):
        """Verify GENERATE_HYPOTHESIS action execution."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")
        director.register_agent("HypothesisGeneratorAgent", "hyp_gen_001")

        with patch.object(director, '_send_to_hypothesis_generator') as mock_send:
            director._execute_next_action(NextAction.GENERATE_HYPOTHESIS)
            mock_send.assert_called_once_with(action="generate")

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_execute_next_action_design_experiment(self, mock_wm, mock_llm):
        """Verify DESIGN_EXPERIMENT action execution."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")
        director.register_agent("ExperimentDesignerAgent", "exp_designer_001")
        director.research_plan.add_hypothesis("hyp1")

        with patch.object(director, '_send_to_experiment_designer') as mock_send:
            director._execute_next_action(NextAction.DESIGN_EXPERIMENT)
            mock_send.assert_called_once()
            # Verify hypothesis_id was passed
            args, kwargs = mock_send.call_args
            assert args[0] == "hyp1" or kwargs.get("hypothesis_id") == "hyp1"

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_execute_next_action_execute_experiment(self, mock_wm, mock_llm):
        """Verify EXECUTE_EXPERIMENT action execution."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")
        director.register_agent("Executor", "executor_001")
        director.research_plan.add_experiment("exp1")

        with patch.object(director, '_send_to_executor') as mock_send:
            director._execute_next_action(NextAction.EXECUTE_EXPERIMENT)
            mock_send.assert_called_once()

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_execute_next_action_analyze_result(self, mock_wm, mock_llm):
        """Verify ANALYZE_RESULT action execution."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")
        director.register_agent("DataAnalystAgent", "analyst_001")
        director.research_plan.add_result("res1")

        with patch.object(director, '_send_to_data_analyst') as mock_send:
            director._execute_next_action(NextAction.ANALYZE_RESULT)
            mock_send.assert_called_once()

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_execute_next_action_refine_hypothesis(self, mock_wm, mock_llm):
        """Verify REFINE_HYPOTHESIS action execution."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")
        director.register_agent("HypothesisRefiner", "refiner_001")
        director.research_plan.add_hypothesis("hyp1")
        director.research_plan.mark_tested("hyp1")

        with patch.object(director, '_send_to_hypothesis_refiner') as mock_send:
            director._execute_next_action(NextAction.REFINE_HYPOTHESIS)
            mock_send.assert_called_once()

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_execute_next_action_converge(self, mock_wm, mock_llm):
        """Verify CONVERGE action execution."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")
        director.register_agent("ConvergenceDetector", "convergence_001")

        with patch.object(director, '_send_to_convergence_detector') as mock_send:
            director._execute_next_action(NextAction.CONVERGE)
            mock_send.assert_called_once()


@pytest.mark.requirement("REQ-ORCH-TASK-006")
@pytest.mark.priority("MUST")
class TestREQ_ORCH_TASK_006_StrategyTracking:
    """
    REQ-ORCH-TASK-006: The Research Director MUST track strategy effectiveness
    (attempts, successes, costs) for adaptive decision-making.
    """

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_strategy_stats_initialized(self, mock_wm, mock_llm):
        """Verify strategy statistics are initialized."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        assert director.strategy_stats is not None
        assert isinstance(director.strategy_stats, dict)

        # Verify key strategies tracked
        expected_strategies = [
            "hypothesis_generation",
            "experiment_design",
            "hypothesis_refinement",
            "literature_review"
        ]

        for strategy in expected_strategies:
            assert strategy in director.strategy_stats

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_update_strategy_effectiveness(self, mock_wm, mock_llm):
        """Verify strategy effectiveness can be updated."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        initial_attempts = director.strategy_stats["hypothesis_generation"]["attempts"]

        director.update_strategy_effectiveness("hypothesis_generation", success=True, cost=100.0)

        assert director.strategy_stats["hypothesis_generation"]["attempts"] == initial_attempts + 1
        assert director.strategy_stats["hypothesis_generation"]["successes"] == 1
        assert director.strategy_stats["hypothesis_generation"]["cost"] == 100.0

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_select_best_strategy(self, mock_wm, mock_llm):
        """Verify best strategy is selected based on effectiveness."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        # Update strategies with different success rates
        director.update_strategy_effectiveness("hypothesis_generation", success=True)
        director.update_strategy_effectiveness("hypothesis_generation", success=True)
        director.update_strategy_effectiveness("experiment_design", success=True)
        director.update_strategy_effectiveness("experiment_design", success=False)

        best_strategy = director.select_next_strategy()

        # hypothesis_generation has 100% success rate (2/2)
        # experiment_design has 50% success rate (1/2)
        assert best_strategy == "hypothesis_generation"


@pytest.mark.requirement("REQ-ORCH-TASK-007")
@pytest.mark.priority("SHOULD")
class TestREQ_ORCH_TASK_007_ResearchPlanGeneration:
    """
    REQ-ORCH-TASK-007: The Research Director SHOULD generate initial research
    plan using LLM (generate_research_plan) to guide investigation.
    """

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_generate_research_plan_exists(self, mock_wm, mock_llm):
        """Verify generate_research_plan method exists."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        assert hasattr(director, 'generate_research_plan')
        assert callable(director.generate_research_plan)

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_generate_research_plan_uses_llm(self, mock_wm, mock_llm):
        """Verify research plan generation uses LLM."""
        mock_client = Mock()
        mock_client.generate = Mock(return_value="Test research plan")
        mock_llm.return_value = mock_client
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test question", domain="biology")

        plan = director.generate_research_plan()

        # Verify LLM was called
        mock_client.generate.assert_called_once()
        call_args = mock_client.generate.call_args
        prompt = call_args[0][0]

        # Verify prompt includes research question and domain
        assert "Test question" in prompt
        assert "biology" in prompt

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_research_plan_stored_in_research_plan(self, mock_wm, mock_llm):
        """Verify generated plan is stored in research plan."""
        mock_client = Mock()
        mock_client.generate = Mock(return_value="Generated research strategy")
        mock_llm.return_value = mock_client
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        plan = director.generate_research_plan()

        assert director.research_plan.initial_strategy == "Generated research strategy"
        assert plan == "Generated research strategy"

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_research_plan_generation_handles_errors(self, mock_wm, mock_llm):
        """Verify research plan generation handles LLM errors gracefully."""
        mock_client = Mock()
        mock_client.generate = Mock(side_effect=Exception("LLM error"))
        mock_llm.return_value = mock_client
        mock_wm.return_value = None

        director = ResearchDirectorAgent(research_question="Test")

        # Should not raise exception
        plan = director.generate_research_plan()

        assert isinstance(plan, str)
        assert "error" in plan.lower() or "Error" in plan
