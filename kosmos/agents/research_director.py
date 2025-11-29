"""
Research Director Agent - Master orchestrator for autonomous research (Phase 7).

This agent coordinates all other agents to execute the full research cycle:
Research Question → Hypotheses → Experiments → Results → Analysis → Refinement → Iteration

Uses message-based async coordination with all specialized agents.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import asyncio
import concurrent.futures
import threading
from contextlib import contextmanager

from kosmos.agents.base import BaseAgent, AgentMessage, MessageType, AgentStatus
from kosmos.utils.compat import model_to_dict
from kosmos.core.workflow import (
    ResearchWorkflow,
    ResearchPlan,
    WorkflowState,
    NextAction
)
from kosmos.core.llm import get_client
from kosmos.core.stage_tracker import get_stage_tracker
from kosmos.models.hypothesis import Hypothesis, HypothesisStatus
from kosmos.world_model import get_world_model, Entity, Relationship
from kosmos.db import get_session
from kosmos.db.operations import get_hypothesis, get_experiment, get_result

logger = logging.getLogger(__name__)


class ResearchDirectorAgent(BaseAgent):
    """
    Master orchestrator for autonomous research.

    Coordinates:
    - HypothesisGeneratorAgent: Generate and refine hypotheses
    - ExperimentDesignerAgent: Design experiment protocols
    - Executor: Run experiments
    - DataAnalystAgent: Interpret results
    - HypothesisRefiner: Refine hypotheses based on results
    - ConvergenceDetector: Detect when research is complete

    Uses message-based coordination for async agent communication.
    """

    def __init__(
        self,
        research_question: str,
        domain: Optional[str] = None,
        agent_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Research Director.

        Args:
            research_question: The research question to investigate
            domain: Optional domain (biology, physics, etc.)
            agent_id: Optional agent ID
            config: Optional configuration (max_iterations, stopping_criteria, etc.)
        """
        super().__init__(
            agent_id=agent_id,
            agent_type="ResearchDirector",
            config=config or {}
        )

        self.research_question = research_question
        self.domain = domain

        # Configuration
        self.max_iterations = self.config.get("max_iterations", 10)
        self.mandatory_stopping_criteria = self.config.get(
            "mandatory_stopping_criteria",
            ["iteration_limit", "no_testable_hypotheses"]
        )
        self.optional_stopping_criteria = self.config.get(
            "optional_stopping_criteria",
            ["novelty_decline", "diminishing_returns"]
        )

        # Initialize research plan and workflow
        self.research_plan = ResearchPlan(
            research_question=research_question,
            domain=domain,
            max_iterations=self.max_iterations
        )

        self.workflow = ResearchWorkflow(
            initial_state=WorkflowState.INITIALIZING,
            research_plan=self.research_plan
        )

        # Claude client for research planning and decision-making
        self.llm_client = get_client()

        # Initialize database if not already initialized
        from kosmos.db import init_from_config
        try:
            init_from_config()
        except RuntimeError:
            # Database already initialized
            pass
        except Exception as e:
            logger.warning(f"Database initialization failed: {e}")

        # Agent registry (will be populated during coordination)
        self.agent_registry: Dict[str, str] = {}  # agent_type -> agent_id

        # Message correlation tracking
        self.pending_requests: Dict[str, Dict[str, Any]] = {}  # correlation_id -> request_info

        # Strategy effectiveness tracking
        self.strategy_stats: Dict[str, Dict[str, Any]] = {
            "hypothesis_generation": {"attempts": 0, "successes": 0, "cost": 0.0},
            "experiment_design": {"attempts": 0, "successes": 0, "cost": 0.0},
            "hypothesis_refinement": {"attempts": 0, "successes": 0, "cost": 0.0},
            "literature_review": {"attempts": 0, "successes": 0, "cost": 0.0}
        }

        # Research history
        self.iteration_history: List[Dict[str, Any]] = []

        # Thread safety locks for concurrent operations
        self._research_plan_lock = threading.RLock()  # Reentrant lock for nested acquisitions
        self._strategy_stats_lock = threading.Lock()
        self._workflow_lock = threading.Lock()
        self._agent_registry_lock = threading.Lock()

        # Concurrent operations support
        self.enable_concurrent = self.config.get("enable_concurrent_operations", False)
        self.max_parallel_hypotheses = self.config.get("max_parallel_hypotheses", 3)
        self.max_concurrent_experiments = self.config.get("max_concurrent_experiments", 4)

        # Initialize ParallelExperimentExecutor if concurrent operations enabled
        self.parallel_executor = None
        if self.enable_concurrent:
            try:
                from kosmos.execution.parallel import ParallelExperimentExecutor
                self.parallel_executor = ParallelExperimentExecutor(
                    max_workers=self.max_concurrent_experiments
                )
                logger.info(
                    f"Parallel execution enabled with {self.max_concurrent_experiments} workers"
                )
            except ImportError:
                logger.warning("ParallelExperimentExecutor not available, using sequential execution")
                self.enable_concurrent = False

        # Initialize AsyncClaudeClient for concurrent LLM calls
        self.async_llm_client = None
        if self.enable_concurrent:
            try:
                from kosmos.core.async_llm import AsyncClaudeClient
                import os
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if api_key:
                    self.async_llm_client = AsyncClaudeClient(
                        api_key=api_key,
                        max_concurrent=self.config.get("max_concurrent_llm_calls", 5),
                        max_requests_per_minute=self.config.get("llm_rate_limit_per_minute", 50)
                    )
                    logger.info("Async LLM client initialized for concurrent operations")
                else:
                    logger.warning("ANTHROPIC_API_KEY not set, async LLM disabled")
            except ImportError:
                logger.warning("AsyncClaudeClient not available, using sequential LLM calls")

        # Initialize world model for persistent knowledge graph
        try:
            self.wm = get_world_model()
            # Create ResearchQuestion entity
            question_entity = Entity.from_research_question(
                question_text=research_question,
                domain=domain,
                created_by=f"ResearchDirectorAgent:{self.agent_id}"
            )
            self.question_entity_id = self.wm.add_entity(question_entity)
            logger.info(f"Research question persisted to knowledge graph: {self.question_entity_id}")
        except Exception as e:
            logger.warning(f"Failed to initialize world model: {e}. Continuing without graph persistence.")
            self.wm = None
            self.question_entity_id = None

        logger.info(
            f"ResearchDirector initialized for question: '{research_question}' "
            f"(max_iterations={self.max_iterations}, concurrent={self.enable_concurrent})"
        )

    # ========================================================================
    # LIFECYCLE HOOKS
    # ========================================================================

    def _on_start(self):
        """Initialize director when started."""
        logger.info(f"ResearchDirector {self.agent_id} starting research cycle")
        with self._workflow_lock:
            self.workflow.transition_to(
                WorkflowState.GENERATING_HYPOTHESES,
                action="Start research cycle"
            )

    def _on_stop(self):
        """Cleanup when stopped."""
        logger.info(f"ResearchDirector {self.agent_id} stopped")

        # Cleanup async resources
        if self.async_llm_client:
            try:
                asyncio.run(self.async_llm_client.close())
            except Exception as e:
                logger.warning(f"Error closing async LLM client: {e}")

    # ========================================================================
    # THREAD-SAFE CONTEXT MANAGERS
    # ========================================================================

    @contextmanager
    def _research_plan_context(self):
        """Context manager for thread-safe research plan access."""
        self._research_plan_lock.acquire()
        try:
            yield self.research_plan
        finally:
            self._research_plan_lock.release()

    @contextmanager
    def _strategy_stats_context(self):
        """Context manager for thread-safe strategy stats access."""
        self._strategy_stats_lock.acquire()
        try:
            yield self.strategy_stats
        finally:
            self._strategy_stats_lock.release()

    @contextmanager
    def _workflow_context(self):
        """Context manager for thread-safe workflow access."""
        self._workflow_lock.acquire()
        try:
            yield self.workflow
        finally:
            self._workflow_lock.release()

    # ========================================================================
    # GRAPH PERSISTENCE HELPERS
    # ========================================================================

    def _persist_hypothesis_to_graph(self, hypothesis_id: str, agent_name: str = "HypothesisGeneratorAgent"):
        """
        Persist hypothesis to knowledge graph with SPAWNED_BY relationship.

        Args:
            hypothesis_id: ID of hypothesis to persist
            agent_name: Name of agent that created the hypothesis
        """
        if not self.wm or not self.question_entity_id:
            return  # Graph persistence disabled

        try:
            with get_session() as session:
                # Fetch hypothesis from database
                hypothesis = get_hypothesis(session, hypothesis_id)
                if not hypothesis:
                    logger.warning(f"Hypothesis {hypothesis_id} not found in database")
                    return

                # Convert to Entity and persist
                entity = Entity.from_hypothesis(hypothesis, created_by=agent_name)
                entity_id = self.wm.add_entity(entity)

                # Create SPAWNED_BY relationship to research question
                rel = Relationship.with_provenance(
                    source_id=entity_id,
                    target_id=self.question_entity_id,
                    rel_type="SPAWNED_BY",
                    agent=agent_name,
                    generation=hypothesis.generation,
                    iteration=self.research_plan.iteration_count
                )
                self.wm.add_relationship(rel)

                # If refined from parent, add REFINED_FROM relationship
                if hypothesis.parent_hypothesis_id:
                    parent_rel = Relationship.with_provenance(
                        source_id=entity_id,
                        target_id=hypothesis.parent_hypothesis_id,
                        rel_type="REFINED_FROM",
                        agent=agent_name,
                        refinement_count=hypothesis.refinement_count
                    )
                    self.wm.add_relationship(parent_rel)

                logger.debug(f"Persisted hypothesis {hypothesis_id} to graph")

        except Exception as e:
            logger.warning(f"Failed to persist hypothesis {hypothesis_id} to graph: {e}")

    def _persist_protocol_to_graph(self, protocol_id: str, hypothesis_id: str, agent_name: str = "ExperimentDesignerAgent"):
        """
        Persist experiment protocol to knowledge graph with TESTS relationship.

        Args:
            protocol_id: ID of protocol to persist
            hypothesis_id: ID of hypothesis being tested
            agent_name: Name of agent that created the protocol
        """
        if not self.wm:
            return

        try:
            with get_session() as session:
                # Fetch protocol from database
                protocol = get_experiment(session, protocol_id)
                if not protocol:
                    logger.warning(f"Protocol {protocol_id} not found in database")
                    return

                # Convert to Entity and persist
                entity = Entity.from_protocol(protocol, created_by=agent_name)
                entity_id = self.wm.add_entity(entity)

                # Create TESTS relationship to hypothesis
                rel = Relationship.with_provenance(
                    source_id=entity_id,
                    target_id=hypothesis_id,
                    rel_type="TESTS",
                    agent=agent_name,
                    iteration=self.research_plan.iteration_count
                )
                self.wm.add_relationship(rel)

                logger.debug(f"Persisted protocol {protocol_id} to graph")

        except Exception as e:
            logger.warning(f"Failed to persist protocol {protocol_id} to graph: {e}")

    def _persist_result_to_graph(self, result_id: str, protocol_id: str, hypothesis_id: str, agent_name: str = "Executor"):
        """
        Persist experiment result to knowledge graph with PRODUCED_BY relationship.

        Args:
            result_id: ID of result to persist
            protocol_id: ID of protocol that produced this result
            hypothesis_id: ID of hypothesis being tested
            agent_name: Name of agent that created the result
        """
        if not self.wm:
            return

        try:
            with get_session() as session:
                # Fetch result from database
                result = get_result(session, result_id)
                if not result:
                    logger.warning(f"Result {result_id} not found in database")
                    return

                # Convert to Entity and persist
                entity = Entity.from_result(result, created_by=agent_name)
                entity_id = self.wm.add_entity(entity)

                # Create PRODUCED_BY relationship to protocol
                rel = Relationship.with_provenance(
                    source_id=entity_id,
                    target_id=protocol_id,
                    rel_type="PRODUCED_BY",
                    agent=agent_name,
                    iteration=self.research_plan.iteration_count
                )
                self.wm.add_relationship(rel)

                # Create TESTS relationship to hypothesis
                tests_rel = Relationship.with_provenance(
                    source_id=entity_id,
                    target_id=hypothesis_id,
                    rel_type="TESTS",
                    agent=agent_name
                )
                self.wm.add_relationship(tests_rel)

                logger.debug(f"Persisted result {result_id} to graph")

        except Exception as e:
            logger.warning(f"Failed to persist result {result_id} to graph: {e}")

    def _add_support_relationship(self, result_id: str, hypothesis_id: str, supports: bool, confidence: float, p_value: float = None, effect_size: float = None):
        """
        Add SUPPORTS or REFUTES relationship based on result analysis.

        Args:
            result_id: ID of result entity
            hypothesis_id: ID of hypothesis entity
            supports: True if result supports hypothesis, False if refutes
            confidence: Confidence score from analyst
            p_value: Statistical p-value if available
            effect_size: Effect size if available
        """
        if not self.wm:
            return

        try:
            rel_type = "SUPPORTS" if supports else "REFUTES"
            metadata = {"iteration": self.research_plan.iteration_count}
            if p_value is not None:
                metadata["p_value"] = p_value
            if effect_size is not None:
                metadata["effect_size"] = effect_size

            rel = Relationship.with_provenance(
                source_id=result_id,
                target_id=hypothesis_id,
                rel_type=rel_type,
                agent="DataAnalystAgent",
                confidence=confidence,
                **metadata
            )
            self.wm.add_relationship(rel)

            logger.debug(f"Added {rel_type} relationship: result {result_id} -> hypothesis {hypothesis_id}")

        except Exception as e:
            logger.warning(f"Failed to add {rel_type} relationship: {e}")

    # ========================================================================
    # MESSAGE HANDLING
    # ========================================================================

    def process_message(self, message: AgentMessage):
        """
        Process incoming message from other agents.

        Routes messages to appropriate handlers based on source agent.
        """
        # Extract sender agent type from message metadata or from_agent
        sender_type = message.metadata.get("agent_type", "unknown")

        logger.debug(f"Processing message from {sender_type} ({message.from_agent})")

        # Route to appropriate handler
        if sender_type == "HypothesisGeneratorAgent":
            self._handle_hypothesis_generator_response(message)
        elif sender_type == "ExperimentDesignerAgent":
            self._handle_experiment_designer_response(message)
        elif sender_type == "Executor":
            self._handle_executor_response(message)
        elif sender_type == "DataAnalystAgent":
            self._handle_data_analyst_response(message)
        elif sender_type == "HypothesisRefiner":
            self._handle_hypothesis_refiner_response(message)
        elif sender_type == "ConvergenceDetector":
            self._handle_convergence_detector_response(message)
        else:
            logger.warning(f"No handler for agent type: {sender_type}")

    def _handle_hypothesis_generator_response(self, message: AgentMessage):
        """
        Handle response from HypothesisGeneratorAgent.

        Expected content:
        - hypotheses: List of generated Hypothesis objects
        - count: Number of hypotheses generated
        """
        content = message.content

        if message.type == MessageType.ERROR:
            logger.error(f"Hypothesis generation failed: {content.get('error')}")
            self.errors_encountered += 1
            # TODO: Implement error recovery strategy
            return

        # Extract hypotheses
        hypothesis_ids = content.get("hypothesis_ids", [])
        count = content.get("count", 0)

        logger.info(f"Received {count} hypotheses from generator")

        # Update research plan (thread-safe)
        with self._research_plan_context():
            for hyp_id in hypothesis_ids:
                self.research_plan.add_hypothesis(hyp_id)

        # Persist hypotheses to knowledge graph
        for hyp_id in hypothesis_ids:
            self._persist_hypothesis_to_graph(hyp_id, agent_name="HypothesisGeneratorAgent")

        # Update strategy stats (thread-safe)
        with self._strategy_stats_context():
            self.strategy_stats["hypothesis_generation"]["attempts"] += 1
            if count > 0:
                self.strategy_stats["hypothesis_generation"]["successes"] += 1

        # Decide next action
        next_action = self.decide_next_action()
        self._execute_next_action(next_action)

    def _handle_experiment_designer_response(self, message: AgentMessage):
        """
        Handle response from ExperimentDesignerAgent.

        Expected content:
        - protocol_id: ID of designed experiment protocol
        - hypothesis_id: ID of hypothesis being tested
        """
        content = message.content

        if message.type == MessageType.ERROR:
            logger.error(f"Experiment design failed: {content.get('error')}")
            self.errors_encountered += 1
            return

        protocol_id = content.get("protocol_id")
        hypothesis_id = content.get("hypothesis_id")

        logger.info(f"Received experiment design: {protocol_id} for hypothesis {hypothesis_id}")

        # Update research plan (thread-safe)
        with self._research_plan_context():
            self.research_plan.add_experiment(protocol_id)

        # Persist protocol to knowledge graph
        if protocol_id and hypothesis_id:
            self._persist_protocol_to_graph(protocol_id, hypothesis_id, agent_name="ExperimentDesignerAgent")

        # Update strategy stats (thread-safe)
        with self._strategy_stats_context():
            self.strategy_stats["experiment_design"]["attempts"] += 1
            if protocol_id:
                self.strategy_stats["experiment_design"]["successes"] += 1

        # Decide next action
        next_action = self.decide_next_action()
        self._execute_next_action(next_action)

    def _handle_executor_response(self, message: AgentMessage):
        """
        Handle response from Executor.

        Expected content:
        - result_id: ID of experiment result
        - protocol_id: ID of protocol executed
        - status: SUCCESS/FAILURE/ERROR
        """
        content = message.content

        if message.type == MessageType.ERROR:
            logger.error(f"Experiment execution failed: {content.get('error')}")
            self.errors_encountered += 1
            return

        result_id = content.get("result_id")
        protocol_id = content.get("protocol_id")
        status = content.get("status")
        hypothesis_id = content.get("hypothesis_id")  # May not be present

        logger.info(f"Received experiment result: {result_id} (status: {status})")

        # Update research plan (thread-safe)
        with self._research_plan_context():
            self.research_plan.add_result(result_id)
            self.research_plan.mark_experiment_complete(protocol_id)

        # Persist result to knowledge graph (get hypothesis_id from protocol if needed)
        if result_id and protocol_id:
            if not hypothesis_id:
                # Fetch hypothesis_id from protocol
                try:
                    with get_session() as session:
                        protocol = get_experiment(session, protocol_id)
                        if protocol:
                            hypothesis_id = protocol.hypothesis_id
                except Exception as e:
                    logger.warning(f"Failed to fetch hypothesis_id from protocol: {e}")

            if hypothesis_id:
                self._persist_result_to_graph(result_id, protocol_id, hypothesis_id, agent_name="Executor")

        # Transition to analyzing state (thread-safe)
        with self._workflow_context():
            self.workflow.transition_to(
                WorkflowState.ANALYZING,
                action=f"Analyze result {result_id}"
            )

        # Send to DataAnalystAgent for interpretation
        next_action = NextAction.ANALYZE_RESULT
        self._execute_next_action(next_action)

    def _handle_data_analyst_response(self, message: AgentMessage):
        """
        Handle response from DataAnalystAgent.

        Expected content:
        - interpretation: ResultInterpretation object
        - result_id: ID of analyzed result
        - hypothesis_supported: bool
        """
        content = message.content

        if message.type == MessageType.ERROR:
            logger.error(f"Result analysis failed: {content.get('error')}")
            self.errors_encountered += 1
            return

        result_id = content.get("result_id")
        hypothesis_id = content.get("hypothesis_id")
        hypothesis_supported = content.get("hypothesis_supported")
        confidence = content.get("confidence", 0.8)  # Default confidence
        p_value = content.get("p_value")
        effect_size = content.get("effect_size")

        logger.info(
            f"Received result interpretation for {result_id}: "
            f"hypothesis {hypothesis_id} supported={hypothesis_supported}"
        )

        # Update hypothesis status in research plan (thread-safe)
        with self._research_plan_context():
            if hypothesis_supported is True:
                self.research_plan.mark_supported(hypothesis_id)
            elif hypothesis_supported is False:
                self.research_plan.mark_rejected(hypothesis_id)
            else:
                # Inconclusive
                self.research_plan.mark_tested(hypothesis_id)

        # Add SUPPORTS/REFUTES relationship to knowledge graph
        if result_id and hypothesis_id and hypothesis_supported is not None:
            self._add_support_relationship(
                result_id,
                hypothesis_id,
                supports=hypothesis_supported,
                confidence=confidence,
                p_value=p_value,
                effect_size=effect_size
            )

        # Transition to refining state (thread-safe)
        with self._workflow_context():
            self.workflow.transition_to(
                WorkflowState.REFINING,
                action=f"Refine based on result {result_id}"
            )

        # Decide next action (may refine hypothesis, generate new ones, or converge)
        next_action = self.decide_next_action()
        self._execute_next_action(next_action)

    def _handle_hypothesis_refiner_response(self, message: AgentMessage):
        """
        Handle response from HypothesisRefiner.

        Expected content:
        - refined_hypothesis_ids: List of refined/spawned hypothesis IDs
        - retired_hypothesis_ids: List of retired hypothesis IDs
        - action_taken: REFINED/RETIRED/SPAWNED
        """
        content = message.content

        if message.type == MessageType.ERROR:
            logger.error(f"Hypothesis refinement failed: {content.get('error')}")
            self.errors_encountered += 1
            return

        refined_ids = content.get("refined_hypothesis_ids", [])
        retired_ids = content.get("retired_hypothesis_ids", [])

        logger.info(f"Hypothesis refinement: {len(refined_ids)} refined, {len(retired_ids)} retired")

        # Add refined hypotheses to pool (thread-safe)
        with self._research_plan_context():
            for hyp_id in refined_ids:
                self.research_plan.add_hypothesis(hyp_id)

        # Persist refined hypotheses to knowledge graph
        for hyp_id in refined_ids:
            self._persist_hypothesis_to_graph(hyp_id, agent_name="HypothesisRefiner")

        # Update strategy stats (thread-safe)
        with self._strategy_stats_context():
            self.strategy_stats["hypothesis_refinement"]["attempts"] += 1
            if refined_ids:
                self.strategy_stats["hypothesis_refinement"]["successes"] += 1

        # Decide next action
        next_action = self.decide_next_action()
        self._execute_next_action(next_action)

    def _handle_convergence_detector_response(self, message: AgentMessage):
        """
        Handle response from ConvergenceDetector.

        Expected content:
        - should_converge: bool
        - reason: str (why convergence detected)
        - metrics: ConvergenceMetrics
        """
        content = message.content

        should_converge = content.get("should_converge", False)
        reason = content.get("reason", "")

        if should_converge:
            logger.info(f"Convergence detected: {reason}")

            # Update research plan (thread-safe)
            with self._research_plan_context():
                self.research_plan.has_converged = True
                self.research_plan.convergence_reason = reason

            # Add convergence annotation to research question in knowledge graph
            if self.wm and self.question_entity_id:
                try:
                    from kosmos.world_model.models import Annotation
                    convergence_annotation = Annotation(
                        text=f"Research converged: {reason}",
                        created_by="ConvergenceDetector"
                    )
                    self.wm.add_annotation(self.question_entity_id, convergence_annotation)
                    logger.debug("Added convergence annotation to research question")
                except Exception as e:
                    logger.warning(f"Failed to add convergence annotation: {e}")

            # Transition to converged state (thread-safe)
            with self._workflow_context():
                self.workflow.transition_to(
                    WorkflowState.CONVERGED,
                    action=f"Research converged: {reason}"
                )

            # Stop the director
            self.stop()
        else:
            logger.debug("Convergence check: not yet converged")

    # ========================================================================
    # MESSAGE SENDING (to other agents)
    # ========================================================================

    def _send_to_hypothesis_generator(
        self,
        action: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """
        Send request to HypothesisGeneratorAgent.

        Args:
            action: Action to request (generate, refine)
            context: Additional context (research_question, literature, etc.)

        Returns:
            AgentMessage: Sent message
        """
        content = {
            "action": action,
            "research_question": self.research_question,
            "domain": self.domain,
            "context": context or {}
        }

        target_agent = self.agent_registry.get("HypothesisGeneratorAgent", "hypothesis_generator")

        message = self.send_message(
            to_agent=target_agent,
            content=content,
            message_type=MessageType.REQUEST
        )

        self.pending_requests[message.id] = {
            "agent": "HypothesisGeneratorAgent",
            "action": action,
            "timestamp": datetime.utcnow()
        }

        logger.debug(f"Sent {action} request to HypothesisGeneratorAgent")
        return message

    def _send_to_experiment_designer(
        self,
        hypothesis_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """Send request to ExperimentDesignerAgent to design protocol."""
        content = {
            "action": "design_experiment",
            "hypothesis_id": hypothesis_id,
            "context": context or {}
        }

        target_agent = self.agent_registry.get("ExperimentDesignerAgent", "experiment_designer")

        message = self.send_message(
            to_agent=target_agent,
            content=content,
            message_type=MessageType.REQUEST
        )

        self.pending_requests[message.id] = {
            "agent": "ExperimentDesignerAgent",
            "hypothesis_id": hypothesis_id,
            "timestamp": datetime.utcnow()
        }

        logger.debug(f"Sent design request to ExperimentDesignerAgent for hypothesis {hypothesis_id}")
        return message

    def _send_to_executor(
        self,
        protocol_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """Send request to Executor to run experiment."""
        content = {
            "action": "execute_experiment",
            "protocol_id": protocol_id,
            "context": context or {}
        }

        target_agent = self.agent_registry.get("Executor", "executor")

        message = self.send_message(
            to_agent=target_agent,
            content=content,
            message_type=MessageType.REQUEST
        )

        self.pending_requests[message.id] = {
            "agent": "Executor",
            "protocol_id": protocol_id,
            "timestamp": datetime.utcnow()
        }

        logger.debug(f"Sent execution request to Executor for protocol {protocol_id}")
        return message

    def _send_to_data_analyst(
        self,
        result_id: str,
        hypothesis_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """Send request to DataAnalystAgent to interpret results."""
        content = {
            "action": "interpret_results",
            "result_id": result_id,
            "hypothesis_id": hypothesis_id,
            "context": context or {}
        }

        target_agent = self.agent_registry.get("DataAnalystAgent", "data_analyst")

        message = self.send_message(
            to_agent=target_agent,
            content=content,
            message_type=MessageType.REQUEST
        )

        self.pending_requests[message.id] = {
            "agent": "DataAnalystAgent",
            "result_id": result_id,
            "timestamp": datetime.utcnow()
        }

        logger.debug(f"Sent interpretation request to DataAnalystAgent for result {result_id}")
        return message

    def _send_to_hypothesis_refiner(
        self,
        hypothesis_id: str,
        result_id: Optional[str] = None,
        action: str = "evaluate",
        context: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """Send request to HypothesisRefiner."""
        content = {
            "action": action,
            "hypothesis_id": hypothesis_id,
            "result_id": result_id,
            "context": context or {}
        }

        target_agent = self.agent_registry.get("HypothesisRefiner", "hypothesis_refiner")

        message = self.send_message(
            to_agent=target_agent,
            content=content,
            message_type=MessageType.REQUEST
        )

        self.pending_requests[message.id] = {
            "agent": "HypothesisRefiner",
            "hypothesis_id": hypothesis_id,
            "timestamp": datetime.utcnow()
        }

        logger.debug(f"Sent {action} request to HypothesisRefiner for hypothesis {hypothesis_id}")
        return message

    def _send_to_convergence_detector(
        self,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """Send request to ConvergenceDetector to check if research is complete."""
        # Use model_dump() for Pydantic v2, fall back to dict() for v1
        try:
            research_plan_dict = model_to_dict(self.research_plan)
        except AttributeError:
            research_plan_dict = self.research_plan.dict()

        content = {
            "action": "check_convergence",
            "research_plan": research_plan_dict,
            "context": context or {}
        }

        target_agent = self.agent_registry.get("ConvergenceDetector", "convergence_detector")

        message = self.send_message(
            to_agent=target_agent,
            content=content,
            message_type=MessageType.REQUEST
        )

        self.pending_requests[message.id] = {
            "agent": "ConvergenceDetector",
            "timestamp": datetime.utcnow()
        }

        logger.debug("Sent convergence check request to ConvergenceDetector")
        return message

    # ========================================================================
    # CONCURRENT OPERATIONS
    # ========================================================================

    def execute_experiments_batch(self, protocol_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Execute multiple experiments in parallel using ParallelExperimentExecutor.

        Args:
            protocol_ids: List of protocol IDs to execute

        Returns:
            List of execution results

        Example:
            results = director.execute_experiments_batch(["proto1", "proto2", "proto3"])
        """
        if not self.enable_concurrent or not self.parallel_executor:
            logger.warning("Concurrent execution not enabled, falling back to sequential")
            results = []
            for protocol_id in protocol_ids:
                # Sequential fallback
                self._send_to_executor(protocol_id=protocol_id)
                results.append({"protocol_id": protocol_id, "status": "queued"})
            return results

        logger.info(f"Executing {len(protocol_ids)} experiments in parallel")

        try:
            # Execute batch using parallel executor
            batch_results = self.parallel_executor.execute_batch(protocol_ids)

            # Process results and update research plan
            for result in batch_results:
                if result.get("success"):
                    result_id = result.get("result_id")
                    protocol_id = result.get("protocol_id")

                    # Thread-safe update
                    with self._research_plan_context():
                        if result_id:
                            self.research_plan.add_result(result_id)
                        if protocol_id:
                            self.research_plan.mark_experiment_complete(protocol_id)

                    logger.info(f"Experiment {protocol_id} completed successfully")
                else:
                    logger.error(f"Experiment {result.get('protocol_id')} failed: {result.get('error')}")

            return batch_results

        except Exception as e:
            logger.error(f"Batch experiment execution failed: {e}")
            return [{"protocol_id": pid, "success": False, "error": str(e)} for pid in protocol_ids]

    async def evaluate_hypotheses_concurrently(self, hypothesis_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Evaluate multiple hypotheses concurrently using AsyncClaudeClient.

        Uses async LLM calls to evaluate testability and potential impact of hypotheses in parallel.

        Args:
            hypothesis_ids: List of hypothesis IDs to evaluate

        Returns:
            List of evaluation results with scores and recommendations

        Example:
            evaluations = await director.evaluate_hypotheses_concurrently(["hyp1", "hyp2", "hyp3"])
        """
        if not self.async_llm_client:
            logger.warning("Async LLM client not available, using sequential evaluation")
            return []

        logger.info(f"Evaluating {len(hypothesis_ids)} hypotheses concurrently")

        try:
            from kosmos.core.async_llm import BatchRequest

            # Create batch requests for hypothesis evaluation
            requests = []
            for i, hyp_id in enumerate(hypothesis_ids):
                # TODO: Load actual hypothesis text from database
                prompt = f"""Evaluate this hypothesis for testability and scientific merit:

Hypothesis ID: {hyp_id}
Research Question: {self.research_question}
Domain: {self.domain or "General"}

Rate on scale 1-10:
1. Testability: Can this be experimentally tested?
2. Novelty: Is this approach novel?
3. Impact: Would confirmation significantly advance the field?

Provide brief JSON response:
{{"testability": X, "novelty": X, "impact": X, "recommendation": "proceed/refine/reject", "reasoning": "brief explanation"}}
"""

                requests.append(BatchRequest(
                    id=hyp_id,
                    prompt=prompt,
                    system="You are a research evaluator. Provide concise, objective assessments.",
                    temperature=0.3  # Lower temperature for more consistent evaluations
                ))

            # Execute concurrent evaluations
            responses = await self.async_llm_client.batch_generate(requests)

            # Process responses
            evaluations = []
            for resp in responses:
                if resp.success:
                    try:
                        import json
                        # Parse JSON response
                        eval_data = json.loads(resp.response)
                        eval_data["hypothesis_id"] = resp.id
                        evaluations.append(eval_data)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse evaluation for {resp.id}")
                        evaluations.append({
                            "hypothesis_id": resp.id,
                            "error": "Parse error",
                            "recommendation": "refine"
                        })
                else:
                    evaluations.append({
                        "hypothesis_id": resp.id,
                        "error": resp.error,
                        "recommendation": "retry"
                    })

            logger.info(f"Completed {len(evaluations)} hypothesis evaluations")
            return evaluations

        except Exception as e:
            logger.error(f"Concurrent hypothesis evaluation failed: {e}")
            return []

    async def analyze_results_concurrently(self, result_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze multiple experiment results concurrently using AsyncClaudeClient.

        Performs parallel interpretation of results to identify patterns and insights.

        Args:
            result_ids: List of result IDs to analyze

        Returns:
            List of analysis results

        Example:
            analyses = await director.analyze_results_concurrently(["res1", "res2", "res3"])
        """
        if not self.async_llm_client:
            logger.warning("Async LLM client not available, using sequential analysis")
            return []

        logger.info(f"Analyzing {len(result_ids)} results concurrently")

        try:
            from kosmos.core.async_llm import BatchRequest

            # Create batch requests for result analysis
            requests = []
            for result_id in result_ids:
                # TODO: Load actual result data from database
                prompt = f"""Analyze this experiment result:

Result ID: {result_id}
Research Question: {self.research_question}

Provide analysis including:
1. Key findings
2. Statistical significance
3. Relationship to hypothesis
4. Next steps

Provide brief JSON response:
{{"significance": "high/medium/low", "hypothesis_supported": true/false/inconclusive, "key_finding": "summary", "next_steps": "recommendation"}}
"""

                requests.append(BatchRequest(
                    id=result_id,
                    prompt=prompt,
                    system="You are a data analyst. Provide objective, evidence-based interpretations.",
                    temperature=0.3
                ))

            # Execute concurrent analyses
            responses = await self.async_llm_client.batch_generate(requests)

            # Process responses
            analyses = []
            for resp in responses:
                if resp.success:
                    try:
                        import json
                        analysis_data = json.loads(resp.response)
                        analysis_data["result_id"] = resp.id
                        analyses.append(analysis_data)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse analysis for {resp.id}")
                        analyses.append({
                            "result_id": resp.id,
                            "error": "Parse error"
                        })
                else:
                    analyses.append({
                        "result_id": resp.id,
                        "error": resp.error
                    })

            logger.info(f"Completed {len(analyses)} result analyses")
            return analyses

        except Exception as e:
            logger.error(f"Concurrent result analysis failed: {e}")
            return []

    # ========================================================================
    # RESEARCH PLANNING (using Claude)
    # ========================================================================

    def generate_research_plan(self) -> str:
        """
        Generate initial research plan using Claude.

        Returns:
            str: Research plan description
        """
        prompt = f"""You are a research director planning an autonomous scientific investigation.

Research Question: {self.research_question}
Domain: {self.domain or "General"}

Please generate a research plan that includes:

1. **Initial Hypothesis Directions** (3-5 high-level directions to explore)
2. **Experiment Strategy** (what types of experiments would be most informative)
3. **Success Criteria** (how will we know when we've answered the question)
4. **Resource Considerations** (estimated experiments needed, complexity)

Provide a structured, actionable plan in 2-3 paragraphs.
"""

        try:
            response = self.llm_client.generate(prompt, max_tokens=1000)

            # Store in research plan
            self.research_plan.initial_strategy = response

            logger.info("Generated initial research plan using Claude")
            return response

        except Exception as e:
            logger.error(f"Failed to generate research plan: {e}")
            return f"Error generating plan: {str(e)}"

    # ========================================================================
    # DECISION MAKING
    # ========================================================================

    def decide_next_action(self) -> NextAction:
        """
        Decide what to do next based on current workflow state and research plan.

        Decision tree:
        - If no hypotheses: GENERATE_HYPOTHESIS
        - If untested hypotheses: DESIGN_EXPERIMENT
        - If experiments in queue: EXECUTE_EXPERIMENT
        - If results need analysis: ANALYZE_RESULT
        - If hypotheses need refinement: REFINE_HYPOTHESIS
        - If convergence criteria met: CONVERGE
        - Otherwise: Check convergence, then GENERATE_HYPOTHESIS

        Returns:
            NextAction: Next action to take
        """
        current_state = self.workflow.current_state

        # Enhanced debug logging
        logger.debug(
            "[DECISION] decide_next_action: state=%s, iteration=%d/%d, "
            "hypotheses=%d, untested=%d, experiments_queued=%d",
            current_state.value,
            self.research_plan.iteration_count,
            self.research_plan.max_iterations,
            len(self.research_plan.hypothesis_pool),
            len(self.research_plan.get_untested_hypotheses()),
            len(self.research_plan.experiment_queue)
        )

        # Check convergence first
        if self._should_check_convergence():
            return NextAction.CONVERGE

        # State-based decision making
        if current_state == WorkflowState.GENERATING_HYPOTHESES:
            # Generate hypotheses
            return NextAction.GENERATE_HYPOTHESIS

        elif current_state == WorkflowState.DESIGNING_EXPERIMENTS:
            # Design experiments for untested hypotheses
            untested = self.research_plan.get_untested_hypotheses()
            if untested:
                return NextAction.DESIGN_EXPERIMENT
            else:
                # All hypotheses tested, check convergence or generate more
                return NextAction.CONVERGE

        elif current_state == WorkflowState.EXECUTING:
            # Execute queued experiments
            if self.research_plan.experiment_queue:
                return NextAction.EXECUTE_EXPERIMENT
            else:
                return NextAction.ANALYZE_RESULT

        elif current_state == WorkflowState.ANALYZING:
            # Analyze recent results
            return NextAction.ANALYZE_RESULT

        elif current_state == WorkflowState.REFINING:
            # Refine hypotheses based on results
            if self.research_plan.tested_hypotheses:
                return NextAction.REFINE_HYPOTHESIS
            else:
                return NextAction.GENERATE_HYPOTHESIS

        elif current_state == WorkflowState.CONVERGED:
            return NextAction.CONVERGE

        elif current_state == WorkflowState.ERROR:
            return NextAction.ERROR_RECOVERY

        else:
            # Default: generate hypotheses
            return NextAction.GENERATE_HYPOTHESIS

    def _execute_next_action(self, action: NextAction):
        """
        Execute the decided next action.

        Uses concurrent execution when enabled and multiple items available.

        Args:
            action: Action to execute
        """
        logger.info("[ACTION] Executing: %s", action.value)

        tracker = get_stage_tracker()
        with tracker.track(f"ACTION_{action.value}", action=action.value):
            self._do_execute_action(action)

    def _do_execute_action(self, action: NextAction):
        """Internal method to execute action (wrapped by stage tracking)."""
        if action == NextAction.GENERATE_HYPOTHESIS:
            self._send_to_hypothesis_generator(action="generate")

        elif action == NextAction.DESIGN_EXPERIMENT:
            # Get untested hypotheses
            with self._research_plan_context():
                untested = self.research_plan.get_untested_hypotheses()

            if untested:
                # Use concurrent evaluation if enabled and multiple hypotheses
                if self.enable_concurrent and self.async_llm_client and len(untested) > 1:
                    # Evaluate multiple hypotheses concurrently (up to max_parallel_hypotheses)
                    batch_size = min(len(untested), self.max_parallel_hypotheses)
                    hypothesis_batch = untested[:batch_size]

                    try:
                        # Run async evaluation with timeout
                        logger.info(f"Starting concurrent evaluation of {len(hypothesis_batch)} hypotheses")
                        timeout_seconds = 300
                        evaluations = None
                        try:
                            # Check if there's already a running event loop
                            loop = asyncio.get_running_loop()
                            # If we're already in an async context, use run_coroutine_threadsafe with timeout
                            future = asyncio.run_coroutine_threadsafe(
                                self.evaluate_hypotheses_concurrently(hypothesis_batch), loop
                            )
                            evaluations = future.result(timeout=timeout_seconds)
                            logger.info(f"Concurrent hypothesis evaluation completed")
                        except RuntimeError:
                            # No running loop - this can cause issues with nested event loops
                            # Fall back to sequential execution for safety
                            logger.warning("No running event loop, falling back to sequential hypothesis evaluation")
                            evaluations = None
                        except concurrent.futures.TimeoutError:
                            logger.warning(f"Hypothesis evaluation timed out after {timeout_seconds}s, falling back to sequential")
                            evaluations = None

                        if evaluations:
                            # Process best candidate(s)
                            for eval_result in evaluations:
                                if eval_result.get("recommendation") == "proceed":
                                    self._send_to_experiment_designer(
                                        hypothesis_id=eval_result["hypothesis_id"]
                                    )
                                    break  # Design experiment for first promising hypothesis
                            else:
                                # No promising hypotheses, design for first untested
                                self._send_to_experiment_designer(hypothesis_id=untested[0])
                        else:
                            # Fallback to sequential
                            self._send_to_experiment_designer(hypothesis_id=untested[0])

                    except Exception as e:
                        logger.error(f"Concurrent hypothesis evaluation failed: {e}")
                        # Fallback to sequential
                        self._send_to_experiment_designer(hypothesis_id=untested[0])
                else:
                    # Sequential: design experiment for first untested hypothesis
                    self._send_to_experiment_designer(hypothesis_id=untested[0])

        elif action == NextAction.EXECUTE_EXPERIMENT:
            # Get queued experiments
            with self._research_plan_context():
                experiment_queue = list(self.research_plan.experiment_queue)

            if experiment_queue:
                # Use batch execution if enabled and multiple experiments queued
                if self.enable_concurrent and self.parallel_executor and len(experiment_queue) > 1:
                    # Execute multiple experiments in parallel
                    batch_size = min(len(experiment_queue), self.max_concurrent_experiments)
                    experiment_batch = experiment_queue[:batch_size]

                    logger.info(f"Executing {batch_size} experiments in parallel")
                    self.execute_experiments_batch(experiment_batch)
                else:
                    # Sequential: execute first queued experiment
                    protocol_id = experiment_queue[0]
                    self._send_to_executor(protocol_id=protocol_id)

        elif action == NextAction.ANALYZE_RESULT:
            # Get recent results
            with self._research_plan_context():
                results = list(self.research_plan.results)

            if results:
                # Use concurrent analysis if enabled and multiple results
                if self.enable_concurrent and self.async_llm_client and len(results) > 1:
                    # Analyze multiple recent results concurrently
                    batch_size = min(len(results), 5)  # Analyze up to 5 recent results
                    result_batch = results[-batch_size:]  # Most recent results

                    try:
                        # Run async analysis with timeout
                        logger.info(f"Starting concurrent analysis of {len(result_batch)} results")
                        timeout_seconds = 300
                        analyses = None
                        try:
                            # Check if there's already a running event loop
                            loop = asyncio.get_running_loop()
                            # If we're already in an async context, use run_coroutine_threadsafe with timeout
                            future = asyncio.run_coroutine_threadsafe(
                                self.analyze_results_concurrently(result_batch), loop
                            )
                            analyses = future.result(timeout=timeout_seconds)
                            logger.info(f"Concurrent result analysis completed")
                        except RuntimeError:
                            # No running loop - this can cause issues with nested event loops
                            # Fall back to sequential execution for safety
                            logger.warning("No running event loop, falling back to sequential result analysis")
                            analyses = None
                        except concurrent.futures.TimeoutError:
                            logger.warning(f"Result analysis timed out after {timeout_seconds}s, falling back to sequential")
                            analyses = None

                        if analyses:
                            # Process analyses and update hypotheses
                            for analysis in analyses:
                                result_id = analysis.get("result_id")
                                # Send to data analyst for full processing
                                if result_id:
                                    self._send_to_data_analyst(result_id=result_id)
                                    break  # Process one at a time in workflow
                        else:
                            # Fallback to sequential
                            result_id = results[-1]
                            self._send_to_data_analyst(result_id=result_id)

                    except Exception as e:
                        logger.error(f"Concurrent result analysis failed: {e}")
                        # Fallback to sequential
                        result_id = results[-1]
                        self._send_to_data_analyst(result_id=result_id)
                else:
                    # Sequential: analyze most recent result
                    result_id = results[-1]
                    self._send_to_data_analyst(result_id=result_id)

        elif action == NextAction.REFINE_HYPOTHESIS:
            # Refine most recently tested hypothesis
            with self._research_plan_context():
                tested = list(self.research_plan.tested_hypotheses)

            if tested:
                hypothesis_id = tested[-1]
                self._send_to_hypothesis_refiner(
                    hypothesis_id=hypothesis_id,
                    action="evaluate"
                )

            # Increment iteration after completing refinement phase
            # This marks the completion of one full research cycle
            with self._research_plan_context():
                self.research_plan.increment_iteration()
            logger.info(f"Completed iteration {self.research_plan.iteration_count}")

        elif action == NextAction.CONVERGE:
            self._send_to_convergence_detector()

            # Increment iteration before checking convergence
            with self._research_plan_context():
                self.research_plan.increment_iteration()
            logger.info(f"Checking convergence at iteration {self.research_plan.iteration_count}")

        elif action == NextAction.PAUSE:
            self.pause()

        else:
            logger.warning(f"Unknown action: {action}")

    def _should_check_convergence(self) -> bool:
        """
        Check if convergence should be evaluated.

        Returns:
            bool: True if convergence check is needed
        """
        # Check iteration limit (mandatory)
        if self.research_plan.iteration_count >= self.research_plan.max_iterations:
            logger.info("Iteration limit reached")
            return True

        # Don't converge if we haven't generated any hypotheses yet
        # Let the state machine handle generating hypotheses first
        if not self.research_plan.hypothesis_pool:
            # Only converge if we're past the hypothesis generation state
            # and hypothesis generation was attempted but failed
            if self.workflow.current_state != WorkflowState.GENERATING_HYPOTHESES:
                logger.info("No hypotheses in pool after generation attempted")
                return True
            # Otherwise, let it try to generate hypotheses first
            return False

        untested = self.research_plan.get_untested_hypotheses()
        if not untested and not self.research_plan.experiment_queue:
            logger.info("No untested hypotheses and no queued experiments")
            return True

        return False

    # ========================================================================
    # STRATEGY ADAPTATION
    # ========================================================================

    def select_next_strategy(self) -> str:
        """
        Select next strategy based on effectiveness tracking.

        Strategies with higher success rates are favored.

        Returns:
            str: Selected strategy name
        """
        # Calculate effectiveness scores
        scores = {}
        for strategy, stats in self.strategy_stats.items():
            attempts = stats["attempts"]
            if attempts == 0:
                # Favor unexplored strategies
                scores[strategy] = 1.0
            else:
                success_rate = stats["successes"] / attempts
                scores[strategy] = success_rate

        # Select strategy with highest score
        best_strategy = max(scores.items(), key=lambda x: x[1])[0]

        logger.debug(f"Selected strategy: {best_strategy} (scores: {scores})")
        return best_strategy

    def update_strategy_effectiveness(self, strategy: str, success: bool, cost: float = 0.0):
        """
        Update strategy effectiveness tracking.

        Args:
            strategy: Strategy name
            success: Whether strategy was successful
            cost: Cost incurred (API tokens, compute time, etc.)
        """
        # Thread-safe strategy stats update
        with self._strategy_stats_context():
            if strategy in self.strategy_stats:
                self.strategy_stats[strategy]["attempts"] += 1
                if success:
                    self.strategy_stats[strategy]["successes"] += 1
                self.strategy_stats[strategy]["cost"] += cost

                logger.debug(f"Updated strategy {strategy}: success={success}, cost={cost}")

    # ========================================================================
    # AGENT REGISTRY
    # ========================================================================

    def register_agent(self, agent_type: str, agent_id: str):
        """
        Register an agent for coordination.

        Args:
            agent_type: Type of agent (HypothesisGeneratorAgent, etc.)
            agent_id: Unique agent ID
        """
        self.agent_registry[agent_type] = agent_id
        logger.info(f"Registered {agent_type} with ID {agent_id}")

    def get_agent_id(self, agent_type: str) -> Optional[str]:
        """
        Get agent ID for a given type.

        Args:
            agent_type: Agent type

        Returns:
            Optional[str]: Agent ID if registered
        """
        return self.agent_registry.get(agent_type)

    # ========================================================================
    # EXECUTE (BaseAgent interface)
    # ========================================================================

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute research task.

        Args:
            task: Task specification (usually {"action": "start_research"})

        Returns:
            dict: Task result
        """
        action = task.get("action", "start_research")

        if action == "start_research":
            # Generate initial research plan
            plan = self.generate_research_plan()

            # Start the workflow
            self.start()

            # Execute first action
            next_action = self.decide_next_action()
            self._execute_next_action(next_action)

            return {
                "status": "research_started",
                "research_plan": plan,
                "next_action": next_action.value
            }

        elif action == "step":
            # Execute one step of research
            next_action = self.decide_next_action()
            self._execute_next_action(next_action)

            return {
                "status": "step_executed",
                "next_action": next_action.value,
                "workflow_state": self.workflow.current_state.value
            }

        else:
            raise ValueError(f"Unknown action: {action}")

    # ========================================================================
    # STATUS & REPORTING
    # ========================================================================

    def get_research_status(self) -> Dict[str, Any]:
        """
        Get comprehensive research status.

        Returns:
            dict: Full research status including plan, workflow, statistics
        """
        return {
            "research_question": self.research_question,
            "domain": self.domain,
            "workflow_state": self.workflow.current_state.value,
            "iteration": self.research_plan.iteration_count,
            "max_iterations": self.research_plan.max_iterations,
            "has_converged": self.research_plan.has_converged,
            "convergence_reason": self.research_plan.convergence_reason,
            "hypothesis_pool_size": len(self.research_plan.hypothesis_pool),
            "hypotheses_tested": len(self.research_plan.tested_hypotheses),
            "hypotheses_supported": len(self.research_plan.supported_hypotheses),
            "hypotheses_rejected": len(self.research_plan.rejected_hypotheses),
            "experiments_completed": len(self.research_plan.completed_experiments),
            "results_count": len(self.research_plan.results),
            "strategy_stats": self.strategy_stats,
            "agent_status": self.get_status()
        }
