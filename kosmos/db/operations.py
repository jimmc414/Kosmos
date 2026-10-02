"""
CRUD operations for database models.

Provides high-level functions for creating, reading, updating, and deleting entities.

Performance Optimizations:
- Eager loading with joinedload() to prevent N+1 queries
- Strategic use of indexes (defined in Alembic migrations)
- Query result caching for frequently accessed data
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import event, func
from kosmos.db.models import (
    Experiment, Hypothesis, Result, Paper, AgentRecord, ResearchSession,
    ExperimentStatus, HypothesisStatus
)
from datetime import datetime, timezone
import logging
import time


logger = logging.getLogger(__name__)


def _validate_json_dict(value: Any, field_name: str, required: bool = True) -> None:
    """Validate that a value is a proper dict for JSON column storage."""
    if value is None:
        if required:
            raise ValueError(f"{field_name} is required and cannot be None")
        return
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a dict, got {type(value).__name__}")


def _validate_json_list(value: Any, field_name: str, required: bool = True) -> None:
    """Validate that a value is a proper list for JSON column storage."""
    if value is None:
        if required:
            raise ValueError(f"{field_name} is required and cannot be None")
        return
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list, got {type(value).__name__}")


# ============================================================================
# QUERY PERFORMANCE MONITORING
# ============================================================================

def log_slow_queries(session_factory, threshold_ms: float = 100.0):
    """
    Log slow database queries for performance monitoring.

    Args:
        session_factory: SQLAlchemy session factory
        threshold_ms: Threshold in milliseconds for slow query logging

    Example:
        >>> from kosmos.db import _engine
        >>> log_slow_queries(_engine, threshold_ms=50.0)
    """
    @event.listens_for(session_factory, "after_cursor_execute")
    def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        """Log queries that exceed threshold."""
        if context.executemany:
            return

        duration_ms = (time.time() - context._query_start_time) * 1000

        if duration_ms > threshold_ms:
            logger.warning(
                f"Slow query ({duration_ms:.2f}ms): {statement[:200]}"
                + ("..." if len(statement) > 200 else "")
            )

    @event.listens_for(session_factory, "before_cursor_execute")
    def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        """Record query start time."""
        context._query_start_time = time.time()


# ============================================================================
# HYPOTHESIS CRUD
# ============================================================================

def create_hypothesis(
    session: Session,
    id: str,
    research_question: str,
    statement: str,
    rationale: str,
    domain: str,
    novelty_score: Optional[float] = None,
    testability_score: Optional[float] = None,
    confidence_score: Optional[float] = None,
    related_papers: Optional[List[str]] = None,
) -> Hypothesis:
    """Create a new hypothesis."""
    _validate_json_list(related_papers, "related_papers", required=False)
    hypothesis = Hypothesis(
        id=id,
        research_question=research_question,
        statement=statement,
        rationale=rationale,
        domain=domain,
        novelty_score=novelty_score,
        testability_score=testability_score,
        confidence_score=confidence_score,
        related_papers=related_papers or [],
    )
    session.add(hypothesis)
    session.commit()
    session.refresh(hypothesis)

    logger.info(f"Created hypothesis {id}")
    return hypothesis


def get_hypothesis(session: Session, hypothesis_id: str, with_experiments: bool = False) -> Optional[Hypothesis]:
    """
    Get hypothesis by ID.

    Args:
        session: Database session
        hypothesis_id: Hypothesis ID
        with_experiments: If True, eager load associated experiments

    Returns:
        Hypothesis or None if not found
    """
    query = session.query(Hypothesis).filter(Hypothesis.id == hypothesis_id)

    if with_experiments:
        query = query.options(joinedload(Hypothesis.experiments))

    return query.first()


def list_hypotheses(
    session: Session,
    domain: Optional[str] = None,
    status: Optional[HypothesisStatus] = None,
    limit: int = 100,
    with_experiments: bool = False
) -> List[Hypothesis]:
    """
    List hypotheses with optional filtering.

    Performance: Uses indexes on domain and status columns for fast filtering.

    Args:
        session: Database session
        domain: Filter by domain
        status: Filter by status
        limit: Maximum results to return
        with_experiments: If True, eager load associated experiments to prevent N+1 queries

    Returns:
        List of hypotheses
    """
    query = session.query(Hypothesis)

    if domain:
        query = query.filter(Hypothesis.domain == domain)
    if status:
        query = query.filter(Hypothesis.status == status)

    if with_experiments:
        query = query.options(joinedload(Hypothesis.experiments))

    return query.order_by(Hypothesis.created_at.desc()).limit(limit).all()


def update_hypothesis_status(
    session: Session,
    hypothesis_id: str,
    status: HypothesisStatus
) -> Hypothesis:
    """Update hypothesis status."""
    hypothesis = get_hypothesis(session, hypothesis_id)
    if not hypothesis:
        raise ValueError(f"Hypothesis {hypothesis_id} not found")

    hypothesis.status = status
    hypothesis.updated_at = datetime.now(timezone.utc)
    session.commit()
    session.refresh(hypothesis)

    logger.info(f"Updated hypothesis {hypothesis_id} status to {status}")
    return hypothesis


# ============================================================================
# EXPERIMENT CRUD
# ============================================================================

def create_experiment(
    session: Session,
    id: str,
    hypothesis_id: str,
    experiment_type: str,
    description: str,
    protocol: Dict[str, Any],
    domain: str,
    code_generated: Optional[str] = None,
) -> Experiment:
    """Create a new experiment."""
    _validate_json_dict(protocol, "protocol", required=True)
    experiment = Experiment(
        id=id,
        hypothesis_id=hypothesis_id,
        experiment_type=experiment_type,
        description=description,
        protocol=protocol,
        domain=domain,
        code_generated=code_generated,
    )
    session.add(experiment)
    session.commit()
    session.refresh(experiment)

    logger.info(f"Created experiment {id}")
    return experiment


def get_experiment(
    session: Session,
    experiment_id: str,
    with_hypothesis: bool = True,
    with_results: bool = False
) -> Optional[Experiment]:
    """
    Get experiment by ID with optional eager loading.

    Args:
        session: Database session
        experiment_id: Experiment ID
        with_hypothesis: If True, eager load associated hypothesis (recommended)
        with_results: If True, eager load associated results

    Returns:
        Experiment or None if not found

    Performance:
        - Uses primary key index for O(1) lookup
        - Eager loading prevents N+1 queries when accessing relationships
    """
    query = session.query(Experiment).filter(Experiment.id == experiment_id)

    if with_hypothesis:
        query = query.options(joinedload(Experiment.hypothesis))
    if with_results:
        query = query.options(joinedload(Experiment.results))

    return query.first()


def list_experiments(
    session: Session,
    hypothesis_id: Optional[str] = None,
    status: Optional[ExperimentStatus] = None,
    domain: Optional[str] = None,
    limit: int = 100,
    with_hypothesis: bool = False,
    with_results: bool = False
) -> List[Experiment]:
    """
    List experiments with optional filtering and eager loading.

    Performance:
        - Uses composite index (domain, status) for fast filtering
        - Eager loading prevents N+1 queries
        - Created_at DESC uses idx_experiments_created_at

    Args:
        session: Database session
        hypothesis_id: Filter by hypothesis ID
        status: Filter by status
        domain: Filter by domain
        limit: Maximum results to return
        with_hypothesis: If True, eager load associated hypotheses
        with_results: If True, eager load associated results

    Returns:
        List of experiments
    """
    query = session.query(Experiment)

    if hypothesis_id:
        query = query.filter(Experiment.hypothesis_id == hypothesis_id)
    if status:
        query = query.filter(Experiment.status == status)
    if domain:
        query = query.filter(Experiment.domain == domain)

    if with_hypothesis:
        query = query.options(joinedload(Experiment.hypothesis))
    if with_results:
        query = query.options(joinedload(Experiment.results))

    return query.order_by(Experiment.created_at.desc()).limit(limit).all()


def update_experiment_status(
    session: Session,
    experiment_id: str,
    status: ExperimentStatus,
    error_message: Optional[str] = None,
    execution_time_seconds: Optional[float] = None,
) -> Experiment:
    """Update experiment status."""
    experiment = get_experiment(session, experiment_id)
    if not experiment:
        raise ValueError(f"Experiment {experiment_id} not found")

    experiment.status = status
    if status == ExperimentStatus.RUNNING and not experiment.started_at:
        experiment.started_at = datetime.now(timezone.utc)
    elif status in [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED]:
        experiment.completed_at = datetime.now(timezone.utc)

    if error_message:
        experiment.error_message = error_message
    if execution_time_seconds:
        experiment.execution_time_seconds = execution_time_seconds

    session.commit()
    session.refresh(experiment)

    logger.info(f"Updated experiment {experiment_id} status to {status}")
    return experiment


# ============================================================================
# RESULT CRUD
# ============================================================================

def create_result(
    session: Session,
    id: str,
    experiment_id: str,
    data: Dict[str, Any],
    statistical_tests: Optional[Dict[str, Any]] = None,
    interpretation: Optional[str] = None,
    key_findings: Optional[List[str]] = None,
    supports_hypothesis: Optional[bool] = None,
    p_value: Optional[float] = None,
    effect_size: Optional[float] = None,
) -> Result:
    """Create a new result."""
    _validate_json_dict(data, "data", required=True)
    _validate_json_dict(statistical_tests, "statistical_tests", required=False)
    _validate_json_list(key_findings, "key_findings", required=False)
    result = Result(
        id=id,
        experiment_id=experiment_id,
        data=data,
        statistical_tests=statistical_tests,
        interpretation=interpretation,
        key_findings=key_findings,
        supports_hypothesis=supports_hypothesis,
        p_value=p_value,
        effect_size=effect_size,
    )
    session.add(result)
    session.commit()
    session.refresh(result)

    logger.info(f"Created result {id} for experiment {experiment_id}")
    return result


def get_result(session: Session, result_id: str, with_experiment: bool = False) -> Optional[Result]:
    """
    Get result by ID.

    Args:
        session: Database session
        result_id: Result ID
        with_experiment: If True, eager load associated experiment

    Returns:
        Result or None if not found
    """
    query = session.query(Result).filter(Result.id == result_id)

    if with_experiment:
        query = query.options(joinedload(Result.experiment))

    return query.first()


def update_result_analysis(
    session: Session,
    result_id: str,
    supports_hypothesis: Optional[bool] = None,
    interpretation: Optional[str] = None,
    key_findings: Optional[List[str]] = None,
) -> Result:
    """Persist the analysis verdict for a result.

    supports_hypothesis is always written, so None records an inconclusive verdict.
    """
    result = get_result(session, result_id)
    if not result:
        raise ValueError(f"Result {result_id} not found")

    result.supports_hypothesis = supports_hypothesis
    if interpretation is not None:
        result.interpretation = interpretation
    if key_findings is not None:
        _validate_json_list(key_findings, "key_findings", required=False)
        result.key_findings = key_findings

    session.commit()
    session.refresh(result)

    logger.info(f"Updated result {result_id} analysis: supports_hypothesis={supports_hypothesis}")
    return result


def get_results_for_experiment(
    session: Session,
    experiment_id: str,
    with_experiment: bool = False
) -> List[Result]:
    """
    Get all results for an experiment.

    Performance: Uses idx_results_experiment_id for fast filtering.

    Args:
        session: Database session
        experiment_id: Experiment ID
        with_experiment: If True, eager load associated experiment (usually not needed)

    Returns:
        List of results for the experiment
    """
    query = session.query(Result).filter(Result.experiment_id == experiment_id)

    if with_experiment:
        query = query.options(joinedload(Result.experiment))

    return query.all()


# ============================================================================
# PAPER CRUD
# ============================================================================

def create_paper(
    session: Session,
    id: str,
    title: str,
    authors: List[str],
    abstract: str,
    source: str,
    url: Optional[str] = None,
    doi: Optional[str] = None,
    arxiv_id: Optional[str] = None,
    publication_date: Optional[datetime] = None,
    domain: Optional[str] = None,
) -> Paper:
    """Create a new paper."""
    _validate_json_list(authors, "authors", required=True)
    paper = Paper(
        id=id,
        title=title,
        authors=authors,
        abstract=abstract,
        source=source,
        url=url,
        doi=doi,
        arxiv_id=arxiv_id,
        publication_date=publication_date,
        domain=domain,
    )
    session.add(paper)
    session.commit()
    session.refresh(paper)

    logger.info(f"Created paper {id}")
    return paper


def get_paper(session: Session, paper_id: str) -> Optional[Paper]:
    """Get paper by ID."""
    return session.query(Paper).filter(Paper.id == paper_id).first()


def search_papers(
    session: Session,
    domain: Optional[str] = None,
    min_relevance: Optional[float] = None,
    limit: int = 100
) -> List[Paper]:
    """Search papers with filtering."""
    query = session.query(Paper)

    if domain:
        query = query.filter(Paper.domain == domain)
    if min_relevance:
        query = query.filter(Paper.relevance_score >= min_relevance)

    return query.order_by(Paper.created_at.desc()).limit(limit).all()


# ============================================================================
# AGENT CRUD
# ============================================================================

def create_agent_record(
    session: Session,
    id: str,
    agent_type: str,
    status: str,
    config: Optional[Dict[str, Any]] = None,
) -> AgentRecord:
    """Create agent record."""
    _validate_json_dict(config, "config", required=False)
    agent = AgentRecord(
        id=id,
        agent_type=agent_type,
        status=status,
        config=config,
    )
    session.add(agent)
    session.commit()
    session.refresh(agent)

    logger.info(f"Created agent record {id}")
    return agent


def update_agent_record(
    session: Session,
    agent_id: str,
    status: Optional[str] = None,
    state_data: Optional[Dict[str, Any]] = None,
) -> AgentRecord:
    """Update agent record."""
    agent = session.query(AgentRecord).filter(AgentRecord.id == agent_id).first()
    if not agent:
        raise ValueError(f"Agent {agent_id} not found")

    if status:
        agent.status = status
        if status == "stopped":
            agent.stopped_at = datetime.now(timezone.utc)

    if state_data is not None:
        _validate_json_dict(state_data, "state_data", required=True)
        agent.state_data = state_data

    agent.updated_at = datetime.now(timezone.utc)
    session.commit()
    session.refresh(agent)

    return agent


# ============================================================================
# RESEARCH SESSION CRUD
# ============================================================================

def create_research_session(
    session: Session,
    id: str,
    research_question: str,
    domain: str,
    max_iterations: int = 10,
    autonomous_mode: bool = True,
) -> ResearchSession:
    """Create research session."""
    research_session = ResearchSession(
        id=id,
        research_question=research_question,
        domain=domain,
        max_iterations=max_iterations,
        autonomous_mode=autonomous_mode,
    )
    session.add(research_session)
    session.commit()
    session.refresh(research_session)

    logger.info(f"Created research session {id}")
    return research_session


def get_research_session(
    session: Session,
    session_id: str
) -> Optional[ResearchSession]:
    """Get research session by ID."""
    return session.query(ResearchSession).filter(ResearchSession.id == session_id).first()


def update_research_session(
    db_session: Session,
    session_id: str,
    status: Optional[str] = None,
    iteration: Optional[int] = None,
    hypotheses_generated: Optional[int] = None,
    experiments_completed: Optional[int] = None,
) -> ResearchSession:
    """Update research session."""
    research_session = get_research_session(db_session, session_id)
    if not research_session:
        raise ValueError(f"Research session {session_id} not found")

    if status:
        research_session.status = status
        if status == "completed":
            research_session.completed_at = datetime.now(timezone.utc)

    if iteration is not None:
        research_session.iteration = iteration
    if hypotheses_generated is not None:
        research_session.hypotheses_generated = hypotheses_generated
    if experiments_completed is not None:
        research_session.experiments_completed = experiments_completed

    research_session.updated_at = datetime.now(timezone.utc)
    db_session.commit()
    db_session.refresh(research_session)

    return research_session
