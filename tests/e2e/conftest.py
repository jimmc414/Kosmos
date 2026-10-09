"""
E2E Test Infrastructure.

Provides unified fixtures, skip decorators, and test utilities
for end-to-end testing.
"""

import os
import subprocess
import pytest
from typing import Optional
from datetime import datetime
from pathlib import Path
# dotenv.main: tests/conftest.py stubs dotenv.load_dotenv for the hermetic suites
from dotenv.main import load_dotenv

# Load .env file before checking environment variables
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=True)


# =============================================================================
# ENVIRONMENT DETECTION
# =============================================================================

def _check_docker_available() -> bool:
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return False


def _check_neo4j_available() -> bool:
    """Check if Neo4j is configured."""
    return bool(os.getenv("NEO4J_URI"))


# Environment flags
HAS_ANTHROPIC_KEY = bool(os.getenv("ANTHROPIC_API_KEY"))
HAS_OPENAI_KEY = bool(os.getenv("OPENAI_API_KEY"))
HAS_ANY_LLM_KEY = HAS_ANTHROPIC_KEY or HAS_OPENAI_KEY
HAS_NEO4J = _check_neo4j_available()
HAS_DOCKER = _check_docker_available()


# =============================================================================
# SKIP DECORATORS
# =============================================================================

requires_llm = pytest.mark.skipif(
    not HAS_ANY_LLM_KEY,
    reason="Requires ANTHROPIC_API_KEY or OPENAI_API_KEY"
)

requires_anthropic = pytest.mark.skipif(
    not HAS_ANTHROPIC_KEY,
    reason="Requires ANTHROPIC_API_KEY"
)

requires_openai = pytest.mark.skipif(
    not HAS_OPENAI_KEY,
    reason="Requires OPENAI_API_KEY"
)

requires_neo4j = pytest.mark.skipif(
    not HAS_NEO4J,
    reason="Requires NEO4J_URI environment variable"
)

requires_docker = pytest.mark.skipif(
    not HAS_DOCKER,
    reason="Requires Docker to be running"
)

requires_full_stack = pytest.mark.skipif(
    not (HAS_ANY_LLM_KEY and HAS_NEO4J and HAS_DOCKER),
    reason="Requires LLM API key, Neo4j, and Docker"
)


# =============================================================================
# CORE FIXTURES
# =============================================================================

@pytest.fixture
def metrics_collector():
    """Fresh MetricsCollector instance for budget testing."""
    from kosmos.core.metrics import MetricsCollector
    collector = MetricsCollector()
    yield collector
    # Cleanup
    collector.reset()


@pytest.fixture
def configured_metrics_collector():
    """MetricsCollector with budget configured."""
    from kosmos.core.metrics import MetricsCollector, BudgetPeriod

    collector = MetricsCollector()
    collector.configure_budget(
        limit_usd=1.00,
        period=BudgetPeriod.DAILY,
        alert_thresholds=[50.0, 75.0, 90.0, 100.0]
    )
    yield collector
    collector.reset()


@pytest.fixture
def circuit_breaker():
    """Circuit breaker with test-friendly timeouts."""
    from kosmos.core.async_llm import CircuitBreaker

    return CircuitBreaker(
        failure_threshold=3,
        reset_timeout=1.0,  # 1 second for testing
        half_open_max_calls=1
    )


@pytest.fixture
def rate_limiter():
    """Rate limiter with test-friendly limits."""
    from kosmos.core.async_llm import RateLimiter

    return RateLimiter(
        max_requests_per_minute=60,
        max_concurrent=5
    )


@pytest.fixture
def in_memory_world_model():
    """In-memory world model for testing without Neo4j."""
    from kosmos.world_model.in_memory import InMemoryWorldModel

    wm = InMemoryWorldModel()
    yield wm
    wm.reset()


@pytest.fixture
def cli_runner():
    """Typer CLI test runner."""
    from typer.testing import CliRunner
    return CliRunner()


@pytest.fixture
def convergence_detector():
    """ConvergenceDetector with test configuration."""
    from kosmos.core.convergence import ConvergenceDetector

    return ConvergenceDetector(
        mandatory_criteria=["iteration_limit", "no_testable_hypotheses"],
        optional_criteria=["novelty_decline", "diminishing_returns"],
        config={
            "novelty_decline_threshold": 0.3,
            "novelty_decline_window": 3,
            "cost_per_discovery_threshold": 100.0
        }
    )


@pytest.fixture
def research_plan():
    """Basic ResearchPlan for testing."""
    from kosmos.core.workflow import ResearchPlan

    return ResearchPlan(
        research_question="Test research question",
        max_iterations=5
    )


@pytest.fixture
def e2e_artifacts_dir(tmp_path):
    """Directory for E2E test artifacts."""
    artifacts = tmp_path / "e2e_artifacts"
    artifacts.mkdir(exist_ok=True)
    return artifacts


# =============================================================================
# ALERT TRACKING FIXTURES
# =============================================================================

@pytest.fixture
def alert_tracker():
    """Tracks budget alerts for testing."""
    class AlertTracker:
        def __init__(self):
            self.alerts = []

        def callback(self, alert):
            self.alerts.append(alert)

        def get_thresholds(self):
            return [a.threshold_percent for a in self.alerts]

        def clear(self):
            self.alerts = []

    return AlertTracker()


# =============================================================================
# ASYNC FIXTURES
# =============================================================================

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# WORLD MODEL FIXTURES
# =============================================================================

@pytest.fixture
def sample_entity():
    """Sample Entity for testing."""
    from kosmos.world_model.models import Entity

    return Entity(
        type="Hypothesis",
        properties={
            "statement": "Test hypothesis statement",
            "domain": "biology",
            "research_question": "Does X affect Y?"
        },
        confidence=0.8,
        created_by="test"
    )


@pytest.fixture
def sample_relationship():
    """Sample Relationship for testing."""
    from kosmos.world_model.models import Relationship

    return Relationship(
        source_id="entity-1",
        target_id="entity-2",
        type="TESTS",
        properties={"iteration": 1},
        confidence=0.9,
        created_by="test"
    )


# =============================================================================
# HYPOTHESIS & EXPERIMENT FIXTURES
# =============================================================================

@pytest.fixture
def sample_hypothesis():
    """Sample Hypothesis for testing."""
    from kosmos.models.hypothesis import Hypothesis, HypothesisStatus

    return Hypothesis(
        id="hyp_test_001",
        statement="Test hypothesis statement",
        rationale="Test rationale for the hypothesis",
        domain="biology",
        research_question="Does X affect Y?",
        status=HypothesisStatus.GENERATED,
        testability_score=0.8,
        novelty_score=0.6,
        confidence_score=0.7
    )


@pytest.fixture
def sample_experiment_result():
    """Sample ExperimentResult for testing."""
    import platform
    import sys
    from kosmos.models.result import ExperimentResult, ResultStatus, ExecutionMetadata

    now = datetime.utcnow()

    metadata = ExecutionMetadata(
        start_time=now,
        end_time=now,
        duration_seconds=1.0,
        python_version=sys.version,
        platform=platform.system(),
        experiment_id="exp_test_001",
        protocol_id="proto_test_001",
        hypothesis_id="hyp_test_001"
    )

    return ExperimentResult(
        id="result_test_001",
        experiment_id="exp_test_001",
        protocol_id="proto_test_001",
        hypothesis_id="hyp_test_001",
        status=ResultStatus.SUCCESS,
        supports_hypothesis=True,
        primary_p_value=0.01,
        metadata=metadata
    )


# =============================================================================
# CLEANUP FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def reset_singletons_e2e():
    """Reset singleton instances for each E2E test."""
    yield

    # Reset metrics singleton
    try:
        from kosmos.core.metrics import get_metrics
        metrics = get_metrics()
        metrics.reset()
    except ImportError:
        pass

    # Reset world model singleton
    try:
        from kosmos.world_model.factory import reset_world_model
        reset_world_model()
    except ImportError:
        pass
