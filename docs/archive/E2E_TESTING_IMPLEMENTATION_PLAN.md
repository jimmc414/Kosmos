# E2E Testing Implementation Plan: Production Readiness

## Overview

This document provides a comprehensive implementation plan to bring Kosmos E2E testing from its current state (45/100 coverage score) to full production readiness. Based on the code review analysis in `E2E_TESTING_CODE_REVIEW.md` and deep codebase exploration, this plan addresses all identified gaps systematically.

**Target Coverage Score: 90/100**

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Phase 1: Critical Gap Resolution (Week 1-2)](#phase-1-critical-gap-resolution)
3. [Phase 2: Infrastructure & Coverage (Week 3-4)](#phase-2-infrastructure--coverage)
4. [Phase 3: Advanced Scenarios (Week 5-6)](#phase-3-advanced-scenarios)
5. [Phase 4: CI/CD Integration (Week 7-8)](#phase-4-cicd-integration)
6. [Implementation Details](#implementation-details)
7. [Test File Structure](#test-file-structure)
8. [Fixture Requirements](#fixture-requirements)
9. [CI/CD Configuration](#cicd-configuration)
10. [Success Criteria](#success-criteria)

---

## Executive Summary

### Current State
- **E2E Test Count**: 39 tests across 3 files
- **Coverage Score**: 45/100
- **Critical Gaps**: Budget enforcement, World Model, Error Recovery, CLI workflows
- **Quality Issues**: Placeholder tests, inconsistent skip conditions, heavy mocking

### Target State
- **E2E Test Count**: 100+ tests across 10+ files
- **Coverage Score**: 90/100
- **All Critical Paths Tested**: Real component integration, error handling, edge cases
- **Quality**: No placeholders, consistent markers, balanced mock/real testing

### Key Deliverables
1. 5 new E2E test files addressing critical gaps
2. Standardized test infrastructure with factories and fixtures
3. CI/CD pipeline with staged test execution
4. Smoke test suite for quick validation
5. Comprehensive error recovery test coverage

---

## Phase 1: Critical Gap Resolution

**Duration**: Week 1-2
**Priority**: HIGH
**Focus**: Address the top 5 missing tests identified in the code review

### 1.1 Budget Enforcement E2E Test

**File**: `tests/e2e/test_budget_enforcement.py`

**Rationale**: Budget enforcement is critical for production use to prevent runaway costs. Currently no E2E tests verify this functionality.

**Test Cases**:

| Test | Description | Components Tested |
|------|-------------|-------------------|
| `test_budget_enforcement_halts_research` | Verify research halts when budget exceeded | MetricsCollector, ResearchWorkflow |
| `test_budget_threshold_alerts` | Verify alerts at 50%, 75%, 90%, 100% | MetricsCollector, Alert callbacks |
| `test_budget_period_reset_hourly` | Verify hourly budget resets correctly | BudgetPeriod, period tracking |
| `test_budget_period_reset_daily` | Verify daily budget resets correctly | BudgetPeriod, period tracking |
| `test_cost_calculation_accuracy` | Verify Claude pricing ($3/$15 per M tokens) | Token counting, cost calculation |
| `test_budget_enforcement_graceful_shutdown` | Verify in-progress work completes before halt | Workflow state management |
| `test_budget_concurrent_tracking` | Verify thread-safe cost accumulation | Concurrent API calls, metrics |

**Implementation**:

```python
# Test outline for test_budget_enforcement_halts_research
class TestBudgetEnforcement:
    @pytest.fixture
    def mock_llm_with_costs(self):
        """Mock LLM that reports realistic token costs."""
        mock = Mock()
        mock.generate.return_value = Mock(
            content="Response",
            usage=Mock(input_tokens=1000, output_tokens=500)
        )
        return mock

    def test_budget_enforcement_halts_research(self, mock_llm_with_costs):
        """Test research halts gracefully when budget exceeded."""
        # 1. Setup: Configure $0.10 budget limit
        metrics = MetricsCollector()
        metrics.configure_budget(
            max_cost_usd=0.10,
            max_requests_per_period=1000,
            period=BudgetPeriod.DAILY,
            alert_thresholds=[0.5, 0.75, 0.9, 1.0]
        )

        # 2. Execute: Run research that exceeds budget
        workflow = ResearchWorkflow(
            research_objective="Test budget limits",
            metrics_collector=metrics
        )
        result = await workflow.run(num_cycles=10)

        # 3. Assert: Workflow stopped with budget_exceeded reason
        assert result['stopped_reason'] == 'budget_exceeded'
        assert result['cycles_completed'] < 10

        # 4. Assert: No API calls after budget exceeded
        final_stats = metrics.get_api_statistics()
        assert final_stats['total_cost_usd'] <= 0.10 * 1.05  # 5% tolerance
```

### 1.2 World Model Persistence E2E Test

**File**: `tests/e2e/test_world_model.py`

**Rationale**: World model is the knowledge persistence layer. Without E2E tests, we can't verify entities persist correctly across research cycles.

**Test Cases**:

| Test | Description | Components Tested |
|------|-------------|-------------------|
| `test_entity_persistence_across_cycles` | Entities from cycle 1 available in cycle 2 | Neo4jWorldModel, Entity |
| `test_relationship_creation` | SPAWNED_BY, TESTS, SUPPORTS relationships | Relationship, provenance |
| `test_hypothesis_to_entity_flow` | Hypothesis → Entity conversion correct | Entity.from_hypothesis() |
| `test_result_to_entity_flow` | Result → Entity conversion correct | Entity.from_result() |
| `test_world_model_export_import` | Full graph export/import roundtrip | export_graph(), import_graph() |
| `test_world_model_statistics` | Statistics accurate after operations | get_statistics() |
| `test_multi_project_isolation` | Projects don't leak data | project_id isolation |

**Implementation**:

```python
class TestWorldModelPersistence:
    @pytest.fixture
    def in_memory_world_model(self):
        """Create in-memory world model for testing."""
        return InMemoryWorldModel()

    def test_entity_persistence_across_cycles(self, in_memory_world_model):
        """Verify entities persist correctly across research cycles."""
        wm = in_memory_world_model

        # Cycle 1: Create hypothesis entity
        hypothesis = Hypothesis(
            id="hyp_001",
            statement="Test hypothesis",
            domain="biology",
            research_question="Does X affect Y?"
        )
        entity = Entity.from_hypothesis(hypothesis)
        wm.add_entity(entity)

        # Cycle 2: Create experiment testing hypothesis
        experiment = ExperimentProtocol(
            id="exp_001",
            hypothesis_id="hyp_001",
            name="Test experiment"
        )
        exp_entity = Entity.from_protocol(experiment)
        wm.add_entity(exp_entity)

        # Create TESTS relationship
        relationship = Relationship(
            source_id=exp_entity.id,
            target_id=entity.id,
            relationship_type="TESTS",
            properties={"iteration": 1}
        )
        wm.add_relationship(relationship)

        # Verify: Query related entities
        related = wm.query_related_entities(entity.id, "TESTS", direction="incoming")
        assert len(related) == 1
        assert related[0].id == exp_entity.id

        # Verify: Statistics
        stats = wm.get_statistics()
        assert stats['entity_count'] >= 2
        assert stats['relationship_count'] >= 1
```

### 1.3 Error Recovery E2E Test

**File**: `tests/e2e/test_error_recovery.py`

**Rationale**: Production systems must handle failures gracefully. Current tests don't verify error recovery, circuit breaker states, or retry logic.

**Test Cases**:

| Test | Description | Components Tested |
|------|-------------|-------------------|
| `test_circuit_breaker_opens_after_failures` | CB opens after 3 consecutive failures | CircuitBreaker |
| `test_circuit_breaker_blocks_when_open` | Requests fail-fast when CB open | CircuitBreaker |
| `test_circuit_breaker_recovers` | CB transitions to HALF_OPEN after timeout | CircuitBreaker |
| `test_retry_on_recoverable_errors` | Rate limit/timeout errors retry | tenacity, is_recoverable_error() |
| `test_no_retry_on_permanent_errors` | Auth/invalid errors fail immediately | is_recoverable_error() |
| `test_workflow_continues_after_agent_failure` | Workflow continues despite agent errors | ResearchWorkflow |
| `test_task_retry_increments` | Failed tasks track retry count | DelegationManager |
| `test_partial_batch_failure_handling` | Batch continues despite some failures | AsyncClaudeClient |

**Implementation**:

```python
class TestErrorRecovery:
    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker with test-friendly timeouts."""
        return CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=5,  # 5 seconds for testing
            half_open_success_threshold=1
        )

    def test_circuit_breaker_state_transitions(self, circuit_breaker):
        """Test complete circuit breaker lifecycle."""
        cb = circuit_breaker

        # Initial state: CLOSED
        assert cb.get_state() == CircuitState.CLOSED
        assert cb.can_execute() is True

        # Record 3 failures → OPEN
        for _ in range(3):
            cb.record_failure()
        assert cb.get_state() == CircuitState.OPEN
        assert cb.can_execute() is False

        # Wait for recovery timeout
        time.sleep(6)

        # Should be HALF_OPEN
        assert cb.get_state() == CircuitState.HALF_OPEN
        assert cb.can_execute() is True

        # Record success → CLOSED
        cb.record_success()
        assert cb.get_state() == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_workflow_continues_after_agent_failure(self):
        """Verify workflow continues despite individual agent failures."""
        # Setup: Create workflow with failing agent on 2nd call
        call_count = [0]

        async def failing_hypothesis_generator(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Simulated agent failure")
            return Mock(hypotheses=[Mock(id="hyp_1")])

        workflow = ResearchWorkflow(
            research_objective="Test error recovery",
            hypothesis_generator=Mock(generate=failing_hypothesis_generator)
        )

        # Execute: Run 3 cycles
        result = await workflow.run(num_cycles=3)

        # Assert: At least 2 cycles completed (1 failed)
        assert result['cycles_completed'] >= 2
        assert 'errors' in result
```

### 1.4 Full Research Convergence E2E Test

**File**: `tests/e2e/test_convergence.py`

**Rationale**: The full research cycle to convergence is the core product feature. Current tests don't verify convergence detection, hypothesis refinement, or knowledge accumulation.

**Test Cases**:

| Test | Description | Components Tested |
|------|-------------|-------------------|
| `test_research_reaches_convergence` | Research converges within iteration limit | ConvergenceDetector |
| `test_convergence_report_complete` | Report has all required sections | Report generation |
| `test_hypothesis_refinement_across_iterations` | Hypotheses refined based on results | HypothesisGenerator |
| `test_knowledge_accumulates` | Each iteration builds on previous | State persistence |
| `test_early_convergence_detection` | Stops early if converged | ConvergenceDetector |
| `test_max_iterations_reached` | Graceful stop at max iterations | Workflow limits |

**Implementation**:

```python
class TestResearchConvergence:
    @pytest.mark.asyncio
    async def test_full_research_cycle_to_convergence(self):
        """Test complete research cycle reaches convergence."""
        workflow = ResearchWorkflow(
            research_objective="Test simple hypothesis: 2+2=4",
            max_iterations=5
        )

        # Execute until convergence
        result = await workflow.run_until_convergence()

        # Verify convergence
        assert result['converged'] is True or result['iterations'] == 5

        # Verify report
        report = await workflow.generate_convergence_report()
        assert '# Research Summary' in report
        assert 'Hypotheses Tested' in report
        assert 'Conclusions' in report

        # Verify knowledge accumulated
        stats = workflow.state_manager.get_statistics()
        assert stats['total_findings'] > 0
        assert stats['total_hypotheses'] > 0
```

### 1.5 CLI Run Command E2E Test

**File**: `tests/e2e/test_cli_workflows.py`

**Rationale**: CLI is the primary user interface. Current CLI tests are placeholders (`assert True`).

**Test Cases**:

| Test | Description | Components Tested |
|------|-------------|-------------------|
| `test_cli_run_complete_workflow` | Full research via CLI | kosmos run |
| `test_cli_status_during_research` | Status command shows progress | kosmos status |
| `test_cli_history_after_completion` | History shows past runs | kosmos history |
| `test_cli_verbose_flag` | Verbose output works | --verbose flag |
| `test_cli_debug_flag` | Debug output works | --debug flag |
| `test_cli_mock_mode` | Mock mode works | --mock flag |
| `test_cli_doctor_diagnostics` | Doctor command validates setup | kosmos doctor |

**Implementation**:

```python
class TestCLIWorkflows:
    @pytest.fixture
    def cli_runner(self):
        """Create CLI test runner."""
        from typer.testing import CliRunner
        return CliRunner()

    def test_cli_run_complete_workflow(self, cli_runner):
        """Test CLI can execute complete research workflow."""
        from kosmos.cli.main import app

        result = cli_runner.invoke(app, [
            "run",
            "Does temperature affect enzyme activity?",
            "--max-iterations", "2",
            "--mock"  # Use mock mode for testing
        ])

        # Verify exit code
        assert result.exit_code == 0

        # Verify output contains progress
        assert "Starting research" in result.stdout or "research" in result.stdout.lower()

        # Verify completion message
        assert "completed" in result.stdout.lower() or "finished" in result.stdout.lower()

    def test_cli_doctor_diagnostics(self, cli_runner):
        """Test doctor command validates system setup."""
        from kosmos.cli.main import app

        result = cli_runner.invoke(app, ["doctor"])

        # Should report on key components
        assert "Python" in result.stdout
        assert "Database" in result.stdout or "database" in result.stdout.lower()
```

---

## Phase 2: Infrastructure & Coverage

**Duration**: Week 3-4
**Priority**: MEDIUM
**Focus**: Standardize infrastructure, remove placeholders, add markers

### 2.1 Test Infrastructure Improvements

#### 2.1.1 Standardize Skip Conditions

**Current Problem**: Inconsistent use of `@pytest.mark.skipif` vs custom markers.

**Solution**: Create unified decorator in `tests/e2e/conftest.py`:

```python
# tests/e2e/conftest.py

import os
import pytest

# Environment checks
HAS_ANTHROPIC_KEY = bool(os.getenv("ANTHROPIC_API_KEY"))
HAS_OPENAI_KEY = bool(os.getenv("OPENAI_API_KEY"))
HAS_ANY_LLM_KEY = HAS_ANTHROPIC_KEY or HAS_OPENAI_KEY
HAS_NEO4J = bool(os.getenv("NEO4J_URI"))
HAS_DOCKER = _check_docker_available()

def _check_docker_available():
    """Check if Docker is available."""
    try:
        import subprocess
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False

# Unified skip decorators
requires_llm = pytest.mark.skipif(
    not HAS_ANY_LLM_KEY,
    reason="Requires ANTHROPIC_API_KEY or OPENAI_API_KEY"
)

requires_neo4j = pytest.mark.skipif(
    not HAS_NEO4J,
    reason="Requires NEO4J_URI environment variable"
)

requires_docker = pytest.mark.skipif(
    not HAS_DOCKER,
    reason="Requires Docker to be running"
)

# Composite decorators
requires_full_stack = pytest.mark.skipif(
    not (HAS_ANY_LLM_KEY and HAS_NEO4J and HAS_DOCKER),
    reason="Requires LLM API key, Neo4j, and Docker"
)
```

#### 2.1.2 Add Test Factories

**File**: `tests/e2e/factories.py`

```python
"""Test factories for creating complex test scenarios."""

from dataclasses import dataclass
from typing import List, Optional
import uuid

from kosmos.models.hypothesis import Hypothesis, ExperimentType
from kosmos.models.experiment import ExperimentProtocol, ProtocolStep
from kosmos.models.result import ExperimentResult, ResultStatus

@dataclass
class ResearchScenarioFactory:
    """Factory for creating complete research scenarios."""

    @staticmethod
    def create_simple_hypothesis(
        domain: str = "biology",
        research_question: str = "Does X affect Y?"
    ) -> Hypothesis:
        """Create a simple testable hypothesis."""
        return Hypothesis(
            id=f"hyp_{uuid.uuid4().hex[:8]}",
            statement=f"X increases Y in {domain}",
            research_question=research_question,
            domain=domain,
            rationale="Based on prior research",
            testability_score=0.8,
            novelty_score=0.6
        )

    @staticmethod
    def create_experiment_protocol(
        hypothesis: Hypothesis,
        num_steps: int = 3
    ) -> ExperimentProtocol:
        """Create an experiment protocol for a hypothesis."""
        steps = [
            ProtocolStep(
                step_number=i,
                title=f"Step {i}",
                description=f"Execute step {i}",
                action=f"action_{i}()"
            )
            for i in range(1, num_steps + 1)
        ]

        return ExperimentProtocol(
            id=f"proto_{uuid.uuid4().hex[:8]}",
            hypothesis_id=hypothesis.id,
            name=f"Test protocol for {hypothesis.id}",
            description="Test protocol",
            domain=hypothesis.domain,
            experiment_type=ExperimentType.COMPUTATIONAL,
            steps=steps
        )

    @staticmethod
    def create_successful_result(
        protocol: ExperimentProtocol,
        p_value: float = 0.01
    ) -> ExperimentResult:
        """Create a successful experiment result."""
        return ExperimentResult(
            id=f"result_{uuid.uuid4().hex[:8]}",
            experiment_id=protocol.id,
            protocol_id=protocol.id,
            hypothesis_id=protocol.hypothesis_id,
            status=ResultStatus.SUCCESS,
            supports_hypothesis=p_value < 0.05,
            primary_p_value=p_value,
            statistics={"p_value": p_value, "effect_size": 0.8}
        )
```

#### 2.1.3 Add Smoke Test Marker

**Update**: `pytest.ini`

```ini
markers =
    # ... existing markers ...
    smoke: Quick smoke tests for CI validation (< 2 minutes total)
```

**File**: `tests/e2e/test_smoke.py`

```python
"""Smoke tests for quick CI validation."""

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.smoke]

class TestSmoke:
    """Quick smoke tests that run in < 2 minutes total."""

    def test_imports(self):
        """Verify all major modules import correctly."""
        from kosmos.workflow.research_loop import ResearchWorkflow
        from kosmos.agents.research_director import ResearchDirectorAgent
        from kosmos.core.metrics import MetricsCollector
        from kosmos.world_model.factory import get_world_model
        from kosmos.execution.executor import CodeExecutor
        assert True

    def test_database_connection(self):
        """Verify database can be initialized."""
        from kosmos.db import init_database, get_session
        import tempfile

        with tempfile.NamedTemporaryFile(suffix='.db') as tmp:
            init_database(f"sqlite:///{tmp.name}")
            with get_session() as session:
                assert session is not None

    def test_code_executor_basic(self):
        """Verify code executor can run simple code."""
        from kosmos.execution.executor import CodeExecutor

        executor = CodeExecutor(use_sandbox=False)
        result = executor.execute("print('hello')")

        assert result.success
        assert "hello" in result.stdout

    def test_safety_validator_blocks_dangerous(self):
        """Verify safety validator blocks dangerous code."""
        from kosmos.safety.code_validator import CodeValidator

        validator = CodeValidator()
        result = validator.validate("import os; os.system('rm -rf /')")

        assert result.passed is False
```

### 2.2 Remove Placeholder Tests

**Action**: Replace all `assert True` placeholders with real implementations.

**Affected Tests**:

| File | Test | Replacement |
|------|------|-------------|
| `test_full_research_workflow.py` | `test_parallel_vs_sequential_speedup` | Benchmark parallel vs sequential execution |
| `test_full_research_workflow.py` | `test_cache_hit_rate` | Verify cache statistics accuracy |
| `test_full_research_workflow.py` | `test_api_cost_reduction` | Measure cost savings from caching |
| `test_full_research_workflow.py` | `test_cli_run_and_view_results` | Full CLI test implementation |
| `test_full_research_workflow.py` | `test_cli_status_monitoring` | Status command test |
| `test_full_research_workflow.py` | `test_service_health_checks` | Docker health check validation |

### 2.3 Add Missing Component Tests

#### 2.3.1 Context Compression E2E

**File**: `tests/e2e/test_context_compression.py`

| Test | Description |
|------|-------------|
| `test_large_context_compression` | Verify 100k+ token contexts compress correctly |
| `test_compression_preserves_key_info` | Important details retained after compression |
| `test_notebook_compression` | Jupyter notebooks compress correctly |
| `test_literature_compression` | Paper summaries compress correctly |

#### 2.3.2 Knowledge Graph Integration E2E

**File**: `tests/e2e/test_knowledge_graph.py`

| Test | Description |
|------|-------------|
| `test_paper_citation_graph` | Citation relationships created correctly |
| `test_concept_extraction_storage` | Concepts stored in graph |
| `test_graph_query_performance` | Queries complete within timeout |
| `test_graph_export_large` | 10k+ node export works |

---

## Phase 3: Advanced Scenarios

**Duration**: Week 5-6
**Priority**: MEDIUM-LOW
**Focus**: Concurrent operations, real LLM tests, performance benchmarks

### 3.1 Concurrent Operation Tests

**File**: `tests/e2e/test_concurrent_operations.py`

| Test | Description |
|------|-------------|
| `test_async_claude_batch_processing` | 10+ concurrent LLM requests |
| `test_parallel_experiment_execution` | Multiple experiments in parallel |
| `test_rate_limiter_enforcement` | Verify 50 req/min limit enforced |
| `test_concurrent_metric_recording` | Thread-safe metrics accumulation |
| `test_concurrent_world_model_updates` | Thread-safe entity updates |

### 3.2 Real LLM Integration Tests

**File**: `tests/e2e/test_real_llm.py`

**Note**: These tests require API keys and incur costs. Run manually or in dedicated CI job.

```python
@pytest.mark.requires_api_key
@pytest.mark.slow
@pytest.mark.real_llm
class TestRealLLMIntegration:
    """Tests using real LLM API calls."""

    def test_hypothesis_generation_real(self):
        """Test hypothesis generation with real Claude API."""
        # Implementation
        pass

    def test_experiment_design_real(self):
        """Test experiment design with real Claude API."""
        pass

    def test_result_analysis_real(self):
        """Test result analysis with real Claude API."""
        pass
```

### 3.3 Performance Benchmark Tests

**File**: `tests/e2e/test_performance.py`

| Test | Description | Target |
|------|-------------|--------|
| `test_workflow_startup_time` | Time to initialize workflow | < 5s |
| `test_hypothesis_generation_latency` | Time for single hypothesis | < 10s |
| `test_parallel_speedup_ratio` | 3 parallel vs sequential | >= 2x speedup |
| `test_cache_response_time` | Cached response time | < 100ms |
| `test_large_batch_throughput` | 100 requests throughput | > 10 req/s |

---

## Phase 4: CI/CD Integration

**Duration**: Week 7-8
**Priority**: HIGH
**Focus**: Automated testing pipeline

### 4.1 GitHub Actions Workflow

**File**: `.github/workflows/e2e-tests.yml`

```yaml
name: E2E Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  workflow_dispatch:
    inputs:
      run_real_llm:
        description: 'Run real LLM tests (costs $)'
        required: false
        default: 'false'

env:
  PYTHON_VERSION: '3.11'

jobs:
  smoke:
    name: Smoke Tests
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - name: Install dependencies
        run: |
          pip install -e ".[test]"
      - name: Run smoke tests
        run: |
          pytest tests/e2e/ -m smoke -v --tb=short

  e2e-mock:
    name: E2E Tests (Mocked)
    runs-on: ubuntu-latest
    timeout-minutes: 30
    needs: smoke
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_USER: kosmos
          POSTGRES_PASSWORD: kosmos_test
          POSTGRES_DB: kosmos_test
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - name: Install dependencies
        run: pip install -e ".[test]"
      - name: Run E2E tests (mocked)
        env:
          DATABASE_URL: postgresql://kosmos:kosmos_test@localhost:5432/kosmos_test
          MOCK_LLM: 'true'
        run: |
          pytest tests/e2e/ -m "e2e and not requires_api_key and not requires_neo4j" \
            -v --tb=short --timeout=300

  e2e-neo4j:
    name: E2E Tests (Neo4j)
    runs-on: ubuntu-latest
    timeout-minutes: 30
    needs: smoke
    services:
      neo4j:
        image: neo4j:5
        env:
          NEO4J_AUTH: neo4j/kosmos_test
        ports:
          - 7687:7687
          - 7474:7474
        options: >-
          --health-cmd "curl -f http://localhost:7474 || exit 1"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 10
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - name: Install dependencies
        run: pip install -e ".[test]"
      - name: Run E2E tests (Neo4j)
        env:
          NEO4J_URI: bolt://localhost:7687
          NEO4J_USER: neo4j
          NEO4J_PASSWORD: kosmos_test
        run: |
          pytest tests/e2e/ -m "requires_neo4j" -v --tb=short

  e2e-real-llm:
    name: E2E Tests (Real LLM)
    runs-on: ubuntu-latest
    timeout-minutes: 60
    needs: [e2e-mock, e2e-neo4j]
    if: github.event.inputs.run_real_llm == 'true' || github.event_name == 'schedule'
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - name: Install dependencies
        run: pip install -e ".[test]"
      - name: Run E2E tests (Real LLM)
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          pytest tests/e2e/ -m "requires_api_key and not slow" \
            -v --tb=short --timeout=600
```

### 4.2 Makefile Updates

Add new targets to `Makefile`:

```makefile
#==============================================================================
# E2E Testing
#==============================================================================

test-e2e:
	@echo "🧪 Running E2E tests..."
	@pytest tests/e2e/ -v --tb=short

test-e2e-mock:
	@echo "🧪 Running E2E tests (mocked mode)..."
	@pytest tests/e2e/ -m "e2e and not requires_api_key" -v

test-e2e-full:
	@echo "🧪 Running full E2E tests (requires API keys)..."
	@pytest tests/e2e/ -m "e2e" -v

test-smoke:
	@echo "💨 Running smoke tests..."
	@pytest tests/e2e/ -m "smoke" -v --timeout=120

test-e2e-cov:
	@echo "🧪 Running E2E tests with coverage..."
	@pytest tests/e2e/ --cov=kosmos --cov-report=html:htmlcov/e2e -v
```

---

## Implementation Details

### New Test Files to Create

| File | Tests | Priority |
|------|-------|----------|
| `tests/e2e/test_budget_enforcement.py` | 7 | HIGH |
| `tests/e2e/test_world_model.py` | 7 | HIGH |
| `tests/e2e/test_error_recovery.py` | 8 | HIGH |
| `tests/e2e/test_convergence.py` | 6 | HIGH |
| `tests/e2e/test_cli_workflows.py` | 7 | HIGH |
| `tests/e2e/test_smoke.py` | 6 | MEDIUM |
| `tests/e2e/test_context_compression.py` | 4 | MEDIUM |
| `tests/e2e/test_knowledge_graph.py` | 4 | MEDIUM |
| `tests/e2e/test_concurrent_operations.py` | 5 | MEDIUM |
| `tests/e2e/test_performance.py` | 5 | LOW |
| `tests/e2e/test_real_llm.py` | 5 | LOW |
| `tests/e2e/conftest.py` | - | HIGH |
| `tests/e2e/factories.py` | - | HIGH |

**Total New Tests**: ~64 tests

### Files to Modify

| File | Changes |
|------|---------|
| `tests/e2e/test_full_research_workflow.py` | Remove placeholder tests, add real implementations |
| `tests/conftest.py` | Add E2E-specific fixtures |
| `pytest.ini` | Add smoke marker |
| `Makefile` | Add E2E test targets |

---

## Test File Structure

```
tests/e2e/
├── __init__.py
├── conftest.py                    # E2E-specific fixtures and skip decorators
├── factories.py                   # Test data factories
│
├── # Critical Gap Tests (Phase 1)
├── test_budget_enforcement.py     # Budget and cost tracking
├── test_world_model.py            # World model persistence
├── test_error_recovery.py         # Error handling and recovery
├── test_convergence.py            # Research convergence
├── test_cli_workflows.py          # CLI command tests
│
├── # Infrastructure Tests (Phase 2)
├── test_smoke.py                  # Quick validation tests
├── test_context_compression.py    # Context compression
├── test_knowledge_graph.py        # Neo4j integration
│
├── # Existing Tests (Modified)
├── test_system_sanity.py          # Component sanity tests
├── test_autonomous_research.py    # Autonomous workflow tests
├── test_full_research_workflow.py # Full workflow tests (placeholders removed)
│
├── # Advanced Tests (Phase 3)
├── test_concurrent_operations.py  # Parallel/async operations
├── test_performance.py            # Performance benchmarks
└── test_real_llm.py               # Real LLM integration (optional)
```

---

## Fixture Requirements

### New Fixtures for `tests/e2e/conftest.py`

```python
@pytest.fixture
def metrics_collector():
    """Fresh MetricsCollector for budget testing."""
    return MetricsCollector()

@pytest.fixture
def in_memory_world_model():
    """In-memory world model for testing without Neo4j."""
    return InMemoryWorldModel()

@pytest.fixture
def circuit_breaker():
    """Circuit breaker with test-friendly timeouts."""
    return CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=5
    )

@pytest.fixture
def rate_limiter():
    """Rate limiter with test-friendly limits."""
    return RateLimiter(
        max_concurrent=3,
        max_per_minute=60
    )

@pytest.fixture
def mock_workflow():
    """Pre-configured workflow with all mocks."""
    return ResearchWorkflow(
        research_objective="Test workflow",
        anthropic_client=None,  # Mock mode
        max_cycles=3
    )

@pytest.fixture
def cli_runner():
    """Typer CLI test runner."""
    from typer.testing import CliRunner
    return CliRunner()

@pytest.fixture
def research_scenario_factory():
    """Factory for creating research test scenarios."""
    return ResearchScenarioFactory()
```

---

## CI/CD Configuration

### Environment Variables Required

| Variable | Required For | Secret? |
|----------|--------------|---------|
| `DATABASE_URL` | All tests | No |
| `NEO4J_URI` | Neo4j tests | No |
| `NEO4J_USER` | Neo4j tests | No |
| `NEO4J_PASSWORD` | Neo4j tests | Yes |
| `ANTHROPIC_API_KEY` | Real LLM tests | Yes |
| `OPENAI_API_KEY` | Alternative LLM tests | Yes |
| `MOCK_LLM` | Mock mode flag | No |

### Test Stages

| Stage | Duration | Trigger | Tests Run |
|-------|----------|---------|-----------|
| Smoke | ~2 min | Every PR | `@pytest.mark.smoke` |
| E2E Mock | ~10 min | Every PR | `@pytest.mark.e2e` without API |
| E2E Neo4j | ~10 min | Every PR | `@pytest.mark.requires_neo4j` |
| E2E Full | ~60 min | Manual/Scheduled | All E2E with API keys |

---

## Success Criteria

### Phase 1 Completion
- [ ] All 5 critical gap test files created
- [ ] 35+ new tests passing
- [ ] No placeholder tests in critical paths
- [ ] Coverage score improved to 65/100

### Phase 2 Completion
- [ ] Standardized skip decorators across all E2E tests
- [ ] Test factories created and documented
- [ ] Smoke test suite running in < 2 minutes
- [ ] All placeholder tests replaced
- [ ] Coverage score improved to 75/100

### Phase 3 Completion
- [ ] Concurrent operation tests passing
- [ ] Performance benchmarks established
- [ ] Real LLM tests working (optional CI job)
- [ ] Coverage score improved to 85/100

### Phase 4 Completion
- [ ] GitHub Actions workflow configured
- [ ] All CI stages passing
- [ ] Makefile targets working
- [ ] Documentation updated
- [ ] Coverage score achieved 90/100

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| API costs during testing | Use mock mode by default; real LLM tests opt-in |
| Neo4j availability | Provide in-memory fallback for most tests |
| Docker dependency | Skip Docker tests when not available |
| Flaky async tests | Add retries and increase timeouts |
| Long test runtime | Parallel test execution with pytest-xdist |

---

## Appendix: Test Implementation Checklist

### Phase 1 Tests

- [ ] `test_budget_enforcement_halts_research`
- [ ] `test_budget_threshold_alerts`
- [ ] `test_budget_period_reset_hourly`
- [ ] `test_budget_period_reset_daily`
- [ ] `test_cost_calculation_accuracy`
- [ ] `test_budget_enforcement_graceful_shutdown`
- [ ] `test_budget_concurrent_tracking`
- [ ] `test_entity_persistence_across_cycles`
- [ ] `test_relationship_creation`
- [ ] `test_hypothesis_to_entity_flow`
- [ ] `test_result_to_entity_flow`
- [ ] `test_world_model_export_import`
- [ ] `test_world_model_statistics`
- [ ] `test_multi_project_isolation`
- [ ] `test_circuit_breaker_opens_after_failures`
- [ ] `test_circuit_breaker_blocks_when_open`
- [ ] `test_circuit_breaker_recovers`
- [ ] `test_retry_on_recoverable_errors`
- [ ] `test_no_retry_on_permanent_errors`
- [ ] `test_workflow_continues_after_agent_failure`
- [ ] `test_task_retry_increments`
- [ ] `test_partial_batch_failure_handling`
- [ ] `test_research_reaches_convergence`
- [ ] `test_convergence_report_complete`
- [ ] `test_hypothesis_refinement_across_iterations`
- [ ] `test_knowledge_accumulates`
- [ ] `test_early_convergence_detection`
- [ ] `test_max_iterations_reached`
- [ ] `test_cli_run_complete_workflow`
- [ ] `test_cli_status_during_research`
- [ ] `test_cli_history_after_completion`
- [ ] `test_cli_verbose_flag`
- [ ] `test_cli_debug_flag`
- [ ] `test_cli_mock_mode`
- [ ] `test_cli_doctor_diagnostics`

---

*Document Version: 1.0*
*Created: 2025-12-05*
*Based on: E2E_TESTING_CODE_REVIEW.md*
