# Resume Prompt: E2E Testing Phase 3

## Context

Phase 2 E2E test infrastructure is complete. All smoke tests and placeholder replacements done. Ready to proceed with Phase 3.

## Goal: Production-Ready E2E Tests

**Critical requirement**: All E2E tests must use real implementations, NOT mocks.

- Do NOT use `Mock()`, `MagicMock()`, `patch()`, or inline mock classes
- Use real components: `InMemoryWorldModel`, `MetricsCollector`, `ConvergenceDetector`, etc.
- Use skip decorators (`@requires_neo4j`, `@requires_llm`) when external infrastructure is needed
- Tests should exercise actual code paths to catch real integration issues

## Phase 2 Completed ✓

- Created `tests/e2e/test_smoke.py` with 6 smoke tests (all passing)
- Replaced placeholders in `tests/e2e/test_full_research_workflow.py`:
  - TestPerformanceValidation: 3 real tests (parallel speedup, cache hit rate, API cost tracking)
  - TestCLIWorkflows: 2 real tests (CLI commands, doctor status monitoring)
  - TestDockerDeployment: 1 real test (service health checks)
- Updated `pytest.ini` with `smoke` marker
- Updated `Makefile` with `test-e2e`, `test-smoke`, `test-e2e-quick` targets (all with `--no-cov`)
- **97+ tests in E2E suite, smoke tests passing**

## Phase 3 Tasks

### 1. Review and Enhance World Model Tests

Review `tests/e2e/test_world_model.py` for any remaining mock usage or test gaps:
- Verify all tests use `in_memory_world_model` fixture
- Add tests for edge cases (empty graph, concurrent operations)
- Ensure Neo4j integration tests work when available

### 2. Review Error Recovery Tests

Review `tests/e2e/test_error_recovery.py`:
- Verify circuit breaker tests use real `CircuitBreaker` class
- Verify rate limiter tests use real `RateLimiter` class
- Add any missing error scenario tests

### 3. Review System Sanity Tests

Review `tests/e2e/test_system_sanity.py`:
- Verify all component tests use real implementations
- Remove any mock dependencies
- Add integration tests for component interactions

### 4. Create Test Documentation

Create `tests/e2e/README.md` documenting:
- Test categories (smoke, e2e, slow)
- How to run tests with different markers
- Skip decorators and their requirements
- Test fixtures available in conftest.py

## Key Source Files

| Component | File Path |
|-----------|-----------|
| InMemoryWorldModel | `kosmos/world_model/in_memory.py` |
| World Model Factory | `kosmos/world_model/factory.py` |
| MetricsCollector | `kosmos/core/metrics.py` |
| CircuitBreaker | `kosmos/core/async_llm.py` |
| RateLimiter | `kosmos/core/async_llm.py` |
| E2E conftest | `tests/e2e/conftest.py` |
| Factories | `tests/e2e/factories.py` |

## Environment

Neo4j is configured and running:
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=kosmos-password
```

## Validation

After Phase 3, run:
```bash
# Run smoke tests
make test-smoke

# Run quick E2E tests (excludes slow)
make test-e2e-quick

# Run all E2E tests
make test-e2e

# Verify no mocks in E2E tests
grep -r "Mock\|MagicMock\|patch" tests/e2e/ --include="*.py"
```

## Reference Documentation (Archived)

- Phase 1 resume: `docs/archive/RESUME_E2E_PHASE1_REVISION.md`
- Phase 2 resume: `docs/archive/RESUME_E2E_PHASE2.md`
- Implementation plan: `docs/archive/E2E_TESTING_IMPLEMENTATION_PLAN.md`
- Code review: `docs/archive/E2E_TESTING_CODE_REVIEW.md`
