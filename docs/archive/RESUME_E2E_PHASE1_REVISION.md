# Resume Prompt: E2E Testing Phase 1 Revision

## Context

Phase 1 E2E tests were created but use mock implementations. Revise to use real system components for production readiness.

## Current State

- 7 test files created with 74 passing tests
- Some tests use `MockWorldModel` and other mock classes
- Need to replace mocks with real implementations

## Files to Revise

### 1. tests/e2e/test_world_model.py

- Remove the `MockWorldModel` class definitions (lines ~100-200)
- Replace with real `Neo4jWorldModel` from `kosmos.world_model.simple`
- Add `@requires_neo4j` decorator to all tests that need Neo4j
- The `Neo4jWorldModel` requires `NEO4J_URI` environment variable

### 2. tests/e2e/test_error_recovery.py

- `TestWorkflowContinuation` uses simulated async functions
- `TestPartialBatchFailureHandling` uses mock `BatchResponse`
- Replace with real `AsyncClaudeClient` integration where possible
- Keep `@requires_llm` decorator for tests needing API keys

### 3. tests/e2e/test_cli_workflows.py

- Remove the `test_cli_run_basic` test that patches internals
- Test real CLI commands - they should work with database
- Use CLI's built-in `--mock` flag for LLM calls

### 4. tests/e2e/conftest.py

- Remove `mock_llm_response` fixture
- Remove `mock_api_call_data` fixture
- Keep skip decorators (`requires_neo4j`, `requires_llm`, etc.)
- Add real component fixtures where needed

## Key Source Files

| Component | File Path |
|-----------|-----------|
| MetricsCollector | `kosmos/core/metrics.py` |
| World Model | `kosmos/world_model/simple.py` |
| Circuit Breaker | `kosmos/core/async_llm.py` |
| Convergence | `kosmos/core/convergence.py` |
| CLI | `kosmos/cli/main.py` |

## Reference Documentation

- Implementation plan: `docs/E2E_TESTING_IMPLEMENTATION_PLAN.md`
- Code review: `docs/E2E_TESTING_CODE_REVIEW.md`
- Full plan: `/home/jim/.claude/plans/typed-puzzling-waterfall.md`

## Goal

E2E tests should exercise real code paths. Skip tests only when external infrastructure (Neo4j, API keys, Docker) is unavailable.

## Validation

After revision, run:

```bash
pytest tests/e2e/ -v --tb=short
```

## Next Phase

After completing Phase 1 revision, continue to Phase 2:
- Create `tests/e2e/test_smoke.py` (6 tests)
- Replace 6 placeholder tests in `tests/e2e/test_full_research_workflow.py`
- Update `pytest.ini` with smoke marker
- Update `Makefile` with E2E test targets
