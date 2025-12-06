# Resume Prompt: E2E Testing Phase 2

## Context

Phase 1 E2E test revision is complete. All mock implementations have been replaced with real components. Ready to proceed with Phase 2.

## Goal: Production-Ready E2E Tests

**Critical requirement**: All E2E tests must use real implementations, NOT mocks.

- Do NOT use `Mock()`, `MagicMock()`, `patch()`, or inline mock classes
- Use real components: `InMemoryWorldModel`, `MetricsCollector`, `ConvergenceDetector`, etc.
- Use skip decorators (`@requires_neo4j`, `@requires_llm`) when external infrastructure is needed
- Tests should exercise actual code paths to catch real integration issues

## Phase 1 Completed ✓

- Created `kosmos/world_model/in_memory.py` - `InMemoryWorldModel` class
- Removed 4 `MockWorldModel` classes from `tests/e2e/test_world_model.py`
- Removed `mock_llm_response` and `mock_api_call_data` fixtures from `tests/e2e/conftest.py`
- Configured Neo4j in `.env` (bolt://localhost:7687, neo4j/kosmos-password)
- **76 tests passing, 0 skipped**

## Phase 2 Tasks

### 1. Create `tests/e2e/test_smoke.py` (6 tests)

Fast sanity checks that verify core system functionality:

```python
# Suggested smoke tests:
1. test_config_loads - Config loads without errors
2. test_database_connection - SQLite DB initializes
3. test_neo4j_connection - Neo4j connects (skip if unavailable)
4. test_metrics_collector_initializes - MetricsCollector works
5. test_world_model_factory - get_world_model() returns valid instance
6. test_cli_help - CLI --help works
```

### 2. Replace placeholders in `tests/e2e/test_full_research_workflow.py`

Current file has placeholder tests. Replace with real workflow tests:
- Full research cycle execution
- Multi-iteration research
- Result persistence
- Convergence detection in workflow

### 3. Update `pytest.ini`

Add smoke marker:
```ini
markers =
    smoke: Quick sanity checks (< 10 seconds total)
    # ... existing markers
```

### 4. Update `Makefile`

Add E2E test targets:
```makefile
test-e2e:
    pytest tests/e2e/ -v

test-smoke:
    pytest tests/e2e/ -m smoke -v

test-e2e-quick:
    pytest tests/e2e/ -m "not slow" -v
```

## Key Source Files

| Component | File Path |
|-----------|-----------|
| InMemoryWorldModel | `kosmos/world_model/in_memory.py` |
| World Model Factory | `kosmos/world_model/factory.py` |
| Config | `kosmos/config.py` |
| CLI | `kosmos/cli/main.py` |
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

After Phase 2, run:
```bash
# Run smoke tests
pytest tests/e2e/ -m smoke -v

# Run all E2E tests
pytest tests/e2e/ -v --tb=short

# Verify test count increased
pytest tests/e2e/ --collect-only | grep "test session starts" -A 5
```

## Reference Documentation (Archived)

- Phase 1 resume: `docs/archive/RESUME_E2E_PHASE1_REVISION.md`
- Implementation plan: `docs/archive/E2E_TESTING_IMPLEMENTATION_PLAN.md`
- Code review: `docs/archive/E2E_TESTING_CODE_REVIEW.md`
