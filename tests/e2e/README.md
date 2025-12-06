# E2E Tests

End-to-end tests that validate complete system functionality using real implementations.

## Running Tests

```bash
# Run all E2E tests
make test-e2e

# Run smoke tests only (fast)
make test-smoke

# Run quick tests (excludes @slow)
make test-e2e-quick
```

## Test Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.e2e` | All E2E tests |
| `@pytest.mark.smoke` | Fast sanity checks (<10s total) |
| `@pytest.mark.slow` | Long-running tests |

## Skip Decorators

| Decorator | Skips When |
|-----------|------------|
| `@requires_neo4j` | NEO4J_URI not set |
| `@requires_llm` | No API key available |
| `@requires_docker` | Docker not running |
| `@requires_full_stack` | Missing any external service |

## Key Fixtures (conftest.py)

| Fixture | Description |
|---------|-------------|
| `in_memory_world_model` | InMemoryWorldModel instance |
| `circuit_breaker` | CircuitBreaker with test timeouts |
| `rate_limiter` | RateLimiter with test limits |
| `metrics_collector` | MetricsCollector instance |
| `cli_runner` | Typer CliRunner for CLI tests |

## Test Categories

1. **Smoke Tests** (`test_smoke.py`) - Core component validation
2. **Budget Tests** (`test_budget_enforcement.py`) - Cost tracking/limits
3. **CLI Tests** (`test_cli_workflows.py`) - Command-line interface
4. **Convergence Tests** (`test_convergence.py`) - Research termination
5. **Error Recovery** (`test_error_recovery.py`) - Circuit breaker, retry logic
6. **World Model** (`test_world_model.py`) - Entity/relationship persistence
7. **System Sanity** (`test_system_sanity.py`) - Component integration

## Important: No Mocks

All E2E tests use real implementations. Do NOT add:
- `Mock()`, `MagicMock()`
- `patch()` decorators
- Inline mock classes

Use skip decorators when external services are needed.
