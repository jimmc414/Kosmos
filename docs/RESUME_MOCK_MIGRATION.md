# Resume: Mock to Real Test Migration

## Quick Start
```
@docs/RESUME_MOCK_MIGRATION.md continue with Phase 4
```

## Context
Converting mock-based tests to real API/service calls for production readiness.

## Completed
| Phase | Tests | Status |
|-------|-------|--------|
| Phase 1: Core LLM | 43 | ✓ Complete |
| Phase 2: Knowledge Layer | 57 | ✓ Complete |
| Phase 3: Agent Tests | 128 | ✓ Complete (bugs fixed) |
| **Total** | **228** | |

## Current Task: Phase 4 - Integration Tests

### Files to Convert
1. `tests/integration/test_analysis_pipeline.py`
2. `tests/integration/test_phase2_e2e.py`
3. `tests/integration/test_phase3_e2e.py`
4. `tests/integration/test_concurrent_research.py`

### Dependencies
All Phase 4 tests require:
- Claude API (ANTHROPIC_API_KEY)
- Neo4j (Docker)
- ChromaDB
- Semantic Scholar API (rate limited: 1 req/sec)

### Expected Patterns
- Use `unique_id()` helpers for test isolation
- Valid workflow state transitions
- Context manager mocks for databases
- Rate limiting for external APIs

### Verification
```bash
# Run Phase 4 integration tests
pytest tests/integration/ -v --no-cov
```

## Reference
- Full checkpoint: `docs/CHECKPOINT_MOCK_MIGRATION.md`
