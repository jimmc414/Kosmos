# Mock to Real Test Migration - Checkpoint

## Date: 2025-12-07

## Overall Progress
- **Phase 1**: Core LLM Tests - 43 tests ✓
- **Phase 2**: Knowledge Layer Tests - 57 tests ✓
- **Phase 3**: Agent Tests - 128 tests ✓ (bugs fixed)
- **Phase 4**: Integration Tests - Pending

**Total Converted: 228 tests**

---

## Phase 3 Complete: Agent Tests (Bugs Fixed)

### Summary
Converted 4 agent test files from mock-based to real Claude API calls. All 128 tests pass.

| File | Tests | Service | Notes |
|------|-------|---------|-------|
| `tests/unit/agents/test_data_analyst.py` | 24 | Claude Haiku | Pure Python logic + LLM |
| `tests/unit/agents/test_research_director.py` | 36 | Claude Haiku | Workflow + planning |
| `tests/unit/agents/test_hypothesis_generator.py` | 19 | Claude Haiku | Generation + DB mocks |
| `tests/unit/agents/test_literature_analyzer.py` | 10 | Claude Haiku | All tests pass |
| `tests/unit/agents/test_skill_loader.py` | 39 | None (pure Python) | Already passing |

### Bugs Fixed

#### Bug 1: `generate_structured` max_tokens parameter ✓
- **File:** `kosmos/core/llm.py:409-417`
- **Fix:** Added `max_tokens`, `temperature`, and `max_retries` parameters to `ClaudeClient.generate_structured()`
- **Enhancement:** Added retry logic for flaky JSON generation (Haiku sometimes truncates responses)

#### Bug 2: JSON reliability improvements ✓
- **Fix:** Added `temperature=0.3` default for structured output (more deterministic)
- **Fix:** Added `max_retries=2` with cache bypass on retries
- **Fix:** Added explicit JSON completion instructions in system prompt

### Key Patterns Used
- `unique_id()` helper for test isolation
- Valid workflow state transitions for ResearchDirector tests
- Context manager mock pattern (`__enter__`/`__exit__`) for database mocks
- Legacy ClaudeClient for tests to avoid provider interface mismatch

---

## Phase 2 Complete: Knowledge Layer Tests

### Summary
Converted 4 knowledge layer test files from mock-based to real services. All 57 tests pass.

| File | Tests | Service |
|------|-------|---------|
| `tests/unit/knowledge/test_embeddings.py` | 13 | SentenceTransformer (SPECTER + MiniLM) |
| `tests/unit/knowledge/test_concept_extractor.py` | 11 | Anthropic Haiku |
| `tests/unit/knowledge/test_vector_db.py` | 16 | ChromaDB + SPECTER embeddings |
| `tests/unit/knowledge/test_graph.py` | 17 | Neo4j |

### Key Patterns Used
- `unique_id()` helper to generate test-specific IDs for isolation
- `unique_paper` fixtures with random suffixes to avoid cache/collision
- Correct method names discovered: `create_paper`, `create_concept`, `create_citation`, etc.
- ChromaDB paper IDs use format: `{source.value}:{primary_identifier}`

### VRAM Note
SPECTER model (~440MB) loads on GPU. Running all knowledge tests together may cause CUDA OOM with 6GB VRAM. Run in batches:
```bash
pytest tests/unit/knowledge/test_embeddings.py tests/unit/knowledge/test_concept_extractor.py -v --no-cov
pytest tests/unit/knowledge/test_vector_db.py tests/unit/knowledge/test_graph.py -v --no-cov
```

---

## Phase 1 Complete: Core LLM Tests

### Summary
Converted 3 core test files from mock-based to real API calls. All 43 tests pass.

| File | Tests | Provider |
|------|-------|----------|
| `tests/unit/core/test_llm.py` | 17 | Anthropic Haiku |
| `tests/unit/core/test_async_llm.py` | 13 | Anthropic Haiku |
| `tests/unit/core/test_litellm_provider.py` | 13 | Anthropic + DeepSeek |

### Key Patterns Established
```python
import os, pytest, uuid

pytestmark = [
    pytest.mark.requires_claude,
    pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Requires API key")
]

def unique_prompt(base: str) -> str:
    return f"{base} [test-id: {uuid.uuid4().hex[:8]}]"
```

---

## Infrastructure Status
- Docker: Running
- Neo4j: Running (kosmos-neo4j, healthy)
- ChromaDB: v1.3.4
- ANTHROPIC_API_KEY: Configured
- DEEPSEEK_API_KEY: Configured
- SEMANTIC_SCHOLAR_API_KEY: Configured (1 req/sec rate limit)

---

## Next Phase: Phase 4 - Integration Tests (4 files)
| File | Dependencies |
|------|--------------|
| `tests/integration/test_analysis_pipeline.py` | All services |
| `tests/integration/test_phase2_e2e.py` | All services |
| `tests/integration/test_phase3_e2e.py` | All services |
| `tests/integration/test_concurrent_research.py` | All services |

---

## Verification Commands

```bash
# Phase 1 - Core LLM (43 tests)
pytest tests/unit/core/test_llm.py tests/unit/core/test_async_llm.py tests/unit/core/test_litellm_provider.py -v --no-cov

# Phase 2 - Knowledge Layer (57 tests) - run in batches for VRAM
pytest tests/unit/knowledge/test_embeddings.py tests/unit/knowledge/test_concept_extractor.py -v --no-cov
pytest tests/unit/knowledge/test_vector_db.py tests/unit/knowledge/test_graph.py -v --no-cov

# Phase 3 - Agent Tests (128 tests)
pytest tests/unit/agents/ -v --no-cov
```

## Commits
- `199c931` - Convert autonomous research tests to use real LLM API calls
- `7ebd56b` - Convert Phase 2 knowledge layer tests from mocks to real services
