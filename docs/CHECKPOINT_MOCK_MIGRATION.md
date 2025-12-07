# Mock to Real Test Migration - Checkpoint

## Date: 2025-12-07

## Overall Progress
- **Phase 1**: Core LLM Tests - 43 tests ✓
- **Phase 2**: Knowledge Layer Tests - 57 tests ✓
- **Phase 3**: Agent Tests - Pending
- **Phase 4**: Integration Tests - Pending

**Total Converted: 100 tests**

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

## Remaining Phases

### Phase 3: Agent Tests (4 files)
| File | Dependencies |
|------|--------------|
| `tests/unit/agents/test_research_director.py` | Claude API |
| `tests/unit/agents/test_literature_analyzer.py` | Claude API + Neo4j |
| `tests/unit/agents/test_data_analyst.py` | Claude API |
| `tests/unit/agents/test_hypothesis_generator.py` | Claude + Semantic Scholar |

### Phase 4: Integration Tests (4 files)
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
```

## Commits
- `199c931` - Convert autonomous research tests to use real LLM API calls
- `7ebd56b` - Convert Phase 2 knowledge layer tests from mocks to real services
