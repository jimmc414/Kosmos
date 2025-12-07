# Resume: Mock to Real Test Migration

## Quick Start
```
@docs/RESUME_MOCK_MIGRATION.md continue with Phase 3
```

## Context
Converting mock-based tests to real API/service calls for production readiness.

## Completed
| Phase | Tests | Status |
|-------|-------|--------|
| Phase 1: Core LLM | 43 | ✓ Complete |
| Phase 2: Knowledge Layer | 57 | ✓ Complete |
| **Total** | **100** | |

## Current Task: Phase 3 - Agent Tests

### Files to Convert (4 files)
1. `tests/unit/agents/test_research_director.py` - Claude API
2. `tests/unit/agents/test_literature_analyzer.py` - Claude API + Neo4j
3. `tests/unit/agents/test_data_analyst.py` - Claude API
4. `tests/unit/agents/test_hypothesis_generator.py` - Claude + Semantic Scholar

### Infrastructure Ready
- ANTHROPIC_API_KEY: ✓ Configured
- SEMANTIC_SCHOLAR_API_KEY: ✓ Configured (1 req/sec)
- Neo4j: ✓ Running (kosmos-neo4j)
- ChromaDB: ✓ Available

### Conversion Pattern
```python
import os, pytest, uuid

pytestmark = [
    pytest.mark.requires_claude,
    pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Requires API key")
]

def unique_prompt(base: str) -> str:
    return f"{base} [test-id: {uuid.uuid4().hex[:8]}]"
```

### Key Learnings from Phase 2
- Use `unique_id()` / `unique_prompt()` helpers to avoid cache hits
- Check actual method signatures (e.g., `create_paper` not `add_paper`)
- ChromaDB IDs: `{source.value}:{primary_identifier}`
- Neo4j node properties use `id` key, stats use `paper_count` not `total_papers`
- Run tests in batches to avoid CUDA OOM (6GB VRAM limit)

## After Phase 3: Phase 4 - Integration Tests
1. `tests/integration/test_analysis_pipeline.py`
2. `tests/integration/test_phase2_e2e.py`
3. `tests/integration/test_phase3_e2e.py`
4. `tests/integration/test_concurrent_research.py`

## Verify Previous Work
```bash
# Phase 1 (43 tests)
pytest tests/unit/core/test_llm.py tests/unit/core/test_async_llm.py tests/unit/core/test_litellm_provider.py -v --no-cov

# Phase 2 (57 tests) - run in batches
pytest tests/unit/knowledge/test_embeddings.py tests/unit/knowledge/test_concept_extractor.py -v --no-cov
pytest tests/unit/knowledge/test_vector_db.py tests/unit/knowledge/test_graph.py -v --no-cov
```

## Reference
- Full checkpoint: `docs/CHECKPOINT_MOCK_MIGRATION.md`
- Migration plan: `/home/jim/.claude/plans/sprightly-finding-puddle.md`
