# Resume Prompt - Post Compaction

## Context

You are resuming work on the Kosmos project after a context compaction. The previous sessions implemented **10 paper implementation gaps** (3 BLOCKER + 5 Critical + 2 High).

## What Was Done

### All Fixed Issues

| Issue | Description | Implementation |
|-------|-------------|----------------|
| #66 | CLI Deadlock | Full async refactor of message passing |
| #67 | SkillLoader | Domain-to-bundle mapping fixed |
| #68 | Pydantic V2 | Model config migration complete |
| #54 | Self-Correcting Code Execution | Enhanced RetryStrategy with 11 error handlers + LLM repair |
| #55 | World Model Update Categories | UpdateType enum (CONFIRMATION/CONFLICT/PRUNING) + conflict detection |
| #56 | 12-Hour Runtime Constraint | `max_runtime_hours` config + runtime tracking in ResearchDirector |
| #57 | Parallel Task Execution | Changed `max_concurrent_experiments` default from 4 to 10 |
| #58 | Agent Rollout Tracking | New RolloutTracker class + integration in ResearchDirector |
| #59 | h5ad/Parquet Data Formats | `DataLoader.load_h5ad()` and `load_parquet()` methods |
| #69 | R Language Execution | `RExecutor` class + Docker image with TwoSampleMR |

### Key Files Created/Modified (This Session)

| File | Changes |
|------|---------|
| `kosmos/execution/data_analysis.py` | Added `load_h5ad()`, `load_parquet()` |
| `kosmos/execution/r_executor.py` | **NEW** - R execution engine |
| `kosmos/execution/executor.py` | Added R integration, `execute_r()` |
| `docker/sandbox/Dockerfile.r` | **NEW** - R-enabled Docker image |
| `pyproject.toml` | Added pyarrow to science dependencies |
| `tests/unit/execution/test_r_executor.py` | **NEW** - 36 R executor tests |
| `tests/integration/test_data_formats.py` | **NEW** - 14 data format tests |
| `tests/integration/test_r_execution.py` | **NEW** - 22 R execution tests |

## Remaining Work (7 gaps)

### Implementation Order (R done early for execution environment)

| Phase | Order | Issue | Description | Status |
|-------|-------|-------|-------------|--------|
| 1 | 1 | #59 | h5ad/Parquet Data Formats | ✅ Complete |
| 1 | 2 | #69 | R Language Support | ✅ Complete |
| 2 | 3 | #60 | Figure Generation | **Next** |
| 2 | 4 | #61 | Jupyter Notebook Generation | Pending |
| 3 | 5 | #70 | Null Model Statistical Validation | Pending |
| 3 | 6 | #63 | Failure Mode Detection | Pending |
| 4 | 7 | #62 | Code Line Provenance | Pending |
| 5 | 8 | #64 | Multi-Run Convergence | Pending |
| 5 | 9 | #65 | Paper Accuracy Validation | Pending |

### Testing Requirements

- All tests must pass (no skipped tests except environment-dependent)
- Mock tests must be accompanied by real-world tests
- Do not proceed until current task is fully working

## Key Documentation

- `docs/CHECKPOINT.md` - Full session summary
- `docs/PAPER_IMPLEMENTATION_GAPS.md` - 17 tracked gaps (10 complete)
- `/home/jim/.claude/plans/peppy-floating-feather.md` - Full implementation plan
- GitHub Issues #54-#70 - Detailed tracking

## Quick Verification Commands

```bash
# Verify new implementations
python -c "
from kosmos.execution.data_analysis import DataLoader
from kosmos.execution.r_executor import RExecutor, is_r_code
from kosmos.execution.executor import CodeExecutor

# Test h5ad/parquet support
print('DataLoader methods:', [m for m in dir(DataLoader) if m.startswith('load_')])

# Test R executor
executor = RExecutor()
print(f'R available: {executor.is_r_available()}')
print(f'Language detection: {executor.detect_language(\"library(dplyr)\")}')

# Test CodeExecutor R integration
ce = CodeExecutor()
print(f'CodeExecutor R available: {ce.is_r_available()}')
print('All imports successful')
"

# Run tests for completed features
python -m pytest tests/unit/execution/test_data_analysis.py::TestDataLoaderH5ad tests/unit/execution/test_data_analysis.py::TestDataLoaderParquet -v --tb=short
python -m pytest tests/unit/execution/test_r_executor.py -v --tb=short
```

## Resume Command

Start by reading the checkpoint:
```
Read docs/CHECKPOINT.md and docs/PAPER_IMPLEMENTATION_GAPS.md, then continue with the next item: #60 - Figure Generation
```

## Progress Summary

**10/17 gaps fixed (59% complete)**

| Priority | Status |
|----------|--------|
| BLOCKER | 3/3 complete ✅ |
| Critical | 5/5 complete ✅ |
| High | 2/5 complete |
| Medium | 0/2 remaining |
| Low | 0/2 remaining |

## Next Step

Continue with **#60 - Figure Generation**:
- Create `FigureManager` class for matplotlib plot handling
- Add code templates for common plot types
- Track figures in world model artifacts
- Save to `artifacts/cycle_N/figures/`
