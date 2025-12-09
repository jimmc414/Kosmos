# Resume Prompt - Post Compaction

## Context

You are resuming work on the Kosmos project after a context compaction. The previous sessions implemented **14 paper implementation gaps** (3 BLOCKER + 5 Critical + 5 High + 1 Medium).

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
| #60 | Figure Generation | `FigureManager` class + code template integration |
| #61 | Jupyter Notebook Generation | `NotebookGenerator` class + nbformat integration |
| #70 | Null Model Statistical Validation | `NullModelValidator` class + ScholarEval integration |
| #63 | Failure Mode Detection | `FailureDetector` class (over-interp, invented metrics, rabbit hole) |

### Key Files Created/Modified (Recent)

| File | Changes |
|------|---------|
| `kosmos/validation/failure_detector.py` | **NEW** - FailureDetector, FailureDetectionResult, FailureModeScore (350+ lines) |
| `kosmos/validation/__init__.py` | Exported failure detector classes |
| `kosmos/world_model/artifacts.py` | Added failure_detection_result field to Finding |
| `tests/unit/validation/test_failure_detector.py` | **NEW** - 60 unit tests |
| `tests/integration/validation/test_failure_detection_pipeline.py` | **NEW** - 22 integration tests |

## Remaining Work (3 gaps)

### Implementation Order

| Phase | Order | Issue | Description | Status |
|-------|-------|-------|-------------|--------|
| 3 | 6 | #63 | Failure Mode Detection | ✅ Complete |
| 4 | 7 | #62 | Code Line Provenance | **Next** |
| 5 | 8 | #64 | Multi-Run Convergence | Pending |
| 5 | 9 | #65 | Paper Accuracy Validation | Pending |

### Testing Requirements

- All tests must pass (no skipped tests except environment-dependent)
- Mock tests must be accompanied by real-world tests
- Do not proceed until current task is fully working

## Key Documentation

- `docs/CHECKPOINT.md` - Full session summary
- `docs/PAPER_IMPLEMENTATION_GAPS.md` - 17 tracked gaps (14 complete)
- `/home/jim/.claude/plans/groovy-questing-allen.md` - Failure mode detection plan
- GitHub Issues #54-#70 - Detailed tracking

## Quick Verification Commands

```bash
# Verify failure detection
python -c "
from kosmos.validation import FailureDetector, FailureDetectionResult, FailureModeScore

# Test FailureDetector
detector = FailureDetector()
finding = {
    'finding_id': 'test_001',
    'summary': 'Genetic analysis shows association with cancer susceptibility',
    'statistics': {
        'p_value': 0.001,
        'effect_size': 0.7,
        'sample_size': 150,
    },
    'interpretation': 'Results suggest genetic factors contribute to cancer risk.',
}
context = {
    'research_question': 'What genetic factors are associated with cancer susceptibility?',
}
result = detector.detect_failures(finding, context)
print(f'Passes validation: {result.passes_validation}')
print(f'Over-interpretation score: {result.over_interpretation.score:.3f}')
print(f'Invented metrics score: {result.invented_metrics.score:.3f}')
print(f'Rabbit hole score: {result.rabbit_hole.score:.3f}')
print('All imports successful')
"

# Run tests
python -m pytest tests/unit/validation/test_failure_detector.py -v --tb=short
python -m pytest tests/integration/validation/test_failure_detection_pipeline.py -v --tb=short
```

## Resume Command

Start by reading the checkpoint:
```
Read docs/CHECKPOINT.md and docs/PAPER_IMPLEMENTATION_GAPS.md, then continue with the next item: #62 - Code Line Provenance
```

## Progress Summary

**14/17 gaps fixed (82% complete)**

| Priority | Status |
|----------|--------|
| BLOCKER | 3/3 complete ✅ |
| Critical | 5/5 complete ✅ |
| High | 5/5 complete ✅ |
| Medium | 1/2 complete |
| Low | 0/2 remaining |

## Next Step

Continue with **#62 - Code Line Provenance**:
- Add `source_file` and `line_number` fields to findings
- Enable hyperlinks from reports to source code
- Create provenance chain: finding → code → hypothesis
- Track which code produced which findings
