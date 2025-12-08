# Paper Implementation Gaps

**Document Purpose**: Track gaps between the original Kosmos paper claims and this implementation.
**Paper Reference**: Mitchener et al., "Kosmos: An AI Scientist for Autonomous Discovery" (arXiv:2511.02824v2)
**Last Updated**: 2025-12-08

---

## Summary

| Priority | Count | Status |
|----------|-------|--------|
| **BLOCKER** | 3 | **3/3 Complete** ✅ |
| Critical | 5 | 0/5 Complete |
| High | 5 | 0/5 Complete |
| Medium | 2 | 0/2 Complete |
| Low | 2 | 0/2 Complete |
| **Total** | **17** | **3/17 Complete** |

> **Note**: BLOCKER priority means the system cannot run at all until fixed. These must be addressed before any other gaps.

---

## BLOCKER Priority Gaps

> **These issues prevent the system from running at all. Fix these first.**

### GAP-013: CLI Hangs Indefinitely ✅ COMPLETE

| Field | Value |
|-------|-------|
| **GitHub Issue** | [#66](https://github.com/jimmc414/Kosmos/issues/66) |
| **Status** | **Complete** (2025-12-08) |
| **Priority** | BLOCKER |
| **Area** | CLI / Message Passing |

**Evidence** (From Runbook Critique):
> "The CLI Deadlock: Section 7.1 notes the main entry point (`kosmos run`) 'hangs indefinitely' due to message-passing failures. A system that cannot be started autonomously violates the core premise of 'Autonomous Discovery.'"

**Solution Implemented**:
- Full async refactor of message passing (send_message, receive_message, process_message)
- Converted ResearchDirector.execute() and all _send_to_*() methods to async
- CLI uses asyncio.run() at entry point
- Agent registration with AgentRegistry sets message router
- 36 unit tests + 14 integration tests pass

**Files Modified**:
- `kosmos/agents/base.py` - Async message passing
- `kosmos/agents/registry.py` - Async routing
- `kosmos/agents/research_director.py` - Async execute, locks, handlers
- `kosmos/cli/commands/run.py` - Async entry point

**Acceptance Criteria**:
- [x] `kosmos run "question"` starts and does not hang
- [x] ResearchDirector can dispatch tasks to Executor
- [x] Messages acknowledged by recipients
- [x] Timeout handling for undelivered messages

---

### GAP-014: SkillLoader Returns None ✅ COMPLETE

| Field | Value |
|-------|-------|
| **GitHub Issue** | [#67](https://github.com/jimmc414/Kosmos/issues/67) |
| **Status** | **Complete** (2025-12-07) |
| **Priority** | BLOCKER |
| **Area** | Agents / Skills |

**Evidence** (From ISSUE_SKILLLOADER_BROKEN.md):
```python
>>> loader.load_skills_for_task(task_type='research', domain='biology')
Skill not found: seaborn
Skill not found: matplotlib
# Returns: None
```

**Solution Implemented**:
- Fixed COMMON_SKILLS to reference actual skill file names
- Added domain-to-bundle mapping for all supported domains
- Skills now load from correct paths and return valid skill objects

**Files Modified**:
- `kosmos/agents/skill_loader.py`

**Acceptance Criteria**:
- [x] `load_skills_for_task(domain='biology')` returns non-None
- [x] Skills injected into HypothesisGeneratorAgent prompts
- [x] 116 available skills accessible by name or bundle
- [x] No "Skill not found" warnings for expected libraries

**Related**: Full documentation in `docs/ISSUE_SKILLLOADER_BROKEN.md`

---

### GAP-015: Pydantic V2 Configuration Failure ✅ COMPLETE

| Field | Value |
|-------|-------|
| **GitHub Issue** | [#68](https://github.com/jimmc414/Kosmos/issues/68) |
| **Status** | **Complete** (2025-12-07) |
| **Priority** | BLOCKER |
| **Area** | Configuration |

**Evidence** (From Unified Bug List):
- Config class uses deprecated Pydantic V1 patterns
- `class Config:` instead of `model_config`
- Validator decorators use V1 syntax

**Solution Implemented**:
- Migrated all Pydantic models to V2 syntax
- Replaced `class Config:` with `model_config = ConfigDict(...)`
- Updated validator decorators to use `@field_validator`
- All deprecation warnings resolved

**Files Modified**:
- `kosmos/config.py`
- Various Pydantic model files

**Acceptance Criteria**:
- [x] Config loads without deprecation warnings
- [x] All Pydantic models use V2 syntax
- [x] `kosmos doctor` passes configuration checks

---

## Critical Priority Gaps

### GAP-001: Self-Correcting Code Execution

| Field | Value |
|-------|-------|
| **GitHub Issue** | [#54](https://github.com/jimmc414/Kosmos/issues/54) |
| **Status** | Not Started |
| **Priority** | Critical |
| **Area** | Execution |

**Paper Claim** (Section 4, Phase B):
> "Data Analysis Agent: If error occurs → reads traceback → fixes code → re-executes (iterative debugging)"

**Current Implementation**:
- Traceback capture exists: `kosmos/execution/executor.py:256`
- Basic retry logic with `max_retries=3`
- `RetryStrategy.modify_code_for_retry()` only handles `KeyError` and `FileNotFoundError`
- Returns `None` for all other error types

**Gap**:
- No intelligent code repair
- No LLM-based analysis of tracebacks
- Most errors trigger retry without fixing the underlying issue

**Files to Modify**:
- `kosmos/execution/executor.py:436-520`

**Acceptance Criteria**:
- [ ] RetryStrategy handles >5 common error types
- [ ] LLM analyzes traceback and suggests fix
- [ ] Code is modified before retry attempt
- [ ] Success rate tracked for auto-repairs

---

### GAP-002: World Model Update Categories

| Field | Value |
|-------|-------|
| **GitHub Issue** | [#55](https://github.com/jimmc414/Kosmos/issues/55) |
| **Status** | Not Started |
| **Priority** | Critical |
| **Area** | World Model |

**Paper Claim** (Section 4, Phase C):
> "The World Model integrates findings using three categories: **Confirmation** (data supports hypothesis), **Conflict** (data contradicts literature), **Pruning** (hypothesis refuted)"

**Current Implementation**:
- `add_finding_with_conflict_check()` exists at `kosmos/world_model/artifacts.py:507-531`
- Function is a **STUB** that always returns `False` for conflicts
- Comment says "Future: More sophisticated conflict detection"
- No enum for update types

**Gap**:
- No Confirmation/Conflict/Pruning categorization
- No semantic conflict detection
- No hypothesis pruning workflow

**Files to Modify**:
- `kosmos/world_model/artifacts.py`
- `kosmos/core/workflow.py` (state transitions for pruning)

**Acceptance Criteria**:
- [ ] `UpdateType` enum with CONFIRMATION, CONFLICT, PRUNING
- [ ] Conflict detection using semantic similarity
- [ ] Pruned hypotheses marked and excluded from future cycles
- [ ] Statistics tracked for each update type

---

### GAP-003: 12-Hour Runtime Constraint

| Field | Value |
|-------|-------|
| **GitHub Issue** | [#56](https://github.com/jimmc414/Kosmos/issues/56) |
| **Status** | Not Started |
| **Priority** | Critical |
| **Area** | Configuration |

**Paper Claim** (Section 1):
> "Standard Runtime: Up to 12 hours (continuous operation)"

**Current Implementation**:
- No `MAX_RUNTIME`, `RUNTIME_LIMIT`, or similar in code
- Only cycle limits (`max_iterations=20`)
- Task-level timeouts (60-600s) but no global timeout

**Gap**:
- System could run indefinitely
- No graceful shutdown at time limit
- No elapsed time tracking

**Files to Modify**:
- `kosmos/config.py` (add MAX_RUNTIME_HOURS)
- `kosmos/workflow/research_loop.py` (check elapsed time)

**Acceptance Criteria**:
- [ ] `MAX_RUNTIME_HOURS` config option (default: 12)
- [ ] Elapsed time tracked from run start
- [ ] Graceful shutdown when limit approached
- [ ] Final report generated before timeout

---

### GAP-004: Parallel Task Execution Mismatch

| Field | Value |
|-------|-------|
| **GitHub Issue** | [#57](https://github.com/jimmc414/Kosmos/issues/57) |
| **Status** | Not Started |
| **Priority** | Critical |
| **Area** | Configuration |

**Paper Claim** (Section 4, Phase A):
> "Generates a batch of **up to 10 parallel tasks**"

**Current Implementation**:
- Task generation: 10 tasks per cycle (`tasks_per_cycle=10`) ✓
- Execution: Default `max_concurrent_experiments=4` at `config.py:708`
- Maximum configurable: 16

**Gap**:
- Paper claims 10 parallel, implementation defaults to 4
- 2.5x lower throughput than paper claims

**Files to Modify**:
- `kosmos/config.py:708`

**Acceptance Criteria**:
- [ ] Default `max_concurrent_experiments=10`
- [ ] Documentation updated to reflect parallel capacity
- [ ] Performance tested at 10 concurrent

---

### GAP-005: Agent Rollout Tracking

| Field | Value |
|-------|-------|
| **GitHub Issue** | [#58](https://github.com/jimmc414/Kosmos/issues/58) |
| **Status** | Not Started |
| **Priority** | Critical |
| **Area** | Agents |

**Paper Claim** (Section 1):
> "Agent Rollouts: ~200 total (~166 data analysis, ~36 literature)"

**Current Implementation**:
- `strategy_stats` tracks "attempts" at `research_director.py:136-141`
- No breakdown by agent type (data vs literature)
- No "rollout" terminology or counting

**Gap**:
- Cannot verify operational scale matches paper
- No metrics for agent-typed rollouts

**Files to Modify**:
- `kosmos/agents/research_director.py`
- `kosmos/core/metrics.py`

**Acceptance Criteria**:
- [ ] `RolloutTracker` class with per-agent-type counts
- [ ] Summary shows "X data analysis + Y literature rollouts"
- [ ] Rollout count included in final report

---

## High Priority Gaps

### GAP-006: h5ad/Parquet Data Format Support

| Field | Value |
|-------|-------|
| **GitHub Issue** | [#59](https://github.com/jimmc414/Kosmos/issues/59) |
| **Status** | Not Started |
| **Priority** | High |
| **Area** | Execution |

**Paper Claim** (Section 3.1):
> "Format: CSV, TSV, Parquet, Excel, or scientific formats (e.g., h5ad for single-cell RNA-seq)"

**Current Implementation**:
- CSV: `DataLoader.load_csv()` at `data_analysis.py:832` ✓
- Excel: `DataLoader.load_excel()` at line 849 ✓
- h5ad: Not implemented
- Parquet: Not implemented

**Gap**:
- Cannot process single-cell RNA-seq datasets (h5ad is standard)
- Cannot process columnar analytics data (Parquet)

**Files to Modify**:
- `kosmos/execution/data_analysis.py`
- `requirements.txt` (add anndata, pyarrow)

**Acceptance Criteria**:
- [ ] `DataLoader.load_h5ad()` using anndata library
- [ ] `DataLoader.load_parquet()` using pyarrow
- [ ] Auto-detection by file extension
- [ ] Tests with real h5ad/parquet files

---

### GAP-007: Figure Generation

| Field | Value |
|-------|-------|
| **GitHub Issue** | [#60](https://github.com/jimmc414/Kosmos/issues/60) |
| **Status** | Not Started |
| **Priority** | High |
| **Area** | Execution |

**Paper Claim** (Section 5):
> "High-resolution figures: High-resolution plots generated by the Data Analysis Agent"

**Current Implementation**:
- No matplotlib/seaborn imports in execution code
- No `plt.savefig()` or figure export
- Code templates have comments like `"# Plot results"` but no actual code

**Gap**:
- System cannot produce visual outputs
- Reports lack figures despite paper claim

**Files to Modify**:
- `kosmos/execution/code_generator.py`
- `kosmos/execution/templates/` (add figure templates)

**Acceptance Criteria**:
- [ ] Code templates include matplotlib plotting
- [ ] Figures saved to `artifacts/cycle_N/figures/`
- [ ] High-resolution export (300+ DPI)
- [ ] Figure references in reports

---

### GAP-008: Jupyter Notebook Generation

| Field | Value |
|-------|-------|
| **GitHub Issue** | [#61](https://github.com/jimmc414/Kosmos/issues/61) |
| **Status** | Not Started |
| **Priority** | High |
| **Area** | Execution |

**Paper Claim** (Section 5):
> "Code Repository: All ~42,000 lines of executable Python code generated during the run (Jupyter notebooks)"

**Current Implementation**:
- `JupyterClient` can EXECUTE notebooks at `jupyter_client.py:326`
- Cannot CREATE notebooks from code
- Compression processes existing notebooks but doesn't generate them

**Gap**:
- System doesn't produce notebook artifacts as claimed
- Code not preserved in reproducible format

**Files to Modify**:
- `kosmos/execution/jupyter_client.py`
- New: `kosmos/execution/notebook_generator.py`

**Acceptance Criteria**:
- [ ] `NotebookGenerator.create_notebook(code, outputs)` function
- [ ] Notebooks saved to `artifacts/cycle_N/notebooks/`
- [ ] Outputs embedded in notebook cells
- [ ] Total line count tracked

---

### GAP-016: R Language Execution Support

| Field | Value |
|-------|-------|
| **GitHub Issue** | [#69](https://github.com/jimmc414/Kosmos/issues/69) |
| **Status** | Not Started |
| **Priority** | High |
| **Area** | Execution |

**Paper Claim** (Statistical Genetics Discoveries):
> Mendelian Randomization linking SOD2 to myocardial fibrosis using `TwoSampleMR` R package

**Current Implementation**:
- Python-only execution environment
- No R language support
- No `rpy2` or R script execution capability

**Gap**:
- Cannot reproduce paper's statistical genetics findings
- Mendelian Randomization requires R packages (`TwoSampleMR`, `susieR`)
- Python-only limits scientific domain coverage

**Files to Modify**:
- `kosmos/execution/executor.py`
- `kosmos/execution/sandbox.py`
- Docker images (add R runtime)

**Acceptance Criteria**:
- [ ] R scripts can be executed in sandbox
- [ ] `TwoSampleMR` package accessible
- [ ] R output captured and integrated with findings
- [ ] Mendelian Randomization analysis can run

---

### GAP-017: Null Model Statistical Validation

| Field | Value |
|-------|-------|
| **GitHub Issue** | [#70](https://github.com/jimmc414/Kosmos/issues/70) |
| **Status** | Not Started |
| **Priority** | High |
| **Area** | Validation |

**Paper Claim** (From Critique Analysis):
> Paper used Null Models - running the exact same analysis on randomized noise data to see if the "discovery" disappears.

**Current Implementation**:
- ScholarEval is 100% LLM-based (8 subjective dimensions)
- No statistical validation against baselines
- No shuffled/permutation testing
- LLM grading LLM creates "sycophancy loop"

**Gap**:
- Circular validation: LLM hallucinates finding → same LLM approves it
- No statistical ground truth
- Cannot distinguish real findings from noise
- Paper achieved 79.4% via human validation, not LLM

**Files to Modify**:
- `kosmos/validation/scholar_eval.py`
- New: `kosmos/validation/null_model.py`

**Acceptance Criteria**:
- [ ] Findings tested against null model (shuffled data)
- [ ] P-value from permutation testing included
- [ ] Findings that persist in noise are flagged/rejected
- [ ] ScholarEval includes statistical validation score

**Related**: Strengthens GAP-012 (#65)

---

## Medium Priority Gaps

### GAP-009: Code Line Provenance

| Field | Value |
|-------|-------|
| **GitHub Issue** | [#62](https://github.com/jimmc414/Kosmos/issues/62) |
| **Status** | Not Started |
| **Priority** | Medium |
| **Area** | Traceability |

**Paper Claim** (Section 5):
> "Code Citation: Hyperlink to the exact Jupyter notebook and line of code that produced the claim"

**Current Implementation**:
- DOI support for literature citations ✓
- No code line → finding mapping
- Phase 4 doc says "PROV-O provenance tracking" is future work

**Gap**:
- Cannot audit which code line produced which finding
- No hyperlinks to source code in reports

**Files to Modify**:
- `kosmos/world_model/artifacts.py`
- `kosmos/execution/executor.py`

**Acceptance Criteria**:
- [ ] Findings include `source_file` and `line_number` fields
- [ ] Report generator creates hyperlinks to code
- [ ] Provenance chain: finding → code → hypothesis

---

### GAP-010: Failure Mode Detection

| Field | Value |
|-------|-------|
| **GitHub Issue** | [#63](https://github.com/jimmc414/Kosmos/issues/63) |
| **Status** | Not Started |
| **Priority** | Medium |
| **Area** | Validation |

**Paper Claim** (Section 6.2):
> "Common failure modes: Over-interpretation, Invented Metrics, Pipeline Pivots, Rabbit Holes"

**Current Implementation**:
- Loop prevention: `MAX_ACTIONS_PER_ITERATION=50` ✓
- Error recovery: `MAX_CONSECUTIVE_ERRORS=3` ✓
- No over-interpretation detection
- No invented metrics validation
- No rabbit hole prevention

**Gap**:
- System may make speculative claims without flagging
- May report non-existent metrics
- May explore irrelevant tangents

**Files to Modify**:
- `kosmos/validation/scholar_eval.py`
- `kosmos/core/convergence.py`

**Acceptance Criteria**:
- [ ] Confidence score for interpretations vs facts
- [ ] Validation that claimed metrics exist in data
- [ ] Relatedness check to original research question
- [ ] Warnings for potential failure modes

---

## Low Priority Gaps

### GAP-011: Multi-Run Convergence Framework

| Field | Value |
|-------|-------|
| **GitHub Issue** | [#64](https://github.com/jimmc414/Kosmos/issues/64) |
| **Status** | Not Started |
| **Priority** | Low |
| **Area** | Workflow |

**Paper Claim** (Section 6.3):
> "Kosmos is non-deterministic. If a finding is critical, run multiple times and look for convergent results."

**Current Implementation**:
- Temperature control exists (0.0-0.7)
- No framework for N independent runs
- No ensemble averaging
- No convergence metrics

**Gap**:
- Cannot validate findings through replication
- No confidence from multiple runs

**Files to Modify**:
- `kosmos/workflow/research_loop.py`
- New: `kosmos/workflow/ensemble.py`

**Acceptance Criteria**:
- [ ] `EnsembleRunner.run(n_runs, research_objective)` function
- [ ] Convergence metrics across runs
- [ ] Report showing findings that appeared in N/M runs

---

### GAP-012: Paper Accuracy Validation

| Field | Value |
|-------|-------|
| **GitHub Issue** | [#65](https://github.com/jimmc414/Kosmos/issues/65) |
| **Status** | Not Started |
| **Priority** | Low |
| **Area** | Validation |

**Paper Claim** (Section 8):
> "79.4% overall accuracy, 85.5% data analysis, 82.1% literature, 57.9% interpretation"

**Current Implementation**:
- ScholarEval framework exists ✓
- Test framework with accuracy targets defined ✓
- No validation study conducted
- `120625_code_review.md` says "Paper claims NOT yet reproduced"

**Gap**:
- Cannot verify system achieves paper accuracy
- No benchmark dataset for validation

**Files to Modify**:
- `tests/requirements/scientific/test_req_sci_validation.py`
- New: benchmark dataset

**Acceptance Criteria**:
- [ ] Validation study with expert-annotated dataset
- [ ] Accuracy measured by statement type
- [ ] Results compared to paper claims
- [ ] Report documenting any deviations

---

## What IS Correctly Implemented

| Feature | Paper Claim | Status |
|---------|-------------|--------|
| 20 research cycles | Up to 20 cycles | ✅ `max_iterations=20` configurable |
| Literature APIs | PubMed, arXiv, Semantic Scholar | ✅ All three with ThreadPoolExecutor |
| ScholarEval validation | 8-dimension peer review | ✅ Fully implemented (but see GAP-017) |
| Context compression | 20:1 compression ratio | ✅ Hierarchical summarization |
| Plan Creator + Reviewer | Task generation with QA | ✅ 5-dimension review scoring |
| Convergence detection | Multiple stopping criteria | ✅ 8 criteria implemented |
| Budget tracking | Cost enforcement | ✅ Graceful convergence |
| **Docker Sandbox** | Secure code execution | ✅ Full isolation, network disabled, resource limits |

### Critique Corrections

Two external critiques (grading C-/C+) made claims that investigation proved **incorrect**:

1. **Security (Critiques claimed Grade F)**: Docker sandbox IS fully implemented:
   - `kosmos/execution/sandbox.py` - DockerSandbox with full isolation
   - `kosmos/execution/docker_manager.py` - Container pool, all capabilities dropped
   - `kosmos/execution/production_executor.py` - Production interface
   - Network disabled, read-only filesystem, resource limits enforced
   - *Critiques were based on outdated info or the dev fallback mode*

2. **Context Compression (Critique claimed "impossible")**: Implementation uses query-based relevance:
   - `LiteratureCompressor.compress_papers()` sorts by relevance_score
   - Processes top 10 papers per batch, not bulk 1,500
   - Extracts statistics rule-based (p-values, correlations)
   - *Architecture is sound, but could benefit from Graph-RAG*

---

## Change Log

| Date | Change |
|------|--------|
| 2025-12-08 | Added 5 gaps from critique analysis (GAP-013 to GAP-017), now 17 total |
| 2025-12-08 | Added BLOCKER priority tier for operational blockers |
| 2025-12-08 | Added critique corrections section for security and compression |
| 2025-12-08 | Initial document created with 12 gaps identified |
