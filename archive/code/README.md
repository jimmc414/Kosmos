# archive/code — modules with no production call path

Moved here by VIAB#P3-4 (evaluation/VIABILITY_ASSESSMENT_AND_CHANGE_PLAN.md §5 P3-4; docs/PLAN.md
M4). Files keep their repository paths under this directory, so `archive/code/kosmos/core/memory.py`
was `kosmos/core/memory.py`. Their tests moved with them to `archive/code/tests/<same path>`.
A test file that also covered kept code was split: the tests that needed an archived module
are here under the same class names, and the rest stayed in `tests/`.

Nothing here is collected or measured: pytest.ini `norecursedirs` names `archive`, and the
coverage omit list in pyproject.toml holds `"archive/*"`. Nothing in `kosmos/` imports from here.
To bring a module back, `git mv` it to its old path and restore its package `__init__` export.

"Last importer removed by" names the commit that removed the module's last production importer
(kosmos/, scripts/, evaluation/, outside its own package `__init__` and its tests). It was found
with `git log -S "<module> import"` and `git log -S "kosmos.<package>.<module>"`. "never" means
no such importer ever existed: the module was reachable only through its package `__init__`
export and its tests.

## Tier A — zero production importers (first P3-4 commit)

| Module | Added in | Last importer removed by | Reason |
|---|---|---|---|
| kosmos/core/domain_router.py | 7a9399c | never | Routes questions to domain agents; `kosmos run` takes `--domain` and the director never routes |
| kosmos/core/feedback.py | 4120337 | never | Phase 7 feedback loop; the director never constructs it |
| kosmos/core/memory.py | 4120337 | never | Phase 7 memory store; the director never constructs it |
| kosmos/validation/failure_detector.py | bb2a6c5 | never | Failure-mode detection for findings; the P2-2 gate (recomputation and permutation null) is the validation path |
| kosmos/validation/accuracy_tracker.py | a576c05 | never | Paper accuracy framework; nothing tracks accuracy against the paper |
| kosmos/validation/accuracy_validator.py | 158122c | never | Benchmark accuracy validator; nothing calls it |
| kosmos/validation/benchmark_dataset.py | a576c05 | never | Its only importer was accuracy_validator.py (archived with it) |
| kosmos/safety/verifier.py | 2bdfd93 | never | Result verifier; the director validates results through the P2-2 gate |
| kosmos/knowledge/graph_builder.py | 7eb25c0 | never | Paper-to-graph builder; nothing builds the graph through it |
| kosmos/knowledge/graph_visualizer.py | 7eb25c0 | never | Graph visualization; nothing renders the graph |
| kosmos/execution/notebook_generator.py | 171de96 | never | Notebook artifacts; the director saves code under artifacts/runs/<run_id>/code (P2-3) |
| kosmos/execution/figure_manager.py | 3c4d7d4 | never | Figure artifacts; nothing registers figures |
| kosmos/execution/production_executor.py | ff3301d | never | Jupyter/Docker executor; the director runs CodeExecutor with DockerSandbox |
| kosmos/execution/result_collector.py | 98e7f1c | never | Result collection; the director writes results through kosmos.db.operations |
| kosmos/analysis/plotly_viz.py | 3d34825 | never | Interactive plots; nothing plots |
| kosmos/analysis/statistics.py | 3d34825 | never | Descriptive statistics helpers; the templates and analysis_fn compute their own |
| kosmos/workflow/ensemble.py | a576c05 | never | Multi-run convergence over research_loop (Tier B); no CLI command runs it |
| kosmos/api/streaming.py | f3f6a12 | never | SSE endpoint; no HTTP server is mounted (plan §9 non-goal) |
| kosmos/api/websocket.py | f3f6a12 | never | WebSocket endpoint; no HTTP server is mounted (plan §9 non-goal) |

Exports removed: kosmos/execution/__init__.py (ProductionExecutor, ProductionConfig,
execute_code_safely), kosmos/validation/__init__.py (failure detector, accuracy and benchmark
names), kosmos/safety/__init__.py (ResultVerifier, VerificationReport, VerificationIssue),
kosmos/knowledge/__init__.py (graph builder and visualizer names), kosmos/workflow/__init__.py
(ensemble names).
