# Kosmos Viability Assessment and Change Plan

**Commit**: 6cfe7f6 (master) | **Date**: 2026-10-02 | **Brief**: prompts/VIABILITY_ASSESSMENT_PROMPT.md

Citations are `path:line` or `path:first-last`, relative to the repository root at commit 6cfe7f6. Every cited line was re-read on 2026-10-02 before it was cited. docker-compose.yml has an uncommitted working-tree change; its citations refer to HEAD. Database facts come from read-only queries of `kosmos.db` at the repository root.

## 1. Verdict

**CONDITIONALLY VIABLE** as a tool that produces honest, reproducible single-dataset findings. **RESEARCH ONLY** as a reproduction of the paper.

Today a user with a CSV and a question does not get a stored finding whose statistic answers the question. The loop runs, but the hop where generated code executes on the user's data is broken by four independent wiring defects, and nothing downstream reads whether execution succeeded. Every blocker has a mocked acceptance test in Section 5. The verdict becomes VIABLE when the four conditions below hold.

| # | Condition | Mechanical check |
|---|---|---|
| 1 | P0 and P1 land with their mocked end-to-end tests green | Section 8, steps 1 to 11 exit 0 |
| 2 | P2-1 and P2-2 land; on each of the four CSVs under evaluation/data a mocked-LLM run yields at least one result with validation_status `validated` whose statistic matches an independent recomputation within 1e-6 and whose shuffled-control pass rate is at or below 0.05 | tests/unit/validation/test_director_gate.py and `kosmos validate-null` (Section 7) |
| 3 | P2-3 and P2-4 land; every result row carries execution_success, data_source, random_seed, provenance, validation_status and cost_usd, and the CLI results table shows failed rows and real cost | tests/unit/cli/test_run_results.py and tests/unit/execution/test_seed_provenance.py |
| 4 | One owner-authorized live DeepSeek run on the climate CSV (`--budget 1 --max-iterations 3 --seed 42`) produces a validated CO2 vs temperature finding (expected Pearson r about 0.93, p about 6e-29) with a provenance record that `kosmos rerun --result-id <id>` reproduces to 1e-9 | Section 8, live step; needs the sandbox image |

Reasons:

1. **No stored run has ever produced a test statistic tied to its research question on real data.** The only persisted real-data result (kosmos.db results row d39936bb, 2026-02-09) reports Pearson r 0.9887, p 8.3e-53 for year against co2_ppm, because the live template correlates the first two numeric columns (kosmos/execution/code_generator.py:631-636); its supports_hypothesis and interpretation are NULL. Every "significant" finding in the evaluation artifacts is a harness mock (evaluation/run_phase2_tests.py:325-360), and the evaluation's loop_completed check is hard-coded True (evaluation/scientific_evaluation.py:521-525).
2. **The blockers are wiring bugs, not design limits.** Generated code imports kosmos (kosmos/execution/code_generator.py:107,161,241,313,378,472,700) into an image that never installs it (docker/sandbox/Dockerfile:30-35); the sandbox returns values only from a `RESULT:` line nobody prints (kosmos/execution/sandbox.py:438-450); the host data path overrides the container path (kosmos/execution/executor.py:573,655); the director never reads success (kosmos/agents/research_director.py:1589); REFINING never exits (kosmos/agents/research_director.py:1807-1990,2549-2554); verdicts are never written back (kosmos/agents/research_director.py:1633-1641); zero-vector cosine clamps to similarity 1.0 (kosmos/hypothesis/novelty_checker.py:350-354); convergence imports a class that does not exist (kosmos/agents/research_director.py:1253).
3. **The components that run are real and domain-appropriate.** Literature fan-out, LiteLLM hypothesis generation (kosmos.db hypothesis 41c8c967 is specific and testable, novelty 1.0, testability 0.9), protocol design with power analysis, the sandbox container configuration (kosmos/execution/sandbox.py:259-277), and the database layer all work.
4. **The validation stack for honesty exists but is off the live path or fails open.** ScholarEval returns an approving mock on any error (kosmos/validation/scholar_eval.py:204-207); the null model without data runs a parametric pseudo-null unrelated to the dataset (kosmos/validation/null_model.py:214-222,436-474); CodeProvenance is never constructed in production (kosmos/execution/provenance.py:71-148); the director's analyze handler validates nothing (kosmos/agents/research_director.py:1677-1805).
5. **Against the paper, nothing has been measured.** No benchmark exists for the 79.4 percent claim (evaluation/SCIENTIFIC_EVALUATION_REPORT.md:160; archive/planning/VALIDATION_ROADMAP.md:246), discoveries stand at zero (archive/planning/VALIDATION_ROADMAP.md:247), and the 17 gaps were ticked on existence of code within two days (archive/PAPER_IMPLEMENTATION_GAPS.md:1-18,249,658). Reproducing the paper needs 12-hour runs over 1,500 papers, a world model and an accuracy harness that do not exist; that goal stays RESEARCH ONLY.

## 2. Intent

**What the authors set out to build.** README.md:3 calls the project a re-implementation of "Kosmos: An AI Scientist for Autonomous Discovery" (arXiv 2511.02824), by Mitchener et al., Edison Scientific, November 2025; archive/PAPER_IMPLEMENTATION_GAPS.md:4 attributes it correctly, while README.md:3,377 and archive/planning/VALIDATION_ROADMAP.md:3 say "Lu et al. (2024)". README.md:12-20 states the loop: hypotheses from literature and data, experiment design, code execution in Docker, 8-dimension validation, a knowledge graph. The paper's system (docs/paper/PAPER_REFERENCE_ARCHITECTURE.md:3-5,11-15,105-108) takes an objective plus a tabular dataset under 5 GB (docs/paper/PAPER_REFERENCE_ARCHITECTURE.md:63-70), runs up to 12 hours and 20 cycles linked by a structured world model, and emits cited reports, code, a literature log and figures (docs/paper/PAPER_REFERENCE_ARCHITECTURE.md:166-184).

**Three goals.** docs/planning/objective.md:62-110 commits to Goal A, faithful reproduction; Goal B, a practical tool for graduate students and individual researchers; Goal C, educational reference. Entry points: CLI (README.md:70-96), Python library (README.md:52-68), streaming service (README.md:114); single-user (README.md:341). Providers: Anthropic, OpenAI, LiteLLM (README.md:162-177); all empirical runs used DeepSeek through LiteLLM (evaluation/SCIENTIFIC_EVALUATION_REPORT.md:21-22). The integration plan's exit criterion is to "replicate ANY kosmos-figures discovery autonomously" (docs/planning/integration-plan.md:620-621; docs/domain-roadmaps/biology.md:1-3).

**What success would do, in the authors' terms.** Publish honestly measured performance against the paper (archive/planning/VALIDATION_ROADMAP.md:3,228-234,289); test every MUST requirement (archive/planning/REQUIREMENTS.md:1-6; traceability stood at 0 of 293 on 2025-11-21, docs/REQUIREMENTS_TRACEABILITY_MATRIX.md:9-12); meet the adoption criteria at docs/planning/objective.md:114-163; tick the domain success boxes, all unchecked (docs/domain-roadmaps/biology.md:398-415); reproduce the 7 discoveries and the 79.4 percent accuracy (archive/120625_code_review.md:515-538). None is met.

**How "complete" was defined.** The repository supplies six components the paper omits (README.md:360-371; archive/implementation/OPEN_QUESTIONS.md:83-170) and tracked 17 gaps, all COMPLETE between 2025-12-07 and 2025-12-09 (archive/PAPER_IMPLEMENTATION_GAPS.md:1-18). Closure meant code exists: GAP-004 is a "Simple 1-line change" (archive/PAPER_IMPLEMENTATION_GAPS.md:240-249); GAP-012 ticks a validation study against a synthetic benchmark built at the paper's rates (archive/PAPER_IMPLEMENTATION_GAPS.md:658; kosmos/validation/benchmark_dataset.py:349-374,501; docs/TODO.md:9-13 admits no expert-annotated data); GAP-016 declares R execution complete with tests that skip without R (archive/PAPER_IMPLEMENTATION_GAPS.md:450), two days after "Python-only, no R support" (archive/120625_code_review.md:221-230). The February 2026 evaluation: "each component was tested with mocks, but nobody wired them together and pressed go" (evaluation/SCIENTIST_NARRATIVE.md:19).

**The arc.** 364 commits: 338 between 2025-11-07 and 2025-12-12, 26 in 2026. A fast architecture build-out declared production-ready, repeated downgrades as end-to-end runs exposed broken wiring, checklist-driven gap closure, then evaluation-driven fixing; the paper's accuracy and discovery claims were never validated. The README says the results are "not yet reproduced" and the system "suitable for experimentation" (README.md:321,331); an external critique graded it C-, "a sophisticated architectural blueprint for a machine that does not currently run" (archive/runbook_critque1.md:9-10).

| Date | Commit | Message or event |
|---|---|---|
| 2025-11-06 | CHANGELOG.md:163-166 | "Initial Production Release" 0.1.0, one day before the first commit; claims 90 percent coverage (CHANGELOG.md:189) |
| 2025-11-07 | f7bbed3 | First commit |
| 2025-11-13 | bd4d689 | "Phase 10 Complete: Production-Ready v1.0 Release" |
| 2025-11-21 | caff417 | "change status from production-ready to E2E testing" |
| 2025-12-07 | 0063c54 | "Add 'yet' to reproduction status language for optimism" |
| 2025-12-09 | 9aac902 | "Update status to 'ready for user testing'" |
| 2026-02-13 | 3ff33c3 | "Fix 42 critical evaluation findings across 10 work packages" |

Claims versus measurement:

| Paper claim | README position | Measured evidence |
|---|---|---|
| 79.4 percent of statements accurate | "Architecture implemented, not validated" (README.md:325); "Not a reproduction study" (README.md:343) | GAP-012 closed on a synthetic 90-finding benchmark generated at the paper's rates (kosmos/validation/benchmark_dataset.py:349-374,501; docs/CHECKPOINT.md:17); evaluation BLOCKER (evaluation/SCIENTIFIC_EVALUATION_REPORT.md:160); roadmap "Unknown" (archive/planning/VALIDATION_ROADMAP.md:246) |
| 85.5 / 82.1 / 57.9 percent by category | not stated | "Not reproduced" (archive/120625_code_review.md:59-62); implementation targets lowered to 75/80/75/50 (archive/PAPER_IMPLEMENTATION_GAPS.md:642-643) |
| 7 discoveries | "Not reproduced" (README.md:326) | 0 discoveries (archive/planning/VALIDATION_ROADMAP.md:247); evaluation PARTIAL (evaluation/SCIENTIFIC_EVALUATION_REPORT.md:165) |
| 1,500 papers per run, 36 literature rollouts | "Architecture supports this" (README.md:327) | PARTIAL (evaluation/SCIENTIFIC_EVALUATION_REPORT.md:161); one query returned 30 papers (evaluation/SCIENTIST_NARRATIVE.md:83); the 20-cycle run had literature search "hanging (disabled)" (archive/planning/VALIDATION_ROADMAP.md:119) |
| 42,000 lines of code per run | "Architecture supports this" (README.md:328) | PARTIAL (evaluation/SCIENTIFIC_EVALUATION_REPORT.md:158); NotebookGenerator has no production importer (P3-4) |
| 200 agent rollouts | "Configurable via max_iterations" (README.md:329) | the scaled climate run hit the 100-action cap with 1 experiment (evaluation/personas/runs/004_climate_data_scientist/v007_20260209/tier1_scaled/EVALUATION_REPORT.md:62-64) |
| 12-hour runtime, 20 cycles | not stated | max_runtime_hours=12 added as a config field (archive/PAPER_IMPLEMENTATION_GAPS.md:207); "Not reproduced/validated" (archive/120625_code_review.md:173); the longest director run lasted 4.6 hours (16,448.5 s, evaluation/personas/runs/004_climate_data_scientist/v007_20260209/tier1_scaled/EVALUATION_REPORT.md:11); the library loop completed 20 cycles in 29.2 minutes for $0.0178 on DeepSeek with no execution results (archive/planning/VALIDATION_ROADMAP.md:127-140) |
| 10 parallel tasks per cycle | "Default now matches paper" (README.md:198) | a one-line default change (archive/PAPER_IMPLEMENTATION_GAPS.md:240-249) |
| 4 to 6 months of expert time per run | not stated | "Not validated" (archive/120625_code_review.md:67) |
| Cited reports, 3 to 4 narratives | not stated | PARTIAL, "Summarizer not importable" (evaluation/SCIENTIFIC_EVALUATION_REPORT.md:170) |
| Self-reported counts | 3704 tests (README.md:8,398); 116 skills (README.md:369); ten examples (examples/README.md:21-103) | the README's own table sums to 3602 (README.md:306-312); 119 SKILL.md files exist under kosmos-claude-scientific-skills/scientific-skills; examples 03 to 10 exist only as placeholder files |

## 3. Current state and one-run trace

**Two orchestrators.** `kosmos run` builds a ResearchDirectorAgent (kosmos/cli/commands/run.py:183-202) that drives the state machine in kosmos/core/workflow.py and calls the worker agents directly from its handlers (kosmos/agents/research_director.py:1402-1990). The README quickstart (README.md:52-68), scripts/smoke_test.py:18-29 and kosmos/workflow/ensemble.py drive a different class in kosmos/workflow/research_loop.py, a plan, review, delegate, ScholarEval cycle loop. The two share the worker classes, the get_client() singleton (kosmos/core/llm.py:613-683), the SQLite database and the event bus, nothing else. Only the director has empirical runs; evaluation/scientific_evaluation.py:258-300 drives it.

**One run, hop by hop.** Persona 004, version v007, 2026-02-09, DeepSeek through LiteLLM, the 64-row climate CSV, 3 iterations and a scaled 10-iteration run. The abbreviation R004 below means evaluation/personas/runs/004_climate_data_scientist/v007_20260209.

| Hop | What happened | Evidence | Real science? |
|---|---|---|---|
| 1 Pre-flight | Provider litellm, model deepseek/deepseek-chat; the CSV loads | evaluation/SCIENTIFIC_EVALUATION_REPORT.md:21-22; R004/tier3/NARRATIVE.md:20 | yes |
| 2 Literature | arXiv returned HTTP 429 and Semantic Scholar refused connections; each search waited the 90 s timeout | R004/tier3/NARRATIVE.md:20 | partial |
| 3 Hypotheses | Real LLM output. kosmos.db hypothesis 41c8c967 (novelty 1.0, testability 0.9): "The correlation between atmospheric CO2 concentrations and global surface temperature anomalies is strongest when CO2 leads temperature by 6-12 months, with a Pearson correlation coefficient exceeding 0.8". The pool grew to 35 then 195 | R004/tier1_scaled/EVALUATION_REPORT.md:44-54,63 | yes |
| 4 Design | "No template found for computational, falling back to LLM"; the LLM protocol had 0 steps. All 7 experiments in kosmos.db are status CREATED with code_generated NULL | R004/tier3/NARRATIVE.md:46; kosmos/agents/experiment_designer.py:380-407 (climate_science is not in the domain map, so the type is COMPUTATIONAL) | partial |
| 5 Execute | **First hop where real science stops.** The generated code was a T-test template indexing columns `group` and `measurement`; KeyError 'group'; synthetic fallback. The harness scored it PASS with success false. On the director path the executor's return value is read without checking success and `{}` is stored | R004/tier3/NARRATIVE.md:48; R004/tier1/artifacts/phase2_components/2.4_code_execution.json:8-15,23; kosmos/agents/research_director.py:1589,1626-1641 | no |
| 6 Analyze | The component test interprets a hard-coded mock (p=0.003, d=0.8, t=-12.5) built by the harness; the narrative reports it as a system result | evaluation/run_phase2_tests.py:325-360; R004/tier1/artifacts/phase2_components/2.5_data_analysis.json:6-13; R004/tier3/NARRATIVE.md:30 | no |
| 7 Refine | REFINING never exits: 100 actions, 96 iterations, 1 experiment, 195 hypotheses, not converged | R004/tier1_scaled/EVALUATION_REPORT.md:62-74; kosmos/agents/research_director.py:2549-2554 | no |
| 8 Cost | $0.00 reported; is_always_zero true; the $1 budget never enforced | R004/tier1/artifacts/phase2_components/2.6_convergence_detection.json:22-26; R004/tier3/NARRATIVE.md:52 | no |

**The one persisted real-data result.** kosmos.db results row d39936bb (2026-02-09 15:27:37 UTC, experiment fa9ac1a3): n_samples 64, data_source file, pearson_r 0.98869, p 8.3e-53, interpretation NULL, supports_hypothesis NULL. Recomputed on the CSV, year against co2_ppm gives exactly r 0.98869; CO2 against temperature gives r 0.9317, p 5.8e-29. The template takes numeric_cols[0] and numeric_cols[1] (kosmos/execution/code_generator.py:631-636). Other real-data attempts: the perovskite run reports p 1.0 and effect 1.7e-18, which is the correlation between the two orthogonal design factors annealing_temp_c and film_thickness_nm, with efficiency never analyzed (evaluation/personas/runs/002_perovskite_solar_cell_researcher/v003_20260208/tier2/TECHNICAL_REPORT.md:99-108); the enzyme run executed the ML template on synthetic data, accuracy 0.7500, supported=None (evaluation/logs/evaluation_20260207_212853.log:343-352,368).

**Why execution fails on both paths at HEAD.** Sandbox path: every template imports kosmos (kosmos/execution/code_generator.py:107,161,241,313,378,472,700) but the image installs only docker/sandbox/requirements.txt (docker/sandbox/Dockerfile:30-35); no template prints the `RESULT:` line the parser needs (kosmos/execution/sandbox.py:438-450); and execute_with_data prepends the host path after the container path, so pandas reads a path that does not exist in the container and the template falls back to synthetic data (kosmos/execution/executor.py:573,655; kosmos/execution/code_generator.py:593-608). Host path: SAFE_BUILTINS (kosmos/execution/executor.py:43-83), installed as the exec builtins by _prepare_globals (kosmos/execution/executor.py:589-597), has no `dir`, so the guard `if 'data_path' in dir()` (kosmos/execution/code_generator.py:112,246,383,476,593) raises NameError, and the kosmos imports are refused (kosmos/execution/executor.py:86-110). Which path runs is not configurable: with the docker SDK installed, CodeExecutor() constructs DockerSandbox() (kosmos/execution/executor.py:223), which raises RuntimeError when the daemon is unreachable (kosmos/execution/sandbox.py:117-122), and the director does not catch it (kosmos/agents/research_director.py:1551).

**What works.** Configuration loading; the provider abstraction; literature fan-out with deduplication and caching; hypothesis generation with database storage; template and LLM protocol design with power analysis; the sandbox container configuration (network disabled, read-only root, all capabilities dropped, kosmos/execution/sandbox.py:259-277); the database layer; ArtifactStateManager; ScholarEval scoring when given a client; the event bus; the CLI scaffolding.

## 4. Viability analysis against the criteria

Criteria (a) to (e) come from the brief. Two more follow from the intent: (f) findings traceable to literature, because the paper's output is a cited report, and (g) the run stops on evidence rather than on a cap, because convergence is the loop's exit.

| Criterion | Status at 6cfe7f6 | Evidence |
|---|---|---|
| (a) The finding came from code that executed on the user's data | FAIL | Section 3, "Why execution fails on both paths" |
| (b) It carries a correct test statistic and p-value | FAIL | Column choice is positional (kosmos/execution/code_generator.py:631-636) or by LLM-chosen variable name (kosmos/execution/code_generator.py:74-78,219-221,364-366); the one real-data row tests the wrong pair; the director keeps only seven statistic keys (kosmos/agents/research_director.py:1593-1598) |
| (c) It is reproducible from a seed and a provenance record | FAIL | Seeds are 42 or LLM-chosen (kosmos/execution/code_generator.py:89,231,368,572; kosmos/agents/experiment_designer.py:675); no `--seed` option (kosmos/cli/commands/run.py:51-61); CodeProvenance never constructed in production (kosmos/execution/provenance.py:71-148); experiments.code_generated is NULL on all 7 rows |
| (d) It survived a non-trivial validation gate | FAIL | The analyze handler validates nothing (kosmos/agents/research_director.py:1677-1805); ScholarEval approves on error (kosmos/validation/scholar_eval.py:204-207); the null model without data draws a parametric null (kosmos/validation/null_model.py:214-222,436-474) |
| (e) It is reported honestly beside failed tasks and cost | FAIL | Failures store `{}` (kosmos/agents/research_director.py:1589,1626-1641); experiments_executed counts every mark_experiment_complete (kosmos/cli/commands/run.py:466; kosmos/agents/research_director.py:1653); api_calls reads an attribute that does not exist (kosmos/cli/commands/run.py:459 against kosmos/core/providers/base.py:384); `--budget` is stored and never read (kosmos/cli/commands/run.py:144-145,161; kosmos/core/metrics.py:157) |
| (f) Findings are traceable to literature | PARTIAL | Literature search works; nothing links a result row to papers; the report summarizer is not importable (evaluation/SCIENTIFIC_EVALUATION_REPORT.md:170) |
| (g) The run stops on evidence, not on a cap | FAIL | REFINING dead end (kosmos/agents/research_director.py:1807-1990,2549-2554); convergence receives no hypotheses and no results (kosmos/agents/research_director.py:1248,1253); the scaled run ended at the action cap, unconverged (R004/tier1_scaled/EVALUATION_REPORT.md:62-66) |

Structural problems are fixable by wiring and bug fixes. Fundamental ones are limits of the design or of LLM-driven science.

| Defect | Class | Reason |
|---|---|---|
| Generated code cannot run in the sandbox or on the host | Structural | Remove the kosmos imports, print a result footer, fix path precedence (P0-1, P0-2, P0-3, P0-5, P0-6) |
| Director ignores execution success and records phantom results | Structural | P0-4 |
| REFINING dead end; unbounded re-analysis and refinement | Structural | P1-2, P2-7 |
| Verdicts never persisted; None p-value crash | Structural | P1-1 |
| Error recovery blocks the event loop | Structural | P1-3 |
| Convergence fed empty data | Structural | P1-4 |
| Budget inert; cost priced as Claude Sonnet | Structural | P1-5, P2-4 |
| Novelty NaN clamp; vector search returns [] | Structural | P2-5 |
| No binding from protocol variables to dataset columns | Structural, at the model level: Variable has no column field (kosmos/models/experiment.py:42-70) and the designer never sees the dataset (kosmos/agents/experiment_designer.py:164-299) | P2-1 |
| Validation off the live path or fail-open | Structural | P2-2 |
| The LLM can affirm a wrong or synthetic result | Fundamental, bounded | The Feb 6 scorecard records "LLM can affirm synthetic data results" (evaluation/artifacts/phase5_scorecard.json:6-118); recomputation, a permutation null on the real data and ScholarEval (P2-2) bound it, they do not remove it |
| Reproduction at the paper's scale: 12 hours, 1,500 papers, world-model coherence, 79.4 percent accuracy | Fundamental for this codebase | Needs a benchmark and a harness that do not exist, plus a system an order of magnitude larger; RESEARCH ONLY |
| Science from one CSV with four template tests | Fundamental scope limit | After P2-1 the product is correlation, t-test, ANOVA and regression on tabular data with honest gates; that is the viable product, not open-ended discovery |

The verdict follows: every FAIL in the first table maps to a Structural row, and the two Fundamental rows bound what the fixed system can claim, which is why the tool is CONDITIONALLY VIABLE for single-dataset findings and RESEARCH ONLY for paper reproduction.

## 5. Change plan by phase

Conventions for every change: paths are relative to the repository root; line numbers are exact at 6cfe7f6 and were re-read on 2026-10-02. Run targeted tests with `python -m pytest <file> --no-cov -p no:cacheprovider -q`. pytest.ini sets `asyncio_mode = auto` (pytest.ini:44) and turns warnings into errors (pytest.ini:47), so new tests must not emit DeprecationWarning (use `datetime.now(timezone.utc)`, never `datetime.utcnow()`). The owner's environment has the docker SDK, so SANDBOX_AVAILABLE is True (kosmos/execution/executor.py:24-29) and CodeExecutor() constructs DockerSandbox() (kosmos/execution/executor.py:223), which raises RuntimeError when the daemon is unreachable (kosmos/execution/sandbox.py:117-122). Mock the LLM with `patch('kosmos.agents.<module>.get_client')`; the director fixture pattern is tests/unit/agents/test_research_director_loops.py:23-33; in-memory SQLite comes from `kosmos.db.init_database("sqlite:///:memory:")` (kosmos/db/__init__.py:26-27,72-78); local execution uses `CodeExecutor(use_sandbox=False)` (fixture at tests/unit/execution/test_executor.py:20-23). The real-data fixture is evaluation/data/climate_co2_temperature_test.csv (64 rows; columns year, co2_ppm, temp_anomaly_c, solar_irradiance_wm2, volcanic_aerosol_index, enso_index, co2_growth_rate, decade; decade is a string).

Reconciliations that hold across phases: P1-5 owns the provider-to-metrics bridge and P2-4 only adds per-call cost_usd and per-result attribution. P0-4 writes execution_success and data_source into the result data JSON; P2-0 promotes them to columns and P2-2 onward read the columns. P1-2 fixes the REFINING exit without touching decide_next_action; P2-7 then changes decide_next_action's GENERATING and REFINING branches. P2-1 bypasses the protocol templates whenever a dataset is supplied, which is what makes Tier C archival (P3-4) an owner decision. Changes are applied in the order listed; each leaves the suite no worse and is independently applicable within its phase. Every line number below is relative to 6cfe7f6; once an earlier change has edited a file, find the later change's target in that file by the quoted code, because the numbers shift. Bare line numbers: in a change's Files field a bare number belongs to the path written most recently before it; elsewhere in that change a bare number refers to the first file listed under Files, and every other file is written with its path.

| Phase | Changes | Hours |
|---|---|---|
| P0: one real experiment runs end to end on the CLI path and is stored honestly | P0-1 Generic template self-contained and syntax-safe; P0-2 `RESULT:` JSON footer in _execute_in_sandbox; P0-3 execute_with_data skips the host prefix when sandboxed; P0-4 director reads exec_result.success, honest result rows, experiment status, SandboxUnavailable failure, no phantom ids; P0-5 TTest and Correlation templates self-contained, `{method}` NameError fixed; P0-6 LogLog and ML templates self-contained | 11 |
| P1: the loop closes | P1-1 verdict and hypothesis status persisted, failed executions skip the LLM, None p-value guard; P1-2 leave REFINING on every exit; P1-3 error recovery never blocks the loop; P1-4 convergence loads real hypotheses and results; P1-5 `--budget` arms enforcement, provider calls recorded, per-model pricing | 8.5 |
| P2: quality of scientific output | P2-0 result columns and helpers; P2-1 dataset schema and variable-to-column binding; P2-2 recomputation, permutation null on real data, provider-agnostic ScholarEval, verdict rule; P2-3 seed and provenance; P2-4 honest run report and cost; P2-5 novelty without sentence-transformers; P2-6 LiteLLM JSON mode and tolerant parsing; P2-7 hypothesis-pool control | 64 |
| P3: hardening and cleanup | P3-1 CodeValidator and emergency stop on the director path; P3-2 image build, declared dependencies, no HTTP probe; P3-3 test suite green for surviving modules, README test count removed; P3-4 archive zero-importer modules; P3-5 README and DEEP_ONBOARD corrections | 22 (+4 for Tier C) |
| Consolidation port | C-1 findings JSON through ArtifactStateManager and a `kosmos report --run-id` command | 5 |
| Metric commands | M-1 `kosmos validate-null` and `kosmos rerun` | 4 |

### P0-1: Generic template self-contained and syntax-safe

**Goal.** Make the GenericComputational template, the live template, run in the sandbox image by removing its only kosmos import, and make its generated code syntax-safe for LLM-chosen names.

**Files.** kosmos/execution/code_generator.py, GenericComputationalCodeTemplate.generate (566-728) and ExperimentCodeGenerator._create_code_generation_prompt (858-905).

**Current.** Line 700 emits `from kosmos.analysis.visualization import PublicationVisualizer` and 701 constructs it; 703-712 is dead because figure_path is never injected. The import fails in the container (no kosmos in docker/sandbox/requirements.txt, installed at docker/sandbox/Dockerfile:30-35) and under restricted host exec (kosmos/execution/executor.py:86-110). Lines 604-607 emit `{x_var}_data` identifiers and `'{x_var}'` literals from protocol.variables.keys() (568-570); a space or apostrophe in a key makes _validate_syntax (982-989) raise ValueError. Line 582 emits `# Protocol: {protocol.name}` unescaped. Prompt line 900 says "Use kosmos.execution.data_analysis.DataAnalyzer".

**Required.** (1) Delete lines 699-712 inclusive; keep 714-725. (2) Replace 604-607 with fixed identifiers and repr'd keys:

```python
"    _x_syn = np.linspace(0, 10, n)",
"    noise = np.random.normal(0, 0.5, n)",
"    _y_syn = 2.0 * np.exp(-0.3 * _x_syn) + noise",
f"    df = pd.DataFrame({{{x_var!r}: _x_syn, {y_var!r}: _y_syn}})",
```

(3) Replace 582 with `f"# Protocol: {' '.join(str(protocol.name).split())}"`. (4) In the prompt (890-900): replace line 900 with "Do NOT import kosmos or any kosmos.* module; only pandas, numpy, scipy.stats, scikit-learn and statsmodels exist in the sandbox." and add a fifth numbered item after 894: "5. Assign the final results dictionary to a top-level variable named results".

**Acceptance test.** tests/unit/execution/test_code_generator.py. Add a fixture generic_protocol by copying ttest_protocol (tests/unit/execution/test_code_generator.py:31-62) with experiment_type=ExperimentType.COMPUTATIONAL, name="CO2 vs temperature's trend", variables keyed "CO2 concentration" and "temp anomaly", and no scaling or ML keywords. (a) `code = code_generator.generate(generic_protocol)`; assert `"kosmos" not in code` and `ast.parse(code)` succeeds. (b) Write a 10-row CSV with header `year,co2_ppm` to tmp_path, `exec(code, {"data_path": str(csv)})` under redirect_stdout, assert `ns["results"]["data_source"] == "file"`, `isinstance(ns["results"]["p_value"], float)`, `ns["results"]["n_samples"] == 10`. (c) Assert `"kosmos" not in code_generator._create_code_generation_prompt(generic_protocol)`. No LLM, no Docker.

**Dependencies.** None.

**Risk and blast radius.** code_generator.py is in coupling cluster 0 (docs/DEEP_ONBOARD.md:1986) but no production code reads the deleted lines; existing tests do not assert on Generic output; kosmos/analysis/visualization.py keeps its direct tests.

**Effort.** 1 h.

### P0-2: `RESULT:` JSON footer in _execute_in_sandbox

**Goal.** Make the container's results dict reach the host by appending a JSON `RESULT:` footer to every sandboxed program.

**Files.** kosmos/execution/executor.py, CodeExecutor._execute_in_sandbox (555-587); new module constant after line 40.

**Current.** DockerSandbox._extract_return_value (kosmos/execution/sandbox.py:438-450) parses only a stdout line starting with `RESULT:`; no template prints one, so return_value is always None on success. _execute_in_sandbox prepends the data path (565-573) and calls self.sandbox.execute(code, ...) (576). Prior art: kosmos/execution/jupyter_client.py:239-273 prints json.dumps(_result, default=str) between markers.

**Required.** Add after line 40:

```python
SANDBOX_RESULT_FOOTER = '''
# --- kosmos sandbox result footer (appended by CodeExecutor._execute_in_sandbox) ---
import json as _kj
def _kdef(o):
    try:
        import numpy as _n
        if isinstance(o, _n.integer): return int(o)
        if isinstance(o, _n.floating): return float(o)
        if isinstance(o, _n.bool_): return bool(o)
        if isinstance(o, _n.ndarray): return o.tolist()
    except Exception:
        pass
    return str(o)
def _kfix(o):
    if isinstance(o, dict): return {str(k): _kfix(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_kfix(v) for v in o]
    return o
_kr = globals().get('results', globals().get('result'))
if _kr is not None:
    print("RESULT:" + _kj.dumps(_kfix(_kr), default=_kdef))
'''
```

In _execute_in_sandbox, after the `if local_vars and 'data_path' in local_vars:` block (565-573) and before line 576, add `code = code.rstrip("\n") + "\n" + SANDBOX_RESULT_FOOTER`. Each retry passes a fresh copy of the code (277-282), so footers do not accumulate. json.dumps emits one physical line, which _extract_return_value requires (kosmos/execution/sandbox.py:442-443). NaN serializes as the NaN token; P0-4 maps it to None.

**Acceptance test.** tests/unit/execution/test_executor.py, class TestSandboxIntegration (390-412), decorators as at 393-394 (`@patch('kosmos.execution.executor.SANDBOX_AVAILABLE', True)`, `@patch('kosmos.execution.executor.DockerSandbox')`). (a) `mock_sandbox.execute.return_value = SandboxExecutionResult(success=True)`; `executor.execute("import numpy as np\nresults={'p': np.float64(0.01), 'n': np.int64(3), 'ok': np.bool_(True), 'arr': np.array([1,2]), 'x': float('nan')}")`; `code = mock_sandbox.execute.call_args.args[0]`; assert `code.count("RESULT:") == 1`; exec(code, ns) under redirect_stdout; the `RESULT:` line json.loads to `{'p': 0.01, 'n': 3, 'ok': True, 'arr': [1, 2], 'x': nan}` (compare x with math.isnan). (b) Retry path: return_value = `SandboxExecutionResult(success=False, error="x", error_type="ExecutionError")`; `CodeExecutor(use_sandbox=True, max_retries=2, retry_delay=0.01).execute("results={}", retry_on_error=True)`; every call's code has exactly one footer. SandboxExecutionResult is importable from kosmos.execution.sandbox even with the class patched in executor.

**Dependencies.** None.

**Risk and blast radius.** Only the sandbox branch of CodeExecutor (_execute_once routes at 474-475). Callers: kosmos/agents/research_director.py:1579-1583, execute_protocol_code (kosmos/execution/executor.py:1017-1088), parallel.py through execute_protocol_code. Unsandboxed tests (fixture at tests/unit/execution/test_executor.py:20-23) are unaffected.

**Effort.** 1 h.

### P0-3: execute_with_data skips the host prefix when sandboxed

**Goal.** Make the container path win so templates read the mounted CSV instead of silently falling back to synthetic data.

**Files.** kosmos/execution/executor.py, CodeExecutor.execute_with_data (632-660).

**Current.** Line 655 prepends `data_path = '<host path>'`; _execute_in_sandbox then prepends `data_path = '/workspace/data/<file>'` above it (573). The later host assignment wins, pd.read_csv fails in the read-only container, and every template's except branch (for example kosmos/execution/code_generator.py:593-608) switches to synthetic data with `_data_source = 'synthetic'`.

**Required.** Replace 653-660 with:

```python
local_vars = {'data_path': data_path}
if self.use_sandbox:
    # _execute_in_sandbox assigns data_path to the container mount (executor.py:573).
    return self.execute(code, local_vars, retry_on_error)
augmented_code = f"# Data path injected by executor\ndata_path = {repr(data_path)}\n\n{code}"
return self.execute(augmented_code, local_vars, retry_on_error)
```

**Acceptance test.** tests/unit/execution/test_executor.py, TestSandboxIntegration, same patches as P0-2: `executor.execute_with_data("results={'d': data_path}", "/host/dir/test_data.csv")`; capture the code passed to the sandbox; assert `code.splitlines()[0] == "data_path = '/workspace/data/test_data.csv'"` and `"/host/dir" not in code`; assert `mock_sandbox.execute.call_args.kwargs["data_files"] == {"test_data.csv": "/host/dir/test_data.csv"}`. The existing unsandboxed test TestExecuteWithData.test_execute_with_data_path (tests/unit/execution/test_executor.py:330-341) must still pass.

**Dependencies.** None.

**Risk and blast radius.** Only execute_with_data; callers kosmos/agents/research_director.py:1579 and kosmos/execution/executor.py:1080.

**Effort.** 0.5 h.

### P0-4: Director reads exec_result.success and stores honest rows

**Goal.** The director reads exec_result.success, stores an honest result row and experiment status for both success and failure, records the executed code, and stores a SandboxUnavailable failure instead of crashing when Docker is down.

**Files.** kosmos/agents/research_director.py, __init__ (68-170, lazy-init fields at 147-149) and _handle_execute_experiment_action (1532-1675) including _json_safe (1606-1627). Uses update_experiment_status (kosmos/db/operations.py:305-332), which has no caller today, ExperimentStatus (kosmos/db/models.py:22-28), and Experiment.code_generated and error_message (kosmos/db/models.py:58,60).

**Current.** Lines 1548-1551 construct CodeExecutor(max_retries=3) inside the try; a RuntimeError from DockerSandbox() propagates to the except at 1668 and into error recovery, which before P1-3 raises TimeoutError. Line 1589 reads only .return_value; .success is never read; failures store `{}` with p_value None. A DB failure is logged (1642-1643) but the result id is still added to the plan (1652). Line 1653 always calls mark_experiment_complete. Experiment status is never updated. _json_safe passes NaN and inf through (1607).

**Required.** (1) After line 149 add `self._sandbox_error: Optional[str] = None`. In the imports at 1539-1544 add update_experiment_status to the kosmos.db.operations import, add `from kosmos.db.models import ExperimentStatus`, and change 1540 to `from kosmos.execution.executor import CodeExecutor, ExecutionResult`. (2) Replace 1550-1551 with:

```python
if self._code_executor is None:
    try:
        self._code_executor = CodeExecutor(max_retries=3)
    except RuntimeError as sandbox_err:   # DockerSandbox() failed (sandbox.py:117-122)
        logger.error(f"Sandbox unavailable; experiment {protocol_id} will be recorded as failed: {sandbox_err}")
        self._sandbox_error = str(sandbox_err)
```

(3) Replace 1577-1583 with:

```python
if self._code_executor is None:
    exec_result = ExecutionResult(success=False, error=self._sandbox_error, error_type="SandboxUnavailable")
elif self.data_path:
    exec_result = self._code_executor.execute_with_data(code, self.data_path, retry_on_error=True)
else:
    exec_result = self._code_executor.execute(code, retry_on_error=True)
success = bool(exec_result.success)
```

(4) Replace 1589 with `return_value = exec_result.return_value if (success and exec_result.return_value) else {}`; keep 1590-1602. (5) In _json_safe, before line 1607, add `if isinstance(obj, float) and (obj != obj or obj in (float('inf'), float('-inf'))): return None`. (6) After 1627 add:

```python
safe_data = _json_safe(return_value) if isinstance(return_value, dict) else {"raw_return_value": str(return_value)}
safe_data.update({
    "execution_success": success,
    "error": exec_result.error,
    "error_type": exec_result.error_type,
    "stderr_tail": (exec_result.stderr or "")[-2000:],
    "execution_time": exec_result.execution_time,
    "executor_mode": "sandbox" if (self._code_executor is not None and self._code_executor.use_sandbox) else ("none" if self._code_executor is None else "host"),
    "data_path": self.data_path,
})
if "data_source" not in safe_data and exec_result.data_source:
    safe_data["data_source"] = exec_result.data_source
def _finite(v):
    try: v = float(v)
    except (TypeError, ValueError): return None
    return v if v == v and v not in (float('inf'), float('-inf')) else None
p_value, effect_size = _finite(p_value), _finite(effect_size)
```

(7) Replace 1630-1643 with one session and no swallowed exception:

```python
result_id = str(uuid4())
with get_session() as session:
    create_result(session, id=result_id, experiment_id=protocol_id, data=safe_data,
                  p_value=p_value, effect_size=effect_size, statistical_tests=safe_stats)
    db_exp = update_experiment_status(
        session, protocol_id,
        ExperimentStatus.COMPLETED if success else ExperimentStatus.FAILED,
        error_message=None if success else f"{exec_result.error_type}: {exec_result.error}",
        execution_time_seconds=exec_result.execution_time or None)
    db_exp.code_generated = code
```

get_session commits on exit (kosmos/db/__init__.py:109-137). A DB failure now reaches the outer except at 1668 and no phantom result id is added. (8) Replace 1651-1653 with:

```python
with self._research_plan_context():
    self.research_plan.add_result(result_id)
    if success:
        self.research_plan.mark_experiment_complete(protocol_id)
    elif protocol_id in self.research_plan.experiment_queue:
        self.research_plan.experiment_queue.remove(protocol_id)
```

Failed experiments leave the queue but are not counted in completed_experiments, which _should_check_convergence (2769-2774), kosmos/core/convergence.py:335, the FDR correction (1305) and the CLI (kosmos/cli/commands/run.py:436-445) read. (9) Keep 1656-1666: graph persistence and the transition to ANALYZING; EXECUTING may only go to ANALYZING, ERROR or PAUSED (kosmos/core/workflow.py:193-197). Make the log at 1645 include success and error_type.

**Acceptance test.** New file tests/unit/agents/test_research_director_execute.py. Fixture: copy the mock_director patches from tests/unit/agents/test_research_director_loops.py:26-33 (get_client, get_world_model, SkillLoader, kosmos.db.init_from_config); before constructing the director call `kosmos.db.init_database("sqlite:///:memory:")` (kosmos/db/__init__.py:26-27, SQLite branch 72-78) and reset_database() in teardown; keep the real ResearchPlan; set `director.workflow = MagicMock()`. Seed with operations.create_hypothesis and `operations.create_experiment(session, id=exp_id, hypothesis_id=h_id, experiment_type="computational", description="d", protocol=proto.to_dict(), domain="climate")` where proto is the ttest_protocol object from tests/unit/execution/test_code_generator.py:31-62 (the designer stores protocol.to_dict(), kosmos/agents/experiment_designer.py:897; if `ExperimentProtocol.model_validate(proto.to_dict())` fails, store `proto.model_dump(mode="json")` and report the round-trip as a defect). Set `director._code_generator = Mock(generate=Mock(return_value="results = {}"))`, `director.data_path = "/x/data.csv"`, plan.add_hypothesis(h_id), plan.add_experiment(exp_id). (a) `director._code_executor = Mock(use_sandbox=True)` with execute_with_data returning `ExecutionResult(success=True, return_value={"p_value": np.float64(0.01), "effect_size": 0.9, "data_source": "file", "n_samples": 64}, execution_time=1.2)`; `await director._handle_execute_experiment_action(exp_id)`; assert one Result row with data["execution_success"] is True, data["data_source"] == "file", p_value == 0.01; Experiment.status == ExperimentStatus.COMPLETED and code_generated == "results = {}"; plan.completed_experiments == [exp_id]; plan.results == [row.id]; workflow.transition_to called with WorkflowState.ANALYZING. (b) `ExecutionResult(success=False, error="Container exited with code 1", error_type="ExecutionError", stderr="Traceback ... KeyError: 'group'")`: row data["execution_success"] is False, p_value is None, stderr_tail contains KeyError; experiment FAILED with error_message starting "ExecutionError:"; plan.completed_experiments == []; exp_id not in plan.experiment_queue; transition ANALYZING. (c) `director._code_executor = None` and `patch("kosmos.execution.executor.CodeExecutor", side_effect=RuntimeError("Docker not available"))`: a row with error_type == "SandboxUnavailable" and no exception escapes. (d) `patch("kosmos.db.operations.create_result", side_effect=RuntimeError("db down"))` and `director._handle_error_with_recovery = Mock()`: recovery called once; plan.results == []. (e) return_value={"p_value": float("nan")}: stored p_value is None and data["p_value"] is None.

**Dependencies.** None at unit level; end to end needs P0-1, P0-2, P0-3.

**Risk and blast radius.** research_director.py has risk 0.88 and 20 hotfixes (docs/DEEP_ONBOARD.md:1996,2001); this handler is the only writer of result rows on the CLI path. evaluation/scientific_evaluation.py counts mark_experiment_complete calls, so after P0-4 that count excludes failed executions, which is the honest number. The CLI results table (kosmos/cli/commands/run.py:436-445) lists only completed experiments.

**Effort.** 3 h.

### P0-5: TTest and Correlation templates self-contained

**Goal.** Make the TTest and Correlation templates runnable in the sandbox and syntax-safe, and fix the Correlation `{method}` NameError.

**Files.** kosmos/execution/code_generator.py, TTestComparisonCodeTemplate.generate (71-190) and CorrelationAnalysisCodeTemplate.generate (216-340); tests/unit/execution/test_code_generator.py:309-310,319-320,499-500.

**Current.** TTest imports DataAnalyzer (107), calls analyzer.ttest_comparison (147-152), imports PublicationVisualizer (161) with dead visualization code at 168-176 and `title='{protocol.name}'` (172). Correlation imports DataAnalyzer (241), calls analyzer.correlation_analysis (274-278), emits `{x_var}_data` identifiers (257-259) and `'{x_var}'` literals (259,266,282-283); the print at 306 contains `{method}` without an f prefix on the generator string, so the generated f-string references an undefined `method` and raises NameError at runtime; dead visualization code at 313-326. With P0-3 both templates read the real CSV and raise KeyError on LLM-chosen column names, which P0-4 stores honestly.

**Required.** TTest: delete 107 and 160-176; replace 146-152 by inlining DataAnalyzer.ttest_comparison (kosmos/execution/data_analysis.py:43-150) with numpy and scipy.stats only, keeping the keys t_statistic, p_value, mean_difference and significance_label and adding effect_size (Cohen's d with pooled SD) and test; honor the log_transform expression at 151 by emitting the transform conditionally at generation time; emit all column and group names with `!r` (for example `df[{group_var!r}] == {groups[1]!r}`) at 110, 126-127, 135, 165-166 and 173; raise ValueError naming the columns when either group has fewer than 2 rows; keep 178-187. Correlation: delete 241 and 312-326; replace 273-278 with an inline block using stats.pearsonr or stats.spearmanr according to `{method}` plus stats.linregress, producing the keys of kosmos/execution/data_analysis.py:153-236 (correlation, p_value, r_squared, slope, intercept, std_err, significance, n_samples, equation, method) and `result['effect_size'] = result['correlation']`; replace 257-259 with `_x_syn` and `_y_syn` and `{x_var!r}` and `{y_var!r}` keys; `!r` at 266 and 282-283; fix 306 to `f"print(f\"Correlation ({method}): {{result['correlation']:.4f}}\")"`. In tests/unit/execution/test_code_generator.py change lines 309 and 499 to `assert "kosmos" not in code`, lines 310 and 500 to `assert "ttest_ind" in code`, line 319 to `assert "kosmos" not in code`, and line 320 to `assert "pearsonr" in code`.

**Acceptance test.** tests/unit/execution/test_code_generator.py: (a) exec the TTest code against a 20-row CSV with columns group and measurement (fixture names at 60-61) and group values control and experimental (fixture at 85-86); assert results["p_value"] is a float, results["data_source"] == "file", "effect_size" in results. (b) exec the Correlation code against a 30-row CSV with columns x and y; assert results["correlation"] is a float, results["method"] == "pearson", and the line "Correlation (pearson)" appears in captured stdout. (c) A protocol with variables keyed "CO2 concentration" and "it's temp" and name "O'Brien's test" generates without ValueError from _validate_syntax.

**Dependencies.** None.

**Risk and blast radius.** Same file as P0-1; DataAnalyzer keeps its direct callers and tests; the result keys consumed by kosmos/agents/research_director.py:1595-1597 are preserved.

**Effort.** 3 h.

### P0-6: LogLog and ML templates self-contained

**Goal.** Make the LogLog and ML templates runnable in the sandbox and syntax-safe.

**Files.** kosmos/execution/code_generator.py, LogLogScalingCodeTemplate.generate (362-441) and MLExperimentCodeTemplate.generate (460-544); tests/unit/execution/test_code_generator.py:328-329,337,512.

**Current.** LogLog imports DataAnalyzer and DataCleaner (378), uses DataCleaner.filter_positive (400) and analyzer.log_log_scaling_analysis (403-404), emits `{x_var}_data` (394-396), dead visualization at 413-427. ML imports MLAnalyzer (472), calls analyzer.run_experiment (496-506) whose result contains `'model': pipeline` (kosmos/execution/ml_experiments.py:477) and a Timestamp, neither JSON serializable; dead visualization at 513-530; treats the last column as the target (492-493).

**Required.** LogLog: delete 378 and 413-427; replace 400 with `df = df[(df[{x_var!r}] > 0) & (df[{y_var!r}] > 0)]`; replace 402-404 by inlining kosmos/execution/data_analysis.py:239-316 with numpy and scipy only (log10 both axes, stats.linregress for slope and intercept, stats.spearmanr on the raw values), producing the keys spearman_rho, p_value, power_law_exponent, power_law_coefficient, r_squared, equation, n_samples, log_log_slope and log_log_intercept plus effect_size = spearman_rho; replace 394-396 with `_x_syn` and `_y_syn` and `!r` keys; `!r` at 420-423. ML: delete 472 and 513-530; replace 495-506 with sklearn only: `Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])`, `train_test_split(test_size=0.2, random_state=42)`, accuracy_score, `f1_score(average="macro")`, `cross_val_score(cv=5)`; `results = {"train_test_results": {"accuracy": ..., "f1_score": ...}, "cv_results": {"mean_score": ..., "std_score": ...}, "n_features": ..., "train_size": ..., "test_size": ..., "task_type": "classification"}` with no model object and no Timestamp; keep the prints at 508-511. A continuous last column raises sklearn's "Unknown label type", which P0-4 stores honestly. In tests/unit/execution/test_code_generator.py change line 328 to `assert "kosmos" not in code`, line 329 to `assert "log10" in code`, and lines 337 and 512 to `assert "LogisticRegression" in code and "kosmos" not in code`.

**Acceptance test.** tests/unit/execution/test_code_generator.py: exec the LogLog code against a 30-row CSV with columns x and y where y = 2 * x ** 0.75 and assert `abs(results["power_law_exponent"] - 0.75) < 0.05`; exec the ML code against a make_classification-shaped CSV (10 feature columns plus a binary target, 60 rows) and assert `0 <= results["train_test_results"]["accuracy"] <= 1`.

**Dependencies.** None.

**Risk and blast radius.** As P0-5; MLAnalyzer keeps tests/unit/execution/test_ml_experiments.py.

**Effort.** 2.5 h.

**End-of-P0 check (needs Docker and the sandbox image).** With the image built (DockerSandbox._verify_image builds it on first use, kosmos/execution/sandbox.py:127-161), `kosmos run "Does CO2 concentration predict temperature anomaly?" --data-path evaluation/data/climate_co2_temperature_test.csv --max-iterations 1` produces a results row with data.execution_success = 1, data.data_source = 'file', a finite p_value, and experiments.status = 'COMPLETED'. The Generic template still correlates numeric_cols[0] against numeric_cols[1] (kosmos/execution/code_generator.py:631-636), which is year against co2_ppm on this CSV; choosing columns from the hypothesis is P2-1. This step makes a paid DeepSeek call and needs owner authorization.

### P1-1: Verdict and hypothesis status persisted

**Goal.** Persist the analysis verdict and hypothesis status, handle failed executions without an LLM call, and remove the `None < 0.05` crash.

**Files.** kosmos/agents/research_director.py _handle_analyze_result_action (1677-1805) and a new helper _db_result_to_experiment_result; kosmos/db/operations.py (add update_result_analysis after get_result, 374-391; existing update_hypothesis_status at 175-191); kosmos/agents/data_analyst.py _create_fallback_interpretation (548-569).

**Current.** Lines 1715-1733 build an ExperimentResult with status=ResultStatus.SUCCESS always (1719) and supports_hypothesis=db_result.supports_hypothesis, which is always NULL; 1748-1751 call the analyst; the verdict goes only to the in-memory plan (1771-1778) and the graph (1781-1789). The result row is never updated, so _apply_multiple_comparison_correction (1309-1313) and the refiner (kosmos/hypothesis/refiner.py:155-167) read NULL and always take the SPAWN_VARIANT branch. The Result columns interpretation, key_findings and supports_hypothesis exist (kosmos/db/models.py:126-128). kosmos/agents/data_analyst.py:560-561 evaluates `result.primary_p_value < 0.05` with None on the fallback path, which is reached from kosmos/agents/data_analyst.py:376-379,543-546.

**Required.** (1) After kosmos/db/operations.py:391 add:

```python
def update_result_analysis(session, result_id, supports_hypothesis=None, interpretation=None, key_findings=None) -> Result:
    result = get_result(session, result_id)
    if not result:
        raise ValueError(f"Result {result_id} not found")
    result.supports_hypothesis = supports_hypothesis
    if interpretation is not None:
        result.interpretation = interpretation
    if key_findings is not None:
        _validate_json_list(key_findings, "key_findings", required=False)
        result.key_findings = key_findings
    session.commit(); session.refresh(result)
    return result
```

(2) Director: add `def _db_result_to_experiment_result(self, db_r)` containing the body of 1712-1733 with `status=ResultStatus.FAILED if (db_r.data or {}).get("execution_success") is False else ResultStatus.SUCCESS`; use it at 1715-1733 and at 1858-1881 in the refine handler. (3) In the analyze handler replace 1747-1760 with:

```python
if pydantic_result.status == ResultStatus.FAILED:
    _d = pydantic_result.raw_data
    hypothesis_supported, confidence = None, 0.0
    summary = f"Execution failed: {_d.get('error_type')}: {_d.get('error')}"
    key_findings = []
else:
    interpretation = self._data_analyst.interpret_results(result=pydantic_result, hypothesis=pydantic_hyp)
    hypothesis_supported = interpretation.hypothesis_supported
    confidence = interpretation.confidence if interpretation.confidence else 0.8
    summary, key_findings = interpretation.summary, list(interpretation.key_findings or [])
self.rollout_tracker.increment("data_analysis")
p_value, effect_size = pydantic_result.primary_p_value, pydantic_result.primary_effect_size
from kosmos.db.operations import update_result_analysis, update_hypothesis_status
from kosmos.db.models import HypothesisStatus as DBHypothesisStatus
with get_session() as session:
    update_result_analysis(session, result_id, supports_hypothesis=hypothesis_supported,
                           interpretation=summary, key_findings=key_findings)
    if hypothesis_id:
        update_hypothesis_status(session, hypothesis_id,
            DBHypothesisStatus.SUPPORTED if hypothesis_supported is True
            else DBHypothesisStatus.REJECTED if hypothesis_supported is False
            else DBHypothesisStatus.INCONCLUSIVE)
```

Use the DB enum (kosmos/db/models.py:31-37), not the kosmos.models.hypothesis.HypothesisStatus imported at 36. Keep 1762-1796. (4) kosmos/agents/data_analyst.py:560-561: compute `_p = result.primary_p_value` and emit 'significant' if `_p is not None and _p < 0.05`, else 'non-significant' if `_p is not None`, else 'undetermined'.

**Acceptance test.** New file tests/unit/agents/test_research_director_analyze.py using the P0-4 fixture. Seed a hypothesis, an experiment, and `create_result(..., data={"execution_success": True, "p_value": 0.01}, p_value=0.01, effect_size=0.9)`; `director._data_analyst = Mock(interpret_results=Mock(return_value=ResultInterpretation(experiment_id=exp_id, hypothesis_supported=True, confidence=0.9, summary="S", key_findings=["k"], significance_interpretation="", biological_significance=None, comparison_to_prior_work=None, potential_confounds=[], follow_up_experiments=[], anomalies_detected=[], patterns_detected=[], overall_assessment="")))`; `await director._handle_analyze_result_action(result_id)`; assert the DB Result has supports_hypothesis True, interpretation == "S", key_findings == ["k"]; Hypothesis.status == HypothesisStatus.SUPPORTED (DB enum); h_id in plan.supported_hypotheses; transition_to called with REFINING. Second test with data={"execution_success": False, "error_type": "ExecutionError", "error": "exit 1"}: interpret_results not called, supports_hypothesis None, status INCONCLUSIVE, h_id in plan.tested_hypotheses, interpretation starts with "Execution failed". Third test (unit, `patch("kosmos.agents.data_analyst.get_client")`): `DataAnalystAgent()._create_fallback_interpretation(ExperimentResult(... primary_p_value=None ...))` returns without TypeError.

**Dependencies.** P0-4 (writes execution_success).

**Risk and blast radius.** ExperimentResult is a critical model with blast radius 98 (docs/DEEP_ONBOARD.md:2048) but is only constructed here; the new operations.py function is additive; the refiner now sees real True and False values and takes the REFINE and RETIRE branches that were unreachable; tests/unit/hypothesis/ is unaffected.

**Effort.** 3 h.

### P1-2: Leave REFINING after every refinement pass

**Goal.** Leave REFINING after every refinement pass.

**Files.** kosmos/agents/research_director.py _handle_refine_hypothesis_action (1807-1990); new method _leave_refining.

**Current.** The handler returns early at 1840 and 1894 and ends at 1978 without any transition_to; decide_next_action in REFINING returns REFINE_HYPOTHESIS whenever tested_hypotheses is non-empty (2549-2554), and the REFINE branch (2706-2723) increments the iteration on each pass, so runs spend every remaining action refining (persona 004 v007: 100 actions, 1 experiment, evaluation/personas/runs/004_climate_data_scientist/v007_20260209/tier1_scaled/EVALUATION_REPORT.md:62-74). Legal targets from REFINING are GENERATING_HYPOTHESES, DESIGNING_EXPERIMENTS, CONVERGED, PAUSED and ERROR (kosmos/core/workflow.py:204-210).

**Required.** Add:

```python
def _leave_refining(self) -> None:
    """Exit REFINING: DESIGNING_EXPERIMENTS if untested hypotheses remain, else GENERATING_HYPOTHESES (workflow.py:204-210)."""
    with self._research_plan_context():
        untested = self.research_plan.get_untested_hypotheses()
    target = WorkflowState.DESIGNING_EXPERIMENTS if untested else WorkflowState.GENERATING_HYPOTHESES
    with self._workflow_context():
        if self.workflow.current_state == WorkflowState.REFINING:
            self.workflow.transition_to(target, action=f"Refinement complete; {len(untested)} untested hypotheses")
```

Call `self._leave_refining()` immediately before the returns at 1840 and 1894, and after 1978. The `current_state == REFINING` guard keeps the recovery paths that return REFINE_HYPOTHESIS from EXECUTING and ANALYZING (2535, 2546) unchanged. Do not change decide_next_action (2549-2554); P2-7 does.

**Acceptance test.** tests/unit/agents/test_research_director_loops.py with the mock_director fixture (23-55) and a real ResearchPlan substituted (`director.research_plan = ResearchPlan(research_question="q", max_iterations=10)`): (a) workflow.current_state = REFINING, plan.add_hypothesis("h1"), plan.mark_tested("h1"), patch kosmos.agents.research_director.get_session to yield a session and patch kosmos.db.operations.get_hypothesis to return None; `await director._handle_refine_hypothesis_action("h1")`; transition_to called with GENERATING_HYPOTHESES. (b) Add untested "h2": DESIGNING_EXPERIMENTS. (c) workflow.current_state = EXECUTING: transition_to not called. (d) Loop-closure test with `director._hypothesis_refiner = Mock(evaluate_hypothesis_status=Mock(return_value=RetirementDecision.CONTINUE_TESTING))` and a seeded in-memory DB (P0-4 fixture): after the handler, decide_next_action() with a real ResearchWorkflow in REFINING no longer returns REFINE_HYPOTHESIS.

**Dependencies.** None; P1-1 makes the refiner's decisions meaningful.

**Risk and blast radius.** test_decide_next_action_refine_hypothesis (tests/unit/agents/test_research_director.py:512-527) tests decide_next_action, which is unchanged, and is skipped without ANTHROPIC_API_KEY (tests/unit/agents/test_research_director.py:20-26). Behavioral consequence: after REFINING to GENERATING_HYPOTHESES with no untested hypotheses and an empty queue, _should_check_convergence (2797-2800) returns True on the next step, so the run converges instead of generating a second batch; another round is the P2-7 policy change.

**Effort.** 1.5 h.

### P1-3: Error recovery never blocks or raises from the event-loop thread

**Goal.** Error recovery never blocks or raises from the event-loop thread, so the 3-strike breaker, the ERROR state and ERROR_RECOVERY become reachable and the CLI stops printing an empty "Research failed:".

**Files.** kosmos/agents/research_director.py _handle_error_with_recovery (599-695, backoff at 664-688).

**Current.** All handlers are coroutines run by `asyncio.run(run_with_progress_async(...))` (kosmos/cli/commands/run.py:196-202, steps at 306 and 379), so asyncio.get_running_loop() at 677 succeeds; `run_coroutine_threadsafe(asyncio.sleep(b), loop).result(timeout=b+5)` (679-682) blocks the loop that must run the sleep, times out, and raises TimeoutError, which `except RuntimeError` (683) does not catch. The exception escapes the handler's except (for example 1668-1675), _do_execute_action and execute (2884-2925), and lands at kosmos/cli/commands/run.py:228-229 with an empty message. Lines 649-662 (breaker to ERROR) and 2732-2744 (ERROR_RECOVERY) are unreachable in practice.

**Required.** Replace 674-685 with:

```python
try:
    asyncio.get_running_loop()
    in_loop = True
except RuntimeError:
    in_loop = False
if in_loop:
    logger.info(f"{ERROR_RECOVERY_LOG_PREFIX} Inside event loop; skipping blocking backoff of {backoff_seconds}s")
else:
    time.sleep(backoff_seconds)
```

Keep 687-688 (`return self.decide_next_action()`); the message-bus handlers at 726-955 use that return value. The CLI step loop already yields between steps (kosmos/cli/commands/run.py:399).

**Acceptance test.** tests/unit/agents/test_research_director_loops.py with mock_director: (a) async test: workflow.current_state = EXECUTING; `t0 = time.monotonic()`; `_handle_error_with_recovery("CodeExecutor", "boom", recoverable=True)`; no exception, elapsed under 1.0 s, `_consecutive_errors == 1`. (b) Call it MAX_CONSECUTIVE_ERRORS (3, line 45) times: transition_to called with WorkflowState.ERROR and the last call returns NextAction.ERROR_RECOVERY. (c) workflow.current_state = ERROR; research_plan.hypothesis_pool = ["h1"]; get_untested_hypotheses returns ["h1"]; `decide_next_action() == NextAction.ERROR_RECOVERY`; `await _execute_next_action(NextAction.ERROR_RECOVERY)`; transition_to called with GENERATING_HYPOTHESES and `_consecutive_errors == 0`. (d) Sync path from a plain `def test_`: elapsed at least ERROR_BACKOFF_SECONDS[0] (2 s), or patch time.sleep and assert it was called with 2.

**Dependencies.** None.

**Risk and blast radius.** 11 call sites, signatures unchanged; removes the only path by which `kosmos run` aborts on a handler exception; later exceptions flow to the breaker.

**Effort.** 1 h.

### P1-4: Feed the convergence detector real hypotheses and results

**Goal.** Feed the convergence detector real hypotheses and results.

**Files.** kosmos/agents/research_director.py _check_convergence_direct (1232-1287); uses _db_result_to_experiment_result from P1-1 and get_results_for_experiment (kosmos/db/operations.py:394-410).

**Current.** Line 1253 imports HypothesisModel from kosmos.db.models, which names the ORM class Hypothesis (kosmos/db/models.py:75); the ImportError is swallowed at 1273-1274, so hypotheses == []; results (1248) is never filled. ConvergenceDetector.check_convergence(research_plan, hypotheses, results, total_cost) (kosmos/core/convergence.py:221-225) computes discovery rate and consistency from r.supports_hypothesis (kosmos/core/convergence.py:479,541,557), so the optional criteria are inert.

**Required.** Replace 1253 with `from kosmos.db.models import Hypothesis as HypothesisModel`. Wrap each Hypothesis(...) construction at 1264-1270 in try/except Exception (Pydantic enforces statement min 10 and rationale min 20, kosmos/models/hypothesis.py:52-53) and skip rows that fail. Inside the same `with get_session() as session:` block, after the hypotheses, add:

```python
from kosmos.db.operations import get_results_for_experiment
for exp_id in list(self.research_plan.completed_experiments)[-100:]:
    for db_r in get_results_for_experiment(session, exp_id):
        try:
            results.append(self._db_result_to_experiment_result(db_r))
        except Exception as conv_err:
            logger.debug(f"Skipping result {db_r.id} for convergence: {conv_err}")
```

Change the logger.debug at 1274 to logger.warning.

**Acceptance test.** tests/unit/agents/test_research_director_execute.py (P0-4 fixture): seed two hypotheses, one experiment, one result with supports_hypothesis=True and data={"execution_success": True}; plan.hypothesis_pool = [h1, h2], plan.completed_experiments = [exp_id]; `director.convergence_detector = Mock(check_convergence=Mock(return_value=MagicMock(should_stop=False, reason=MagicMock(value="x"), details="")))`; call `director._check_convergence_direct()`; assert check_convergence called once with len(hypotheses) == 2, len(results) == 1, results[0].supports_hypothesis is True, results[0].status == ResultStatus.SUCCESS.

**Dependencies.** P1-1 (helper); P0-4.

**Risk and blast radius.** One caller (1358); the optional criteria novelty_decline and diminishing_returns (kosmos/core/convergence.py:194-201) become live and may stop runs earlier; the mandatory criteria are unchanged.

**Effort.** 1.5 h.

### P1-5: `--budget` arms enforcement, provider calls recorded, per-model pricing

**Goal.** `--budget` arms enforcement, every provider call is recorded in MetricsCollector, and cost is priced by the model actually used.

**Files.** kosmos/agents/research_director.py __init__ (after line 101); kosmos/core/providers/base.py LLMProvider._update_usage_stats (391-402); kosmos/core/metrics.py _calculate_period_cost (747-762).

**Current.** kosmos/cli/commands/run.py:144-145 stores the flag in config_obj.research.budget_usd and line 161 copies it to flat_config["budget_usd"]; nothing reads it. Enforcement at kosmos/agents/research_director.py:2422-2440 runs only when get_metrics().budget_enabled is True (default False, kosmos/core/metrics.py:157), which only configure_budget (kosmos/core/metrics.py:553-592) sets, and nothing in production calls it. record_api_call (kosmos/core/metrics.py:176-215) has no production caller. Providers track cost in _update_usage_stats (kosmos/core/providers/base.py:391-402); LiteLLM calls it at kosmos/core/providers/litellm_provider.py:195 with UsageStats.model set at 190. _calculate_period_cost prices everything as claude-sonnet-4-5 (kosmos/core/metrics.py:762), overstating DeepSeek (kosmos/core/pricing.py:35) by about 20 times.

**Required.** (1) Director __init__ after 101:

```python
budget_usd = self.config.get("budget_usd")
if budget_usd:
    from kosmos.core.metrics import get_metrics
    get_metrics().configure_budget(limit_usd=float(budget_usd))
    logger.info(f"[BUDGET] Enforcement armed at ${float(budget_usd):.2f}")
```

(2) After kosmos/core/providers/base.py:402 inside _update_usage_stats add:

```python
try:
    from kosmos.core.metrics import get_metrics
    get_metrics().record_api_call(
        model=usage.model or getattr(self, "model", "unknown"),
        input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
        duration_seconds=0.0, success=True)
except Exception as e:
    logger.debug(f"Metrics recording skipped: {e}")
```

(3) Replace kosmos/core/metrics.py:757-762 with:

```python
return sum(
    get_model_cost(call.get("model") or "claude-sonnet-4-5",
                   call.get("input_tokens", 0), call.get("output_tokens", 0))
    for call in period_calls)
```

record_api_call already stores "model" per call (kosmos/core/metrics.py:207). tests/e2e/test_budget_enforcement.py:33-59 records claude-3-5-sonnet-20241022, priced (3.0, 15.0) in MODEL_PRICING (kosmos/core/pricing.py:20), identical to the previous default, so its numbers do not change.

**Acceptance test.** New file tests/unit/core/test_budget_wiring.py; call get_metrics(reset=True) before and after each test (singleton at kosmos/core/metrics.py:922-935). (a) A director with the mock_director patches and config={"max_iterations": 1, "budget_usd": 1.0}: get_metrics().budget_enabled is True and budget_limit_usd == 1.0; with no budget_usd: False. (b) `class _P(LLMProvider)` implementing the abstract methods (kosmos/core/providers/base.py:197,226,255,280,365) with pass; `_P(...)._update_usage_stats(UsageStats(input_tokens=1000, output_tokens=500, total_tokens=1500, cost_usd=0.0, model="deepseek/deepseek-chat"))`; get_metrics().api_calls == 1 and `_calculate_period_cost() == pytest.approx(get_model_cost("deepseek/deepseek-chat", 1000, 500))`. (c) mock_director with workflow.current_state = GENERATING_HYPOTHESES; `configure_budget(limit_usd=0.0001)`; `record_api_call("claude-sonnet-4-5", 10000, 10000, 1.0)`; `decide_next_action() == NextAction.CONVERGE` and transition_to called with WorkflowState.CONVERGED.

**Dependencies.** None.

**Risk and blast radius.** providers/base.py has blast radius 124 (docs/DEEP_ONBOARD.md:1991,2012); the change is additive inside try/except and does not alter generate() signatures or LLMResponse; the legacy ClaudeClient in kosmos/core/llm.py does not pass through _update_usage_stats, the owner's LiteLLM path does. Changing _calculate_period_cost affects check_budget (kosmos/core/metrics.py:647) and enforce_budget (kosmos/core/metrics.py:785-805); the e2e budget tests keep their numbers.

**Effort.** 1.5 h.

### P2-0: Result columns for execution, validation, provenance and cost

**Goal.** Extend the results table and create_result so every downstream P2 change has a place to store execution, validation, provenance and cost fields.

**Files.** kosmos/db/models.py:110-145 Result (columns data, statistical_tests, interpretation, key_findings, supports_hypothesis, p_value, effect_size, confidence_interval, figures, created_at) and 40-72 Experiment (code_generated at 58 and error_message at 60, unused before P0-4); kosmos/db/operations.py:339-372 create_result; alembic/versions/ (current head dc24ead48293_add_profiling_tables.py); kosmos/db/__init__.py:100,201 (create_all, so fresh databases need no migration; the owner's existing kosmos.db does).

**Current.** Result rows carry statistics only; there are no columns for success, source, seed, provenance, validation, cost or run id. After P0-4, execution_success and data_source live inside the data JSON.

**Required.** Add nullable columns to Result: run_id String, execution_success Boolean, data_source String, random_seed Integer, provenance JSON, validation_status String (one of validated, rejected, unvalidated, rejected_unsafe), validation_detail JSON (null-model result, ScholarEval score, recomputation check), cost_usd Float, error_message Text. Extend create_result with the same optional keyword arguments plus `code: Optional[str]` that writes Experiment.code_generated for experiment_id. Add one alembic revision add_result_provenance_columns with down_revision 'dc24ead48293' whose upgrade also backfills execution_success and data_source from the data JSON of existing rows. Change the P0-4 handler to pass `execution_success=success` and `data_source=safe_data.get("data_source")` to create_result in addition to the JSON keys. Add `get_results_for_run(session, run_id)` and `update_result_validation(session, result_id, status, detail, supports_hypothesis)` to kosmos/db/operations.py.

**Acceptance test.** tests/unit/db/test_database.py with in-memory SQLite: `create_result(..., execution_success=False, data_source='file', run_id='r1', provenance={'git_sha': 'abc'}, cost_usd=0.001)` round-trips every field; `get_results_for_run(session, 'r1')` returns 1 row; update_result_validation changes validation_status and supports_hypothesis. The P0-4 test (a) additionally asserts the row's execution_success column is True and data_source column is 'file'.

**Dependencies.** P0, P1.

**Risk and blast radius.** Low; adding columns with defaults is listed safe (docs/DEEP_ONBOARD.md:2122); create_result has one production caller (kosmos/agents/research_director.py:1633-1641).

**Effort.** 3 h.

### P2-1: Dataset schema and variable-to-column binding

**Goal.** Bind every protocol variable to an actual CSV column at design time, and make the code generator use those bindings instead of positional numeric columns or LLM-invented names.

**Files.** kosmos/execution/code_generator.py:631-636 (Generic takes numeric_cols[0] and [1]), 568-570 (variable names computed but unused), 74-78 (TTest), 219-221 (Correlation), 364-366 (LogLog), 492-493 (ML takes the last column as target), 797-833 generate, 858-905 _create_code_generation_prompt (no column list; line 900 already replaced by P0-1). kosmos/agents/experiment_designer.py:164-299 design_experiment (no dataset input), 380-407 _select_experiment_type (unknown domains default to COMPUTATIONAL), 409-439 _generate_from_template, 441-516 _generate_with_claude (prompt at 452-460; schema at 463-490 has "variables": {"type": "object"} with no column field), 542-565 variable parsing, 589-602 placeholder defaults independent_var and dependent_var. kosmos/experiments/templates/data_analysis.py:90-92 (literal variables outcome_variable and group) and 329-342 (variable_x and variable_y); kosmos/experiments/templates/computational.py:173-179 (the generic computational template is_applicable returns True for every hypothesis) and 194-195. kosmos/core/prompts.py:202-497 EXPERIMENT_DESIGNER (variables list at 494 has no dataset slot; schema example at 252-260). kosmos/models/experiment.py:42-70 Variable (no column field) and 471-573 to_dict. kosmos/execution/data_provider.py:310-396 DataProvider.get_data (suffix dispatch at 360-372). kosmos/agents/research_director.py:1469-1530 _handle_design_experiment_action (design_experiment(hypothesis_id=..., store_in_db=True) at 1485-1488 with no data context), 1402-1467 (generate_hypotheses at 1419-1424), 101 (self.data_path). kosmos/core/workflow.py:149-151 get_untested_hypotheses.

**Current.** On the director path the domain climate_science is not in the defaults map, so the type is COMPUTATIONAL; the always-applicable generic protocol template supplies placeholder variables; the generic code template ignores them and correlates the first two numeric columns (year and co2_ppm on the climate CSV). On the LLM protocol path variables are free-text names that the templates index with df['<name>'], producing KeyError on real data.

**Required.** (1) New module kosmos/execution/data_schema.py with `@dataclass DatasetSchema(path, sha256, n_rows, columns, numeric_columns, categorical_columns: Dict[str, List[str]] (levels, at most 10), dtypes)`, `describe_dataset(path) -> DatasetSchema` (same suffix dispatch as DataProvider.get_data), and `to_prompt_block()` listing each column with dtype, example values and categorical levels. (2) `Variable.column: Optional[str] = None` in kosmos/models/experiment.py; include it in ExperimentProtocol.to_dict() (hand-written at 471-573; add the key under each variable). (3) `design_experiment(..., dataset_schema: Optional[DatasetSchema] = None)`: when given, skip _generate_from_template and call _generate_with_claude; add dataset_context to EXPERIMENT_DESIGNER.variables and body with the text "Every independent and dependent variable MUST include \"column\": \"<exact column name from the dataset>\"; do not invent columns"; add `"column": {"type": "string"}` to the variable schema; in _parse_claude_protocol read var_data.get("column") and resolve it against schema.columns in this order: exact, case-insensitive, normalized (`re.sub(r'[^a-z0-9]', '', s.lower())`), then `difflib.get_close_matches(cutoff=0.8)`; store the match in Variable.column. If any INDEPENDENT or DEPENDENT variable is unbound raise `UnboundVariableError(hypothesis_id, unbound_names, available_columns)`, a new exception in kosmos/agents/experiment_designer.py. No synthetic fallback when a dataset was supplied. Without a schema, behavior is unchanged. (4) Director: `self.dataset_schema = describe_dataset(self.data_path)` once in __init__ when data_path is set; pass it to design_experiment; on UnboundVariableError append the id to a new ResearchPlan.untestable_hypotheses and (owner question 1) either call update_hypothesis_status(REJECTED) or leave the status and exclude it from get_untested_hypotheses (change kosmos/core/workflow.py:149-151 to exclude untestable_hypotheses). Also pass `dataset_context=schema.to_prompt_block()` into hypothesis generation by appending it to the research_question handed to HYPOTHESIS_GENERATOR.format (kosmos/agents/hypothesis_generator.py:360 region). (5) Code generator: every template reads `col = var.column or var.name`. TTest: group_col and measure_col from the bound INDEPENDENT (categorical, 2 levels) and DEPENDENT; Correlation and LogLog: the bound INDEPENDENT as x and DEPENDENT as y; Generic: replace 631-636 with the bound x and y, correlation when both are numeric, Welch t-test when x is categorical with exactly 2 levels, one-way ANOVA with 3 to 10 levels; emit test_type ('pearson_correlation', 'welch_t_test' or 'one_way_anova'), 'statistic', 'p_value', 'effect_size', 'n' and `'columns': {'x': ..., 'y': ...}`. Every template prepends after `df = pd.read_csv(data_path)`: `missing = [c for c in [<bound cols>] if c not in df.columns]; if missing: raise KeyError(f"Dataset is missing required columns: {missing}; available: {list(df.columns)}")`. generate raises ValueError("protocol has unbound variables") when a data_path binding exists but any used variable has column None. Replace the former line 900 instruction with the column list through a new `generate(protocol, dataset_schema=None)` parameter. (6) The data_path override and the `RESULT:` footer are P0.

**Acceptance test.** New file tests/unit/execution/test_column_binding.py: (a) `describe_dataset(climate_csv)` returns 8 columns, 7 numeric, decade categorical with 7 levels, 64 rows, a 64-character sha256. (b) With get_client mocked to return a structured protocol whose variables are `{"co2": {"type":"independent","column":"CO2_ppm"}, "temp": {"type":"dependent","column":"temp_anomaly_c"}}`, `design_experiment(hypothesis, dataset_schema=schema)` yields Variable.column == 'co2_ppm' and 'temp_anomaly_c'. (c) `"column": "ocean_heat"` raises UnboundVariableError listing the 8 columns. (d) `ExperimentCodeGenerator(use_llm=False).generate(protocol)` executed with `CodeExecutor(use_sandbox=False).execute_with_data(code, climate_csv)` yields results['columns'] == {'x':'co2_ppm','y':'temp_anomaly_c'}, test_type 'pearson_correlation', statistic above 0.85, p_value below 1e-6 (the correct analysis gives r 0.93). (e) decade (categorical) against temp_anomaly_c generates ANOVA code that runs with test_type 'one_way_anova'. (f) Director test: UnboundVariableError from a mocked designer leaves hypothesis_pool unchanged, adds the id to untestable_hypotheses, and get_untested_hypotheses() excludes it.

**Dependencies.** P0, P2-6.

**Risk and blast radius.** Medium. kosmos/models/experiment.py has blast radius 102 (docs/DEEP_ONBOARD.md:2047), but an optional field with a default is listed safe (docs/DEEP_ONBOARD.md:2069-2070); to_dict() is the sole database serialization path, so the key must be added there; design_experiment gains a keyword-only parameter; all 9 protocol templates become unreachable when a dataset is supplied, which is intended.

**Effort.** 14 h.

### P2-2: Recomputation, permutation null on real data, provider-agnostic ScholarEval, verdict rule

**Goal.** Gate every analyzed result through an in-process recomputation check, a real permutation null model on the actual data, and a provider-agnostic ScholarEval, so a finding is marked validated only with a real test statistic that fails on shuffled data.

**Files.** kosmos/agents/research_director.py:1677-1805 _handle_analyze_result_action (ExperimentResult at 1715-1733 with status SUCCESS unconditionally; interpret at 1748-1751; mark at 1771-1778; transition at 1792-1796; no validation). kosmos/validation/scholar_eval.py:103-125 __init__(anthropic_client=None, ...), 127-207 evaluate_finding (self.client.messages.create at 146-151; thresholds at 160-163; null model at 168-185; exception to _mock_evaluation at 204-207), 298-340 _parse_llm_response (brace slicing; neutral 0.5 fallback at 338-340), 421-492 _mock_evaluation (base score 0.78 at 434). kosmos/validation/null_model.py:135-139 STATISTIC_KEYS, 166-256 validate_finding (full permutation only when data and analysis_func are passed, 214-222; otherwise _parametric_null at 436-474, unrelated to the data), 336-369 _extract_test_statistic, 382-434 _full_permutation_test, 573-596 _determine_shuffle_method (keyed on statistics['test_type']), 260-280 shuffle_columns, 295-313 shuffle_labels. kosmos/agents/data_analyst.py:508-546 _parse_interpretation_response and 548-569 _create_fallback_interpretation. kosmos/core/llm.py:613-683 get_client.

**Current.** The director never validates; on the library loop ScholarEval with a LiteLLM client raises on .messages.create and silently approves through the mock; the null model without data runs a parametric pseudo-null.

**Required.** (1) `ScholarEvalValidator.__init__(anthropic_client=None, llm_client=None, threshold=0.75, min_rigor_score=0.70, model=..., temperature=0.3, allow_mock=False)`; in evaluate_finding, if llm_client is set call `llm_client.generate(prompt, max_tokens=1500, temperature=self._temperature)` and take .content when present else str(); keep the anthropic_client branch; parse with parse_json_response (kosmos/core/utils/json_parser.py:31-154); on any exception return `ScholarEvalScore(overall_score=0.0, passes_threshold=False, feedback=f"evaluation_error: {e}")` unless allow_mock=True (True only in scripts/smoke_test.py and the existing unit tests). Add `run_null_model: bool = True` and pass False from the director. (2) New module kosmos/validation/analysis_fn.py: `build_analysis_fn(test_type, x_col, y_col) -> Callable[[pd.DataFrame], Dict]` implementing pearson_correlation, spearman_correlation, welch_t_test, mann_whitney, one_way_anova and linear_regression with scipy, returning {'statistic','p_value','effect_size','test_type','n'}; `shuffle_target(df, test_type, x_col, y_col, rng)` permuting y_col for correlation and regression and x_col (the labels) for group tests. Same formulas as the templates (consistency test below). (3) Director, before interpret_results: `exec_ok = db_result.execution_success is True`; `from_file = db_result.data_source == 'file'`; `stats = db_result.statistical_tests or {}`; `has_stat = any(k in stats for k in STATISTIC_KEYS) and stats.get('p_value') is not None`. If exec_ok and from_file and has_stat and self.dataset_schema: `df = pd.read_csv(self.data_path)`; `fn = build_analysis_fn(stats['test_type'], stats['columns']['x'], stats['columns']['y'])`; `recomputed = fn(df)`; `recomputed_match = isclose(recomputed['statistic'], stats['statistic'], rel_tol=1e-6)`; `null = NullModelValidator(n_permutations=self.config.get('null_permutations', 500), random_seed=self.random_seed).validate_finding({'statistics': stats}, data=df, analysis_func=fn)`; `scholar = ScholarEvalValidator(llm_client=self.llm_client, run_null_model=False).evaluate_finding({'summary': interpretation.summary, 'statistics': stats, 'methods': f"{protocol_name}; {stats['test_type']} on {columns}; n={n}", 'interpretation': interpretation.significance_interpretation})`. `validated = recomputed_match and null.passes_null_test and not null.persists_in_noise and scholar.passes_threshold`. validation_status is 'validated' or 'rejected'; when the precondition fails it is 'unvalidated' with validation_detail['reason'] in {'execution_failed', 'synthetic_data', 'no_statistic', 'no_dataset'}. Persist with `update_result_validation(session, result_id, status, detail={'null_model': null.to_dict(), 'scholar_eval': scholar.to_dict(), 'recomputed': recomputed, 'recomputed_match': ...}, supports_hypothesis=verdict)`. (4) Verdict rule: `supported = interpretation.hypothesis_supported`; mark_supported only when supported is True and validation_status == 'validated'; mark_rejected only when supported is False and validation_status in ('validated', 'rejected') and exec_ok and from_file; otherwise mark_tested. (5) The ExperimentResult at 1715-1733 gets status SUCCESS if exec_ok else FAILED, and metadata.data_source and metadata.random_seed from the row. (6) Replace the brace slicing in data_analyst._parse_interpretation_response with parse_json_response.

**Acceptance test.** New file tests/unit/validation/test_director_gate.py (loops fixture, get_client mocked so generate returns ScholarEval JSON with all eight dimensions at 0.85 and analyst JSON `{"hypothesis_supported": true, "confidence": 0.9, ...}`): (a) seed a Result with execution_success=True, data_source='file', statistical_tests={'test_type':'pearson_correlation','statistic':r,'p_value':p,'columns':{'x':'co2_ppm','y':'temp_anomaly_c'}} computed by build_analysis_fn on the climate dataframe; after the handler: validation_status 'validated', null_model passes_null_test True, recomputed_match True, hypothesis in supported_hypotheses. (b) A shuffled copy of the CSV (temp_anomaly_c permuted with default_rng(0)), director.data_path pointed at it, the row computed on the shuffled file: 'rejected', hypothesis only in tested_hypotheses. (c) Statistic tampered to 0.99: recomputed_match False and rejected. (d) data == {} or execution_success=False: 'unvalidated' with reason, no mark_supported. (e) `ScholarEvalValidator(llm_client=Mock(generate=Mock(side_effect=RuntimeError)))` gives passes_threshold False. (f) Consistency: for each of the 4 CSVs and the templates' default test type, exec of the generated code equals build_analysis_fn within 1e-9. 500 permutations on 64 rows complete in under 1 s.

**Dependencies.** P0, P1, P2-0, P2-1, P2-6.

**Risk and blast radius.** Medium; additive inside the handler; the ScholarEval signature stays backward compatible; tests/unit/validation/test_scholar_eval.py must set allow_mock=True; one extra LLM call per analyzed result (about $0.001 on DeepSeek).

**Effort.** 12 h.

### P2-3: Seed and provenance

**Goal.** Propagate one run seed from the CLI through protocol, generated code and executor into the stored result, and capture provenance (git SHA, model, temperature, data hash, code hash, sandbox flag) on every result row.

**Files.** kosmos/cli/commands/run.py:51-61 options (no --seed) and 154-180 flat_config. kosmos/config.py:631-633 SafetyConfig.default_random_seed. kosmos/agents/research_director.py:68-170 __init__ (llm_client at 128), 1532-1675 execute handler (create_result at 1630-1643), 1715-1733 and 1859-1880 ExecutionMetadata built without random_seed. kosmos/agents/experiment_designer.py:675 `random_seed=data.get("random_seed")` (LLM-chosen). kosmos/execution/code_generator.py:89,231,368,572 `seed = getattr(protocol,'random_seed',42) or 42`; 485 make_classification(random_state=42); 496 MLAnalyzer(random_state=42). kosmos/execution/executor.py:632-662 execute_with_data and 113-137 ExecutionResult. kosmos/safety/reproducibility.py:127-166 set_seed. kosmos/execution/provenance.py:71-148 CodeProvenance (optional seed, git_sha, model, temperature and data_hash at 123-127; get_git_sha at 146-148; never constructed in production). kosmos/models/result.py:42 ExecutionMetadata.random_seed and 58 data_source. kosmos/db/models.py:58 Experiment.code_generated.

**Current.** Seeds are 42 by default and LLM-overridable; host-process seeding never happens on the director path; generated code is not stored; there is no provenance record.

**Required.** (1) CLI: `seed: Optional[int] = typer.Option(None, "--seed")`; `flat_config["random_seed"] = seed if seed is not None else config_obj.safety.default_random_seed`. (2) Director __init__: `self.random_seed = int(self.config.get("random_seed", 42))`; `ReproducibilityManager(default_seed=self.random_seed).set_seed(self.random_seed)`; `self.run_id = f"run_{uuid4().hex[:12]}"`; store a ResearchSession row (kosmos/db/models.py:222-250) with id=self.run_id. (3) Designer: `design_experiment(..., random_seed: Optional[int] = None)`; at kosmos/agents/experiment_designer.py:675 use random_seed if not None else data.get("random_seed"); the director passes self.random_seed. (4) Code generator: templates keep {seed} from protocol.random_seed; ML uses {seed} at kosmos/execution/code_generator.py:485,496; all templates emit `results['random_seed'] = <seed>`. (5) Executor: `execute_with_data(code, data_path, retry_on_error=False, seed: Optional[int] = None)`; when seed is given prepend `random_seed = <seed>\nimport random as _r; _r.seed(random_seed)\nimport numpy as _np; _np.random.seed(random_seed)\n` (random is already in _ALLOWED_MODULES, kosmos/execution/executor.py:86-94); ExecutionResult gains random_seed and sandbox_used. (6) kosmos/execution/provenance.py: `build_run_provenance(code, data_path, llm_client, seed, protocol, sandbox_used) -> Dict` with git_sha (get_git_sha()), model (getattr(llm_client,'model',None)), provider (getattr(llm_client,'provider_name',None)), temperature (getattr(llm_client,'temperature_default',None)), data_path, data_sha256 (file bytes), data_rows, code_sha256, code_path, template (protocol.template_name or 'llm'), seed, sandbox_used, python_version, kosmos_version (importlib.metadata.version('kosmos')) and timestamp. The director writes the code to `<artifacts_dir>/<run_id>/code/<result_id>.py` (artifacts_dir is a new config key defaulting to `<cwd>/artifacts`, owner question 8) and passes code=, provenance=, random_seed= and run_id= to create_result (P2-0). (7) Both ExecutionMetadata constructions (kosmos/agents/research_director.py:1724-1732,1871-1879) set random_seed, data_source and sandbox_used from the row.

**Acceptance test.** New file tests/unit/execution/test_seed_provenance.py: (a) `CodeExecutor(use_sandbox=False).execute_with_data(code_with_synthetic_fallback, '/nonexistent.csv', seed=7)` twice returns identical return_value; seed 8 differs. (b) build_run_provenance on the climate CSV returns data_sha256 == hashlib.sha256(open(csv,'rb').read()).hexdigest(), a 40-hex git_sha, model == 'deepseek/deepseek-chat' when llm_client.model is that string. (c) Loops fixture with mocked generator and executor: after _handle_execute_experiment_action the Result row has random_seed == 7, run_id == director.run_id, provenance['code_sha256'] == sha256(code), Experiment.code_generated == code; `artifacts/<run_id>/code/<result_id>.py` exists (tmp_path through the artifacts_dir config key).

**Dependencies.** P0, P2-0.

**Risk and blast radius.** Low to medium; execute_with_data callers are kosmos/agents/research_director.py:1579 and kosmos/execution/executor.py:1080 and the new parameter is keyword-only; the host-side set_seed touches global numpy state (no test asserts randomness after a director is constructed).

**Effort.** 8 h.

### P2-4: Honest run report and real cost

**Goal.** Make the end-of-run report and JSON export show execution success, data source, validation status, failed tasks and real LLM cost.

**Files.** kosmos/cli/commands/run.py:414-468 result assembly (hypothesis.to_dict() at 433 and experiment.to_dict() at 440 on ORM rows that define only __repr__, so the lists hold repr strings), 458-467 metrics (director.llm_client.total_requests does not exist on LLMProvider; the counter is request_count, exposed as total_requests by get_usage_stats at kosmos/core/providers/base.py:375-389; no cost), 205-211 display. kosmos/cli/views/results_viewer.py:84-115 display_hypotheses_table, 172-211 display_experiments_table, 267 display_metrics_summary. kosmos/core/providers/base.py:375-389 get_usage_stats (has total_cost_usd) and 391-402 _update_usage_stats. kosmos/core/providers/litellm_provider.py:173-205 _parse_response (cost from get_model_cost at 183; _update_usage_stats at 195). kosmos/core/metrics.py:176-215 record_api_call, 217-250 get_api_statistics (Sonnet pricing at 231-235), 747-762 _calculate_period_cost, 785-805 enforce_budget. kosmos/core/pricing.py:35-36,54. kosmos/agents/research_director.py:2942-2968 get_research_status.

**Current.** The CLI tables receive strings; the API call count reads a nonexistent attribute; cost is never shown; "experiments_executed" counts mark_experiment_complete calls regardless of success.

**Required.** (1) Extract 414-468 into `build_run_results(director, question, max_iterations) -> Dict` in a new module kosmos/cli/commands/run_results.py. Hypotheses: {'id', 'claim': statement, 'novelty_score', 'testability_score', 'priority_score': None, 'status': status.value, 'tested', 'supported', 'untestable'}. Experiments: {'id', 'type', 'status', 'created_at', 'hypothesis_id'}. A new results list from get_results_for_run(session, director.run_id): {'result_id', 'experiment_id', 'hypothesis_id', 'execution_success', 'data_source', 'test_type', 'statistic', 'p_value', 'effect_size', 'n', 'supports_hypothesis', 'validation_status', 'validation_reason', 'random_seed', 'error_message', 'cost_usd'}. Metrics: `usage = director.llm_client.get_usage_stats()` (guard with getattr), api_calls=usage['total_requests'], total_cost_usd, input_tokens, output_tokens, experiments_attempted, experiments_succeeded, experiments_failed, results_from_file, results_synthetic, findings_validated, findings_rejected, cost_per_validated_finding (None when zero validated), hypotheses_untestable. (2) kosmos/cli/views/results_viewer.py: `display_results_table(results)` with columns Result, Hyp, Exec (OK/FAIL), Data (file/synthetic/none), Test, Stat, p, Verdict, Validation; failed rows in red with the first 60 characters of error_message; extend display_metrics_summary with cost, tokens, succeeded/failed and validated counts; kosmos/cli/commands/run.py:205-211 calls the new table; export_to_json and export_to_markdown include results. (3) Per-result cost attribution: snapshot self.llm_client.total_cost_usd at the start of the design, execute and analyze handlers; accumulate the deltas into self._result_cost[result_id]; write cost_usd through update_result_validation (P2-0). (4) The provider-to-metrics bridge is P1-5; here only extend record_api_call with `cost_usd: Optional[float] = None` stored in the history record, and make _calculate_period_cost and get_api_statistics sum the stored cost_usd with a get_model_cost fallback.

**Acceptance test.** New file tests/unit/cli/test_run_results.py: loops fixture with in-memory SQLite seeded with 1 hypothesis, 1 experiment and 2 results (one execution_success=True, data_source='file', validation_status='validated'; one False, None, 'unvalidated', error_message='KeyError ...'); `llm_client = Mock(get_usage_stats=lambda: {'total_requests': 9, 'total_cost_usd': 0.0123, ...}, total_cost_usd=0.0123)`; assert results['hypotheses'][0]['claim'] is the statement, metrics experiments_succeeded 1, experiments_failed 1, findings_validated 1, total_cost_usd 0.0123, cost_per_validated_finding 0.0123. New file tests/unit/core/test_metrics_bridge.py: patch litellm.completion to return usage prompt_tokens=1000 and completion_tokens=500; after one `LiteLLMProvider(...).generate('x')`, `get_metrics().get_api_statistics()['estimated_cost_usd'] == pytest.approx(get_model_cost('deepseek/deepseek-chat', 1000, 500))` and api_calls == 1.

**Dependencies.** P0, P1, P2-0, P2-2, P2-3.

**Risk and blast radius.** Low for the CLI (run.py is a leaf); medium for providers/base.py (blast radius 124, docs/DEEP_ONBOARD.md:2012), additive and wrapped.

**Effort.** 8 h.

### P2-5: Novelty without sentence-transformers

**Goal.** Make novelty scoring correct without sentence-transformers, fix the vector-search path, and demote SPECTER to an optional dependency.

**Files.** kosmos/knowledge/embeddings.py:17-24 import guard, 56-60 PaperEmbedder.model=None, 201-203 embed_query returns np.zeros(768). kosmos/hypothesis/novelty_checker.py:67 `self.embedder = get_embedder()` (truthy even when model is None), 346-354 _compute_similarity (0/0 gives nan; `float(max(0.0, min(1.0, nan)))` is 1.0 because `nan < 1.0` is False), 384-391 _compute_hypothesis_similarity, 272-275 and 296-300 _check_existing_hypotheses, 242-250 _vector_search_papers (PaperMetadata without the required id, kosmos/literature/base_client.py:45; the except at 256-258 returns []), 397 _keyword_similarity (Jaccard). kosmos/agents/hypothesis_generator.py:79-83 defaults and 207-228 filter (drops when score is below 0.5 at 217-219; fail-open at 222-224). kosmos/agents/research_director.py:1939-1949 variants stored without a novelty check. pyproject.toml:69 sentence-transformers in core dependencies.

**Current.** With sentence-transformers absent every embedding is a zero vector; similarity to any same-domain row is 1.0, novelty 0.0, and every generated hypothesis is filtered once the database has a row in that domain (the climate run's component 2.1 recorded zero hypotheses yet PASS, evaluation/personas/runs/004_climate_data_scientist/v007_20260209/tier1/artifacts/phase2_components/2.1_hypothesis_generation.json:2,6); vector paper search always returns nothing.

**Required.** (1) kosmos/knowledge/embeddings.py: `PaperEmbedder.is_available -> bool` (self.model is not None); kosmos/hypothesis/novelty_checker.py:67 becomes `emb = get_embedder() if use_vector_db else None; self.embedder = emb if (emb is not None and emb.is_available) else None`. (2) Guard both cosine sites: `na, nb = norm(a), norm(b); if na == 0 or nb == 0: return 0.0`, and return 0.0 for a non-finite result. (3) Replace _keyword_similarity with TF-IDF cosine (sklearn `TfidfVectorizer(ngram_range=(1,2), stop_words='english', sublinear_tf=True)` fitted on the pair plus the current pool); keep Jaccard as the fallback if the sklearn import fails. (4) _vector_search_papers: `id = metadata.get('id') or metadata.get('doi') or metadata.get('arxiv_id') or f"vec_{sha1(title)[:12]}"`, `source=PaperSource(metadata.get("source","unknown"))` with an UNKNOWN fallback (enum at kosmos/literature/base_client.py:17-24), and `abstract=result.get("document","")` because abstract and authors are not in the stored metadata (kosmos/knowledge/vector_db.py:401-416). (5) _check_existing_hypotheses: limit 500 ordered by created_at descending, excluding the hypothesis's own id. (6) Director refinement path: run check_novelty on each variant or refined hypothesis before db_create_hypothesis; drop it when report.max_similarity >= 0.85 and log the dropped statement. (7) pyproject.toml: move sentence-transformers to `[project.optional-dependencies] embeddings`; the embeddings.py warning points to `pip install kosmos[embeddings]`. Decision: SPECTER is not worth a hard dependency. It pulls torch and fails in the owner's environment, and the need is near-duplicate detection among short statements plus a keyword literature search that works without it.

**Acceptance test.** Additions to tests/unit/hypothesis/test_novelty_checker.py with `patch('kosmos.knowledge.embeddings.HAS_SENTENCE_TRANSFORMERS', False)` and UnifiedLiteratureSearch mocked: (a) NoveltyChecker().embedder is None. (b) `_compute_hypothesis_similarity(h, h_copy) > 0.95`; unrelated statements below 0.2. (c) An embedder stub returning zeros with model set: _compute_similarity returns 0.0. (d) _vector_search_papers with one hit `{'metadata': {'title': 'T', 'abstract': 'A', 'doi': '10.1/x'}}` returns length 1 with id '10.1/x'. (e) An in-memory DB holding one climate hypothesis: a new distinct climate hypothesis gets novelty_score above 0.5 and is not filtered by HypothesisGeneratorAgent (LLM mocked).

**Dependencies.** None; P2-7 consumes step 6.

**Risk and blast radius.** Low; the get_embedder importers (kosmos/knowledge/literature_analyzer.py, kosmos/knowledge/graph_builder.py, kosmos/knowledge/semantic_search.py) see an unchanged return type.

**Effort.** 6 h.

### P2-6: LiteLLM JSON mode and tolerant parsing

**Goal.** Make LiteLLM structured output use JSON mode when the backend supports it, parse with the tolerant parser, and retry once on malformed JSON.

**Files.** kosmos/core/providers/litellm_provider.py:407-468 generate_structured (instruction at 434-437; single fence strip at 448-457; json.loads at 460; non-recoverable ProviderAPIError at 463-468) and 275-285 generate, which passes **kwargs to litellm.completion. kosmos/core/utils/json_parser.py:31-154 parse_json_response (object-only regex at 112). kosmos/core/providers/anthropic.py:545 and kosmos/core/providers/openai.py:493 already use it. Secondary brace-slicing parsers: kosmos/agents/data_analyst.py:515-520, kosmos/hypothesis/refiner.py:453-457 (array), kosmos/validation/scholar_eval.py:311-316.

**Current.** Any preamble text, trailing comma or unclosed fence makes generate_structured raise; no response_format is sent.

**Required.** (1) `kwargs_json = dict(kwargs); kwargs_json.setdefault('response_format', {'type': 'json_object'})`; call `self.generate(..., **kwargs_json)`; on an exception whose message mentions response_format or json_object, or whose class name starts with BadRequest or UnsupportedParams, log once and retry without response_format; memoize `self._supports_json_mode` so the fallback happens once per instance. (2) Parse with `parse_json_response(content, schema=schema)`; on JSONParseError issue one repair call `self.generate(prompt=f"Return ONLY the JSON object. Your previous reply was not valid JSON:\n\n{content[:4000]}", system=json_system, temperature=0.0, max_tokens=max_tokens)` and parse again; then raise ProviderAPIError(recoverable=True) carrying the first 300 characters. (3) If schema.get('required') is a list, log a warning listing missing keys. (4) kosmos/core/utils/json_parser.py: add parse_json_array_response (same strategies with the pattern `\[[\s\S]*\]`) and switch the three secondary parsers to the tolerant functions.

**Acceptance test.** New file tests/unit/core/test_litellm_structured.py with litellm.completion patched: (a) content `'Sure. ```json\n{"name": "x", "steps": [],}\n```'` returns {"name": "x", "steps": []} and the first call's kwargs include response_format={'type':'json_object'}. (b) The first call raises Exception("litellm.BadRequestError: response_format not supported"), the second returns clean JSON; parsed; a third generate_structured on the same instance does not send response_format. (c) Content 'not json' then a repair reply '{"a":1}' returns {"a":1} with completion called twice. (d) `parse_json_array_response('Here: [{"statement":"s"}]')` returns a list of length 1.

**Dependencies.** None; P2-1 and P2-2 depend on it.

**Risk and blast radius.** Low to medium; litellm_provider.py has blast radius 113 (docs/DEEP_ONBOARD.md:1993) but the change is internal to one method; _parse_response and _update_usage_stats are untouched.

**Effort.** 5 h.

### P2-7: Hypothesis-pool control

**Goal.** Stop hypothesis-pool runaway: cap generation, spawn variants only from validated results, deduplicate, and always test the best untested hypothesis before generating more.

**Files.** kosmos/agents/research_director.py:2549-2553 (REFINING branch; P1-2 fixes the exit transition), 2637-2638 (DESIGN picks untested[0], the oldest), 2706-2722 (REFINE always refines tested[-1] then increments the iteration), 1419-1424 (num_hypotheses default 3 per GENERATE), 1897-1899 (evaluate_hypothesis_status), 1932-1953 (spawns num_variants=2 without a novelty check), 1964-1966 (variants added to the pool). kosmos/hypothesis/refiner.py:107-167 evaluate_hypothesis_status (supports_hypothesis None gives SPAWN_VARIANT at 158-160; supported with 2 or more results gives SPAWN_VARIANT at 163-165) and 404-500 spawn_variant. kosmos/core/workflow.py:57-151 ResearchPlan (ids only; get_untested_hypotheses at 149-151). Evidence: evaluation/personas/runs/004_climate_data_scientist/v007_20260209/tier1_scaled/EVALUATION_REPORT.md:62-74 (195 hypotheses, 1 tested, 100 actions, 96 iterations).

**Current.** Every analyzed result with supports_hypothesis None spawns two LLM variants per REFINE; nothing caps the pool; the oldest untested hypothesis is designed next; duplicates are never removed.

**Required.** (1) Config keys with defaults: max_hypothesis_pool=12, max_untested_backlog=4, num_variants=1, max_refinements_per_hypothesis=2. (2) `ResearchPlan.hypothesis_scores: Dict[str, float]` and `hypothesis_generation: Dict[str, int]`; `add_hypothesis(id, score=None, generation=0)`; score = 0.5 * testability + 0.5 * novelty from the generator response; variants inherit parent_score * 0.9; get_untested_hypotheses() returns ids sorted by score descending then insertion order, excluding untestable_hypotheses (P2-1). (3) decide_next_action: in GENERATING_HYPOTHESES, if len(untested) >= max_untested_backlog or len(hypothesis_pool) >= max_hypothesis_pool, transition to DESIGNING_EXPERIMENTS and return DESIGN_EXPERIMENT; in REFINING (after P1-2) return DESIGN_EXPERIMENT when untested exist, else GENERATE_HYPOTHESIS. (4) _handle_refine_hypothesis_action: skip evaluate_hypothesis_status (decision CONTINUE_TESTING, no LLM call) when the latest result has execution_success False or validation_status in ('unvalidated', 'rejected_unsafe'); spawn variants only when validation_status == 'validated'; never when the pool is at the cap; count refinements per hypothesis and stop at max_refinements_per_hypothesis; run the P2-5 duplicate filter (max_similarity >= 0.85 against pool statements) before storing. (5) HypothesisRefiner.evaluate_hypothesis_status: the supports_hypothesis None branch returns CONTINUE_TESTING when result.status != SUCCESS; SPAWN_VARIANT only for SUCCESS with None. (6) get_research_status adds untested_backlog, untestable and variants_dropped_duplicate.

**Acceptance test.** New file tests/unit/agents/test_pool_control.py using the loops fixture: the generator mock returns 3 new ids per call; designer, executor and analyst mocks complete one experiment per DESIGN; the refiner mock's spawn_variant returns 2 variants with statements identical to existing ones. Drive decide_next_action plus _do_execute_action for 60 actions. Assert len(hypothesis_pool) <= 12, len(tested_hypotheses) >= 8, spawn_variant call count 0 when every result row is 'unvalidated', and when rows are 'validated' duplicates are dropped (variants_dropped_duplicate >= 1). Separate test: get_untested_hypotheses() returns the highest-score id first.

**Dependencies.** P1, P2-2, P2-5.

**Risk and blast radius.** Medium; decide_next_action is the central decision function, and the changes are confined to two branches; the ResearchPlan additions are additive; kosmos/core/convergence.py reads existing fields by name (docs/DEEP_ONBOARD.md:2089).

**Effort.** 8 h.

### P3-1: CodeValidator and emergency stop on the director path

**Goal.** Validate generated code with CodeValidator on the director path, consult the emergency-stop flag without hijacking process signals, and fix the SafetyIncident model so an emergency stop does not raise.

**Files.** kosmos/agents/research_director.py:1575-1583 (generate then execute, no validation). kosmos/safety/code_validator.py:36-40 DANGEROUS_MODULES, 59-86 __init__, 93 `if path and Path(path).exists()` (a Mock path raises TypeError), 160-232 validate. kosmos/safety/guardrails.py:46-93 __init__ (get_config at 58; CodeValidator(ethical_guidelines_path=...) at 65-70; enable_signal_handlers=True default at 49), 95-107 _register_signal_handlers (replaces SIGTERM and SIGINT), 227-298 trigger_emergency_stop (SafetyIncident(violation=None) at 286-298). kosmos/models/safety.py:97-110 SafetyIncident (violation required at 104). kosmos/execution/executor.py:1017-1088 execute_protocol_code (the only guardrails caller; SafetyGuardrails() at 1042; CodeValidator(allow_file_read=True) at 1062). tests/unit/safety/test_guardrails.py:24-38 (the Mock config pattern).

**Current.** The director executes unvalidated code; execute_protocol_code registers signal handlers on construction and its emergency stop raises a pydantic ValidationError; the 29 guardrails tests fail because a Mock path reaches Path().

**Required.** (1) kosmos/models/safety.py:104: `violation: Optional[SafetyViolation] = None`. (2) kosmos/safety/code_validator.py:93: `if isinstance(path, (str, os.PathLike)) and Path(path).exists()`. (3) kosmos/safety/guardrails.py: default enable_signal_handlers=False; register only when `threading.current_thread() is threading.main_thread()`; chain to the previous handler (`prev = signal.getsignal(sig)`; call it after triggering). (4) Director: lazy `self._code_validator = CodeValidator(allow_file_read=True)` and `self._guardrails = SafetyGuardrails(enable_signal_handlers=False)`; in _handle_execute_experiment_action after generate: `self._guardrails.sync_from_flag_file()`; if active raise RuntimeError("emergency stop active"); `report = self._code_validator.validate(code, context={'protocol': protocol.name})`; if not report.passed: `create_result(..., execution_success=False, data_source=None, validation_status='rejected_unsafe', error_message='; '.join(v.message for v in report.violations), data={}, code=code)`, remove the experiment from the queue, transition to ANALYZING (analyze marks it unvalidated), and return without executing. Pass `self._guardrails.enforce_resource_limits()` values into `CodeExecutor(sandbox_config={...})` at 1551. (5) execute_protocol_code: `SafetyGuardrails(enable_signal_handlers=False)`.

**Acceptance test.** (a) tests/unit/safety/test_guardrails.py passes unmodified (29 tests). (b) `SafetyGuardrails(enable_signal_handlers=False).trigger_emergency_stop('test', 'reason')` with incident_log_path=tmp_path/'x.jsonl' raises nothing and writes one incident line with `"violation": null`. (c) Loops fixture with a generator mock returning `import os\nos.system('echo hi')\nresults={}`: the executor mock is not called, a Result row exists with validation_status 'rejected_unsafe' and execution_success False. (d) After `signal.signal(SIGINT, custom)`, constructing SafetyGuardrails(enable_signal_handlers=True) in the main thread and `os.kill(os.getpid(), SIGINT)` calls custom (chained).

**Dependencies.** P0, P2-0.

**Risk and blast radius.** Low; CodeValidator is re-exported from kosmos/execution/executor.py:664 with an unchanged constructor; the optional violation field is backward compatible; the signal default change affects only execute_protocol_code and parallel.py.

**Effort.** 5 h.

### P3-2: Image build, declared dependencies, no HTTP probe

**Goal.** Make the Docker image build and run, declare the dependencies the code imports, and remove the HTTP health-probe assumption that no server satisfies.

**Files.** Dockerfile:44 `COPY pyproject.toml README.md ./`, 47-48 pip install ., 99 HEALTHCHECK imports kosmos, 102 CMD --help. pyproject.toml:26-100 core dependencies (no litellm, fastapi or Postgres driver; docker only in the optional execution group at 142-145; sentence-transformers at 69), 200-203 `[tool.setuptools.data-files]` requiring .env.example, alembic.ini and alembic/**, so `pip install .` in the image fails. docker-compose.yml:13 (8000:8000) and 31-36 healthcheck `requests.get(":8000/health")` with requests undeclared. k8s/kosmos-deployment.yaml:25,91-100; k8s/kosmos-service.yaml:11-12; k8s/ingress.yaml:28. kosmos/api/health.py:18 HealthChecker (a plain class; imported only by kosmos/monitoring/alerts.py:277); kosmos/api/streaming.py and kosmos/api/websocket.py import fastapi and have zero importers. kosmos/execution/code_generator.py:768 and kosmos/core/providers/litellm_provider.py import litellm at runtime.

**Current.** The image build fails at the data-files step; the compose healthcheck fails because nothing listens on 8000; litellm must be installed by hand.

**Required.** (1) Dockerfile:44: `COPY pyproject.toml README.md .env.example alembic.ini ./` and `COPY alembic/ ./alembic/`; keep CMD; HEALTHCHECK becomes `python -m kosmos.cli.main version` (no --version flag exists; version is a subcommand). (2) pyproject.toml: add "litellm>=1.40.0" to core; optional groups `server = ["fastapi>=0.110", "uvicorn[standard]>=0.29", "requests>=2.31"]`, `postgres = ["psycopg[binary]>=3.1"]`, `embeddings = ["sentence-transformers>=2.2.0"]` (P2-5); document the existing execution group (docker) in the README. (3) Decision: no HTTP server; single-user CLI. docker-compose.yml:31-36 test becomes `["CMD", "python", "-m", "kosmos.cli.main", "version"]`; remove the 8000:8000 mapping and its comment at docker-compose.yml:13; move k8s/ to archive/k8s/ with a README note; move kosmos/api/streaming.py and kosmos/api/websocket.py to archive (P3-4); keep kosmos/api/health.py. (4) Add scripts/check_env.py (or extend `kosmos doctor`) reporting provider, model, litellm importable, Docker daemon reachable, sentence_transformers importable, DB URL and the pinned sandbox image tag.

**Acceptance test.** `docker build -t kosmos:test .` succeeds and `docker run --rm kosmos:test python -m kosmos.cli.main version` prints a version (needs Docker). `pip install -e . --dry-run` in a clean venv resolves litellm. `python -c "import kosmos.cli.main"` succeeds without fastapi installed.

**Dependencies.** P2-5, P3-4.

**Risk and blast radius.** Low; packaging only; the removed modules have zero importers.

**Effort.** 4 h.

### P3-3: Test suite green for surviving modules; README test count removed

**Goal.** Return the unit suite to green at HEAD for the modules that survive consolidation, and stop the README from asserting a test count.

**Files.** tests/requirements/core/test_req_configuration.py:325 (asserts "claude-3-5-sonnet-20241022" against _DEFAULT_CLAUDE_SONNET_MODEL at kosmos/config.py:17-18). tests/unit/workflow/test_research_loop.py:215-220,256-259,287-292,466-471 (AsyncMock for create_plan, review_plan and revise_plan, which kosmos/workflow/research_loop.py:272,290,300-301 call synchronously; execute_plan, save_finding_artifact and generate_cycle_summary are awaited at 310, 336 and 367 and stay AsyncMock). tests/unit/safety/test_guardrails.py (29 tests; fixed by P3-1 step 2). kosmos/literature/base_client.py:230-234 _handle_api_error re-raises; tests expecting swallowed errors: tests/unit/literature/test_arxiv_client.py:98, tests/unit/literature/test_pubmed_client.py:70, tests/unit/literature/test_semantic_scholar.py:107,223,230, tests/unit/literature/test_unified_search.py:122 (unified search catches per source at kosmos/literature/unified_search.py:170-171, so it may already pass). README.md:8 badge, 306-312 table, 398 footer. pytest.ini (80 percent gate, warnings as errors).

**Current.** One assertion pins an old model name; seven of sixteen research-loop tests await synchronous methods; all 29 guardrails tests fail on a Mock config; five literature tests expect swallowed errors; the README asserts a test count its own table contradicts.

**Required.** (1) tests/requirements/core/test_req_configuration.py:325: import _DEFAULT_CLAUDE_SONNET_MODEL and assert equality with it. (2) tests/unit/workflow/test_research_loop.py: change the three synchronous methods to `Mock(return_value=...)` at the listed lines, or delete the file with the module under Section 6. (3) Literature client tests: the contract is "clients raise; UnifiedLiteratureSearch isolates per source"; rewrite the five client tests to pytest.raises; keep tests/unit/literature/test_unified_search.py:122. (4) README: delete the numeric badge at README.md:8, replace README.md:306-312 with the command `pytest tests/unit --no-cov -q` and the sentence "counts change; CI is authoritative", update README.md:398. (5) Add .github/workflows/unit.yml running `pytest tests/unit --no-cov -p no:cacheprovider -q` on push (no secrets; tests/conftest.py loads .env with override, so the job must not have one).

**Acceptance test.** `python -m pytest tests/requirements/core/test_req_configuration.py tests/unit/safety/test_guardrails.py tests/unit/literature --no-cov -p no:cacheprovider -q` exits 0; if tests/unit/workflow/test_research_loop.py is kept it exits 0 too.

**Dependencies.** P3-1, Section 6 decision.

**Risk and blast radius.** Minimal.

**Effort.** 4 h.

### P3-4: Archive zero-importer modules

**Goal.** Archive modules with no production call path so the package reflects what runs.

**Files.** Tier A, zero production importers (verified on 2026-10-02 with a grep over kosmos/, scripts/ and evaluation/ excluding __init__ exports): kosmos/core/domain_router.py, kosmos/core/feedback.py, kosmos/core/memory.py, kosmos/validation/failure_detector.py, kosmos/validation/accuracy_tracker.py, kosmos/validation/accuracy_validator.py (the sole importer of kosmos/validation/benchmark_dataset.py, which goes with it), kosmos/safety/verifier.py, kosmos/knowledge/graph_builder.py, kosmos/knowledge/graph_visualizer.py, kosmos/execution/notebook_generator.py, kosmos/execution/figure_manager.py, kosmos/execution/production_executor.py, kosmos/execution/result_collector.py, kosmos/analysis/plotly_viz.py, kosmos/analysis/statistics.py, kosmos/workflow/ensemble.py, kosmos/api/streaming.py, kosmos/api/websocket.py; with tests/unit/validation/test_failure_detector.py, test_accuracy_tracker.py, test_benchmark_dataset.py, tests/unit/execution/test_notebook_generator.py, test_figure_manager.py, test_production_executor.py, tests/unit/safety/test_verifier.py. Tier B, library-loop only (Section 6): kosmos/workflow/research_loop.py (importers scripts/smoke_test.py, scripts/verify_e2e.py, kosmos/workflow/ensemble.py), kosmos/orchestration/plan_creator.py, plan_reviewer.py, delegation.py, novelty_detector.py, kosmos/compression/ (importers kosmos/workflow/research_loop.py and scripts/smoke_test.py), tests/unit/workflow/, tests/unit/orchestration/, tests/unit/compression/. Tier C, reachable only through protocol templates that P2-1 bypasses when a dataset is supplied: kosmos/domains/ (imported by kosmos/experiments/templates/biology, materials and neuroscience packages and kosmos/knowledge/domain_kb.py); owner decision. Keep: kosmos/api/health.py (imported by kosmos/monitoring/alerts.py:277), kosmos/world_model/artifacts.py (ported in C-1), kosmos/validation/scholar_eval.py and null_model.py (P2-2), kosmos/agents/skill_loader.py. Prune exports: kosmos/execution/__init__.py:89,134 (CodeProvenance stays), kosmos/validation/__init__.py:29-54, kosmos/safety/__init__.py:7-16.

**Current.** About 27 percent of kosmos/ has no production call path; the __init__ modules export it, so every import of kosmos.validation or kosmos.execution loads it.

**Required.** `git mv` Tier A to archive/code/<same path> in one commit with archive/code/README.md listing each module, the commit that removed its last importer, and the reason; remove the __init__ exports; Tier B in a second commit after the Section 6 decision; Tier C only after owner question 4. After each commit run the unit suite with --no-cov, then once with coverage to confirm the 80 percent gate still passes.

**Acceptance test.** `python -c "import kosmos, kosmos.execution, kosmos.validation, kosmos.safety, kosmos.agents.research_director"` succeeds; `rg` for the archived module names under kosmos/ returns nothing; `python -m pytest tests/unit --no-cov -p no:cacheprovider -q` has no ImportError collection failures.

**Dependencies.** Section 6 decision; P3-2.

**Risk and blast radius.** Low for Tier A; Tier C touches the template registry in kosmos/experiments/templates/base.py (archive the domain templates together with the domain analyzers).

**Effort.** 6 h for Tiers A and B, plus 4 h for Tier C.

### P3-5: README and DEEP_ONBOARD corrections

**Goal.** Correct the README and the onboarding document so they describe the director path, the measured state, and the right paper.

**Files.** README.md:3 and 377 ("Lu et al. (2024)"; the paper is Mitchener et al., Edison Scientific, November 2025), 7 badge "paper_gaps 17/17 complete", 8 and 306-312 and 398 test counts, 46 smoke test, 52-68 Python quickstart (research_loop), 70-96 CLI examples (no --data-path or --seed), 321-331 claims table, 335 "Without Docker, code execution falls back to direct exec()" (false: a missing SDK fails the kosmos.execution import; a stopped daemon raises RuntimeError at kosmos/execution/sandbox.py:117-122), 343. docs/DEEP_ONBOARD.md known-wrong items: the host-exec fallback, getattr in SAFE_BUILTINS, 28 instead of 33 state-machine edges, LRU instead of oldest-written cache eviction, the cache directories, the --stream display queue, the conftest resets. archive/PAPER_IMPLEMENTATION_GAPS.md.

**Current.** The README names the wrong paper, quotes a test count its own table contradicts, shows a Python quickstart for the library loop, omits --data-path and --seed from the CLI examples, and claims a host-exec fallback that does not exist; docs/DEEP_ONBOARD.md carries the seven known-wrong items.

**Required.** README Quick Start becomes `kosmos run "<question>" --domain <d> --data-path <csv> --seed 42 --max-iterations 3 --budget 1`; Verify becomes the new smoke test (Section 6); add a "What it does today" paragraph built from the Section 7 metrics; the paper status table gains a "Measured" column reading "no" on every row; fix the attribution at 3 and 377; replace 335 with "Docker is required for sandboxed execution; kosmos run records a SandboxUnavailable failure when the daemon is unreachable" (matching P0-4); add a "Not a reproduction" banner at the top. Fix the seven DEEP_ONBOARD items in place. Add a banner to archive/PAPER_IMPLEMENTATION_GAPS.md: "Checked on existence of code, not on measured behavior; see evaluation/VIABILITY_ASSESSMENT_AND_CHANGE_PLAN.md."

**Acceptance test.** `grep -n "Lu et al\|3704\|research_loop" README.md` returns nothing; `grep -n "Mitchener" README.md` returns at least one line; `kosmos run --help` lists --seed and --data-path.

**Dependencies.** P2-3, Section 6, P3-3.

**Risk and blast radius.** None to code.

**Effort.** 3 h.

### C-1: Findings JSON and `kosmos report`

**Goal.** Port the library loop's two useful outputs, a findings artifact per validated result and a rendered report, onto the director path.

**Files.** kosmos/world_model/artifacts.py:51-85 Finding and 199-240 save_finding_artifact; kosmos/workflow/research_loop.py:426-476 generate_report (source to port, then archived); kosmos/agents/research_director.py _handle_analyze_result_action (after the P2-2 gate); kosmos/cli/main.py (new `report` command); kosmos/db/operations.py get_results_for_run (P2-0).

**Current.** The director writes result rows and nothing else; the library loop writes findings JSON and a markdown report, but from mocked statistics (Section 6).

**Required.** After P2-2 marks a result validated or rejected, build a Finding from the row (summary, statistics, methods, interpretation, hypothesis_id, null_model_result, scholar_eval, code_provenance with notebook_path = the saved .py from P2-3 and cell_index=0) and call ArtifactStateManager.save_finding_artifact to write `<artifacts_dir>/<run_id>/findings/<result_id>.json`. Add `kosmos report --run-id <id> [--output <path>]` that reads get_results_for_run and renders validated and rejected findings with their provenance, the Section 7 metrics, and the failed experiments with error messages, using the structure of generate_report.

**Acceptance test.** New file tests/unit/cli/test_report.py: seed an in-memory DB with one validated and one rejected result for run 'r1' plus provenance dicts; `kosmos report --run-id r1 --output tmp/report.md` through typer's CliRunner exits 0 and the file contains both result ids, the string "validated", the git_sha from provenance and the failed experiment's error message; the findings JSON written by the analyze handler (loops fixture, P2-2 mocks) parses and carries scholar_eval and null_model keys.

**Dependencies.** P2-2, P2-3.

**Risk and blast radius.** Low; additive CLI command; ArtifactStateManager is already production code.

**Effort.** 5 h.

### M-1: `kosmos validate-null` and `kosmos rerun`

**Goal.** Add the two commands that produce the metrics requiring re-execution: the shuffled-control null pass rate and the reproducibility rate across seeds.

**Files.** kosmos/cli/main.py (two new commands); kosmos/validation/analysis_fn.py (P2-2, build_analysis_fn and shuffle_target); kosmos/execution/executor.py execute_with_data with the seed parameter (P2-3); kosmos/db/operations.py get_results_for_run and update_result_validation (P2-0); kosmos/validation/null_model.py:166-256 validate_finding.

**Current.** There is no way to re-execute a stored result or to measure the null model's false-positive rate on shuffled data.

**Required.** `kosmos validate-null --run-id R --k 20 --alpha 0.05 --seed 0`: for each result in the run with validation_status in ('validated', 'rejected') and a columns entry in statistical_tests, load the dataset at provenance['data_path'], verify its sha256 equals provenance['data_sha256'] (otherwise skip the row and print why), build `fn = build_analysis_fn(test_type, x, y)`, and for i in range(k): `df_s = shuffle_target(df, test_type, x, y, rng=default_rng(seed + i))`; `passes += NullModelValidator(n_permutations, random_seed=seed + i).validate_finding({'statistics': fn(df_s)}, data=df_s, analysis_func=fn).passes_null_test`; `shuffled_pass_rate = passes / k`; store it in validation_detail['shuffled_pass_rate'] through update_result_validation with the status unchanged; print one row per result and the run-level mean; exit 1 when the mean exceeds alpha. `kosmos rerun --result-id X --seeds 1,2,3`: load the result, its experiment's code_generated and its provenance; verify the data sha256; execute `CodeExecutor(use_sandbox=provenance['sandbox_used']).execute_with_data(code, data_path, seed=provenance['seed'])`; `exact_match = isclose(result['statistic'], stored statistic, rel_tol=1e-9)`; for each extra seed re-execute and record whether the p below alpha conclusion holds; store `provenance['reproducibility'] = {'exact_match': bool, 'seeds': {seed: p_value}, 'conclusion_stable': bool}`; print; exit 1 when exact_match is False.

**Acceptance test.** New file tests/unit/cli/test_metric_commands.py: seed an in-memory DB with one validated pearson_correlation result on the climate CSV (statistics from build_analysis_fn, provenance carrying the file's sha256 and seed 42, code_generated set to the P2-1 generated code); through typer's CliRunner, `validate-null --run-id r1 --k 20 --seed 0` exits 0 and the stored shuffled_pass_rate is at most 0.15 (deterministic given the fixed seeds); `rerun --result-id X --seeds 1,2` exits 0 with exact_match True; after tampering the stored statistic to 0.5, `rerun` exits 1.

**Dependencies.** P2-0, P2-2, P2-3.

**Risk and blast radius.** Low; additive CLI commands; no production module changes.

**Effort.** 4 h.

## 6. Orchestrator consolidation

**Recommendation: keep the director; archive the library loop.**

The library loop cannot be the base. DelegationManager._execute_data_analysis (kosmos/orchestration/delegation.py:388-422) sends `{'description': <task text>}` to DataAnalystAgent.execute('interpret_results') (kosmos/orchestration/delegation.py:404-409) and returns the LLM's words as statistics (kosmos/orchestration/delegation.py:417); no code is generated or executed, there is no data path, no database row and no protocol. It wires real agents only when given a raw Anthropic client (kosmos/workflow/research_loop.py:117-133), which the owner's policy forbids; otherwise planner, reviewer and ScholarEval return approving mocks on any error (kosmos/orchestration/plan_creator.py:194-198; kosmos/orchestration/plan_reviewer.py:160-162; kosmos/validation/scholar_eval.py:204-207). Its null model needs a STATISTIC_KEYS entry or a p_value (kosmos/validation/null_model.py:349-369) that delegation never emits, so only blank findings validate. The director has database persistence, the state machine, event streaming, the CLI, the data path, the sandbox, and every empirical run.

Port into the director:

1. ScholarEval and the null model into _handle_analyze_result_action, with the permutation test on the real data (P2-2).
2. Seed and temperature (kosmos/workflow/research_loop.py:62-63,79-89) become `--seed` and config (P2-3).
3. Findings JSON and the report (C-1): reuse Finding and save_finding_artifact (kosmos/world_model/artifacts.py:51-85,199-240) and port generate_report (kosmos/workflow/research_loop.py:426-476) into `kosmos report --run-id`.
4. Plan and review: do not port now. Each costs an LLM call per iteration and the fallback approves on error. Revisit once the Section 7 metrics exist.
5. NoveltyDetector (kosmos/orchestration/novelty_detector.py): archive; the director uses kosmos/hypothesis/novelty_checker.py (P2-5).

Archive with tests (P3-4 Tier B): kosmos/workflow/research_loop.py, kosmos/workflow/ensemble.py, kosmos/orchestration/, kosmos/compression/, tests/unit/workflow/, tests/unit/orchestration/, tests/unit/compression/ and scripts/verify_e2e.py. Keep kosmos/world_model/artifacts.py, kosmos/validation/scholar_eval.py, kosmos/validation/null_model.py and kosmos/agents/skill_loader.py.

README: replace the Python quickstart at README.md:52-68 with the CLI command, and point programmatic users at evaluation/scientific_evaluation.py:258-300, which already drives the director from Python. Add no new Python facade.

scripts/smoke_test.py: replace the import list at scripts/smoke_test.py:18-29 with ResearchDirectorAgent, ExperimentCodeGenerator, CodeExecutor, describe_dataset, NullModelValidator, build_analysis_fn and ArtifactStateManager; delete test_compression (scripts/smoke_test.py:43-61), test_orchestration (scripts/smoke_test.py:101-133) and test_workflow (scripts/smoke_test.py:161-179); add test_real_data_pipeline: build the P2-1 protocol fixture bound to the climate CSV, generate with ExperimentCodeGenerator(use_llm=False), execute with `CodeExecutor(use_sandbox=False).execute_with_data(code, csv, seed=42)`, assert success, data_source 'file', p_value below 0.05 and a statistic present; `NullModelValidator(n_permutations=200, random_seed=42).validate_finding({'statistics': results}, data=df, analysis_func=fn)` passes on the real file and fails on a temp_anomaly_c-shuffled copy; ScholarEvalValidator(allow_mock=True) still scores. Keep test_state_manager and test_skill_loader. Under 30 seconds, no LLM, no Docker.

## 7. Output-quality metrics

| Metric | Definition | Emitted at | Stored in |
|---|---|---|---|
| experiments_executed_per_run | Result rows with execution_success True per run_id, reported beside attempted and failed | _handle_execute_experiment_action after create_result (kosmos/agents/research_director.py:1630-1643 as rewritten by P0-4); aggregated by build_run_results (P2-4) and get_research_status (kosmos/agents/research_director.py:2942-2968) | results.execution_success; summary metrics.experiments_succeeded |
| findings_with_real_test_statistic | Rows whose statistical_tests has a STATISTIC_KEYS entry (kosmos/validation/null_model.py:135-139) and a p_value, data_source 'file', and validation_detail.recomputed_match True | P2-2 gate | results.validation_detail; summary metrics.findings_with_statistic |
| null_model_pass_rate_real_vs_shuffled | Real: fraction of statistic-bearing results passing the null model. Shuffled control: fraction of the same protocols passing when the dependent column is permuted k times (target at or below alpha) | Real: P2-2 gate. Shuffled: `kosmos validate-null --run-id R --k 20` (M-1) using shuffle_target and NullModelValidator | results.validation_detail.null_model and .shuffled_pass_rate; summary metrics.null_pass_rate_real and null_pass_rate_shuffled |
| reproducibility_rate_across_seeds | Fraction of validated findings whose statistic matches to 1e-9 when experiments.code_generated is re-executed with the stored seed on a file whose sha256 matches provenance.data_sha256, plus the fraction whose p below alpha conclusion holds under 3 further seeds | `kosmos rerun --result-id X --seeds 1,2,3` (M-1) through execute_with_data(code, data_path, seed=s) (P2-3) | results.provenance.reproducibility; summary metrics.reproducibility_rate |
| cost_per_validated_finding | llm_client.get_usage_stats()['total_cost_usd'] divided by the count of rows with validation_status 'validated'; per-result cost_usd from handler deltas | build_run_results (P2-4); handler deltas (P2-4 step 3); the budget check through the P1-5 bridge | summary metrics.total_cost_usd and cost_per_validated_finding; results.cost_usd |
| fraction_results_data_source_file | Rows with data_source 'file' divided by all rows for the run | create_result, from ExecutionResult.data_source (kosmos/execution/executor.py:517 on the host path; the `RESULT:` payload's data_source key on the sandbox path, P0-2 and P0-4) | results.data_source; summary metrics.results_from_file and results_synthetic |
| hypotheses_tested_ratio and untested_backlog | len(tested_hypotheses) / len(hypothesis_pool); mean untested backlog per iteration | get_research_status (P2-7) | summary metrics.hypotheses_tested and untested_backlog |

evaluation/scientific_evaluation.py replaces the hard-coded loop_completed check (evaluation/scientific_evaluation.py:521-525) with assertions on experiments_succeeded >= 1, findings_with_statistic >= 1 and null_pass_rate_shuffled <= 0.05, read from build_run_results. The existence-based component checks in evaluation/run_phase2_tests.py (PASS whenever the function returns, evaluation/run_phase2_tests.py:40-45) stay as smoke tests but must be labeled "component imports" in their report section, not "data analysis".

## 8. Verification plan

Run everything from the repository root with the conda environment that has the docker SDK installed. Never run the bare `pytest` command: pytest.ini enforces an 80 percent coverage gate and treats warnings as errors, and tests/conftest.py loads .env with override, so the e2e directory makes live calls. Steps 1 to 22 make no LLM calls and need no Docker daemon. Line numbers in Section 5 are exact at 6cfe7f6; after each change in a file, locate the next change in the same file by the quoted code, not by the number.

| Step | After | Command | Expected outcome |
|---|---|---|---|
| 1 | P0-1 | `python -m pytest tests/unit/execution/test_code_generator.py --no-cov -p no:cacheprovider -q` | exit 0; the three generic_protocol tests pass; `grep -n "DataAnalyzer" tests/unit/execution/test_code_generator.py` still lists the TTest, Correlation and LogLog assertions (changed in P0-5 and P0-6) |
| 2 | P0-2, P0-3 | `python -m pytest tests/unit/execution/test_executor.py --no-cov -p no:cacheprovider -q` | exit 0; the TestSandboxIntegration additions pass; test_execute_with_data_path (tests/unit/execution/test_executor.py:330-341) still passes |
| 3 | P0-4 | `python -m pytest tests/unit/agents/test_research_director_execute.py --no-cov -p no:cacheprovider -q` | 5 passed; if `ExperimentProtocol.model_validate(proto.to_dict())` fails in the fixture, record the round-trip defect and store `proto.model_dump(mode="json")` instead |
| 4 | P0-5, P0-6 | repeat step 1 | exit 0; `grep -c "DataAnalyzer\|MLAnalyzer" tests/unit/execution/test_code_generator.py` prints 0; the LogLog exponent test and the ML accuracy test pass |
| 5 | end of P0 (needs Docker, the sandbox image, and owner authorization because it calls DeepSeek) | `kosmos run "Does CO2 concentration predict temperature anomaly?" --data-path evaluation/data/climate_co2_temperature_test.csv --max-iterations 1` then `sqlite3 kosmos.db "select json_extract(data,'$.execution_success'), json_extract(data,'$.data_source'), p_value from results order by created_at desc limit 1; select status from experiments order by created_at desc limit 1"` | `1\|file\|<finite float>` and `completed`; the p_value belongs to year against co2_ppm until P2-1 |
| 6 | P1-1 | `python -m pytest tests/unit/agents/test_research_director_analyze.py --no-cov -p no:cacheprovider -q` | 3 passed |
| 7 | P1-2, P1-3 | `python -m pytest tests/unit/agents/test_research_director_loops.py --no-cov -p no:cacheprovider -q` | exit 0; the four _leave_refining tests and the four error-recovery tests pass; no test takes longer than 3 s except the sync backoff test |
| 8 | P1-4 | repeat step 3 | 6 passed |
| 9 | P1-5 | `python -m pytest tests/unit/core/test_budget_wiring.py --no-cov -p no:cacheprovider -q && python -m pytest tests/e2e/test_budget_enforcement.py --no-cov -p no:cacheprovider -q -k "halts_on_exceeded"` | 3 passed, then 1 passed with the same dollar figures as before (the test uses claude-3-5-sonnet-20241022, priced at tests/e2e/test_budget_enforcement.py:33-59 and kosmos/core/pricing.py:20) |
| 10 | P2-0 | `python -m pytest tests/unit/db/test_database.py --no-cov -p no:cacheprovider -q`, then on a copy of the owner's database, `cp kosmos.db /tmp/k.db` and `alembic upgrade head` with the URL pointed at the copy, then `sqlite3 /tmp/k.db "select execution_success, data_source from results"` | exit 0; the migration prints the add_result_provenance_columns revision; the query prints `1\|file` for row d39936bb |
| 11 | P2-1 | `python -m pytest tests/unit/execution/test_column_binding.py --no-cov -p no:cacheprovider -q` | 6 passed; test (d) reports statistic above 0.85 on the climate CSV (the true value is 0.9317) |
| 12 | P2-2 | `python -m pytest tests/unit/validation --no-cov -p no:cacheprovider -q` | exit 0; test_director_gate (a) validated, (b) rejected, (c) rejected, (d) unvalidated, (e) passes_threshold False, (f) all four CSVs within 1e-9; tests/unit/validation/test_scholar_eval.py passes with allow_mock=True |
| 13 | P2-3 | `python -m pytest tests/unit/execution/test_seed_provenance.py --no-cov -p no:cacheprovider -q` | 3 passed |
| 14 | P2-4 | `python -m pytest tests/unit/cli/test_run_results.py tests/unit/core/test_metrics_bridge.py --no-cov -p no:cacheprovider -q` | exit 0; cost_per_validated_finding == 0.0123; estimated_cost_usd equals get_model_cost('deepseek/deepseek-chat', 1000, 500) |
| 15 | P2-5 | `python -m pytest tests/unit/hypothesis --no-cov -p no:cacheprovider -q` | exit 0; the five novelty additions pass with HAS_SENTENCE_TRANSFORMERS patched False |
| 16 | P2-6 | `python -m pytest tests/unit/core/test_litellm_structured.py --no-cov -p no:cacheprovider -q && python -m pytest tests/unit/core -k litellm --no-cov -p no:cacheprovider -q` | 4 passed, then exit 0 |
| 17 | P2-7 | `python -m pytest tests/unit/agents/test_pool_control.py --no-cov -p no:cacheprovider -q` | 2 passed; pool never exceeds 12 over 60 actions |
| 18 | P3-1 | `python -m pytest tests/unit/safety --no-cov -p no:cacheprovider -q` | exit 0; all 29 tests in tests/unit/safety/test_guardrails.py pass unmodified plus the four new tests |
| 19 | P3-2 (needs Docker) | `docker build -t kosmos:test . && docker run --rm kosmos:test python -m kosmos.cli.main version`; `pip install -e . --dry-run`; `python -c "import kosmos.cli.main"` in an environment without fastapi | a version string; litellm resolved; exit 0 |
| 20 | P3-3 | `python -m pytest tests/requirements/core/test_req_configuration.py tests/unit/safety/test_guardrails.py tests/unit/literature --no-cov -p no:cacheprovider -q` | exit 0 |
| 21 | P3-4 | `python -c "import kosmos, kosmos.execution, kosmos.validation, kosmos.safety, kosmos.agents.research_director; print('ok')"`; `rg -l "domain_router\|failure_detector\|accuracy_tracker\|notebook_generator\|production_executor\|graph_visualizer\|plotly_viz" kosmos/`; `python -m pytest tests/unit --co -q --no-cov -p no:cacheprovider \| tail -3` | ok; no files; "N tests collected" with no errors |
| 22 | P3-5 | `grep -n "Lu et al\|3704\|research_loop" README.md; grep -c "Mitchener" README.md; kosmos run --help` | nothing; at least 1; `--seed` and `--data-path` listed |
| 23 | C-1, M-1 | `python -m pytest tests/unit/cli/test_report.py tests/unit/cli/test_metric_commands.py --no-cov -p no:cacheprovider -q` | exit 0; shuffled_pass_rate at most 0.15; rerun exact_match True then exit 1 after tampering |
| 24 | all phases | `python -m pytest tests/unit/execution tests/unit/agents tests/unit/core tests/unit/db tests/unit/validation tests/unit/hypothesis tests/unit/safety tests/unit/cli --no-cov -p no:cacheprovider -q` | exit 0 |
| 25 | live acceptance (owner-authorized, under $1, needs the sandbox image) | `kosmos run "Does atmospheric CO2 concentration predict global temperature anomaly?" --domain climate_science --data-path evaluation/data/climate_co2_temperature_test.csv --seed 42 --max-iterations 3 --budget 1`; then `kosmos report --run-id <run_id>`, `kosmos rerun --result-id <id> --seeds 1,2,3`, `kosmos validate-null --run-id <run_id> --k 20` | the results table shows at least one row with Exec OK, Data file, Test pearson_correlation, Stat about 0.93, p about 6e-29, Validation validated; the metrics summary shows a non-zero total_cost_usd below 1.00 (DeepSeek is priced at $0.14 and $0.28 per million tokens, kosmos/core/pricing.py:35, so three iterations cost cents); rerun exits 0 with exact_match True; validate-null reports a mean shuffled pass rate at or below 0.05 |

Expected failures that are not regressions: tests/unit/agents/test_research_director.py is skipped without ANTHROPIC_API_KEY (tests/unit/agents/test_research_director.py:20-26); tests/unit/workflow, tests/unit/orchestration and tests/unit/compression disappear with Tier B.

## 9. Non-goals and things not to change

- **Paper reproduction at scale.** No 12-hour runs, no 1,500-paper literature rollouts, no world-model coherence work, no 79.4 percent accuracy harness. The verdict for that goal is RESEARCH ONLY.
- **HTTP server, SSE and WebSocket.** kosmos/api/streaming.py and kosmos/api/websocket.py are archived (P3-4); the health probes are removed (P3-2); no server is added.
- **Multi-tenancy, authentication, Kubernetes.** k8s/ moves to archive/k8s/.
- **R execution.** docker/sandbox/Dockerfile.r and the R executor stay as they are; they are not on the live path.
- **Neo4j.** The in-memory world model fallback stays the default; no graph features are added or removed.
- **The parallel execution path.** kosmos/execution/parallel.py and execute_protocol_code (kosmos/execution/executor.py:1017-1088) are left alone apart from the signal-handler default in P3-1.
- **Host-path retry rewrites.** The rewrites that turn failures into success dicts (kosmos/execution/executor.py:869-1008) stay; after P0-4 the sandbox path is the live path and its error types never match a rewrite branch (kosmos/execution/executor.py:791-822), so they are inert. Do not revive them.
- **Sandbox timeout handling.** docker-py's wait(timeout) raises a requests ReadTimeout rather than docker.errors.APIError, so the branch at kosmos/execution/sandbox.py:306-319 is skipped and the failure is reported by kosmos/execution/sandbox.py:381-391 with the container removed at 393-400. Leave it; the outcome is a stored failure after P0-4.
- **DataProvider on the director path.** It is constructed with a file path as a directory and never used (kosmos/agents/research_director.py:1552-1555); P2-1's describe_dataset replaces its role. Do not wire it.
- **The message bus and AgentRegistry.** They carry no traffic; leave them.
- **Phase 4 polyglot persistence.** Not started; not planned.
- **kosmos/domains and the domain protocol templates.** Untouched until owner question 4 is answered.

## 10. Open questions for the owner

1. **Unbound hypotheses (P2-1).** When no dataset column can be bound to a hypothesis's variables, mark it REJECTED in the database, or leave it GENERATED and only exclude it from the untested list? The plan excludes it and leaves the status; say if you prefer REJECTED.
2. **Live runs.** Authorize the end-of-P0 check (step 5, one DeepSeek call sequence, cents) and the live acceptance run (step 25, under $1), and confirm the climate CSV as the acceptance dataset.
3. **sentence-transformers.** Move it to an optional `embeddings` extra (P2-5)? It pulls torch and fails to import in your environment today.
4. **Tier C.** Archive kosmos/domains and the biology, materials and neuroscience protocol templates now that a supplied dataset forces LLM-designed protocols, or keep them for dataset-free runs?
5. **Library loop.** Delete kosmos/workflow/research_loop.py, kosmos/orchestration and kosmos/compression with their tests, or keep archive/code copies in-tree? In-tree copies count toward the 80 percent coverage gate.
6. **Runs without a dataset.** Should `kosmos run` without `--data-path` still execute synthetic-data experiments (data_source 'synthetic', never validated), or refuse to execute experiments?
7. **Thresholds on DeepSeek.** The defaults are null_permutations 500 and ScholarEval threshold 0.75 with min_rigor 0.70, tuned for Claude Sonnet. Keep them for DeepSeek, or lower the ScholarEval threshold after a first calibration run?
8. **Artifacts directory.** Default `<cwd>/artifacts/<run_id>/` for code and findings, and add it to .gitignore?
9. **Sandbox image.** It is not built on this machine as of 2026-10-01. DockerSandbox builds it on first use from docker/sandbox (kosmos/execution/sandbox.py:127-161, pulling python:3.11-slim). Build it before step 5, or pin a tag and build in CI?
10. **One generation round per run.** After P1-2 a run whose only hypothesis is tested converges on the next step instead of generating a second batch; P2-7 adds the policy. Acceptable in the interim?
11. **SandboxUnavailable.** Should the first SandboxUnavailable failure stop the run, or should the run keep recording failed experiments for every hypothesis up to `--max-iterations`? The plan continues.

## 11. Appendix A: evidence index of every file:line cited

Every file:line citation in Sections 1 to 10, grouped by file and relative to the repository root at 6cfe7f6. R004 in Section 3 expands to evaluation/personas/runs/004_climate_data_scientist/v007_20260209. Short names inside code comments (executor.py:573, sandbox.py:117-122, workflow.py:204-210) refer to kosmos/execution/executor.py, kosmos/execution/sandbox.py and kosmos/core/workflow.py and are listed under those paths. Bare line numbers inside Section 5 change blocks follow the convention stated at the top of Section 5 and are not repeated here.

| File | Lines cited | What it evidences |
|---|---|---|
| CHANGELOG.md | 163-166, 189 | 0.1.0 "Initial Production Release" dated 2025-11-06; 90 percent coverage claim |
| Dockerfile | 44 | copies only pyproject and README, so the data-files step fails; healthcheck and CMD |
| README.md | 3, 8, 12-20, 52-68, 70-96, 114, 162-177, 198, 306-312, 321, 325, 326, 327, 328, 329, 331, 341, 343, 360-371, 369, 377, 398 | attribution, loop description, quickstarts, CLI, providers, status claims, test and skill counts, limitations |
| archive/120625_code_review.md | 59-62, 67, 173, 221-230, 515-538 | paper claims not reproduced; R absent; prioritized action items |
| archive/PAPER_IMPLEMENTATION_GAPS.md | 1-18, 4, 207, 240-249, 249, 450, 642-643, 658 | correct paper attribution; 17 gaps marked complete; existence-based closure criteria |
| archive/implementation/OPEN_QUESTIONS.md | 83-170 | the six gaps the paper omits |
| archive/planning/REQUIREMENTS.md | 1-6 | requirements specification v1.4 purpose |
| archive/planning/VALIDATION_ROADMAP.md | 3, 119, 127-140, 228-234, 246, 247, 289 | validation goal, literature disabled, 20-cycle run, unmeasured accuracy, zero discoveries, honesty principle |
| archive/runbook_critque1.md | 9-10 | external grade C- |
| docker-compose.yml | 13, 31-36 | port 8000 mapping and HTTP healthcheck (at HEAD) |
| docs/CHECKPOINT.md | 17 | synthetic benchmark generator |
| docs/DEEP_ONBOARD.md | 1986, 1991, 1993, 1996, 2001, 2012, 2047, 2048, 2069-2070, 2089, 2122 | Change Impact Index: coupling cluster, blast radii, hotfix counts, safe changes |
| docs/REQUIREMENTS_TRACEABILITY_MATRIX.md | 9-12 | 0 of 293 requirements tested on 2025-11-21 |
| docs/TODO.md | 9-13 | no expert-annotated ground truth |
| docs/domain-roadmaps/biology.md | 1-3, 398-415 | kosmos-figures basis; unchecked success boxes |
| docs/paper/PAPER_REFERENCE_ARCHITECTURE.md | 3-5, 11-15, 63-70, 105-108, 166-184 | the paper's system: runtime, inputs, loop, outputs |
| docs/planning/integration-plan.md | 620-621 | exit criterion: replicate any kosmos-figures discovery |
| docs/planning/objective.md | 62-110, 114-163 | three goals; adoption criteria |
| evaluation/SCIENTIFIC_EVALUATION_REPORT.md | 21-22, 158, 160, 161, 165, 170 | provider used; paper-claim verdicts |
| evaluation/SCIENTIST_NARRATIVE.md | 19, 83 | never wired end to end; 30 papers per query |
| evaluation/artifacts/phase5_scorecard.json | 6-118 | Feb 6 scorecard; LLM can affirm synthetic results |
| evaluation/logs/evaluation_20260207_212853.log | 343-352, 368 | ML template on synthetic data; supported=None |
| evaluation/personas/runs/002_perovskite_solar_cell_researcher/v003_20260208/tier2/TECHNICAL_REPORT.md | 99-108 | perovskite result tests the wrong column pair |
| evaluation/personas/runs/004_climate_data_scientist/v007_20260209/tier1/artifacts/phase2_components/2.1_hypothesis_generation.json | 2, 6 | zero hypotheses yet PASS |
| evaluation/personas/runs/004_climate_data_scientist/v007_20260209/tier1/artifacts/phase2_components/2.4_code_execution.json | 8-15, 23 | execution success false yet PASS |
| evaluation/personas/runs/004_climate_data_scientist/v007_20260209/tier1/artifacts/phase2_components/2.5_data_analysis.json | 6-13 | mock result interpreted as a finding |
| evaluation/personas/runs/004_climate_data_scientist/v007_20260209/tier1/artifacts/phase2_components/2.6_convergence_detection.json | 22-26 | cost always zero |
| evaluation/personas/runs/004_climate_data_scientist/v007_20260209/tier1_scaled/EVALUATION_REPORT.md | 11, 44-54, 62-66, 62-74, 62-64, 63 | duration; 100 actions, 1 experiment, 195 hypotheses, not converged |
| evaluation/personas/runs/004_climate_data_scientist/v007_20260209/tier3/NARRATIVE.md | 20, 30, 46, 48, 52 | literature timeouts; mock result; 0-step protocol; wrong template; $0.00 |
| evaluation/run_phase2_tests.py | 40-45, 325-360 | PASS whenever the function returns; the hard-coded mock result |
| evaluation/scientific_evaluation.py | 258-300, 521-525 | director-driven smoke test; loop_completed hard-coded True |
| examples/README.md | 21-103 | ten examples listed by name |
| k8s/ingress.yaml | 28 | port 8000 |
| k8s/kosmos-deployment.yaml | 25, 91-100 | port 8000 and HTTP probes |
| k8s/kosmos-service.yaml | 11-12 | port 8000 |
| kosmos/agents/data_analyst.py | 376-379, 508-546, 515-520, 543-546, 560-561 | brace-slicing parser; fallback interpretation with None p-value |
| kosmos/agents/experiment_designer.py | 164-299, 380-407, 675, 897 | design_experiment, type selection, LLM-chosen seed, protocol stored through to_dict |
| kosmos/agents/hypothesis_generator.py | 79-83, 360 | novelty defaults and filter; prompt format site |
| kosmos/agents/research_director.py | 68-170, 1248, 1253, 1402-1990, 1469-1530, 1551, 1552-1555, 1575-1583, 1579-1583, 1579, 1589, 1593-1598, 1595-1597, 1626-1641, 1630-1643, 1633-1641, 1653, 1677-1805, 1724-1732, 1807-1990, 1871-1879, 1939-1949, 2422-2440, 2549-2554, 2549-2553, 2942-2968 | constructor, action handlers, decision function, convergence, budget check, status |
| kosmos/api/health.py | 18 | HealthChecker class |
| kosmos/cli/commands/run.py | 51-61, 144-145, 161, 183-202, 196-202, 205-211, 228-229, 399, 414-468, 436-445, 459, 466 | options, budget flag, director construction, step loop, result assembly, metrics |
| kosmos/cli/views/results_viewer.py | 84-115 | hypotheses table |
| kosmos/config.py | 17-18, 631-633 | default model names; default random seed |
| kosmos/core/convergence.py | 194-201, 221-225, 335, 479, 541, 557 | detector signature; optional criteria; supports_hypothesis reads |
| kosmos/core/llm.py | 613-683 | get_client singleton |
| kosmos/core/metrics.py | 157, 176-215, 207, 553-592, 647, 757-762, 762, 785-805, 922-935 | budget flag default, record_api_call, period cost pricing, singleton |
| kosmos/core/pricing.py | 20, 35-36, 35, 54 | pricing table and get_model_cost |
| kosmos/core/prompts.py | 202-497 | EXPERIMENT_DESIGNER template |
| kosmos/core/providers/anthropic.py | 545 | parse_json_response use |
| kosmos/core/providers/base.py | 197, 226, 255, 280, 365, 375-389, 384, 391-402, 402 | abstract methods; usage statistics |
| kosmos/core/providers/litellm_provider.py | 173-205, 195, 407-468 | response parsing and cost; generate_structured |
| kosmos/core/providers/openai.py | 493 | parse_json_response use |
| kosmos/core/utils/json_parser.py | 31-154 | tolerant JSON parser |
| kosmos/core/workflow.py | 57-151, 149-151, 193-197, 204-210 | ResearchPlan; untested list; allowed transitions |
| kosmos/db/__init__.py | 26-27, 72-78, 100, 109-137, 201 | init_database, SQLite branch, get_session, create_all |
| kosmos/db/models.py | 22-28, 31-37, 58, 60, 75, 110-145, 126-128, 222-250 | status enums; Experiment and Result columns; ResearchSession |
| kosmos/db/operations.py | 305-332, 339-372, 391, 394-410 | update_experiment_status, create_result, get_results_for_experiment |
| kosmos/execution/__init__.py | 89, 134 | CodeProvenance export |
| kosmos/execution/code_generator.py | 74-78, 89, 107, 112, 161, 219-221, 231, 241, 246, 313, 364-366, 368, 378, 383, 472, 476, 485, 496, 572, 593-608, 593, 631-636, 700, 768 | templates, kosmos imports, data loading guards, column choice, prompt text, syntax check |
| kosmos/execution/data_analysis.py | 43-150, 153-236, 239-316 | DataAnalyzer methods to inline |
| kosmos/execution/data_provider.py | 310-396 | get_data |
| kosmos/execution/executor.py | 24-29, 43-83, 86-94, 86-110, 223, 517, 573, 589-597, 632-662, 655, 664, 791-822, 869-1008, 1017-1088, 1080 | sandbox availability, restricted builtins, allowed imports, data path handling, retries, execute_protocol_code |
| kosmos/execution/jupyter_client.py | 239-273 | result-marker prior art |
| kosmos/execution/ml_experiments.py | 477 | fitted pipeline placed in results |
| kosmos/execution/provenance.py | 71-148 | CodeProvenance dataclass |
| kosmos/execution/sandbox.py | 117-122, 127-161, 259-277, 306-319, 381-391, 438-450, 442-443 | Docker client, image build, container config, timeout branches, RESULT parser |
| kosmos/experiments/templates/computational.py | 173-179 | always-applicable generic protocol template |
| kosmos/experiments/templates/data_analysis.py | 90-92 | literal variable names |
| kosmos/hypothesis/novelty_checker.py | 67, 350-354 | embedder init; cosine clamp |
| kosmos/hypothesis/refiner.py | 107-167, 155-167, 453-457 | retirement decisions; array parser |
| kosmos/knowledge/embeddings.py | 17-24 | optional import guard |
| kosmos/knowledge/vector_db.py | 401-416 | metadata stored per paper |
| kosmos/literature/base_client.py | 17-24, 45, 230-234 | PaperSource enum; PaperMetadata.id; re-raise on error |
| kosmos/literature/unified_search.py | 170-171 | per-source error isolation |
| kosmos/models/experiment.py | 42-70 | Variable model without a column field |
| kosmos/models/hypothesis.py | 52-53 | statement and rationale minimum lengths |
| kosmos/models/result.py | 42 | ExecutionMetadata.random_seed |
| kosmos/models/safety.py | 97-110, 104 | SafetyIncident with required violation |
| kosmos/monitoring/alerts.py | 277 | health checker import |
| kosmos/orchestration/delegation.py | 388-422, 404-409, 417 | data analysis returns LLM words as statistics |
| kosmos/orchestration/plan_creator.py | 194-198 | mock plan on error |
| kosmos/orchestration/plan_reviewer.py | 160-162 | mock review on error |
| kosmos/safety/__init__.py | 7-16 | exports |
| kosmos/safety/code_validator.py | 36-40, 93 | dangerous modules; Path check on a Mock |
| kosmos/safety/guardrails.py | 46-93 | constructor and signal-handler default |
| kosmos/safety/reproducibility.py | 127-166 | set_seed |
| kosmos/validation/__init__.py | 29-54 | exports |
| kosmos/validation/benchmark_dataset.py | 349-374, 501 | synthetic benchmark at the paper's rates |
| kosmos/validation/null_model.py | 135-139, 166-256, 214-222, 349-369, 436-474 | statistic keys; validate_finding; parametric null; statistic extraction |
| kosmos/validation/scholar_eval.py | 103-125, 204-207, 311-316 | constructor; mock on error; brace parser |
| kosmos/workflow/research_loop.py | 62-63, 79-89, 117-133, 272, 290, 300-301, 426-476 | seed handling; agent wiring; synchronous calls; report generator |
| kosmos/world_model/artifacts.py | 51-85, 199-240 | Finding; save_finding_artifact |
| pyproject.toml | 26-100, 69 | core dependencies; sentence-transformers |
| pytest.ini | 44, 47 | asyncio mode; warnings as errors |
| scripts/smoke_test.py | 18-29, 43-61, 101-133, 161-179 | library-loop imports and tests |
| tests/e2e/test_budget_enforcement.py | 33-59 | budget test model and pricing |
| tests/requirements/core/test_req_configuration.py | 325 | old model name assertion |
| tests/unit/agents/test_research_director.py | 20-26, 512-527 | skip marker; refine decision test |
| tests/unit/agents/test_research_director_loops.py | 23-33, 26-33 | mock_director fixture |
| tests/unit/execution/test_code_generator.py | 31-62, 309-310, 319-320, 328-329, 337, 499-500, 512 | protocol fixtures; DataAnalyzer assertions |
| tests/unit/execution/test_executor.py | 20-23, 330-341 | executor fixture; data path test; sandbox patches |
| tests/unit/literature/test_arxiv_client.py | 98 | expects swallowed error |
| tests/unit/literature/test_pubmed_client.py | 70 | expects swallowed error |
| tests/unit/literature/test_semantic_scholar.py | 107, 223, 230 | expects swallowed errors |
| tests/unit/literature/test_unified_search.py | 122 | per-source isolation test |
| tests/unit/safety/test_guardrails.py | 24-38 | Mock config pattern |
| tests/unit/workflow/test_research_loop.py | 215-220, 256-259, 287-292, 466-471 | AsyncMock on synchronous methods |

## 12. Appendix B: assumptions

1. The code state is commit 6cfe7f6 on master. The working tree's uncommitted changes to docker-compose.yml, human_review_audit.jsonl and the literature cache, and the untracked documents, are not part of the assessed state.
2. The live provider is LiteLLM with deepseek/deepseek-chat; ANTHROPIC_API_KEY is never set; every change is written for that path and the legacy ClaudeClient is not exercised.
3. sentence-transformers does not import in the owner's environment, so the novelty path runs on zero vectors today.
4. The Docker daemon is reachable, the docker SDK is installed (so SANDBOX_AVAILABLE is True), and the kosmos-sandbox image is not built as of 2026-10-01.
5. The evaluation datasets under evaluation/data are constructed CSVs with planted effects, not observations; the expected statistics quoted (Pearson r about 0.93 for CO2 against temperature) describe the planted data.
6. The Change Impact Index numbers in docs/DEEP_ONBOARD.md (blast radius, risk, hotfix counts) are taken as given and were not recomputed.
7. Line numbers are exact at 6cfe7f6. Once an earlier change edits a file, later changes in that file are located by the quoted code rather than by number.
8. kosmos.db at the repository root is the owner's database. It was queried read-only; the rows cited (result d39936bb, experiment fa9ac1a3, hypothesis 41c8c967, 7 experiments, 15 hypotheses, 1 result) existed on 2026-10-02.
9. The February 2026 persona runs and the 17-agent verification journal are accepted as evidence only where this report re-read the cited lines; journal findings that were not re-read are not cited.
10. Effort estimates assume one engineer who has read docs/DEEP_ONBOARD.md, and exclude the Docker build and the live run except where marked.
11. ExperimentProtocol.model_validate(protocol.to_dict()) is assumed to round-trip for designer-emitted protocols; the P0-4 fixture exposes any mismatch.
