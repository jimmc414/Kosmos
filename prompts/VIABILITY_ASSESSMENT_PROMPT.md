# Kosmos Viability Assessment and Change Plan Prompt

## Your role

You are a principal engineer and research lead. The Kosmos repository is checked out at /mnt/c/python/kosmos on commit 6cfe7f6 (master). If you are not working in that checkout, say so and stop.

Your task has three parts:

1. State the intent behind the project: what its authors were trying to build and what a successful version would do.
2. Decide whether Kosmos is viable as a working autonomous-research tool, or whether its value is limited to research and reference.
3. If it is viable, specify the changes that would most improve the quality of its research output.

The deliverable is a single report that a coding model can execute without further clarification.

## What Kosmos is meant to be

Kosmos is an open-source implementation of the architecture in "Kosmos: An AI Scientist for Autonomous Discovery" (arXiv 2511.02824). The intended loop is: generate hypotheses from literature and data, design experiments, generate and execute analysis code in a sandbox, interpret results, refine hypotheses, and stop at convergence, producing validated findings with provenance. Establish the intent yourself from README.md, paper/, archive/PAPER_IMPLEMENTATION_GAPS.md, archive/120525_implementation_gaps_v2.md, docs/, evaluation/personas/definitions/, and the commit history. Write the intent section in your own words, with citations, before you assess the current state.

## Verified state of the code as of 2026-10-01

A 17-agent source verification established the facts below at commit 6cfe7f6. Each was confirmed by an independent reviewer. Re-verify any fact before you build a recommendation on it, and cite file:line in your report.

### Architecture

- Two independent orchestrators exist and never call each other. `kosmos run` (kosmos/cli/commands/run.py:183-202) drives ResearchDirectorAgent (kosmos/agents/research_director.py) over the state machine in kosmos/core/workflow.py. The README Python quickstart, scripts/smoke_test.py and kosmos/workflow/ensemble.py drive a different class with the same name in kosmos/workflow/research_loop.py, a plan, review, delegate, ScholarEval cycle loop. They share the worker agent classes, the get_client() LLM singleton, the SQLite database and the EventBus, nothing else. evaluation/scientific_evaluation.py uses the director.
- The director calls workers directly from its _handle_*_action methods (research_director.py:1402-1990). The BaseAgent message bus and AgentRegistry carry no traffic.
- Persistence is SQLite at <repo>/kosmos.db through DatabaseConfig.normalized_url (kosmos/config.py:269-300), with hand-written Pydantic to ORM conversion at four sites. A world model mirrors entities to Neo4j or an in-memory fallback (kosmos/world_model/factory.py:105-134).
- One LLM client comes from kosmos/core/llm.py get_client(), with the provider chosen by LLM_PROVIDER. The owner's environment uses LLM_PROVIDER=litellm with deepseek/deepseek-chat and sets no ANTHROPIC_API_KEY. Recommendations must work on the LiteLLM path.

### Confirmed defects that block real output on the CLI path

1. Generated experiment code cannot run. All five code templates import kosmos (kosmos/execution/code_generator.py:107,161,241,313,378,472,700). The sandbox image never installs kosmos (docker/sandbox/Dockerfile:30) and restricted exec forbids importing it (kosmos/execution/executor.py:86-110). Sandbox return values are read only from a "RESULT:" stdout line that no template prints (kosmos/execution/sandbox.py:438-450). Retry rewrites turn failures into success=True error dicts (executor.py:869-1008). The director never reads exec_result.success and stores {} as the result (research_director.py:1589,1626-1653).
2. The director state machine dead-ends in REFINING. _handle_refine_hypothesis_action never transitions (research_director.py:1807-1990), and decide_next_action returns REFINE_HYPOTHESIS whenever any hypothesis has been tested (2549-2554). Persona runs show 100 actions with 1 experiment completed (evaluation/personas/runs/004_climate_data_scientist/v007_20260209/tier1_scaled/EVALUATION_REPORT.md:62-66).
3. Analysis verdicts are never persisted. create_result is called without supports_hypothesis (research_director.py:1633-1641) and nothing updates it afterward, so refinement and the FDR correction read NULL.
4. The first exception in any handler aborts the run. _handle_error_with_recovery blocks the event loop with run_coroutine_threadsafe(...).result() and raises TimeoutError, which `except RuntimeError` does not catch (research_director.py:676-685). The 3-strike breaker, the ERROR state and ERROR_RECOVERY are unreachable. The CLI prints "Research failed:" with an empty message.
5. Novelty filtering is broken while sentence-transformers fails to import, which is the case in the owner's environment. Zero-vector cosine gives NaN clamped to similarity 1.0 (kosmos/hypothesis/novelty_checker.py:350-354,387-391), so every new hypothesis in a domain with existing database rows is dropped (kosmos/agents/hypothesis_generator.py:211-219). The vector-search branch always returns [] because PaperMetadata is built without its required id (novelty_checker.py:242-258).
6. Convergence is fed empty data. The hypotheses load imports a nonexistent HypothesisModel (research_director.py:1253) and results are never loaded, so the optional stopping criteria are inert.
7. Budget controls do nothing. --budget is stored but never read (kosmos/cli/commands/run.py:144-145,161). Enforcement requires configure_budget(), which nothing in production calls (kosmos/core/metrics.py:157,553-588). record_api_call has no production caller.
8. The safety layer is off the live path. The director runs generated code without CodeValidator, guardrails or human review (research_director.py:1551,1578-1583). Those run only in execute_protocol_code (executor.py:1017-1088), where the emergency-stop path raises a pydantic ValidationError (kosmos/safety/guardrails.py:220-298 against kosmos/models/safety.py:104) and process SIGINT and SIGTERM handlers are replaced.

### Confirmed defects on the library loop

- The null-model gate fails any finding whose statistics lack a recognized test statistic (kosmos/validation/null_model.py:188-205,349-369; kosmos/validation/scholar_eval.py:171-182). DelegationManager never emits one, so only blank findings validate.
- data_analysis tasks pass a dict where an ExperimentResult is required and silently complete empty (kosmos/orchestration/delegation.py:404-422; kosmos/agents/data_analyst.py:343). literature_review tasks return corpus_size 0 (kosmos/agents/literature_analyzer.py:695-702).
- Planner, reviewer and ScholarEval fall back to approving mock results on any LLM error (plan_creator.py:194-198; plan_reviewer.py:160-162; scholar_eval.py:204-207). All agent calls are synchronous inside async functions, so there is no real concurrency and no effective task timeout.

### Deployment and tooling

- The Docker image build fails: Dockerfile:44-48 copies only pyproject.toml and README.md, but pyproject data-files require .env.example and alembic.ini. No HTTP server exists, although docker-compose and k8s probe :8000/health. fastapi, litellm, the docker SDK and a Postgres driver are not declared dependencies.
- About 27 percent of kosmos/, roughly 23k lines, has no production call path. Examples: core/domain_router.py, core/feedback.py, core/memory.py, validation/{failure_detector,accuracy_tracker,accuracy_validator,benchmark_dataset}.py, safety/verifier.py, knowledge/{graph_builder,graph_visualizer}.py, execution/{notebook_generator,figure_manager,production_executor}.py, analysis/{plotly_viz,statistics}.py, and the kosmos/domains analyzers.
- Known failing tests at HEAD: tests/requirements/core/test_req_configuration.py:325 asserts an old model name; 7 of 16 tests in tests/unit/workflow/test_research_loop.py use AsyncMock on now-synchronous methods; all 29 tests in tests/unit/safety/test_guardrails.py fail on a Mock config; 5 literature error tests broke when _handle_api_error began re-raising. README.md still claims 3704 passing tests.

### What works

Configuration loading and validation; the provider abstraction for Anthropic, OpenAI and LiteLLM; literature search fan-out with deduplication and caching; hypothesis generation through the LLM with database storage; template-based and LLM-based protocol design with power analysis and validation; the Docker sandbox container configuration itself; ArtifactStateManager JSON artifacts; ScholarEval scoring when given a client; the event bus and stage tracker; the CLI scaffolding.

## Source materials

- docs/DEEP_ONBOARD.md, untracked, generated 2026-04-10 from this commit. Its Critical Paths, Gotchas and Change Impact Index sections are accurate except for the items below. Treat its evidence text as more reliable than its claim text.
- Known-wrong items in that document: it says a missing Docker SDK falls back silently to host exec, but the kosmos.execution import fails outright and a stopped daemon raises RuntimeError; it says SAFE_BUILTINS includes getattr, which it never did; the state machine has 33 edges, not 28; literature cache eviction is oldest-written, not LRU; `kosmos cache` uses the same CWD .kosmos_cache as the runtime, and only info and doctor use ~/.kosmos_cache; the CLI --stream display prints directly with no queue; tests/conftest.py never resets config or the event bus.
- docs/xray.md: structural index, import graph and class skeletons.
- evaluation/CRITICAL_EVALUATION_REPORT.md and evaluation/*_findings.md, dated 2026-02-12. Findings F-01, F-02 and F-04 about the mocked pipeline were fixed by commit 3ff33c3. F-03 (two orchestrators), F-05 (silent mocks), F-26 (no checkpointing) and F-35 (seed never propagated) remain true.
- evaluation/personas/runs/*/EVALUATION_REPORT.md: the only empirical evidence of real runs. Read at least v007 of persona 004 and the genomics persona runs before judging what the system produces today.
- Full verification output: /home/jim/.claude/projects/-mnt-c-python-kosmos/47a87f3f-fa9c-4f95-9328-1706409442a8/subagents/workflows/wf_8880eb92-262/journal.jsonl. One JSON line per agent, with claim verdicts and about 150 additional file:line findings.

## Method

1. Read the intent sources first and write the intent section before reading the defect lists, so the assessment is anchored on purpose rather than on bugs.
2. Define viability criteria explicitly before applying them. At minimum: can a user with a dataset and a research question obtain, within one run, at least one finding that (a) came from code that actually executed on that data, (b) carries a correct test statistic and p-value, (c) is reproducible from a seed and a provenance record, (d) survived a non-trivial validation gate, and (e) is reported honestly alongside failed tasks and cost. Add any criteria the intent sources imply.
3. Reconstruct one real run end to end from the persona reports and the code, hop by hop, and mark the first hop where real science stops. Do not launch `kosmos run` to find out; it makes paid DeepSeek calls, and the owner has not authorized spend. If you must execute anything, prefer unit-level checks with mocked LLM clients.
4. Separate structural problems, fixable by wiring and bug fixes, from fundamental ones, which are limits of the design or of LLM-driven science. Say which category each major defect falls in.
5. Decide. Use exactly one of: VIABLE, CONDITIONALLY VIABLE with the conditions stated, or RESEARCH ONLY. Give the three to five reasons that most determine the verdict.
6. If VIABLE or CONDITIONALLY VIABLE, produce the change plan. Order changes so that a coding model can execute them sequentially and each leaves the test suite no worse. Group them into phases:
   - P0: one real experiment runs end to end on the CLI path, and its result is stored with success status, statistics and provenance.
   - P1: the loop closes. Analysis verdicts persist, the state machine leaves REFINING, error recovery works, convergence sees real data, and the budget is enforced.
   - P2: quality of scientific output. Novelty works on the LiteLLM path with sentence-transformers either fixed or replaced; the null-model and validation gates pass real findings and fail fabricated ones; seed and provenance propagate; failed tasks and cost are reported honestly.
   - P3: hardening and cleanup. The safety layer is on the live path, deployment builds and serves, dead code is archived, and docs and README are corrected.
   For each change give: an ID; the goal in one sentence; the files and functions to touch with current line references; the current behavior; the required behavior; an acceptance test that runs without live LLM calls where possible, or is marked as needing a live call; dependencies on other change IDs; risk and blast radius, using the Change Impact Index in docs/DEEP_ONBOARD.md; and an effort estimate in hours.
7. If RESEARCH ONLY, list what is worth preserving as reference, what should be archived, and what a from-scratch rebuild should reuse.
8. Decide what to do about the second orchestrator. Recommend consolidating on one and specify which pieces of the other to port.
9. Define output-quality metrics the owner can track after the changes, for example experiments executed per run, findings with a real test statistic, null-model pass rate on real versus shuffled data, reproducibility rate across seeds, and cost per validated finding. Say where in the code each metric would be emitted.

## Constraints

- Cite file:line for every factual claim about the code. Re-read the code at the cited line before citing it; the line numbers above are exact at commit 6cfe7f6 but may drift if the working tree changes.
- Do not propose changes to dead code paths unless reviving that path is the recommendation, and say so explicitly.
- Keep the LiteLLM provider as the primary target. Do not require ANTHROPIC_API_KEY; the owner's policy forbids setting that variable.
- Do not run the full pytest suite as a check. pytest.ini enforces an 80 percent coverage gate and treats warnings as errors, and tests/conftest.py loads .env with override, so e2e tests make live calls. Use targeted runs with --no-cov and -p no:cacheprovider.
- Do not commit, push, or modify files other than the report you write.
- Write for a coding model: imperative and specific, with no "consider" or "might". Each change must be executable from its own text without re-reading this prompt.

## Report format

Title: Kosmos Viability Assessment and Change Plan, with the commit hash and the date.

Sections, in order:

1. Verdict
2. Intent
3. Current state and one-run trace
4. Viability analysis against the criteria
5. Change plan by phase, or Salvage plan if the verdict is RESEARCH ONLY
6. Orchestrator consolidation
7. Output-quality metrics
8. Verification plan with exact commands and expected outcomes
9. Non-goals and things not to change
10. Open questions for the owner
11. Appendix A: evidence index of every file:line cited
12. Appendix B: assumptions

Length: as long as needed for a coding model to execute every change without asking questions, and no longer. Use tables for change lists. Keep each narrative section under 400 words.
