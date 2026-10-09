# Stages S1–S4: P2 remainder · P3 hardening · reporting and reruns · live acceptance — Execution Spec
> Implements: evaluation/VIABILITY_ASSESSMENT_AND_CHANGE_PLAN.md §5 items P2-4, P2-7, P3-1, P3-4,
> P3-2, P3-3, P3-5, C-1, M-1 and §8 step 25; verification §8 steps 14, 17–24 · Governed by:
> docs/MAP.md §2 rows S1–S4 · Status: FROZEN 2026-10-04 (s0, JUDGMENT Fable high) · Displaces:
> evaluation/VIABILITY_PROGRESS.md (frozen in place as the record of P0–P2-3; not archived) ·
> Displaced by: the next freeze (an interstitial when S4 waits on the owner).

**Mission.** When this plan is exhausted, `kosmos run` on the climate CSV produces a validated
Pearson finding with honest cost, the package contains only what runs, the unit suite is green
without a baseline, `kosmos report`, `kosmos rerun` and `kosmos validate-null` exist, the README
tells the truth, and the owner has signed one live run's report.

**Self-contained.** This spec + docs/execution/BACKLOG.md rows by id + the plan §5 item text
each milestone names + the frozen tracker's Notes for the items it depends on + the files it
anchors = everything needed. Read anchored files before editing; import no prior-session
assumptions. Plan §5 line numbers are exact at 6cfe7f6 only: locate by the quoted code.

## Tag vocabulary
BUILD·high · BUILD·xhigh · JUDGMENT·high · JUDGMENT·xhigh · OWNER · CALENDAR; qualifier
EXPLORE (a whole session). Every milestone below is BUILD·high unless its heading says otherwise
(MAP D-22).

## How a session picks its milestone
1. Lane 0 fires by its own triggers, never by lane order.  2. NEXT ACTION if its tag matches.
3. Lane 1 first open row for your tier, in printed order (= tracker order, MAP D-13).  4. Lane 2
under the WIP cap (one lane in flight beside lane 1), printed order.  5. BACKLOG by routing.
6. No open row ⇒ write a successor worklist (/factory-spec interstitial); never stop, never
manufacture a stage.

## Locked decisions (do not relitigate) — each names its durable home
1. Every plan §10 decision, D-01 to D-12 (docs/MAP.md §4). The ones a milestone here touches:
   D-01 unbound hypotheses excluded only (M1 reports them as hypotheses_untestable); D-04 Tier C
   kept (M4 moves Tiers A and B only); D-05 Tier B archived in-tree with its tests (M4);
   D-06 synthetic rows never supported (M1's results table shows Data synthetic); D-07
   ScholarEval advisory (M8 stores and shows it, never gates); D-08 artifacts under
   ./artifacts/runs/<run_id>/ (M8 findings JSON, M10 report); D-11 fail fast on
   SandboxUnavailable (M3 adds the validator before execution, not around it).
2. Order and discipline: D-13 (tracker order, P3-4 before P3-2), D-15 (one commit per item,
   `s<N> — <ID>: <summary>`, push every session), D-16 (never stage the owner's paths),
   D-17 (test command; tests/e2e never in the ladder), D-18 (live calls only in M10, `--budget 1`),
   D-19 (ANTHROPIC_API_KEY never set).
3. D-21 (ADR-0002): gate 2 judges against scripts/test_baseline.txt until M6 retires it; every
   re-stamp carries its cause in the commit.
4. D-22: every row here is BUILD·high except F1 (JUDGMENT·high) and the OWNER-gated steps of
   M5 and M10; D-24: the runner uses the skip-permissions flag with the deny-list.
5. s0 freeze decisions (this file is their home): (a) M5 splits into M5a (everything but the
   compose edit) and M5b (the compose edit, once OWNER#compose-edit-approval is RULED) so an
   unruled ask never idles a session; (b) M6 may span several sessions, each commit re-stamping
   a SMALLER baseline with the cause "P3-3 part n: <what was fixed>", and the last commit
   retires it; (c) M4 lands as two commits (Tier A, Tier B) in one or two sessions, both under
   VIAB#P3-4; (d) M10 is launched only by a session that holds the live-spend ruling (Lane 0).

## Verified codebase facts — checked 2026-10-04 (s0); re-read before editing
- Result assembly the CLI still does inline: `kosmos/cli/commands/run.py:499` (`hypothesis.to_dict()`
  on ORM rows), `:506` (`experiment.to_dict()`), `:525` (`getattr(director.llm_client, 'total_requests', 0)`).
- `kosmos/cli/views/results_viewer.py:84` display_hypotheses_table, `:172` display_experiments_table,
  `:267` display_metrics_summary (no results table yet).
- `kosmos/core/providers/base.py:375` get_usage_stats (has total_cost_usd), `:391` _update_usage_stats.
- `kosmos/core/metrics.py:176` record_api_call, `:217` get_api_statistics, `:747` _calculate_period_cost,
  `:788` enforce_budget.
- `kosmos/db/operations.py:351` create_result, `:468` update_result_validation (takes `cost_usd`, P2-0),
  `:504` get_results_for_run.
- `kosmos/agents/research_director.py:56` NEAR_DUPLICATE_SIMILARITY = 0.85, `:166` artifacts_dir,
  `:308` _save_run_code, `:1490` num_hypotheses default 3, `:1612` _handle_execute_experiment_action,
  `:1633` CodeExecutor( construction, `:1848` _db_result_to_experiment_result, `:1888` _validate_result
  (the P2-2 gate), `:1963` _handle_analyze_result_action, `:2130` _leave_refining, `:2150`
  _is_near_duplicate, `:2175` _handle_refine_hypothesis_action, `:2755` decide_next_action,
  `:3291` get_research_status.
- `kosmos/hypothesis/refiner.py:108` evaluate_hypothesis_status, `:405` spawn_variant.
- `kosmos/core/workflow.py:57` ResearchPlan, `:73` untestable_hypotheses, `:126` mark_untestable.
- `kosmos/safety/code_validator.py:28` CodeValidator, `:93` `if path and Path(path).exists():`,
  `:160` validate. `kosmos/safety/guardrails.py:49` enable_signal_handlers=True default, `:95`
  _register_signal_handlers, `:227` trigger_emergency_stop. `kosmos/models/safety.py:104`
  `violation: SafetyViolation` (required). `kosmos/execution/executor.py:1074` execute_protocol_code,
  `:1099` SafetyGuardrails(), `:1119` CodeValidator(allow_file_read=True), `:710` seed_prelude.
- Tier A and Tier B modules of plan P3-4 all exist at their listed paths (checked by `ls`);
  `kosmos/execution/__init__.py:89` exports CodeProvenance; `pytest.ini:10` testpaths = tests
  (no norecursedirs); `pyproject.toml:230` coverage omit list.
- `Dockerfile:44` `COPY pyproject.toml README.md ./`, `:98` HEALTHCHECK, `:102` CMD --help;
  `pyproject.toml:141` execution extra, `:151` embeddings extra (P2-5), `:205` data-files;
  `docker-compose.yml:13` 8000:8000, `:32-33` healthcheck with requests; k8s/ holds configmap,
  hpa, ingress and more; `kosmos/api/health.py:18` HealthChecker (kept). `.dockerignore` exists:
  it excludes .git/, htmlcov/, logs/, *.db, .env and `.env.*`, and `*.md` except README.md, but NOT
  neo4j_data/ (518 MB), postgres_data/, redis_data/, .literature_cache/, artifacts/ or chroma_db/;
  its `.env.*` rule also drops the .env.example that P3-2's Dockerfile COPY needs (checked 2026-10-04).
- `tests/requirements/core/test_req_configuration.py:325` asserts "claude-3-5-sonnet-20241022";
  `kosmos/literature/base_client.py:222` _handle_api_error; no .github/workflows/ directory.
- README.md: `:3` and `:394` cite Lu et al.; `:7` paper_gaps badge; `:8` tests-3704 badge; `:56`
  quickstart imports kosmos.workflow.research_loop. archive/PAPER_IMPLEMENTATION_GAPS.md exists.
- `kosmos/world_model/artifacts.py:51` Finding (`:61` notebook_path, `:79` code_provenance),
  `:199` save_finding_artifact, `:146` ArtifactStateManager;
  `kosmos/workflow/research_loop.py:426` generate_report (port source; archived by M4 Tier B —
  M8 reads it from `git show 1247379:kosmos/workflow/research_loop.py` or archive/code after M4).
- `kosmos/validation/analysis_fn.py:44` build_analysis_fn, `:133` shuffle_target;
  `kosmos/validation/null_model.py:117` NullModelValidator, `:166` validate_finding;
  `kosmos/cli/main.py:36` app, `:173-174` the `version` command (the pattern for new commands).
- The ladder: scripts/verify.sh (4 gates; about 8 min, 471 s measured s0); scripts/test_baseline.txt 394 node ids
  stamped at dc62f8a; the climate CSV truth r 0.9317, p 5.77e-29, n 64 (gate 4).

## Design
Each milestone's design is plan §5's "Required" paragraph for that item, read in full at
session start, plus the pre-decisions written under the milestone. New files: plan §5 names
them (kosmos/cli/commands/run_results.py, tests/unit/cli/test_run_results.py,
tests/unit/core/test_metrics_bridge.py, tests/unit/agents/test_pool_control.py,
archive/code/README.md, scripts/check_env.py, .github/workflows/unit.yml,
tests/unit/cli/test_report.py, tests/unit/cli/test_metric_commands.py).

## Milestones — lanes; ONE per session; verify each before the next

## Lane 0 — calendar / stop-the-line (READ-ONLY table)
| # | Event | Owner·tag | Trigger | What a session does | Where the reading is recorded |
|---|---|---|---|---|---|
| L0-1 | Live-spend authorization for M10 (`OWNER#live-spend-authorization`) | Owner · OWNER | the session that reaches M10 | Launch `kosmos run` with a live provider ONLY when the ask is RULED in SESSION_STATE §Waiting-on-owner or a PRE-ANSWERED line in the prompt says "launch"; otherwise do not launch, write HALT_NEEDS_OWNER (runner) or report (attended), and take the next open row | SESSION_STATE §Waiting-on-owner; backlog/OWNER-ruled.md |
| L0-2 | Compose edit approval for M5b (`OWNER#compose-edit-approval`) | Owner · OWNER | the session that reaches M5 | Unruled ⇒ do M5a only, leave M5b open; never stage docker-compose.yml without the ruling | same |
| L0-3 | Signals RED on docker, sandbox-image or kosmos-db | Session · stop-the-line | any session start | Record a SIGNAL# row; gates 3/4 cannot pass ⇒ the ladder is red by the environment: do not fix the environment mid-milestone beyond `docker build -t kosmos-sandbox:latest docker/sandbox` (D-09); if the ladder cannot go green, HALT_RED_GATE | the session block's `signals:` line; BACKLOG |

## Lane 1 — the critical path (never capped, always first)

### M1 — P2-4 Honest run report and real cost — BUILD·high — M — closes `VIAB#P2-4`
Design: plan §5 P2-4 "Required" (1)–(4) verbatim. Pre-decided: the new module is
kosmos/cli/commands/run_results.py with `build_run_results(director, question, max_iterations)
-> Dict`; results rows come from get_results_for_run(session, director.run_id) (P2-0) reading
the COLUMNS (execution_success, data_source, validation_status, …), not the data JSON; cost per
result is written through update_result_validation(cost_usd=…) (P2-0 Notes); P1-5 owns the
provider→metrics bridge, so (4) only adds `cost_usd` to record_api_call and sums it;
hypotheses_untestable = len(plan.untestable_hypotheses) (D-01). Keep `kosmos run`'s exit code
and the A-2 provider line in the panel (tracker row 14 Notes).
*Verify:* `python -m pytest tests/unit/cli/test_run_results.py tests/unit/core/test_metrics_bridge.py --no-cov -p no:cacheprovider -q`
→ exit 0; cost_per_validated_finding == 0.0123; estimated_cost_usd == get_model_cost('deepseek/deepseek-chat', 1000, 500)
(plan §8 step 14); `bash scripts/verify.sh` → `VERIFY PASS (4 gates)`.
*DoD:* ladder green · explicit-path commit `s<N> — P2-4: …` · SESSION_STATE.md updated ·
`VIAB#P2-4` flipped to done in backlog/BUILD-inplan.md in the same commit · tracker-style
Notes of any deviation recorded in the session block.

### M2 — P2-7 Hypothesis-pool control — BUILD·high — M — closes `VIAB#P2-7`
Design: plan §5 P2-7 "Required" (1)–(6) verbatim. Pre-decided: the duplicate filter is the
existing `ResearchDirectorAgent._is_near_duplicate` with NEAR_DUPLICATE_SIMILARITY (P2-5 Notes),
counted into `variants_dropped_duplicate`; untested ordering excludes untestable_hypotheses
(P2-1); the REFINING branch edit sits after `_leave_refining` (P1-2) and must keep that exit;
`evaluate_hypothesis_status` returns CONTINUE_TESTING for supports_hypothesis None unless
result.status == SUCCESS. D-10 is superseded by this milestone's landing (note it in the block).
*Verify:* `python -m pytest tests/unit/agents/test_pool_control.py --no-cov -p no:cacheprovider -q`
→ 2 passed; the pool never exceeds 12 over 60 actions (plan §8 step 17); ladder green.
*DoD:* as M1, for `VIAB#P2-7`. Then check MAP §2 S1's exit gate: steps 14 and 17 green and the
ladder green ⇒ THIS session makes the stage-gate commit (S1 done, S2 active, changelog row).

### M3 — P3-1 CodeValidator and emergency stop on the director path — BUILD·high — M — closes `VIAB#P3-1`
Design: plan §5 P3-1 "Required" (1)–(5) verbatim. Pre-decided: the rejected-unsafe row goes
through create_result(..., validation_status='rejected_unsafe', execution_success=False, code=code)
(P2-0's ValueError rule: the experiment must exist); the P2-2 gate `_validate_result` then
reads reason execution_failed for it; signal handlers default off and chain. Acceptance (a):
the 29 guardrails tests pass unmodified — they are in scripts/test_baseline.txt today and will
show as "retired" in the judge output; that is the expected evidence, not a red.
*Verify:* `python -m pytest tests/unit/safety --no-cov -p no:cacheprovider -q` → exit 0 (29 + 4
new) (plan §8 step 18); ladder green.
*Amended s3 (CLAUDE.md §AMENDMENT):* step 18 cannot exit 0 at M3. tests/unit/safety also holds
27 baseline reds outside P3-1's files: test_verifier.py, 26 errors (its fixture builds
ExecutionMetadata without duration_seconds, python_version and platform), and
test_reproducibility.py::TestConsistencyValidation::test_validate_different_types, 1 failure (int −
str in validate_consistency). M3's verify is therefore test_guardrails.py (29) + the 4 new tests
passing, with those 27 ids unchanged. Step 18's exit 0 moves to M6, which fixes every remaining
baseline id (TEST#safety-suite-baseline-reds). The S2 exit gate is unchanged. Plan §5 P3-1 says
the 29 tests are fixed by its step (2); they also need the guardrails' numeric config reads to
tolerate the tests' Mock config (max_cpu_cores is a Mock), and a cwd-isolating
tests/unit/safety/conftest.py, because the tests leave the stop flag file in the cwd.
*DoD:* as M1, for `VIAB#P3-1`.

### M4 — P3-4 Archive zero-importer modules (Tiers A and B; Tier C kept) — BUILD·high — M — closes `VIAB#P3-4`
Design: plan §5 P3-4 "Required" verbatim: Tier A `git mv` to archive/code/<same path> with
archive/code/README.md (module · the commit that removed its last importer · reason); prune the
__init__ exports (kosmos/execution/__init__.py, kosmos/validation/__init__.py,
kosmos/safety/__init__.py); Tier B in a second commit including scripts/verify_e2e.py and
tests/unit/{workflow,orchestration,compression}; archived tests move to archive/code/tests/<same
path>; `norecursedirs = .* build dist node_modules venv archive` in pytest.ini and `"archive/*"`
in the coverage omit list. Tier C stays (D-04). Pre-decided: scripts/smoke_test.py imports
research_loop and compression (tracker row 18 Notes) — it is Tier B's last importer and moves
with it; kosmos/world_model/artifacts.py, scholar_eval.py, null_model.py, skill_loader.py,
api/health.py stay. The baseline's ids for archived tests become "retired" in the judge output
(expected). Run `python -m pytest tests/unit --co -q --no-cov -p no:cacheprovider | tail -3`
after each commit: no collection errors.
*Verify:* `python -c "import kosmos, kosmos.execution, kosmos.validation, kosmos.safety, kosmos.agents.research_director; print('ok')"`
→ ok; `rg -l "domain_router\|failure_detector\|accuracy_tracker\|notebook_generator\|production_executor\|graph_visualizer\|plotly_viz" kosmos/`
→ no files; collection → "N tests collected" with no errors (plan §8 step 21); ladder green after each commit.
*DoD:* as M1, for `VIAB#P3-4` (two commits `s<N> — P3-4: Archive Tier A …` / `… Tier B …`).

### M5 — P3-2 Image build, declared dependencies, no HTTP probe — BUILD·high — S — closes `VIAB#P3-2` (M5b OWNER-gated)
Design: plan §5 P3-2 "Required" (1), (2), (4) and the k8s and api parts of (3) are **M5a**;
the docker-compose.yml edit (healthcheck `["CMD","python","-m","kosmos.cli.main","version"]`,
drop `8000:8000` and its comment) is **M5b** and needs `OWNER#compose-edit-approval` RULED
(Lane 0 L0-2). Pre-decided: extend .dockerignore with neo4j_data/, neo4j_logs/, neo4j_import/,
neo4j_plugins/, postgres_data/, redis_data/, .literature_cache/, artifacts/runs/, chroma_db/,
.chroma_db/ and the negation `!.env.example` (its `.env.*` rule would otherwise drop the file the
new `COPY .env.example` needs), so `docker build -t kosmos:test .` ships neither the volumes nor a
broken context; `litellm>=1.40.0` to core; extras `server`, `postgres` (the
`embeddings` extra exists since P2-5); scripts/check_env.py (not `kosmos doctor`, which A-2
already extended) reporting provider, model, litellm importable, Docker reachable,
sentence_transformers importable, DB URL, the pinned sandbox tag; fold
HYG#pyproject-ruff-top-level-keys into the pyproject edit. kosmos/api/streaming.py and
websocket.py are already gone after M4.
*Verify:* `docker build -t kosmos:test . && docker run --rm kosmos:test python -m kosmos.cli.main version`
→ a version string; `pip install -e . --dry-run` → litellm resolved; `python -c "import kosmos.cli.main"`
in an env without fastapi → exit 0 (plan §8 step 19; the no-fastapi check may use
`python -c "import sys; sys.modules['fastapi']=None; import kosmos.cli.main"`); ladder green.
*Ruled s3 (OWNER#compose-edit-approval):* M5 runs whole; M5b commits the owner's `${VAR:?}`
hardening as-is together with the edit, staged by name. Genesis (SESSION_STATE §STANDING) binds
every Docker step: the .dockerignore edit lands BEFORE the first `docker build .`; the compose
file is checked only with `docker compose --profile prod config --quiet`, never `up`/`down`;
`docker inspect kosmos-postgres -f '{{.Id}} {{.State.StartedAt}}'` is read before and after and
must match; the build respects the Genesis no-go windows; scripts/init_db.sql never moves.
*DoD:* M5a: as M1 but `VIAB#P3-2` stays `in-plan, blocked(OWNER#compose-edit-approval)` until
M5b; M5b (XS, once ruled): the compose edit staged by name in its own commit, `VIAB#P3-2` →
done, OWNER#compose-edit-approval → ruled in backlog/OWNER-ruled.md.

### M6 — P3-3 Test suite green for surviving modules; README test count removed; retire the baseline — BUILD·high — L — closes `VIAB#P3-3`, `FACTORY#retire-test-baseline`
Design: plan §5 P3-3 "Required" (1)–(5) verbatim, plus: every remaining red node id in
scripts/test_baseline.txt is fixed or its test archived with its module (D-05 for Tier B
tests); TEST#order-dependent-caplog, TEST#director-tests-write-configured-db (a tests/conftest.py
fixture pointing DATABASE_URL at tmp_path), TEST#prioritizer-fixture-rationales,
TEST#validation-pipeline-parametric-null and HYG#makefile-bare-pytest are folded in. Size:
394 ids at s0, roughly 150 of them in Tier A/B tests that M4 archives and 29 that M3 fixes;
expect two or three sessions. Pre-decided split (Locked decisions 5b): each session's commit
re-stamps a smaller baseline with `--accept-baseline "P3-3 part n: fixed <files>"`; the LAST
commit empties the baseline (stamp line "retired by P3-3 at <hash>") and edits
scripts/verify.sh gate 2 so a bare `exit 0` of the judge with an empty baseline is the pass
(no script change is needed if the file holds only the stamp). (5) .github/workflows/unit.yml
runs `python -m pytest tests/unit --no-cov -p no:cacheprovider -q` on push with no secrets
(tests/conftest.py loads .env with override, so the job has none).
*Verify:* `python -m pytest tests/requirements/core/test_req_configuration.py tests/unit/safety/test_guardrails.py tests/unit/literature --no-cov -p no:cacheprovider -q`
→ exit 0 (plan §8 step 20); `python -m pytest tests/unit/execution tests/unit/agents tests/unit/core tests/unit/db tests/unit/validation tests/unit/hypothesis tests/unit/safety tests/unit/cli --no-cov -p no:cacheprovider -q`
→ exit 0 (step 24); scripts/test_baseline.txt holds only its stamp line; ladder green.
*DoD:* as M1, for `VIAB#P3-3` and `FACTORY#retire-test-baseline` (same commit); ADR-0002 gets
an append-only correction note "retired <date> <hash>".

### M7 — P3-5 README and DEEP_ONBOARD corrections — BUILD·high — S — closes `VIAB#P3-5`
Design: plan §5 P3-5 "Required" verbatim, plus the A-2 flags (`--provider`, `--model`) in the
README provider section (tracker row 27 Notes) and the ladder (`bash scripts/verify.sh`) in the
Verify section. docs/DEEP_ONBOARD.md is UNTRACKED (the owner's file): fix the seven items in
place and say so in the block, but never stage it — the commit carries README.md and
archive/PAPER_IMPLEMENTATION_GAPS.md only. On landing, lift README from STALE to CANON in
docs/MAP.md §3 (same commit, changelog row).
*Verify:* `grep -n "Lu et al\|3704\|research_loop" README.md` → nothing; `grep -c "Mitchener" README.md`
→ ≥ 1; `kosmos run --help` lists --seed and --data-path (plan §8 step 22); ladder green.
*DoD:* as M1, for `VIAB#P3-5`. Then check MAP §2 S2's exit gate (steps 18, 21, 19, 20, 22, 24,
baseline retired, ladder green) ⇒ the stage-gate commit (S2 done, S3 active).

### M8 — C-1 Findings JSON and `kosmos report` — BUILD·high — M — closes `VIAB#C-1`
Design: plan §5 C-1 "Required" verbatim. Pre-decided: the Finding is built in
`_handle_analyze_result_action` right after `_validate_result` returns validated or rejected
(P2-2), with code_provenance = {notebook_path: the P2-3 saved .py path from provenance
code_path, cell_index: 0}, null_model_result and scholar_eval from validation_detail (P2-2 keys:
recomputed, recomputed_match, null_model, scholar_eval); written by
ArtifactStateManager.save_finding_artifact to `<artifacts_dir>/<run_id>/findings/<result_id>.json`
(D-08); `kosmos report --run-id <id> [--output <path>]` added to kosmos/cli/main.py after the
`version` command pattern, rendering validated and rejected findings with provenance, the
plan §7 metrics and the failed experiments with error messages, structured like
generate_report (read from archive/code after M4). The artifacts.py VALID_TYPES warning
("DERIVES_FROM", DEEP_ONBOARD gotcha) is NOT fixed here (register row if it bites).
*Verify:* `python -m pytest tests/unit/cli/test_report.py --no-cov -p no:cacheprovider -q` → exit 0;
the file contains both result ids, "validated", the git_sha and the error message; the findings
JSON parses with scholar_eval and null_model keys (plan §8 step 23, C-1 half); ladder green.
*DoD:* as M1, for `VIAB#C-1`.

### M9 — M-1 `kosmos validate-null` and `kosmos rerun` — BUILD·high — S — closes `VIAB#M-1`
Design: plan §5 M-1 "Required" verbatim (the exact algorithm is in the item text). Pre-decided:
`CodeExecutor(use_sandbox=provenance['sandbox_used']).execute_with_data(code, data_path,
seed=provenance['seed'])` (P2-3 Notes); the sha256 check uses provenance['data_sha256']; the
test fixture's code_generated is the P2-1 generated code (ExperimentCodeGenerator(use_llm=False)
on the bound protocol, as in tests/unit/execution/test_column_binding.py). The unit test runs
the host executor path (use_sandbox False in the seeded provenance) so it needs no Docker.
*Verify:* `python -m pytest tests/unit/cli/test_report.py tests/unit/cli/test_metric_commands.py --no-cov -p no:cacheprovider -q`
→ exit 0; stored shuffled_pass_rate ≤ 0.15; rerun exact_match True, then exit 1 after tampering
(plan §8 step 23); ladder green.
*DoD:* as M1, for `VIAB#M-1`. Then check MAP §2 S3's exit gate ⇒ the stage-gate commit (S3 done,
S4 active) — and surface `OWNER#live-spend-authorization` in the END-OF-SESSION REPORT.

### M10 — LIVE-25 Live acceptance run on the climate CSV — BUILD·high, OWNER-gated — S — closes `VIAB#LIVE-25`
Design: Lane 0 L0-1 first: no launch without the ruling. Then plan §8 step 25 verbatim:
`kosmos run "Does atmospheric CO2 concentration predict global temperature anomaly?" --domain climate_science --data-path evaluation/data/climate_co2_temperature_test.csv --seed 42 --max-iterations 3 --budget 1`
(DeepSeek via .env, D-02/D-18; report the cost shown in the metrics summary), then
`kosmos report --run-id <run_id>` (save the output under artifacts/runs/<run_id>/report.md and
quote its path in the block), `kosmos rerun --result-id <id> --seeds 1,2,3`,
`kosmos validate-null --run-id <run_id> --k 20`. This run WRITES the owner's kosmos.db (it is
the product's database; D-02): say so in the block with the row counts before and after. Step
25b (the subscription variant) is optional and the owner's call: not part of this milestone.
*Verify:* the results table shows ≥ 1 row with Exec OK, Data file, Test pearson_correlation,
Stat ≈ 0.93, p ≈ 6e-29, Validation validated; total_cost_usd non-zero and < 1.00; rerun exit 0
with exact_match True; validate-null mean shuffled pass rate ≤ 0.05 (plan §8 step 25); ladder
green (the ladder itself is unchanged by a live run).
*DoD:* the four command outputs quoted in the session block · `VIAB#LIVE-25` → done ·
`OWNER#live-report-signature` surfaced as the first "Needs owner" line · S4 flips ONLY when the
owner's signature is recorded (a RULED line in SESSION_STATE §Waiting-on-owner and a
docs/MAP_CHANGELOG.md row); until then S4 stays active with the lag recorded as a blocker.

## Lane 2 — the factory (at most one lane in flight beside lane 1)
### F1 — FACTORY#runner Build the unattended queue — JUDGMENT·high (Session 4, owner present) — M — closes `FACTORY#runner`
Precondition: three attended /factory-continue passes recorded green in SESSION_STATE
§Session log (rows s1, s2, s3 with `VERIFY PASS (4 gates)`). Design: process §7 and Appendix L
verbatim with the Kosmos values: ~/kosmos-queue/ as its own git repo; driver.sh, run_watched.sh
(build = claude-opus-5-5, judgment = claude-fable-5-1, `--dangerously-skip-permissions` per
D-24, unset ANTHROPIC_API_KEY and every CLAUDE* variable; the secrets file holds nothing the
repo's .env does not: Kosmos needs no runner secret, so the NOSECRETS halt is replaced by a
`.env` presence check); template/CONTINUE_PROMPT.md with the RUNNER ADDENDUM and halt classes;
WORK_QUEUE.md with two continue rows at pos 10 and 20 (build/high, wrap commit); the signals
gate (`scripts/signals_check.sh` rc 0 before every launch) and the no-progress pause (a
continue row that leaves HEAD unchanged pauses the queue with a needs-owner reason) from the
start; refill ≤ 2 rows per drain, 8 drains per day; ORCHESTRATOR_RESULTS.md at the repo root
as the runner's one tracked file; a push on HALT (ntfy or the owner's chosen channel; recorded
in docs/STANDING_SIGNALS.md). One watched row with the owner present; then hand over the night
launch command, the attach hint and the pre-flight rules (/factory-queue).
*Verify:* `bash ~/kosmos-queue/driver.sh --dry-run` renders both continue rows with LIVE FACTS;
one watched row ends WRAP_OK with a results-ledger row and a commit; `/factory-queue status`
parses the queue; ladder green on the repo (ORCHESTRATOR_RESULTS.md added by explicit path).
*DoD:* explicit-path commit of ORCHESTRATOR_RESULTS.md and the STANDING_SIGNALS.md push entry ·
`FACTORY#runner` → done · SESSION_STATE §STANDING "How the work reaches the queue" updated with
the launch command · ADR-0003 (the runner's permission posture and halt classes) in the same commit.

### F2 — FACTORY#ladder-gates Review the ladder after three milestones — BUILD·high — S — closes `FACTORY#ladder-gates`
Design: read the three sessions' Verify cells and /tmp/kosmos-verify/run_*/verify.log
durations; decide (and record in SESSION_STATE §STANDING) whether a fast ladder per milestone
plus the full ladder nightly is warranted (process §12 knob; default: no, the ladder is under
10 minutes); whether gate 4 should also run a Welch and an ANOVA template (default: yes if any
milestone touched the templates); whether any false red occurred (record TEST# rows).
*Verify:* the decision paragraph in §STANDING; ladder green if scripts/verify.sh changed.
*DoD:* explicit-path commit · `FACTORY#ladder-gates` → done.

## Risk playbook (decided mitigations — no open questions live here)
- A milestone's acceptance test cannot pass as written (a fixture is missing, a quoted line is
  gone): make the smallest correct change, record the deviation in the session block in the
  tracker's Notes style, and fix the plan text per CLAUDE.md §AMENDMENT in the same commit.
- Gate 2 shows "retired" ids after M3/M4/M6: expected; they are not a red. A NEW RED outside
  the baseline is fixed forward or the milestone wraps; the baseline is never widened to pass.
- Gate 4 or gate 3 red by the environment (daemon down, image missing, kosmos.db locked):
  Lane 0 L0-3. Rebuilding the sandbox image is allowed (D-09); nothing else is touched.
- A session on the build tier meets a decision the design did not pre-decide (a schema shape,
  a changed verdict rule, anything in plan §9): PARK (SESSION_STATE §Waiting-on-judgment), wrap
  or take the next row. Nothing in this plan should need it; if it does, that is a freeze defect
  to record.
- The ladder red at session start: the first work item is that red (continue the dirty tree
  as milestone-in-progress; never restart).
- Context runs low: the mid-milestone wrap (commit nothing; NEXT ACTION = "resume M<N> at
  <step>"; list the touched files).

## Out of scope (for every session under this spec)
Plan §9 non-goals in full (paper reproduction at scale; the HTTP server, SSE, WebSocket;
multi-tenancy, auth, Kubernetes beyond moving k8s/ to archive; R execution; Neo4j features; the
parallel execution path; the host-path retry rewrites; sandbox timeout handling; DataProvider on
the director path; the message bus; Phase 4 persistence; Tier C; ScholarEval as a gate;
ANTHROPIC_API_KEY). Also: the 4,241-finding ruff sweep (HYG#ruff-lint-debt), untracking
.literature_cache (OWNER#literature-cache-untrack), step 25b, anything in the untracked
evaluation reports.

## Definition of done (the stage exit gates, restated concretely)
- S1: plan §8 steps 14 and 17 green at M2's landing; ladder green; the stage-gate commit flips
  S1 done / S2 active with a docs/MAP_CHANGELOG.md row.
- S2: steps 18, 21, 19, 20, 22, 24 green; scripts/test_baseline.txt holds only its stamp line;
  ladder green; M5b landed (or S2 stays active with the lag recorded as blocked(OWNER#compose-edit-approval)
  while S3's rows proceed in printed order — the owner-gated item blocks the flip, never the
  sibling lane).
- S3: step 23 green; ladder green; S3 done / S4 active.
- S4: step 25's outcomes quoted; the report saved under artifacts/runs/<run_id>/; the owner's
  signature recorded; ladder green; S4 done ⇒ every map row done ⇒ archive this plan to
  docs/execution/archive/PLAN_S1-S4.md and write HALT_PLAN_COMPLETE (runner) or report (attended).

## Open items register (honesty section: item · owner · what it blocks)
- `DEC#tier-c-archive` · owner · decide-by after S4 · blocks TEST#tier-c-shap-undefined-name only.
- `DEC#scholar-eval-gate` · owner · decide-by after S4 · blocks nothing.
- `OWNER#literature-cache-untrack` · owner · any pack sitting · blocks nothing (the four dirty
  .pkl files are noise in every `git status`).
- M6 size · the M6 session · decide-by M6 start · the split rule (Locked decisions 5b) makes it
  safe; the honest estimate is two to three sessions, not the plan's 4 h.
- M5 docker build on Docker Desktop from WSL: the image build was never run on this machine
  (`docker build -t kosmos:test .` failed at the data-files step per the plan); M5a extends
  .dockerignore first and records the build time · the M5 session · blocks nothing beyond M5.
- The runner's push channel (ntfy, email, or none) · owner, asked at Session 4 · blocks nothing
  before F1.

## Commit trail (append during execution)
| Commit | Milestone | Delivered |
|---|---|---|
| 2600661 | freeze | docs/PLAN.md frozen s0 2026-10-04 · DOC-CHECK run: ADR-0001, ADR-0002 lint A1–A7 pass · MAP_CHANGELOG references resolve · every anchor above dated 2026-10-04 · findings: none |
| `s1 — P2-4:` (2026-10-08) | M1 | VIAB#P2-4: kosmos/cli/commands/run_results.py build_run_results (columns, not repr strings; usage from get_usage_stats); results table and metrics in the viewer and both exports; per-result cost through create_result and update_result_validation; record_api_call cost_usd with a running total. Plan §8 step 14 green (13 passed) |
| `s2 — P2-7:` (2026-10-08) | M2 | VIAB#P2-7: pool caps (max_hypothesis_pool 12, max_untested_backlog 4, num_variants 1, max_refinements_per_hypothesis 2); ResearchPlan.hypothesis_scores/hypothesis_generation with score-ordered untested ids; refinement skips failed/unvalidated/rejected_unsafe results without an LLM call, spawns variants only from validated results, counts variants_dropped_duplicate; REFINING leaves after one pass per analysis; the empty-backlog convergence waits while the pool can grow. Plan §8 step 17 green (2 passed; pool peak 12 over 52 actions to convergence) |
| `s3 — P3-1:` (2026-10-08) | M3 | VIAB#P3-1: SafetyIncident.violation optional; CodeValidator path guard; guardrails signal handlers default off, main thread only, chained to the previous handler (SIG_DFL re-delivered); numeric config reads tolerate a Mock config; director validates generated code after an emergency-stop check, stores unsafe code as a rejected_unsafe result (experiment FAILED, off the queue, ANALYZING) and never executes it; analysis keeps rejected_unsafe without an LLM call; guardrails limits into the sandbox config; execute_protocol_code builds guardrails without signal handlers. 29 guardrails tests + 4 new pass (step 18 exit 0 moved to M6, see the M3 amendment) |
| `s4 — P3-4:` ×2 (2026-10-08) | M4 | VIAB#P3-4: Tier A (19 modules) in 51ea888, Tier B (kosmos/workflow, kosmos/orchestration, kosmos/compression, scripts/smoke_test.py, verify_e2e.py, verify_production.sh) in the second commit, all under archive/code/<same path> with archive/code/README.md; tests moved whole or split (archived-dependent tests to archive/code/tests/<same path>); exports pruned; norecursedirs and the coverage omit exclude archive/. Step 21 green after each commit (imports ok, rg no files, no collection errors) |

## Operational learnings (append during execution; graduate keepers)
- (s0) The judge prints "retired" ids whenever a baseline test starts passing; after M3, M4 and
  each M6 part this list is long and is the expected evidence, not noise.
- (s1, M1) A plan §8 verify command is a bare pytest run: it loads .env and reaches the configured DB. Any test in it that constructs a ResearchDirectorAgent must use an in-memory DB fixture, or it writes a ResearchSession row into kosmos.db. Checked by md5 of kosmos.db around the step-14 run.
- (s2, M2) Plan §5 P2-7 (3)'s REFINING clause read literally ("return DESIGN_EXPERIMENT when untested exist, else GENERATE_HYPOTHESIS") would never refine, and its acceptance test needs refinement; it applies once the pass has run (`_refined_since_analysis`). Plan §5 P1-2's risk note is the other half: with no untested work the run converged before a second generation round, so the empty-backlog convergence now waits while the pool is under its cap and the last generation was not empty (`_can_grow_pool`).
- (s3, M3) Tests that exercise SafetyGuardrails write `.kosmos_emergency_stop` and safety_incidents.jsonl into the cwd. A flag file left in the repo root makes every later run (and the director) refuse to execute. tests/unit/safety/conftest.py runs each test in tmp_path; any new test that triggers an emergency stop must chdir the same way. Check `ls .kosmos_emergency_stop` in the repo root after a ladder run.
- (s4, M4) A test file can import an archived module lazily, inside a test, or through a package re-export (`from kosmos.validation import FailureDetector`), or patch one by string. A grep for `kosmos.<pkg>.<module>` alone misses the last two. The s4 census used an AST pass that flags a test when its body, its patch strings or a fixture it requests touches an archived name. Fixture names repeat across classes, so key fixtures by class.
