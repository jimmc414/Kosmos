# Kosmos Viability Build — MAP

§0 How to read · §1 Mission and the gate ladder · §2 Stage table · §3 Document router and trust
table · §4 Locked decisions · §5 Standing traps · §6 Guardrails · §7 Verification ladder (gates
per stage) · §8 Open inputs · §9 Session protocol · §10 Changelog (docs/MAP_CHANGELOG.md)

## §0 How to read
This is the map: what is being built, in what order, and what is settled. It changes only in a
stage-gate commit or a freeze (§2 rules). The plan for the active stages is docs/PLAN.md; the
session ledger is SESSION_STATE.md; the long tail is docs/execution/BACKLOG.md. A session reads
§2 at start (CLAUDE.md step 1) and §4 before touching anything a decision governs. Drafted s0,
2026-10-04; the owner signed §2 and §4 on 2026-10-04 (docs/process/INVENTORY_S0.md §10 (b)).

## §1 Mission and the gate ladder
**Mission.** Complete the viability plan (evaluation/VIABILITY_ASSESSMENT_AND_CHANGE_PLAN.md §5)
on branch `viability-fixes`: the ten remaining items P2-4, P2-7, P3-1, P3-4, P3-2, P3-3, P3-5,
C-1, M-1 and LIVE-25, in that order, each landed as one green commit behind the ladder, until the
live acceptance run (plan §8 step 25) passes and the owner signs its report.

**Truth.** The plan's acceptance tests (mocked LLM) are the oracle for every code item; the
climate CSV's known statistics (co2_ppm vs temp_anomaly_c: r 0.9317, p 5.8e-29, n 64) are the
oracle for the sandbox gate and the live run; the package's own import graph is the oracle for
the archive items.

**The ladder** is `bash scripts/verify.sh` (§7): 4 gates, one pass string `VERIFY PASS (4 gates)`.

## §2 Stage table
Status vocabulary: done · active · next · blocked(<on>) · later. Exactly one row is `active`.
Status changes only in a stage-gate commit (exceptions: a freeze may move later→next; a fired
owner trigger may set blocked(<item>)). Plan §8 step numbers below are the evaluation plan's
verification table; every conjunct is a non-live command unless marked LIVE.

| Stage | Name | Entry criteria | Exit gate (evidence conjunction) | Oracle | Execution spec | Owner-gated | Size (sessions) | Status |
|---|---|---|---|---|---|---|---|---|
| S0 | Factory bootstrap | viability-fixes at 1247379 (P2-3 landed); owner present for Phase A | `VERIFY PASS (4 gates)` on the finished tree, with each gate's known-RED and known-GREEN control recorded in docs/process/INVENTORY_S0.md · CLAUDE.md, docs/MAP.md, docs/PLAN.md, SESSION_STATE.md, docs/execution/BACKLOG.md, docs/adr/0001 and 0002, scripts/signals_check.sh (rc 0), scripts/hot_files_check.sh (rc 0) and the four /factory-* commands present · owner rulings recorded verbatim in INVENTORY_S0.md §10 · branch pushed | none (process installation) | the Session 0 prompt (owner, 2026-10-04) recorded in INVENTORY_S0.md §0 | Phase A rulings (recorded) | S (1) | done |
| S1 | P2 remainder (P2-4, P2-7) | S0 done; docs/PLAN.md frozen | plan §8 step 14: `python -m pytest tests/unit/cli/test_run_results.py tests/unit/core/test_metrics_bridge.py --no-cov -p no:cacheprovider -q` exit 0, cost_per_validated_finding == 0.0123, estimated_cost_usd == get_model_cost('deepseek/deepseek-chat', 1000, 500) · step 17: `python -m pytest tests/unit/agents/test_pool_control.py --no-cov -p no:cacheprovider -q` 2 passed, pool never exceeds 12 over 60 actions · ladder green at the landing commit of the last item | plan §5 acceptance tests (mocked LLM) | docs/PLAN.md | none | M (2) | active |
| S2 | P3 hardening (P3-1, P3-4, P3-2, P3-3, P3-5) | S1 done | step 18: `python -m pytest tests/unit/safety --no-cov -p no:cacheprovider -q` exit 0 (29 guardrails tests unmodified + 4 new) · step 21: `python -c "import kosmos, kosmos.execution, kosmos.validation, kosmos.safety, kosmos.agents.research_director; print('ok')"` prints ok, `rg -l "domain_router\|failure_detector\|accuracy_tracker\|notebook_generator\|production_executor\|graph_visualizer\|plotly_viz" kosmos/` lists no files, `python -m pytest tests/unit --co -q --no-cov -p no:cacheprovider \| tail -3` reports no errors · step 19: `docker build -t kosmos:test .` then `docker run --rm kosmos:test python -m kosmos.cli.main version` prints a version, `pip install -e . --dry-run` resolves litellm, `python -c "import kosmos.cli.main"` succeeds without fastapi · step 20: `python -m pytest tests/requirements/core/test_req_configuration.py tests/unit/safety/test_guardrails.py tests/unit/literature --no-cov -p no:cacheprovider -q` exit 0 · step 22: `grep -n "Lu et al\|3704\|research_loop" README.md` prints nothing, `grep -c "Mitchener" README.md` ≥ 1, `kosmos run --help` lists --seed and --data-path · step 24: `python -m pytest tests/unit/execution tests/unit/agents tests/unit/core tests/unit/db tests/unit/validation tests/unit/hypothesis tests/unit/safety tests/unit/cli --no-cov -p no:cacheprovider -q` exit 0 · scripts/test_baseline.txt retired (FACTORY#retire-test-baseline done; gate 2 judges a bare exit 0) · ladder green | plan §5 acceptance tests; the package import graph; the Dockerfile build | docs/PLAN.md | P3-2: the owner approves the docker-compose.yml edit and its staging over the uncommitted hardening (or commits the hardening first) | L (5–6) | later |
| S3 | Reporting and reruns (C-1, M-1) | S2 done | step 23: `python -m pytest tests/unit/cli/test_report.py tests/unit/cli/test_metric_commands.py --no-cov -p no:cacheprovider -q` exit 0, stored shuffled_pass_rate ≤ 0.15, rerun exact_match True then exit 1 after tampering · ladder green | plan §5 acceptance tests on the climate CSV | docs/PLAN.md | none | M (2) | later |
| S4 | Live acceptance (LIVE-25) | S3 done; the owner's go for the live spend recorded in docs/PLAN.md Lane 0 | LIVE step 25: `kosmos run "Does atmospheric CO2 concentration predict global temperature anomaly?" --domain climate_science --data-path evaluation/data/climate_co2_temperature_test.csv --seed 42 --max-iterations 3 --budget 1` shows ≥ 1 results row with Exec OK, Data file, Test pearson_correlation, Stat ≈ 0.93, p ≈ 6e-29, Validation validated, and a non-zero total_cost_usd below 1.00 · `kosmos report --run-id <run_id>` written under artifacts/runs/<run_id>/ · `kosmos rerun --result-id <id> --seeds 1,2,3` exit 0 with exact_match True · `kosmos validate-null --run-id <run_id> --k 20` mean shuffled pass rate ≤ 0.05 · the owner's signature on the run's report (a RULED line in SESSION_STATE §Waiting-on-owner and a docs/MAP_CHANGELOG.md row) · ladder green | the climate CSV's known truth (r 0.9317, p 5.8e-29) | docs/PLAN.md | LIVE-25: the owner authorizes the spend at launch time and signs the report | S (1) | later |

Critical path: S0 → S1 → S2 → S3 → S4. Parallel: none (one code lane; the register's EXPLORE
rows may run beside it as read-only sessions).

## §3 Document router and trust table
Trust: CANON (build from it) · MIXED (cite only the parts the table names) · STALE (never a build
input) · SNAPSHOT (true at a date; re-verify).

| Asset | Trust | Use |
|---|---|---|
| evaluation/VIABILITY_ASSESSMENT_AND_CHANGE_PLAN.md | CANON | the spec: §5 item text, §8 verification table, §9 non-goals, §10 owner decisions. Line numbers exact at 6cfe7f6 only; locate by the quoted code |
| evaluation/VIABILITY_PROGRESS.md | CANON, FROZEN at s0 | the record of P0 through P2-3 and the A-2 spec; its Notes are facts later items rely on. Not edited after s0 |
| docs/process/SOFTWARE_FACTORY_PROCESS.md | CANON | the process; its laws bind every session |
| docs/MAP.md (this file) · docs/PLAN.md · SESSION_STATE.md · docs/execution/BACKLOG.md · docs/adr/ | CANON | the factory's four files plus the decision records |
| CLAUDE.md | CANON | the contract |
| docs/process/INVENTORY_S0.md | SNAPSHOT 2026-10-04 | environment readings and the owner's Phase A rulings |
| docs/DEEP_ONBOARD.md (untracked) | MIXED | grep Gotchas and the Change Impact Index for touched files; trust its evidence over its claims; the known-wrong items are listed in plan P3-5 |
| README.md | STALE until P3-5 lands | never a build input; P3-5 rewrites it |
| docs/xray.md · docs/DEEP_ONBOARD_VALIDATION.md · evaluation/*_findings.md · evaluation/CRITICAL_EVALUATION_REPORT.md (untracked) | SNAPSHOT 2026-02 to 2026-10 | owner's reports; evidence only, never a build input, never staged |
| archive/ · .claude/export/WARM_START.md · docs/planning, docs/phase-reports | STALE | history |
| .claude/skills/next-plan-item/SKILL.md | STALE (superseded s0) | pointer only |

## §4 Locked decisions (re-arguing one in a build session is drift)
| Id | Decision | Source | Revisit point | ADR |
|---|---|---|---|---|
| D-01 | Unbound hypotheses are excluded only: DB status stays GENERATED, the id goes to untestable_hypotheses | plan §10 #1, owner 2026-10-02 | none | — |
| D-02 | Live DeepSeek steps (plan §8 steps 5 and 25) are run by the coding session with `--budget 1`, cost reported; the climate CSV is the acceptance dataset | plan §10 #2 | none | — |
| D-03 | sentence-transformers is the optional `embeddings` extra with the TF-IDF fallback | plan §10 #3 | none | — |
| D-04 | kosmos/domains and the domain protocol templates (Tier C) are kept | plan §10 #4 | after S4 (the live acceptance run) | — |
| D-05 | The library loop (Tier B) is archived in-tree under archive/code with its tests, excluded from collection and coverage | plan §10 #5 | none | — |
| D-06 | Runs without --data-path are allowed; synthetic rows carry data_source 'synthetic' and validation_status 'unvalidated' and never mark a hypothesis supported | plan §10 #6 | none | — |
| D-07 | ScholarEval is advisory only; the validated rule ignores it unless `scholar_eval_gate` is set | plan §10 #7; plan §9 | after S4 (calibration on live runs) | — |
| D-08 | Artifacts live under ./artifacts/runs/<run_id>/ (config key artifacts_dir), gitignored | plan §10 #8 | none | — |
| D-09 | The sandbox image is built by the coding session (`docker build -t kosmos-sandbox:latest docker/sandbox`); the tag is pinned through sandbox_image | plan §10 #9 | none | — |
| D-10 | One generation round per run is acceptable until P2-7 lands | plan §10 #10 | P2-7 landing (S1) supersedes it | — |
| D-11 | SandboxUnavailable fails fast: one failed experiment, has_converged with a "halted:" reason, ERROR, the CLI stops | plan §10 #11 | none | — |
| D-12 | Anthropic models through KOSMOS_ANTHROPIC_API_KEY or the ClaudeCodeProvider; LiteLLM stays the default | plan §10 #12 | none | — |
| D-13 | Order: P0 → P1 → providers → the rest; the remaining order is the tracker's printed order (P3-4 ahead of P3-2 because P3-2 step 3 archives api modules) | tracker ground rules 2026-10-02; tracker row 24 note | none | — |
| D-14 | Provider and model switching through `kosmos run --provider/--model`, with .env as the default | tracker A-2 spec, owner 2026-10-02 | none | — |
| D-15 | One commit per item, subject `s<N> — <ID>: <summary>`, no attribution lines, never force-push, push the branch at the end of every session | tracker ground rules; owner 2026-10-03; CLAUDE.md | none | — |
| D-16 | Never stage the owner's paths (CLAUDE.md §STANDING TRAPS list); docker-compose.yml is edited only with the owner's approval | tracker ground rules; s0 prompt | none | — |
| D-17 | Tests run as `python -m pytest <paths> --no-cov -p no:cacheprovider -q`; never bare `pytest`; tests/e2e is never in the ladder | tracker ground rules; s0 prompt | none | — |
| D-18 | Live calls: DeepSeek for plan §8 steps 5 and 25 with `--budget 1` only; one Claude smoke test, spent in A-1; nothing else live, nothing live in the ladder | tracker ground rules | none | — |
| D-19 | ANTHROPIC_API_KEY is never set or written by Kosmos, its tests, its .env files or the runner | plan §9; tracker; s0 prompt | none | — |
| D-20 | The Session Factory process governs the remaining work; the tracker is frozen | s0, owner 2026-10-04 | none | ADR-0001 |
| D-21 | Ladder gate 2 judges against a stamped baseline of pre-existing failures until P3-3 retires it | s0, owner 2026-10-04 | P3-3 landing (S2) | ADR-0002 |
| D-22 | Model tiers: judgment = Fable (effort high), build = Opus (effort high); every remaining plan item defaults to BUILD·high; P3-2 and LIVE-25 are OWNER-gated | s0 prompt, owner 2026-10-04 | none | — |
| D-23 | Session 0's files land on viability-fixes; every factory commit stays on that branch | owner ruling 2026-10-04 (INVENTORY_S0 §10 (d)) | none | — |
| D-24 | The runner launches each row with `--dangerously-skip-permissions`; the safety is the .claude/settings.json deny-list, the contract and the pushed branch (no per-project allowlist) | owner ruling 2026-10-04 (INVENTORY_S0 §10 (c)) | FACTORY#runner (Session 4) | — |

## §5 Standing traps
The list lives in CLAUDE.md §STANDING TRAPS (the contract is the operative text). Of note for
every milestone: tests/conftest.py loads .env with override=True, so DATABASE_URL reaches every
test; only the ladder's scripts/verify_isolate.py plugin keeps the suite off kosmos.db.

## §6 Guardrails
- STOP-AND-ASK: CLAUDE.md §STOP AND ASK THE OWNER (seven items, owner-ruled 2026-10-04).
- The harness deny-list: .claude/settings.json `permissions.deny` (47 rules: bulk staging, `commit -a`,
  `reset --hard`, `clean -f`, `stash -u`, force-push, push to master, staging the owner's paths,
  `compose down -v`, deleting kosmos.db or a data directory).
- Owner-gated rows: P3-2 (docker-compose.yml) and LIVE-25 (live spend and signature).

## §7 Verification ladder (gates per stage)
Gates now (s0): 1 compile-lint · 2 tests (baseline-judged) · 3 alembic · 4 template-run.
Per-stage changes: S2 retires the gate-2 baseline when P3-3 lands (gate 2 becomes a bare exit 0)
and P3-3 adds .github/workflows/unit.yml (CI, no secrets). S3 and S4 add no gate. Live steps are
never gates.

## §8 Open inputs (pointers, blocking only)
None blocking at s0. All twelve plan §10 questions are answered. Factory items are register rows:
FACTORY#runner (Session 4, after three attended passes), FACTORY#ladder-gates,
FACTORY#retire-test-baseline (closes with P3-3). See docs/execution/BACKLOG.md.

## §9 Session protocol
CLAUDE.md is the operative text; docs/process/SOFTWARE_FACTORY_PROCESS.md §5 is its rationale.

## §10 Changelog
docs/MAP_CHANGELOG.md (append-only; columns Date · Section · Change · Why · Commit).
