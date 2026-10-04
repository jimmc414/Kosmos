# SESSION STATE — Kosmos viability build
<!-- The ONLY mutable session ledger. Updated at the end of EVERY session.
     Stage status lives in docs/MAP.md §2 — not here. -->
<!-- RETENTION RULE. HOT = this file: ONE "## NEXT ACTION" · "## Waiting-on-owner"
     · "## Waiting-on-judgment" · "## HOLD" · "## STANDING" (edited in place, never
     appended) · "## Recent sessions" (newest ~10 blocks, newest first, each opened
     by <!-- session s<N> -->) · "## Session log" (rows of the sessions whose blocks
     are here; NEWEST FIRST; a new row goes DIRECTLY BELOW the separator) · "## Fresh
     traps". ROLL when the file passes 200,000 B or 12 blocks:
     python3 scripts/session_state_roll.py --write moves the oldest blocks AND
     their log rows to docs/execution/archive/SESSION_STATE_s<a>-s<b>.md (frozen;
     their log rows at the top are the index). Anchor edits on the LINE-START
     header; this comment quotes the headers and a substring search lands here. -->

## NEXT ACTION (rewritten s0, 2026-10-04)
Owner: skim docs/PLAN.md §Locked decisions and §DoD; then the first /factory-continue pass takes
VIAB#P2-4 (Opus, high). VIAB#P2-4 is milestone M1 of docs/PLAN.md (P2-4 Honest run report and
real cost: a new kosmos/cli/commands/run_results.py, the results table in the CLI, per-result
cost attribution, the cost field in record_api_call), verified by plan §8 step 14 and the
ladder. Sessions s1–s3 are attended passes on Opus high (SESSION_STATE §STANDING "How the work
reaches the queue"); FACTORY#runner waits for three green rows in the Session log.

## Waiting-on-owner (same ids as BACKLOG §OWNER; the sweep keys on this section)
- `OWNER#compose-edit-approval` — P3-2 edits docker-compose.yml (healthcheck → `python -m kosmos.cli.main version`, drop the 8000:8000 mapping) on top of the owner's uncommitted hardening: approve the edit and its staging, or commit the hardening first — escalate-by S2 start (the session that reaches VIAB#P3-2) — NEW s0
- `OWNER#live-spend-authorization` — LIVE-25 spends DeepSeek money (`--budget 1`, under $1): confirm at launch time that the run may be launched by the session, attended or through a PRE-ANSWERED line — escalate-by S4 start — NEW s0
- `OWNER#live-report-signature` — after LIVE-25, read `kosmos report --run-id <run_id>` and sign it (a RULED line here and a docs/MAP_CHANGELOG.md row); S4 cannot flip without it — escalate-by S4 — NEW s0
- `DEC#tier-c-archive` — MAP D-04 revisit: archive kosmos/domains and the domain protocol templates (Tier C) after the live acceptance run, or keep them — escalate-by after S4 — NEW s0
- `DEC#scholar-eval-gate` — MAP D-07 revisit: calibrate ScholarEval thresholds on live DeepSeek runs and decide whether `scholar_eval_gate` joins the validated rule — escalate-by after S4 — NEW s0

## Waiting-on-judgment (parked by a build session: item · why judgment-grade · what it blocks)
(none)

## HOLD
Status: LIFTED
Queue (in order): (none)
PARKED (never started under the hold): (none)

## STANDING (facts and rules that outlive a session — edit IN PLACE; name the source session)
- **Ground rules, carried from the frozen tracker (evaluation/VIABILITY_PROGRESS.md, 2026-10-02/03) into the contract (s0):** branch `viability-fixes` cut from master at 73f4d2a (code identical to 6cfe7f6, the commit the plan's line numbers refer to) · one commit per item, subject `s<N> — <ID>: <imperative summary>`, no attribution lines, push the branch at the end of every session (`git push -u origin viability-fixes`, never force), find an item's commit with `git log --oneline --grep "<ID>:"` · never stage the owner's paths (CLAUDE.md §STANDING TRAPS) · tests only as `python -m pytest <paths> --no-cov -p no:cacheprovider -q` · plan line numbers exact at 6cfe7f6 only · live calls only per MAP D-18 · plan §10 decisions binding (MAP D-01 to D-12).
- **Item-level facts live in the frozen tracker's Notes column** (helper names, deviations, changed line positions, defects found for later items): read the tracker row of every item your milestone depends on before editing. The tracker is CANON and frozen; it is not edited again.
- **The ladder (s0):** `bash scripts/verify.sh` (4 gates, about 8 minutes; gate 2 alone about 7 minutes). Gate 2 judges against scripts/test_baseline.txt (ADR-0002) until VIAB#P3-3 retires it. The baseline moves only through `--accept-baseline "<cause>"`. Gate 2 runs under scripts/verify_isolate.py and never opens kosmos.db (proven s0).
- **How the work reaches the queue (owner, 2026-10-04):** Sessions 1 to 3: the owner runs `/factory-continue` attended on Opus, high. Each takes one milestone and writes anything that stumbled into this §STANDING; nothing graduates to CLAUDE.md before it bites twice. Session 4 (Fable, high, owner present): FACTORY#runner per process Appendix L: ~/kosmos-queue/ as its own git repo, driver.sh, run_watched.sh, template/CONTINUE_PROMPT.md with the RUNNER ADDENDUM and halt classes, two continue rows at pos 10 and 20, the signals gate and the no-progress pause from process §7.8 on from the start, refill limited to two rows per drain, `--dangerously-skip-permissions` with the deny-list carrying the safety (MAP D-24). One watched row with the owner present, then the night launch command, the attach hint and the pre-flight rules are handed over. Then the serial queue runs continue rows until a session writes HALT_PLAN_COMPLETE (every map row done and PLAN.md archived) or HALT_NEEDS_OWNER (P3-2, LIVE-25, or any STOP-AND-ASK item). The owner answers at a pack sitting; the answers go into PRE-ANSWERED blocks; the queue relaunches.
- **Signals (s0):** no scheduled job and no push channel exist for Kosmos yet; scripts/signals_check.sh checks branch, stray test processes, Docker, disk, kosmos.db and the sandbox image (docs/STANDING_SIGNALS.md is the definition). FACTORY#runner adds the push.
- **Process-rule incubator:** (none yet; rules 1–11 in CLAUDE.md are inherited from the process document at adoption)
- **Last documentation-checklist run:** s0 2026-10-04 (B7 freeze; token in docs/MAP_CHANGELOG.md and docs/PLAN.md §Commit trail)

## Recent sessions (newest first; each block opens with `<!-- session s<N> -->`)
<!-- session s0 -->
### s0 — 2026-10-04 01:50 → 03:19:02 CDT (JUDGMENT Fable high; attended, owner present for Phase A) — Factory bootstrap: contract, map, ladder, ledger, register, ADRs, signals, commands, first freeze
- signals: signals: 7 OK · tree 6M/18U
- **What landed:** Phase A dc62f8a (process doc, CLAUDE.md contract merged over the existing rules, .claude/settings.json deny-list of 47 rules, docs/MAP.md S0–S4 + D-01..D-24 signed by the owner, docs/process/INVENTORY_S0.md with the four rulings verbatim) · B1 fee5198 (scripts/verify.sh 4 gates with RED/GREEN controls recorded in INVENTORY §B1; verify_isolate.py keeps the suite off kosmos.db, proven by md5 over three full runs; test baseline 394 ids and lint baseline 1 finding stamped with causes) · B2 df24d97 (this ledger; tracker frozen with a pointer header) · B3 123d17c (register: 10 VIAB in-plan rows, 3 FACTORY rows, 6 owner/decision asks, 9 TEST/HYG rows, 22 closed rows with hashes) · B4 51b48f7 (ADR-0001, ADR-0002) · B5 94f18b8 (signals: 7 checks + tree reading) · B6 a095809 (/factory-continue, /factory-spec, /factory-verify, /factory-queue; hot_files_check.sh; session_state_roll.py proven on a synthetic 14-block ledger; /next-plan-item → pointer) · B7 2600661 (docs/PLAN.md frozen: Lane 0 two owner triggers, M1–M10, F1–F2; DOC-CHECK token) · B8 this commit (S0 done, S1 active).
- **Defaults taken:** un-ignored CLAUDE.md and .claude/settings.json in .gitignore (they were ignored; the contract must be tracked) · gate 1 lint limited to ruff E9/F63/F7/F82 against a baseline because the full rule set has 4,241 findings · gate 2's real-suite RED control deferred to FACTORY#ladder-gates (the judge's RED control proves the comparison) · M5 pre-split into M5a/M5b and M6 allowed to span sessions with shrinking baselines (PLAN §Locked decisions 5) · pre-factory tracker sessions logged as `pre-s0` rows · the process's rules 1–11 inherited into CLAUDE.md at adoption.
- **Recorded, not fixed:** TEST#order-dependent-caplog, TEST#director-tests-write-configured-db, TEST#alembic-env-ignores-runtime-url, TEST#prioritizer-fixture-rationales, TEST#validation-pipeline-parametric-null, TEST#tier-c-shap-undefined-name, HYG#ruff-lint-debt, HYG#pyproject-ruff-top-level-keys, HYG#makefile-bare-pytest, OWNER#literature-cache-untrack (backlog/BUILD-open.md, OWNER-open.md). Prompt-fact correction: the tracker's 98+62 figure covers four directories; the ladder's full set is 285+107 (INVENTORY §9).
- **Tests / verify:** `VERIFY PASS (4 gates): compile-lint tests alembic template-run  (486 s)` 03:10:56 → 03:19:02 (second full run, on the finished tree at 2600661; first run 471 s at 03:00:09 → 03:08:00 on the B1 tree). Gate 2 red sets identical across all three full runs.
- **Review:** SELF-REVIEW only (the one spawned agent measured the read ceiling). Findings fixed before commit: the plan's "no .dockerignore exists" claim was wrong (the file exists and its `.env.*` rule would drop .env.example; corrected in PLAN facts and M5a); the DOC-CHECK row's anchor count (26) is corrected to 24 in the changelog.
- **Wrap:** clean.

## Session log (append-only; NEWEST FIRST; insert directly below the separator)
| Date | Did | Commits | Verify |
|---|---|---|---|
| 2026-10-04 s0 | **Factory bootstrap complete (S0 → done, S1 → active; JUDGMENT Fable high, attended).** Contract, deny-list, map, inventory, 4-gate ladder with baselines, ledger, register, ADR-0001/0002, signals, commands, hygiene tools, docs/PLAN.md frozen for S1–S4; nine commits dc62f8a … (this one). Review: SELF-REVIEW only. | dc62f8a, fee5198, df24d97, 123d17c, 51b48f7, 94f18b8, a095809, 2600661, (hash: next session) | `VERIFY PASS (4 gates): compile-lint tests alembic template-run  (486 s)` 03:10:56 → 03:19:02 |
| 2026-10-04 pre-s0 | **P2-3 done (`VIAB#P2-3` → done; Opus, attended, /next-plan-item).** --seed through director, designer, templates and executor; build_run_provenance on every result row; code saved under artifacts/runs/<run_id>/code; ResearchSession row per run. Regression agents/execution/core/cli/db/safety plus integration execution pipeline: no new failures against HEAD (128 failed, 99 errors pre-existing; two order-dependent caplog tests passed this time). No live call. | 1247379 | none (pre-factory; the item's acceptance tests, 22 passed) |
| 2026-10-03 pre-s0 | **P2-2 done (`VIAB#P2-2` → done; Opus, attended).** analysis_fn.py recomputation, permutation null on the real data via shuffle_target, provider-agnostic fail-closed ScholarEval as advisory, verdict rule. Regression: no new failures against HEAD (100 failed, 74 errors pre-existing). Owner asked to commit and push at the end of every session; skill and ground rules updated; branch pushed. | 43dbf18, 1ad249e | none (pre-factory; 33 acceptance tests) |
| 2026-10-03 pre-s0 | **P2-1 done (`VIAB#P2-1` → done; Opus, attended).** data_schema.py, Variable.column, designer binding with UnboundVariableError, untestable_hypotheses, bound code templates; host executor fixes for dir() and split namespaces. Regression: no new failures against HEAD (105 failed, 63 errors pre-existing). | 8026985 | none (pre-factory; 23 acceptance tests) |
| 2026-10-03 pre-s0 | **P2-6 done (`VIAB#P2-6` → done; Opus, attended).** LiteLLM JSON mode with one-time fallback, tolerant parse plus one repair call, parse_json_array_response, three brace-slicing parsers switched. Regression core/agents/hypothesis/validation identical to HEAD (70 failed, 63 errors pre-existing). | 67e7c0a | none (pre-factory; 10 acceptance tests) |
| 2026-10-03 pre-s0 | **P2-0 done (`VIAB#P2-0` → done; Opus, attended).** Nine nullable result columns, alembic revision a0aa37ea19f2 with JSON backfill, get_results_for_run, update_result_validation. Regression identical to HEAD (99 failed, 73 errors). Owner kosmos.db migrated to a0aa37ea19f2. | 5fced2c | none (pre-factory) |
| 2026-10-03 pre-s0 | **P2-5 done (`VIAB#P2-5` → done; Opus, attended).** TF-IDF novelty fallback, guarded cosine, vector-search PaperMetadata fix, newest-500 DB query, near-duplicate drop on the refinement path, sentence-transformers moved to the `embeddings` extra. Regression: no new failures against HEAD, two fewer. | 1b57f6a | none (pre-factory) |
| 2026-10-02 pre-s0 | **P0-1 through P1-5, P0-CHECK, A-1, A-2 done (tracker rows 1–14 → done; Opus, attended).** Tracker and /next-plan-item skill created. Live step 5 run after P1-5: exit 0, 101 provider calls, $0.0155, budget armed; zero hypotheses (novelty filter, moved P2-5 to the front of P2). A-1 live subscription smoke test passed ($0.077 API-equivalent). | ec6ed67 … 3e3b80b (17 commits) | none (pre-factory; per-item acceptance tests; two live DeepSeek runs and one Claude smoke test) |

## Fresh traps (graduate to CLAUDE.md at a stage gate, then delete here)
- (s0) CLAUDE.md and .claude/*.json were gitignored, so the contract was never tracked before s0; `.gitignore` now negates `/CLAUDE.md` and `.claude/settings.json`. If a `git add` of either is refused as ignored again, the negation was lost: restore it, never `-f`.
- (s0) tests/conftest.py loads .env with override=True at import; any test run outside the ladder's `-p verify_isolate` plugin writes ResearchSession rows into the owner's kosmos.db. Run ad-hoc tests as `VERIFY_RUN_DIR=/tmp/kosmos-verify/adhoc PYTHONPATH=scripts python -m pytest <paths> -p verify_isolate --no-cov -p no:cacheprovider -q`.
