# Kosmos Viability Build — Session Contract

Kosmos = an autonomous-research CLI (`kosmos run`) being made viable against the plan in
evaluation/VIABILITY_ASSESSMENT_AND_CHANGE_PLAN.md: one real experiment per run, executed in the
Docker sandbox on the owner's dataset, validated by recomputation and a permutation null, and
reported honestly with real cost. You are one session in a multi-session build run by the Session
Factory process (docs/process/SOFTWARE_FACTORY_PROCESS.md). The map is docs/MAP.md. Trust the docs
in the order docs/MAP.md §3 gives, not your priors about this repo.

## Codebase Onboarding
Read docs/DEEP_ONBOARD.md before starting any task. It contains verified behavioral documentation,
critical paths, gotchas, and conventions for this codebase. It is MIXED trust (docs/MAP.md §3): its
known-wrong claims are listed in plan item P3-5. It is 542 KB and cannot be read in one call: grep
its `## Gotchas` section and its Change Impact Index for the files your milestone touches.

## Commit Guidelines
- Do not include "Co-Authored-By: Claude" in commit messages
- Do not include "Generated with Claude Code" attribution
- Commits should appear as the repository owner
- Subject: `s<N> — <milestone id>: <imperative summary>` (Session 0 steps: `s0 — <step>: <summary>`).
  One commit per milestone. Stage by explicit path. Never force-push. Push right after EVERY
  commit (owner, 2026-10-08), and again at session end: `git push -u origin viability-fixes`.
  Stay on `viability-fixes`.

## ON SESSION START — AUTONOMOUS MODE
0. Run `bash scripts/signals_check.sh` (read-only, under 60 s). rc 0 = OK · 2 = a RED · 1 = could
   not look, which is NOT a green. Paste its `signals:` line into your block verbatim. On RED:
   record a register row (docs/execution/BACKLOG.md), never fix mid-milestone.
1. Read docs/MAP.md §2 (stage table): know the active stage and its exit gate.
2. Read SESSION_STATE.md: its NEXT ACTION is your task, UNLESS §HOLD reads ACTIVE, in which case
   the hold's queue is your task list and NEXT ACTION is parked. Queue empty ⇒ stop and tell the
   owner.
3. `git log --oneline` since the newest `## Session log` row (its date and Commits cell). A commit
   already delivering NEXT ACTION = done-but-unrecorded: record it, advance.
4. Read docs/PLAN.md, find the milestone NEXT ACTION names. If blocked or done, take the first open
   row in Lanes 1–2 whose tier tag matches your model, in printed order; a Lane-0 row preempts by
   its own trigger. Long tail: docs/execution/BACKLOG.md by id. No PLAN.md ⇒ write the
   interstitial (/factory-spec interstitial) before any other work.
5. Begin immediately. Sessions run unattended; asking "should I continue?" blocks the work. Stop
   only where this file says to. Never re-plan the stage.

User override always wins: comply, then record the deviation. A question or a problem described is
answered, not acted on; a request phrased as a question is a request.

## ON SESSION END (or before compaction) — MANDATORY
1. `bash scripts/verify.sh` → `VERIFY PASS (4 gates)` (or record which gate is red and why).
2. Commit milestone-complete GREEN work by EXPLICIT paths (never `git add -A`). Partial or red work
   is NEVER committed.
3. Update SESSION_STATE.md: rewrite NEXT ACTION (never stack), add your `<!-- session s<N> -->`
   block, insert your log row directly below the `|---|---|---|---|` separator, flip the BACKLOG
   rows you touched, all in the SAME commit. Then `bash scripts/hot_files_check.sh`: exit 1 ⇒ roll
   (`python3 scripts/session_state_roll.py --write`) or split BEFORE the commit. Anchor every edit
   on the LINE-START header (`^## NEXT ACTION`); afterwards `grep -c '^## NEXT ACTION'` must print 1.
4. `git push -u origin viability-fixes`.
An unrecorded session is a lost session.

## STAGE NAVIGATION
- After EVERY milestone check the exit-gate row of its own stage in docs/MAP.md §2. If met, THIS
  session makes the stage-gate commit: flip the status, append the docs/MAP_CHANGELOG.md line, set
  the next critical-path row `active`.
- docs/PLAN.md exhausted ⇒ archive it and /factory-spec the next stage. Stages start ONLY through
  /factory-spec.
- Blocked on the owner ⇒ never idle: take the next open row for your tier.
- Work ONLY the milestone. An out-of-milestone defect gets a register row. A plan item that looks
  trivial still waits for its own /factory-continue pass.
- Mid-milestone wrap: commit NOTHING. NEXT ACTION = "resume <milestone> at <step>".
- At session end, sweep §Waiting-on-owner and the register's OWNER rows: surface every item whose
  escalate-by is at or before the active stage.

## UNDER THE RUNNER (when the prompt carries "### LIVE FACTS")
Nobody is at the keyboard. Never ask a question through the question tool: take the recommended
default and lodge an OWNER row. A STOP-AND-ASK item or an unfixable red ⇒ the mid-milestone wrap,
one BLOCKED line, and write your halt class (HALT_HARD_STOP · HALT_RED_GATE · HALT_NEEDS_OWNER ·
HALT_PLAN_COMPLETE · HALT_HOLD_EMPTY · HALT_PRECONDITION) into your row's done file. Never end your
turn to wait; wait inside it. Never match a process by a pattern this prompt contains.

## STANDING TRAPS
- NEVER `git add -A` · `git add .` · `git add --all` · `git add -u` · `git commit -a` ·
  `git reset --hard` · `git clean -f` · `git stash -u` · `git push --force` (enforced by the
  deny-list in .claude/settings.json).
- NEVER stage: docker-compose.yml (the owner's uncommitted hardening) · .literature_cache/ ·
  human_review_audit.jsonl · the untracked docs/ and evaluation/ reports · kosmos.db ·
  kosmos_test.db · postgres_data/ · neo4j_*/ · redis_data/ · chroma_db/ · .chroma_db/ · logs/ ·
  htmlcov/ · coverage.xml · tests/test_run.log. Stage by name.
- Private data: .env, .env.backup, human_review_audit.jsonl, .literature_cache/ and
  ~/.claude/.credentials.json. Never copy their rows or values into docs, fixtures, commits,
  prompts or temp dirs. Print key NAMES only, never values.
- NEVER set or write ANTHROPIC_API_KEY anywhere (code, tests, .env files, the runner's env). The
  API path uses KOSMOS_ANTHROPIC_API_KEY; the subscription path reads ~/.claude/.credentials.json.
- Shared state: kosmos.db is the owner's database. Tests and the ladder never open it (gate 2 runs
  under scripts/verify_isolate.py; gate 3 migrates a scratch copy). Never delete or re-initialize
  kosmos.db, a data directory or a compose volume. The containers kosmos-postgres, kosmos-redis
  and kosmos-neo4j and ports 5432, 6379, 7474, 7687 are the owner's; the ladder opens no port and
  starts no service.
- Live LLM calls spend money. Authorized only where docs/MAP.md §4 D-18 says (DeepSeek, plan §8
  steps 5 and 25, `--budget 1`; the one Claude smoke test is spent). Never inside the ladder.
  Everything else is mocked (`patch('kosmos.agents.<module>.get_client')`).
- Baselines (scripts/test_baseline.txt, scripts/lint_baseline.txt) move only through
  `bash scripts/verify.sh --accept-baseline "<cause>"` with the cause written in the same commit;
  "the gate is red" is never a cause.
- Behavior you do not know: extract from the plan, the tracker Notes, or the code; never invent.
  Plan line numbers are exact at 6cfe7f6 only; locate targets by the quoted code.
- Tests: `python -m pytest <paths> --no-cov -p no:cacheprovider -q`. NEVER bare `pytest` (80 %
  coverage gate, warnings-as-errors, tests/conftest.py loads .env so tests/e2e makes live calls).
  tests/e2e is never in the ladder. Ad-hoc runs that must not touch kosmos.db:
  `VERIFY_RUN_DIR=/tmp/kosmos-verify/adhoc PYTHONPATH=scripts python -m pytest <paths> -p verify_isolate --no-cov -p no:cacheprovider -q`.
- .gitignore negates `/CLAUDE.md` and `.claude/settings.json`; if a `git add` of either is refused as
  ignored, the negation was lost: restore it, never `git add -f`.
- Match the surrounding code: imports in stdlib / third-party / kosmos blocks;
  `logger = logging.getLogger(__name__)`; `datetime.now(timezone.utc)`; absolute `kosmos.*` imports.

## VERIFY DISCIPLINE
`bash scripts/verify.sh` = the 4-gate ladder: 1 compile-lint (compileall + ruff critical subset
against scripts/lint_baseline.txt) · 2 tests (tests/unit + tests/integration, serial, uncached,
isolated from kosmos.db, judged against scripts/test_baseline.txt) · 3 alembic (upgrade head and
downgrade one on a scratch copy) · 4 template-run (the no-LLM bound template through the real
sandbox on the climate CSV). Pass string: `VERIFY PASS (4 gates)`. Launch it detached
(`bash scripts/verify.sh --detach`, then poll the marker it names) and read the string from the
log, never from an exit code. A red from a parallel or cached run is not a result. Needs: the
Docker daemon and kosmos-sandbox:latest. Env: .env (never printed). Duration: about 8 minutes (471 s measured s0 2026-10-04, first
green run: 03:00:09 → 03:08:00; gate 2 alone about 7 minutes).

## PROCESS RULES (graduated from SESSION_STATE §STANDING after biting twice; numbers are stable ids,
never reused; rules 1–11 are inherited from the process document §9.2 at adoption)
(1) Never read a result off a pipeline's exit status. (2) Never read a result off a backgrounded
wrapper's exit code. (3) Never trust a liveness probe that can match its own command line; confirm
with a process listing that shows elapsed time; a `pkill` by pattern can kill the session running
it. (4) Never trust a sleep-based clock; read the real time. (5) Never read a census through a
truncation. (6) The adversarial pass binds by blast radius; the self-run pass is the floor; agents
are additive. (7) Spawned agents share the live tree: checksum before, diff after, no ladder run
during a mutation pass. (8) Never run two test suites against one shared database. (9) A mutation
harness needs its own RED and GREEN controls. (10) Know which typecheck command really checks.
(11) A red under machine load is a timeout, not a result: re-run the file alone, then the full
ladder on the unchanged tree.

## STOP AND ASK THE OWNER
- Spending money or any live LLM call beyond the tracker's authorization (MAP §4 D-18).
- Staging or editing docker-compose.yml.
- Deleting or re-initializing a data directory, a compose volume, or any database (kosmos.db
  included).
- A force-push, or any push to master.
- Changing an owner decision in plan §10 (MAP §4 D-01 to D-12).
- Anything in plan §9 non-goals.
- A suspected secret or private-data exposure.
Everything not on this list proceeds. A session blocked on an ask never idles (take the next open
row for your tier). Under the runner, a STOP-AND-ASK item is HALT_HARD_STOP.

## AMENDMENT
Reality contradicts a doc ⇒ verify with file:line evidence ⇒ fix the doc + one
docs/MAP_CHANGELOG.md line in the same commit ⇒ proceed. Never silently diverge.

## DECISION RECORDS
A significant decision (costly to reverse · external contract · correctness posture · schema or
money shape · security/privacy · cross-session convention) gets an immutable record in docs/adr/
in the SAME commit. Never edit an accepted record; append a dated correction note or supersede.

## COMMANDS
/factory-continue · /factory-spec · /factory-verify · /factory-queue live in .claude/commands/.
The old /next-plan-item skill is superseded; its tracker evaluation/VIABILITY_PROGRESS.md is frozen.
