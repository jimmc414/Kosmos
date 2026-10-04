# Session 0 inventory — Kosmos Session Factory bootstrap

Date 2026-10-04 · Session s0 (Fable 5.1, judgment tier, effort high; owner present for Phase A) ·
Branch `viability-fixes` at 1247379 · master at 73f4d2a · Trust: SNAPSHOT (true on 2026-10-04;
re-verify a reading before relying on it). Process: docs/process/SOFTWARE_FACTORY_PROCESS.md §11.1.

## §0 The Session 0 prompt (condensed; owner, 2026-10-04)
Install the factory in this repository and freeze the plan that carries the remaining viability
work under an unattended queue; start no plan item. Phases: A inventory, contract, map, one
owner question round; B ladder (B1), ledger (B2), register (B3), ADRs (B4), signals (B5),
commands and hygiene (B6), the first plan freeze (B7), final ladder run, record, push, report
(B8). Facts asserted by the prompt (verified below, §9): remaining tracker rows P2-4, P2-7, P3-1,
P3-4, P3-2, P3-3, P3-5, C-1, M-1, LIVE-25 in that order; commit discipline (one commit per item,
no attribution, never force-push, push at session end) now behind the ladder with the session
number in the subject; the never-stage list; the test command; the master-baseline of
pre-existing failures until P3-3; live calls only per the tracker; ANTHROPIC_API_KEY never set;
plan line numbers exact at 6cfe7f6 only; tiers judgment = Fable high, build = Opus high. Queue
plan (§4 of the prompt): Sessions 1–3 attended `/factory-continue` on Opus high; Session 4
(Fable, owner present) builds FACTORY#runner; then the serial queue runs until
HALT_PLAN_COMPLETE or HALT_NEEDS_OWNER.

## §1 Spec, plan, tracker, skill, commands
- Spec: evaluation/VIABILITY_ASSESSMENT_AND_CHANGE_PLAN.md (181,653 B). §5 item text (28 items),
  §8 verification table (steps 1–25b), §9 non-goals, §10 twelve owner decisions, all answered
  2026-10-02. Line numbers exact at 6cfe7f6 only.
- Tracker: evaluation/VIABILITY_PROGRESS.md (30,537 B): ground rules, queue of 30 rows (20 done,
  10 todo; numbered 1–30 with 15a and no 21), the A-2 spec, a 7-line session log (2026-10-02 to
  2026-10-04). Frozen by step B2.
- Remaining rows, in the tracker's printed order: P2-4, P2-7, P3-1, P3-4, P3-2, P3-3, P3-5, C-1,
  M-1, LIVE-25 (P3-4 ahead of P3-2 by the tracker's own note). Matches the prompt.
- Branch: 24 commits on master..viability-fixes (ec6ed67 … 1247379), all pushed to
  origin/viability-fixes (origin = github.com/jimmc414/Kosmos.git). Listed with hashes in
  docs/execution/backlog/CLOSED.md after step B3.
- Skill: .claude/skills/next-plan-item/SKILL.md (superseded by /factory-continue in B6);
  .claude/skills/kosmos-e2e-testing (untouched). Agent: .claude/agents/kosmos_architect.md.
- Commands: .claude/commands/README.md and hn-rewrite.md only. No .claude/settings.json before
  s0; .claude/settings.local.json carries the owner's allowlist (git add/commit/push, python,
  pytest, …) and an empty deny list.
- Working tree at start: modified docker-compose.yml (the owner's uncommitted hardening,
  25 lines), human_review_audit.jsonl, four .literature_cache/*.pkl; untracked docs/DEEP_ONBOARD.md,
  docs/DEEP_ONBOARD_VALIDATION.md, docs/xray.md, evaluation/CRITICAL_EVALUATION_REPORT.md,
  evaluation/*_findings.md, evaluation/data/perovskite_solar_cell_test.csv, persona definition
  002 and persona runs. None of these are ever staged.

## §2 Tests and their runtimes
- Runner: `python -m pytest <paths> --no-cov -p no:cacheprovider -q`. Never bare `pytest`:
  pytest.ini addopts add `--cov-fail-under=80`, `filterwarnings = error`, `timeout = 300` per
  test (thread), `asyncio_mode = auto`, `log_file = tests/test_run.log`.
- Collection (2026-10-04, 0 collection errors): tests/unit + tests/integration 2,951 tests
  (12 s to collect, 29 s wall); tests/requirements 815; tests/e2e 121 (live calls:
  tests/e2e/conftest.py loads .env with override=True; never in the ladder); tests/manual 14.
- Runtime, measured 2026-10-04 on the s0 tree with the isolation plugin, serial, uncached:
  tests/unit + tests/integration = **447 s (7 min 27 s) of pytest, 468 s wall**:
  285 failed, 2,353 passed, 212 skipped, 107 errors. The ladder's gate 2 is this run.
- Pre-existing failures: the tracker's figure (98 failed + 62 errors) covers only
  tests/unit/{agents,cli,core,db} on master at 73f4d2a; the P2-3 session saw 128 failed + 99
  errors over a wider path set. The full unit + integration set on the s0 tree has 285 + 107 =
  392 red node ids, concentrated in: test_feedback 31, test_guardrails 29, test_convergence 28,
  test_domain_router 27, test_verifier 26, integration/test_parallel_execution 22, test_memory 18,
  cli/test_commands 18, test_skill_loader 16, integration/test_iterative_loop 15, test_citations
  14, integration/test_cli 12, test_graph_commands 11, integration/test_execution_pipeline 11,
  and a long tail. Two single failures in P2-6/A-1 test files (test_litellm_structured.py,
  test_claude_code_provider.py) are checked alone in B1 before stamping. The stamped list is
  scripts/test_baseline.txt (ADR-0002).
- tests/conftest.py:25-27 `load_dotenv(.env, override=True)` at import, so
  `DATABASE_URL=sqlite:///./kosmos.db` reaches every test; director tests that use the
  configured DB write ResearchSession rows into the owner's kosmos.db (the P2-3 session added
  46). **Mechanism proven 2026-10-04:** a pytest plugin passed with `-p` sets DATABASE_URL,
  KOSMOS_ARTIFACTS_DIR, CHROMA_PERSIST_DIRECTORY and LOG_FILE_PATH in `pytest_configure`, which
  runs AFTER the initial conftest import; across the full run the owner's kosmos.db md5
  (2758f443…) was unchanged and research_sessions stayed 0, while the scratch DB received 92
  ResearchSession rows and was migrated to a0aa37ea19f2. Shipped as scripts/verify_isolate.py (B1).
- Lint: pyproject [tool.ruff] is configured (select E, W, F, I, B, C4, UP; the top-level keys
  are deprecated in ruff 0.14.4 and warn). `ruff check kosmos` reports **4,241 errors** (3,731
  auto-fixable) on the s0 tree, so a full-ruff gate would be permanently red. The critical subset
  `--select E9,F63,F7,F82` reports **1 error**: kosmos/experiments/templates/materials/
  shap_analysis.py:491 F821 undefined name `top_3_features` (Tier C, kept by D-04). Gate 1
  judges that subset against scripts/lint_baseline.txt. black is configured but nothing enforces
  it. Makefile `lint` runs pylint/mypy/flake8 with `|| true` (never fails). .pre-commit-config.yaml
  exists (black, ruff, verify-imports, smoke-test) but no hook is installed.
- CI: none (.github/workflows/ absent). P3-3 step 5 adds .github/workflows/unit.yml.

## §3 Scheduled jobs
- crontab (user jim): six entries, all /mnt/c/python/collection-software/genesis/scripts/*
  (another project): partition-autocreate 04:00, monthly-close 04:30, nightly-derivation-diff
  05:00, nightly-shadow-replay 05:30, report-runner 06:00, serve-keepalive every 10 min. None
  touches Kosmos; the only effect is machine load 04:00–06:15 (a slow ladder, never a red).
- systemd timers: OS defaults only (apt-daily, e2scrub_all, man-db). None for Kosmos.
- Kosmos has no scheduled job of its own; the signals script therefore has no scheduler check.

## §4 Docker services
- Daemon: Docker 29.0.1 (Windows Docker Desktop, reachable from WSL). Running at s0:
  kosmos-neo4j (neo4j:5.14-community, 127.0.0.1:7474/7687, healthy), kosmos-redis (redis:7-alpine,
  127.0.0.1:6379, healthy), kosmos-postgres (postgres:15-alpine, 0.0.0.0:5432),
  genesis-walreceiver (another project).
- Sandbox image kosmos-sandbox:latest = b4d4829898c1, 2.57 GB, built 2026-10-02 (P0-CHECK).
  Required by ladder gate 4 and by `kosmos run`.
- docker-compose.yml (committed version): services kosmos (8000:8000, healthcheck
  `requests.get(:8000/health)` that nothing answers), postgres, redis, neo4j, pgadmin; volumes
  postgres_data, redis_data, neo4j_data/logs/import/plugins, pgadmin_data. The working tree holds
  the owner's uncommitted hardening. P3-2 edits this file and is OWNER-gated.
- What the ladder needs: the daemon and kosmos-sandbox:latest (gate 4). The unit and integration
  suites skip or fail identically with or without Neo4j/Redis/Postgres (212 skipped; the
  baseline is stamped with the three containers up, which the signals script reports).

## §5 Shared mutable state
| Path | What | Size | Tracked | Rule |
|---|---|---|---|---|
| kosmos.db | the owner's SQLite DB: alembic a0aa37ea19f2; 22 hypotheses, 8 experiments, 2 results, 0 research_sessions | 416 K | no (`*.db` ignored) | the ladder never opens it; gate 3 migrates a copy; delete or re-init = STOP-AND-ASK |
| kosmos_test.db | stray test DB, no alembic_version table | 52 K | no | leave it |
| postgres_data/ redis_data/ neo4j_data/ neo4j_logs/ neo4j_import/ neo4j_plugins/ | compose bind volumes (neo4j_data 518 M) | | no | never delete (STOP-AND-ASK) |
| chroma_db/ and .chroma_db/ | Chroma persistence; two directories because the CLI and the core resolve the path differently (DEEP_ONBOARD gotcha) | 164 K each | no | gate 2 redirects CHROMA_PERSIST_DIRECTORY |
| logs/ | kosmos.log (LOG_TO_FILE) | 32 K | no | gate 2 redirects LOG_FILE_PATH |
| artifacts/ | tracked baseline_* files; artifacts/runs/ is gitignored (D-08) | 64 K | partly | gate 2 redirects KOSMOS_ARTIFACTS_DIR |
| htmlcov/ coverage.xml tests/test_run.log .kosmos_cache/ | test and coverage outputs | 25 M, 1.3 M, 504 K, 16 K | no | never staged; gate 2 writes its log to the scratch dir |
| ports | 5432 postgres · 6379 redis · 7474/7687 neo4j · 8000 compose kosmos (nothing listens) | | | the ladder opens no port |

## §6 Paths that may carry secrets or private data
- .env (3,855 B, gitignored): DEEPSEEK_API_KEY, SEMANTIC_SCHOLAR_API_KEY, OPENAI_API_KEY,
  NEO4J_PASSWORD, NEO4J_AUTH, REDIS_PASSWORD, KOSMOS_PG_SUPER_PASSWORD, KOSMOS_PGADMIN_PASSWORD,
  plus non-secret settings (LLM_PROVIDER, LITELLM_MODEL, DATABASE_URL=sqlite:///./kosmos.db, …).
  No ANTHROPIC_API_KEY, and none may ever be added (D-19). .env.backup (3,296 B, gitignored).
- ~/.claude/.credentials.json (outside the repo): Max-subscription OAuth, read by the claude_code
  provider. ANTHROPIC_API_KEY is unset in the shell.
- human_review_audit.jsonl (61,560 B, 384 lines, TRACKED and modified in the tree): the owner's
  review audit. Never staged, never copied.
- .literature_cache/ (4.5 MB; 5 .pkl files TRACKED despite .gitignore:146, 4 modified): pickled
  literature responses. Never staged; pickle loads execute code (DEEP_ONBOARD gotcha).
- The untracked docs/ and evaluation/ reports and persona runs (§1): the owner's; never staged.
- Not secrets: k8s/secrets.yaml.template (template), kosmos-claude-scientific-skills/**/secrets.md
  and tokenizers.md (documentation). docker-compose.yml commits default passwords
  (kosmos-password, kosmos-dev-password; DEEP_ONBOARD gotcha).

## §7 Tool read ceiling (measured 2026-10-04 on docs/DEEP_ONBOARD.md, 541,848 B, 4,666 lines)
- Read tool: refuses a whole file over 256 KB (`File content (529.1KB) exceeds maximum allowed
  size (256KB)`), AND refuses any slice over 25,000 tokens (`File content (93221 tokens) exceeds
  maximum allowed tokens (25000)` for lines 1–2000 = 225 KB; even lines 4001–4666 = 75 KB =
  31,161 tokens were refused). Usable page: about 500 lines / 55 KB of prose.
- Bash tool: 40,000 B of output returns intact; a 570 KB output is spilled to a tool-results file
  with a 2 KB inline preview (the spill threshold lies between 40 KB and 570 KB; not narrowed).
- Consequence: hot-file lines stay at the process defaults (SESSION_STATE.md 200,000 B or 12
  blocks; docs/MAP.md, docs/PLAN.md and each register file 225,000 B; all under 262,144 B), and
  every canonical file is written so that any one section fits a 500-line page.

## §8 Toolchain
- Python 3.11.11, conda env `llm` (/home/jim/miniconda3/envs/llm/bin/python); docker SDK 7.1.0,
  litellm, SQLAlchemy 2.0.44, alembic, pytest 9.0.2 (pytest-timeout, pytest-asyncio, pytest-cov),
  ruff 0.14.4, tmux 3.4, Claude Code 2.1.289. Disk: /mnt/c 76 G free (82 % used); / 883 G free.
- alembic: alembic.ini:19 fallback `sqlite:///kosmos.db`; alembic/env.py:31-38 `get_url()`
  prefers `get_config().database.url`, and pydantic-settings lets os.environ win over .env, so
  `DATABASE_URL` in the environment redirects a migration (gate 3 relies on this and proves it
  by md5). Revisions: 2ec489a3eb6b → fb9e61f33cbf → dc24ead48293 → a0aa37ea19f2 (head).
- Makefile targets test/test-unit/test-int call bare `pytest` (never use them).

## §9 Corrections to the prompt's §1 facts (the repo wins)
- "98 failed and 62 errors across tests/unit/agents, cli, core, db" is the tracker's figure for
  those four directories only. The ladder's gate 2 covers tests/unit + tests/integration, where
  the s0 tree has 285 failed + 107 errors; the stamped baseline is that larger list, with the
  cause "pre-existing on master at 73f4d2a or recorded in the tracker Notes; stamped on the s0
  tree at 1247379". Nothing else in §1 was found wrong.
- The prompt's deny-list spelling: `Bash(git add .)` is listed as an exact match (a prefix form
  would also deny `git add .claude/settings.json`).
- CLAUDE.md and .claude/settings.json were gitignored (.gitignore:155 `CLAUDE.md`, :12 `.claude/*.json`),
  so the pre-factory CLAUDE.md was never tracked. The contract and the deny-list must live in the
  repository (process §1 law 1, §3.1), so s0 adds `!/CLAUDE.md` and `!.claude/settings.json` to
  .gitignore (settings.local.json stays ignored). Default taken without asking: not a STOP-AND-ASK
  item; recorded here and in the s0 session block.
- Added to the never-stage list from the inventory: .chroma_db/, tests/test_run.log,
  kosmos_test.db (all untracked outputs).

## §10 Owner rulings (Phase A, recorded verbatim with a `put:` clause)
One AskUserQuestion round, 2026-10-04, s0. Each answer is the owner's selected option, quoted.

- **(a) STOP-AND-ASK list.** Question: the seven-item list as drafted in CLAUDE.md (spending
  money or any live LLM call beyond the tracker's authorization · staging or editing
  docker-compose.yml · deleting or re-initializing a data directory, a compose volume, or any
  database · a force-push, or any push to master · changing an owner decision in plan §10 ·
  anything in plan §9 non-goals · a suspected secret or private-data exposure).
  Answer: "Approve the seven items as drafted (Recommended)".
  put: CLAUDE.md §STOP AND ASK THE OWNER (unchanged from the draft); docs/MAP.md §6.
- **(b) Locked decisions and exit gates.** Question: sign D-01 to D-24 and the S0–S4 exit gates
  as drafted in docs/MAP.md §2 and §4.
  Answer: "Sign D-01–D-24 and the S0–S4 exit gates as drafted (Recommended)".
  put: docs/MAP.md §2 (stage table) and §4 (locked decisions), signed; docs/MAP_CHANGELOG.md
  row dated 2026-10-04 carries the signature.
- **(c) Runner permission mode.** Question: the skip-permissions flag with the settings deny-list
  carrying the safety (Genesis practice) or a per-project allowlist.
  Answer: "Skip-permissions flag + deny-list (Recommended; Genesis practice)".
  put: docs/MAP.md §4 D-24; .claude/settings.json `permissions.deny` (47 rules) is the guard;
  FACTORY#runner (Session 4) passes `--dangerously-skip-permissions` in run_watched.sh.
- **(d) Where Session 0's files land.** Question: viability-fixes (default) or a branch of its own.
  Answer: "On viability-fixes (Recommended; default)".
  put: docs/MAP.md §4 D-23; every s0 commit is on viability-fixes.
