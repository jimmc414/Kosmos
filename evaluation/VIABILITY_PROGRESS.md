# Viability Plan Progress

Tracker for executing `evaluation/VIABILITY_ASSESSMENT_AND_CHANGE_PLAN.md` (the plan) across sessions. The `/next-plan-item` skill (`.claude/skills/next-plan-item/SKILL.md`) reads this file, does the next item, and updates it. This file and `git log viability-fixes` are the only state that survives a context clear.

## Ground rules

- **Branch**: `viability-fixes`, cut from master at 73f4d2a (code identical to 6cfe7f6, the commit the plan's line numbers refer to).
- **One commit per item**, subject `<ID>: <imperative summary>`, made only after the item's acceptance tests pass. No attribution lines. No push until the owner asks. Find an item's commit with `git log --oneline --grep "^<ID>:"`.
- **Never stage**: docker-compose.yml (owner's uncommitted hardening), `.literature_cache/`, human_review_audit.jsonl, the untracked docs/ and evaluation/ reports, `kosmos.db`. Stage files by name.
- **Tests**: `python -m pytest <paths> --no-cov -p no:cacheprovider -q`. Never bare `pytest` (80 percent coverage gate, warnings are errors, tests/conftest.py loads .env so tests/e2e makes live calls).
- **Line numbers** in the plan are exact at 6cfe7f6. Once an earlier item has edited a file, locate targets by the quoted code, not the number.
- **Pre-existing failures** (master at 73f4d2a): `tests/unit/agents tests/unit/cli tests/unit/core tests/unit/db` has 98 failed and 62 errors (test_feedback 31, test_convergence 28, test_domain_router 27, test_memory 18, cli/test_commands 18, test_skill_loader 16, test_graph_commands 11, test_cache 8, core/test_workflow 2, test_research_director_loops 1 (`_actions_this_iteration` missing on the mock_director fixture)). To prove an item adds none, run the same paths in a master worktree (`git worktree add <scratchpad>/master_wt master`) and diff the `^(FAILED|ERROR) tests/` lines.
- **Live calls authorized**: DeepSeek for plan Section 8 steps 5 and 25 with `--budget 1` (report cost). Claude subscription: one smoke test only (one `generate`, one `generate_structured`) during A-1; no research run on the subscription.
- **Owner decisions** in plan Section 10 are binding. Decisions added 2026-10-02 at execution start: order P0 then P1 then providers then the rest; provider switching through CLI flags with .env as the default (spec A-2 below).

## Queue

Status values: `todo`, `in progress`, `done`, `blocked` (reason in Notes). Work top to bottom; a `blocked` item does not block later items unless they depend on it.

| # | ID | Title | Plan section | Status | Notes |
|---|---|---|---|---|---|
| 1 | P0-1 | Generic template self-contained and syntax-safe | 5 / P0-1 | done | Deviation: plan test (c) asserted "kosmos" not in the prompt, which contradicts the required prompt text "Do NOT import kosmos"; the test asserts no DataAnalyzer / kosmos.execution and that the new instructions are present. Fixture Variable descriptions need at least 10 characters. |
| 2 | P0-2 | `RESULT:` JSON footer in _execute_in_sandbox | 5 / P0-2 | done | SANDBOX_RESULT_FOOTER sits after DEFAULT_EXECUTION_TIMEOUT in kosmos/execution/executor.py |
| 3 | P0-3 | execute_with_data skips the host prefix when sandboxed | 5 / P0-3 | done | |
| 4 | P0-4 | Director reads exec_result.success, honest rows, fail-fast | 5 / P0-4 | done | Protocol to_dict/model_validate round-trip works (no defect). Additions beyond the plan text: `kosmos run` exits 1 after printing the halt reason; _json_safe also maps numpy NaN/inf to None; result data carries executor_mode (sandbox/host/none). |
| 5 | P0-5 | TTest and Correlation templates self-contained | 5 / P0-5 | done | Column and group names are emitted once as module variables (`_group_col`, `_measure_col`, `_label1`, `_label2`, `_x_col`, `_y_col`) instead of inline `!r` literals, because a repr containing a quote breaks inside the generated f-strings. The result prints emitted doubled braces (printed literal `{...}`); fixed to single braces. Extra stale assertions updated: TestLLMFallback.test_template_preferred_over_llm_when_available and tests/integration/test_execution_pipeline.py:99,222 (that file's other 11 tests error on master too: `StatisticalTest` not imported in its fixture). |
| 6 | P0-6 | LogLog and ML templates self-contained | 5 / P0-6 | done | Same module-variable pattern as P0-5; LogLog also sets effect_size = spearman_rho. Extra stale assertion updated: test_ml_code_generation expected run_experiment/cross_validate, now cross_val_score. Section 8 step 4's `grep -c DataAnalyzer` prints 1, not 0: the remaining hit is P0-1's negative assertion `"DataAnalyzer" not in prompt`. |
| 7 | P0-CHECK | Build sandbox image; live end-of-P0 DeepSeek run | 5 / "End-of-P0 check"; 8 / steps 4b, 5 | done | Image kosmos-sandbox:latest built 2026-10-02 (sha256:b4d48298...). A no-LLM template run through the real container on the climate CSV returned success, data_source file, n 64. Live step 5 ran at 32f5ecb (P0 plus P1-1): result b9e3bf60 has execution_success 1, data_source file, p 8.3e-53 (year vs co2_ppm, as predicted until P2-1); experiment e7d2683b COMPLETED with code stored. P1-1 then marked hypothesis d489043b SUPPORTED from that wrong pair: the false positive P2-1/P2-2 remove. The CLI crashed at the end display (`domain` None -> `.title()`, pre-existing for any run without --domain); fixed in this item with tests/unit/cli/test_results_viewer.py. Cost not reported (no tracking before P1-5). Also seen: logs/kosmos.log is not written although LOG_TO_FILE=true; novelty_checker NaN warning (P2-5). |
| 8 | P1-1 | Verdict and hypothesis status persisted | 5 / P1-1 | done | Helper `_db_result_to_experiment_result` sits just above `_handle_analyze_result_action`; P1-4 reuses it. The DB-backed director fixture moved to tests/unit/agents/conftest.py as `db_director` (constants H_ID, EXP_ID, CODE importable from tests.unit.agents.conftest). |
| 9 | P1-2 | Leave REFINING after every refinement pass | 5 / P1-2 | done | Loop-closure test needs an untested hypothesis in the pool: with none, decide_next_action in REFINING converges before refining, so the dead end only reproduces while untested work remains. `_leave_refining` sits just above `_handle_refine_hypothesis_action`. |
| 10 | P1-3 | Error recovery never blocks the event loop | 5 / P1-3 | done | Tests (c) ERROR_RECOVERY and (d) sync backoff also pass on the old code: those paths were unreachable, not broken. |
| 11 | P1-4 | Convergence detector gets real hypotheses and results | 5 / P1-4 | done | Test lives in tests/unit/agents/test_research_director_execute.py. Optional criteria (novelty_decline, diminishing_returns) are now live and may stop runs earlier. |
| 12 | P1-5 | `--budget` armed, provider calls recorded, per-model pricing | 5 / P1-5 | todo | After committing, re-run Section 8 step 5 once (authorized, `--budget 1`) to confirm the end display renders and a non-zero cost is reported |
| 13 | A-1 | Anthropic via KOSMOS_ANTHROPIC_API_KEY and Claude Code subscription | 5 / A-1 | todo | Includes the one-call live subscription smoke test |
| 14 | A-2 | `--provider` / `--model` flags on `kosmos run` | this file, "A-2 spec" | todo | Not in the plan; spec below |
| 15 | P2-0 | Result columns for execution, validation, provenance, cost | 5 / P2-0 | todo | |
| 16 | P2-6 | LiteLLM JSON mode and tolerant parsing | 5 / P2-6 | todo | Moved ahead of P2-1, which depends on it |
| 17 | P2-1 | Dataset schema and variable-to-column binding | 5 / P2-1 | todo | |
| 18 | P2-2 | Recomputation, permutation null, ScholarEval advisory, verdict rule | 5 / P2-2 | todo | |
| 19 | P2-3 | Seed and provenance | 5 / P2-3 | todo | |
| 20 | P2-4 | Honest run report and real cost | 5 / P2-4 | todo | |
| 21 | P2-5 | Novelty without sentence-transformers | 5 / P2-5 | todo | |
| 22 | P2-7 | Hypothesis-pool control | 5 / P2-7 | todo | |
| 23 | P3-1 | CodeValidator and emergency stop on the director path | 5 / P3-1 | todo | |
| 24 | P3-4 | Archive zero-importer modules (Tiers A and B; Tier C kept) | 5 / P3-4 | todo | Ahead of P3-2 because P3-2 step 3 archives api modules |
| 25 | P3-2 | Image build, declared dependencies, no HTTP probe | 5 / P3-2 | todo | Touches docker-compose.yml: ask the owner before staging it |
| 26 | P3-3 | Test suite green for surviving modules | 5 / P3-3 | todo | |
| 27 | P3-5 | README and DEEP_ONBOARD corrections | 5 / P3-5 | todo | Document the A-2 flags in the README too |
| 28 | C-1 | Findings JSON and `kosmos report` | 5 / C-1 | todo | |
| 29 | M-1 | `kosmos validate-null` and `kosmos rerun` | 5 / M-1 | todo | |
| 30 | LIVE-25 | Live acceptance run on the climate CSV | 8 / step 25 | todo | Live, DeepSeek, `--budget 1`; then report, rerun, validate-null |

## A-2 spec: provider and model switching flags

Owner request 2026-10-02: users must be able to use DeepSeek or Anthropic models and switch easily. Depends on A-1 (the `claude_code` provider and KOSMOS_ANTHROPIC_API_KEY).

**Files.** kosmos/cli/commands/run.py (`run_research` options, the config overrides before `flat_config`, the start panel); kosmos/core/llm.py `get_client(reset=True)`; kosmos/config.py (`llm_provider`, `litellm.model`, `claude.model`/`anthropic.model`, the A-1 `claude_code.model`); kosmos/cli/main.py `doctor` and `config` display.

**Required.**
1. Options on `kosmos run`: `--provider/-P` with choices `deepseek`, `litellm`, `anthropic`, `claude-code`; `--model/-m` taking a model id or alias. Both default to None, meaning .env decides, exactly as today.
2. Provider mapping: `deepseek` sets llm_provider `litellm` and, when `--model` is absent, model `deepseek/deepseek-chat` (LiteLLM reads DEEPSEEK_API_KEY). `litellm` keeps the .env LiteLLM settings. `anthropic` sets llm_provider `anthropic` (API billing through KOSMOS_ANTHROPIC_API_KEY, A-1). `claude-code` sets llm_provider `claude_code` (subscription, A-1).
3. Aliases, resolved before the mapping: `opus` claude-opus-5-5, `sonnet` claude-sonnet-5-5, `haiku` claude-haiku-4-5, `fable` claude-fable-5-1, `deepseek` and `deepseek-chat` deepseek/deepseek-chat, `deepseek-reasoner` deepseek/deepseek-reasoner. Any other string passes through unchanged. Keep the table in one module-level dict so `kosmos doctor` can print it.
4. Validation: a Claude model (alias or an id starting with `claude-`) with an effective provider of litellm or deepseek, or a deepseek model with anthropic or claude-code, exits 1 with an actionable message naming the matching `--provider` values. `--provider anthropic` without KOSMOS_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY exits 1 and names KOSMOS_ANTHROPIC_API_KEY, and mentions `--provider claude-code` as the no-key option. Never write ANTHROPIC_API_KEY into os.environ.
5. Apply the overrides to the `get_config()` object before the director is constructed, then call `get_client(reset=True)` so every agent's `get_client()` returns the selected provider.
6. The start panel and the end-of-run metrics summary show `Provider: <name>  Model: <id>`. `kosmos doctor` reports the .env default provider and model, whether DEEPSEEK_API_KEY and KOSMOS_ANTHROPIC_API_KEY are set (never their values), whether `claude --version` succeeds, and the alias table.
7. .env.example and the README provider section (README.md:162-177) show switching examples: `kosmos run "Q" --provider deepseek`, `kosmos run "Q" --provider claude-code --model opus`, `kosmos run "Q" --provider anthropic --model sonnet`.

**Acceptance test.** New file tests/unit/cli/test_provider_flags.py using typer's CliRunner with ResearchDirectorAgent, `run_with_progress_async` and `get_client` patched, and `reset_config()` around each test: (a) `--provider deepseek` leaves llm_provider `litellm` and litellm.model `deepseek/deepseek-chat`, and get_client was called with reset=True; (b) `--provider claude-code --model sonnet` gives llm_provider `claude_code` and claude_code.model `claude-sonnet-5-5`; (c) `--provider deepseek --model opus` exits 1 and the output names `--provider claude-code`; (d) `--provider anthropic` with neither key variable set exits 1 naming KOSMOS_ANTHROPIC_API_KEY; (e) no flags leaves the .env provider and model unchanged and does not reset the client; (f) os.environ has no ANTHROPIC_API_KEY after any of these.

**Effort.** 2 h.

## Session log

Newest last. One line per session: date, items completed, anything the next session must know.

- 2026-10-02: tracker and `/next-plan-item` skill created; P0-1 started.
