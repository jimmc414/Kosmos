---
name: next-plan-item
description: Resume the Kosmos viability change plan - read evaluation/VIABILITY_PROGRESS.md, pick the next queued item, load only the context it needs, implement it, run its acceptance tests, commit it on the viability-fixes branch, and update the tracker. Use after a context clear to continue the plan one item at a time. Args - none (next item), an item ID such as P1-3, "status" (show progress only), or "continue N" (do up to N items).
argument-hint: "[status | <item ID> | continue N]"
---

# Next plan item

Executes `evaluation/VIABILITY_ASSESSMENT_AND_CHANGE_PLAN.md` (the plan) one item at a time so the owner can stop, clear context, and resume. State lives in two places only: `evaluation/VIABILITY_PROGRESS.md` (the tracker) and the commits on branch `viability-fixes`. Assume no memory of earlier sessions.

## 1. Orient (always)

1. `git branch --show-current` must print `viability-fixes`. If not, stop and ask; do not switch branches over uncommitted work.
2. Read the tracker in full: ground rules, queue, A-2 spec, session log.
3. `git log --oneline master..viability-fixes` and `git status --short`. Expected noise that is never staged: docker-compose.yml, `.literature_cache/`, human_review_audit.jsonl, untracked docs/ and evaluation/ reports, kosmos.db.
4. If `$ARGUMENTS` is `status`: print the queue (ID, title, status, notes), the commits on the branch, and the next item, then stop.

## 2. Pick the item

- `$ARGUMENTS` names an ID: take that item; warn if its dependencies are not done.
- Otherwise: the first row with status `in progress`, else the first `todo`. An `in progress` row means a previous session stopped mid-item: run `git diff` to see the partial work and continue from it rather than restarting.
- `continue N`: loop through sections 2 to 6 for up to N items, stopping early on any failure or block.

## 3. Load only the context this item needs

1. The item's plan section: `grep -n '^### <ID>:' evaluation/VIABILITY_ASSESSMENT_AND_CHANGE_PLAN.md`, then read from that line to the next `^### ` or `^## ` heading. For P0-CHECK and LIVE-25 read the matching rows of plan Section 8 and the "End-of-P0 check" paragraph. For A-2 the spec is in the tracker.
2. The Section 5 conventions paragraph (the paragraph that starts "Conventions for every change") and the "Reconciliations that hold across phases" paragraph, since they govern every item.
3. Owner decisions: plan Section 10 table and the tracker ground rules.
4. docs/DEEP_ONBOARD.md is a reference, not a read-through: grep its Gotchas section (`## Gotchas`) and Change Impact Index for the files this item touches. Trust its evidence over its claims; known-wrong claims are listed in plan P3-5.
5. Read every source region the item cites before editing. Line numbers are exact at 6cfe7f6 only; files already edited on this branch have shifted, so locate by the quoted code.

## 4. Implement

1. Mark the row `in progress` in the tracker before the first edit.
2. Follow the plan text exactly, including its tests. Where the plan is wrong about the code (a quoted line is missing, a fixture does not exist, a test cannot pass as written), make the smallest correct change, and record the deviation in the row's Notes with the reason. Do not expand scope into later items.
3. Match the surrounding code: imports in stdlib, third-party, kosmos blocks; `logger = logging.getLogger(__name__)`; `datetime.now(timezone.utc)`; absolute `kosmos.*` imports.
4. Never set or write ANTHROPIC_API_KEY anywhere (code, tests, .env files).

## 5. Verify

1. Run the item's acceptance tests, then the plan's Section 8 step for the item, with `python -m pytest <paths> --no-cov -p no:cacheprovider -q`. Never bare `pytest`.
2. Run the regression set for touched areas, at minimum the test directories of every package edited (for example tests/unit/execution, tests/unit/agents, tests/unit/core). Compare against the pre-change state with `git stash` if a failure looks pre-existing, and say which failures pre-date the item.
3. Live steps run only where the tracker's ground rules authorize them. Report cost.
4. If tests fail and the fix is outside this item's scope, set the row to `blocked` with the failing test and reason, leave the work uncommitted, and stop and report.

## 6. Commit and record

1. Update the tracker: row status `done`, Notes with deviations and anything a later item must know (renamed helpers, changed line positions of later targets, defects found). Append one line to the session log.
2. Stage the edited source and test files and the tracker by name, then `git commit -m "<ID>: <imperative summary>"`. No attribution lines. Do not push.
3. Report to the owner in a few lines: what changed, the test command and its result, deviations, cost of any live call, and the next item's ID and title. Then stop, unless running `continue N`.
