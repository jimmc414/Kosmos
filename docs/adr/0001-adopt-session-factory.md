# ADR-0001: Adopt the Session Factory process for the remaining viability work
Status: accepted
Date: 2026-10-04 (s0)

## Context
Twenty of thirty tracker rows of the viability plan landed between 2026-10-02 and 2026-10-04 under
a simpler setup: a tracker file (evaluation/VIABILITY_PROGRESS.md), a resume skill
(/next-plan-item), one commit per item after its acceptance tests, and the owner relaunching a
session after every context clear. Ten rows remain (P2-4 … LIVE-25). The owner wants the rest to
run under an unattended queue. The existing setup has no single scripted gate (each item ran only
its own tests, with a hand-made master worktree diff to prove no regression), no stage table with
exit gates, no register that outlives the plan, no health probe, and a record discipline that
depends on the owner reading a 30 KB tracker. The process document
(docs/process/SOFTWARE_FACTORY_PROCESS.md) describes a factory derived from about 490 sessions of
a comparable build, with the mapping for exactly this kind of plan-and-tracker setup (§11.2).

## Options considered
1. Keep the tracker and the skill, add a cron-driven loop that reruns /next-plan-item — rejected
   because the skill stops and asks on every block, has no ladder and no halt classes, so an
   unattended loop would either idle or commit red work; and the tracker's Notes cells (several
   KB each) are the only record, which a cold session reads partially.
2. Write a bespoke lighter process (a ladder plus the tracker) — rejected because every rule in
   the process document has a recorded failure behind it; re-deriving the subset we need would
   repeat those failures one at a time, and the owner asked for this process by name.
3. Adopt the Session Factory process as written, mapping the tracker per §11.2, freezing the
   tracker as the record of the first twenty rows — chosen.

## Decision
We will run the remaining viability work under the Session Factory process: CLAUDE.md is the
contract, docs/MAP.md the map, docs/PLAN.md the frozen plan, SESSION_STATE.md the ledger,
docs/execution/BACKLOG.md the register, scripts/verify.sh the ladder, scripts/signals_check.sh
the probe, /factory-continue the resume command; the tracker is frozen and
/next-plan-item is a pointer. Sessions 1–3 run attended on the build tier; Session 4 builds the
runner; the queue then runs until HALT_PLAN_COMPLETE or HALT_NEEDS_OWNER.

## Consequences
- Every milestone is proven by one script with one pass string, committed by explicit path, and
  recorded so that a cold session can continue; the owner is asked only for the seven
  STOP-AND-ASK items.
- Bookkeeping cost: the process document measures about 31 % record-only commits and a 1.2 : 1
  bookkeeping-to-code byte ratio; each milestone also pays the full ladder (about 9 minutes).
- The factory's own files (about 15 of them) are new surface a session can get wrong; the
  hot-files check and the structural assertions exist for that reason.

## Enforcement
CLAUDE.md §ON SESSION START/END; the deny-list in .claude/settings.json; `bash scripts/verify.sh`
before every commit; `bash scripts/hot_files_check.sh` before every record commit.
