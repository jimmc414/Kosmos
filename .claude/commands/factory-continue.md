---
description: Resume the Kosmos viability build from where the last session stopped
argument-hint: [--status-only]
---
# Continue the build
Resume autonomous build work on the active stage of docs/MAP.md.

## Automatic actions
1. **Read, in order:** `bash scripts/signals_check.sh` FIRST (paste its line; rc 1 is never a
   green; RED = record a SIGNAL# row, never fix mid-milestone) · docs/MAP.md §2 ·
   SESSION_STATE.md (NEXT ACTION, §HOLD, blockers, Waiting-on-owner) · docs/PLAN.md (the
   milestone NEXT ACTION names: Design, Verify, DoD) · docs/execution/BACKLOG.md ONLY when NEXT
   ACTION is blocked or done. For a VIAB# milestone also read its plan §5 item text (`grep -n
   '^### <ID>:' evaluation/VIABILITY_ASSESSMENT_AND_CHANGE_PLAN.md`, read to the next `### `)
   and the frozen tracker's Notes for the items it depends on.
2. **Determine the task (first match wins):**
   a0. §HOLD ACTIVE ⇒ first unfinished hold-queue item; empty ⇒ stop, tell the owner (under
       the runner: write HALT_HOLD_EMPTY to your done file).
   a. A recorded blocker now resolved ⇒ clear it, take its milestone.
   b. NEXT ACTION's tag matches this session's tier, or it is a Lane-0 row firing by its
      trigger ⇒ take it verbatim. A JUDGMENT row met by a build session is not started and not
      decided. A BUILD row met by a judgment session is skipped for the first open JUDGMENT row.
      OWNER rows are startable by no session. An EXPLORE row is a whole session.
   c. NEXT ACTION done-but-unrecorded (git log + verify evidence) ⇒ record it, run the
      exit-gate check, advance.
   d. All PLAN.md milestones done ⇒ catch missed flips (a calendar gate may lag: record a
      blocker), archive PLAN.md → docs/execution/archive/PLAN_<stages>.md, run /factory-spec.
      An interstitial is never archived on exhaustion.
   e. Blocked on the owner or mismatched by tag ⇒ never idle: first open lane row for this
      tier in printed order, then BACKLOG by routing; say so in the log row.
   f. Nothing above applies ⇒ stop with a named reason (under the runner: HALT_PLAN_COMPLETE
      or HALT_NEEDS_OWNER in your done file).
3. **Sanity-check anchors:** re-verify every file:line the milestone cites; read the printed
   line against its claim. Plan §5 line numbers are exact at 6cfe7f6 only: locate by the quoted
   code. Mismatch ⇒ fix the spec per CLAUDE.md §AMENDMENT, proceed.
4. **Work exactly ONE milestone.** Then: ladder green (`bash scripts/verify.sh --detach`, poll
   the marker, read `VERIFY PASS (4 gates)` from the log) → adversarial pass (the self-run
   three-lens floor: correctness, spec fidelity, test efficacy; agents additive at zero
   wall-clock; record the tally) → commit by explicit paths with the register flips and doc
   fixes in the same commit, subject `s<N> — <ID>: <summary>` → SESSION_STATE.md (rewrite NEXT
   ACTION; block; log row below the separator; `bash scripts/hot_files_check.sh`;
   `grep -c '^## NEXT ACTION'` = 1) → exit-gate check of the milestone's own stage; if met, the
   stage-gate commit in THIS session → `git push -u origin viability-fixes`. A second small
   milestone MAY follow; never a third.

## PARK, DON'T DECIDE
A build session that meets a judgment-grade decision its task did not pre-decide records it in
SESSION_STATE §Waiting-on-judgment (item · why · what it blocks) and takes another row or
wraps. A judgment session PRE-DECIDES such items into NEXT ACTION; its report never says
"escalate X next session".

## Options
`--status-only`: report the active stage, milestone progress, blockers, waiting-on-owner, and
the last verify result (`/tmp/kosmos-verify/latest/verify.log`). Change nothing. (This replaces
the old `/next-plan-item status`.)

## LOOP MODE (`/loop /factory-continue`)
Each firing = one full pass. Stop (no re-arm) when the phase is done, when context is near
500K tokens (clean wrap by 600K), or when the hold queue is empty. Never schedule idle wakeups.

## DO NOT
Ask permission to continue · re-plan the stage · re-open locked decisions (docs/MAP.md §4) ·
re-summarize the project · start milestone N+1 while N is red · touch the STOP-AND-ASK list
without the owner · run tests outside the `-p verify_isolate` plugin (they write kosmos.db).

## END-OF-SESSION REPORT (mandatory, last in the final message)
1. **Needs owner** — one line each, new first, standing ones with escalate-by; or "None".
2. **Next session** — NEXT ACTION in one line + alternates if blocked.
3. **Tier + effort** — explicit ("BUILD, high"); never "BUILD but escalate X".
