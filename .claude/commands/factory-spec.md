---
description: Spec-freeze — write the next stage's docs/PLAN.md (or the interstitial) from the template
argument-hint: <stage>|interstitial
---
# Freeze the next stage's execution spec
1. **Read:** docs/MAP.md §2 (the target row: entry criteria, exit gate, governing docs; for an
   interstitial read §2 WHOLE and record "no row enterable" in the header) · the governing
   plan §5 items and §8 steps · docs/execution/BACKLOG.md (lanes are drawn from it by id) ·
   the plan template in docs/process/SOFTWARE_FACTORY_PROCESS.md Appendix C.
2. **Check entry criteria** for EVERY stage the plan will span, including that every
   stage-start decision is resolved. Unresolved ⇒ a one-paragraph decision brief in
   SESSION_STATE §Waiting-on-owner, and that path stops. A fired time-fused owner trigger ⇒ a
   dated ESCALATION line first in Waiting-on-owner AND first in NEXT ACTION; mark the stage
   blocked(<item>).
3. **Archive the exhausted docs/PLAN.md** by `git mv` to docs/execution/archive/
   PLAN_<stages>.md (interstitial: PLAN_interstitial-<freeze-date>.md). FIRST move (not copy)
   every still-open lane row and open-items row to BACKLOG as `open` or `in-plan`. Un-moved
   rows fail this freeze.
4. **Write the new docs/PLAN.md from the template.** Re-verify and DATE every file:line anchor
   at write time. Every milestone: a tier tag, *Verify* (exact commands + expected output),
   *DoD* (incl. the same-commit register flip). Oracle milestones name the comparison and
   define "match".
5. **High-stakes stage:** set NEXT ACTION = "Owner: skim docs/PLAN.md §Locked decisions +
   §DoD". Otherwise proceed.
6. **Run the documentation checklist** (decision-record lint A1–A7: number matches filename ·
   sections in order · valid Status · body unchanged since minting · referenced somewhere · a
   real rejected option · a downside; map changelog reference health; every file:line anchor
   dated) and leave the token `DOC-CHECK run: <scope> · <findings>` in the docs/MAP_CHANGELOG.md
   line and the plan's commit trail. No token ⇒ the freeze is incomplete and the next session
   reports it.

## DO NOT
Freeze with silent unknowns (every open question is resolved, routed to extraction, or in the
honesty register with an owner) · pre-write later stages' specs · cite a STALE source (README
until P3-5; archive/).
## DO
Make the spec executable by a session with zero prior context. Update docs/MAP.md §2 only when
the previous stage's exit evidence is in the same commit.
