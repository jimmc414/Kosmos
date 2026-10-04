---
name: next-plan-item
description: Superseded by /factory-continue on 2026-10-04 (Session 0). The tracker evaluation/VIABILITY_PROGRESS.md is frozen. Use /factory-continue to resume the viability build, and /factory-continue --status-only for the old "status". Kept as a pointer; do not delete.
argument-hint: "(superseded; use /factory-continue)"
---

# next-plan-item — superseded

Superseded by `/factory-continue` on 2026-10-04 (Session 0, ADR-0001). The tracker
`evaluation/VIABILITY_PROGRESS.md` is frozen: it is the record of P0 through P2-3 and the A-2
spec and is not edited again. State now lives in SESSION_STATE.md (the ledger), docs/PLAN.md
(the frozen plan), docs/MAP.md (the map) and docs/execution/BACKLOG.md (the register).

- Resume the build: `/factory-continue`
- The old `status`: `/factory-continue --status-only`
- An item by id: there is no argument; the plan's NEXT ACTION names the next milestone, and a
  session takes the first open lane row for its tier when that one is blocked.
