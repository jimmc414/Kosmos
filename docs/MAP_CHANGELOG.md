# MAP changelog (append-only; corrections are new rows)

| Date | Section | Change | Why | Commit |
|---|---|---|---|---|
| 2026-10-04 | all | Map created by Session 0: stages S0–S4, trust table, locked decisions D-01 to D-24; the owner signed §2 and §4 (INVENTORY_S0 §10 (b)) | Factory bootstrap (process §11.1 step 4) | (hash: next session) |
| 2026-10-04 | §2, §7, §8 | Freeze: docs/PLAN.md spans S1–S4 (Lane 0 two owner triggers, Lane 1 M1–M10 in tracker order, Lane 2 F1–F2); S1 stays `next` until S0's gate flips it in B8. DOC-CHECK run: ADR-0001, ADR-0002 lint A1–A7 pass · 26 anchor files resolve, 9 line anchors spot-checked · changelog hashes resolve · findings: none | /factory-spec step 6 (process §4.5) | (hash: next session) |
