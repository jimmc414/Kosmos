# MAP changelog (append-only; corrections are new rows)

| Date | Section | Change | Why | Commit |
|---|---|---|---|---|
| 2026-10-04 | all | Map created by Session 0: stages S0–S4, trust table, locked decisions D-01 to D-24; the owner signed §2 and §4 (INVENTORY_S0 §10 (b)) | Factory bootstrap (process §11.1 step 4) | dc62f8a |
| 2026-10-04 | §2, §7, §8 | Freeze: docs/PLAN.md spans S1–S4 (Lane 0 two owner triggers, Lane 1 M1–M10 in tracker order, Lane 2 F1–F2); S1 stays `next` until S0's gate flips it in B8. DOC-CHECK run: ADR-0001, ADR-0002 lint A1–A7 pass · 26 anchor files resolve, 9 line anchors spot-checked · changelog hashes resolve · findings: none | /factory-spec step 6 (process §4.5) | 2600661 |
| 2026-10-04 | §2 | Stage-gate: S0 Factory bootstrap → done; S1 P2 remainder → active. Evidence: `VERIFY PASS (4 gates): compile-lint tests alembic template-run  (486 s)` on the finished tree (03:10:56 → 03:19:02); every S0 exit conjunct in the s0 session block | S0 exit gate met (B8) | 3725d4b |
| 2026-10-04 | §10 | Correction to the freeze row above: the DOC-CHECK resolved 24 anchor files, not 26 (the count was mistyped) | accuracy | 3725d4b |
| 2026-10-08 | §2, §4 | Stage-gate: S1 P2 remainder → done; S2 P3 hardening → active; D-10 marked superseded by VIAB#P2-7. Evidence: step 14 green at s1 (13 passed), step 17 green at 50a6ee1 (2 passed, pool peak 12), `VERIFY PASS (4 gates): compile-lint tests alembic template-run  (448 s)` 09:09:03 → 09:16:31 on the landing tree. Plan §5 P2-7 (3)'s REFINING clause read as applying after the one refinement pass (docs/PLAN.md §Operational learnings, s2) | S1 exit gate met (s2) | `s2 — S1-gate:` |
