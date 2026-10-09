# ADR-0002: The test gate judges against a stamped baseline of pre-existing failures until P3-3
Status: accepted
Date: 2026-10-04 (s0)

## Context
The unit and integration suites are not green: on the Session 0 tree (code identical to
viability-fixes at 1247379) `python -m pytest tests/unit tests/integration --no-cov
-p no:cacheprovider -q` reports 285 failed, 2,353 passed, 212 skipped and 107 errors in 7 min
27 s. The tracker records the failures as pre-existing on master at 73f4d2a (98 failed + 62
errors in four directories alone; the rest are in modules P3-4 archives or P3-3 repairs), plus
two order-dependent caplog tests. Plan item P3-3 ("test suite green for surviving modules") is
the sixth of the ten remaining items, after P3-1 and P3-4. The ladder needs a test gate that is
red for a NEW failure and green otherwise, from Session 1 on, without waiting for P3-3.

## Options considered
1. Gate 2 = bare `exit 0` of the full suite — rejected because it is red on every tree until
   P3-3 lands, so no milestone before it could commit behind the ladder, or sessions would
   learn to ignore gate 2.
2. Gate 2 restricted to the test files each milestone names (the old per-item practice) —
   rejected because it cannot catch a regression outside the named files, which is exactly what
   the master-worktree diff was hand-made to catch.
3. Gate 2 judges the full suite against a stamped baseline of red node ids, accepted once with a
   written cause, moved only by an explicit accept command that refuses a blank cause and prints
   the delta first; retired when P3-3 lands — chosen.
4. An auto-accepting ratchet (any run with fewer reds rewrites the baseline) — rejected because
   the triage is the point: a test that flips from red to green without a cause is as suspicious
   as the reverse (an order-dependent test, a skipped import).

## Decision
We will keep scripts/test_baseline.txt, a sorted list of `STATUS node-id` lines headed by
`# baseline accepted: <cause> (<date>, HEAD <hash>)`, stamped by
`bash scripts/verify.sh --accept-baseline "<cause>"`. Gate 2 runs the full unit and integration
suites serially, uncached, isolated from kosmos.db, and is red when any red node id is not in
the baseline; node ids in the baseline that now pass are printed as "retired" and are not a red.
The cause of every re-stamp is written in the same commit. When VIAB#P3-3 lands, the same commit
empties the baseline (the stamp line reads "retired") and gate 2 becomes a bare exit 0
(FACTORY#retire-test-baseline).

## Consequences
- Milestones can land behind a meaningful regression gate from Session 1.
- A test that is in the baseline and that a milestone breaks WORSE (a different assertion, or a
  new error in the same node) is invisible to the gate; the self-review's test-efficacy lens
  must read the "retired" and summary lines, and P3-3 is the deadline.
- The baseline file holds 392 node ids and ties the suite to the three Docker containers that
  were up when it was stamped (postgres, redis, neo4j); the signals script reports them.

## Enforcement
scripts/verify.sh gate 2 and scripts/verify_judge.py (known-RED/GREEN controls in their
docstrings); CLAUDE.md §STANDING TRAPS ("the gate is red" is never a cause).

## Correction note (2026-10-09, s6)
Retired 2026-10-09 by VIAB#P3-3 (commit `s6 — P3-3: …`): every one of the 394 stamped ids was fixed or rewritten against the real API, and gate 2 ran with 0 red ids (run_20261009_022750 and the re-stamp run). scripts/test_baseline.txt keeps its stamp lines and holds 0 node ids, so the judge's "no red id outside the baseline" is now a plain "no red id". The unit and integration suites are hermetic since the same commit (tests/conftest.py: no .env, no shell credentials, temp paths), which changes what a red means: it can no longer come from the owner's environment.
