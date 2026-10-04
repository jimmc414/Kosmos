---
description: Run the 4-gate ladder detached and interpret the first red gate
argument-hint: [--gate N | --accept-baseline "<cause>"]
---
# Run the ladder
1. Refuse to start while another test run is alive: `bash scripts/signals_check.sh` must not
   report `RED tests-idle`. Never run two suites at once (rule 8).
2. Launch detached: `bash scripts/verify.sh --detach` prints the run directory `<dir>`.
   Poll with real time, never a sleep-based clock: every 60 s run `cat <dir>/verify.done` (absent
   while running) and `ps -eo pid,etime,args | awk '$1==<pid>'` to confirm the runner is alive.
   Expect about 8 minutes (471 s measured s0; gate 2 alone about 7 minutes).
3. Read the result from the log, never from an exit code:
   `grep -E '^(VERIFY PASS|=== gate [0-9]+ RED)' <dir>/verify.log`.
   Green = exactly `VERIFY PASS (4 gates): compile-lint tests alembic template-run  (<s> s)`.
4. Interpret the first red gate:
   - gate 1 compile-lint: `python -m compileall -q kosmos tests scripts alembic` names the
     file; or `<dir>/lint.txt` holds a finding outside scripts/lint_baseline.txt (fix the code;
     a baseline move needs `--accept-lint-baseline "<cause>"` with the cause in the commit).
   - gate 2 tests: `<dir>/gate2.log` is the pytest output; the judge prints every `NEW RED`
     node id. Re-run that file alone under the plugin
     (`VERIFY_RUN_DIR=<dir>/adhoc PYTHONPATH=scripts python -m pytest <file> -p verify_isolate
     --no-cov -p no:cacheprovider -q`). A green isolated run never makes the ladder green:
     re-run the full ladder on the unchanged tree; the identical red on two full runs is a
     result (rule 11). "the gate is red" is never a cause to touch the baseline.
   - gate 3 alembic: `<dir>/alembic.log`; the migration or its downgrade is broken, or
     `alembic heads` is not what the scratch copy reached.
   - gate 4 template-run: `<dir>/template_run.json` lists the failed checks; "dataset missing"
     or "sandbox execution raised" is the environment (daemon, image) before it is the code.
   - "touched the owner's kosmos.db": stop; the isolation failed; record a SIGNAL# row.
5. Record the pass string with start and end times in the session block and the log row's
   Verify cell. A docs-only session does not re-run the ladder; its cell reads
   `none (docs-only; inherited <pass> from s<N> <hash>)`.
`--gate N` runs one gate for a control or a diagnosis and is never a result.
