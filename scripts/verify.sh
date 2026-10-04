#!/usr/bin/env bash
# scripts/verify.sh — the Kosmos gate ladder. ONE script, ordered gates, stops at the first red,
# prints exactly one pass string:  VERIFY PASS (4 gates): compile-lint tests alembic template-run
# Read the result from the log, never from an exit code (CLAUDE.md process rules 1 and 2).
#
# Usage
#   bash scripts/verify.sh                           all gates, foreground (about 8 minutes; 471 s measured s0)
#   bash scripts/verify.sh --detach                  launch detached; prints the run dir; poll <dir>/verify.done
#   bash scripts/verify.sh --accept-baseline "<cause>"       re-run gate 2, print the delta, stamp scripts/test_baseline.txt
#   bash scripts/verify.sh --accept-lint-baseline "<cause>"  re-run the lint, print the delta, stamp scripts/lint_baseline.txt
#   bash scripts/verify.sh --gate N                  one gate only: for controls and debugging, NEVER a result
#
# Every run writes to $VERIFY_SCRATCH (default /tmp/kosmos-verify)/run_<ts>/ : verify.log (this
# output), gate2.log (pytest), ladder.db (the test DB), alembic.db (the migration copy),
# template_run.json. `latest` symlinks the newest run. The ladder NEVER opens the owner's
# kosmos.db (gate 2 runs under scripts/verify_isolate.py; gate 3 copies the file first; gates 3
# and 4 prove it by md5), makes no LLM call, opens no port and starts no service. Serial and
# uncached: a red from any other invocation of the suite is not a result.
#
# Known-RED / known-GREEN recipes (each run once in s0; results in docs/process/INVENTORY_S0.md §B1)
#   gate 1  RED-a: printf 'def f(:\n' > kosmos/zz_probe.py; bash scripts/verify.sh --gate 1   → "gate 1 RED" (compileall)
#           RED-b: printf 'x = undefined_probe\n' > kosmos/zz_probe.py; --gate 1          → "gate 1 RED" (lint not in baseline)
#           GREEN: rm kosmos/zz_probe.py; --gate 1                                         → "gate 1 OK"
#   gate 2  RED:   printf 'def test_probe():\n    assert False\n' > tests/unit/test_zz_probe.py; full run → "gate 2 RED: 1 new"
#           GREEN: rm tests/unit/test_zz_probe.py; full run                                → "VERIFY PASS (4 gates)"
#           judge-only controls (milliseconds): see scripts/verify_judge.py docstring
#   gate 3  RED:   VERIFY_ALEMBIC_SOURCE=/nonexistent bash scripts/verify.sh --gate 3 is NOT red (an empty DB
#                  migrates from nothing); the real RED is a broken newest revision: add `raise RuntimeError` at
#                  the top of upgrade() in alembic/versions/a0aa37ea19f2_*.py; --gate 3 → "gate 3 RED"; revert → OK
#   gate 4  RED:   VERIFY_TEMPLATE_CSV=/nonexistent.csv bash scripts/verify.sh --gate 4     → "gate 4 RED" (dataset missing)
#           RED-b: docker stop of the daemon, or `docker image rm kosmos-sandbox:latest`    → "gate 4 RED" (sandbox)
#           GREEN: the climate CSV with the image present; --gate 4                        → "gate 4 OK ... r=0.93"
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
ROOT=$(pwd)
PY=${PYTHON:-python}
SCRATCH_BASE=${VERIFY_SCRATCH:-/tmp/kosmos-verify}
TEST_PATHS=(tests/unit tests/integration)
PYTEST_FLAGS=(--no-cov -p no:cacheprovider -q)
LINT_SELECT=E9,F63,F7,F82
TEST_BASELINE=scripts/test_baseline.txt
LINT_BASELINE=scripts/lint_baseline.txt
CSV=${VERIFY_TEMPLATE_CSV:-evaluation/data/climate_co2_temperature_test.csv}
ALEMBIC_SOURCE=${VERIFY_ALEMBIC_SOURCE:-kosmos.db}

MODE=all; ONLY=; CAUSE=; RUN=
while [ $# -gt 0 ]; do
  case "$1" in
    --detach) MODE=detach;;
    --accept-baseline) MODE=accept; CAUSE=${2:-}; shift;;
    --accept-lint-baseline) MODE=accept-lint; CAUSE=${2:-}; shift;;
    --gate) MODE=one; ONLY=${2:-}; shift;;
    --run-dir) RUN=${2:-}; shift;;
    -h|--help) sed -n '2,32p' "$0"; exit 0;;
    *) echo "unknown argument: $1"; exit 64;;
  esac; shift
done

if [ -z "$RUN" ]; then RUN=$SCRATCH_BASE/run_$(date +%Y%m%d_%H%M%S); fi
mkdir -p "$RUN"; ln -sfn "$RUN" "$SCRATCH_BASE/latest"

if [ "$MODE" = detach ]; then
  rm -f "$RUN/verify.done"
  setsid nohup bash "$0" --run-dir "$RUN" > "$RUN/verify.out" 2>&1 &
  echo "verify.sh detached: pid $! · run dir $RUN"
  echo "poll:  cat $RUN/verify.done   (absent while running)"
  echo "read:  grep -E '^(VERIFY PASS|=== gate [0-9]+ RED)' $RUN/verify.log"
  echo "alive: ps -eo pid,etime,args | awk '\$1==$!'"
  exit 0
fi

LOG=$RUN/verify.log
exec > >(tee -a "$LOG") 2>&1
t0=$(date +%s)
trap 'rc=$?; echo "rc=$rc finished=$(date -Is) run=$RUN" > "$RUN/verify.done"' EXIT
PASS=()
gate()  { echo "=== gate $1: $2 ($(date +%T))"; }
red()   { echo "=== gate $CUR RED: $*"; exit 1; }
ok()    { echo "=== gate $CUR OK"; PASS+=("$CURNAME"); }
want()  { [ "$MODE" != one ] || [ "$ONLY" = "$1" ]; }
owner_db_md5() { [ -f kosmos.db ] && md5sum kosmos.db | cut -d' ' -f1 || echo absent; }
peer_pytest() {  # anchored on argv, never a pattern a prompt could contain (rule 3, rule 8)
  ps -eo pid,etime,args | awk -v me=$$ 'NR>1 && $1!=me && $3 ~ /python[0-9.]*$/ && $4=="-m" && $5=="pytest" {print}'
}
echo "verify.sh start $(date -Is) HEAD $(git rev-parse --short HEAD) mode=$MODE run=$RUN"

# ---------------------------------------------------------------- accept modes
if [ "$MODE" = accept-lint ]; then
  [ -n "${CAUSE// /}" ] || { echo "refusing: --accept-lint-baseline needs a non-blank cause"; exit 3; }
  ruff check kosmos --select "$LINT_SELECT" --output-format concise --quiet --no-cache --exit-zero 2>/dev/null \
    | sed -E 's/^([^:]+):[0-9]+:[0-9]+: /\1: /' | sort > "$RUN/lint.txt"
  echo "--- delta against $LINT_BASELINE (new findings first):"
  grep -vxF -f <(grep -v '^#' "$LINT_BASELINE" 2>/dev/null || true) "$RUN/lint.txt" | sed 's/^/  NEW: /' || true
  grep -vxF -f "$RUN/lint.txt" <(grep -v '^#' "$LINT_BASELINE" 2>/dev/null || true) | sed 's/^/  retired: /' || true
  { echo "# baseline accepted: $CAUSE ($(date +%F), HEAD $(git rev-parse --short HEAD))"
    echo "# ruff --select $LINT_SELECT over kosmos/, line and column stripped. Moves only through \`bash scripts/verify.sh --accept-lint-baseline \"<cause>\"\`."
    cat "$RUN/lint.txt"; } > "$LINT_BASELINE"
  echo "lint baseline written: $LINT_BASELINE ($(wc -l < "$RUN/lint.txt") findings)"; exit 0
fi

run_gate2() {
  while [ -n "$(peer_pytest)" ]; do echo "waiting: another pytest is alive (rule 8)"; peer_pytest; sleep 30; done
  local before; before=$(owner_db_md5)
  VERIFY_RUN_DIR="$RUN" PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}" \
    "$PY" -m pytest "${TEST_PATHS[@]}" -p verify_isolate "${PYTEST_FLAGS[@]}" -o log_file="$RUN/pytest_run.log" \
    > "$RUN/gate2.log" 2>&1 || true
  tail -1 "$RUN/gate2.log"
  [ "$(owner_db_md5)" = "$before" ] || { echo "gate 2 touched the owner's kosmos.db (md5 changed): refusing"; return 9; }
}

if [ "$MODE" = accept ]; then
  [ -n "${CAUSE// /}" ] || { echo "refusing: --accept-baseline needs a non-blank cause"; exit 3; }
  run_gate2 || exit 9
  "$PY" scripts/verify_judge.py "$RUN/gate2.log" "$TEST_BASELINE" --write-baseline "$CAUSE"
  exit $?
fi

# ---------------------------------------------------------------- gate 1: compile + lint
CUR=1; CURNAME=compile-lint
if want 1; then
  gate 1 "$CURNAME"
  probe=$(mktemp -d)
  printf 'def f(:\n' > "$probe/bad.py"
  if "$PY" -m compileall -q "$probe" >/dev/null 2>&1; then red "compileall cannot fail on a syntax error — refusing"; fi
  printf 'x = undefined_probe_name\n' > "$probe/lint.py"
  if ruff check --select "$LINT_SELECT" --quiet --no-cache --isolated "$probe/lint.py" >/dev/null 2>&1; then red "ruff cannot fail on an undefined name — refusing"; fi
  rm -rf "$probe"
  "$PY" -m compileall -q kosmos tests scripts alembic >/dev/null || red "compileall found a syntax error (run: python -m compileall -q kosmos tests scripts alembic)"
  ruff check kosmos --select "$LINT_SELECT" --output-format concise --quiet --no-cache --exit-zero 2>/dev/null \
    | sed -E 's/^([^:]+):[0-9]+:[0-9]+: /\1: /' | sort > "$RUN/lint.txt"
  new=$(grep -vxF -f <(grep -v '^#' "$LINT_BASELINE") "$RUN/lint.txt" || true)
  [ -z "$new" ] || red "lint findings outside $LINT_BASELINE:"$'\n'"$new"
  echo "lint: $(wc -l < "$RUN/lint.txt") findings, all in the baseline"
  ok
fi

# ---------------------------------------------------------------- gate 2: tests vs the stamped baseline
CUR=2; CURNAME=tests
if want 2; then
  gate 2 "$CURNAME (tests/unit tests/integration, serial, uncached, isolated)"
  [ -f "$TEST_BASELINE" ] || red "no $TEST_BASELINE; stamp one with --accept-baseline \"<cause>\""
  run_gate2 || red "the run touched kosmos.db"
  if ! "$PY" scripts/verify_judge.py "$RUN/gate2.log" "$TEST_BASELINE"; then red "new failures outside $TEST_BASELINE (see above; log $RUN/gate2.log)"; fi
  ok
fi

# ---------------------------------------------------------------- gate 3: alembic on a scratch copy
CUR=3; CURNAME=alembic
if want 3; then
  gate 3 "$CURNAME (upgrade head, downgrade -1, upgrade head on a copy of $ALEMBIC_SOURCE)"
  before=$(owner_db_md5)
  if [ -f "$ALEMBIC_SOURCE" ]; then cp "$ALEMBIC_SOURCE" "$RUN/alembic.db"; else : > "$RUN/alembic.db"; echo "no $ALEMBIC_SOURCE: migrating an empty database"; fi
  head=$(alembic -c alembic.ini heads 2>/dev/null | awk 'NR==1{print $1}')
  [ -n "$head" ] || red "alembic heads printed nothing"
  export DATABASE_URL="sqlite:///$RUN/alembic.db"
  alembic -c alembic.ini upgrade head   > "$RUN/alembic.log" 2>&1 || red "upgrade head failed (see $RUN/alembic.log)"
  alembic -c alembic.ini downgrade -1  >> "$RUN/alembic.log" 2>&1 || red "downgrade -1 failed (see $RUN/alembic.log)"
  alembic -c alembic.ini upgrade head  >> "$RUN/alembic.log" 2>&1 || red "re-upgrade failed (see $RUN/alembic.log)"
  unset DATABASE_URL
  cur=$(sqlite3 "$RUN/alembic.db" 'select version_num from alembic_version' 2>/dev/null || true)
  [ "$cur" = "$head" ] || red "version after upgrade is '$cur', head is $head"
  [ "$(owner_db_md5)" = "$before" ] || red "the migration touched the owner's kosmos.db"
  echo "alembic: head $head reached twice; owner kosmos.db untouched"
  ok
fi

# ---------------------------------------------------------------- gate 4: no-LLM template run through the sandbox
CUR=4; CURNAME=template-run
if want 4; then
  gate 4 "$CURNAME ($CSV through kosmos-sandbox:latest)"
  before=$(owner_db_md5)
  DATABASE_URL="sqlite:///$RUN/ladder.db" KOSMOS_ARTIFACTS_DIR="$RUN/artifacts" \
    "$PY" scripts/verify_template_run.py --csv "$CSV" --out "$RUN/template_run.json" || red "see $RUN/template_run.json"
  [ "$(owner_db_md5)" = "$before" ] || red "the template run touched the owner's kosmos.db"
  ok
fi

if [ "$MODE" = one ]; then echo "single-gate run finished (never a result)"; exit 0; fi
echo "VERIFY PASS (${#PASS[@]} gates): ${PASS[*]}  ($(( $(date +%s) - t0 )) s)"
