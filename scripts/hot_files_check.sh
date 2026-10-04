#!/usr/bin/env bash
# scripts/hot_files_check.sh — every canonical file must stay readable in one tool call.
# Lines (bytes), all under the 262,144 B Read ceiling measured in s0 (INVENTORY_S0 §7):
#   SESSION_STATE.md 200,000 B and at most 12 session blocks · docs/MAP.md, docs/PLAN.md and
#   each docs/execution/BACKLOG.md + backlog/*.md file 225,000 B.
# Exit 0 fine · 1 a line is crossed (roll the ledger with scripts/session_state_roll.py --write
# or split a register file BEFORE the commit) · 2 the ledger's structure is broken (the
# message names the repair). Read-only.
set -u
cd "$(git rev-parse --show-toplevel)"
rc=0
line() { local f=$1 cap=$2; [ -f "$f" ] || { echo "hot: $f (absent)"; return; }; local n; n=$(stat -c %s "$f"); if [ "$n" -gt "$cap" ]; then echo "hot: $f $n B > $cap B  ← OVER THE LINE"; rc=1; else echo "hot: $f $n B / $cap B"; fi; }
line SESSION_STATE.md 200000
line docs/MAP.md 225000
line docs/PLAN.md 225000
line docs/execution/BACKLOG.md 225000
for f in docs/execution/backlog/*.md; do [ -f "$f" ] && line "$f" 225000; done
if [ -f SESSION_STATE.md ]; then
  blocks=$(grep -c '^<!-- session s[0-9]\+ -->' SESSION_STATE.md)
  if [ "$blocks" -gt 12 ]; then echo "hot: SESSION_STATE.md holds $blocks session blocks > 12  ← ROLL"; rc=1; else echo "hot: SESSION_STATE.md holds $blocks session blocks / 12"; fi
  for h in '^## NEXT ACTION' '^## Waiting-on-owner' '^## Waiting-on-judgment' '^## HOLD' '^## STANDING' '^## Recent sessions' '^## Session log' '^## Fresh traps'; do
    c=$(grep -c "$h" SESSION_STATE.md); [ "$c" -eq 1 ] || { echo "structure: '$h' appears $c times (want exactly 1): repair the header line"; rc=2; }
  done
  sep=$(awk '/^## Session log/{f=1} f && /^\|---\|---\|---\|---\|$/{print NR; exit}' SESSION_STATE.md)
  [ -n "$sep" ] || { echo "structure: no '|---|---|---|---|' separator under '## Session log': restore the table header"; rc=2; }
fi
[ $rc -eq 0 ] && echo "hot_files_check: OK" || echo "hot_files_check: rc=$rc"
exit $rc
