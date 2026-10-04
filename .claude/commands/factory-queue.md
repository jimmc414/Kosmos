---
description: Runner control for ~/kosmos-queue/ — pre-flight, launch, status, review, dry-run, stop
argument-hint: [status | review [run] | dry-run | stop | launch]
---
# The queue
The runner lives OUTSIDE the repo at ~/kosmos-queue/ (its own git repo; process Appendix L).
It is built by FACTORY#runner (Session 4, Fable high, owner present) after three attended
`/factory-continue` passes are recorded green in SESSION_STATE §Session log. Until that
directory exists, every subcommand reports "runner not built; see FACTORY#runner" and changes
nothing.

## Pre-flight (process §7.8; first matching rule wins; a blocking rule from either summary blocks)
```bash
cd ~/kosmos-queue
ls -dt runs/*_serial 2>/dev/null | head -1; ls -dt runs/*_burn 2>/dev/null | head -1
for l in driver.lock burn.lock; do [ -f "$l" ] && echo "$l: pid $(cat $l) $(kill -0 $(cat $l) 2>/dev/null && echo LIVE || echo stale)"; done
ls STOP 2>/dev/null
bash /mnt/c/python/kosmos/scripts/signals_check.sh
```
1. A lock is LIVE → a driver is running; report "already running".
2. No prior runs → first run; launch.
3. A run dir with a driver log but no SUMMARY.md and no live lock → the driver died hard; do
   not launch; show the log tail; the owner decides.
4. SUMMARY.md attention has a HALT, VIOLATED or HALT_* line → do not launch. A dirty-class
   halt needs a supervised session; a config-class halt (model, cwd, secrets) needs a queue fix.
5. Outcome REFUSED or UNKNOWN → surface it; STOP_HELD and LOCK_HELD are safe after checking
   locks; DIRTY_TIMEOUT means find out what was live; UNKNOWN is rule 3.
6. Failed rows, hold proposals, unfolded rows, a stopped run → do not block; one-line
   recommendation each.
7. Attention empty and signals rc 0 → launch, and say so.

## Launch
`tmux new-window -t kosmos-queue -n "driver-$(date +%H%M)" 'bash ~/kosmos-queue/driver.sh; ec=$?; echo "[driver exited rc=$ec]"; exec sleep infinity'`
Report the run directory, the attach hint (`tmux attach -t kosmos-queue`) and the previous
run's outcome line; return immediately. Never wait on or poll the queue from the launching
session.

## Subcommands
`status` · `review [run]` · `dry-run` run the driver's zero-launch subcommands inline
(`bash ~/kosmos-queue/driver.sh --status | --review | --dry-run`). `stop` touches
~/kosmos-queue/STOP (the driver finishes its current row). Rows and prompts are edited only
right after a LAUNCH line in the driver log (process §7.13).
