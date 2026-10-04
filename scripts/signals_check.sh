#!/usr/bin/env bash
# scripts/signals_check.sh — read-only, ≤60 s. The definition is docs/STANDING_SIGNALS.md §2;
# this script implements it and changes nothing. rc 0 = all OK · 2 = a RED · 1 = could not look
# (NEVER a green). Prints ONE line: counts and classes, never identifiers.
set -u
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
red=(); unable=(); looked=0
check() { local name=$1; shift; local out; if out=$("$@" 2>&1); then looked=$((looked+1)); case "$out" in RED*) red+=("$name");; esac; else unable+=("$name"); fi; }

check branch        bash -c '[ "$(git branch --show-current)" = viability-fixes ] && echo OK || echo RED'
check tests-idle    bash -c 'n=$(ps -eo pid,args | awk -v me=$PPID '"'"'NR>1 && $1!=me && (($2 ~ /python[0-9.]*$/ && $3=="-m" && $4=="pytest") || ($2=="bash" && $3=="scripts/verify.sh")) {c++} END{print c+0}'"'"'); [ "$n" -eq 0 ] && echo OK || echo "RED $n"'
check docker        bash -c 'timeout 20 docker info >/dev/null 2>&1 && echo OK || echo RED'
check sandbox-image bash -c 'timeout 20 docker image inspect kosmos-sandbox:latest >/dev/null 2>&1 && echo OK || echo RED'
check services      bash -c 'up=$(timeout 20 docker ps --format "{{.Names}}" 2>/dev/null | grep -cE "^kosmos-(postgres|redis|neo4j)$"); [ "$up" -eq 3 ] && echo OK || echo "RED $up/3"'
check disk          bash -c 'a=$(df -BG --output=avail . | tail -1 | tr -dc 0-9); b=$(df -BG --output=avail /tmp | tail -1 | tr -dc 0-9); [ "${a:-0}" -ge 10 ] && [ "${b:-0}" -ge 5 ] && echo OK || echo "RED ${a}G/${b}G"'
check kosmos-db     bash -c '[ -f kosmos.db ] || { echo RED; exit 0; }; timeout 10 sqlite3 kosmos.db "begin immediate; rollback;" >/dev/null 2>&1 && echo OK || echo "RED locked"'
tree=$(git status --porcelain 2>/dev/null | awk '{if($1=="??")u++; else m++} END{printf "%dM/%dU", m+0, u+0}') || tree="?"

if [ ${#unable[@]} -gt 0 ]; then echo "signals: COULD-NOT-LOOK ${unable[*]} · ${looked} looked · tree $tree"; exit 1; fi
if [ ${#red[@]} -gt 0 ];    then echo "signals: RED ${red[*]} · ${looked} checks · tree $tree"; exit 2; fi
echo "signals: ${looked} OK · tree $tree"; exit 0
