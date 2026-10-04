# Standing signals — the session-start health probe

This document is the definition; scripts/signals_check.sh implements it. When they disagree
the script has the bug. The script is read-only, finishes in under 60 s, prints ONE line of
counts and classes (never identifiers), and exits 0 = all OK · 2 = a RED · 1 = could not look,
which is never a green. A session pastes the line verbatim into its block (CLAUDE.md step 0).
On RED or 1 a session records a `SIGNAL#` register row with an escalate-by and fixes nothing
mid-milestone.

## §1 What Kosmos has, and has not
Kosmos has no scheduled job of its own (the crontab entries on this machine belong to another
project), no server to probe, no backups to age, and no push channel. The checks below are
therefore the environment the ladder and `kosmos run` need. FACTORY#runner adds a push on RED
when the queue exists; until then the line in the session block is the reader.

## §2 The checks (name · what is read · OK · RED)
| # | Name | Reads | OK when | RED when |
|---|---|---|---|---|
| 1 | branch | `git branch --show-current` | it prints `viability-fixes` | anything else (a session on master or a detached HEAD must stop) |
| 2 | tests-idle | `ps -eo pid,args`, anchored on argv: `python -m pytest …` or `bash scripts/verify.sh …`, excluding the probe itself | no such process | one is alive (never run two suites at once, rule 8); the line names the count only |
| 3 | docker | `docker info` | the daemon answers | it does not (gate 4 and `kosmos run` need it) |
| 4 | sandbox-image | `docker image inspect kosmos-sandbox:latest` | the image exists | missing (rebuild: `docker build -t kosmos-sandbox:latest docker/sandbox`, MAP D-09) |
| 5 | services | `docker ps` for kosmos-postgres, kosmos-redis, kosmos-neo4j | all three running | any is not: the gate-2 baseline was stamped with them up (ADR-0002), so a red test may be the environment |
| 6 | disk | `df` on the repo and on /tmp | ≥ 10 GB free on the repo's drive and ≥ 5 GB on /tmp | less |
| 7 | kosmos-db | `kosmos.db` exists; `sqlite3 kosmos.db 'begin immediate; rollback'` | present and not locked | missing, or the lock is held (a `kosmos run` or a stray process is writing it) |
| 8 | tree | `git status --porcelain` counts | informational: tracked-modified and untracked counts are printed; the owner's six known-dirty paths are expected | never RED on its own; a count far above the expected (6 tracked, ~20 untracked at s0) is worth a look |

## §3 Output
`signals: 7 OK · tree 7M/28U` (checks 1–7 counted; the tree reading is appended) or `signals: RED <names> · 7 checks · tree …` or
`signals: COULD-NOT-LOOK <names> · <n> looked`. The exit code follows the process rule above.

## §4 Changelog
- 2026-10-04 s0: created with checks 1–7 plus the tree reading (8). First run: `signals: RED tests-idle · 7 checks · tree 7M/28U` while the baseline run was alive; 5.9 s.
