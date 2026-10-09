# BACKLOG — the durable open-items register
How-to: (1) every row has a stable id `<SOURCE>#<slug>`; (2) schema: id · essence · source
(file+anchor) · blocks · routing·tier · size · deps · escalate-by · status; (3) status:
open → in-plan <lane> → done <commit/session> | refuted <file:line why> | merged → <id>; never
deleted or renumbered; (4) OWNER and DEC rows mirror SESSION_STATE §Waiting-on-owner under the
same id; (5) a row you touch in a commit is part of that commit; (6) routing ∈ BUILD ·
JUDGMENT · EXPLORE · OWNER · CALENDAR; (7) files split at 225,000 B: closed rows → *-CLOSED.md,
in-plan rows → *-inplan.md, then by id-prefix subject, then a numbered successor.
Sources: `VIAB#<item>` (the viability plan §5 item), `FACTORY#` (the process installation),
`OWNER#`, `DEC#` (an open decision), `TEST#`, `HYG#` (hygiene), `OPS#`, `SIGNAL#`.
Count: `cd docs/execution && cat backlog/*.md BACKLOG.md | grep -c '^- \*\*'` (tree must match HEAD).

## Where the rows are
| File | Holds |
|---|---|
| backlog/BUILD-inplan.md | rows in the frozen plan's lanes, in lane order (any tier; the tag is on the row) |
| backlog/BUILD-open.md | open build-tier rows not in the plan |
| backlog/JUDGMENT-open.md | open judgment-tier rows not in the plan |
| backlog/OWNER-open.md | open owner asks and open decisions (escalate-by on every row) |
| backlog/OWNER-ruled.md | ruled asks, verbatim ruling + `put:` clause |
| backlog/EXPLORE.md | whole-session investigations |
| backlog/CLOSED.md | done / refuted / merged |

## §OWNER router
| id | ask | escalate-by | status |
|---|---|---|---|
| OWNER#compose-edit-permission | make M5b's docker-compose.yml edit by hand (the session's write was refused by the permission classifier), or allow it | S2 exit | ruled 2026-10-09 (backlog/OWNER-ruled.md) |
| OWNER#compose-edit-approval | approve P3-2's docker-compose.yml edit and staging, or commit the hardening first | S2 start (the session reaching VIAB#P3-2) | ruled 2026-10-08 (backlog/OWNER-ruled.md) |
| OWNER#live-spend-authorization | confirm at launch time that LIVE-25 may spend (DeepSeek, `--budget 1`) | S4 start | open |
| OWNER#live-report-signature | read and sign the LIVE-25 `kosmos report` | S4 | open |
| DEC#tier-c-archive | archive Tier C (kosmos/domains, domain templates) or keep (MAP D-04 revisit) | after S4 | open |
| DEC#scholar-eval-gate | calibrate ScholarEval on live runs; gate or stay advisory (MAP D-07 revisit) | after S4 | open |
| OWNER#literature-cache-untrack | `.literature_cache/*.pkl` are tracked despite .gitignore:146 and dirty in the tree: `git rm --cached` them or keep tracking | S4 (any pack sitting) | open |
| OWNER#stop-and-ask-list · OWNER#map-signoff · OWNER#runner-permission-mode · OWNER#s0-branch | Phase A rulings | — | ruled 2026-10-04 (backlog/OWNER-ruled.md) |
