# The Session Factory

## A process specification for building a large spec to completion across many autonomous Claude Code sessions

Version 1.0 · 2026-10-04 · Project-neutral. Derived from the Genesis build (about 490 sessions between 2026-07 and 2026-10, a Go/PostgreSQL/React rewrite of a 40-year legacy system), with the project-specific content removed and the mechanisms kept. Every MUST and NEVER in this document has a recorded failure behind it in that build; where the failure is instructive it is quoted in §9.

---

## 0. How to use this document

**Who this is for.** Two readers. First, a Claude Code session in a project that already has a large specification, asked to set the factory up (Session 0, §11) and then to run it. Second, the project's owner, who decides the short list of things only a human decides (§5.12), signs stage gates, and relaunches the runner.

**What it assumes.** A git repository. A large spec, or the material for one. A command that builds and tests the project. Claude Code CLI on one machine, with at least one model available and preferably two tiers (§8). tmux if sessions will run unattended. Shared mutable state such as one development database or a fixed port is allowed and is handled explicitly (§6.2, §7.11).

**What it delivers.** A build that advances one validated unit of work per session, records itself so that every session can start cold, and can be driven by a queue that launches sessions until the plan is complete or a human decision is needed.

**Reading paths.**

| Reader | Read |
|---|---|
| Owner | §1, §2, §5.12, §7.8, §7.13, §12 |
| Session 0 (bootstrap) | §0 to §4, §11, Appendices A to K |
| Build sessions | Appendix J is the operative text; §5 is its rationale |
| Runner implementer | §7 and Appendix L |

**Normative language.** MUST and NEVER mark rules with a recorded failure behind them. SHOULD marks the default; deviate with a written reason in the record. MAY marks a knob.

**Vocabulary.** This document uses generic names. The second column gives the Genesis name for readers who know the source project.

| This document | Genesis name | Purpose |
|---|---|---|
| Contract | `CLAUDE.md` | The per-session protocol, loaded every session |
| Map | `genesis/docs/BLUEPRINT.md` | Stage table, locked decisions, trust table, amendment protocol |
| Spec | root `SPEC_*.md`, `concepts/` | What to build |
| Plan | `genesis/docs/PLAN.md` | The frozen execution spec for the active stage or stages |
| Ledger | `SESSION_STATE.md` | The one mutable session record |
| Register | `execution/BACKLOG.md` + `backlog/*.md` | The durable open-items register, keyed by id |
| Decision records | `docs/adr/` | Immutable numbered records |
| Ladder | `scripts/verify.sh` | The gate ladder: one script, ordered gates, one pass string |
| Signals | `scripts/signals_check.sh` | Read-only health probe at session start |
| Runner | `~/genesis-orchestrator/driver.sh` | The session queue, outside the repo |
| Results ledger | `ORCHESTRATOR_RESULTS.md` | The runner's one tracked file, at the repo root |
| `/factory-continue` | `/genesis-continue` | The resume command |
| `/factory-spec` | `/genesis-spec` | The freeze command |
| `/factory-queue` | `/genesis-orchestrator` | Runner control |
| Judgment tier, Build tier | Fable, Opus | The two model tiers |
| Owner | Jim | The human |

---

## 1. The model in one page

**The problem.** A build that takes months is executed by sessions that each last one context window and remember nothing of the others. Left alone, such sessions re-plan, redo finished work, commit half-built changes, drift from the spec, and stop to ask questions nobody is present to answer. The factory is the set of files, scripts and rules that make a cold session productive within minutes and safe to run unattended.

**The shape.** Every session runs the same loop:

```
orient → select → build ONE unit → validate → review → commit → record → navigate → report
```

The repository is the only memory. State lives in four tracked files with fixed structure (map, plan, ledger, register) plus git history. A session's first act is a read-only health probe followed by a fixed read order; its last act is the record. Between them it does exactly one plan milestone, proves it with a scripted gate ladder, and commits only if the ladder is green.

**The runner adds one thing.** Because every session selects its own task from the repo state, an unattended queue does not need to carry tasks. It carries slots (tier, effort, mode), launches a fresh session into each slot with the same continue command, judges the session by a declared deliverable rather than its exit code, and halts when a session says it must stop. The repo is the planner; the runner is a scheduler.

**The ten laws.** Everything else in this document is a consequence of these.

1. **The repository is the only memory.** Nothing a session knows survives unless it is written to a tracked file or a commit. Memory kept outside the repo holds lessons and methods, never status (§10).
2. **Every session starts cold and ends with the record.** A fixed read order comes first; the ledger update comes last and is part of the work, not cleanup. An unrecorded session is a lost session.
3. **One unit of work per session.** A milestone is sized to one context window. No drive-by changes: an out-of-scope defect becomes a register row, not an edit.
4. **Commits carry only complete, green work, staged by explicit path.** Partial or red work is never committed. A session that must stop mid-unit leaves the dirty tree as the hand-off, with a resume pointer in the ledger.
5. **The plan is frozen.** It changes only through the freeze procedure or a dated addendum. Locked decisions are not re-argued in build sessions; re-arguing one is drift.
6. **Validation is scripted and evidence-based.** One ladder, one terminal pass string, read from the log and never from an exit code. Baselines and goldens move only on a recorded cause. Stage exit gates are checkable conjunctions of evidence.
7. **Review binds to blast radius, not to who did the work.** The author's self-review is the floor. Independent review agents are additive, never on the critical path, and an agent's silence is unknown, never clean.
8. **Judgment and build are routed by tag.** A build-tier session parks judgment-grade calls instead of making them. A judgment-tier session pre-decides them so that its successors have none to make.
9. **The owner decides a short, explicit list.** Everything else proceeds without asking. The owner's override always wins and is recorded. A blocked session never idles and never invents scope.
10. **Rules are earned.** A rule is born in the incubator when something bites, graduates to the contract on the second bite, and keeps a stable id forever. Reality outranks documents through a protocol that fixes the document in the same commit as the code.

---

## 2. Roles and tiers

| Role | Who | Does | Never does |
|---|---|---|---|
| Owner | The human | Rules on the STOP-AND-ASK list, signs stage gates, declares and lifts holds, packs and relaunches the runner | Is asked "should I continue?" |
| Judgment session | Judgment-tier model | Spec freezes, stage-gate flips, rulings, stop-the-line triage, milestones whose failure mode is a wrong decision | Burns itself on turnkey work |
| Build session | Build-tier model | Pre-decided milestones, intakes, digests, template authoring, with the mandatory review backstop | Decides a judgment-grade item (it parks it) |
| Reviewer agents | Spawned sub-agents, cheaper tier for refutation | The adversarial pass on a diff or a canon rewrite | Hold up a commit |
| Runner | A shell driver outside the repo | Launches sessions from a queue, detects their end by a wrap gate, reconciles, halts | Edits the repo (its one tracked file is its results ledger) |
| Supervisor | The session or script that launched the runner | Watches events, nudges a stalled row once, fixes queue cells, relaunches after a config halt | Kills a live row, resolves a dirty halt, edits tracked files |
| Signals | A read-only script and the scheduled jobs it reads | Reports health at session start | Fixes anything |

**The allocation stance.** A buggy implementation is caught by the ladder, the goldens and the review backstop. A wrong decision is caught by nothing. Spend the judgment tier where the failure mode is a wrong decision, and the build tier plus the backstop everywhere else. Never recommend the judgment tier for everything "to be safe": exhausting it on turnkey work risks having none left for the decision that needed it.

**PARK, DON'T DECIDE.** There is no mid-session path from a build session to a judgment session. When a build session meets a decision its task did not pre-decide and that is of locked-decision grade (money or domain semantics, a schema shape, a security or privacy posture, an external contract, a conflict with a locked decision), it records the item in the ledger's Waiting-on-judgment section as `item · why it is judgment-grade · what it blocks`, then takes another row or wraps. It does not decide.

**PRE-DECIDE.** The corollary binds the judgment tier. A judgment session's final report never says "escalate X next session." It decides X and writes the decision into NEXT ACTION, so the next session has zero judgment content. It recommends the judgment tier for a whole session only when the judgment content is too large to pre-decide.

---

## 3. The artifact set

### 3.1 The files

| File | Written by | Read when | Size ceiling | Roll or split |
|---|---|---|---|---|
| Contract (`CLAUDE.md`) | Owner, and the graduation step of §9 | Every session, automatically | Keep short; rules only | Never rolls; graduates rules in, retires them to an archive |
| Map (`docs/MAP.md`) | Stage-gate commits and freezes only | Session start, step 1 | Below the tool read ceiling | Changelog table lives in its own file |
| Plan (`docs/PLAN.md`) | The freeze; append-only commit trail and learnings during execution | Session start, step 4 | Below the ceiling | Archived whole when exhausted |
| Ledger (`SESSION_STATE.md`) | Every session, at the end | Session start, step 2 | 200,000 B or 12 session blocks | Oldest blocks roll to frozen archives |
| Register (`docs/execution/BACKLOG.md` + `backlog/*.md`) | Any session, in the same commit as the work | Only when NEXT ACTION is blocked or done, or by id | 225,000 B per file | Split by status, then by subject |
| Decision records (`docs/adr/NNNN-*.md`) | The session that resolves the decision | By reference | One decision each | Never edited; superseded |
| Ladder (`scripts/verify.sh`) | Rarely; adding a gate is a milestone | Every milestone | n/a | n/a |
| Signals (`scripts/signals_check.sh`) + its definition doc | Rarely | Every session, step 0 | Runs in under 60 s | n/a |
| Results ledger (`ORCHESTRATOR_RESULTS.md`) | The runner, committed by explicit path | By the review digest | Below the ceiling | Roll like the ledger |
| Queue (outside the repo) | The runner and the owner | By the runner | n/a | Per-run directories |
| Memory (outside the repo) | Any session | At recall | Index below the ceiling | Pruned on the owner's approval |

**The altitude stack.** Strategy documents (why) sit above the map (what, in what order), which sits above the plan (how, this stage), which sits above the ledger (where we are today). A session reads top-down to orient and writes bottom-up to record. Stage status lives in the map and nowhere else; the ledger points at it.

### 3.2 The read ceiling

Every tool that reads a file has a ceiling (the Genesis harness truncates at 262,144 bytes). A canonical file that cannot be read in one call is a file the next session will read partially and act on wrongly. Every hot file therefore has a byte ceiling below the tool's, a script that checks it (`hot_files_check.sh`, Appendix I), and a roll or split procedure that runs before the commit that would cross the line. The Genesis ledger reached 2.29 MB and 138 session blocks before this rule existed; it was unreadable for weeks without anyone noticing.

### 3.3 The trust ladder

The map carries a trust table over every source document: CANON (build from it), MIXED (cite only the parts the table names), STALE (never a build input), SNAPSHOT (true at a date; re-verify). A build milestone may cite only CANON sources. Citing a STALE source is a spec defect, fixed in the spec, not argued in the session.

### 3.4 Line-start anchoring

The ledger's header comment quotes the names of the ledger's own sections to explain the retention rule. A substring search for a header therefore lands inside the comment, and an edit anchored there renders as nothing. Every edit to a structured file MUST anchor on the line-start form of the marker (`^## NEXT ACTION`, the `|---|` separator of a specific table) and MUST be followed by an assertion: `grep -c '^## NEXT ACTION'` returns exactly 1, and the new block or row sits below its real header. Two sessions lost their records before this rule.

---

## 4. The planning layer

### 4.1 Spec readiness

The factory needs a spec that can be executed by a session with zero prior context. Before Session 0 freezes anything, the spec (or the map written from it) must contain:

1. **A stage decomposition.** Each stage with entry criteria, an exit gate written as a conjunction of checkable evidence, an oracle statement (what truth this stage's behavior is checked against), a size estimate in sessions, and the owner-gated items it needs.
2. **Locked decisions.** What is settled, each with its source and a named revisit point if one exists.
3. **A trust table** over the spec's own sources (§3.3).
4. **Verified anchors.** Every claim about the existing codebase carries a `file:line` anchor and the date it was checked.
5. **An open-items register** with an owner and a decide-by for each unknown. A spec with silent unknowns is not frozen.
6. **Non-goals.**
7. **The verification contract.** Which ladder gates exist now, and which each stage adds.

Where the spec lacks these, Session 0 writes the map from the spec and gets the owner's sign-off on the locked decisions and the exit gates before the first freeze. That sign-off is a STOP-AND-ASK item by construction.

### 4.2 The map

The map's stage table has these columns: **Stage · Name · Entry criteria · Exit gate · Oracle requirement · Execution spec · Owner-gated · Size · Status**.

Rules:

- **Closed status vocabulary:** `done` · `active` · `next` · `blocked(<on>)` · `later`. Nothing else.
- **Exactly one row is `active`.** It is the default critical-path row. It does not fence off sibling rows: a `next` row with milestones in the same frozen plan is legal parallel work.
- **Status changes only in a stage-gate commit** that carries the gate's evidence, with two exceptions: a freeze whose plan spans a stage may move it `later` to `next`, and a fired time-fused owner item may set `blocked(<item>)`.
- **A flip fires the moment a row's exit gate is met,** even mid-plan. The session that lands the evidence flips the row (§5.9).
- **A calendar-bound gate legitimately lags.** The row stays `active` with all milestones delivered; the lag is recorded as a blocker, and the flip catches up when the window passes.
- **An owner-gated item blocks the flip, never the sibling lanes.**
- **The changelog is a separate append-only file** with columns Date · Section · Change · Why · Commit. Corrections are new rows.
- **Status cells do not accrete narrative.** The Genesis map's cells grew to multi-kilobyte histories; keep history in the changelog and the cell to the status word plus one hash.

### 4.3 Locked decisions and decision records

**Locked decisions** are the do-not-relitigate list. Re-arguing one in a build session is drift. The only doors are the named revisit points.

**Decision records** (ADRs) are minted for any decision that is costly to reverse, an external contract, a correctness or parity posture, a schema or money-core shape, a security or privacy posture, or a cross-session convention. Format: `# ADR-NNNN: <noun phrase>` · Status · Date · Context (value-neutral) · Options considered (mandatory, with at least one real rejected option) · Decision ("We will…") · Consequences (at least one downside) · Enforcement (optional, one line). Rules:

- Written **in the same commit** as the decision it records.
- **Immutable.** The minting sitting may correct a factual error in place, flagged on the Status line. After that, only an append-only dated `## Correction note`. A changed decision is a new record with supersession links both ways. An unflagged Status means an unamended body.
- **Lint at mint, at every flip and freeze:** number matches filename, sections in order, valid Status token, body unchanged since minting (checked against git), referenced from somewhere, a real rejected option, a downside.
- **Open decisions** live in one file as D-entries (context · options · recommended default · decide-by · cost-if-wrong). Each has a same-id `DEC#` row in the register and in the ledger's Waiting-on-owner. Resolution writes the record and collapses the D-entry to a pointer in the same commit.
- **Parallel sessions collide on numbers.** Two sessions minting at once took the same number; a session that mints checks the newest number at commit time, not at draft time.
- **Exit clause.** If records go unreferenced for about two stages, stop writing them.

### 4.4 The register

The register is the durable long tail: everything open that is not in the current plan. It exists because registers kept inside archived plans were never moved: Genesis found more than 100 open rows across six archives, 14 of them already done but recorded open.

- **Index file plus section files**, split by routing and status (`BUILD-inplan.md`, `BUILD-open.md`, `JUDGMENT-open.md`, `OWNER-open.md`, `OWNER-ruled.md`, `CLOSED.md`), each below 225,000 B. The index routes; it holds no rows itself beyond the owner router.
- **Row schema:** `id · essence · source (file + anchor) · blocks · routing · size · deps · escalate-by · status`.
- **Ids are `<SOURCE>#<slug>`**, stable forever. Sources are the document or lane the row came from: `PLAN_S3#`, `OWNER#`, `DEC#`, `HYG#` (hygiene), `OPS#`, `ARCH#`, `TEST#`, `LEDGER#`. Lane milestones are `<LANE>#m<N>-<slug>`.
- **Status lifecycle:** `open` → `in-plan <lane-id>` → `done <commit or session>`, or `refuted <file:line reason>`, or `merged → <id>`. A row is never deleted or renumbered; a refuted row carries the reason so evidence can reopen it.
- **Closed routing vocabulary:** `BUILD` · `JUDGMENT` · `EXPLORE` · `OWNER` · `CALENDAR`.
- **The same-commit rule.** A row you touch in a commit is part of that commit. A milestone's commit flips its row to `done`; a freeze flips its rows to `in-plan`.
- **Owner rows carry `escalate-by`** (a stage or a date). The ledger's Waiting-on-owner section is the ask; the register row is the ledger of record; same id in both. The session-end sweep surfaces every row whose trigger is at or before the active stage.
- **Counting is one reproducible command** (`cat backlog/*.md BACKLOG.md | grep -c '^- \*\*'`) plus a check that the tree matches HEAD. Carried counts do not reproduce.
- **Split levers, in order:** move closed rows to a `*-CLOSED.md` sibling; move in-plan rows to `*-inplan.md`; split by subject on an id-prefix cluster; start a numbered successor. Moves are byte-verified by reassembly.

### 4.5 The freeze (`/factory-spec`)

A stage starts only through the freeze. The freeze produces the plan: a self-contained execution spec that a session with no prior context can execute. Procedure (operative text in Appendix K):

1. **Read** the target stage's map row, the governing spec sections, the design documents, the open-items files, and the plan template.
2. **Check entry criteria** for every stage the plan will span, including that every stage-start decision is resolved. An unresolved one gets a one-paragraph decision brief in Waiting-on-owner, and that path stops; the freeze takes other lanes. A fired time-fused owner trigger gets a dated ESCALATION line first in Waiting-on-owner and first in NEXT ACTION, and the stage is marked `blocked(<item>)`.
3. **Archive the exhausted plan** by `git mv` to `docs/execution/archive/PLAN_<stages>.md`. Before the move, every still-open row of its lanes and its open-items register MOVES (not copies) to the register with status `open` or `in-plan`. An archive with un-moved open rows fails the freeze.
4. **Write the new plan from the template** (Appendix C), re-verifying and dating every `file:line` anchor at write time. Every milestone gets a *Verify* line (exact commands and expected output) and a *DoD* (including the same-commit rule). Milestones with an oracle name the comparison and what "match" means.
5. **For high-stakes stages** set NEXT ACTION to "Owner: skim §Locked decisions + §DoD" before the build starts. Other stages proceed without the pause.
6. **Run the documentation checklist** (decision-record lint, diagram currency, reference health) and leave a greppable token in the changelog line and the plan's commit trail. A freeze without the token is incomplete, and the next session reports it. The token exists because two of six freezes had skipped the checklist and nothing noticed.

**Milestone shape.** A heading line with id, name, tier tag, size and the register ids it closes; then Design; then `*Verify:*` with exact commands and expected output; then `*DoD:*`. The common DoD tail: ladder green · explicit-path commit · ledger updated · register rows flipped in the same commit.

**Milestone sizing.** One milestone is one session: it fits one context window including the ladder run, the review pass and the record. Heuristics from the source project: two to eight files; one register row; a Verify line that runs in less time than the ladder; design fully decided at freeze time. A session MAY take a second small milestone after the first lands and never a third.

**Tags.** Every milestone and every register row carries one: `BUILD·high` · `BUILD·xhigh` · `JUDGMENT·high` · `JUDGMENT·xhigh` · `OWNER` · `CALENDAR`, optionally qualified `EXPLORE` (a whole-session investigation). `BUILD` means turnkey: decided, with the backstop mandatory. `JUDGMENT` means the failure mode is a wrong decision. `OWNER` means a business ruling or a physical act no session can perform. `CALENDAR` means blocked on a clock.

**Lanes.** Lane 0 is a read-only calendar and stop-the-line table: events with owner, trigger, what a session must do, and where the reading is recorded. It preempts everything, by its own triggers and never by lane order. Lane 1 is the critical path, never capped and always first. Lanes 2 and 3 are parallel tracks under a WIP cap: at most one lane in flight beside lane 1. The cap counts lanes, not sessions.

**The interstitial.** When no stage is enterable (a calendar-bound gate is counting, or the owner holds a decision), the plan file still exists: an interstitial written under the same template with an empty stage list. It flips no status, draws its lanes from the register by id, lists only the ranked working set, and is displaced (not exhausted) by the next stage freeze. If its lanes empty while no stage is enterable, write a successor interstitial. Never stop, never manufacture a stage. The interstitial exists because a missing plan file was read as "nothing to do" and because a prose worklist in the ledger "rots by burial."

**Addenda.** A mid-life addition to a frozen plan is a dated addendum section with its own DoD and its own decision records, never an edit to frozen text.

### 4.6 The amendment protocol

When reality contradicts a document: stop, verify the contradiction with `file:line` evidence, classify it, fix the document, add a changelog line, and ship the fix in the same commit as the code that depends on it. Classification: an execution detail fixes the plan; an architecture fact fixes the map and adds a changelog line; a spec defect proven by extraction is corrected in place with a dated appendix note and a severity line for the owner; a conflict with a locked decision is a STOP-AND-ASK. Never silently diverge.

---

## 5. The session loop (`/factory-continue`)

The operative text is Appendix J. This section explains each step and the failure it prevents.

### 5.1 Orient (read-only, in this order)

0. **Signals.** Run the signals script. Paste its one-line output verbatim into the session block. Exit 0 is OK, 2 is a RED, 1 means the script could not look, which is never a green. A RED is recorded as a register row with an escalate-by and is never fixed mid-milestone; a scheduled job missing from the scheduler is an observation, never a scheduler edit.
1. **The map's stage table.** Which row is active, its exit gate, its execution spec.
2. **The ledger.** NEXT ACTION, the hold, blockers, Waiting-on-owner. If the hold is ACTIVE, the hold's queue is the task list and NEXT ACTION is parked (§8).
3. **Reconcile.** `git log` since the newest session-log row's date and Commits cell. A commit that already delivers NEXT ACTION is done-but-unrecorded: record it, advance. This step exists because a session that crashed after its commit and before its record would otherwise have its work redone.
4. **The plan.** The milestone NEXT ACTION names: its Design, Verify and DoD.
5. **The register,** only when NEXT ACTION is blocked or done, by id.

### 5.2 Select (the rung ladder, first match wins)

- **a0.** The hold is ACTIVE: take the first unfinished hold-queue item. Queue empty: stop and tell the owner. Never fall through to the parked list.
- **a.** A recorded blocker is now resolved: clear it, take its milestone.
- **b.** NEXT ACTION's tag matches this session's tier, or it is a Lane-0 item firing by its own trigger: take it verbatim. A JUDGMENT row met by a build session is not started and not decided; it stays where the plan put it. A BUILD row met by a judgment session is skipped for the first open JUDGMENT row. An OWNER row is startable by no session. An EXPLORE row is a whole session; take it only when NEXT ACTION names it or earlier lanes for this tier are empty.
- **c.** NEXT ACTION is done-but-unrecorded: record it, run the exit-gate check for its stage, advance to the next milestone.
- **d.** All plan milestones are done: catch any missed flip (a calendar-bound gate may lag; record it as a blocker), archive the plan, run `/factory-spec`. Under an interstitial this rung does not apply; an interstitial is displaced, not archived on exhaustion.
- **e.** The next milestone is blocked on the owner or mismatched by tag: do not idle. Take the first open row in the plan's lanes for this tier in printed order, then the register by routing, and say so in the log row. Under a hold, this rung resolves to the hold's queue only.
- **f.** Nothing above applies (no open row for this tier anywhere, no stage enterable, no interstitial to write): stop with a named reason. Under the runner, the reason goes into the row's done file as a halt class (§7.5) so the queue halts instead of idling.

### 5.3 Sanity-check anchors

Re-verify every `file:line` the milestone cites against the working tree. Specs are written from a snapshot; code moves. A cite that resolves can still be the wrong line (15 of 120 anchors in one audit resolved to a neighbouring line): read the printed line against its claim. A mismatch means fix the spec per §4.6, then proceed.

### 5.4 Build exactly one unit

Work only the milestone. No drive-by refactors, renames, or "while I'm here" fixes. An out-of-milestone defect gets a register row. Behavior you do not know is extracted from the reference (the legacy system, the owner, the spec), never invented. If the project has a dirty tree from a previous wrap, that tree is milestone-in-progress: continue it, do not restart it.

### 5.5 Validate

Run the ladder (§6.2). The result is the terminal pass string in the log, never the exit code. On a red: fix forward or revert; never commit around it; never start milestone N+1 while milestone N is red. A red at session start is the first work item. The ladder is slow (the source project's reached 65 minutes), so sessions launch it detached with a done-marker file and poll the marker, never a sleep-based clock (§9, rules 1 to 4). A red whose cause is the machine (a dead network mount, a worker that failed to start, a setup phase measured at zero milliseconds) is an environment fault, not a result: check the mount and the process list before reading the gate.

### 5.6 Review (the adversarial pass)

Binds by blast radius: any change to code, any rewrite of canon (map, plan, a frozen lane spec, a decision-governed doc, the register's structure), or any census the register is re-keyed from. Runs after the ladder is green and before the commit.

- **The floor is the author's own pass**, recorded as SELF-REVIEW and weighted as one careful reading. For code: (1) correctness, try to construct inputs or state that break the new logic; (2) spec fidelity, re-read the milestone and its cites against what was built; (3) test efficacy, would the tests fail if the logic were subtly wrong (mutation-style, §6.6). For canon: cite accuracy against the tree, claim-versus-tree, what is missing.
- **Spawned agents are additive**, budgeted at zero wall-clock. Never sequence the commit behind an idle agent; a late finding lands as a follow-up commit. Refuters of concrete cited claims run on the cheaper tier.
- **Trust after smoke.** A wave of agents is trusted only after one has returned a real report on a trivial task. Count results against agents spawned before quoting a tally. An agent that idles silently is DOWN for scheduling and unknown for fact: never "clean", never "found nothing".
- **Agents mutate the live tree** even when told not to, and worktree isolation does not cover untracked new files. Checksum and back up authored files before a wave; diff, search for mutant markers, and check `git status` after; never run the ladder concurrently with a mutation pass. Agent reports truncate; ask for the remainder in parts rather than re-running the agent.
- **The record names what ran:** `SELF-REVIEW only` · `SELF-REVIEW + n of m agents returned` · `agents DOWN`, plus the finding count, in the session block and the log row. A milestone without a backstop paragraph is un-backstopped by definition.

Origin: an audit found 17 confirmed defects shipped by sessions without this pass; across 33 build milestones, the agent mechanism returned before the commit exactly once, which is why agents are additive and the self-run pass is the floor.

### 5.7 Commit

- Explicit paths only. `git add -A`, `git add .`, `commit -a`, `reset --hard`, `clean -f` and `stash -u` are on the harness deny-list (the project settings file), not merely in the contract. A deny pattern needs its exact spelling (the pattern for `git add .` must not also match `git add .claude/`).
- The same-commit rule: the code, its document fix, its register flips, and its decision record ship together.
- Subject: `s<N> — <milestone id>: <imperative summary>`. The session number in the subject makes `git log --grep '^s<N> '` the index into history.
- **The landing-hash problem.** A commit cannot contain its own hash, yet the log row wants the landing hash. The source project pairs every landing with a stamp commit (`s<N> — landing hash <h> on the session-log row`); 31 percent of its commits are record-only. Two designs: (a) the Genesis way, stamp immediately, always true at HEAD; (b) deferred: the log row carries the session number and `(hash: next session)`, and the next session fills it during its own record commit as part of step 5.1.3. This document recommends (b) unless the runner reconciles by hash; the runner described in §7 judges a row by the ledger appearing in the commit range, not by the hash, so (b) is compatible with it.

### 5.8 Record

In the ledger (structure in Appendix D):

1. **Rewrite NEXT ACTION**, never stack a new paragraph on the old one. Write it for a session that saw none of yours: the task in the first sentence, complete sentences, every file, commit or register id with a plain clause saying what it is, no arrow chains. Keep in it only what the next session acts on; what this session delivered belongs in the block.
2. **Append your block** under `## Recent sessions`, newest first, opened by `<!-- session s<N> -->`, then `### s<N> — <date, time span> — <tier/effort/launch mode> — <headline>`, then labelled bullets: What landed · Defaults taken · Recorded not fixed · Tests · Verify · Review · Wrap. The `signals:` line opens the block.
3. **Insert your log row** directly below the `|---|---|---|---|` separator (the table is newest-first). Columns: `Date | Did | Commits | Verify`. The Did cell carries the register id marked done, the tier and effort, the launch mode, the delivery summary and the review tally. The Verify cell carries the pass string with start and end times, or the red gate and why, or `none (docs-only; inherited <pass> from s<N> <hash>)`.
4. **Flip the register rows** you touched, in the same commit.
5. **Run the hot-files check.** Exit 1 means roll the ledger or split a register file before the commit, not after. Exit 2 means the file's structure is broken and the message names the repair.
6. **Assert the structure:** exactly one `^## NEXT ACTION`; the block and the row sit below their real headers.

### 5.9 Navigate

After every milestone, check the exit-gate row of the milestone's own stage in the map. If met, this session performs the stage-gate commit, ideally the landing commit itself: ladder green first; flip the status; append the changelog line; set the next critical-path row `active`; run the documentation checklist and leave its token. Never wait for the plan to be exhausted; one plan may span several stages. If the plan is exhausted, archive it and freeze the next (§4.5 step 3).

### 5.10 Report

The final message ends with a fixed block:

1. **Needs owner.** Every open decision or waiting-on-owner item live right now, new ones first, standing ones with their escalate-by. "None" if none. Before listing an item, check it against git log and its register row: in one sweep two of six "needs owner" items had already been done.
2. **Next session.** NEXT ACTION in one line, plus parallel-lane alternates if it is blocked.
3. **Tier and effort.** Explicit: "BUILD, high" or "JUDGMENT, xhigh". Never "BUILD but escalate X": pre-decide X (§2). Under a hold: the next hold-queue item and how many remain.

### 5.11 The mid-session wrap

When context runs low or a red cannot be fixed this session: commit nothing, not even the ledger. The dirty tree carries the wrap. Write NEXT ACTION as `resume M<N> at <step, file or red gate>`, list the touched files and the red gate with its cause. The record outranks the ladder run when context cannot fit both. The next session treats the dirty tree as milestone-in-progress. Automation honours the same rule: scheduled jobs that act from the build SKIP when the tree is dirty ("not posting from an uncommitted build"), one held night is recorded, the second consecutive one is RED. Under the runner, a wrap that leaves tracked dirt halts the queue by design (§7.5); the halt is the correct outcome, because the next step is a human's.

### 5.12 Owner interaction

**STOP-AND-ASK.** A short list, written into the contract by the owner at Session 0. The source project's: correctness unreachable after extraction · suspected privacy exposure · anything touching a frozen or production system · spending money · a locked decision that appears wrong · dropping or reloading the reference data. Everything not on the list proceeds.

**Waiting-on-owner.** Each ask is a register row with an `escalate-by` and a same-id line in the ledger section. The session-end sweep surfaces every row whose trigger is at or before the active stage, in the final message. A session blocked on an ask never idles (§5.2 rung e).

**Override.** "Stop", "let me drive", "actually do X" always wins; the deviation is recorded. The owner's request then sets the scope the way a milestone does. When the owner asks a question, describes a problem or thinks aloud, the assessment is the deliverable: report and change nothing. A request phrased as a question ("can you run X?") is still a request. The record is still owed; a pure question session with no tracked change says once that the record is unwritten and asks whether to skip it.

**Rulings.** When the owner rules on a batch of questions, record each answer verbatim beside the question with a `put:` clause naming where the ruling now lives. Quote a row's criteria to the owner, never a paraphrase of them. Batch the questions into one sitting at the start, and write the answers into every prompt the night will run (§7.13).

### 5.13 Attended loop mode

Under an attended loop (`/loop /factory-continue`), each iteration is one full pass. The loop stops, without re-arming, when the phase is done (the last buildable milestone landed, or a flip or archive happened, or no open row remains for this tier), when context is heavy (about 500K tokens; stop by 600K with a clean wrap), or when the hold queue is empty. It never schedules idle wakeups; stopping is the normal ending, and the owner relaunches after clearing context. The unattended equivalent is the runner (§7).

---

## 6. Exit validation

### 6.1 Four levels

| Level | Checked by | Passes when |
|---|---|---|
| Milestone DoD | The session, per the plan's *Verify* and *DoD* lines | The exact commands print the expected output and every DoD item is done |
| The ladder | One script, every milestone | The terminal pass string prints |
| Stage exit gate | The session that lands the last evidence | Every conjunct of the map row's exit gate has evidence; the owner has signed where the row says so |
| Plan and spec completion | `/factory-continue` rung d, the freeze | Every milestone done, every spanned stage flipped, the plan archived; the map has no `next` or `later` rows |

### 6.2 The ladder contract

One script yields one result. Properties, each with a failure behind it:

- **Ordered gates, cheap to costly:** build and generated-code currency first, then unit and integration tests, then data and domain checks, then client checks, then the slow replays. The script runs under `set -euo pipefail` and stops at the first red gate.
- **One terminal string:** `VERIFY PASS (<N> gates): <names>`. Sessions and runners key on that string in the log, never on an exit code or a pipeline status. A waiter that greps for a different string ("11/11") never matches; read the terminal string off the script itself.
- **Serial and uncached.** Test packages share one database; parallel packages produced false reds (a foreign-key violation, a partition-create deadlock) that passed clean serially. The test cache keyed on source files, not database state, once served a stale pass after a migration. Serial, `-count=1` or the equivalent, always; a red from a run without these is not a result.
- **Isolated.** The ladder builds its own binary and starts its own server on its own port; it never depends on the running dev server.
- **Checks prove they can fail.** The format check first confirms the tool flags a deliberately misformatted probe; before that, `GOFMT=/bin/true` was a surviving mutant. Each check carries a hand-runnable known-RED and known-GREEN recipe in its comment. A stray guard turns the gate red when a file falls outside the explicit scan scope, so nothing is skipped silently.
- **Prerequisites run inside the ladder.** The ladder seeds its own fixtures, so a red never means "someone skipped a setup command".
- **Summaries print on pass and on fail.** A failing replay must not hide its summary behind `set -e`.
- **Never skip a gate in the committed script.** When a registered data-state red in one test hides the gates behind it, a never-committed scratch copy skips only that test, and the record says the gate was evidenced by two runs.
- **Duration is measured and recorded** per run (start and end in the Verify cell). Design for under 20 minutes; the source project's grew from 30 to 65 minutes as stages wired gates in, and every milestone pays it.
- **Detached execution.** The Bash tool caps a call at 600 s. Launch with `setsid nohup`, write a done-marker file, poll the marker and the runner's liveness by a process listing with elapsed time (keyed on the runner's PID, not the launcher's), and read the result from the log. The launcher waits until no peer test process is running (§9 rule 8).
- **Docs-only sessions do not re-run the ladder.** Their Verify cell reads `none (docs-only; inherited <pass> from s<N> <hash>)`.

### 6.3 Baselines, goldens, exclusions

- **A baseline is a stamped acceptance with a cause.** The invariants gate compares against the newest run stamped `baseline accepted: <cause>`; only an explicit accept command writes the stamp, it refuses a blank cause, and it prints the delta first. An auto-accepting ratchet was considered and rejected: the triage is the point of the gate.
- **Goldens move only on a recorded external event.** Never regenerate a golden to make a gate pass; "the gate is red" is never a cause. A regeneration for a real data event needs no permission provided the cause (the event, its evidence, what changed) is written in the same commit. Volatile fields are masked and the comparison clock is frozen at the capture date, rather than excluding drifting counters.
- **A new output key breaks every golden.** Prefer absent-means-old or a sibling route over re-capturing for a contract change.
- **Exclusions are assertions, not an ignore list.** An exclusion must fire when its trigger occurs or the run fails; entry needs positive evidence; a merely plausible explanation goes to an investigation queue. A replay lane that meets an exclusion its stance map does not know aborts the run, and a new exclusion needs a stance in every lane's map before the gate that reads them runs (that gate sat an hour after the test gate; the refusal was invisible until then).

### 6.4 When a reference system exists

If the project replaces a running system, that system is the test suite, not the spec. Match exactly, replicate documented quirks (each flagged in the exclusion register) rather than fix them, and on a mismatch stop and extract more, never rationalize. "The code is wrong until proven otherwise" is the stance for every differential gate. The harness's own self-test compares the reference with itself and uses none of the new code, so "the oracle is inconsistent" and "the engine is wrong" are separable. The reference's own reports can be blind to what they drop; a figure derived backward from a report is marked `derived (from X)` everywhere and never quoted as a reading.

### 6.5 A minimum ladder for a project without a reference system

Proposed by the validation inventory, not yet run anywhere:

1. Build, format and generated-code currency, with a format-tool self-test and a stray-file guard.
2. The full test suite, integration tests required, serial against any shared state, cache disabled.
3. A differential crosscheck: each output compared across two independent paths (an API response against a direct query of the source of truth). This stands in for the missing oracle.
4. Domain invariants over real or representative data, against a baseline accepted by an explicit stamp with a written cause.
5. Golden snapshots of approved outputs, volatile fields masked, clock frozen, regenerated only for a recorded cause.
6. Client typecheck, tests and build, if the project has a client. Know which typecheck command really checks: one no-op variant exits 0 having checked nothing.
7. Optional: a fixed regression corpus with an accepted baseline, failing only on new mismatches.

### 6.6 Mutation discipline

A test that passes proves little until a test that should fail has failed.

- Run a known-RED and a known-GREEN control before believing any verdict. One harness lied in both directions: a missing PATH made every mutant read KILLED; the fix still read green on two real kills.
- Confirm the patch applied (compare a checksum) before reading a verdict; write patches as files, not shell one-liners; a compile failure is inconclusive, not a kill.
- Uncached runs, always.
- Keep two snapshots: the pre-work backup for the audit, and the authored snapshot as the only restore target. Restoring the wrong one reverted the work and stayed green.
- Survivors have three causes: killed-but-missed, equivalent, or unreachable on this data. The fix for the third is a test that plants the missing data in a rolled-back transaction, never a weaker assertion.
- A mutant on a commit path performs real writes; register actor-keyed cleanup before the test, because counting the rows afterwards detects and cleans nothing.
- Self-authored mutants share the author's blind spots: fifteen of them missed a resume bug an independent reviewer found. Pin the refused path's figure, not only the accepted one. A derived value needs two environment values to pin it: one forced setting cannot tell a derived answer from a literal that happens to match.

### 6.7 Signals between sessions

A fixed, read-only, time-boxed set of checks runs at every session start, defined in a document and implemented by a script; the document is the definition, and when they disagree the script has the bug. Generic checks: every scheduled job present by name at a path that exists; each timed job fired on schedule with its last verdict (one dirty-tree skip is held, two in a row is RED); service liveness read as the age of the newest probe, not a count; new error lines against a count carried forward on the signals line; backup and replication health; disk headroom; data horizon (partitions, quotas); domain readings; job exit-code split; orphaned long queries, with the count prefixed so an empty answer reads as could-not-look. Output is one line, counts and classes only, never identifiers. Exit codes: 0 OK, 2 RED, 1 could not look, and the last is never a green. One outbound act: a phone push on RED or 1, never on green. Every scheduled job joins the signal inventory in the commit that installs it; the rule exists because five of six scheduled lanes had no reader and 188 foreign-key errors went unread for 22 days.

### 6.8 False reds

A red is a result only when the run was clean. The catalog (each one bit at least twice): a parallel test run against the shared database; a cached test result; a client test timing out under machine load (read the environment setup timings first, re-run the file alone, then re-run the full ladder on the unchanged tree; a green isolated run never makes the ladder green); a dead network mount presenting as a worker that failed to start; two test runs alive at once deleting each other's fixtures; a result read off a pipeline exit status; a backgrounded wrapper's exit code; a liveness check that matched its own command line; a `sleep`-based clock; a census truncated by `head`. The numbered rules are in §9.

---

## 7. The runner

The runner turns the session loop into an unattended queue. It is a shell driver living outside the repository, launching one fresh Claude Code session per queue row into its own tmux window, judging each session by a declared deliverable, and halting when a session says it must. Everything in this section is the Genesis implementation, generalized; Appendix L holds the schemas and skeletons.

### 7.1 Placement

- **Outside the repo, on purpose.** Orchestration state is never canon and never outranks the project's own record. The directory holds the queue, the prompts, the templates, per-run directories and logs. Make it its own small git repository, or at least back up every file before editing it (the source project kept `.bak-<date>` copies and had no other undo).
- **One tracked file.** The results ledger (`ORCHESTRATOR_RESULTS.md`) lives at the repo root, is appended by the runner and committed by explicit path. Columns: `when · id · type · outcome · output · dur_min · cost · session · run dir · folded?`. Investigation rows start `UNFOLDED` and a later session flips them `FOLDED` when their output has been read into the record.
- **The runner never edits the repo** beyond that file. The session does the work; the runner schedules and judges.

### 7.2 The queue

One markdown table in one file. The runner parses only rows whose first cell is numeric and that have at least ten fields; prose, dated notes and holds live in the same file around the table, and the runner ignores them. Rows are sorted by position, so position order is run order.

| Column | Values |
|---|---|
| `pos` | Integer. Serial rows below 90; parallel read-only rows at 90 and above |
| `id` | Slug. Also the tmux window name and the log-file stem |
| `type` | `continue` (the generic continue prompt; the session picks its own task) · `kickoff` (a specific prompt file for work the repo cannot derive: a pack sitting, a ruling, a freeze) · `maintenance` (a charter such as queue reconcile) · `burn` (read-only investigation, may run in parallel) |
| `mode` | `watched` (an interactive TUI in a tmux window; the default) · `unattended` (headless `claude -p`; kept for bounded automated rows) · `interactive` (skipped by the runner; a human will run it) |
| `model/effort` | For example `build/high`, `judgment/xhigh`. Aliases resolve inside the runner, never in the queue |
| `prompt` | `prompts/<id>.md`, or `-` for continue rows, which use the template |
| `wrap` | The deliverable class the runner judges by: `commit` · `recon:<dir>` · `maint` · `none` |
| `pre` | Comma tokens: `db` (held through the scheduled-job blackout) · `excl` (waits for the parallel wave to drain) · `nocron` (launcher-only hold) |
| `status` | Runner-written: `running` · `done` · `failed` · `halted`. Human- or launcher-written: `ready` · `hold` · `cronhold` · `budget` · `deadline` · `superseded`. Only an exact `ready` launches. A status cell may carry an annotation in parentheses |

**Why continue rows carry no task.** `/factory-continue` selects the task from the repo state (§5.2). A queue of continue rows is therefore a schedule of slots: how many sessions, of which tier and effort, in what order, with which read-only investigations beside them. The plan is the planner. Kickoff rows exist for the work the repo cannot derive on its own.

### 7.3 The continue prompt

The rendered prompt is composed at launch, in this order:

1. **The source:** `template/CONTINUE_PROMPT.md` for continue rows, else the row's prompt file.
2. **`### LIVE FACTS`, injected at launch and never authored:** HEAD, the commits landed since the last reconcile (capped), tracked and untracked dirt counts, free disk, the count of live sessions, the highest decision-record number, the newest data drop. Prompts carry rules; readings are injected.
3. **Opt-in blocks** the effort cell asks for (a multi-agent opt-in, for example).
4. **One watched line:** the owner may be watching; never sit waiting for input; end your turn when your wrap is complete.

The template says, in substance: run the contract's session-start protocol exactly, the repo schedules the task; wrap in full (ladder green, explicit-path commit, ledger block and row); nobody is at the keyboard, so never ask a question through the question tool, take the recommended default and lodge an OWNER row instead; a STOP-AND-ASK item or an unfixable red gate means the mid-milestone wrap: commit nothing, say BLOCKED in one line, and write a halt class into your own done file so the queue halts, which is correct.

**Traps, each of which cost a night:**

- A prompt that starts with `/` is parsed as an unknown slash command and dropped; the session idles at zero tokens. The row sweep refuses such a prompt.
- Clock clauses are TIMES ("after 06:15"), never weekday names: "Friday = BLOCKED" halted every post-midnight row.
- A dated RE-CUT NOTE in a reused prompt must declare that it outranks the text before it, the text after it, the addendum and the injected facts; otherwise two clauses contradict and the session picks one.
- The halt done-file path is the newest session-id FILE, not the newest run directory: a refused launch leaves a newer, empty directory.
- Never match a process by a pattern the prompt quotes (`pgrep -f`, `pkill -f`): the TUI's argv is the whole prompt, so the pattern matches the session itself, and a `pkill` kills it.
- A watched session that ends its turn is never resumed. A prompt that must wait for a window (a blackout) waits inside the turn, with repeated short sleeps, not by ending.

### 7.4 Launching a row

- One tmux session per project queue; one window per row, named by the row id, running the per-row launcher and then `sleep infinity` so the window stays open for review.
- The per-row launcher: resolve the model alias (unknown → `HALT_BADMODEL`); `cd` to the repo (failure → `HALT_BADCWD`); scrub the environment (unset the API key and every harness variable that would change auth or mode); source the secrets file (missing → `HALT_NOSECRETS`); launch `claude "<rendered prompt>" --session-id <uuid> --model <id> --effort <level>` with a permission mode that cannot block on a prompt. The source project uses the skip-permissions flag; that removes the harness guard, so the deny-list in the project settings, the contract, and a backed-up tree carry the safety. A per-project allowlist is the stricter alternative.
- A background poller reads the model from the first assistant record of the transcript within three minutes; on a mismatch it kills only its own child and writes `HALT_MODEL_MISMATCH`. Every halt class is written before any work is done.
- The headless variant (`claude -p --output-format stream-json`) verifies the model from the init event, sleeps through a usage limit until the reset time plus five minutes and then resumes the same session, kills a row with no output for 90 minutes, retries transient failures up to six times, and fires one resume with a "finish your wrap or say BLOCKED" prompt if the wrap is missing when the process exits.
- The session id is chosen by the runner before launch and reused if the file already exists; moving the file aside forces a fresh session.

### 7.5 Completion detection: the wrap gate

The verdict is the declared deliverable, never the exit code. A session that exits 0 without a wrap is a failure; a session still alive with its wrap complete is done.

| Wrap class | Gate |
|---|---|
| `commit` | HEAD moved and the ledger file appears in `prehead..HEAD` |
| `recon:<dir>` | The directory holds at least one file |
| `maint` | HEAD moved, the hot-files check returns 0, the tracked tree is clean |
| `curate` | The rationale file exists and the queue still parses |

The watcher polls every 60 seconds and returns one of four verdicts:

- **DONE_HALT:** the row's done file holds a `HALT_*` class. This is the session's only in-band way to end its own row. Classes the session may write: `HALT_HARD_STOP` (a STOP-AND-ASK item) · `HALT_RED_GATE` (unfixable red) · `HALT_NEEDS_OWNER` · `HALT_PLAN_COMPLETE` (rung f: nothing left for any tier and nothing enterable) · `HALT_HOLD_EMPTY` · `HALT_PRECONDITION` (a prompt's own precondition failed).
- **WRAP_OK:** the gate reads OK on two consecutive polls, the tree is clean (serial rows), and the session is out of the way: the TUI exited, the window is gone, or the window has been idle for ten minutes.
- **EXITED_MISS:** the TUI is over and the gate is still missing after ten minutes.
- **TIMEOUT_LIVE:** eight hours have passed with the session still live. The queue halts and kills nothing; a human decides what happens to a live session.

After the verdict the row ends one of three ways: gate OK and tree clean → `done`, a results-ledger row, a ledger commit by explicit path; tracked dirt remaining → `halted` with `DIRTY_TREE_AFTER_WRAP` (a mid-milestone wrap is in progress; a human must look); otherwise → `failed`, and the queue continues with the next row.

### 7.6 Gates before each launch

In order, per serial row:

1. Commit the runner's own results-ledger dirt (its first night, that dirt halted the queue).
2. Any other tracked dirt → halt with `DIRTY_BEFORE_LAUNCH`. Never launch onto a dirty tree; dirt between rows halts, it never waits.
3. `excl` rows wait while the parallel wave's lock is live.
4. `db` rows sleep through the scheduled-job blackout window.
5. The meter gate: hold while the provider's short usage window is at or above 85 percent, sleeping until the reset time plus five minutes. A reading of "allowed with warning" is allowed (treating it as a hold once froze a wave for five hours). The built driver proceeds on an unknown reading; the target design fails closed.

At start: refuse if the lock file's pid is alive (`REFUSED_LOCK_HELD`); write the lock; trap EXIT to remove it and write `SUMMARY.md`. A `STOP` file left from an earlier run is cleared at start unless another lane is live (`REFUSED_STOP_HELD`). A dirty tree at start is waited out with 120-second polls needing two consecutive clean reads, up to twelve hours, then `REFUSED_DIRTY_TIMEOUT`.

### 7.7 The main loop

```
refuse if lock pid alive; write lock; trap EXIT → rm lock, write SUMMARY.md
STOP present and another lane live → REFUSED_STOP_HELD; else rm STOP
wait_for_clean_tree (120 s polls, 2 clean in a row, ≤ 720 min) else REFUSED_DIRTY_TIMEOUT
loop:
  for row in snapshot(queue sorted by pos):          # one launch per pass; the snapshot is re-read after each
    skip unless status == ready and type != burn and mode in {watched, unattended}
    STOP present → outcome STOPPED, exit
    commit own ledger dirt; other tracked dirt → halted, HALT DIRTY_BEFORE_LAUNCH, push, exit
    excl → wait for burn lock;  db → sleep past the blackout;  meter_gate
    prehead = HEAD; render = source + LIVE FACTS (+ opt-ins) (+ watched line); status = running
    tmux new-window → run_watched.sh <id> <rendered> <model> <effort> <logdir>
    poll every 60 s:
      HALT_* in done file                                        → DONE_HALT
      gate OK ×2 and tree clean and (exited | window gone | idle ≥ 10 min) → WRAP_OK
      TUI over and gate missing for 10 min                       → EXITED_MISS
      8 h elapsed and still live                                 → TIMEOUT_LIVE (halt, kill nothing)
    WRAP_OK and clean → done + ledger row + ledger commit
    tracked dirt      → halted (DIRTY_TREE_AFTER_WRAP), push, exit
    else              → failed; continue
  if ++pass > PASS_CAP: break                        # the cap counts rows, since one pass launches one row
  if ready rows remain: continue
  once: refill allowed → append up to max_continue canned continue rows; continue
  once: curator allowed → run the bounded curator session (append-only; dirt → halt); continue if ready rows
  break
outcome COMPLETE
```

The parallel mode (`--burn N`) selects up to N ready read-only rows in position order, launches them all, waits for every one, and exits without backfilling a free slot. Parallel rows must carry a `recon:<dir>` wrap and be read-only by charter; with the default setting they skip the clean-tree wait when the serial lane's lock is live, because a read-only row cannot dirty the tree.

### 7.8 Outcomes, stopping and relaunch

The first line of the run's done file is the outcome:

| Outcome | Meaning |
|---|---|
| `COMPLETE` | No ready rows remain after the refill and the curator each fired at most once (a pass-cap break also reads COMPLETE but is flagged in the summary) |
| `STOPPED` | A `STOP` file was honoured after the current row |
| `HALTED at <id>: <class>` | `DIRTY_BEFORE_LAUNCH` · `DIRTY_TREE_AFTER_WRAP` · `WATCHED_TIMEOUT_SESSION_LIVE` · `CURATOR_DIRTIED_TREE` · any `HALT_*` a session wrote |
| `REFUSED_*` | `LOCK_HELD` · `STOP_HELD` · `DIRTY_TIMEOUT` · `DIRTY_TREE` |
| `UNKNOWN` | No done file; the driver died |

Observed over 113 runs in the source project: 30 parallel waves complete, 19 serial COMPLETE, 8 STOPPED, 7 HALTED, 2 UNKNOWN.

**"Runs until the plan is complete."** The driver's COMPLETE means the queue drained, not that the plan is done. Plan completion is signalled by the session: rung d archives and freezes; rung f, when no row for any tier remains and no stage is enterable, writes `HALT_PLAN_COMPLETE` to its done file. The queue halts, a push goes out, and the owner reads the summary. With the refill enabled (below), the queue keeps appending continue rows until that halt or a halt for an owner decision; without it, the owner packs a night's worth of rows at a time (§7.13).

**Refill.** The built driver can append up to `max_continue` canned continue rows once per drain, and a bounded "curator" session can append read-only rows and propose holds. Both are off in the source project: the refill once jumped ahead of a held row, a trailing comment on its config line stopped the setting binding at all, and the curator once dirtied the tree. The accepted target design moves the refill into the outer launcher with limits (at most two rows per drain, eight drains per day) and a no-progress detector: a continue row that leaves HEAD unchanged pauses the queue with a needs-owner reason. Use those limits from the start.

**No signals-RED gate exists in the built driver.** Sessions record a RED but never stop the queue for it. The target design adds it as a launch gate (§7.14); add it from the start.

**Stopping:** `touch STOP`. The serial driver finishes its current row and writes STOPPED; a parallel wave stops launching and drains. The file is cleared at the next start, or honoured if a lane is still live. `rm STOP` un-stops before anything noticed.

**Relaunch is a human act in a clean session.** The control command runs a pre-flight over the newest run of each mode, first matching rule wins, and a blocking rule from either summary blocks:

1. A lock is LIVE → a driver is running; a serial launch reports "already running"; a parallel launch beside a live serial lock is fine.
2. No prior runs → first run; launch.
3. A run directory with a driver log but no summary and no live lock → the driver died hard. Do not launch; show the log tail; the owner decides.
4. The summary's attention list has a HALT, VIOLATED or `HALT_*` line → do not launch. A dirty-class halt needs a supervised session; a config-class halt (model, cwd, secrets) needs a queue or runner fix.
5. Outcome REFUSED or UNKNOWN → surface it; STOP_HELD and LOCK_HELD are safe to relaunch after checking locks; DIRTY_TIMEOUT means find out what was live; UNKNOWN is treated as rule 3.
6. Failed rows, hold proposals, unfolded investigation rows, a stopped run → do not block, but surface each with a one-line recommendation.
7. Attention empty → launch, and say so.

Then: `tmux new-window` running the driver, report the run directory and the attach hint, return immediately. Never wait on or poll the queue from the launching session. The owner's standing preference in the source project is to launch the night from a fresh session of their own, given a console to-do written by the packing session.

### 7.9 Reconcile

Rows go stale because work lands by hand in attended sessions. Mechanisms:

- **LANDED SINCE.** The driver injects `git log --oneline --since=<RECONCILED_AT>` into every prompt (capped at 40 commits), so a session sees what landed since the queue was last reconciled. `RECONCILED_AT` is stamped only by an evidence-backed reconcile (a reconcile charter row, a pack sitting, the re-arm script), never mid-run: re-stamping it mid-run erases the landed list for rows cut earlier.
- **The reconcile charter:** audit every non-done row at its source, reading its prompt rather than trusting its id; flip to `done` only with a commit hash AND a session number; check each conjunct of a compound hold separately; fix the narrative prose to match the table with dated `[UPDATED …]` clauses.
- **Only humans or evidence-backed reconciles change status.** Automated planners (the curator) append and report; a curator lists rows that look overtaken under its own heading and flips nothing.
- **Zero-launch subcommands:** `--status` (parsed queue, freshest meter reading, unfolded count, recent runs, locks) · `--dry-run` (renders every ready prompt and prints the launch plan and the gates that would fire now) · `--review [run]` (the morning digest: summary, wrap-gate lines, a six-line final-message excerpt per row, the pre-captured `git show --stat` per commit row, a `claude --resume <sid>` line per row, the ledger tail).

### 7.10 The supervisor

The session that launches the runner supervises it, or hands the duty to a script. The supervisor script tails the run and emits events: LAUNCHED, MODEL_VERIFIED, one LAUNCH_CHECK (OK, HOLDING or PROBLEM), filtered driver lines, `STALL <id>` when a row's transcript has been quiet for 30 minutes and its wrap is missing, COMPLETE when the summary lands.

What the supervisor MAY do: send one nudge per stalled row (a `continue` typed into the pane, with Enter as a separate keystroke after a pause, only after a probe reads the window as usable, at most three per row); fix queue cells; clear a stale lock or STOP; relaunch after a config-class halt; reach a hand session with one factual message.

What it MUST NOT do: resolve a dirty-class halt; kill a live row; edit tracked files; relaunch onto dirt.

Notifications: a fail-open, silent push on every HALT (halts once left the lane dead for between 2.5 and 54 hours before the push existed), on transitions only, never on green. A delayed push is a reminder channel for the morning.

A long-lived monitor replays or loses lines past its own cap; a single-pass sweep script woken on an interval is the sturdier shape.

### 7.11 Concurrency and the blackout

- **One code lane, serial.** Parallel rows are read-only investigations, each naming the consumer that will fold its output.
- **Never two test suites on the one shared database** (§9 rule 8). Launchers refuse while a ladder run or a test run is alive, detected by an anchored argv match or a pidfile, never by a pattern a prompt might contain.
- **A clean-tree hand session is invisible to the driver.** The driver's wait gate sees tracked modifications only; a supervised code session with a clean tree can be launched over. The hand launcher waits on the harness's session-status file reading idle.
- **Parallel lanes race on the ledger.** Two sessions editing the ledger at once commit their hunks through a temporary index and take the session number from the dirty tree.
- **The blackout.** Scheduled jobs that act on the database own a window (04:15 to 06:15 in the source project). The driver holds `db` rows at launch inside it; the outer launcher flips `db` and `nocron` rows to `cronhold` before the band and restores them after, so docs-only rows fill the band. Because the jobs start before the band and the gate only covers the launch, prompts must wait inside the turn.

### 7.12 Budget

- **Per row:** headless rows report cost and duration in their result event; watched rows have no result event, so their cost cell is unknown unless the transcript carries it, and duration is wall clock. Budget in meter points, not currency.
- **Driver gate:** hold a serial row or a parallel wave while the short window is at or above 85 percent; a long-window reading at or above 95 percent only warns. The reading comes from the freshest rate-limit event in a log younger than ten minutes, else one cheap probe.
- **Launcher rules** (one-way flips; touch STOP first to keep rows `ready`):

| Knob | Default | Effect |
|---|---|---|
| Wave threshold | 75 % short window | No parallel wave above it |
| Parallel long-window stop | 90 % | Ready parallel rows flip to `budget` |
| Docs-only threshold | 96 % | Only rows with no `pre` token may start |
| Serial long-window stop | 96 % | Ready and cronhold serial rows flip to `budget` |

- **Measured rates** in the source project, in long-window points: a build-tier serial row about 0.5 per hour; eight serial rows about 2; three or four parallel lanes about 2 per hour; a 13-agent review pack about 4; a judgment-tier sitting 6 to 10. A short-window gate can refuse waves while the long window is the one that binds; gate on the window that binds, measured from the provider's reset time.

### 7.13 Packing a night

Until the refill is trusted, a human-led sitting packs the queue. In order:

1. Pre-flight (§7.8) and a meter reading.
2. Reconcile (§7.9) and stamp `RECONCILED_AT`.
3. Pick rows: the critical-path continue row first; docs-only rows at the highest serial position so they fill the blackout band; parallel investigations that name a consumer.
4. One question round with the owner, answers written verbatim into PRE-ANSWERED blocks in every prompt the night will run.
5. Writers draft the kickoff prompts in parallel; one refuter and one fixer per prompt; then mechanical checks (no leading slash, clock clauses as times, the halt done-file path, the precondition tests run).
6. Insert the rows with a dated header note; write the night's launcher script with its manifest (each prompt present and at least its minimum size).
7. Run `--status` and `--dry-run`.
8. Hand the owner a console to-do: the launch command, the attach hint, the deadline.

A live top-off edits the queue only right after a `LAUNCH` line in the driver log, because the driver rewrites the whole file through a temporary file and a rename when a cell flips; an edit in the same second is lost.

### 7.14 The perpetual launcher (designed, not yet built in the source project)

The accepted target design replaces the nightly human pack with a scheduled launcher:

- **Control plane by files:** `STOP` (only the owner clears it), `state/PAUSE <reason>` (the system pauses itself), `state/RESUME` (a consume-once token).
- **Eleven gates before any launch:** no STOP or PAUSE · no launcher alive (pidfile, never a pattern match) · the driver dead with COMPLETE or STOPPED · no ladder or test run alive (exact command line) · a clean tree · no busy hand session (the harness's session files) · the scheduler table identical to its baseline · free disk above a floor · the signals script returning 0 · the meter under a pace line keyed to the reset time, failing closed when unknown · a ready row or a gated refill.
- **Refill in the launcher:** at most two rows per drain and eight drains per day; a no-progress commit triggers PAUSE with a needs-owner reason.
- **Sessions report needs-owner or WRAPPED through their row done file.**

Build this after the nightly pack has run cleanly for a few weeks; the gates above are the lessons of those weeks.

---

## 8. Model routing

Two tiers, routed by tag (§4.5). A single-model project sets every tag to one tier and skips the hold; the rest of this section still applies to effort.

- **Tags are on rows, not sessions.** A session knows its own tier (the runner passes it; an attended session reads it from the model name) and matches rows to it (§5.2 rung b).
- **The judgment tier does** freezes, flips, rulings, stop-the-line triage, high-stakes semantic milestones, and the design of anything a wrong decision would make expensive. **The build tier does** everything pre-decided, with the mandatory backstop.
- **Refuters run on the cheaper tier** in all sessions. Finding things keeps the session's model; refuting a concrete cited claim is verification work.
- **The hold.** When the judgment tier is unavailable or rationed, the owner declares a HOLD in the ledger: a status word, an ordered queue of pre-vetted build-tier items (non-judgment, fully specified, free of locked-decision content), and a PARKED list. While ACTIVE, build sessions work the queue in order, one item per session, never start anything on the parked list, and stop when the queue empties. The owner lifts the hold. It is ledger state, never a decision record, because it must be cheap to reverse.
- **Effort is explicit.** Every recommendation and every queue row names the effort level ("high", "xhigh"); never leave it implied.
- **Cost.** Record per-row duration and cost class in the results ledger (§7.12) so outliers are visible: a five-minute "sitting" or a four-hour maintenance row means the prompt or the session went wrong.

---

## 9. Process rules: incubation, graduation, the catalog

### 9.1 How a rule is born

A rule starts in the ledger's STANDING section (the incubator) when something bites, naming the session that was bitten. It is copied into the contract when it has bitten twice, taking the next unused number. Numbers are stable ids cited across the repo and are never renumbered or reused; a retired number stays retired. Fresh traps that have not yet bitten twice sit in the ledger's Fresh traps section and graduate or are deleted at a stage gate.

### 9.2 The graduated rules, stated generically

1. Never read a result off a pipeline's exit status.
2. Never read a result off a backgrounded wrapper's exit code.
3. Never trust a liveness probe that can match its own command line; confirm with a process listing that shows elapsed time. A `pkill` by pattern can kill the session running it.
4. Never trust a sleep-based clock; read the real time.
5. Never read a census through a truncation, and never read a field from a multiplexed record store without the discriminator that selects the record type.
6. The adversarial pass binds by blast radius; the self-run pass is the floor; agents are additive (§5.6).
7. Spawned agents share the live tree: checksum before, diff after, no ladder run during a mutation pass.
8. Never run two test suites against one shared database.
9. A mutation harness needs its own RED and GREEN controls, and every survivor is hand-confirmed.
10. Know which typecheck command really checks; a no-op variant exits 0.
11. A red under machine load is a timeout, not a result: read the setup timings, re-run the file alone, re-run the full ladder on the unchanged tree; after the identical red on two full ladders, apply the pre-filled configuration fix instead of a third re-run.

### 9.3 The failure catalog

What bit, and the rule it produced. Keep this list; it is the argument for every MUST above.

| What happened | Rule |
|---|---|
| The ledger grew to 2.29 MB and 138 blocks; sessions read it truncated | Hot-file ceilings and the roll (§3.2) |
| An edit anchored on a substring landed inside the header comment; two sessions' records vanished | Line-start anchors plus a post-edit assertion (§3.4) |
| Log rows inserted above the separator broke rendering and crashed the roll tool for 14 sessions | Rows go directly below the separator, newest first |
| A roll tool with a fixed `--keep 10` refused silently at every roll it attempted | The keep derives itself from the size headroom; refusals name the repair |
| Registers inside archived plans held 100+ open rows, 14 already done | One durable register; rows MOVE at archive (§4.4) |
| A missing plan file was read as "nothing to do"; a prose worklist rotted by burial | The interstitial (§4.5) |
| 17 confirmed defects shipped by sessions with no review pass; agents returned before commit once in 33 milestones | Blast-radius review with a self-run floor (§5.6) |
| A mutant reached a commit; agents created untracked files worktree isolation did not cover | Rule 7; mutant-marker audit in the staging command |
| Parallel test packages on one database produced an FK violation and a deadlock that passed serially | Serial, uncached ladder (§6.2) |
| Five of six scheduled lanes had no reader; 188 FK errors went unread for 22 days | Signals with three exit codes; every lane has a reader (§6.7) |
| A FATAL verdict read OK because the capture pattern missed it | Probe for holes; a check must prove it can fail |
| Zero-diff days accrued while the maintainer process was dead | A clean day needs a clean health row too |
| Seven notes in off-repo memory still said "uncommitted" after the work landed | Memory holds lessons, never status (§10) |
| 2 of 6 freezes skipped the documentation checklist | Greppable obligation tokens; a freeze without one is incomplete |
| A session exited 0 with no wrap | Judge a row by its wrap gate, never its exit code (§7.5) |
| Authored prompts carried stale facts | LIVE FACTS injected at launch, never authored (§7.3) |
| Hand-landed work was re-proposed by the queue | LANDED SINCE plus `RECONCILED_AT`; planners append and report (§7.9) |
| The runner's own ledger row dirtied the tree and halted the night | The dirt check ignores that file; the runner commits it first (§7.6) |
| A queue-file edit in the same second as a status flip was lost | Rewrite only on a flip; top off only right after a launch line (§7.13) |
| A finished row never exited because of hung sub-agents (about 7 % of rows) | Eight-hour backstop; the supervisor verifies outputs, then interrupts the TUI (§7.5) |
| A BLOCKED row idled for eight hours | The session writes a halt class to its own done file (§7.5) |
| A continue prompt's "Friday = BLOCKED" clause halted every post-midnight row | Clock clauses are TIMES, not weekday names (§7.3) |
| A prompt file beginning with `/` was parsed as a slash command and the row ran nothing | Prompt files never start with `/` (§7.3) |
| `pkill -f` on a quoted pattern killed the session running it | Kill by anchored argv or PID (§7.3, rule 3) |
| The refill jumped ahead of a held row; a trailing comment defeated its config line | Refill with limits in the launcher; parse config strictly (§7.8) |
| The curator dirtied the tree | Automated planners are append-only; dirt halts (§7.8) |
| A live bash script was edited while running | Swap scripts by rename, never in place |
| Halts left the lane dead for 2.5 to 54 hours | A push on every halt (§7.10) |
| A session waiting on a long sub-agent wave looked idle and was closed | Run multi-agent waves outside the queue |
| 15 of 120 cite anchors resolved to a neighbouring line | Read the printed line against its claim (§5.3) |
| A new bundle key diffed every golden as ADDED | Absent-means-old or a sibling route, never a re-capture for a contract change |
| Carried register counts did not reproduce | One reproducible count command |
| A rule numbered twice in two parallel sessions | Stable ids assigned at graduation only; parallel sessions coordinate numbers |
| A dead network mount read as a client-test timeout | Check the environment before reading a gate (§5.5) |

---

## 10. Memory versus record

Off-repo memory (the harness's per-project memory directory) holds only what the repo does not record: lessons, traps, working methods, preferences, pointers to external resources. It never holds a "milestone landed", a status, or a commit hash: the ledger, the register and git already hold those, and the copy goes stale within days. A memory is a hint, not a record: check its claim against the tree before acting on it, and fix or delete a note found wrong in the same session. The memory index is itself a hot file with a read ceiling; when it overflows, move full hooks to a long-form index and keep one line per note. A one-time prune, with the owner approving every deletion, is a register row.

---

## 11. Adoption

### 11.1 Session 0: the bootstrap procedure

Run by one judgment-tier session with the owner present. Each step ends in a commit by explicit path.

0. **Preconditions.** A git repo on a working branch; a spec or its material; a build-and-test command that runs; the CLI authenticated; tmux installed if unattended runs are planned. Measure the tool read ceiling once (read a large file; note where it truncates).
1. **Inventory.** What plan, tracker, tests, CI, scheduled jobs and shared state (databases, ports, external services) already exist. List every path that may carry secrets or private data. Record all of it in `docs/process/INVENTORY_S0.md`.
2. **Write the contract** (Appendix A) with the project's traps filled in: the staging deny-list in the harness settings, the off-limits systems, the privacy rule, the shared-state rule. Agree the STOP-AND-ASK list with the owner and write it in.
3. **Build the ladder** (Appendix G) with at least gates 1 and 2, each with a known-RED and known-GREEN control. Run it twice; record its duration in the contract.
4. **Write the map** (Appendix B) from the spec: stages with entry criteria and evidence-conjunction exit gates, the oracle statement per stage, the locked decisions, the trust table, the changelog file. The owner signs the locked decisions and the exit gates. This is the one place Session 0 asks.
5. **Create the ledger** (Appendix D), the register (Appendix E), the decision-record directory with the template (Appendix F), and the signals script with at least three checks (Appendix H): scheduler or CI present, service liveness, disk headroom.
6. **Install the commands** (Appendices J and K) and the hot-files check (Appendix I).
7. **Freeze the first plan** with `/factory-spec` for the active stage. The owner skims the locked decisions and the DoD.
8. **Run three attended sessions** with `/factory-continue`. Fix the contract wherever a session stumbles; write what bit into the ledger's STANDING section. Graduate nothing yet.
9. **Set up the runner** (Appendix L): the directory, the queue with two continue rows, the templates, the per-row launcher. Run one watched row with the owner present, then one serial night of two or three rows. Read the morning digest. Then run unattended.
10. **Measure after the first two weeks** (§12): bookkeeping ratio, ladder duration, wrap rate, false-red count, owner asks per week, rows per night, halts per week and their classes.

### 11.2 Mapping an existing plan-and-tracker setup

Many projects already run a simpler version: a plan file with numbered items, a progress tracker with a queue table and a session log, acceptance tests per item, one commit per item, resumed by a human after each context clear. The mapping:

| Existing piece | Becomes |
|---|---|
| The plan's items | Plan milestones, each with a tier tag, a *Verify* line and a *DoD* |
| The tracker's queue table | The plan's lanes (printed order is run order); the tracker's status words map to the register lifecycle |
| The tracker's ground rules | The contract's standing traps and verify discipline |
| The tracker's session log | The ledger's session log and session blocks |
| Owner decisions listed in the plan | Locked decisions in the map, each with a decision record when significant |
| Per-item acceptance tests | The milestone's *Verify* line; the regression set becomes ladder gate 2 |
| One commit per item after its tests | Unchanged, but behind the ladder, with the same-commit rule and explicit paths |
| "Resume after context clear" | `/factory-continue` under the runner; the human relaunches only on halt |
| Items marked blocked with a reason | Register rows with routing and escalate-by; the session takes the next row instead of stopping |

What the existing setup usually lacks, in order of value: the gate ladder as a single script with a pass string; the stage table with exit gates; the register that outlives the plan; the signals probe; the record discipline that lets a session start cold; the runner.

### 11.3 The bootstrap prompt

Paste into a Claude Code session in the target project, with this document copied to `docs/process/SOFTWARE_FACTORY_PROCESS.md`:

```
Read docs/process/SOFTWARE_FACTORY_PROCESS.md in full. Then run Session 0 (§11.1)
for this project: steps 0 to 7 today, with me present for steps 2 and 4. The spec
is at <path>. Our build-and-test command is <command>. Shared state: <list>.
Paths that may carry private data: <list>. Use the templates in the appendices;
fill in every <placeholder>; commit each step by explicit path with the subject
"s0 — <step>: <summary>". Stop only at the STOP-AND-ASK points the document names.
```

---

## 12. Known costs and the knobs

Measured on the source project after about 490 sessions:

| Measure | Reading |
|---|---|
| Record-only commits | 286 of 932 (31%) |
| Commits that only stamp a landing hash | 202 |
| Bookkeeping bytes to code bytes | 1.2 : 1 |
| Record written per session | about 10 KB |
| Ledger roll commits in six weeks | 70 |
| Ladder duration | 30 to 40 min at month two, 65 to 67 min at month three |
| Agent-returned-before-commit rate | 1 in 33 build milestones |
| Runner outcomes over 113 runs | 49 complete, 8 stopped, 7 halted, 2 died |
| Rows that never exit because of hung sub-agents | about 7 % |

The knobs, with the source project's setting and the trade-off:

- **Landing-hash stamp:** immediate (always true at HEAD, one extra commit per session) or deferred to the next session's record commit (§5.7). Choose deferred unless the runner reconciles by hash.
- **Full ladder every milestone** (the source default; every milestone pays the full duration) or a fast ladder per milestone plus the full ladder per N milestones and nightly. The second weakens the guarantee that every commit is fully green; if chosen, the record must say which ladder ran, and a nightly red stops the line.
- **Session blocks in the ledger** (rich, newest first, rolled at 12) or a one-row log plus a per-session file under `docs/execution/sessions/`. The second keeps the ledger small at the cost of one more file per session.
- **Review agents:** off, self-review only (the floor), or additive agents. Never make agents blocking.
- **Two tiers or one.** One tier removes routing and the hold but keeps the PRE-DECIDE rule: a session still writes a decision-free NEXT ACTION for its successor.
- **Owner checkpoints per stage:** a skim of locked decisions and DoD for high-stakes stages only (the source default) or for every freeze.
- **Watched or headless rows.** Watched (a TUI in tmux) lets the owner look in and lets a stalled row be nudged; headless reports cost and can resume itself through a usage limit. The source project chose watched wholesale after headless rows died silently.
- **Refill on or off.** Off means a human packs each night and the queue cannot run away; on means the queue runs until a halt, with the limits in §7.8.

---

## Appendix A. Contract template

Place in `CLAUDE.md` at the repo root. Keep it to rules; the rationale lives in this document.

```markdown
# <Project> Build — Session Contract

<Project> = <one line: what is being built, and against what truth>. You are one
session in a months-long build. The map is docs/MAP.md. Trust the docs in the
order below, not your priors about this repo.

## ON SESSION START — AUTONOMOUS MODE
0. Run `bash scripts/signals_check.sh` (read-only, under 60 s). rc 0 = OK · 2 = a
   RED · 1 = could not look, which is NOT a green. Paste its `signals:` line into
   your block verbatim. On RED: record a register row, never fix mid-milestone.
1. Read docs/MAP.md §2 (stage table): know the active stage and its exit gate.
2. Read SESSION_STATE.md: its NEXT ACTION is your task, UNLESS §HOLD reads ACTIVE,
   in which case the hold's queue is your task list and NEXT ACTION is parked.
   Queue empty ⇒ stop and tell the owner.
3. `git log --oneline` since the newest `## Session log` row (its date and Commits
   cell). A commit already delivering NEXT ACTION = done-but-unrecorded: record it,
   advance.
4. Read docs/PLAN.md, find the milestone NEXT ACTION names. If blocked or done, take
   the first open row in Lanes 1–3 whose tier tag matches your model, in printed
   order; a Lane-0 row preempts by its own trigger. Long tail: docs/execution/
   BACKLOG.md by id. No PLAN.md ⇒ write the interstitial (/factory-spec
   interstitial) before any other work.
5. Begin immediately. Sessions run unattended; asking "should I continue?" blocks
   the work. Stop only where this file says to. Never re-plan the stage.

User override always wins: comply, then record the deviation. A question or a
problem described is answered, not acted on; a request phrased as a question is
a request.

## ON SESSION END (or before compaction) — MANDATORY
1. `bash scripts/verify.sh` → `VERIFY PASS (<N> gates)` (or record which gate is
   red and why).
2. Commit milestone-complete GREEN work by EXPLICIT paths (never `git add -A`).
   Partial or red work is NEVER committed.
3. Update SESSION_STATE.md: rewrite NEXT ACTION (never stack), add your
   `<!-- session s<N> -->` block, insert your log row directly below the
   `|---|---|---|---|` separator, flip the BACKLOG rows you touched, all in the
   SAME commit. Then `bash scripts/hot_files_check.sh`: exit 1 ⇒ roll
   (`python3 scripts/session_state_roll.py --write`) or split BEFORE the commit.
   Anchor every edit on the LINE-START header (`^## NEXT ACTION`); afterwards
   `grep -c '^## NEXT ACTION'` must print 1.
An unrecorded session is a lost session.

## STAGE NAVIGATION
- After EVERY milestone check the exit-gate row of its own stage. If met, THIS
  session makes the stage-gate commit: flip the status, append the changelog
  line, set the next critical-path row `active`.
- PLAN.md exhausted ⇒ archive it and /factory-spec the next stage. Stages start
  ONLY through /factory-spec.
- Blocked on the owner ⇒ never idle: take the next open row for your tier.
- Work ONLY the milestone. An out-of-milestone defect gets a register row.
- Mid-milestone wrap: commit NOTHING. NEXT ACTION = "resume M<N> at <step>".
- At session end, sweep §Waiting-on-owner and the register's OWNER rows: surface
  every item whose escalate-by is at or before the active stage.

## UNDER THE RUNNER (when the prompt carries "### LIVE FACTS")
Nobody is at the keyboard. Never ask a question through the question tool: take
the recommended default and lodge an OWNER row. A STOP-AND-ASK item or an
unfixable red ⇒ the mid-milestone wrap, one BLOCKED line, and write your halt
class (HALT_HARD_STOP · HALT_RED_GATE · HALT_NEEDS_OWNER · HALT_PLAN_COMPLETE ·
HALT_HOLD_EMPTY) into your row's done file. Never end your turn to wait; wait
inside it. Never match a process by a pattern this prompt contains.

## STANDING TRAPS
- NEVER `git add -A` / `git add .` (enforced by the settings deny-list).
- <Private-data paths>: never copy rows into docs, fixtures, commits or temp dirs.
- <Off-limits systems and ports>.
- Goldens/baselines are never regenerated to make a gate pass; a regeneration
  needs its cause written in the same commit.
- Behavior you do not know: extract from <the reference>, never invent.

## VERIFY DISCIPLINE
`bash scripts/verify.sh` = the <N>-gate ladder: <gate names>. Tests run SERIALLY
and uncached; a red from a parallel or cached run is not a result. Env: <paths>.
Shared DB: <how to reach it>. Duration: about <m> minutes; run it detached.

## PROCESS RULES (graduated from SESSION_STATE §STANDING after biting twice;
numbers are stable ids, never reused)
(1) Never read a result off a pipeline's exit status. (2) ... (see §9.2)

## STOP AND ASK THE OWNER
<The agreed list, one line each.>

## AMENDMENT
Reality contradicts a doc ⇒ verify with file:line evidence ⇒ fix the doc + one
changelog line in the same commit ⇒ proceed. Never silently diverge.

## DECISION RECORDS
A significant decision (costly to reverse · external contract · correctness
posture · schema/money shape · security/privacy · cross-session convention) gets
an immutable record in docs/adr/ in the SAME commit. Never edit an accepted
record; append a dated correction note or supersede.

## COMMANDS
/factory-continue · /factory-spec · /factory-verify · /factory-queue live in
.claude/commands/.
```

## Appendix B. Map skeleton

```markdown
# <Project> — MAP
§0 How to read · §1 Mission and the gate ladder · §2 Stage table · §3 Document
router and trust table · §4 Locked decisions · §5 Standing traps · §6 Guardrails
· §7 Verification ladder (gates per stage) · §8 Open inputs (pointers, blocking
only) · §9 Session protocol (points at CLAUDE.md) · §10 Changelog (table in
MAP_CHANGELOG.md)

## §2 Stage table
Status vocabulary: done · active · next · blocked(<on>) · later. Exactly one
row is `active`. Status changes only in a stage-gate commit (exceptions: a
freeze may move later→next; a fired owner trigger may set blocked(<item>)).

| Stage | Name | Entry criteria | Exit gate (evidence conjunction; §7 refs) | Oracle | Execution spec | Owner-gated | Size (sessions) | Status |
|---|---|---|---|---|---|---|---|---|
| S0 | Foundations | repo + ladder gates 1–2 green | build+tests green on CI; schema v1 migrated up and down; gate 3 wired | none | PLAN_S0.md | — | S (1–2) | active |
| S1 | <name> | S0 done; <decision D-01> resolved | <checkable conjuncts> | <what truth> | PLAN.md | <item, by when> | M (3–6) | next |

Critical path: S0 → S1 → S3 → S5. Parallel: S2 beside S1; S4 beside S3.

## §3 Trust table
| Asset | Trust | Use |
| spec/REQUIREMENTS.md | CANON | build input |
| docs/old-design.md | STALE | never a build input |

## §4 Locked decisions (re-arguing one in a build session is drift)
| Id | Decision | Source | Revisit point | ADR |
| D-01 | ... | owner, <date> | <stage or event> | ADR-0003 |

## §7 Verification ladder
Gates now: 1 build · 2 tests · 3 <...>. Per-stage additions: S1 adds gate 4 (...)
```

`MAP_CHANGELOG.md`: `| Date | Section | Change | Why | Commit |`, append-only.

## Appendix C. Plan template

```markdown
# <Stage Sn>: <Name> — Execution Spec
> Implements: <spec §> · Governed by: MAP §2 row Sn · Status: FROZEN <date> (s<N>,
> <tier>) · Displaces: <prior plan, archive name> · Displaced by: <next freeze>

**Mission.** (2–4 sentences: what exists when this stage is done.)
**Self-contained.** This spec + BACKLOG rows by id + the files it anchors =
everything needed. Read anchored files before editing; import no prior-session
assumptions.

## Tag vocabulary
BUILD·high · BUILD·xhigh · JUDGMENT·high · JUDGMENT·xhigh · OWNER · CALENDAR;
qualifier EXPLORE (a whole session).

## How a session picks its milestone
1. Lane 0 fires by its own triggers, never by lane order.  2. NEXT ACTION if its
tag matches.  3. Lane 1 first open row for your tier.  4. Lanes 2–3 under the
WIP cap (one lane in flight beside lane 1), printed order.  5. BACKLOG by
routing.  6. No open row ⇒ write a successor worklist; never stop, never
manufacture a stage.

## Locked decisions (do not relitigate) — each names its durable home
1. <decision> (MAP D-xx / ADR-nnnn / owner <date>)

## Verified codebase facts — checked <date>; re-read before editing
- <claim> — `path/file.ext:123`

## Design
(architecture; new files; file-by-file edit list for existing files)

## Milestones — lanes; ONE per session; verify each before the next
## Lane 0 — calendar / stop-the-line (READ-ONLY table)
| # | Event | Owner·tag | Trigger | What a session does | Where the reading is recorded |

## Lane 1 — the critical path (never capped, always first)
### M1 — <name> — BUILD·high — S — closes `<LANE>#m1-<slug>`
Design: ...
*Verify:* `<exact command>` → `<expected output>`; `bash scripts/verify.sh` →
`VERIFY PASS (<N> gates)`.
*DoD:* <checklist> · commit by explicit paths · SESSION_STATE.md updated · the
BACKLOG rows this milestone closes flipped in the same commit.
### M2 — ... — DONE s<N> <date> (<hash>)   ← done is stamped inline

## Lane 2 — <parallel track> (at most one lane in flight beside lane 1)
## Lane 3 — <the rest> (printed order = the register's lane order)

## Risk playbook (decided mitigations — no open questions live here)
## Out of scope (for every session under this spec)
## Definition of done (the stage exit gate, restated concretely)
## Open items register (honesty section: item · owner · what it blocks)
## Commit trail (append during execution)
| Commit | Milestone | Delivered |
## Operational learnings (append during execution; graduate keepers)
```

## Appendix D. Ledger template

```markdown
# SESSION STATE — <Project> build
<!-- The ONLY mutable session ledger. Updated at the end of EVERY session.
     Stage status lives in docs/MAP.md §2 — not here. -->
<!-- RETENTION RULE. HOT = this file: ONE "## NEXT ACTION" · "## Waiting-on-owner"
     · "## Waiting-on-judgment" · "## HOLD" · "## STANDING" (edited in place, never
     appended) · "## Recent sessions" (newest ~10 blocks, newest first, each opened
     by <!-- session s<N> -->) · "## Session log" (rows of the sessions whose blocks
     are here; NEWEST FIRST; a new row goes DIRECTLY BELOW the separator) · "## Fresh
     traps". ROLL when the file passes 200,000 B or 12 blocks:
     python3 scripts/session_state_roll.py --write moves the oldest blocks AND
     their log rows to docs/execution/archive/SESSION_STATE_s<a>-s<b>.md (frozen;
     their log rows at the top are the index). Anchor edits on the LINE-START
     header; this comment quotes the headers and a substring search lands here. -->

## NEXT ACTION (rewritten s<N>, <date>)
<The task in the first sentence. Complete sentences. Every file, commit or
register id with a clause saying what it is. Only what the next session acts on.>

## Waiting-on-owner (same ids as BACKLOG §OWNER; the sweep keys on this section)
- `OWNER#<slug>` — <the ask> — escalate-by <stage or date> — NEW s<N> | RULED <date>

## Waiting-on-judgment (parked by a build session: item · why judgment-grade · what it blocks)

## HOLD
Status: LIFTED | ACTIVE (declared by the owner <date>)
Queue (in order): 1. ... 2. ...
PARKED (never started under the hold): ...

## STANDING (facts and rules that outlive a session — edit IN PLACE; name the source session)
- Process-rule incubator: <rule> (bit s<N>; bites: 1)
- Last documentation-checklist run: s<N> <date>

## Recent sessions (newest first; each block opens with `<!-- session s<N> -->`)
<!-- session s<N> -->
### s<N> — <date hh:mm → hh:mm tz> (<tier> <effort>; <attended | UNATTENDED under the runner, row <id>>) — <headline>
- signals: <the line verbatim>
- **What landed:** ...
- **Defaults taken:** ...
- **Recorded, not fixed:** ...
- **Tests / verify:** `VERIFY PASS (<N> gates)` <start → end>
- **Review:** SELF-REVIEW only | SELF-REVIEW + <n> of <m> agents returned (<findings>) | agents DOWN
- **Wrap:** clean | mid-milestone (see NEXT ACTION)

## Session log (append-only; NEWEST FIRST; insert directly below the separator)
| Date | Did | Commits | Verify |
|---|---|---|---|
| <date> s<N> | **<headline> (`<register id>` → done; <tier> <effort>, <launch mode>).** <summary>. Review: <tally>. | <hash> or (hash: next session) | `VERIFY PASS (<N> gates)` <start → end> |

## Fresh traps (graduate to CLAUDE.md at a stage gate, then delete here)
```

## Appendix E. Register template

`docs/execution/BACKLOG.md` (the index):

```markdown
# BACKLOG — the durable open-items register
How-to: (1) every row has a stable id `<SOURCE>#<slug>`; (2) schema: id · essence ·
source (file+anchor) · blocks · routing · size · deps · escalate-by · status;
(3) status: open → in-plan <lane> → done <commit/session> | refuted <file:line
why> | merged → <id>; never deleted or renumbered; (4) OWNER rows mirror
SESSION_STATE §Waiting-on-owner under the same id; (5) a row you touch in a
commit is part of that commit; (6) routing ∈ BUILD · JUDGMENT · EXPLORE · OWNER ·
CALENDAR; (7) files split at 225,000 B: closed rows → *-CLOSED.md, in-plan rows
→ *-inplan.md, then by id-prefix subject, then a numbered successor.
Count: `cat backlog/*.md BACKLOG.md | grep -c '^- \*\*'` (tree must match HEAD).

## Where the rows are
| File | Holds |
| backlog/BUILD-inplan.md | rows in the frozen plan's lanes, in lane order |
| backlog/BUILD-open.md | open build-tier rows |
| backlog/JUDGMENT-open.md | open judgment-tier rows |
| backlog/OWNER-open.md | open owner asks (escalate-by on every row) |
| backlog/OWNER-ruled.md | ruled asks, verbatim ruling + `put:` clause |
| backlog/EXPLORE.md | whole-session investigations |
| backlog/CLOSED.md | done / refuted / merged |

## §OWNER router
| id | ask | escalate-by | status |
```

Row shape (one line, in a section file):

```markdown
- **`HYG#ledger-roll-tool`** — the ledger roll refuses silently at `--keep 10` — source `scripts/session_state_roll.py:12` — blocks every session end past 200 KB — BUILD — S — deps: none — escalate-by: S1 — **open** (s41)
```

## Appendix F. Decision record template

```markdown
# ADR-NNNN: <noun phrase>
Status: proposed | accepted | rejected | deprecated | superseded by ADR-MMMM
  (amendments flagged here: "corrected in minting sitting s<N>" · "correction note <date>")
Date: <YYYY-MM-DD> (s<N>)

## Context
<Value-neutral. The forces. Extraction facts are evidence, not decisions.>

## Options considered
1. <option> — rejected because <reason>   (at least one real rejected option)
2. <option> — chosen

## Decision
We will <...>.

## Consequences
- <benefit>
- <downside>   (at least one)

## Enforcement
<one line: the gate, lint, or review step that makes the decision hold>

## Correction note — <date> s<N>   (append-only; the decision unchanged)
```

Lint at mint, flip and freeze: number matches filename · sections in order · valid Status · body unchanged since minting (git) · referenced somewhere · a real rejected option · a downside.

## Appendix G. Ladder skeleton

```bash
#!/usr/bin/env bash
# scripts/verify.sh — the gate ladder. ONE script, ordered gates, stops at the
# first red, prints exactly one pass string. Read the result from the log, never
# from an exit code. Known-RED / known-GREEN recipe for each gate in its comment.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export GOFLAGS=-p=1            # or the language's serial flag
LOG=${VERIFY_LOG:-/tmp/verify.$$.log}; exec > >(tee -a "$LOG") 2>&1
PASS=(); t0=$(date +%s)
gate() { echo "=== gate $1: $2 ($(date +%T))"; PASS+=("$2"); }

# 1 build + format + generated-code currency
#   RED control: introduce a syntax error → stops here. GREEN: revert.
gate 1 build
make build
# format tool self-test: it MUST flag a deliberately misformatted probe
probe=$(mktemp --suffix=.src); printf 'bad  formatting\n' > "$probe"
if format_check "$probe"; then echo "format tool cannot fail — refusing"; exit 1; fi; rm -f "$probe"
format_check ./src ./cmd
# stray guard: any source outside the scanned roots turns the gate red
stray=$(find . -name '*.ext' -not -path './src/*' -not -path './cmd/*' | head -1)
[ -z "$stray" ] || { echo "stray source outside scanned roots: $stray"; exit 1; }
generated_up_to_date_check

# 2 tests — serial, uncached, integration REQUIRED (not skippable), own fixtures
#   RED control: flip an assertion. GREEN: revert.
gate 2 test
seed_fixtures                          # the ladder runs its own prerequisites
run_tests --serial --no-cache --require-integration

# 3 differential crosscheck (API vs direct query) — RED: break one mapping
gate 3 crosscheck
./scripts/crosscheck.sh

# 4 invariants vs a stamped baseline — RED: exception count above the accepted run
gate 4 invariants
./bin/app check-invariants --baseline auto      # auto = newest run stamped "baseline accepted: <cause>"

# 5 goldens — masked volatile fields, frozen clock — RED: any unmasked drift
gate 5 goldens
./scripts/goldens_diff.sh

# 6 client — the typecheck that really checks, tests, build
gate 6 web
( cd web && npm run typecheck && npx vitest run && npm run build )

echo "VERIFY PASS (${#PASS[@]} gates): ${PASS[*]}  ($(( $(date +%s) - t0 )) s)"
```

Detached launch, from a session:

```bash
MARK=/tmp/verify.done; rm -f "$MARK"
while pgrep -x app.test >/dev/null; do sleep 10; done     # no peer test run (rule 8); -x, never -f
setsid nohup bash -c 'bash scripts/verify.sh; echo "rc=$?" > /tmp/verify.done' >/tmp/verify.out 2>&1 &
# poll the MARKER and the runner's liveness (ps -eo pid,etime,comm, keyed on the runner's PID);
# read the pass string from /tmp/verify.out
```

## Appendix H. Signals skeleton

```bash
#!/usr/bin/env bash
# scripts/signals_check.sh — read-only, ≤60 s. The definition is docs/STANDING_SIGNALS.md §2;
# this script implements it and changes no definition. rc 0 = all OK · 2 = a RED ·
# 1 = could not look (NEVER a green). Prints ONE line: counts and classes, never ids.
set -u
red=(); looked=0; unable=()
check() { name=$1; shift; if out=$("$@" 2>&1); then looked=$((looked+1)); case "$out" in RED*) red+=("$name");; esac; else unable+=("$name"); fi; }

check scheduler  bash -c 'for j in nightly-build nightly-report keepalive; do crontab -l | grep -q "$j" && [ -x "scripts/$j.sh" ] || { echo RED; exit 0; }; done; echo OK'
check lanes      bash -c 'f=logs/nightly-build.last; [ -f $f ] || { echo RED; exit 0; }; age=$(( ($(date +%s) - $(stat -c %Y $f))/3600 )); [ $age -lt 26 ] && grep -q PASS $f && echo OK || echo RED'
check liveness   bash -c 'curl -fsS -m 5 http://localhost:${PORT:-8081}/healthz >/dev/null && echo OK || echo RED'
check errors     bash -c 'n=$(grep -c " ERROR " logs/server.log 2>/dev/null || echo 0); prev=${PREV_ERRORS:-0}; [ "$n" -le "$prev" ] && echo "OK $n" || echo "RED $n"'
check disk       bash -c 'u=$(df --output=pcent . | tail -1 | tr -dc 0-9); [ $u -lt 90 ] && echo OK || echo RED'
check backups    bash -c 'f=$(ls -t backups/*.tar.zst 2>/dev/null | head -1); [ -n "$f" ] && [ $(( ($(date +%s) - $(stat -c %Y $f))/3600 )) -lt 30 ] && echo OK || echo RED'

n=$((looked))
if [ ${#unable[@]} -gt 0 ]; then echo "signals: COULD-NOT-LOOK ${unable[*]} · ${n} looked"; push "signals could not look"; exit 1; fi
if [ ${#red[@]} -gt 0 ];    then echo "signals: RED ${red[*]} · ${n} checks"; push "signals RED: ${red[*]}"; exit 2; fi
echo "signals: ${n} OK"; exit 0
```

Rules: a push fires on RED or exit 1 only; every scheduled job joins this inventory in the commit that installs it; on RED a session records a register row tagged SIGNAL with an escalate-by and fixes nothing mid-milestone.

## Appendix I. Hot-files check and the roll contract

`scripts/hot_files_check.sh`: prints every hot file's size against its line (ledger 200,000 B; each register file, the map and the plan 225,000 B; all derived from the tool's 262,144 B read ceiling) and exits 1 when any line is crossed. It also exits 1 when the ledger holds more than 12 session blocks (the source project's version checked bytes only; check both).

`scripts/session_state_roll.py` contract:

- Dry-run by default; `--write` applies; `--check` exits 1 when a roll is due; `--keep N` forces one pass.
- Default keep DERIVES ITSELF: the largest keep at or above a floor of 4 blocks that leaves at least 45,000 B of headroom under the line, in as many passes as the archive cap requires. A fixed default refused silently at every roll it attempted.
- Moves the oldest hot blocks together with their log rows, matched by session number, to `docs/execution/archive/SESSION_STATE_s<a>-s<b>.md`; appends to the newest archive while it is under 230,000 B, otherwise opens a new one; writes a FROZEN header with the moved log rows at the top as the index.
- Refuses to exceed the cap; refuses if the target already exists (a half-applied run).
- Validates structure first and names the repair on failure. Exit codes: 0 fine · 1 roll due (under `--check`) · 2 structure broken · 3 the block roll alone cannot clear the line (sweep superseded NEXT ACTION paragraphs).
- Deletes nothing; edits no archive.

## Appendix J. `/factory-continue` command

`.claude/commands/factory-continue.md`:

```markdown
---
description: Resume the <Project> build from where the last session stopped
argument-hint: [--status-only]
---
# Continue the build
Resume autonomous build work on the active stage.

## Automatic actions
1. **Read, in order:** `bash scripts/signals_check.sh` FIRST (paste its line; rc 1
   is never a green; RED = record, never fix mid-milestone) · docs/MAP.md §2 ·
   SESSION_STATE.md (NEXT ACTION, §HOLD, blockers, Waiting-on-owner) · docs/PLAN.md
   (the milestone NEXT ACTION names: Design, Verify, DoD) · docs/execution/
   BACKLOG.md ONLY when NEXT ACTION is blocked or done.
2. **Determine the task (first match wins):**
   a0. §HOLD ACTIVE ⇒ first unfinished hold-queue item; empty ⇒ stop, tell the owner
       (under the runner: write HALT_HOLD_EMPTY to your done file).
   a. A recorded blocker now resolved ⇒ clear it, take its milestone.
   b. NEXT ACTION's tag matches this session's tier, or it is a Lane-0 row firing by
      its trigger ⇒ take it verbatim. A JUDGMENT row met by a build session is not
      started and not decided. A BUILD row met by a judgment session is skipped for
      the first open JUDGMENT row. OWNER rows are startable by no session. An EXPLORE
      row is a whole session.
   c. NEXT ACTION done-but-unrecorded (git log + verify evidence) ⇒ record it, run
      the exit-gate check, advance.
   d. All PLAN.md milestones done ⇒ catch missed flips (a calendar gate may lag:
      record a blocker), archive PLAN.md → docs/execution/archive/PLAN_<stages>.md,
      run /factory-spec. An interstitial is never archived on exhaustion.
   e. Blocked on the owner or mismatched by tag ⇒ never idle: first open lane row for
      this tier in printed order, then BACKLOG by routing; say so in the log row.
   f. Nothing above applies ⇒ stop with a named reason (under the runner:
      HALT_PLAN_COMPLETE or HALT_NEEDS_OWNER in your done file).
3. **Sanity-check anchors:** re-verify every file:line the milestone cites; read the
   printed line against its claim. Mismatch ⇒ fix the spec per MAP §10, proceed.
4. **Work exactly ONE milestone.** Then: ladder green → adversarial pass (the self-run
   three-lens floor; agents additive at zero wall-clock; record the tally) → commit
   by explicit paths with the register flips and doc fixes in the same commit →
   SESSION_STATE.md (rewrite NEXT ACTION; block; log row below the separator;
   hot_files_check; `grep -c '^## NEXT ACTION'` = 1) → exit-gate check of the
   milestone's own stage; if met, the stage-gate commit in THIS session. A second
   small milestone MAY follow; never a third.

## PARK, DON'T DECIDE
A build session that meets a judgment-grade decision its task did not pre-decide
records it in SESSION_STATE §Waiting-on-judgment (item · why · what it blocks) and
takes another row or wraps. A judgment session PRE-DECIDES such items into NEXT
ACTION; its report never says "escalate X next session".

## Options
`--status-only`: report stage, milestone progress, blockers, waiting-on-owner, last
verify result. Change nothing.

## LOOP MODE (`/loop /factory-continue`)
Each firing = one full pass. Stop (no re-arm) when the phase is done, when context
is near 500K tokens (clean wrap by 600K), or when the hold queue is empty. Never
schedule idle wakeups.

## DO NOT
Ask permission to continue · re-plan the stage · re-open locked decisions ·
re-summarize the project · start milestone N+1 while N is red · touch the
STOP-AND-ASK list without the owner.

## END-OF-SESSION REPORT (mandatory, last in the final message)
1. **Needs owner** — one line each, new first, standing ones with escalate-by; or "None".
2. **Next session** — NEXT ACTION in one line + alternates if blocked.
3. **Tier + effort** — explicit ("BUILD, high"); never "BUILD but escalate X".
```

## Appendix K. `/factory-spec` command

`.claude/commands/factory-spec.md`:

```markdown
---
description: Spec-freeze — write the next stage's PLAN.md (or the interstitial) from the template
argument-hint: <stage>|interstitial
---
# Freeze the next stage's execution spec
1. **Read:** MAP §2 (the target row: entry criteria, exit gate, governing docs; for an
   interstitial read §2 WHOLE and record "no row enterable" in the header) · the
   governing spec sections · the open-items files · docs/templates/PLAN_TEMPLATE.md
   · BACKLOG (lanes are drawn from it by id).
2. **Check entry criteria** for EVERY stage the plan will span, including that every
   stage-start decision is resolved. Unresolved ⇒ a one-paragraph decision brief in
   SESSION_STATE §Waiting-on-owner, and that path stops. A fired time-fused owner
   trigger ⇒ a dated ESCALATION line first in Waiting-on-owner AND first in NEXT
   ACTION; mark the stage blocked(<item>).
3. **Archive the exhausted PLAN.md** by `git mv` to docs/execution/archive/
   PLAN_<stages>.md (interstitial: PLAN_interstitial-<freeze-date>.md). FIRST move
   (not copy) every still-open lane row and open-items row to BACKLOG as `open` or
   `in-plan`. Un-moved rows fail this freeze.
4. **Write the new PLAN.md from the template.** Re-verify and DATE every file:line
   anchor at write time. Every milestone: a tier tag, *Verify* (exact commands +
   expected output), *DoD* (incl. the same-commit register flip). Oracle milestones
   name the comparison and define "match".
5. **High-stakes stage:** set NEXT ACTION = "Owner: skim PLAN.md §Locked decisions +
   §DoD". Otherwise proceed.
6. **Run the documentation checklist** (decision-record lint A1–A7, diagram currency,
   reference health) and leave the token `DOC-CHECK run: <scope> · <findings>` in
   the MAP changelog line and the plan's commit trail. No token ⇒ the freeze is
   incomplete and the next session reports it.

## DO NOT
Freeze with silent unknowns (every open question is resolved, routed to extraction,
or in the honesty register with an owner) · pre-write later stages' specs · cite a
STALE source.
## DO
Make the spec executable by a session with zero prior context. Update MAP §2 only
when the previous stage's exit evidence is in the same commit.
```

## Appendix L. The queue

### L.1 Directory layout (outside the repo, its own git repo)

```
~/<project>-queue/
  WORK_QUEUE.md            # the table (§7.2) plus dated prose around it
  RECIPE.md                # queue format, pack procedure, machinery changelog
  AUTOPLAN.conf            # refill=0|1 max_continue=2 curator=0 (strict key=value, no trailing comments)
  RECONCILED_AT            # one timestamp line; stamped only by an evidence-backed reconcile
  STOP                     # graceful stop when present
  driver.lock  burn.lock   # one pid each
  driver.sh                # the runner (§7.7)
  run_watched.sh           # per-row launcher, TUI in a tmux window
  run_one.sh               # per-row launcher, headless
  supervise.sh  SUPERVISE.md
  template/CONTINUE_PROMPT.md  template/STANDING_RULES.md
  prompts/<id>.md          # kickoff and maintenance prompts (never start with "/")
  runs/<ts>_<mode>/        # driver.log · rendered/<id>.md · queue.done · SUMMARY.md
  logs/<id>.{session_id,meta,done,runner.log,final.md,commits.txt}
```

### L.2 Queue table

```markdown
| pos | id | type | mode | model/effort | prompt | wrap | pre | status |
|---|---|---|---|---|---|---|---|---|
| 10 | continue-a | continue | watched | build/high | - | commit | - | ready |
| 20 | continue-b | continue | watched | build/high | - | commit | - | ready |
| 30 | docs-digest | kickoff | watched | build/high | prompts/docs-digest.md | commit | - | ready |
| 40 | queue-reconcile | maintenance | watched | build/high | prompts/maint-queue-reconcile.md | maint | nocron | ready |
| 90 | recon-perf | burn | watched | build/high | prompts/recon-perf.md | recon:analysis/recon-perf | excl | ready |
```

### L.3 Continue prompt template

```markdown
Run the session-start protocol in CLAUDE.md exactly, with /factory-continue. The
repo schedules the task: NEXT ACTION, the hold, the plan's lanes, the register.
Wrap in full: ladder green, explicit-path commit, SESSION_STATE block and log row.

### RUNNER ADDENDUM
Nobody is at the keyboard. Never ask a question through the question tool: take the
recommended default and lodge an OWNER row. A STOP-AND-ASK item or a red you cannot
fix this session ⇒ the mid-milestone wrap: commit nothing, state one BLOCKED line,
and write your halt class into your done file:
  sid=$(ls -t ~/<project>-queue/logs/*.session_id | head -1)   # newest FILE, not dir
  echo HALT_<CLASS> > "${sid%.session_id}.done"; touch ~/<project>-queue/STOP
Classes: HALT_HARD_STOP · HALT_RED_GATE · HALT_NEEDS_OWNER · HALT_PLAN_COMPLETE ·
HALT_HOLD_EMPTY · HALT_PRECONDITION. Wait inside your turn if you must wait (the
scheduled-job window is <HH:MM>–<HH:MM>; repeated `sleep 600`); a watched session
that ends its turn is never resumed. Never match a process by a pattern this prompt
contains. End your turn when your wrap is complete.

### PRE-ANSWERED (the owner's rulings for tonight, verbatim)
- <question> → <answer> (put: <where it now lives>)

### LIVE FACTS
(injected by the driver at launch; never author this block)
```

### L.4 Per-row launcher (watched)

```bash
#!/usr/bin/env bash
# run_watched.sh <id> <rendered-prompt> <model-alias> <effort> <logdir>
set -u
id=$1; prompt=$2; alias=$3; effort=$4; logdir=$5; done="$logdir/$id.done"
case "$alias" in build) model=<build-model-id>;; judgment) model=<judgment-model-id>;; *) echo HALT_BADMODEL > "$done"; exit 2;; esac
cd /path/to/repo || { echo HALT_BADCWD > "$done"; exit 2; }
unset ANTHROPIC_API_KEY; for v in $(env | grep -o '^CLAUDE[A-Z_]*'); do unset "$v"; done
[ -f ~/.secrets/<project>.env ] && . ~/.secrets/<project>.env || { echo HALT_NOSECRETS > "$done"; exit 2; }
sid_file="$logdir/$id.session_id"; [ -f "$sid_file" ] || uuidgen > "$sid_file"; SID=$(cat "$sid_file")
echo "start=$(date -Is)" > "$logdir/$id.meta"
( # model verifier: first assistant record within 3 min, else kill OUR child only
  for i in $(seq 1 36); do sleep 5; t=$(ls -t ~/.claude/projects/*/"$SID".jsonl 2>/dev/null | head -1)
    [ -n "$t" ] && m=$(grep -m1 '"role":"assistant"' "$t" | grep -o '"model":"[^"]*"' | head -1) && [ -n "$m" ] && {
      case "$m" in *"$model"*) echo "model verified $m" >> "$logdir/$id.runner.log"; exit 0;; *) echo HALT_MODEL_MISMATCH > "$done"; pkill -P $$ -x claude; exit 0;; esac; }
  done ) &
claude "$(cat "$prompt")" --session-id "$SID" --model "$model" --effort "$effort" <permission-mode-flag>
rc=$?
echo "end=$(date -Is)" >> "$logdir/$id.meta"; echo "rc=$rc" >> "$logdir/$id.meta"
[ -s "$done" ] || echo "EXITED rc=$rc" > "$done"
```

### L.5 Driver skeleton

```bash
#!/usr/bin/env bash
# driver.sh [--burn N | --status | --dry-run | --review [run]]
set -u
Q=~/<project>-queue; cd "$Q"; REPO=/path/to/repo; TMUX=<project>-queue
PASS_CAP=${PASS_CAP:-8}; WATCHED_POLL=60; IDLE_MIN=10; EXIT_GRACE_MIN=10; MAX_HOURS=8; LIMIT_WAIT_PCT=85
run=runs/$(date +%Y%m%d_%H%M%S)_serial; mkdir -p "$run/rendered" logs
log() { echo "$(date +%T) $*" | tee -a "$run/driver.log"; }
rows() { awk -F'|' 'NF>=10 && $2 ~ /^[ ]*[0-9]+[ ]*$/ {print}' WORK_QUEUE.md | sort -t'|' -k2 -n; }
cell() { echo "$1" | awk -F'|' -v c="$2" '{gsub(/^ +| +$/,"",$c); print $c}'; }
set_status() { tmp=$(mktemp); awk -F'|' -v id="$1" -v st="$2" 'BEGIN{OFS="|"} NF>=10 && $3 ~ id {$10=" "st" "} {print}' WORK_QUEUE.md > "$tmp" && mv "$tmp" WORK_QUEUE.md; }
dirty() { git -C "$REPO" status --porcelain | grep -v ORCHESTRATOR_RESULTS.md | grep -q .; }
finish() { echo "$1" > "$run/queue.done"; }
summary() { { echo "# SUMMARY $run"; echo "outcome: $(head -1 "$run/queue.done" 2>/dev/null || echo UNKNOWN)"; echo "attention:"; grep -E 'HALT|FAILED|VIOLATED' "$run/driver.log" | sed 's/^/- /'; rows | awk -F'|' '{print "- "$3" "$10}'; } > "$run/SUMMARY.md"; }
live_facts() { since=$(cat RECONCILED_AT 2>/dev/null || date -d '-30 days' -Is)
  { echo "### LIVE FACTS ($(date -Is))"; echo "HEAD: $(git -C "$REPO" rev-parse --short HEAD)";
    echo "LANDED SINCE $since:"; git -C "$REPO" log --oneline --since="$since" | head -40 | cut -c1-200;
    echo "dirt: $(git -C "$REPO" status --porcelain | grep -c '^ M') tracked, $(git -C "$REPO" status --porcelain | grep -c '^??') untracked";
    echo "disk free: $(df -h "$REPO" | awk 'NR==2{print $4}')"; echo "claude processes: $(pgrep -xc claude)"; }; }
meter_gate() { pct=$(./meter_probe.sh 2>/dev/null || echo unknown); case "$pct" in unknown) log "meter unknown — failing closed"; sleep 600;; *) [ "$pct" -lt $LIMIT_WAIT_PCT ] || { log "meter $pct% ≥ $LIMIT_WAIT_PCT — holding"; sleep 900; meter_gate; };; esac; }
wrap_ok() { case "$1" in
  commit) [ "$(git -C "$REPO" rev-parse HEAD)" != "$2" ] && git -C "$REPO" diff --name-only "$2..HEAD" | grep -q '^SESSION_STATE.md$';;
  recon:*) [ -n "$(ls -A "$REPO/${1#recon:}" 2>/dev/null)" ];;
  maint) [ "$(git -C "$REPO" rev-parse HEAD)" != "$2" ] && bash "$REPO/scripts/hot_files_check.sh" >/dev/null && ! dirty;;
  *) false;; esac; }

# --- start ---
if [ -f driver.lock ] && kill -0 "$(cat driver.lock)" 2>/dev/null; then finish REFUSED_LOCK_HELD; summary; exit 1; fi
echo $$ > driver.lock; trap 'rm -f driver.lock; summary' EXIT
if [ -f STOP ]; then if [ -f burn.lock ] && kill -0 "$(cat burn.lock)" 2>/dev/null; then finish REFUSED_STOP_HELD; exit 1; else rm -f STOP; fi; fi
n=0; while dirty; do n=$((n+1)); [ $n -gt 360 ] && { finish REFUSED_DIRTY_TIMEOUT; exit 1; }; sleep 120; done
tmux has-session -t "$TMUX" 2>/dev/null || tmux new-session -d -s "$TMUX" -n driver

pass=0; refilled=0
while :; do
  launched=0
  while IFS= read -r row; do
    id=$(cell "$row" 3); type=$(cell "$row" 4); mode=$(cell "$row" 5); me=$(cell "$row" 6); src=$(cell "$row" 7); wrap=$(cell "$row" 8); pre=$(cell "$row" 9); st=$(cell "$row" 10)
    [ "$st" = ready ] && [ "$type" != burn ] && [ "$mode" != interactive ] || continue
    [ -f STOP ] && { finish STOPPED; exit 0; }
    git -C "$REPO" add ORCHESTRATOR_RESULTS.md 2>/dev/null && git -C "$REPO" diff --cached --quiet || git -C "$REPO" commit -qm "runner ledger" ORCHESTRATOR_RESULTS.md
    dirty && { set_status "$id" halted; log "HALT DIRTY_BEFORE_LAUNCH at $id"; ./push.sh "queue HALT dirty before $id"; finish "HALTED at $id: DIRTY_BEFORE_LAUNCH"; exit 1; }
    case ",$pre," in *,excl,*) while [ -f burn.lock ] && kill -0 "$(cat burn.lock)" 2>/dev/null; do sleep 60; done;; esac
    case ",$pre," in *,db,*) while [[ "$(date +%H%M)" > "0414" && "$(date +%H%M)" < "0616" ]]; do sleep 300; done;; esac
    meter_gate
    prehead=$(git -C "$REPO" rev-parse HEAD)
    rendered="$run/rendered/$id.md"; { [ "$src" = - ] && cat template/CONTINUE_PROMPT.md || cat "$src"; echo; live_facts; echo; echo "The owner MAY be watching. NEVER sit waiting for input. END YOUR TURN when your wrap is complete."; } > "$rendered"
    head -c1 "$rendered" | grep -q '/' && { set_status "$id" failed; log "prompt starts with / at $id"; continue; }
    set_status "$id" running; log "LAUNCH $id ($me, wrap=$wrap)"; launched=1
    tmux new-window -t "$TMUX" -n "$id" "bash $Q/run_watched.sh $id $rendered ${me%/*} ${me#*/} $Q/logs; exec sleep infinity"
    t0=$(date +%s); ok=0; verdict=
    while :; do sleep $WATCHED_POLL
      d=$(cat "logs/$id.done" 2>/dev/null || true)
      case "$d" in HALT_*) verdict=DONE_HALT; break;; esac
      if wrap_ok "$wrap" "$prehead" && ! dirty; then ok=$((ok+1)); else ok=0; fi
      over=0; [ -n "$d" ] && over=1; tmux list-windows -t "$TMUX" -F '#W' | grep -qx "$id" || over=1
      idle=$(( $(date +%s) - $(tmux display -p -t "$TMUX:$id" '#{window_activity}' 2>/dev/null || date +%s) ))
      [ $ok -ge 2 ] && { [ $over = 1 ] || [ $idle -ge $((IDLE_MIN*60)) ]; } && { verdict=WRAP_OK; break; }
      [ $over = 1 ] && [ $(( $(date +%s) - t0 )) -gt $((EXIT_GRACE_MIN*60)) ] && ! wrap_ok "$wrap" "$prehead" && { verdict=EXITED_MISS; break; }
      [ $(( $(date +%s) - t0 )) -gt $((MAX_HOURS*3600)) ] && { verdict=TIMEOUT_LIVE; break; }
    done
    log "VERDICT $id $verdict"
    case "$verdict" in
      WRAP_OK) set_status "$id" done; echo "| $(date -Is) | $id | $type | WRAP_OK | $(git -C "$REPO" rev-parse --short HEAD) | $(( ($(date +%s)-t0)/60 )) | ? | $(cat logs/$id.session_id) | $run | - |" >> "$REPO/ORCHESTRATOR_RESULTS.md"
               git -C "$REPO" commit -qm "runner ledger: $id WRAP_OK" ORCHESTRATOR_RESULTS.md;;
      DONE_HALT) set_status "$id" halted; ./push.sh "queue HALT $id: $d"; finish "HALTED at $id: $d"; exit 1;;
      TIMEOUT_LIVE) set_status "$id" halted; ./push.sh "queue HALT $id: session live after ${MAX_HOURS}h"; finish "HALTED at $id: WATCHED_TIMEOUT_SESSION_LIVE"; exit 1;;
      *) if dirty; then set_status "$id" halted; ./push.sh "queue HALT $id: dirty after wrap"; finish "HALTED at $id: DIRTY_TREE_AFTER_WRAP"; exit 1; else set_status "$id" failed; fi;;
    esac
    break   # one launch per pass; re-read the queue
  done < <(rows)
  pass=$((pass+1)); [ $pass -gt $PASS_CAP ] && { log "pass cap"; break; }
  rows | awk -F'|' '{gsub(/ /,"",$10)} $10=="ready"' | grep -q . && continue
  if [ $refilled = 0 ] && grep -qx 'refill=1' AUTOPLAN.conf 2>/dev/null; then refilled=1; ./append_continue_rows.sh "$(grep -oP '^max_continue=\K\d+' AUTOPLAN.conf)"; continue; fi
  break
done
finish COMPLETE
```

### L.6 Night launcher gates (the wrapper that starts the driver)

Before starting the driver: no launcher alive (pidfile) · no live lock · every manifest prompt present and at least its minimum size · no ladder or test run alive (exact command line) · a clean tree with the results ledger exempt · `--status` parses the queue · the signals script returns 0 · free disk above the floor. Then: clear STOP, pre-hold `db` and `nocron` rows if launching inside the blackout band, start the driver in a tmux window, and exit. At the deadline flip unlaunched rows to `deadline`; a re-arm script flips them back to `ready` and stamps `RECONCILED_AT`.

### L.7 Relaunch pre-flight (the `/factory-queue` command)

```bash
cd ~/<project>-queue
ls -dt runs/*_serial 2>/dev/null | head -1; ls -dt runs/*_burn 2>/dev/null | head -1
for l in driver.lock burn.lock; do [ -f "$l" ] && echo "$l: pid $(cat $l) $(kill -0 $(cat $l) 2>/dev/null && echo LIVE || echo stale)"; done
ls STOP 2>/dev/null
```

Then apply §7.8's seven rules to each listed run's `SUMMARY.md`; launch with `tmux new-window -t <project>-queue -n "driver-$(date +%H%M)" 'bash ~/<project>-queue/driver.sh; ec=$?; echo "[driver exited rc=$ec]"; exec sleep infinity'`; report the run directory, the attach hint, and the previous run's outcome line; return immediately. `status`, `review` and `dry-run` run the driver's zero-launch subcommands inline. `stop` touches the STOP file.

### L.8 Supervisor events

`LAUNCHED <id>` · `MODEL_VERIFIED <id> <model>` · `LAUNCH_CHECK OK|HOLDING|PROBLEM` · `DRIVER <line>` (filtered) · `STALL <id>` (transcript quiet 30 min, wrap missing) · `COMPLETE <outcome>`. Permitted responses and prohibitions are in §7.10.

## Appendix M. Session block, log row and final report

The block and row shapes are in Appendix D. The final-message report block:

```
## Needs owner
- OWNER#<slug> — <ask> — escalate-by <trigger> (NEW)
- None

## Next session
NEXT ACTION: <one line>. Alternates if blocked: <lane rows>.

## Tier + effort
BUILD, high.  (Under a hold: next queue item <id>; <k> items remain.)
```

## Appendix N. Glossary

| Term | Meaning |
|---|---|
| Blast radius | What a session changed: code, canon, or a census the register is re-keyed from; determines whether the review pass is owed |
| Canon | A document rated CANON in the trust table; the only kind a build milestone may cite |
| Done-but-unrecorded | A commit that delivered NEXT ACTION before the record was written; detected at orient |
| Done file | The per-row file a session or launcher writes to end the row; a `HALT_*` class in it halts the queue |
| Exit gate | A conjunction of checkable evidence in the map's stage row; met means flip |
| Freeze | The procedure that produces a plan; the only way a stage starts |
| Hold | Ledger state that parks judgment-tier work behind a pre-vetted build-tier queue |
| Interstitial | The plan file written when no stage is enterable; displaced, never exhausted |
| Lane 0 | The read-only calendar and stop-the-line table; preempts by its own triggers |
| Ladder | The gate script; one pass string |
| Landing hash | The hash of the commit that delivered a milestone; stamped into the log row immediately or by the next session |
| LIVE FACTS | The block of readings the runner injects into a prompt at launch; never authored |
| PARK, DON'T DECIDE | The build-tier rule for judgment-grade decisions |
| PRE-DECIDE | The judgment-tier duty to leave a decision-free NEXT ACTION |
| Register | The durable open-items file set, keyed by stable ids |
| Signals | The read-only session-start probe with three exit codes |
| Stage-gate commit | The commit that flips a map row with its evidence |
| Wrap | The session's end-of-work deliverable (ladder, commit, record); also the mid-milestone stop that commits nothing and leaves the dirty tree as the hand-off |
| Wrap gate | The runner's check that a row's declared deliverable exists; the verdict, never the exit code |
