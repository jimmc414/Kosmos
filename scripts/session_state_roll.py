#!/usr/bin/env python3
"""Roll the oldest session blocks (and their log rows) out of SESSION_STATE.md.

Contract (process Appendix I):
  dry-run by default; --write applies; --check exits 1 when a roll is due; --keep N forces one pass.
  The default keep DERIVES ITSELF: the largest keep at or above a floor of 4 blocks that leaves at
  least 45,000 B of headroom under the 200,000 B line, in as many passes as the archive cap
  requires. Moves the oldest hot blocks together with their log rows, matched by session number,
  to docs/execution/archive/SESSION_STATE_s<a>-s<b>.md; appends to the newest archive while it is
  under 230,000 B, otherwise opens a new one; writes a FROZEN header with the moved log rows at
  the top as the index. Refuses to exceed the cap; refuses if the target already exists (a
  half-applied run). Validates structure first and names the repair. Deletes nothing; edits no
  archive body. Exit: 0 fine · 1 roll due (under --check) · 2 structure broken · 3 the block roll
  alone cannot clear the line (sweep superseded NEXT ACTION paragraphs by hand).
"""
import argparse
import datetime
import pathlib
import re
import sys

LEDGER = pathlib.Path("SESSION_STATE.md")
ARCHIVE_DIR = pathlib.Path("docs/execution/archive")
LINE = 200_000
HEADROOM = 45_000
FLOOR = 4
MAX_BLOCKS = 12
ARCHIVE_CAP = 230_000
BLOCK_OPEN = re.compile(r"^<!-- session s(\d+) -->$")
ROW = re.compile(r"^\| [^|]*\bs(\d+)\b[^|]*\|")


def fail(code, msg):
    print(msg)
    sys.exit(code)


def parse(text):
    lines = text.split("\n")
    idx = {}
    for name in ("## NEXT ACTION", "## Recent sessions", "## Session log", "## Fresh traps"):
        hits = [i for i, l in enumerate(lines) if l.startswith(name)]
        if len(hits) != 1:
            fail(2, f"structure: '{name}' appears {len(hits)} times (want 1): repair the header")
        idx[name] = hits[0]
    rs, sl, ft = idx["## Recent sessions"], idx["## Session log"], idx["## Fresh traps"]
    if not rs < sl < ft:
        fail(2, "structure: sections out of order (Recent sessions < Session log < Fresh traps)")
    # blocks: from each <!-- session sN --> to the next opener or the Session log header
    openers = [i for i in range(rs, sl) if BLOCK_OPEN.match(lines[i])]
    blocks = []
    for k, start in enumerate(openers):
        end = openers[k + 1] if k + 1 < len(openers) else sl
        n = int(BLOCK_OPEN.match(lines[start]).group(1))
        blocks.append((n, start, end))
    # log table: header, separator, rows
    sep = None
    for i in range(sl, ft):
        if lines[i] == "|---|---|---|---|":
            sep = i
            break
    if sep is None:
        fail(2, "structure: no '|---|---|---|---|' separator under '## Session log'")
    rows = {}
    for i in range(sep + 1, ft):
        m = ROW.match(lines[i])
        if m:
            rows.setdefault(int(m.group(1)), []).append(i)
    return lines, blocks, rows, sep


def newest_archive():
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    cands = sorted(ARCHIVE_DIR.glob("SESSION_STATE_s*-s*.md"))
    return cands[-1] if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--keep", type=int)
    a = ap.parse_args()
    text = LEDGER.read_text(encoding="utf-8")
    size = len(text.encode("utf-8"))
    lines, blocks, rows, sep = parse(text)
    nblocks = len(blocks)
    due = size > LINE or nblocks > MAX_BLOCKS
    print(f"ledger: {size} B / {LINE} B · {nblocks} blocks / {MAX_BLOCKS}")
    if a.check:
        sys.exit(1 if due else 0)
    if not due and a.keep is None:
        print("no roll due")
        return
    # blocks are newest-first in the file; the oldest are at the end
    order = sorted(blocks, key=lambda b: b[0])  # ascending session number
    if a.keep is not None:
        keep = max(a.keep, 0)
    else:
        keep = nblocks
        while keep > FLOOR:
            moved = order[: nblocks - keep]
            freed = sum(len("\n".join(lines[s:e]).encode()) + 1 for _, s, e in moved)
            freed += sum(len(lines[i].encode()) + 1 for n, _, _ in moved for i in rows.get(n, []))
            if size - freed <= LINE - HEADROOM and nblocks - len(moved) <= MAX_BLOCKS:
                break
            keep -= 1
        if keep < FLOOR:
            keep = FLOOR
    moved = order[: max(nblocks - keep, 0)]
    if not moved:
        fail(3, "the block roll alone cannot clear the line: sweep superseded NEXT ACTION paragraphs by hand")
    freed = sum(len("\n".join(lines[s:e]).encode()) + 1 for _, s, e in moved)
    freed += sum(len(lines[i].encode()) + 1 for n, _, _ in moved for i in rows.get(n, []))
    if a.keep is None and size - freed > LINE:
        fail(3, f"rolling to keep={keep} leaves {size - freed} B > {LINE} B: sweep NEXT ACTION by hand")
    lo, hi = moved[0][0], moved[-1][0]
    target = newest_archive()
    if target is None or target.stat().st_size + freed > ARCHIVE_CAP:
        target = ARCHIVE_DIR / f"SESSION_STATE_s{lo}-s{hi}.md"
        if target.exists():
            fail(2, f"refusing: {target} already exists (a half-applied roll?)")
        new_archive = True
    else:
        new_archive = False
    print(f"roll: move {len(moved)} blocks s{lo}..s{hi} ({freed} B) → {target} ({'new' if new_archive else 'append'}); keep {keep}")
    if not a.write:
        print("dry run; add --write to apply")
        return
    moved_rows = [lines[i] for n, _, _ in sorted(moved, key=lambda b: -b[0]) for i in rows.get(n, [])]
    moved_blocks = ["\n".join(lines[s:e]).rstrip("\n") for _, s, e in sorted(moved, key=lambda b: -b[0])]
    header = ""
    if new_archive:
        header = (f"# SESSION_STATE archive s{lo}–s{hi} — FROZEN {datetime.date.today().isoformat()}\n"
                  "Rolled out of SESSION_STATE.md by scripts/session_state_roll.py; never edited. "
                  "The log rows below are the index; the blocks follow.\n\n"
                  "| Date | Did | Commits | Verify |\n|---|---|---|---|\n")
    chunk = header + "\n".join(moved_rows) + "\n\n" + "\n\n".join(moved_blocks) + "\n"
    with target.open("a", encoding="utf-8") as fh:
        fh.write(chunk)
    drop = set()
    for n, s, e in moved:
        drop.update(range(s, e))
        drop.update(rows.get(n, []))
    kept = [l for i, l in enumerate(lines) if i not in drop]
    LEDGER.write_text("\n".join(kept), encoding="utf-8")
    # post-roll assertion
    t2 = LEDGER.read_text(encoding="utf-8")
    parse(t2)
    print(f"rolled: ledger now {len(t2.encode())} B; archive {target} {target.stat().st_size} B")
    if target.stat().st_size > ARCHIVE_CAP:
        fail(2, f"archive {target} exceeds {ARCHIVE_CAP} B after the roll: open a new one next time (the cap check failed)")


if __name__ == "__main__":
    main()
