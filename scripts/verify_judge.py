#!/usr/bin/env python3
"""Judge a gate-2 pytest log against the stamped baseline (ADR-0002).

Usage:
    python scripts/verify_judge.py <gate2.log> <baseline.txt>
    python scripts/verify_judge.py <gate2.log> <baseline.txt> --write-baseline "<cause>"

The key is the pytest node id; the FAILED/ERROR status is informational. Exit 0 when every
red node id in the run is in the baseline; exit 1 when any is not (each is printed as
NEW RED); exit 2 when the log carries no pytest summary line (the run did not complete,
which is never a result); exit 3 when --write-baseline is given a blank cause.
--write-baseline prints the delta against the existing baseline FIRST, then rewrites the
file with a stamp line naming the cause, the date and HEAD.

Known-RED control: a log holding `FAILED tests/x.py::test_new` not in the baseline -> exit 1.
Known-GREEN control: the same log with that line in the baseline -> exit 0.
"""
import datetime
import re
import subprocess
import sys

LINE = re.compile(r"^(FAILED|ERROR) (\S.*?)(?: - .*)?$")
SUMMARY = re.compile(r"^=+ .*\b(passed|failed|error|errors|skipped)\b.* in [0-9.]+s")


def parse_log(path):
    ids, summary = {}, None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            m = LINE.match(line)
            if m:
                ids[m.group(2)] = m.group(1)
            if SUMMARY.match(line):
                summary = line.strip("= ").strip()
    return ids, summary


def parse_baseline(path):
    ids, stamp = {}, None
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except FileNotFoundError:
        return ids, None
    for line in lines:
        if line.startswith("# baseline accepted:"):
            stamp = line[2:]
        if line.startswith("#") or not line.strip():
            continue
        status, _, node = line.partition(" ")
        ids[node] = status
    return ids, stamp


def main(argv):
    write_cause = None
    if "--write-baseline" in argv:
        i = argv.index("--write-baseline")
        write_cause = argv[i + 1] if i + 1 < len(argv) else ""
        del argv[i : i + 2]
    if len(argv) != 2:
        print(__doc__)
        return 3
    log, base = argv
    run_ids, summary = parse_log(log)
    if summary is None:
        print(f"gate 2: no pytest summary line in {log}; the run did not complete, not a result")
        return 2
    base_ids, stamp = parse_baseline(base)
    new = sorted(n for n in run_ids if n not in base_ids)
    retired = sorted(n for n in base_ids if n not in run_ids)
    print(f"gate 2 summary: {summary}")
    print(
        f"gate 2 judge: {len(run_ids)} red node ids in the run; {len(base_ids)} in the baseline"
        f" ({stamp or 'no stamp'}); {len(new)} new; {len(retired)} retired (now passing or gone)"
    )
    for n in retired[:60]:
        print(f"  retired: {n}")
    for n in new:
        print(f"  NEW RED: {run_ids[n]} {n}")
    if write_cause is not None:
        if not write_cause.strip():
            print("refusing: --write-baseline needs a non-blank cause")
            return 3
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        with open(base, "w", encoding="utf-8") as fh:
            fh.write(
                f"# baseline accepted: {write_cause.strip()} "
                f"({datetime.date.today().isoformat()}, HEAD {head})\n"
            )
            fh.write(f"# {len(run_ids)} red node ids; run summary: {summary}\n")
            fh.write(
                '# Moves only through `bash scripts/verify.sh --accept-baseline "<cause>"` '
                "(ADR-0002). Key = node id; the status is informational.\n"
            )
            for n in sorted(run_ids):
                fh.write(f"{run_ids[n]} {n}\n")
        print(f"baseline written: {base} ({len(run_ids)} node ids)")
        return 0
    return 1 if new else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
