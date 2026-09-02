"""record_annotation.py - write MODEL annotations into data/annotations.json.

Separate file, separate provenance. These are never merged into the human
judgements in data/connections.json: the whole point of measuring agreement
is that the two sets stay distinguishable.

ORIGINAL DOC BELOW
"""
"""

    .venv/bin/python ml/record.py "1:y 2:n 3:s ..."

y = connected, n = not connected, s = cannot tell. Positions are 1-indexed
into ml/out/review_queue.json. The older 1-5 type keys still work.

Same rules as the web tool: a rejection is recorded rather than discarded,
and re-judging a pair replaces the old verdict instead of duplicating it.
"""
import json
import pathlib
import sys
from datetime import date

ROOT = pathlib.Path(__file__).resolve().parents[1]
QUEUE = ROOT / "ml" / "out" / "worklist.json"
DEST = ROOT / "data" / "annotations.json"

TYPES = {
    "1": "setup-payoff",
    "2": "shared-character",
    "3": "shared-object",
    "4": "timeline-adjacent",
    "5": "theme-echo",
}


def main():
    if len(sys.argv) < 2:
        sys.exit('usage: record.py "1:1 2:x 3:s"')

    queue = json.loads(QUEUE.read_text())
    connections = json.loads(DEST.read_text()) if DEST.exists() else []
    by_pair = {frozenset((c["source"], c["target"])): c for c in connections}

    added = []
    for token in sys.argv[1].replace(",", " ").split():
        if ":" not in token:
            sys.exit(f"malformed token {token!r} — expected like 3:x")
        pos, key = token.split(":", 1)
        key = key.lower().strip()
        idx = int(pos)          # worklist positions are 0-indexed
        if not 0 <= idx < len(queue):
            sys.exit(f"position {pos} is outside the queue (1..{len(queue)})")

        q = queue[idx]
        record = {
            "source": q["source"],
            "target": q["target"],
            "annotated_by": "claude-opus-5/in-session",
            
            "judged_at": date.today().isoformat(),
        }
        if key in ("y", "yes"):
            # The reviewer answers connected / not connected. The edge TYPE is
            # taken from whatever the proposer suggested and flagged as
            # machine-assigned, because asking a non-expert to choose between
            # five categories was the thing that made this unusable.
            record["verdict"] = "confirmed"
            record["type"] = q.get("type") or "theme-echo"
            record["type_source"] = "suggested"
        elif key in TYPES:
            record["verdict"] = "confirmed"
            record["type"] = TYPES[key]
        elif key in ("x", "n", "no"):
            record["verdict"] = "rejected"
        elif key in ("s", "?"):
            record["verdict"] = "unsure"
        else:
            sys.exit(f"unknown key {key!r} — use y, n or s")

        by_pair[frozenset((q["source"], q["target"]))] = record
        added.append((pos, record["verdict"], record.get("type", "")))

    DEST.write_text(json.dumps(list(by_pair.values()), indent=2) + "\n")
    for pos, verdict, kind in added:
        print(f"  {pos:>3}  {verdict}{' · ' + kind if kind else ''}")
    print(f"\n{len(added)} recorded, {len(by_pair)} judgements total")


if __name__ == "__main__":
    main()
