"""worklist.py - the labelling queue, and a compact view of the next batch.

    .venv/bin/python ml/worklist.py build          rebuild from proposed edges
    .venv/bin/python ml/worklist.py next [n]       show the next n unjudged
    .venv/bin/python ml/worklist.py status         how far through

Ordered by evidence weight, strongest first, so the edges that matter most to
the graph get judged first and the weak tail can be cut wholesale later
without anyone reading it.

Judgements land in data/annotations.json rather than data/connections.json.
Both feed the graph, but they stay in separate files because they are
different claims: one is a person's opinion, the other is a model's, and
merging them would make the distinction unrecoverable.
"""
import json
import pathlib
import sys
import textwrap

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "ml" / "out"
WORKLIST = OUT / "worklist.json"


def load(path, default):
    p = ROOT / path
    return json.loads(p.read_text()) if p.exists() else default


def judged_pairs():
    pairs = set()
    for f in ("data/connections.json", "data/annotations.json"):
        for c in load(f, []):
            pairs.add(frozenset((c["source"], c["target"])))
    return pairs


def build():
    proposals = json.loads((OUT / "proposed_edges.json").read_text())
    proposals.sort(key=lambda p: -p.get("weight", 0))
    WORKLIST.write_text(json.dumps(proposals, indent=2) + "\n")
    print(f"{len(proposals):,} proposals -> ml/out/worklist.json (strongest first)")


def status():
    work = json.loads(WORKLIST.read_text())
    done = judged_pairs()
    left = [p for p in work if frozenset((p["source"], p["target"])) not in done]
    print(f"{len(work) - len(left):,} judged / {len(work):,} total   "
          f"({len(left):,} remaining)")


def show(n):
    work = json.loads(WORKLIST.read_text())
    moments = {m["id"]: m for m in load("data/moments.json", [])}
    done = judged_pairs()
    shown = 0
    for i, p in enumerate(work):
        if frozenset((p["source"], p["target"])) in done:
            continue
        a, b = moments[p["source"]], moments[p["target"]]
        ev = ", ".join(p.get("evidence", [])[:3]) or "wording only"
        af = a["film"].rsplit(" (", 1)[0]
        bf = b["film"].rsplit(" (", 1)[0]
        print(f"[{i}] {ev}")
        print(f"   A {af}: {textwrap.shorten(a['description'], 118)}")
        print(f"   B {bf}: {textwrap.shorten(b['description'], 118)}")
        shown += 1
        if shown >= n:
            break


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "build":
        build()
    elif cmd == "next":
        show(int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    else:
        status()
