"""evaluate.py - how often is each edge source actually right?

    .venv/bin/python ml/evaluate.py

Reads the judgements in data/connections.json and reports precision per source
and per strength band.

PRECISION is the share of proposals that turned out to be real:

    precision = confirmed / (confirmed + rejected)

It is the number the project has been missing. "The map found a connection
between Fury's pager and Captain Marvel" is an anecdote. "Entity overlap is
right 84% of the time and cosine similarity 41%" is a result, and it tells you
where to set a threshold.

Note what this does NOT measure. RECALL - the share of all real connections
that were found - cannot be computed here, because nobody knows how many real
connections exist in the MCU. Estimating it would need a set of moment pairs
exhaustively labelled by hand. Reporting precision alone is honest; quoting it
as though it were accuracy is not.
"""
import collections
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def wilson(k, n, z=1.96):
    """95% confidence interval for a proportion.

    With 50 labels a raw percentage is a noisy estimate, and reporting it bare
    overstates what the sample can support. This gives the range the true value
    plausibly sits in.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def report(title, rows):
    print(f"\n{title}")
    print(f"  {'group':22s} {'n':>5s} {'confirmed':>10s} {'precision':>10s}   95% CI")
    for name, judged in sorted(rows.items(), key=lambda kv: -len(kv[1])):
        decided = [v for v in judged if v != "unsure"]
        n = len(decided)
        k = sum(1 for v in decided if v == "confirmed")
        lo, hi = wilson(k, n)
        pct = f"{k / n:.0%}" if n else "–"
        print(f"  {name:22s} {n:5d} {k:10d} {pct:>10s}   {lo:.0%}–{hi:.0%}")


def main():
    connections = json.loads((ROOT / "data" / "connections.json").read_text())
    labelled = [c for c in connections if c.get("bucket")]

    if not labelled:
        print("No labelled judgements yet.")
        print("Run  ml/sample_for_review.py  then  ml/review.py  and judge some pairs.")
        return

    by_bucket = collections.defaultdict(list)
    by_source = collections.defaultdict(list)
    for c in labelled:
        by_bucket[c["bucket"]].append(c["verdict"])
        by_source[c["bucket"].split("-")[0]].append(c["verdict"])

    decided = [c for c in labelled if c["verdict"] != "unsure"]
    unsure = len(labelled) - len(decided)
    total = len(decided)
    confirmed = sum(1 for c in decided if c["verdict"] == "confirmed")
    if not total:
        print(f"{unsure} judgements, all marked unsure - nothing to measure yet.")
        return
    lo, hi = wilson(confirmed, total)
    print(f"{total} decided   overall precision {confirmed / total:.0%}  ({lo:.0%}–{hi:.0%})"
          + (f"   [{unsure} marked unsure, excluded]" if unsure else ""))

    report("By source", by_source)
    report("By source and strength", by_bucket)

    types = collections.Counter(
        c.get("type") for c in labelled if c["verdict"] == "confirmed"
    )
    if types:
        print("\nConfirmed edge types")
        for t, n in types.most_common():
            print(f"  {n:5d}  {t}")

    if total < 60:
        print(f"\nOnly {total} labels so far - the intervals above are wide. "
              "Aim for 200 before quoting these numbers anywhere.")


if __name__ == "__main__":
    main()
