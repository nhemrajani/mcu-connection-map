"""sample_for_review.py - build a balanced queue of pairs for a human to judge.

    .venv/bin/python ml/sample_for_review.py [n]

Output: ml/out/review_queue.json

Why sample rather than review everything: there are thousands of proposals and
the point is not to confirm them all, it is to MEASURE how often each source is
right. For that you need a sample that is balanced across sources and strength
bands, not whatever happens to be at the top of the list.

Each source is sampled separately so precision can be reported per source:

    entity-strong / entity-weak      from relate.py  (shared rare entities)
    similar-strong / similar-weak    from embed.py   (cosine similarity)

Without the strength split you learn "the system is 60% right", which is not
actionable. With it you learn where the threshold should sit.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "out"
SEED = 20260901


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    rng = random.Random(SEED)

    moments = {m["id"]: m for m in json.loads((ROOT / "data" / "moments.json").read_text())}
    judged = {
        frozenset((c["source"], c["target"]))
        for c in json.loads((ROOT / "data" / "connections.json").read_text())
    }

    entity = [p for p in json.loads((OUT / "proposed_edges.json").read_text())
              if frozenset((p["source"], p["target"])) not in judged]
    similar = [c for c in json.loads((OUT / "candidates.json").read_text())
               if frozenset((c["source"], c["target"])) not in judged]

    # Split each source at its own median so the bands mean "strong / weak for
    # this source", not "strong on some shared scale these two do not share".
    entity.sort(key=lambda p: -p["weight"])
    similar.sort(key=lambda c: -c["score"])
    buckets = {
        "entity-strong": entity[: len(entity) // 2],
        "entity-weak": entity[len(entity) // 2:],
        "similar-strong": similar[: len(similar) // 2],
        "similar-weak": similar[len(similar) // 2:],
    }

    per = max(1, target // len(buckets))
    queue = []
    for name, pool in buckets.items():
        pool = [p for p in pool if p["source"] in moments and p["target"] in moments]
        for p in rng.sample(pool, min(per, len(pool))):
            queue.append({
                "source": p["source"],
                "target": p["target"],
                "bucket": name,
                "proposed_by": p.get("proposed_by", "?"),
                "suggested_type": p.get("type"),
                "evidence": p.get("evidence"),
                "score": p.get("score", p.get("weight")),
            })

    rng.shuffle(queue)   # interleave sources so you cannot guess from position
    (OUT / "review_queue.json").write_text(json.dumps(queue, indent=2) + "\n")

    print(f"{len(queue)} pairs queued -> ml/out/review_queue.json")
    for name in buckets:
        print(f"  {sum(1 for q in queue if q['bucket'] == name):4d}  {name}")


if __name__ == "__main__":
    main()
