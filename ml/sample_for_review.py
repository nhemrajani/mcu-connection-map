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
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "out"
SEED = 20260901


def plot_path(article):
    return ROOT / "plots" / (re.sub(r"[^\w\-. ]", "_", article) + ".txt")


def source_lookup():
    """Map each film title to the cached plot text it came from."""
    titles = json.loads((ROOT / "data" / "titles.json").read_text())
    out = {}
    for t in titles:
        p = plot_path(t["article"])
        if p.exists():
            out[t["title"]] = p.read_text()
    return out


STOPWORDS = set("""the a an and or but of to in on at for with from by as is are was were
be been being his her its their they he she it that this these those who whom which what when
while after before during into onto out up down over under again then than so not no nor only
own same too very can will just also him them him's himself herself itself one two three
""".split())


def shared_terms(a, b, limit=4):
    """Significant words both descriptions use, for pairs with no shared entity.

    Similarity candidates have no entity to search on, so without this half the
    queue would arrive with no evidence at all and be unjudgeable.
    """
    def words(m):
        return {w for w in re.findall(r"[A-Za-z][\w'-]{3,}", m["description"])
                if w.lower() not in STOPWORDS}
    common = words(a) & words(b)
    # Prefer proper nouns, then longer words - both carry more signal.
    return sorted(common, key=lambda w: (w[0].islower(), -len(w)))[:limit]


def quotes_for(moment, terms, sources, limit=2):
    """Pull sentences from the source plot that mention the shared terms.

    A reviewer who has not seen the films cannot judge from memory, but they
    can judge from evidence. This turns "do you know the MCU?" into "does the
    text support this link?", which is a question anyone can answer.
    """
    text = sources.get(moment.get("film", "").rsplit(" (", 1)[0])
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    hits = []
    for s in sentences:
        if any(term.lower() in s.lower() for term in terms):
            s = s.strip()
            if 40 < len(s) < 320 and s not in hits:
                hits.append(s)
        if len(hits) >= limit:
            break
    return hits


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

    sources = source_lookup()
    per = max(1, target // len(buckets))
    queue = []
    for name, pool in buckets.items():
        pool = [p for p in pool if p["source"] in moments and p["target"] in moments]
        for p in rng.sample(pool, min(per, len(pool))):
            a, b = moments[p["source"]], moments[p["target"]]
            terms = p.get("evidence") or shared_terms(a, b)
            queue.append({
                "source": p["source"],
                "target": p["target"],
                "bucket": name,
                "proposed_by": p.get("proposed_by", "?"),
                "suggested_type": p.get("type"),
                "evidence": terms,
                "score": p.get("score", p.get("weight")),
                # Sentences from the original Wikipedia plot text, so the pair
                # can be judged on evidence rather than recall.
                "quotes_a": quotes_for(a, terms, sources),
                "quotes_b": quotes_for(b, terms, sources),
                "year_a": int(re.search(r"\((\d{4})\)", a.get("film", "")).group(1))
                          if re.search(r"\((\d{4})\)", a.get("film", "")) else None,
                "year_b": int(re.search(r"\((\d{4})\)", b.get("film", "")).group(1))
                          if re.search(r"\((\d{4})\)", b.get("film", "")) else None,
            })

    rng.shuffle(queue)   # interleave sources so you cannot guess from position
    (OUT / "review_queue.json").write_text(json.dumps(queue, indent=2) + "\n")

    print(f"{len(queue)} pairs queued -> ml/out/review_queue.json")
    for name in buckets:
        print(f"  {sum(1 for q in queue if q['bucket'] == name):4d}  {name}")


if __name__ == "__main__":
    main()
