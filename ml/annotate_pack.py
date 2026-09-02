"""annotate_pack.py - build blind judging packets from the primary source.

    .venv/bin/python ml/annotate_pack.py [n]
    Output: ml/out/annotate_pack.json

The model that wrote the moment descriptions and proposed the edges cannot
credibly grade its own work: its errors would be correlated, so a
misunderstanding made while writing a description would be repeated exactly
while judging it, and stay invisible.

Two things reduce that, and this script does both.

  PRIMARY TEXT.  A packet carries the passages from the original Wikipedia
  plot summaries, not the moment descriptions. The annotator judges the source
  rather than the project's own interpretation of it.

  BLIND.  The packet does not say which source proposed the pair, what score it
  scored, or what type was suggested. Knowing "this came from strong entity
  overlap" would bias the verdict toward yes and inflate exactly the number the
  evaluation is trying to measure.

What this does NOT fix: the same model still wrote the moments, so the pairing
itself is downstream of its earlier choices. Model annotations are therefore
kept in data/annotations.json, never merged into the human judgements, and are
only trustworthy to the extent they agree with a human on the same pairs.
Run ml/agreement.py to find out whether they do.
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "ml" / "out"


def plot_text(article):
    p = ROOT / "plots" / (re.sub(r"[^\w\-. ]", "_", article) + ".txt")
    return p.read_text() if p.exists() else ""


def passages(moment, article_of, terms, window=2):
    """Sentences around the best match for this moment in its source text.

    Anchors on the moment's own distinctive words rather than the proposed
    link's evidence, so the passage describes what actually happens rather
    than being selected to support the connection.
    """
    text = plot_text(article_of.get(moment["film"].rsplit(" (", 1)[0], ""))
    if not text:
        return []
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
             if 30 < len(s.strip()) < 400]
    if not sents:
        return []
    key = {w.lower() for w in re.findall(r"[A-Z][a-zA-Z'\-]{3,}", moment["description"])}
    key |= {w.lower() for w in re.findall(r"\b[a-z]{6,}\b", moment["description"])}
    best, score = 0, -1
    for i, s in enumerate(sents):
        words = {w.lower() for w in re.findall(r"[A-Za-z'\-]{4,}", s)}
        overlap = len(key & words)
        if overlap > score:
            best, score = i, overlap
    lo, hi = max(0, best - window), min(len(sents), best + window + 1)
    return sents[lo:hi]


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    moments = {m["id"]: m for m in json.loads((ROOT / "data" / "moments.json").read_text())}
    titles = json.loads((ROOT / "data" / "titles.json").read_text())
    article_of = {t["title"]: t["article"] for t in titles}
    queue = json.loads((OUT / "review_queue.json").read_text())

    pack = []
    for i, q in enumerate(queue[:limit], 1):
        a, b = moments[q["source"]], moments[q["target"]]
        pack.append({
            "position": i,                       # index back into review_queue
            "a_title": a["title"], "a_film": a["film"],
            "b_title": b["title"], "b_film": b["film"],
            "a_passage": passages(a, article_of, q.get("evidence")),
            "b_passage": passages(b, article_of, q.get("evidence")),
            # deliberately omitted: bucket, proposed_by, score, suggested_type
        })

    (OUT / "annotate_pack.json").write_text(json.dumps(pack, indent=2) + "\n")
    withtext = sum(1 for p in pack if p["a_passage"] and p["b_passage"])
    print(f"{len(pack)} packets -> ml/out/annotate_pack.json")
    print(f"{withtext} have primary passages on both sides")


if __name__ == "__main__":
    main()
