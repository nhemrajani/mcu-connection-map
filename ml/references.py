"""references.py - edges the source text states outright.

    .venv/bin/python ml/references.py
    Output: ml/out/reference_edges.json

Every other edge source in this project INFERS a connection. Cosine similarity
infers it from wording; relate.py infers it from a shared name. This one does
not infer anything: it finds places where the plot summary says the connection
out loud.

    "Following the Battle of New York, Agent Phil Coulson assembles a team..."
    "After stealing the Tesseract during the events of Avengers: Endgame..."
    "returns to life following the Blip"

These are cross-references written by Wikipedia's editors, so they are as close
to ground truth as this corpus gets, and each one carries a literal quote as
evidence. That matters twice over: it makes the edge checkable by someone who
has never seen the films, and it gives the evaluation a set of positives that
needed no judgement call.

How an edge is resolved:

  1. Find a sentence carrying a temporal or causal marker next to a named event
     or another title ("following the Blip", "after the events of Endgame").
  2. The SOURCE end is whichever moment of the referencing title best matches
     that sentence.
  3. The TARGET end is whichever moment anywhere in an EARLIER title best
     matches the event phrase itself.
  4. Both matches must clear a similarity floor, or the reference is skipped
     rather than guessed at.

Direction is always earlier -> later, so these are setup-payoff edges.
"""
import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent / "out"
MODEL_NAME = "all-MiniLM-L6-v2"

# A reference needs a temporal or causal marker. Without one, a bare mention of
# "Thanos" is just a mention, not a claim that one thing followed another.
MARKER = (r"(?:after|following|since|prior to|in the aftermath of|"
          r"years? after|weeks? after|months? after|days? after|during)")

# Named EVENTS only - specific occurrences, not characters. An early version
# included "Thanos" and "Ultron" and it wrecked precision: every sentence with
# a marker and a character name fired, so "After defeating Thanos, the
# Illuminati executed their Strange" - a different universe entirely - was
# linked to the snap. A character is not an event.
EVENTS = [
    "Battle of New York", "Battle of Sokovia", "Battle of Earth",
    "Battle of Wakanda", "the Blip", "the Snap", "the Decimation",
    "Sokovia Accords", "the Convergence", "the Emergence",
]

MATCH_FLOOR = 0.34      # below this the resolved end is a guess, not a match


def year_of(text):
    m = re.search(r"\((\d{4})\)", text or "")
    return int(m.group(1)) if m else 0


def plot_file(article):
    return ROOT / "plots" / (re.sub(r"[^\w\-. ]", "_", article) + ".txt")


def find_references(text, titles):
    """Sentences that explicitly point at another title or a named event."""
    out = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")):
        s = sentence.strip()
        if not 40 < len(s) < 400:
            continue
        for name in titles:
            if len(name) > 12 and re.search(rf"{MARKER}\s+[^.]{{0,30}}{re.escape(name)}", s, re.I):
                out.append((s, name, "title"))
                break
        else:
            for ev in EVENTS:
                if (re.search(rf"{MARKER}\s+[^.]{{0,40}}{re.escape(ev)}", s, re.I)
                        or re.search(rf"{re.escape(ev)}[^.]{{0,30}}\b{MARKER}\b", s, re.I)):
                    out.append((s, ev, "event"))
                    break
    return out


def main():
    moments = json.loads((DATA / "moments.json").read_text())
    titles = json.loads((DATA / "titles.json").read_text())
    judged = {
        frozenset((c["source"], c["target"]))
        for c in json.loads((DATA / "connections.json").read_text())
    }

    by_film = {}
    for i, m in enumerate(moments):
        by_film.setdefault(m["film"].rsplit(" (", 1)[0], []).append(i)
    title_names = sorted({t["title"] for t in titles}, key=len, reverse=True)

    # Collect every reference sentence across the corpus first, so the model is
    # loaded once and all sentences are embedded in a single batch.
    found = []
    for t in titles:
        path = plot_file(t["article"])
        if not path.exists() or t["title"] not in by_film:
            continue
        for sentence, target_name, kind in find_references(path.read_text(), title_names):
            if target_name == t["title"]:
                continue
            found.append({"from_title": t["title"], "sentence": sentence,
                          "target_name": target_name, "kind": kind})

    if not found:
        print("No explicit references found.")
        return

    print(f"Embedding {len(moments)} moments and {len(found)} reference sentences ...")
    model = SentenceTransformer(MODEL_NAME)
    mvecs = model.encode([f'{m["title"]}. {m["description"]}' for m in moments],
                         normalize_embeddings=True, show_progress_bar=False)
    svecs = model.encode([f["sentence"] for f in found],
                         normalize_embeddings=True, show_progress_bar=False)
    tvecs = model.encode([f["target_name"] for f in found],
                         normalize_embeddings=True, show_progress_bar=False)

    edges, skipped = [], 0
    for k, f in enumerate(found):
        here = by_film.get(f["from_title"], [])
        if not here:
            continue
        # Source end: the moment in this title closest to the reference sentence.
        sims = mvecs[here] @ svecs[k]
        si = here[int(np.argmax(sims))]

        # Target end: the best match in any EARLIER title. Restricting to
        # earlier titles enforces the direction rather than assuming it.
        this_year = year_of(moments[si]["film"])
        pool = [j for j, m in enumerate(moments)
                if year_of(m["film"]) < this_year or
                (year_of(m["film"]) == this_year and m["film"] != moments[si]["film"])]
        if f["kind"] == "title":
            pool = [j for j in pool
                    if moments[j]["film"].rsplit(" (", 1)[0] == f["target_name"]] or pool
        if not pool:
            continue
        tsims = mvecs[pool] @ tvecs[k]
        ti = pool[int(np.argmax(tsims))]

        if float(sims.max()) < MATCH_FLOOR or float(tsims.max()) < MATCH_FLOOR or si == ti:
            skipped += 1
            continue
        if frozenset((moments[ti]["id"], moments[si]["id"])) in judged:
            continue

        edges.append({
            "source": moments[ti]["id"],      # earlier
            "target": moments[si]["id"],      # later
            "type": "setup-payoff",
            "quote": f["sentence"],
            "refers_to": f["target_name"],
            "confidence": round(min(float(sims.max()), float(tsims.max())), 4),
            "proposed_by": "references.py/stated-in-source",
        })

    # One reference sentence can match several times; keep the strongest.
    best = {}
    for e in edges:
        key = (e["source"], e["target"])
        if key not in best or e["confidence"] > best[key]["confidence"]:
            best[key] = e
    edges = sorted(best.values(), key=lambda e: -e["confidence"])

    OUT.mkdir(exist_ok=True)
    (OUT / "reference_edges.json").write_text(json.dumps(edges, indent=2) + "\n")
    print(f"\n{len(edges)} stated references resolved "
          f"({skipped} skipped as too weak to resolve)")
    by_id = {m["id"]: m for m in moments}
    for e in edges[:8]:
        print(f'\n  {e["confidence"]:.2f}  refers to "{e["refers_to"]}"')
        print(f'    {by_id[e["source"]]["title"][:44]:46s} ({by_id[e["source"]]["film"][:28]})')
        print(f'    -> {by_id[e["target"]]["title"][:44]:43s} ({by_id[e["target"]]["film"][:28]})')
        print(f'    "{e["quote"][:120]}..."')
    print("\nwrote ml/out/reference_edges.json")


if __name__ == "__main__":
    main()
