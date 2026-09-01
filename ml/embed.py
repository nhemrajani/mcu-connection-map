"""embed.py - turn each moment into a vector and surface candidate connections.

The FIRST ML step. Three ideas:

  1. Embeddings - a sentence-transformer maps each moment's description to a
     high-dimensional vector, so "meaning" becomes geometry.
  2. Cosine similarity - moments that mean similar things point in similar
     directions, measured as a number between -1 and 1.
  3. Candidate generation - every pair above a threshold becomes a *suggested*
     connection for a human to confirm or reject.

Run:    .venv/bin/python ml/embed.py
Output: ml/out/candidates.json   ranked suggestions you have not yet judged
        ml/out/embeddings.npy    the vectors, reused by graph.py for layout

Pairs already recorded in data/connections.json - confirmed or rejected - are
never re-proposed. That is what stops the review queue handing you the same
rejected pair forever, and it is why rejections are worth storing.
"""
import itertools
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent / "out"

MODEL_NAME = "all-MiniLM-L6-v2"   # small, fast, free - ideal for learning
THRESHOLD = 0.45                  # tune me: higher = fewer, stronger suggestions
TOP_N = 400                       # cap the review queue at something a human can face


def main():
    moments = json.loads((DATA / "moments.json").read_text())
    connections = json.loads((DATA / "connections.json").read_text())

    ids = [m["id"] for m in moments]
    # Feed the model the title and description together - both carry meaning.
    texts = [f'{m["title"]}. {m["description"]}' for m in moments]

    # Anything already judged is settled; don't ask again.
    judged = {frozenset((c["source"], c["target"])) for c in connections}

    print(f"Embedding {len(moments)} moments with {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    # Normalised vectors mean the dot product IS cosine similarity.
    sim = vectors @ vectors.T

    candidates, skipped = [], 0
    for i, j in itertools.combinations(range(len(moments)), 2):
        score = float(sim[i][j])
        if score < THRESHOLD:
            continue
        if frozenset((ids[i], ids[j])) in judged:
            skipped += 1
            continue
        candidates.append({
            "source": ids[i],
            "target": ids[j],
            "score": round(score, 4),
            "proposed_by": MODEL_NAME,
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)

    OUT.mkdir(exist_ok=True)
    np.save(OUT / "embeddings.npy", vectors)
    (OUT / "moment_ids.json").write_text(json.dumps(ids))
    (OUT / "candidates.json").write_text(json.dumps(candidates[:TOP_N], indent=2) + "\n")

    total_pairs = len(moments) * (len(moments) - 1) // 2
    print(f"\n{total_pairs:,} possible pairs")
    print(f"{len(candidates):,} above threshold {THRESHOLD} ({skipped} already judged, skipped)")
    print(f"wrote top {min(TOP_N, len(candidates))} -> ml/out/candidates.json")

    by_id = {m["id"]: m for m in moments}
    print("\nStrongest suggestions:")
    for c in candidates[:12]:
        a, b = by_id[c["source"]], by_id[c["target"]]
        print(f'  {c["score"]:.3f}  {a["title"][:34]:36s} <-> {b["title"][:34]}')


if __name__ == "__main__":
    main()
