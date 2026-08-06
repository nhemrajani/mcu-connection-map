"""
embed.py - turn each MCU "moment" into a vector and surface candidate connections.

This is the FIRST ml step in the pipeline. It teaches three ideas:

  1. Embeddings - a sentence-transformer maps each moment's text description to a
     high-dimensional vector, so "meaning" becomes geometry.
  2. Cosine similarity - moments that mean similar things end up pointing in
     similar directions, which we measure as a number between -1 and 1.
  3. Candidate generation - every pair above a similarity threshold becomes a
     *suggested* connection for a human to confirm or reject.

Run:    python embed.py
Output: ml/out/candidates.json  (ranked suggested connections)
"""

import json
import itertools
from pathlib import Path

from sentence_transformers import SentenceTransformer, util

# --- config ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "moments.json"
OUT = Path(__file__).resolve().parent / "out"

MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, free - ideal for learning
THRESHOLD = 0.35                 # tune me: higher = fewer, stronger suggestions


def main():
    moments = json.loads(DATA.read_text())
    ids = [m["id"] for m in moments]
    # feed the model the title AND description - both carry meaning
    texts = [f'{m["title"]}. {m["description"]}' for m in moments]

    print(f"Loaded {len(moments)} moments. Embedding with {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)

    # cosine similarity of every moment against every other moment
    sim = util.cos_sim(embeddings, embeddings)

    candidates = []
    for i, j in itertools.combinations(range(len(moments)), 2):
        score = float(sim[i][j])
        if score >= THRESHOLD:
            candidates.append({
                "source": ids[i],
                "target": ids[j],
                "score": round(score, 3),
                "status": "suggested",  # promote to a real edge by hand
            })

    candidates.sort(key=lambda c: c["score"], reverse=True)

    OUT.mkdir(exist_ok=True)
    (OUT / "candidates.json").write_text(json.dumps(candidates, indent=2))

    print(f"\nWrote {len(candidates)} candidate connections -> ml/out/candidates.json")
    print("Top suggestions:")
    for c in candidates[:10]:
        print(f'  {c["score"]:.3f}   {c["source"]}  <->  {c["target"]}')
    print("\nReview these, then move the real ones into data/connections.json.")


if __name__ == "__main__":
    main()
