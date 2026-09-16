"""finetune.py - train the embedding on this project's own judgements.

    .venv/bin/python ml/finetune.py          train, then evaluate held-out
    .venv/bin/python ml/finetune.py eval     evaluate an existing run only

Every edge in this project so far comes from a heuristic. The embedding model
is used off the shelf and frozen: feed it the corpus twice and it returns the
same vectors, having learned nothing from any judgement made here.

This is the step that changes that. The 114 judged pairs are labels, and a
label is a training signal. Contrastive fine-tuning pulls the vectors of
confirmed pairs together and pushes rejected pairs apart, so the model stops
measuring generic semantic similarity and starts measuring what this project
means by a narrative connection.

METHOD

  Pairs are split by MOMENT, not at random. A random split would put the two
  halves of a connection on opposite sides, so a moment seen in training would
  reappear at test time and the score would flatter itself. Splitting on
  moments means the test set contains only moments the model never adjusted
  for.

  Training uses CosineSimilarityLoss over (text_a, text_b, label) triples.
  OnlineContrastiveLoss was the first choice and is the more usual one, but it
  mines positive and negative pairs from within each batch, and this dataset is
  80% positive: some batches contained no negatives at all, which produced an
  empty tensor and crashed the MPS backward pass. CosineSimilarityLoss scores
  each pair independently against its label, so batch composition cannot break
  it.

  Training runs on CPU. The model is 22M parameters and the training set is
  tens of examples, so it takes seconds either way, and CPU avoids the MPS
  numerical quirks entirely.

  The metric is the same one used to filter the graph: precision at a
  threshold, plus ROC AUC, which is threshold-free and so cannot be gamed by
  moving the cutoff.

TWO CONDITIONS, BOTH REPORTED

  Negatives can be HARD (pairs a heuristic proposed and a reviewer rejected:
  plausible but wrong) or EASY (random pairs nobody ever proposed). The choice
  changes the result more than the training does, so reporting one alone would
  mislead.

  Easy negatives inflate every number. Two unrelated moments already have low
  cosine similarity, so separating them is close to free and the baseline
  scores well before any training happens. Hard negatives are the task this
  project actually has: the graph is built from proposals that all look
  plausible, and the work is deciding which of them are real.

WHAT THIS CANNOT SHOW

  114 labels is a small training set and a smaller test set. A result here is
  a demonstration that the loop closes, not evidence that the method scales.
  The honest framing for any write-up is the direction and the size of the
  gap, with the sample size stated beside it.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "out"
MODEL = "all-MiniLM-L6-v2"
TUNED = OUT / "tuned-model"
SEED = 7


def random_negatives(moments, judged, proposed, n, rng):
    """Pairs nobody proposed, used as negatives.

    The judged set is 80% positive by construction: every pair in it was
    already nominated by a heuristic, and most were sampled from the top of the
    weight distribution where precision is 92%. Training on that teaches the
    model to say yes.

    There are ~179,000 unproposed pairs in a 599-moment corpus. A pair that no
    heuristic nominated and no human confirmed is almost certainly unconnected,
    which makes these cheap, plentiful and honest negatives. They are also the
    right KIND of negative: the model needs to separate connection from mere
    co-occurrence, and a random pair is the clearest example of "not
    connected" available.
    """
    ids = [m["id"] for m in moments]
    by_id = {m["id"]: m for m in moments}
    out, tries = [], 0
    while len(out) < n and tries < n * 40:
        tries += 1
        a, b = rng.sample(ids, 2)
        k = frozenset((a, b))
        if k in judged or k in proposed:
            continue
        judged.add(k)
        out.append({
            "a_id": a, "b_id": b,
            "a": f'{by_id[a]["title"]}. {by_id[a]["description"]}',
            "b": f'{by_id[b]["title"]}. {by_id[b]["description"]}',
            "label": 0,
        })
    return out


def load_pairs(balance=True):
    """Every judged pair, plus random unproposed pairs as negatives."""
    moments = {m["id"]: m for m in json.loads((ROOT / "data" / "moments.json").read_text())}
    rows = []
    for path in ("data/connections.json", "data/annotations.json"):
        p = ROOT / path
        if not p.exists():
            continue
        for c in json.loads(p.read_text()):
            if c.get("verdict") not in ("confirmed", "rejected"):
                continue
            a, b = moments.get(c["source"]), moments.get(c["target"])
            if not a or not b:
                continue
            rows.append({
                "a_id": a["id"], "b_id": b["id"],
                "a": f'{a["title"]}. {a["description"]}',
                "b": f'{b["title"]}. {b["description"]}',
                "label": 1 if c["verdict"] == "confirmed" else 0,
            })
    # A pair judged in both files should count once; the human verdict wins.
    seen, uniq = set(), []
    for r in rows:
        k = frozenset((r["a_id"], r["b_id"]))
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)

    if balance:
        pos = sum(r["label"] for r in uniq)
        neg = len(uniq) - pos
        proposed = {frozenset((e["source"], e["target"]))
                    for e in json.loads((OUT / "proposed_edges.json").read_text())}
        uniq += random_negatives(list(moments.values()), seen, proposed,
                                 max(0, pos - neg), random.Random(SEED))
    return uniq


def split(rows, test_frac=0.3):
    """Hold out whole MOMENTS, not random pairs.

    Splitting pairs at random leaks: moment X appears in a training pair and
    again in a test pair, so the model has already been adjusted for half of
    every test example. Holding out moments means the test set is genuinely
    unseen on both ends.
    """
    rng = random.Random(SEED)
    ids = sorted({r["a_id"] for r in rows} | {r["b_id"] for r in rows})
    rng.shuffle(ids)
    held = set(ids[: int(len(ids) * test_frac)])
    train = [r for r in rows if r["a_id"] not in held and r["b_id"] not in held]
    test = [r for r in rows if r["a_id"] in held or r["b_id"] in held]
    return train, test


def scores(model, rows):
    a = model.encode([r["a"] for r in rows], normalize_embeddings=True,
                     show_progress_bar=False)
    b = model.encode([r["b"] for r in rows], normalize_embeddings=True,
                     show_progress_bar=False)
    return (a * b).sum(axis=1)


def report(name, sims, labels):
    labels = np.asarray(labels)
    order = np.argsort(-sims)
    ranked = labels[order]
    pos, neg = labels.sum(), len(labels) - labels.sum()

    # ROC AUC by rank, so no threshold has to be chosen.
    if pos and neg:
        ranks = np.empty(len(sims))
        ranks[order] = np.arange(len(sims), 0, -1)
        auc = (ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)
    else:
        auc = float("nan")

    k = max(1, int(pos))
    p_at_k = ranked[:k].mean()
    best_t, best_f1 = 0.0, 0.0
    for t in np.unique(sims):
        pred = sims >= t
        tp = (pred & (labels == 1)).sum()
        if not tp:
            continue
        prec, rec = tp / pred.sum(), tp / pos
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    print(f"  {name:22s} AUC {auc:.3f}   P@{k} {p_at_k:.0%}   best F1 {best_f1:.3f}")
    return {"auc": float(auc), f"p_at_{k}": float(p_at_k), "f1": float(best_f1)}


def run(condition, balance, train_it):
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    rows = load_pairs(balance=balance)
    train, test = split(rows)
    print(f"{len(rows)} judged pairs  ->  {len(train)} train, {len(test)} test")
    print(f"train {sum(r['label'] for r in train)} positive, "
          f"test {sum(r['label'] for r in test)} positive\n")

    if len(test) < 8 or not sum(r["label"] for r in test):
        sys.exit("test split too small or has no positives; adjust test_frac")

    base = SentenceTransformer(MODEL, device="cpu")
    before = report("off the shelf", scores(base, test), [r["label"] for r in test])

    if train_it:
        examples = [InputExample(texts=[r["a"], r["b"]], label=float(r["label"]))
                    for r in train]
        loader = DataLoader(examples, shuffle=True, batch_size=8)
        loss = losses.CosineSimilarityLoss(base)
        base.fit(train_objectives=[(loader, loss)], epochs=4,
                 warmup_steps=max(1, len(loader) // 4), show_progress_bar=False)
        TUNED.parent.mkdir(exist_ok=True)
        base.save(str(TUNED))

    after = report("fine-tuned", scores(base, test), [r["label"] for r in test])
    return {"condition": condition, "pairs": len(rows), "train": len(train),
            "test": len(test), "before": before, "after": after}


def main(train_it=True):
    results = []
    for condition, balance in (("hard negatives only", False),
                               ("plus random negatives", True)):
        print(f"\n=== {condition} ===")
        results.append(run(condition, balance, train_it))

    (OUT / "finetune-result.json").write_text(json.dumps(results, indent=2) + "\n")
    print("\n" + "-" * 62)
    for r in results:
        d = r["after"]["auc"] - r["before"]["auc"]
        print(f"  {r['condition']:24s} AUC {r['before']['auc']:.3f} -> "
              f"{r['after']['auc']:.3f}  ({d:+.3f})")
    print("\nwrote ml/out/finetune-result.json")


if __name__ == "__main__":
    main(train_it=(len(sys.argv) < 2 or sys.argv[1] != "eval"))
