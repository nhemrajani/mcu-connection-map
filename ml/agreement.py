"""agreement.py - do the model's labels match the human's?

    .venv/bin/python ml/agreement.py

Model annotations are only worth anything at scale if they agree with a human
on the pairs where both have an opinion. This reports that overlap two ways.

  RAW AGREEMENT is the share of shared pairs where both said the same thing.
  On its own it flatters: if both parties say "connected" 80% of the time,
  they agree 68% of the time by chance alone.

  COHEN'S KAPPA corrects for that. It asks how much better than chance the
  agreement is, on a scale where 0 is no better than guessing and 1 is
  perfect. The usual reading:

      < 0.20   negligible - the model labels cannot be trusted
      0.21-0.40  fair
      0.41-0.60  moderate
      0.61-0.80  substantial - usable at scale, with the caveat reported
      > 0.80   strong

Whatever it comes out at is the finding, and it belongs in the write-up
either way. A low kappa is not a failed experiment; it is the discovery that
the model cannot stand in for a reader on this task.
"""
import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(path):
    p = ROOT / path
    return json.loads(p.read_text()) if p.exists() else []


def key(c):
    return frozenset((c["source"], c["target"]))


def main():
    human = {key(c): c["verdict"] for c in load("data/connections.json") if c.get("bucket")}
    model = {key(c): c["verdict"] for c in load("data/annotations.json")}

    shared = [k for k in human if k in model
              and human[k] != "unsure" and model[k] != "unsure"]

    print(f"{len(human)} human judgements, {len(model)} model annotations, "
          f"{len(shared)} decided by both")
    if len(shared) < 5:
        print("\nNot enough overlap to measure agreement yet.")
        return

    both = collections.Counter((human[k], model[k]) for k in shared)
    n = len(shared)
    agree = sum(v for (h, m), v in both.items() if h == m)
    po = agree / n

    labels = ("confirmed", "rejected")
    pe = sum(
        (sum(v for (h, _), v in both.items() if h == lab) / n) *
        (sum(v for (_, m), v in both.items() if m == lab) / n)
        for lab in labels
    )
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0

    print(f"\nraw agreement  {po:.0%}")
    print(f"Cohen's kappa  {kappa:.2f}", end="  ")
    print("(negligible)" if kappa < .21 else "(fair)" if kappa < .41 else
          "(moderate)" if kappa < .61 else "(substantial)" if kappa < .81 else "(strong)")

    print("\n           model:  confirmed  rejected")
    for h in labels:
        row = "  ".join(f"{both[(h, m)]:9d}" for m in labels)
        print(f"  human {h:9s} {row}")

    disagreements = [k for k in shared if human[k] != model[k]]
    if disagreements:
        moments = {m["id"]: m for m in load("data/moments.json")}
        print(f"\n{len(disagreements)} disagreement(s):")
        for k in disagreements[:8]:
            a, b = sorted(k)
            print(f"  human={human[k]:9s} model={model[k]:9s}  "
                  f"{moments[a]['title'][:32]} <-> {moments[b]['title'][:32]}")


if __name__ == "__main__":
    main()
