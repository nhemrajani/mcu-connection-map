"""add_visuals.py - merge written scene descriptions into data/moments.json.

    python3 ingest/add_visuals.py visuals-p1.json

Takes a flat {moment_id: "scene description"} map and writes each one onto the
matching moment's `visual` field. Checks every id exists before writing
anything, so a typo fails loudly rather than silently dropping a description.

Also warns on descriptions long enough to be truncated. CLIP reads 77 tokens
and discards the rest without complaint, which silently cost this project
several rounds of wrong images before it was caught: the model was drawing
from half a prompt and there was nothing in the output to say so.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "moments.json"
BUDGET = 60          # leaves room for the style suffix inside CLIP's 77


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: add_visuals.py <visuals.json>")

    incoming = json.loads(pathlib.Path(sys.argv[1]).read_text())
    moments = json.loads(DEST.read_text())
    by_id = {m["id"]: m for m in moments}

    missing = [k for k in incoming if k not in by_id]
    if missing:
        for k in missing:
            print(f"  ERROR  no such moment: {k}")
        sys.exit(f"{len(missing)} unknown id(s); nothing written.")

    try:
        from transformers import CLIPTokenizer
        tok = CLIPTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer")
        count = lambda t: len(tok(t)["input_ids"]) - 2
    except Exception:
        count = lambda t: int(len(t.split()) * 1.35)   # rough fallback

    added, long = 0, []
    for mid, text in incoming.items():
        n = count(text)
        if n > BUDGET:
            long.append((mid, n))
        by_id[mid]["visual"] = text.strip()
        added += 1

    DEST.write_text(json.dumps(moments, indent=2) + "\n")
    have = sum(1 for m in moments if m.get("visual"))
    print(f"+{added} visuals -> {have}/{len(moments)} moments now have one")
    for mid, n in long:
        print(f"  warning  {mid}: {n} tokens, over the {BUDGET} budget")


if __name__ == "__main__":
    main()
