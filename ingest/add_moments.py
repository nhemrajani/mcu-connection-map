"""Merge a batch of extracted moments into data/moments.json.

    python3 ingest/add_moments.py batch.json

Extraction happens in batches of a few titles at a time, so this appends
rather than overwrites, refuses to introduce duplicate ids, and validates
required fields before writing. Re-running the same batch is a no-op.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "moments.json"
REQUIRED = ("id", "title", "film", "description")


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: add_moments.py <batch.json>")

    batch = json.loads(pathlib.Path(sys.argv[1]).read_text())
    existing = json.loads(DEST.read_text()) if DEST.exists() else []
    by_id = {m["id"]: m for m in existing}

    added, skipped, bad = 0, 0, []
    for m in batch:
        missing = [f for f in REQUIRED if not m.get(f)]
        if missing:
            bad.append(f"{m.get('id', '?')}: missing {', '.join(missing)}")
            continue
        if m["id"] in by_id:
            skipped += 1
            continue
        by_id[m["id"]] = m
        added += 1

    if bad:
        for b in bad:
            print(f"  ERROR  {b}")
        sys.exit(f"{len(bad)} invalid moment(s); nothing written.")

    merged = list(by_id.values())
    DEST.write_text(json.dumps(merged, indent=2) + "\n")
    print(f"+{added} added, {skipped} already present -> {len(merged)} moments total")


if __name__ == "__main__":
    main()
