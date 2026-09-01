"""Check the data files before anything downstream trusts them.

Run this after every hand-edit:   python3 ml/validate.py

It catches the mistakes that are easy to make by hand and annoying to debug
later: duplicate ids, edges pointing at moments that don't exist, missing
required fields, and self-links.
"""
import json
import pathlib
import sys

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
REQUIRED = ("id", "title", "film", "description")
TYPES = {
    "setup-payoff",
    "shared-character",
    "shared-object",
    "timeline-adjacent",
    "theme-echo",
}

errors = []
warnings = []

moments = json.loads((DATA / "moments.json").read_text())
connections = json.loads((DATA / "connections.json").read_text())

ids = set()
for i, m in enumerate(moments):
    where = m.get("id") or f"moment #{i}"
    for field in REQUIRED:
        if not m.get(field):
            errors.append(f"{where}: missing required field '{field}'")
    if m.get("id") in ids:
        errors.append(f"{where}: duplicate id")
    ids.add(m.get("id"))
    if m.get("id") and not m["id"].startswith("m-"):
        warnings.append(f"{where}: id should start with 'm-'")
    # The description is what gets embedded, so thin ones weaken the ML.
    if len(m.get("description", "").split()) < 12:
        warnings.append(f"{where}: description is very short; embeddings need meaning")

# "unsure" is a real answer: the reviewer looked and the evidence did not settle it.
VERDICTS = {"confirmed", "rejected", "unsure"}

for i, c in enumerate(connections):
    where = f"connection #{i} ({c.get('source')} -> {c.get('target')})"
    if c.get("verdict") not in VERDICTS:
        errors.append(f"{where}: verdict must be one of {sorted(VERDICTS)}")
    for end in ("source", "target"):
        if c.get(end) not in ids:
            errors.append(f"{where}: '{end}' is not an existing moment id")
    if c.get("source") == c.get("target"):
        errors.append(f"{where}: links a moment to itself")
    # Only confirmed edges reach the graph, so only they need a type.
    if c.get("verdict") == "confirmed" and c.get("type") not in TYPES:
        errors.append(f"{where}: unknown type {c.get('type')!r}")

linked = {e for c in connections if c.get("verdict") == "confirmed"
          for e in (c.get("source"), c.get("target"))}
for orphan in sorted(ids - linked):
    warnings.append(f"{orphan}: no connections yet")

confirmed = sum(1 for c in connections if c.get("verdict") == "confirmed")
rejected = sum(1 for c in connections if c.get("verdict") == "rejected")
unsure = sum(1 for c in connections if c.get("verdict") == "unsure")
print(f"{len(moments)} moments, {confirmed} confirmed, {rejected} rejected, "
      f"{unsure} unsure ({len(connections)} judgements total)")
for w in warnings:
    print(f"  warning  {w}")
for e in errors:
    print(f"  ERROR    {e}")

if errors:
    print(f"\n{len(errors)} error(s). Fix these before running embed.py.")
    sys.exit(1)
print("\nData is valid.")
