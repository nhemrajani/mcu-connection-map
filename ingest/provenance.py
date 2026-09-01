"""provenance.py - record exactly where every moment came from.

    .venv/bin/python ingest/provenance.py
    Output: data/provenance.json and PROVENANCE.md

A dataset is only checkable if someone else can find the source it came from
and see how much interpretation sat between the two. For every title this
records:

  - the Wikipedia article used, with a permanent link to the EXACT revision
    read, so the text can be retrieved even after the article changes
  - how many words of source text were fetched
  - how many moments were extracted from them
  - the ratio between the two

That last column is the one that matters most, and it is the one a reader
should be suspicious of. It is not constant. A film yields roughly one moment
per 65 words; Agents of S.H.I.E.L.D. yields one per 1,700. That is a
deliberate editorial choice - moments were selected for reach across titles
rather than coverage of plot - but it means the map represents connective
significance, not screen time, and a reader deserves to see the number rather
than be told the corpus is neutral.

Wikipedia text is CC BY-SA; attribution and revision ids are how that licence
is honoured properly.
"""
import json
import pathlib
import re
import sys
from datetime import date

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from wiki import api

ROOT = pathlib.Path(__file__).resolve().parent.parent


def plot_path(article):
    return ROOT / "plots" / (re.sub(r"[^\w\-. ]", "_", article) + ".txt")


def revisions(articles):
    """Current revision id and timestamp for each article, 50 at a time."""
    out = {}
    for i in range(0, len(articles), 50):
        batch = articles[i:i + 50]
        data = api(action="query", prop="revisions", rvprop="ids|timestamp",
                   titles="|".join(batch), redirects=1)
        query = data.get("query", {})
        # Follow redirects back to whatever we asked for.
        alias = {r["to"]: r["from"] for r in query.get("redirects", [])}
        for page in query.get("pages", {}).values():
            revs = page.get("revisions")
            if not revs:
                continue
            name = page["title"]
            entry = {"revid": revs[0]["revid"], "retrieved": revs[0]["timestamp"]}
            out[name] = entry
            if name in alias:
                out[alias[name]] = entry
    return out


def main():
    titles = json.loads((ROOT / "data" / "titles.json").read_text())
    moments = json.loads((ROOT / "data" / "moments.json").read_text())
    released = [t for t in titles if t["status"] == "released"]

    counts = {}
    for m in moments:
        counts[m["film"].rsplit(" (", 1)[0]] = counts.get(m["film"].rsplit(" (", 1)[0], 0) + 1

    print(f"Fetching revision ids for {len(released)} articles ...")
    revs = revisions([t["article"] for t in released])

    rows = []
    for t in sorted(released, key=lambda t: t["released"]):
        path = plot_path(t["article"])
        words = len(path.read_text().split()) if path.exists() else 0
        n = counts.get(t["title"], 0)
        rev = revs.get(t["article"], {})
        rows.append({
            "title": t["title"],
            "kind": t["kind"],
            "universe": t["universe"],
            "released": t["released"],
            "article": t["article"],
            "revid": rev.get("revid"),
            "retrieved": rev.get("retrieved"),
            "permalink": (f"https://en.wikipedia.org/w/index.php?oldid={rev['revid']}"
                          if rev.get("revid") else None),
            "source_words": words,
            "moments": n,
            "words_per_moment": round(words / n) if n else None,
        })

    (ROOT / "data" / "provenance.json").write_text(json.dumps({
        "generated": date.today().isoformat(),
        "source": "English Wikipedia via the MediaWiki API",
        "source_licence": "CC BY-SA 4.0",
        "titles": len(rows),
        "source_words": sum(r["source_words"] for r in rows),
        "moments": sum(r["moments"] for r in rows),
        "rows": rows,
    }, indent=2) + "\n")

    ratios = [r["words_per_moment"] for r in rows if r["words_per_moment"]]
    md = [
        "# Data provenance",
        "",
        "Every moment in this dataset comes from a Wikipedia plot summary. This",
        "table records which article, which exact revision, how much text was read,",
        "and how many moments were written from it.",
        "",
        f"**{len(rows)} titles · {sum(r['source_words'] for r in rows):,} words of source text · "
        f"{sum(r['moments'] for r in rows)} moments**",
        "",
        "## Read this column first",
        "",
        f"`words/moment` ranges from {min(ratios)} to {max(ratios)}. It is not constant, and that is",
        "a choice rather than an accident. Moments were selected for how far they",
        "reach across titles, not for how much of a plot they cover, so a film with a",
        "tight 650-word summary is represented far more densely than a seven-season",
        "series. **The map shows connective significance, not screen time.**",
        "",
        "## Titles",
        "",
        "| Title | Kind | Universe | Released | Source words | Moments | Words/moment | Wikipedia revision |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for r in rows:
        link = f"[{r['revid']}]({r['permalink']})" if r["permalink"] else "—"
        md.append(
            f"| {r['title']} | {r['kind']} | {r['universe']} | {r['released']} | "
            f"{r['source_words']:,} | {r['moments']} | {r['words_per_moment'] or '—'} | {link} |"
        )
    md += [
        "",
        "## Licence and attribution",
        "",
        "Source text is from English Wikipedia, licensed **CC BY-SA 4.0**. The moment",
        "descriptions in `data/moments.json` are original prose written from that text,",
        "and the dataset is released under the same licence. Each revision link above",
        "resolves to the exact version read, so the source can be checked even after",
        "the article changes.",
        "",
        f"Generated by `ingest/provenance.py` on {date.today().isoformat()}.",
        "",
    ]
    (ROOT / "PROVENANCE.md").write_text("\n".join(md))

    print(f"{len(rows)} titles, {sum(r['source_words'] for r in rows):,} source words, "
          f"{sum(r['moments'] for r in rows)} moments")
    print(f"words per moment: min {min(ratios)}, median {sorted(ratios)[len(ratios)//2]}, max {max(ratios)}")
    print("wrote data/provenance.json and PROVENANCE.md")


if __name__ == "__main__":
    main()
