"""Build the title registry: every MCU film and series, with release dates.

    python3 ingest/registry.py

Writes data/titles.json. This is the "what exists" half of the pipeline — it is
what lets the project notice a new release without anyone editing a list by
hand. The "what happens in it" half is ingest/plots.py.

Source is the rendered HTML of Wikipedia's list articles rather than wikitext,
because the tables are far more regular once rendered. Wikidata would be the
tidier source but its query service is unreliable.
"""
import json
import pathlib
import re
import urllib.parse
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from datetime import date
from html.parser import HTMLParser

from wiki import page_html

ROOT = pathlib.Path(__file__).resolve().parent.parent
ISO = re.compile(r"\((\d{4}-\d{2}-\d{2})\)")

SOURCES = [
    ("List of Marvel Cinematic Universe films", "film", "mcu"),
    ("List of Marvel Cinematic Universe television series (Marvel Studios)", "series", "mcu"),
    ("List of Marvel Cinematic Universe television series (Marvel Television)", "series", "marvel-tv"),
]

# In scope because No Way Home pulls them into the multiverse. See data/schema.md.
MULTIVERSE = [
    ("Spider-Man (2002 film)", "Spider-Man", 2002, "raimi"),
    ("Spider-Man 2", "Spider-Man 2", 2004, "raimi"),
    ("Spider-Man 3", "Spider-Man 3", 2007, "raimi"),
    ("The Amazing Spider-Man (film)", "The Amazing Spider-Man", 2012, "webb"),
    ("The Amazing Spider-Man 2", "The Amazing Spider-Man 2", 2014, "webb"),
]


def fetch_html(page):
    return page_html(page)


class TableParser(HTMLParser):
    """Collect every table row as a list of (text, first_href) cells."""

    def __init__(self):
        super().__init__()
        self.rows, self._row, self._cell, self._buf, self._href = [], None, None, "", None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("th", "td") and self._row is not None:
            self._cell, self._buf, self._href = tag, "", None
        elif tag == "a" and self._cell and not self._href:
            self._href = dict(attrs).get("href", "")

    def handle_data(self, data):
        if self._cell:
            self._buf += data

    def handle_endtag(self, tag):
        if tag in ("th", "td") and self._row is not None:
            self._row.append((re.sub(r"\s+", " ", self._buf).strip(), self._href))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None


def titles_from(page, kind, universe):
    parser = TableParser()
    parser.feed(fetch_html(page))
    found = {}
    for row in parser.rows:
        if len(row) < 2:
            continue
        title, href = row[0]
        if not href or "/wiki/" not in href or "redlink" in href:
            continue
        # Strip Wikipedia footnote markers, e.g. "Thunderbolts* [e]".
        title = re.sub(r"\s*\[[a-z0-9]{1,3}\]\s*$", "", title).strip()
        # Multi-season shows repeat with the season number in the first cell;
        # those rows are extra seasons of a series already captured.
        if not re.search(r"[A-Za-z]", title):
            continue
        # The sortable ISO date in parentheses only appears in the real
        # release tables, which filters out box-office and summary tables.
        # Films put it in the second cell, series in the fourth (after
        # season and episode counts), so scan the whole row.
        for cell, _ in row[1:]:
            m = ISO.search(cell)
            if m and title and title not in found:
                article = urllib.parse.unquote(href.split("/wiki/")[1]).replace("_", " ")
                found[title] = {
                    "title": title,
                    "article": article,
                    "released": m.group(1),
                    "kind": kind,
                    "universe": universe,
                }
                break
    return list(found.values())


def main():
    today = date.today().isoformat()
    out = ROOT / "data" / "titles.json"

    titles, failed = [], []
    for page, kind, universe in SOURCES:
        try:
            got = titles_from(page, kind, universe)
        except Exception as exc:
            print(f"  !! {page}: {exc}  (keeping previously known titles)")
            failed.append(page)
            continue
        print(f"  {len(got):3d}  {page}")
        titles += got

    # Fall back to what we already knew for anything this run couldn't fetch.
    # A failed request must never shrink the registry — silently losing titles
    # is worse than being briefly stale. Freshly fetched entries come first so
    # they win the dedup below and corrections actually take effect.
    if out.exists():
        titles += [t for t in json.loads(out.read_text()) if re.search(r"[A-Za-z]", t["title"])]

    for article, title, year, universe in MULTIVERSE:
        titles.append({
            "title": title,
            "article": article,
            "released": f"{year}-01-01",
            "kind": "film",
            "universe": universe,
        })

    seen, uniq = set(), []
    for t in titles:
        if t["article"] in seen:
            continue
        seen.add(t["article"])
        t["status"] = "released" if t["released"] <= today else "upcoming"
        uniq.append(t)
    uniq.sort(key=lambda t: t["released"])

    out.write_text(json.dumps(uniq, indent=2) + "\n")
    rel = sum(1 for t in uniq if t["status"] == "released")
    print(f"\n{len(uniq)} titles ({rel} released, {len(uniq) - rel} upcoming) -> {out.relative_to(ROOT)}")
    if failed:
        print(f"{len(failed)} source(s) failed; rerun to refresh them.")


if __name__ == "__main__":
    main()
