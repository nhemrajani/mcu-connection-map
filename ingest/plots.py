"""Fetch the narrative text for every released title in data/titles.json.

    python3 ingest/plots.py            # only what's missing
    python3 ingest/plots.py --refresh  # refetch everything

Writes one plain-text file per title into plots/ (gitignored — it's a cache,
regenerate it any time). This is the raw material the extraction step reads.

Wikipedia stores narrative text in three different shapes, so this handles all
three:

  film    -> a "Plot" section, 400-700 words by WP:FILMPLOT policy
  series  -> episode summaries inlined in the article's "Episodes" section
  series  -> episode summaries living in separate "<Show> season N" articles,
             in which case we follow those links

Text is CC BY-SA. Keep the attribution if you redistribute it.
"""
import json
import pathlib
import re
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import time
from wiki import sections, wikitext

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "plots"
NARRATIVE = ("Plot", "Plot summary", "Synopsis", "Premise")
# Anthology articles (Marvel One-Shots) have no single Plot section; each short
# gets its own italicised, year-stamped heading instead.
ANTHOLOGY = re.compile(r"^<i>.+</i>\s*\((?:series\s*)?\d{4}")
# Season articles look like "Loki season 1"; long-running shows instead use a
# single "List of <Show> episodes" article.
EPISODE_ARTICLE = re.compile(r"(season \d+$|^List of .* episodes$)", re.I)
SEASON_HEADING = re.compile(r"^Season \d+\b", re.I)


def clean(text):
    """Strip wiki markup down to readable prose."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"<ref.*?</ref>", "", text, flags=re.S)
    text = re.sub(r"</?onlyinclude>|</?includeonly>|</?noinclude>", "", text)
    # Keep episode titles and summaries, drop the rest of the table scaffolding.
    text = re.sub(r"\|\s*(ShortSummary|Title)\s*=\s*", "\n", text)
    text = re.sub(r"\{\{[Ee]pisode list.*?\n", "\n", text)
    text = re.sub(r"\|\s*\w+\s*=[^\n|]*", "", text)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]", r"\2", text)
    text = re.sub(r"'''?", "", text)
    text = re.sub(r"==+[^=]*==+", "", text)
    # Orphaned template parameters, e.g. a stray " |2021|1|15}}" left behind
    # when the opening {{Episode list was stripped. Leading space matters.
    text = re.sub(r"^\s*[|!{}].*$", "", text, flags=re.M)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def season_articles(episodes_wikitext, show):
    """Find '<Show> season N' articles referenced from an Episodes section.

    Wikipedia points at season articles three different ways, and only one of
    them is an ordinary wikilink:

        {{:Loki season 1}}                      full-page transclusion
        {{main|Loki season 1{{!}}...}}          hatnote
        [[Daredevil season 1]]                  plain link

    Miss these and a multi-season show yields only its one-paragraph Premise,
    which is what happened on the first run.
    """
    patterns = (
        r"\{\{:\s*([^}|]+?)\s*\}\}",
        r"\{\{\s*[Mm]ain\s*\|\s*([^}|]+?)\s*(?:\{\{!\}\}|\||\}\})",
        r"\[\[([^\]|]+?)(?:\|[^\]]*)?\]\]",
    )
    found = []
    for pattern in patterns:
        for m in re.finditer(pattern, episodes_wikitext):
            name = m.group(1).strip()
            if EPISODE_ARTICLE.search(name) and name not in found:
                found.append(name)
    return found


def narrative_for(article, kind):
    parts, used = [], []
    secs = sections(article)
    by_name = {s["line"]: s["index"] for s in secs}

    for name in NARRATIVE:
        if name in by_name:
            parts.append(clean(wikitext(article, by_name[name])))
            used.append(name)
            if kind == "film":
                return "\n\n".join(parts), used

    if not parts:
        for name, idx in by_name.items():
            if ANTHOLOGY.match(name):
                parts.append(clean(wikitext(article, idx)))
                used.append(name)
        if parts:
            return "\n\n".join(parts), used

    if "Episodes" in by_name or any(SEASON_HEADING.match(n) for n in by_name):
        got, sources = collect_episodes(article)
        parts += got
        used += sources

    return "\n\n".join(p for p in parts if p), used


def collect_episodes(article, depth=0, seen=None):
    """Gather episode summaries, following transclusions until we find prose.

    Wikipedia nests these arbitrarily deep. Agents of S.H.I.E.L.D. is the worst
    case and takes three hops:

        Agents of S.H.I.E.L.D.                    -> Episodes section
        List of Agents of S.H.I.E.L.D. episodes   -> Season N sections
        Agents of S.H.I.E.L.D. season 1           -> the actual episode tables

    So when a section turns out to be a stub that only transcludes something
    else, follow it rather than accepting the empty result.
    """
    seen = set() if seen is None else seen
    if depth > 2 or article in seen:
        return [], []
    seen.add(article)

    try:
        secs = sections(article)
    except Exception:
        return [], []

    wanted = [x["index"] for x in secs if SEASON_HEADING.match(x["line"])]
    if not wanted:
        wanted = [x["index"] for x in secs if x["line"] == "Episodes"]

    texts, sources = [], []
    for idx in wanted[:12]:
        try:
            raw = wikitext(article, idx)
        except Exception:
            continue
        text = clean(raw)
        if len(text.split()) >= 40:
            texts.append(text)
            if article not in sources:
                sources.append(article)
        else:
            # A stub section — follow whatever it points at.
            for nxt in season_articles(raw, article):
                sub_texts, sub_sources = collect_episodes(nxt, depth + 1, seen)
                texts += sub_texts
                sources += sub_sources
    return texts, sources


def main():
    refresh = "--refresh" in sys.argv
    OUT.mkdir(exist_ok=True)
    titles = json.loads((ROOT / "data" / "titles.json").read_text())
    released = [t for t in titles if t["status"] == "released"]

    total = 0
    for i, t in enumerate(released, 1):
        dest = OUT / (re.sub(r"[^\w\-. ]", "_", t["article"]) + ".txt")
        if dest.exists() and not refresh:
            total += len(dest.read_text().split())
            continue
        try:
            text, used = narrative_for(t["article"], t["kind"])
        except Exception as exc:
            print(f"  {i:3d}/{len(released)}  !! {t['title']}: {exc}")
            continue
        words = len(text.split())
        total += words
        if words < 50:
            print(f"  {i:3d}/{len(released)}  ?? {t['title']}: only {words} words {used}")
            continue
        dest.write_text(text)
        print(f"  {i:3d}/{len(released)}  {words:6d}w  {t['title'][:44]}")
        time.sleep(1.0)   # be a polite API citizen

    print(f"\n{total:,} words cached in {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
