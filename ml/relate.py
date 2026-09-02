"""relate.py - propose typed, directional edges without asking a human.

Cosine similarity (embed.py) measures "these two sound alike". It does not
measure "this caused that", and it systematically misses the connections that
matter most: Samuel Sterns takes Banner's blood in a cut in 2008 and pays off
seventeen years later, but the two descriptions share almost no vocabulary.

This step uses three signals instead, none of which needs a judgement call:

  1. SHARED RARE ENTITIES
     Proper nouns that appear in only a few moments are strong evidence of a
     real link. "Tony Stark" appears everywhere and means nothing; "Quantum
     Realm" appears in five moments and means those five belong together.
     Whether the shared entity is a person or a thing decides the edge type.

  2. CHRONOLOGY
     A setup-payoff is directional: the earlier moment sets up the later one.
     Release order gives us that direction for free.

  3. ADJACENCY WITHIN A TITLE
     Consecutive moments from the same film are timeline-adjacent by
     construction.

Run:    .venv/bin/python ml/relate.py
Output: ml/out/proposed_edges.json

Nothing here is written to data/connections.json. These are proposals, and
anything already judged there is left alone.
"""
import collections
import itertools
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent / "out"

# An entity in more than this fraction of moments is too common to be evidence.
COMMON_CUTOFF = 0.06

# Places are the one class of entity that passes the rarity filter and still
# means nothing. "New York" appears in about ten moments - rare enough to look
# informative - but two events happening in New York are not connected, they
# are just both in New York. This was inflating centrality: "Three buildings
# holding a shield" ranked sixth most central in the MCU purely on London,
# New York and Hong Kong.
#
# A curated list is blunt, and it is a known limitation: a place that also
# carries story weight (Wakanda, Westview) is excluded here even where the
# link would have been real. Frequency alone cannot separate the two cases,
# so this errs toward dropping edges rather than keeping false ones.
PLACES = {
    "Earth", "New York", "New York City", "London", "Hong Kong", "Manhattan",
    "Harlem", "Asgard", "New Asgard", "Wakanda", "Sokovia", "Knowhere",
    "Kamar-Taj", "San Francisco", "Chicago", "Oakland", "Los Angeles",
    "Washington", "Brazil", "Afghanistan", "Norway", "Hala", "Talokan",
    "Westview", "Hell's Kitchen", "New Jersey", "Siberia", "Berlin", "Vienna",
    "Budapest", "Sakaar", "Jotunheim", "Svartalfheim", "Xandar", "Vormir",
    "Titan", "Ta Lo", "Madripoor", "New Orleans", "Portland", "Queens",
}
# Words that start sentences and would otherwise be mistaken for proper nouns.
SENTENCE_STARTERS = {
    "The", "A", "An", "In", "On", "At", "After", "Before", "During", "While",
    "With", "To", "He", "She", "They", "His", "Her", "It", "When", "As", "From",
    "For", "And", "But", "By", "Rather", "Unable", "Believing", "Living", "Held",
    "Sent", "Cornered", "Demonstrating", "Watching", "Investigating", "Fleeing",
    "Realising", "Denied", "Carrying", "Using", "Returning", "Offered", "Newly",
    "Mortally", "Wearing", "Twenty", "Five", "Half", "Three", "Two", "One",
    "Posing", "Imprisoned", "Stopping", "Having", "Only", "Both", "Their",
    "Disgusted", "Humiliated", "Ruined", "Sleepless", "Now", "Then", "Later",
    "Thousands", "Eons", "According", "Because", "This", "That", "These",
}

# Entities that name a person get shared-character; the rest get shared-object.
# Crude but effective: people have forenames, objects and places usually don't.
PERSON_HINT = re.compile(
    r"\b(Stark|Rogers|Parker|Banner|Romanoff|Barton|Odin|Loki|Thor|Thanos|Quill|"
    r"Gamora|Nebula|Rocket|Groot|Drax|Strange|Wanda|Pietro|Vision|Ultron|Fury|"
    r"Coulson|Barnes|Wilson|Rhodes|Pym|Lang|Hope|Janet|Ava|Killian|Toomes|"
    r"T'Challa|Killmonger|Klaue|Zemo|Yondu|Ego|Hela|Valkyrie|Danvers|Talos|"
    r"Carter|Erskine|Schmidt|Zola|Pierce|Ross|Blonsky|Sterns|Vanko|Hammer|"
    r"Stane|Yinsen|Potts|Hogan|Kaecilius|Mordo|Wong|Dormammu|Malekith|Frigga|"
    r"Selvig|Foster|Beck|Jameson|MJ|Ned|Liz|Gwen|Harry|Octavius|Connors|May)\b"
)


def entities(text):
    found = set()
    for match in re.finditer(r"\b([A-Z][a-zA-Z.'\-]+(?:\s+[A-Z][a-zA-Z.'\-]+)*)", text):
        parts = [p for p in match.group(1).split() if p not in SENTENCE_STARTERS]
        if not parts:
            continue
        name = " ".join(parts)
        if len(name) > 3:
            found.add(name)
    return found


def year_of(moment):
    match = re.search(r"\((\d{4})\)", moment.get("film", ""))
    return int(match.group(1)) if match else 0


def main():
    moments = json.loads((DATA / "moments.json").read_text())
    judged = {
        frozenset((c["source"], c["target"]))
        for c in json.loads((DATA / "connections.json").read_text())
    }

    ents = {m["id"]: entities(m["description"]) for m in moments}
    freq = collections.Counter(e for s in ents.values() for e in s)
    cutoff = max(2, int(len(moments) * COMMON_CUTOFF))
    informative = {e for e, n in freq.items()
                   if 2 <= n <= cutoff and e not in PLACES}

    by_id = {m["id"]: m for m in moments}
    proposals = []

    for a, b in itertools.combinations(moments, 2):
        if frozenset((a["id"], b["id"])) in judged:
            continue
        shared = ents[a["id"]] & ents[b["id"]] & informative
        if not shared:
            continue

        ya, yb = year_of(a), year_of(b)
        same_film = a.get("film") == b.get("film")
        person = any(PERSON_HINT.search(s) for s in shared)

        # A shared name across universes is not a shared person. The Reed
        # Richards the Illuminati lose on Earth-838 is not the Reed Richards
        # of Earth-828; the Peggy Carter of 1946 is not the one Wanda kills;
        # the Riri Williams of the zombie timeline is a different Riri. Entity
        # matching cannot see this and it was a recurring source of false
        # edges in the low-precision bands.
        #
        # Cross-universe links are not banned outright - No Way Home makes
        # some of them the point of the project - but a person's name alone
        # cannot carry one. An object or concept still can, demoted to a
        # theme-echo, because a rhyme across universes is a real observation
        # even when the causal chain is not.
        cross_universe = a.get("universe") != b.get("universe")
        if cross_universe:
            if person:
                continue
            proposals.append({
                "source": a["id"], "target": b["id"],
                "type": "theme-echo",
                "evidence": sorted(shared),
                "weight": round(sum(1.0 / freq[s] for s in shared) * 0.5, 4),
                "cross_universe": True,
                "proposed_by": "relate.py/entities+chronology",
            })
            continue

        if same_film:
            kind, source, target = "timeline-adjacent", a["id"], b["id"]
        elif ya and yb and ya != yb:
            # The earlier moment sets up the later one.
            kind = "setup-payoff"
            source, target = (a["id"], b["id"]) if ya < yb else (b["id"], a["id"])
        else:
            kind = "shared-character" if person else "shared-object"
            source, target = a["id"], b["id"]

        if kind == "setup-payoff" and not person:
            kind = "shared-object"

        # Rarer shared entities are stronger evidence; more of them, stronger still.
        weight = sum(1.0 / freq[s] for s in shared)

        proposals.append({
            "source": source,
            "target": target,
            "type": kind,
            "evidence": sorted(shared),
            "weight": round(weight, 4),
            "proposed_by": "relate.py/entities+chronology",
        })

    proposals.sort(key=lambda p: p["weight"], reverse=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "proposed_edges.json").write_text(json.dumps(proposals, indent=2) + "\n")

    kinds = collections.Counter(p["type"] for p in proposals)
    cross = sum(1 for p in proposals if by_id[p["source"]]["film"] != by_id[p["target"]]["film"])
    print(f"{len(moments)} moments, {len(informative)} informative entities "
          f"(seen 2..{cutoff} times)")
    print(f"{len(proposals):,} proposed edges  ({cross:,} cross-title)")
    for k, n in kinds.most_common():
        print(f"  {n:5d}  {k}")
    print("\nStrongest evidence:")
    for p in proposals[:10]:
        a, b = by_id[p["source"]], by_id[p["target"]]
        print(f'  {p["weight"]:.2f} {p["type"]:17s} {", ".join(p["evidence"][:2])[:34]}')
        print(f'       {a["title"][:44]:46s} -> {b["title"][:44]}')
    print("\nwrote ml/out/proposed_edges.json")


if __name__ == "__main__":
    main()
