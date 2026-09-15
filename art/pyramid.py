"""pyramid.py - derive what each tile depicts, then render the pyramid.

    .venv/bin/python art/pyramid.py plan 3     what every tile at levels 0..3 shows
    .venv/bin/python art/pyramid.py build 3    render them

A tile is not decoration placed near the data. A tile IS a region of the graph,
and what it depicts is decided by the moments and connections whose coordinates
fall inside its bounds.

    level 0     1 tile      the whole map
    level 1     4 tiles     each a quadrant of level 0
    level 2    16 tiles
    level 3    64 tiles

Each tile is rendered from its PARENT's corresponding quadrant, upscaled and
re-inked rather than invented from nothing. That single decision solves three
problems that were previously being fought separately:

  Neighbouring tiles agree, because they were painted out of the same parent.
  Zooming is coherent, because a child is literally a detail of its parent.
  Content sharpens with depth, because a deeper tile contains fewer moments and
  so takes a more specific prompt. At level 0 the prompt describes a continent;
  at level 3 it describes two events and the link between them.

The prompt is built from THREE things, in descending order of weight:

  1. The connections inside the tile. An edge is the subject, not the two
     moments separately: "a scepter's gem, later set into a synthetic being's
     brow" says more than either moment alone and is what the project is about.
  2. The moments themselves, as concrete nouns rather than plot summary.
  3. The threads present, which set the palette and the terrain type.

Depiction rules, enforced in code rather than left to the prompt: no character
names, no costumes, no insignia. Objects, places and forces only.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
GRAPH = ROOT / "ml" / "out" / "graph.json"

# Names are stripped before a prompt is built. The project depicts what happens,
# never who it happens to: no likenesses, no costumes, no insignia.
NAME = re.compile(
    r"\b(Tony Stark|Steve Rogers|Peter Parker|Bruce Banner|Natasha Romanoff|"
    r"Clint Barton|Nick Fury|Phil Coulson|Wanda Maximoff|Pietro Maximoff|"
    r"Thor|Loki|Odin|Hela|Thanos|Gamora|Nebula|Rocket|Groot|Drax|Mantis|"
    r"Peter Quill|Stephen Strange|Doctor Strange|Wong|Mordo|T'Challa|Shuri|"
    r"Killmonger|Erik Stevens|Ramonda|Namor|Scott Lang|Hope van Dyne|"
    r"Hank Pym|Janet van Dyne|Carol Danvers|Monica Rambeau|Kamala Khan|"
    r"Sam Wilson|Bucky Barnes|James Barnes|John Walker|Yelena Belova|"
    r"Matt Murdock|Wilson Fisk|Maya Lopez|Jessica Jones|Luke Cage|"
    r"Danny Rand|Frank Castle|Agatha Harkness|Billy|Tommy|Vision|Ultron|"
    r"Jane Foster|Gorr|Shang-Chi|Wenwu|Xialing|Riri Williams|Jennifer Walters|"
    r"Emil Blonsky|Samuel Sterns|Thaddeus Ross|Peggy Carter|Howard Stark|"
    r"Obadiah Stane|Ivan Vanko|Aldrich Killian|Adrian Toomes|Quentin Beck|"
    r"MJ|Ned|Gwen Stacy|Harry Osborn|Norman Osborn|Otto Octavius|"
    r"Jean Grey|Ego|Yondu|Sersi|Ikaris|Arishem|Kang|Bob|Sylvie|Mobius)\b")

# Terrain vocabulary per edge type. A setup-payoff is a river: one thing
# flowing into another. A theme-echo is a pair of matching landforms with no
# channel between them. The geography carries the semantics.
TERRAIN = {
    "setup-payoff": "a river running from one landform into another",
    "shared-object": "two sites joined by a worn road",
    "shared-character": "two settlements on one long ridge",
    "timeline-adjacent": "adjacent valleys separated by a low pass",
    "theme-echo": "two distant landforms of matching shape, unconnected",
}


# Words that carry an image. A description is reduced to these plus what
# surrounds them, because a diffusion model can draw a crater or a hammer and
# cannot draw "unwilling to let him use its power".
IMAGERY = re.compile(
    r"\b(cave|crater|desert|forest|river|sea|ocean|ice|snow|mountain|valley|"
    r"island|city|tower|bridge|road|tunnel|cavern|pit|chasm|fissure|ruin|"
    r"wreck|wreckage|ship|vessel|plane|train|lab|laboratory|workshop|forge|"
    r"prison|cell|vault|throne|temple|shrine|gate|portal|doorway|stair|"
    r"hammer|sword|axe|shield|spear|scepter|sceptre|gem|stone|cube|orb|ring|"
    r"crown|mask|helmet|armour|armor|suit|reactor|machine|engine|core|"
    r"lightning|fire|flame|smoke|ash|dust|storm|flood|wave|light|shadow|"
    r"darkness|glow|beam|blast|explosion|grave|memorial|statue|monument)\w*",
    re.I)


def clean(text, limit=90):
    """A description reduced to what can actually be drawn.

    Two jobs. Names come out, because the project depicts what happens and
    never who it happens to. And the sentence is trimmed toward its concrete
    nouns: an image model can draw a crater, a hammer or a flooded street, and
    cannot draw "unwilling to let him use its power".

    An earlier version substituted "a figure" for each name, which produced
    "Facing an a figure who has won the Infinity Stones". Dropping the subject
    outright reads better than patching a placeholder into its place.
    """
    text = NAME.sub("", text)
    text = re.sub(r"\b(?:he|she|they|his|her|their|him|them)\b", "", text, flags=re.I)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,;")
    first = re.split(r"(?<=[.;])\s", text)[0]

    # Prefer the clause containing the strongest imagery.
    clauses = [c.strip(" ,") for c in re.split(r",", first) if c.strip()]
    if clauses:
        best = max(clauses, key=lambda c: len(IMAGERY.findall(c)))
        if IMAGERY.search(best):
            first = best

    return " ".join(first.split()[:limit]).rstrip(",.;")


def load():
    g = json.loads(GRAPH.read_text())
    return g["nodes"], g["edges"], g.get("meta", {})


def tile_bounds(level, col, row):
    """Bounds of one tile in the map's 0..1000 coordinate space."""
    n = 2 ** level
    step = 1000 / n
    return col * step, row * step, (col + 1) * step, (row + 1) * step


def contents(nodes, edges, bounds, mode="threads"):
    """Which moments and connections fall inside this tile."""
    xk, yk = ("x", "y") if mode == "threads" else ("sx", "sy")
    x0, y0, x1, y1 = bounds
    inside = [n for n in nodes if x0 <= n[xk] < x1 and y0 <= n[yk] < y1]
    ids = {n["id"] for n in inside}
    # An edge belongs to the tile if either end is inside it, so a connection
    # crossing a tile boundary is depicted on both sides rather than lost.
    links = [e for e in edges if e["source"] in ids or e["target"] in ids]
    return inside, links


def describe(inside, links, nodes_by_id, level):
    """Turn the contents of a tile into a description of what it depicts."""
    if not inside:
        return None

    detail = min(level, 3)
    threads = sorted({n["community"] for n in inside})

    # Connections first: an edge is the subject, because the relationship is
    # what the project is about. The most confident ones lead.
    links = sorted(links, key=lambda e: (e.get("verdict") != "confirmed",
                                         -e.get("weight", 0)))
    phrases = []
    for e in links[: 1 + detail]:
        a, b = nodes_by_id.get(e["source"]), nodes_by_id.get(e["target"])
        if not a or not b:
            continue
        shape = TERRAIN.get(e.get("type"), TERRAIN["theme-echo"])
        if detail >= 2:
            phrases.append(f"{shape}, from {clean(a['description'], 14)} "
                           f"to {clean(b['description'], 14)}")
        else:
            phrases.append(shape)

    # Then the moments, as objects in the landscape.
    hubs = sorted(inside, key=lambda n: -n.get("centrality", 0))
    for n in hubs[: 1 + detail * 2]:
        phrases.append(clean(n["description"], 10 + detail * 6))

    # At the shallow levels the tile holds hundreds of moments, so naming two
    # of them says nothing. Describe the territory instead: how many threads
    # meet here, and the dominant imagery across everything inside.
    if level <= 1:
        words = {}
        for n in inside:
            for w in IMAGERY.findall(n["description"]):
                w = w.lower()
                words[w] = words.get(w, 0) + 1
        top = [w for w, _ in sorted(words.items(), key=lambda kv: -kv[1])[:8]]
        phrases = [f"{len(threads)} distinct territories meeting",
                   "coastline and inland plateaus",
                   ", ".join(top)]

    scale = ["a whole continent seen from very high above",
             "a region of that continent",
             "a district within that region",
             "a small area, individual features visible"][min(level, 3)]

    return {
        "scale": scale,
        "threads": threads,
        "moments": len(inside),
        "links": len(links),
        "prompt": "; ".join(phrases[: 2 + detail * 2]),
    }


def plan(max_level=3, mode="threads"):
    nodes, edges, _ = load()
    by_id = {n["id"]: n for n in nodes}
    out = []
    for level in range(max_level + 1):
        n = 2 ** level
        filled = 0
        for row in range(n):
            for col in range(n):
                b = tile_bounds(level, col, row)
                inside, links = contents(nodes, edges, b, mode)
                d = describe(inside, links, by_id, level)
                if d:
                    filled += 1
                    out.append({"level": level, "col": col, "row": row, **d})
        print(f"level {level}: {filled}/{n*n} tiles have content")
    (HERE / "pyramid-plan.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n{len(out)} tiles to render -> art/pyramid-plan.json")
    return out




# ---------------------------------------------------------------- rendering

TILE = 768
STYLE = ("silver age comic book map illustration, flat bold ink colour, "
         "ben-day dot halftone, heavy black ink linework, printed on aged "
         "paper, seen from directly overhead, continuous terrain to every edge")
NEGATIVE = ("horizon, sky, clouds, perspective, vanishing point, text, "
            "lettering, labels, place names, speech bubble, frame, border, "
            "margin, people, faces, figures, photorealistic, 3d, blurry")


def load_pipe():
    import torch
    from diffusers import StableDiffusionXLImg2ImgPipeline
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("mps")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def build(max_level=2, mode="threads"):
    """Render the pyramid top down, each tile inherited from its parent.

    Level 0 is inked from the terrain render, so its composition comes from the
    graph rather than from the model's imagination. Every deeper tile starts as
    its parent's corresponding quadrant, upscaled: neighbours therefore already
    agree with each other, and a child is by construction a detail of its
    parent rather than a separate picture that has to be talked into matching.

    Strength falls with depth. A shallow tile is mostly invention over a flat
    colour field; a deep tile is mostly its parent, sharpened.
    """
    import time
    import torch
    from PIL import Image
    from terrain import render_base

    plan_by = {(t["level"], t["col"], t["row"]): t
               for t in json.loads((HERE / "pyramid-plan.json").read_text())}
    out = HERE / "tiles"
    out.mkdir(exist_ok=True)
    pipe = load_pipe()

    base = render_base(size=TILE, mode=mode)
    parents = {}
    made = 0

    for level in range(max_level + 1):
        n = 2 ** level
        strength = [0.62, 0.5, 0.42, 0.36][min(level, 3)]
        for row in range(n):
            for col in range(n):
                spec = plan_by.get((level, col, row))
                if not spec:
                    continue
                if level == 0:
                    src = base
                else:
                    p = parents.get((level - 1, col // 2, row // 2))
                    if p is None:
                        continue
                    half = TILE // 2
                    src = p.crop(((col % 2) * half, (row % 2) * half,
                                  (col % 2) * half + half,
                                  (row % 2) * half + half)
                                 ).resize((TILE, TILE), Image.LANCZOS)

                t0 = time.time()
                img = pipe(
                    prompt=f"{STYLE}, {spec['scale']}, {spec['prompt']}",
                    negative_prompt=NEGATIVE, image=src, strength=strength,
                    guidance_scale=6.5, num_inference_steps=26,
                    generator=torch.Generator("mps").manual_seed(
                        1000 * level + row * 64 + col),
                ).images[0]
                img.save(out / f"{level}_{col}_{row}.png")
                parents[(level, col, row)] = img
                made += 1
                print(f"  L{level} ({col},{row})  {time.time()-t0:4.0f}s  "
                      f"{spec['moments']}m {spec['links']}l")

    print(f"\n{made} tiles -> art/tiles/")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plan"
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    if cmd == "build":
        plan(depth)
        build(depth)
    elif cmd == "plan":
        tiles = plan(depth)
        print("\nsample prompts:")
        for t in tiles:
            if t["level"] in (0, 2) and t["links"]:
                print(f"\n  L{t['level']} ({t['col']},{t['row']})  "
                      f"{t['moments']} moments, {t['links']} links")
                print(f"    {t['scale']}")
                print(f"    {t['prompt'][:260]}")
                if t["level"] == 2:
                    break
