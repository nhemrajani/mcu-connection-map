"""atlas.py - render every moment as a scene, for the canvas to place.

    .venv/bin/python art/atlas.py           render all 599
    .venv/bin/python art/atlas.py 20        render the 20 most central

Earlier attempts built ONE image: a terrain map, then a quilt of nine scenes
with the gaps painted in. Neither scales to the whole corpus. The layout is
1000x1000 units with a median gap of 12 units between neighbours, so drawing
every moment at a size where you can see it means a canvas around 42,000px
square. That is 1.8 gigapixels, which cannot be generated, stored or served.

So the world is not composited here at all. This renders 599 independent
scenes and the browser places them at their graph coordinates. The blending
that makes it one world instead of 599 pictures happens at draw time.

HOW THE BLEND WORKS

Each scene is saved with a soft radial alpha baked into it: opaque through the
middle, falling smoothly to nothing at the edge. Drawn at their layout
positions the scenes overlap, and overlapping soft edges cross-dissolve.

The important part is WHY that shows connections rather than just looking
pretty. The layout is not arbitrary. Moments sit near each other because the
graph put them there, and the graph put them there because they are connected.
So two scenes only blend into one another if the model judged them related:
the dissolve between them IS the edge, drawn. Unconnected moments end up far
apart and never touch.

Zooming in spreads the scenes and the blend bands widen, which is why detail
and connection both increase together as you go in.

DRAW ORDER

Scenes are rendered most-central last, so hub moments sit on top where scenes
pile up. Centrality already measures how much of the story runs through a
moment, so the thing the graph considers load-bearing is also the thing you
see. It costs nothing: the ordering is metadata the canvas reads.

OUTPUT

  art/world/<id>.webp        512px, alpha, ~40KB
  art/world/thumb/<id>.webp  128px, alpha, ~4KB, used when zoomed out
  art/world/manifest.json    id, position, draw order, size on disk

WebP because 599 PNGs with alpha is about 240MB and will not go in a repo;
the same set as WebP is around 27MB. Seeds are derived from the moment id, so
re-running reproduces the same world rather than a different one.
"""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from quilt import STYLE, NEGATIVE, fit, subject

OUT = HERE / "world"
FULL, THUMB = 512, 128

# Where the alpha falloff starts and ends, as a fraction of the half-width.
# Opaque well past the middle so the subject is never washed out, then a long
# smooth tail so the join between neighbours has room to happen.
SOLID, FADE = 0.55, 1.0


def seed_for(mid):
    """A stable seed per moment, so the world is the same world each run."""
    return int(hashlib.sha256(mid.encode()).hexdigest()[:8], 16) % (2 ** 31)


def falloff(size):
    """A radial alpha mask: opaque core, smooth edge.

    Linear falloff leaves a visible ring where the gradient starts. Smoothstep
    has zero derivative at both ends, so the opaque core and the transparent
    edge both join the ramp without an edge of their own.
    """
    mask = Image.new("L", (size, size))
    px = mask.load()
    c = (size - 1) / 2
    for y in range(size):
        dy = (y - c) / c
        for x in range(size):
            dx = (x - c) / c
            r = math.hypot(dx, dy)
            if r <= SOLID:
                a = 1.0
            elif r >= FADE:
                a = 0.0
            else:
                t = (r - SOLID) / (FADE - SOLID)
                a = 1 - (t * t * (3 - 2 * t))       # smoothstep
            px[x, y] = int(a * 255)
    return mask


def main(limit=None):
    from diffusers import StableDiffusionXLPipeline

    g = json.loads((ROOT / "ml" / "out" / "graph.json").read_text())
    nodes = sorted(g["nodes"], key=lambda m: m.get("centrality", 0))
    if limit:
        nodes = nodes[-limit:]

    OUT.mkdir(exist_ok=True)
    (OUT / "thumb").mkdir(exist_ok=True)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to("mps")
    pipe.set_progress_bar_config(disable=True)

    mask_full = falloff(FULL)
    mask_thumb = mask_full.resize((THUMB, THUMB), Image.LANCZOS)

    entries, started, done = [], time.time(), 0
    for i, m in enumerate(nodes):
        path = OUT / f'{m["id"]}.webp'
        if not path.exists():
            prompt = fit(subject(m), STYLE)
            img = pipe(prompt=prompt, negative_prompt=NEGATIVE,
                       num_inference_steps=6, guidance_scale=2.0,
                       height=FULL, width=FULL,
                       generator=torch.Generator("mps").manual_seed(seed_for(m["id"]))
                       ).images[0].convert("RGB")
            img.putalpha(mask_full)
            img.save(path, "WEBP", quality=82)
            t = img.resize((THUMB, THUMB), Image.LANCZOS)
            t.putalpha(mask_thumb)
            t.save(OUT / "thumb" / f'{m["id"]}.webp', "WEBP", quality=80)
            done += 1
            if done % 25 == 0:
                rate = (time.time() - started) / done
                left = (len(nodes) - i - 1) * rate
                print(f"  {i+1}/{len(nodes)}  {rate:.1f}s each, ~{left/60:.0f} min left",
                      flush=True)

        entries.append({
            "id": m["id"], "x": m["x"], "y": m["y"],
            "order": i,                       # ascending centrality = draw order
            "bytes": path.stat().st_size,
        })

    total = sum(e["bytes"] for e in entries)
    (OUT / "manifest.json").write_text(json.dumps({
        "scenes": entries,
        "meta": {"count": len(entries), "full": FULL, "thumb": THUMB,
                 "solid": SOLID, "fade": FADE,
                 "megabytes": round(total / 1e6, 1)},
    }, indent=2) + "\n")
    print(f"\n{len(entries)} scenes, {total/1e6:.1f} MB -> art/world/")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
