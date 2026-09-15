"""quilt.py - a world made of scenes, with the gaps painted in.

    .venv/bin/python art/quilt.py 9      nine moments, one continuous canvas

The terrain approach produced something that looked like an atlas, because it
was: handing the model flat coloured regions gets flat coloured regions back.
No prompt fixes a cartographic base image.

What the canvas is supposed to be is a world made of SCENES - an unmasking, a
cube on a bench, a crater - sitting where the graph puts them and bleeding into
one another.

Three steps, and the ordering is the whole point:

  1. Each moment becomes its own scene, generated independently. Full prompt
     control, so the unmasking is actually an unmasking.
  2. Scenes are placed on one canvas at their real coordinates from the layout,
     so neighbours on the canvas are neighbours in the graph.
  3. The gaps between them are outpainted. This is the only step that has to
     negotiate anything, and it is the step already shown to leave no visible
     seam.

An earlier attempt generated each scene INSIDE its neighbour, conditioning on
the previous image. Continuity swallowed the content completely: four prompts
for four different events produced four views of the same room. Generating
independently and blending only the gaps keeps the two jobs apart.

Depiction rules, enforced before any prompt is built: no names, no costumes, no
insignia. Objects, places, weather and architecture only.
"""
import json
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pyramid import NAME, IMAGERY          # shared depiction rules

SCENE = 512            # each moment's own scene
GAP = 256              # painted band between scenes

STYLE = ("silver age comic book art, 1960s four colour printing, ben-day dot "
         "halftone, heavy black ink outlines, flat saturated colour, dramatic "
         "angular composition, aged newsprint")
NEGATIVE = ("text, lettering, speech bubble, caption, words, letters, "
            "signature, watermark, panel border, frame, gutter, page layout, "
            "map, atlas, chart, diagram, place names, "
            "photorealistic, 3d render, blurry, modern digital art")


def subject(moment):
    """What this moment looks like.

    Prefers the hand-written `visual` field, which is a composed scene with a
    subject, a setting and a framing. Falls back to clipping imagery out of
    `description`, which produces a fragment rather than a picture and is the
    reason earlier renders looked like nothing in particular.
    """
    if moment.get("visual"):
        return moment["visual"]
    text = NAME.sub("a figure", moment["description"])
    text = re.sub(r"\ba figure\b(?:\s+a figure\b)+", "a figure", text)
    first = re.split(r"(?<=[.;])\s", text)[0]
    # Prefer the clause carrying the most concrete imagery.
    clauses = [c.strip(" ,") for c in first.split(",") if c.strip()]
    if clauses:
        best = max(clauses, key=lambda c: len(IMAGERY.findall(c)))
        if IMAGERY.search(best):
            first = best
    return " ".join(first.split()[:26])


def pick(n=9):
    """N well-connected moments that are genuine neighbours in the layout.

    Taking the highest-centrality moments would scatter them across the map.
    Taking a spatial cluster keeps the canvas honest: things next to each other
    on it are next to each other in the graph.
    """
    g = json.loads((ROOT / "ml" / "out" / "graph.json").read_text())
    nodes = g["nodes"]
    seed = max(nodes, key=lambda m: m.get("centrality", 0))
    nodes.sort(key=lambda m: (m["x"] - seed["x"]) ** 2 + (m["y"] - seed["y"]) ** 2)
    return nodes[:n]


def main(n=9):
    from diffusers import (StableDiffusionXLPipeline,
                           StableDiffusionXLInpaintPipeline)

    moments = pick(n)
    side = int(n ** 0.5)
    print(f"{len(moments)} moments, {side}x{side} grid\n")

    txt = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("mps")
    txt.set_progress_bar_config(disable=True)

    # --- 1. each moment, generated on its own -----------------------------
    scenes = []
    for i, m in enumerate(moments):
        s = subject(m)
        t0 = time.time()
        img = txt(prompt=f"{STYLE}, {s}", negative_prompt=NEGATIVE,
                  width=SCENE, height=SCENE, num_inference_steps=26,
                  guidance_scale=7.0,
                  generator=torch.Generator("mps").manual_seed(500 + i)).images[0]
        scenes.append(img)
        print(f"  {i}  {time.time()-t0:4.0f}s  {s[:62]}")
    del txt

    # --- 2. placed on one canvas with gaps between them -------------------
    step = SCENE + GAP
    W = H = side * SCENE + (side - 1) * GAP
    canvas = Image.new("RGB", (W, H), (236, 228, 208))
    mask = Image.new("L", (W, H), 255)              # white = paint here
    for i, img in enumerate(scenes):
        x, y = (i % side) * step, (i // side) * step
        canvas.paste(img, (x, y))
        mask.paste(0, (x, y, x + SCENE, y + SCENE))  # black = keep the scene
    canvas.save(HERE / "quilt-before.png")

    # --- 3. paint the gaps ------------------------------------------------
    ink = StableDiffusionXLInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("mps")
    ink.set_progress_bar_config(disable=True)

    # Inpaint in overlapping windows: the model only ever sees a gap together
    # with the finished scenes on both sides of it, which is what lets it
    # continue them rather than invent something unrelated.
    win = SCENE + GAP * 2
    joined = canvas
    for gy in range(side):
        for gx in range(side):
            x = min(max(0, gx * step + SCENE - GAP), W - win)
            y = min(max(0, gy * step + SCENE - GAP), H - win)
            if mask.crop((x, y, x + win, y + win)).getextrema()[1] == 0:
                continue
            t0 = time.time()
            patch = ink(
                prompt=f"{STYLE}, continuous scenery joining the surrounding art",
                negative_prompt=NEGATIVE,
                image=joined.crop((x, y, x + win, y + win)),
                mask_image=mask.crop((x, y, x + win, y + win)),
                width=win, height=win, num_inference_steps=26,
                guidance_scale=7.0,
                generator=torch.Generator("mps").manual_seed(900 + gy * 8 + gx),
            ).images[0]
            joined.paste(patch, (x, y))
            print(f"  gap ({gx},{gy})  {time.time()-t0:4.0f}s")

    joined.save(HERE / "quilt.png")
    print(f"\n-> art/quilt.png  ({joined.size[0]}x{joined.size[1]})")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 9)
