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

On depiction. An earlier version stripped every character name before building
a prompt, which produced "a figure" where the scene needed a person and made
the images vague. That restriction is gone: prompts name what is in frame. The
README discloses that Marvel's house style is present in the base model's
training data and that recognisable characters therefore appear.

The gaps between scenes are NOT generated. Asked to paint comic art surrounded
by comic art, the inpainting model concluded it was painting a picture OF
comics and produced covers, magazine pages and framed prints, turning a world
into a gallery wall. The gaps are now filled deterministically from the colours
of the scenes on either side, which is instant, cannot wander, and leaves the
scenes as the only thing carrying content.
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

# CLIP accepts 77 tokens and silently discards the rest. The previous style
# string was 39 of them, and a written visual description is around 41, so
# every scene prompt overflowed and the model never saw the end of the
# description it was supposed to draw. That, not vagueness, is why renders kept
# missing their subject.
#
# Two consequences. The style string is now short enough to leave real room,
# and the scene goes FIRST so that if anything is lost it is styling rather
# than content.
STYLE = "silver age comic art, ben-day dots, heavy ink, flat colour"
NEGATIVE = ("text, lettering, speech bubble, caption, words, signature, "
            "watermark, panel border, frame, gutter, map, atlas, chart, "
            "photorealistic, 3d render, blurry")

_TOK = None


def fit(*parts, limit=74):
    """Join prompt fragments and trim to what CLIP will actually read.

    Silent truncation is the worst kind: the pipeline reports success, the
    image comes back wrong, and nothing points at the cause. Trimming here
    makes the loss explicit and puts it at the end, where the least important
    words have been placed deliberately.
    """
    global _TOK
    if _TOK is None:
        from transformers import CLIPTokenizer
        _TOK = CLIPTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer")
    text = ", ".join(p.strip(" ,") for p in parts if p and p.strip())
    words = text.split()
    while len(_TOK(" ".join(words))["input_ids"]) - 2 > limit and len(words) > 4:
        words.pop()
    return " ".join(words)


def subject(moment):
    """What this moment looks like.

    Prefers the hand-written `visual` field, which is a composed scene with a
    subject, a setting and a framing. Falls back to clipping imagery out of
    `description`, which produces a fragment rather than a picture.
    """
    if moment.get("visual"):
        return moment["visual"]
    first = re.split(r"(?<=[.;])\s", moment["description"])[0]
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
        img = txt(prompt=fit(s, STYLE), negative_prompt=NEGATIVE,
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

    # --- 3. bridge the gaps without generating anything -------------------
    # Each gap pixel takes a blend of the nearest scene edges, then a halftone
    # dot screen is laid over it so the join reads as printing rather than as
    # a blur. Deterministic, instant, and incapable of inventing a comic cover.
    import numpy as np

    arr = np.asarray(canvas).astype(np.float32)
    gap = np.asarray(mask) > 0
    filled = canvas.filter(ImageFilter.GaussianBlur(GAP * 0.55))
    fa = np.asarray(filled).astype(np.float32)
    arr[gap] = fa[gap]

    yy, xx = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
    dots = (((xx % 6) - 3) ** 2 + ((yy % 6) - 3) ** 2) <= 2
    screen = np.where(dots[..., None], 0.86, 1.04)
    arr[gap] = np.clip(arr[gap] * screen[gap], 0, 255)

    joined = Image.fromarray(arr.astype(np.uint8))
    joined.save(HERE / "quilt.png")
    print(f"\n-> art/quilt.png  ({joined.size[0]}x{joined.size[1]})")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 9)
