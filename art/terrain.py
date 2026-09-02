"""terrain.py - draw the graph as a map, then let the model ink it.

    .venv/bin/python art/terrain.py base       render the data as flat terrain
    .venv/bin/python art/terrain.py ink        stylise it into comic art
    .venv/bin/python art/terrain.py tiles 2    crop and stylise a 2x2 grid

Text-to-image was the wrong tool for this. Asking a model to invent a top-down
map from a sentence gave perspective landscapes with horizons in two separate
attempts, and a tile with a horizon in it cannot join the tile above it.

So the model stops being the author and becomes the stylist. The composition
comes from the data that already exists: 599 moments with x/y positions from
the layout, and a thread number each. Rendered as flat colour fields, that IS
a top-down map - no horizon, no sky, no vanishing point, and every region in
the right place by construction.

img2img then inks it. At the strength used here the model repaints texture,
linework and halftone while the underlying shapes survive, so the terrain
still says what the graph says.

Three problems solved at once:

  TOP-DOWN is guaranteed, because the input is top-down.
  CONTINUITY comes free, because neighbouring tiles are crops of one image
  rather than separate inventions that have to be persuaded to agree.
  MEANING is preserved: the shapes of the land are the shape of the graph,
  which was the point of the project rather than a decoration on top of it.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
GRAPH = ROOT / "ml" / "out" / "graph.json"

# Flat four-colour comic inks, one per thread. Deliberately saturated and
# limited - a four-colour press could not print subtle, and neither should
# this. Repeats after twelve; threads that share a hue are far apart.
INKS = [
    (222, 62, 52), (46, 122, 196), (247, 196, 43), (58, 158, 96),
    (156, 74, 168), (232, 128, 42), (64, 190, 196), (206, 74, 128),
    (118, 156, 52), (86, 92, 178), (224, 152, 88), (52, 148, 148),
]
PAPER = (238, 230, 210)
INK = (24, 22, 28)


def load_nodes():
    g = json.loads(GRAPH.read_text())
    return g["nodes"], g["edges"], g.get("meta", {})


def render_base(size=1024, mode="threads", coast=True):
    """Flat colour territory built from moment positions and thread membership.

    Every point on the map belongs to whichever moment is nearest, and takes
    that moment's thread colour. That is a nearest-neighbour partition, and it
    is the right algorithm for territory: it produces hard borders and leaves
    no unclaimed ground, which is what a map is.

    The first version stamped blurred discs and posterised them. It made mush -
    overlapping colours blended into a smear with no borders anywhere, which is
    exactly what img2img has nothing to hold on to.
    """
    from scipy.spatial import cKDTree

    nodes, edges, _ = load_nodes()
    xk, yk = ("x", "y") if mode == "threads" else ("sx", "sy")
    pad, span = size * 0.05, size * 0.90
    pts = np.array([[pad + n[xk] / 1000 * span, pad + n[yk] / 1000 * span]
                    for n in nodes])
    thread = np.array([n["community"] for n in nodes])

    grid = 512                       # partition coarsely, then upscale hard
    ys, xs = np.mgrid[0:grid, 0:grid]
    q = np.stack([xs.ravel(), ys.ravel()], axis=1) * (size / grid)
    dist, nearest = cKDTree(pts).query(q)
    owner = thread[nearest].reshape(grid, grid)
    dist = dist.reshape(grid, grid)

    # A nearest-neighbour partition is unbounded: the outermost moments claim
    # everything out to the frame, so a single moment on the edge was being
    # given a quarter of the map. Ground further than this from any moment is
    # not territory, it is sea - which also gives the landmass a coastline and
    # makes it read as an island rather than a filled square.
    sea = dist > size * 0.075
    owner = np.where(sea, -1, owner)

    palette = np.array(INKS, dtype=np.uint8)
    img = palette[owner % len(INKS)]
    img[sea] = PAPER

    # Coastlines: ink wherever the owning thread changes. Borders are what
    # make a partition read as a map rather than a colour field.
    if coast:
        edge = np.zeros((grid, grid), bool)
        edge[:-1, :] |= owner[:-1, :] != owner[1:, :]
        edge[:, :-1] |= owner[:, :-1] != owner[:, 1:]
        img[edge] = INK

    canvas = Image.fromarray(img).resize((size, size), Image.NEAREST)

    # A few confirmed links drawn as ink channels. The full 1,200 edges was an
    # unreadable hairball that buried the terrain underneath it.
    pos = {n["id"]: (pad + n[xk] / 1000 * span, pad + n[yk] / 1000 * span)
           for n in nodes}
    draw = ImageDraw.Draw(canvas)
    for e in edges:
        if e.get("verdict") != "confirmed":
            continue
        a, b = pos.get(e["source"]), pos.get(e["target"])
        if a and b:
            draw.line([a, b], fill=INK, width=max(2, size // 400))

    return canvas


def load_img2img():
    from diffusers import StableDiffusionXLImg2ImgPipeline
    print("loading img2img pipeline ...")
    t0 = time.time()
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("mps")
    pipe.set_progress_bar_config(disable=True)
    print(f"  ready in {time.time() - t0:.0f}s")
    return pipe


STYLE = (
    "silver age comic book map, flat bold ink colours, ben-day dot halftone "
    "shading, heavy black ink outlines around every region, printed on aged "
    "paper, aerial map of strange terrain, coastlines and rivers and plateaus, "
    "no text"
)
NEGATIVE = (
    "horizon, sky, clouds, perspective, buildings, people, figures, text, "
    "lettering, speech bubble, frame, border, photorealistic, 3d, blurry, "
    "neon, garish, oversaturated"
)


def ink(pipe, base, strength=0.55, seed=3, steps=30):
    return pipe(
        prompt=STYLE, negative_prompt=NEGATIVE, image=base,
        strength=strength, guidance_scale=6.5, num_inference_steps=steps,
        generator=torch.Generator("mps").manual_seed(seed),
    ).images[0]


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "base"
    HERE.mkdir(exist_ok=True)
    if cmd == "base":
        for mode in ("threads", "meaning"):
            render_base(mode=mode).save(HERE / f"terrain-{mode}.png")
            print(f"-> art/terrain-{mode}.png")
    elif cmd == "ink":
        base = render_base()
        base.save(HERE / "terrain-threads.png")
        pipe = load_img2img()
        for s in (0.4, 0.55, 0.7):
            t0 = time.time()
            ink(pipe, base, strength=s).save(HERE / f"terrain-ink-{s}.png")
            print(f"  strength {s}: {time.time()-t0:.0f}s -> art/terrain-ink-{s}.png")
