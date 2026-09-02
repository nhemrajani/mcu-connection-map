"""style.py - the Silver Age recipe, and a sheet to judge whether it holds.

    .venv/bin/python art/style.py sheet        12 varied prompts, one grid
    .venv/bin/python art/style.py one "..."    a single test image

The question this answers is not "can it draw a nice picture" - it can. It is
whether the SAME visual language survives across a dozen different subjects.
A tile pyramid needs hundreds of images that look like one continuous world;
a style that drifts between subjects cannot do that, and it is much cheaper to
find that out now than after building the pyramid.

On the style itself. The look is the Silver Age comic idiom - Ben-Day dots,
heavy black ink, flat limited colour, visible printing registration. That is a
period printing process, not anyone's property. The specific hands associated
with it, Jack Kirby and Steve Ditko, are credited in the finished work as
influences.

Two rules the prompts follow, from the project's own licence position:

  NO CHARACTERS. Nothing is asked for by name and nothing recognisable as a
  Marvel character is generated. Subjects are objects, places and forces.

  NO TRAINING ON MARVEL ART. This uses a general base model. No LoRA is
  trained on scanned comics.

Symbols and landscapes carry a map better than figures would anyway.
"""
import sys
import time
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image

HERE = Path(__file__).resolve().parent
MODEL = "stabilityai/sdxl-turbo"

# The style prefix is the thing under test. Every tile in the finished map
# would carry it, so it has to be strong enough to dominate whatever subject
# follows it.
STYLE = (
    "silver age comic book art, 1960s offset printing, visible ben-day dot "
    "halftone, heavy black ink outlines, flat limited colour palette, "
    "slight off-register colour, aged newsprint paper texture, bold graphic "
    "shapes, dramatic angular composition, no text, no lettering, no speech "
    "bubbles"
)

NEGATIVE = (
    "photorealistic, 3d render, modern digital art, smooth gradients, "
    "soft focus, photograph, text, watermark, signature, logo, letters, "
    "faces, portrait, recognisable superhero costume"
)

# Twelve subjects spanning what the real map would need: objects, places,
# forces, abstractions. If the style holds across all of these it will hold
# across the threads.
SUBJECTS = [
    "a glowing cube on a laboratory bench",
    "a wide desert with a crater and a fallen hammer",
    "a golden city of spires floating above a void",
    "an empty suit of riveted armour in a cave workshop",
    "a dense forest of antennae and satellite dishes on a hillside",
    "a shattered bridge of light over black water",
    "a vast petrified figure rising out of the sea",
    "an operating theatre with an empty chair and hanging cables",
    "a ring of standing stones under a green sky",
    "a derelict cargo ship half sunk in a harbour",
    "a spiral staircase descending into darkness",
    "a field of wheat with one burning tree",
]


def load():
    print("loading model ...")
    t0 = time.time()
    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL, torch_dtype=torch.float16, variant="fp16")
    pipe = pipe.to("mps")
    pipe.set_progress_bar_config(disable=True)
    print(f"  ready in {time.time() - t0:.0f}s")
    return pipe


def generate(pipe, subject, seed=0, steps=4):
    gen = torch.Generator("mps").manual_seed(seed)
    return pipe(
        prompt=f"{STYLE}, {subject}",
        negative_prompt=NEGATIVE,
        num_inference_steps=steps,
        guidance_scale=0.0,          # turbo is trained for guidance-free
        generator=gen,
    ).images[0]


def sheet():
    from PIL import Image
    pipe = load()
    out = HERE / "sheet"
    out.mkdir(exist_ok=True)
    tiles, times = [], []
    for i, subject in enumerate(SUBJECTS):
        t0 = time.time()
        img = generate(pipe, subject, seed=1000 + i)
        times.append(time.time() - t0)
        img.save(out / f"{i:02d}.png")
        tiles.append(img)
        print(f"  {i:2d}  {times[-1]:5.1f}s  {subject[:52]}")

    w, h = tiles[0].size
    cols, rows = 4, 3
    grid = Image.new("RGB", (w * cols, h * rows), "black")
    for i, t in enumerate(tiles):
        grid.paste(t, ((i % cols) * w, (i // cols) * h))
    grid.save(HERE / "style-sheet.png")
    print(f"\n{len(tiles)} images, {sum(times)/len(times):.1f}s each")
    print(f"grid -> art/style-sheet.png  ({grid.size[0]}x{grid.size[1]})")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sheet"
    if cmd == "one":
        p = load()
        generate(p, sys.argv[2]).save(HERE / "one.png")
        print("-> art/one.png")
    else:
        sheet()
