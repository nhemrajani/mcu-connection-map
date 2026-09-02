"""world.py - edge-to-edge tiles for a continuous canvas.

    .venv/bin/python art/world.py test        six tiles, full SDXL with guidance
    .venv/bin/python art/world.py seam        two tiles, second outpainted from the first

Differences from art/style.py, all of them forced by wanting ONE continuous
world rather than a page of panels:

  GUIDANCE IS ON.  style.py used SDXL-Turbo at guidance_scale=0, which
  silently disabled the negative prompt entirely - classifier-free guidance is
  the mechanism negative prompts act through, so every "no text, no faces"
  instruction was ignored and every tile came out covered in gibberish speech
  bubbles. Full SDXL with guidance is slower per image and is the only way to
  get those instructions honoured.

  NO PANEL LANGUAGE.  Asking for a "comic book panel" gets a panel: border,
  margin, gutter and all. A continuous world needs the ink and colour of the
  idiom without its page furniture, so the prompt asks for a printed
  ILLUSTRATION that bleeds to the edge, and the negative prompt works hard to
  suppress frames.

  AERIAL VIEW.  Tiles have to join at their edges. Ground-level scenes with a
  horizon cannot: two neighbouring tiles would each want their own horizon at
  their own height. Looking down removes the problem - terrain meets terrain
  at any edge.
"""
import sys
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline
from PIL import Image

HERE = Path(__file__).resolve().parent
MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

# Colour first, ageing last. The previous version led with "1960s offset
# printing / ben-day dots / aged newsprint" and the model obeyed the ageing
# and produced near-monochrome engravings: at guidance 7 those three cues
# collectively outvoted "flat limited colour palette". Silver Age comics are
# not old-looking, they are LOUD - saturated flat primaries straight off a
# four-colour press - so the colour instruction now leads and carries the
# most weight.
#
# "Orthographic" and "no horizon" are doing real work too. Asking for an
# "aerial view" still produced perspective landscapes with skylines, and a
# tile containing a horizon cannot join the tile above it.
STYLE = (
    "bold flat vivid colour comic art, saturated primary colours, cyan "
    "magenta yellow red blue, four colour process printing, ben-day dot "
    "halftone texture, heavy black ink outlines, poster-like graphic shapes, "
    "orthographic top-down map view, seen from directly overhead, "
    "flat terrain filling the whole image edge to edge"
)

NEGATIVE = (
    "horizon line, sky, clouds, perspective view, vanishing point, landscape "
    "photograph, "
    "panel border, frame, white border, cream border, margin, gutter, page "
    "edge, torn paper, "
    "speech bubble, caption box, text, lettering, words, letters, signature, "
    "watermark, "
    "people, figures, faces, vehicles, "
    "monochrome, greyscale, desaturated, sepia, muted colours, pencil "
    "sketch, engraving, etching, photorealistic, 3d render, blurry"
)

SUBJECTS = [
    "orange crater field with a glowing yellow fissure",
    "dense blue and grey city blocks in a tight grid",
    "green forest canopy split by a winding cyan river",
    "white and pale blue ice plain scored with dark cracks",
    "red and yellow dunes with angular black wreckage",
    "deep blue water with a magenta spiral of foam",
]


def load(inpaint=False):
    cls = StableDiffusionXLInpaintPipeline if inpaint else StableDiffusionXLPipeline
    print(f"loading {'inpaint' if inpaint else 'base'} pipeline ...")
    t0 = time.time()
    pipe = cls.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1" if inpaint else MODEL,
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe = pipe.to("mps")
    pipe.set_progress_bar_config(disable=True)
    print(f"  ready in {time.time() - t0:.0f}s")
    return pipe


def make(pipe, subject, seed=0, steps=28, guidance=7.0, size=768):
    return pipe(
        prompt=f"{STYLE}, {subject}",
        negative_prompt=NEGATIVE,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=size, height=size,
        generator=torch.Generator("mps").manual_seed(seed),
    ).images[0]


def test():
    pipe = load()
    out = HERE / "world"
    out.mkdir(exist_ok=True)
    tiles, times = [], []
    for i, s in enumerate(SUBJECTS[:4]):
        t0 = time.time()
        img = make(pipe, s, seed=2000 + i)
        times.append(time.time() - t0)
        img.save(out / f"t{i}.png")
        tiles.append(img)
        print(f"  {i}  {times[-1]:5.1f}s  {s[:54]}")
    w, h = tiles[0].size
    grid = Image.new("RGB", (w * 3, h * 2), "black")
    for i, t in enumerate(tiles):
        grid.paste(t, ((i % 3) * w, (i // 3) * h))
    grid.save(HERE / "world-test.png")
    print(f"\n{sum(times)/len(times):.1f}s each -> art/world-test.png")


def seam():
    """Generate one tile, then outpaint its right-hand neighbour.

    This is the load-bearing experiment for the whole continuous-canvas idea.
    The new tile is generated with the left third of its canvas already filled
    by the previous tile's right edge, and only the rest masked for painting,
    so the model has to continue what is already there. If the join is
    visible, tiles cannot be grown outward and the canvas has to be produced
    as one large image instead.
    """
    size = 768
    overlap = size // 3

    base = make(load(), "a crater field of shattered rock with a deep glowing fissure",
                seed=7, size=size)
    base.save(HERE / "world" / "seam_a.png")
    print("tile A done")

    pipe = load(inpaint=True)
    canvas = Image.new("RGB", (size, size), "black")
    canvas.paste(base.crop((size - overlap, 0, size, size)), (0, 0))
    mask = Image.new("L", (size, size), 255)
    mask.paste(0, (0, 0, overlap, size))          # black = keep, white = repaint

    nxt = pipe(
        prompt=f"{STYLE}, a crater field giving way to cracked lava plains",
        negative_prompt=NEGATIVE,
        image=canvas, mask_image=mask,
        num_inference_steps=30, guidance_scale=7.0,
        width=size, height=size,
        generator=torch.Generator("mps").manual_seed(8),
    ).images[0]
    nxt.save(HERE / "world" / "seam_b.png")

    joined = Image.new("RGB", (size * 2 - overlap, size), "black")
    joined.paste(base, (0, 0))
    joined.paste(nxt, (size - overlap, 0))
    joined.save(HERE / "world-seam.png")
    print(f"-> art/world-seam.png  ({joined.size[0]}x{joined.size[1]})")


if __name__ == "__main__":
    (seam if len(sys.argv) > 1 and sys.argv[1] == "seam" else test)()
