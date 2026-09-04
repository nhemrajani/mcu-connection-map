"""scenes.py - moments drawn as scenes, bleeding into each other.

    .venv/bin/python art/scenes.py strip

The territory map in terrain.py answered the wrong question. It showed WHERE
threads sit, as coloured regions, with nothing depicted. What the canvas is
supposed to do is show the moments themselves - an unmasking in one place
bleeding into the crossover it leads to - so panning across the world moves
you from one scene to the next without a seam.

So each moment becomes a scene, scenes are laid out in the order the graph
puts them, and each one is outpainted from its neighbour's edge so the join
is continuous. That is the Million Dollar Canvas mechanic: every tile is
generated to continue whatever its neighbours already show.

ON DEPICTION. Every scene here is written to be recognisable as the moment
without drawing anyone's character. No costumes, no masks with a known design,
no insignia - a torn hood rather than a spider-mask, a lit screen in a square
rather than a broadcast anyone could name. That is this project's own rule
(original art, no studio imagery) and it also makes better pictures: a
silhouette pulling a hood away is a stronger image than a licensed costume.

The four moments below are genuine neighbours in the layout - 7 and 15 units
apart - not a hand-picked sequence. The map chose the order; this only draws it.
"""
import sys
import time
from pathlib import Path

import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
SIZE = 768
OVERLAP = SIZE // 2          # half of each new tile is already painted

STYLE = (
    "silver age comic book art, bold black ink outlines, flat saturated "
    "colour, ben-day dot halftone shading, dramatic angular composition, "
    "cinematic lighting, printed on aged paper"
)
NEGATIVE = (
    "text, lettering, speech bubble, caption, words, signature, watermark, "
    "panel border, frame, white margin, gutter, "
    "superhero costume, spandex, insignia, logo, mask design, cape, "
    "photorealistic, 3d render, blurry, deformed hands, extra limbs"
)

# Four moments that sit next to each other in the layout, in map order.
SCENES = [
    ("m-ned-discovers-identity",
     "a teenage boy frozen in a bedroom doorway at night, dropping a bag, "
     "staring up at a dark figure crouched on the ceiling above him, "
     "warm desk lamp light, blue night window"),
    ("m-identity-exposed-to-world",
     "an enormous television screen above a crowded night city square, "
     "hundreds of small upturned faces lit blue by it, a grainy figure "
     "frozen mid-frame on the screen, rain on the pavement"),
    ("m-harry-unmasks-spider-man",
     "gloved hands pulling a torn dark hood away from the face of an "
     "unconscious figure slumped on a workshop floor, one harsh overhead "
     "lamp, deep shadows, scattered machinery"),
    ("m-harry-vows-revenge",
     "a lone young man in a black coat at a graveside in heavy rain, "
     "a ring of black umbrellas behind him, a grey mansion on the hill, "
     "bare trees"),
]


def load(inpaint=False):
    from diffusers import (StableDiffusionXLPipeline,
                           StableDiffusionXLInpaintPipeline)
    name = ("diffusers/stable-diffusion-xl-1.0-inpainting-0.1" if inpaint
            else "stabilityai/stable-diffusion-xl-base-1.0")
    cls = StableDiffusionXLInpaintPipeline if inpaint else StableDiffusionXLPipeline
    t0 = time.time()
    pipe = cls.from_pretrained(name, torch_dtype=torch.float16,
                               variant="fp16", use_safetensors=True).to("mps")
    pipe.set_progress_bar_config(disable=True)
    print(f"  {'inpaint' if inpaint else 'base'} pipeline ready "
          f"({time.time()-t0:.0f}s)")
    return pipe


def strip():
    """One continuous strip, each scene outpainted from the previous one.

    The new tile starts with its left half already filled by the previous
    tile's right half, and only the right half masked for painting. The model
    therefore has to continue a picture that already exists rather than invent
    a fresh one, which is what makes the join disappear.
    """
    print("generating the first scene ...")
    base = load()
    first = base(
        prompt=f"{STYLE}, {SCENES[0][1]}", negative_prompt=NEGATIVE,
        width=SIZE, height=SIZE, num_inference_steps=30, guidance_scale=7.0,
        generator=torch.Generator("mps").manual_seed(11),
    ).images[0]
    del base

    pipe = load(inpaint=True)
    panels = [first]
    for i, (mid, desc) in enumerate(SCENES[1:], start=1):
        canvas = Image.new("RGB", (SIZE, SIZE))
        canvas.paste(panels[-1].crop((SIZE - OVERLAP, 0, SIZE, SIZE)), (0, 0))
        canvas.paste(panels[-1].crop((SIZE - OVERLAP, 0, SIZE, SIZE)), (OVERLAP, 0))
        mask = Image.new("L", (SIZE, SIZE), 255)
        mask.paste(0, (0, 0, OVERLAP, SIZE))      # keep the left half

        t0 = time.time()
        nxt = pipe(
            prompt=f"{STYLE}, {desc}", negative_prompt=NEGATIVE,
            image=canvas, mask_image=mask,
            width=SIZE, height=SIZE, num_inference_steps=34,
            guidance_scale=7.5, strength=1.0,
            generator=torch.Generator("mps").manual_seed(20 + i),
        ).images[0]
        panels.append(nxt)
        print(f"  {i}  {time.time()-t0:5.0f}s  {mid}")

    width = SIZE + (len(panels) - 1) * OVERLAP
    out = Image.new("RGB", (width, SIZE))
    out.paste(panels[0], (0, 0))
    for i, p in enumerate(panels[1:], start=1):
        out.paste(p.crop((OVERLAP, 0, SIZE, SIZE)), (SIZE + (i - 1) * OVERLAP, 0))
    out.save(HERE / "scene-strip.png")
    print(f"\n-> art/scene-strip.png  ({out.size[0]}x{out.size[1]})")


if __name__ == "__main__":
    strip()
