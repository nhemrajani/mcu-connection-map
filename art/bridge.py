"""bridge.py - the space between two scenes depicts the connection between them.

    .venv/bin/python art/bridge.py 6     a connected chain, rendered as one strip

Earlier attempts treated the gap between scenes as a problem to be hidden.
Generated gaps invented comic covers; deterministic gaps were inert and the
result read as pictures pinned to a board.

The gap is not a problem. It is the subject. Two moments are adjacent on this
canvas because the graph says they are connected, so the space between them is
where that connection should be visible: the cube passing from one hand to
another, the shield changing owner, the blood drop travelling.

Three consequences for how this is built:

  ADJACENCY IS CONNECTION.  Scenes are ordered by walking the graph rather than
  by spatial proximity, so every neighbouring pair shares a real edge and the
  seam between them has something true to depict.

  THE SEAM PROMPT COMES FROM THE EDGE.  Its type and its two endpoints decide
  what is drawn. A setup-payoff bridge shows the thing carried from the first
  scene into the second; a theme-echo shows the two rhyming without contact.

  THE SEAM IS NARROW.  A wide gap is a blank canvas and the model fills it with
  whatever it likes, which is how the comic covers appeared. At this width it
  can only bridge what is already on both sides.
"""
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from quilt import subject, STYLE, NEGATIVE

SCENE = 640
SEAM = 128          # narrow on purpose: a wide gap is a blank canvas

# What the space between two connected scenes should show, by edge type.
BRIDGE = {
    "setup-payoff":
        "the object from the left scene being carried through into the right "
        "scene, motion lines and energy trailing behind it",
    "shared-object":
        "the same object appearing in both scenes, connected by a streak of "
        "light across the middle",
    "shared-character":
        "the same figure crossing from the left scene into the right scene",
    "timeline-adjacent":
        "the left scene flowing directly into the right scene, continuous "
        "ground and sky",
    "theme-echo":
        "the two scenes mirroring one another across a narrow divide, "
        "matching shapes, nothing passing between them",
}


def chain(n=6, seed_id=None):
    """A walk through the graph: each scene genuinely connected to the next.

    Picking nearest neighbours by position gave adjacency without connection,
    so the seams had nothing to depict. Walking edges instead means every join
    on the canvas corresponds to an edge in the data.
    """
    g = json.loads((ROOT / "ml" / "out" / "graph.json").read_text())
    nodes = {m["id"]: m for m in g["nodes"]}
    # Retained edges first, then everything proposed. The cutoff is tuned for
    # precision across the whole graph and it severs some true chains: the
    # weight is 1/frequency of the shared entity, so a recurring object scores
    # low precisely because it recurs. "Tesseract" links four scenes at 0.100,
    # just under the 0.125 threshold, for the same reason it is a through-line.
    # Walking a chain wants those edges even though the map is right to drop
    # them from its structural claims.
    extra = json.loads((ROOT / "ml" / "out" / "proposed_edges.json").read_text())
    adj = {}
    for e in list(g["edges"]) + extra:
        adj.setdefault(e["source"], []).append(e)
        adj.setdefault(e["target"], []).append(e)

    # Start where the visual descriptions are, so the chain has real scenes.
    have = [m for m in g["nodes"] if m.get("visual")]
    start = (nodes[seed_id] if seed_id in nodes
             else max(have, key=lambda m: len(adj.get(m["id"], []))))

    walk, edges, seen = [start], [], {start["id"]}
    while len(walk) < n:
        here = walk[-1]["id"]
        options = [e for e in adj.get(here, [])
                   if (e["target"] if e["source"] == here else e["source"]) not in seen]
        if not options:
            break
        # Prefer a neighbour that already has a written visual, then strength.
        def score(e):
            other = nodes[e["target"] if e["source"] == here else e["source"]]
            return (not other.get("visual"), -e.get("weight", 0))
        e = min(options, key=score)
        other = nodes[e["target"] if e["source"] == here else e["source"]]
        walk.append(other)
        edges.append(e)
        seen.add(other["id"])
    return walk, edges


def main(n=6, seed_id=None):
    from diffusers import (StableDiffusionXLPipeline,
                           StableDiffusionXLInpaintPipeline)

    walk, edges = chain(n, seed_id)
    print(f"chain of {len(walk)} scenes, {len(edges)} bridges\n")
    for m in walk:
        print(f"   {m['title'][:40]:42s} {'(visual)' if m.get('visual') else '(derived)'}")
    print()

    txt = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("mps")
    txt.set_progress_bar_config(disable=True)

    scenes = []
    for i, m in enumerate(walk):
        t0 = time.time()
        img = txt(prompt=f"{STYLE}, {subject(m)}", negative_prompt=NEGATIVE,
                  width=SCENE, height=SCENE, num_inference_steps=28,
                  guidance_scale=7.5,
                  generator=torch.Generator("mps").manual_seed(70 + i)).images[0]
        scenes.append(img)
        print(f"  scene {i}  {time.time()-t0:4.0f}s")
    del txt

    step = SCENE + SEAM
    W = len(scenes) * SCENE + (len(scenes) - 1) * SEAM
    strip = Image.new("RGB", (W, SCENE), (240, 234, 216))
    for i, img in enumerate(scenes):
        strip.paste(img, (i * step, 0))
    strip.save(HERE / "bridge-before.png")

    ink = StableDiffusionXLInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("mps")
    ink.set_progress_bar_config(disable=True)

    # Each seam is painted inside a window that includes a good slice of both
    # neighbours, so the model is continuing two finished images rather than
    # filling a void.
    win = 768
    for i, e in enumerate(edges):
        cx = i * step + SCENE + SEAM // 2
        x = max(0, min(cx - win // 2, W - win))
        mask = Image.new("L", (win, win), 0)
        mask.paste(255, (cx - x - SEAM // 2 - 24, 0, cx - x + SEAM // 2 + 24, SCENE))

        a, b = walk[i], walk[i + 1]
        how = BRIDGE.get(e.get("type"), BRIDGE["theme-echo"])
        prompt = (f"{STYLE}, {how}; on the left {subject(a)[:70]}; "
                  f"on the right {subject(b)[:70]}")

        t0 = time.time()
        patch = ink(prompt=prompt, negative_prompt=NEGATIVE,
                    image=strip.crop((x, 0, x + win, win)),
                    mask_image=mask, width=win, height=win,
                    num_inference_steps=30, guidance_scale=7.0,
                    generator=torch.Generator("mps").manual_seed(300 + i)).images[0]
        strip.paste(patch.crop((0, 0, win, SCENE)), (x, 0))
        print(f"  bridge {i} [{e.get('type')}]  {time.time()-t0:4.0f}s")

    strip.save(HERE / "bridge.png")
    print(f"\n-> art/bridge.png  ({strip.size[0]}x{strip.size[1]})")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6,
         sys.argv[2] if len(sys.argv) > 2 else None)
