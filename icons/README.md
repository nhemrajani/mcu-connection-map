# Icons

Every film gets one **original symbol** rendered in a single, consistent art
style. A symbol *evokes* a film through a motif — an object, an element, a shape —
rather than reproducing any official logo, prop, or character design. This keeps
the set unmistakably hand-made and legally clean.

## The style pipeline (planned)

The house style is produced by a small **LoRA fine-tuned on the author's own
artwork**, on top of an open base model (SDXL recommended for its LoRA ecosystem;
FLUX.1 [schnell], Apache-2.0, is the license-clean alternative).

Rough flow:

1. Collect 15-30 pieces of original art that define the style.
2. Caption them and train a style LoRA (free tier of Colab / Kaggle is enough for
   SDXL LoRA training).
3. For each film, write a short motif prompt ("a stylised hammer wreathed in
   storm-light", etc.) and generate a few options.
4. Pick one, clean it up, export at a few sizes for semantic zoom.

## Files

- `raw/`    generated candidates (git-ignored)
- `final/`  the chosen, cleaned symbol per film
- Name each final file after the film's moment ids where possible.

## Reuse

These symbols are original artwork by the author. If you contribute symbols,
you agree they can be shared under the project's content terms (CC BY-SA 4.0)
unless noted otherwise.
