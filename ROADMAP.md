# Roadmap

Where the project is going, in order. Each step produces something you can see,
and each depends on the one before it.

**Where we are now:** 266 moments across 26 titles, 8 confirmed edges, a working
`embed → graph → canvas` pipeline, and a debug canvas that renders the map with
positions computed from meaning.

---

## Phase A — Make the graph real

The map cannot develop regions until it has edges. Everything visual downstream
depends on this phase.

### 1. Generate edges without a human · `ml/relate.py`
Cosine similarity measures *"these sound alike"*, not *"this caused that"*. It
misses the connections that matter most — Samuel Sterns getting Banner's blood
in a cut in 2008 pays off seventeen years later, and the two descriptions share
almost no vocabulary.

Three signals, none needing judgement:
- **Shared rare entities** → `shared-character` / `shared-object` edges
- **Release chronology** → direction, turning symmetric pairs into `setup-payoff`
- **Same film + adjacency** → `timeline-adjacent`

Expected: ~1,300 typed edges from 266 moments.

### 2. Kill the false positives · cross-encoder
A bi-encoder embeds each moment separately and compares. A **cross-encoder**
reads both moments *together* and scores the pair directly — much higher
precision, still zero-shot, still no labels. Runs locally.

### 3. Real threads and real hubs
With enough edges, `graph.py` switches from spatial clustering to genuine
community detection, and `centrality` starts meaning something. Node size
finally reflects how load-bearing a moment is.

### 4. Finish extraction
The remaining 45 titles, taking the corpus to roughly 800 moments. Re-run 1–3.
Deliberately after the pipeline is proven, not before.

---

## Phase B — Make it learn

### 5. Evaluation · the honest number
Hand-label a random sample of ~200 proposed edges as real or not. Compute
precision and recall per edge source. This is the step that turns the project
from a nice visual into something with a result — and the labels double as the
training set for step 6.

### 6. Fine-tune the embeddings on your own edges
Contrastive training on confirmed vs rejected pairs. The model stops measuring
generic semantic similarity and starts measuring *this project's* notion of a
narrative connection. Report precision@k before and after.

**Not** training a language model on MCU text. The corpus is ~138k words, which
is far too small, and any current LLM has already read everything written about
the MCU. The knowledge is free; the judgement is the part that is ours.

### 7. Link prediction · the emergent layer
Train node2vec or a GNN on the graph's own topology to propose edges that no
source ever stated. This is the model learning from patterns with no human
labels at all, and it is the part that is genuinely novel.

---

## Phase C — Make it beautiful

### 8. Stable regions with names
Communities become regions. An LLM reads each cluster and names the thread, so
the map has territories rather than numbered blobs.

### 9. The art pipeline
- **FLUX.1 [schnell]** locally on the M4 Pro, roughly 5s a tile
- **Tile pyramid**, precomputed like map tiles — generation is far too slow to
  run live on zoom
- Each tile generated **conditioned on its neighbours** (img2img outpainting) so
  the canvas is one continuous illustration, not a grid of separate pictures
- **Silver Age idiom**: Ben-Day dots, heavy ink, flat four-colour palette, panel
  gutters. Original symbolic scenes influenced by Jack Kirby and Steve Ditko,
  credited in the final product. No Marvel imagery, no recognisable characters,
  no training on scans.

### 10. Canvas v2
A deep-zoom renderer over the illustration, with the interactive graph as a
layer on top. Zoom levels: world silhouette → regional scenes → moment panels.
The current orb view stays as the developer view — it is how three real bugs
were caught.

---

## Phase D — Ship it and let it grow

### 11. Publish
`mcu.neeha.xyz` on Cloudflare Pages, static, no backend. Methodology write-up,
contributor guide, fan-project disclaimer, art credits.

### 12. Self-growing ingestion
Poll TMDB for new releases → fetch the Wikipedia plot → extract moments →
recompute → open a pull request for review. The registry already tracks six
upcoming titles, so the trigger has something to fire on.

---

## Stopping points

Phase A alone is a finished, legitimate project. Phase B is where it becomes
research. Phase C is where it becomes beautiful. Phase D is where it becomes
alive. Stopping after any of them leaves something real.
