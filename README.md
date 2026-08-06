# The MCU Connection Map

An infinite, zoomable canvas that maps the growing web of connections across the
Marvel Cinematic Universe — the setups and payoffs, the shared objects, the
cameos and callbacks — as a living **knowledge graph** rather than a timeline you
have to read top-to-bottom.

Each **moment** is a node. Each **connection** is an edge. Machine learning helps
surface connections a human would miss, groups moments into "threads," and decides
what deserves to be big and bright on the map. The art is original — hand-made
symbols in a personal style, not studio imagery.

> This is a learning project and a community one. If you want to add a moment or a
> connection, see [CONTRIBUTING.md](./CONTRIBUTING.md) — you don't need to touch a
> line of app code.

## How it works

```
data/moments.json ─┐
                   ├─►  ml/embed.py   embeddings + cosine similarity
data/connections ──┘        │         → ml/out/candidates.json  (suggested links)
                            ▼
                     ml/graph.py      build graph, detect "threads," rank hubs
                            │         → ml/out/graph.json
                            ▼
                     app/              infinite canvas: colour = thread,
                                       size = importance, zoom reveals detail
```

The pipeline is deliberately small so each step teaches one idea:

1. **Embeddings** — turn each moment's text into a vector where meaning becomes
   geometry, so similar moments sit close together.
2. **Similarity** — cosine similarity proposes *candidate* connections for you to
   confirm or reject by hand (curated data, ML-assisted).
3. **Graph + communities** — an unsupervised algorithm clusters the confirmed
   connections into threads; centrality finds the load-bearing moments.
4. **Canvas** — the graph is rendered on an infinite pan/zoom surface with
   semantic zoom (clusters when far, full moments when close).

## Quickstart (ML pipeline)

```bash
cd ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python embed.py     # writes ml/out/candidates.json
python graph.py     # writes ml/out/graph.json
```

## Structure

```
data/     the map's content — moments + connections (this is the community layer)
ml/       embeddings + graph clustering (the "teach me ML" part)
icons/    original per-film symbols + notes on the style pipeline
app/      the infinite-canvas frontend (see app/README.md for the plan)
```

## Licensing

- **Code** (everything in `ml/` and `app/`): MIT — see [LICENSE](./LICENSE).
- **Content** (the descriptions and connections in `data/`): CC BY-SA 4.0, so the
  map can be shared and built on with attribution.
- **Artwork** (`icons/`): original work by the author; reuse terms noted in
  `icons/README.md`.

(These are sensible defaults, not legal advice — adjust to taste.)

## Disclaimer

This is an unofficial, non-commercial fan project. It is **not** affiliated with,
endorsed by, or sponsored by Marvel, Marvel Studios, or The Walt Disney Company.
Film titles and character names are the property of their respective owners and are
used here only for identification and reference.

## Roadmap

- [ ] Seed the first ~50 moments and hand-confirm connections
- [ ] Tune the similarity threshold against a hand-labelled ground-truth set
- [ ] Ship the canvas MVP (React Flow) with semantic zoom
- [ ] Train the style-LoRA and generate the first symbol set
- [ ] Write up the method as a post
