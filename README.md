# The MCU Connection Map

**An experiment at the intersection of design and machine learning: can we
understand a story better by seeing it?**

Twenty-five years of the Marvel Cinematic Universe is roughly seventy-six films
and series, and nobody holds the whole shape of it in their head. It is usually
explained as a timeline — a list, read top to bottom, one thing after another.
But a story is not a list. It is a web of setups and payoffs, objects that
change hands, people who reappear decades later, and beats that rhyme across
films that have nothing else in common.

This project asks whether that web can be **found by a model** and then
**drawn**, and whether the drawing teaches you something the list never could.

Two halves, each a real research question:

1. **Can a model predict narrative connection and causality?** Not "do these two
   scenes use similar words", but "does this one set that one up".
2. **Can that structure become an image?** The graph is handed to an open-source
   image model, which renders it as a continuous illustrated world that grows
   more specific as you zoom in.

---

## Where it stands

| | |
|---|---|
| **Moments** | 599, from all 76 released MCU titles |
| **Source text** | 130,393 words of Wikipedia plot summary, with per-title revision IDs |
| **Connections** | 1,248 in the graph, filtered at a measured precision cutoff |
| **Threads** | 39, found by community detection with no labels |
| **Evaluation** | 114 judged pairs — 27 human, 87 model |
| **Live map** | [mcu.neeha.xyz](https://mcu.neeha.xyz) |

**Honest status:** the graph half works and is measured. The visual half is
partly built — the style is proven and tiles blend invisibly, but the tile
pyramid is not finished. See [Limits](#limits).

---

## What it found

Three things nobody entered, and none of them are timeline facts:

**The two "I am Iron Man" moments land 7 units apart.** The first line of the
MCU in 2008 and its last in 2019, thirty-three films apart, placed as
near-neighbours by an embedding that was never told they were related.

**Iron Man and Shang-Chi are in the same thread.** The Ten Rings kidnap Tony
Stark in 2008 and are never explained. Thirteen years later they turn out to be
a thousand-year-old organisation led by Shang-Chi's father. The clustering put
both films in one region unprompted.

**The most central moment in the MCU is Phil Coulson's death.** Not a snap, not
a battle. The death that forces the Avengers into a team, and the hinge on
which the films hand off to *Agents of S.H.I.E.L.D.*

---

## How it works

```
Wikipedia ──► registry.py ──► plots.py ──► 599 moments
   API         what exists     the text      data/moments.json
                                                   │
                        ┌──────────────────────────┼──────────────────────┐
                        ▼                          ▼                      ▼
                   embed.py                  relate.py             references.py
              meaning → 384 numbers      shared rare entities    links the source
              cosine similarity          + release chronology     states outright
                        └──────────────────────────┼──────────────────────┘
                                                   ▼
                                              graph.py
                                   threads · centrality · layout
                                                   │
                        ┌──────────────────────────┴──────────────────────┐
                        ▼                                                 ▼
                   index.html                                        art/
              interactive canvas                            image model renders
                                                              the same graph
```

### 1. Meaning becomes geometry

A sentence-transformer turns each moment's description into 384 numbers.
Descriptions that mean similar things produce similar numbers, so every moment
gains a **position**, and the angle between two of them measures how alike they
are.

### 2. But similarity is not causality

This is the central finding. Cosine similarity measures *"these sound alike"*.
It cannot measure *"this caused that"*, and it misses the connections that
matter most. A drop of gamma-irradiated blood lands in Samuel Sterns' cut in
2008 and pays off seventeen years later in *Brave New World* — two descriptions
sharing almost no vocabulary, invisible to similarity, obvious to a method that
tracks **who and what** appears in each.

So the graph is built from three signals instead:

- **Shared rare entities** — "Tony Stark" appears 35 times and carries no
  information; "Quantum Realm" appears in five moments and means those five
  belong together
- **Release chronology** — gives a connection direction: the earlier moment
  sets up the later one
- **Stated references** — places where the source says it outright: *"Following
  the Battle of New York, Adrian Toomes and his salvage company…"*

### 3. Measure it, then cut

Every edge source is guessing, so the guesses were checked. 114 pairs were
judged — a mix of human and model annotation — sampled across ten bands of
evidence strength. Precision is not flat; **it collapses**:

| evidence weight | precision |
|---|---:|
| 0.25 and above | **92%** |
| 0.167 – 0.25 | 83% |
| 0.125 – 0.167 | 67% |
| 0.100 – 0.125 | 33% |
| below 0.100 | 40% |

That curve is the reason the graph is trustworthy. Everything below `0.125` is
discarded — two-thirds of the proposed edges — because it is barely better than
chance. A single blended accuracy figure would have hidden this completely.

### 4. Position from meaning, pulled toward connection

Two things want to decide where a moment sits, and they disagree. Meaning put
the two "I am Iron Man" moments 7 units apart. Connection is what a *thread* is
made of, and threads are what the illustration has to draw. Forcing a
compromise was worse at both jobs, so the map carries **both layouts** and
toggles between them.

### 5. The graph becomes a world

The image model is a **stylist, not an author**. Asking it to invent a top-down
map from a sentence produced perspective landscapes with horizons — and a tile
containing a horizon cannot join the tile above it. Instead the composition
comes from the data: moments and threads rendered as territory, which the model
then inks in a Silver Age comic idiom.

---

## Limits

Stated plainly, because a portfolio piece that hides its seams is worth less
than one that shows them.

- **The predictive model is not built yet.** The edges come from heuristics that
  have been *measured*, not from a trained model. Fine-tuning the embeddings on
  the collected judgements, and link prediction over the graph's own topology,
  are designed but unbuilt.
- **590 of 599 moments were written by a model**, from the source text. Nine
  were hand-written to set the standard. This is declared because it affects
  what any result means.
- **Moment density is not uniform** — from one moment per 40 words to one per
  1,808. Beats were chosen for reach across titles, not plot coverage, so the
  map shows *connective significance*, not screen time. Full table in
  [PROVENANCE.md](./PROVENANCE.md).
- **Precision only, never recall.** Nobody knows how many real connections exist
  in the MCU, so the share of proposals that are correct can be measured and the
  share of real connections found cannot.
- **The tile pyramid is unfinished.** The style holds across subjects and tiles
  blend invisibly, but making a tile both its own scene *and* continuous with
  its neighbour is still open.

---

## Run it

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -r ml/requirements.txt

./.venv/bin/python ingest/registry.py     # what titles exist
./.venv/bin/python ingest/plots.py        # fetch the source text
./.venv/bin/python ml/embed.py            # meaning → vectors
./.venv/bin/python ml/relate.py           # propose typed edges
./.venv/bin/python ml/graph.py            # threads, centrality, layout
python3 -m http.server 8777               # then open localhost:8777
```

Every intermediate result is a JSON file you can open and read. There is no
database and no backend — the site is a static page reading `graph.json`, which
is why a contributor's merged pull request changes the live map.

---

## Structure

```
ingest/   Wikipedia → titles, plot text, provenance
ml/       embeddings, edge proposal, evaluation, graph building
art/      image-model experiments: style, terrain, scenes
data/     moments, connections, judgements, provenance  ← the content layer
index.html  the canvas
```

---

## Credits, licence, disclaimer

Source text is English Wikipedia, **CC BY-SA 4.0**, cited to exact revision IDs
in [PROVENANCE.md](./PROVENANCE.md). Moment descriptions are original prose
written from it and released under the same licence. Code is **MIT**.

Generated artwork uses **Stable Diffusion XL**, an open-source model, and is
original: no Marvel imagery is used, no model is trained on Marvel art, and no
recognisable character, costume or insignia is depicted. The visual idiom is
Silver Age comics printing, whose visual language was shaped above all by **Jack
Kirby** and **Steve Ditko**, credited here as influences.

An unofficial, non-commercial project. Not affiliated with, endorsed by, or
sponsored by Marvel, Marvel Studios, or The Walt Disney Company. Titles and
character names are the property of their respective owners and used here for
identification and reference only.
