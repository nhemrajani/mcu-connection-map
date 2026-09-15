# The MCU Connection Map

**An experiment at the intersection of design and machine learning, asking
whether a story can be understood better by being seen.**

The Marvel Cinematic Universe comprises seventy-six released films and series.
It is conventionally described as a timeline: an ordered list, read from top to
bottom. A narrative of that scale is not well served by a list. It is a web of
setups and payoffs, of objects that change hands, of characters who reappear
decades later, and of beats that rhyme across works with little else in common.

This project investigates two questions:

1. **Can a model predict narrative connection and causality?** The task is not
   to identify scenes that use similar language, but to identify where one
   moment establishes something that another later resolves.
2. **Can the resulting structure be rendered as an image?** The graph is passed
   to an open-source image model, which draws it as a continuous illustrated
   world that becomes more specific under magnification.

The first question is a machine learning problem. The second is a design
problem. The premise of the project is that they are the same problem, and that
neither is complete without the other.

---

## Current state

| | |
|---|---|
| Moments | 599, drawn from all 76 released MCU titles |
| Source text | 130,393 words of Wikipedia plot summary, cited to revision IDs |
| Connections | 1,248 retained in the graph, filtered at a measured cutoff |
| Threads | 39, recovered by community detection without labels |
| Evaluation | 114 judged pairs (27 human, 87 model) |
| Live map | [mcu.neeha.xyz](https://mcu.neeha.xyz) |

The graph half of the project is complete and measured. The visual half is
partly built: the visual idiom has been demonstrated to hold across subjects,
and adjacent tiles have been shown to join without a visible seam, but the tile
pyramid itself is unfinished. See [Limitations](#limitations).

---

## Findings

Three results that were not entered by hand, and that a timeline would not
surface.

**The two "I am Iron Man" moments are placed seven units apart.** The opening
declaration of the MCU in 2008 and its closing one in 2019, separated by
thirty-three films, are positioned as near neighbours by an embedding that was
never told they were related.

**Iron Man and Shang-Chi fall within the same thread.** The Ten Rings abduct
Tony Stark in 2008 and are left unexplained. Thirteen years later they are
revealed as a thousand-year-old organisation led by Shang-Chi's father.
Community detection grouped both films into a single region without prompting.

**The most central moment in the corpus is the death of Phil Coulson.** Not a
battle, and not the snap. It is the death that forces the Avengers into a team,
and the point at which the films hand off to *Agents of S.H.I.E.L.D.*

---

## Method

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

### 1. Meaning as geometry

A sentence-transformer converts each moment's description into a vector of 384
numbers. Descriptions with similar meaning produce similar vectors, so every
moment acquires a position, and the angle between two vectors measures their
similarity.

### 2. Similarity is not causality

This is the central methodological finding. Cosine similarity measures whether
two passages sound alike. It cannot measure whether one caused the other, and
it systematically misses the connections that matter most. A drop of
gamma-irradiated blood enters Samuel Sterns' wound in 2008 and pays off
seventeen years later in *Brave New World*. The two descriptions share almost
no vocabulary, so similarity cannot see the link, whilst a method tracking
which entities appear in each finds it immediately.

The graph is therefore built from three signals:

- **Shared rare entities.** "Tony Stark" occurs in 35 moments and carries no
  information. "Quantum Realm" occurs in five, and those five belong together.
- **Release chronology**, which supplies direction: the earlier moment
  establishes what the later one resolves.
- **Stated references**, being passages where the source asserts the connection
  explicitly, as in "Following the Battle of New York, Adrian Toomes and his
  salvage company are contracted to clean up the city".

### 3. Measurement, then filtering

Each edge source is a heuristic, so each was evaluated. 114 pairs were judged,
sampled across ten bands of evidence strength. Precision is not uniform across
those bands; it collapses.

| Evidence weight | Precision |
|---|---:|
| 0.25 and above | **92%** |
| 0.167 to 0.25 | 83% |
| 0.125 to 0.167 | 67% |
| 0.100 to 0.125 | 33% |
| below 0.100 | 40% |

This curve is the basis on which the graph can be trusted. Everything below
0.125, amounting to two-thirds of the proposed edges, is discarded as barely
better than chance. A single aggregate accuracy figure would have concealed the
distinction entirely, which is the argument for stratified sampling rather than
uniform sampling.

### 4. Layout: position from meaning, drawn toward connection

Two criteria compete for the placement of each moment. Meaning placed the two
"I am Iron Man" moments seven units apart. Connection is what constitutes a
thread, and threads are what the illustration must depict as contiguous
regions. A blended compromise performed worse against both criteria than either
pure layout did against its own, so the map retains both and allows the reader
to switch between them.

### 5. From graph to world

The image model functions as a stylist rather than an author. Prompting it to
invent a top-down map from a textual description produced perspective
landscapes containing horizons across two separate attempts, and a tile
containing a horizon cannot be joined to the tile above it. Composition is
therefore supplied by the data, with moments and threads rendered as territory,
which the model then renders in a Silver Age comics idiom.

---

## Limitations

Stated directly, since a project that conceals its seams is worth less than one
that exhibits them.

- **The predictive model is not yet built.** Edges are produced by heuristics
  that have been measured rather than by a trained model. Fine-tuning the
  embedding on the collected judgements, and link prediction over the graph's
  own topology, are specified but unimplemented.
- **590 of 599 moment descriptions were written by a language model** from the
  source text; nine were written by hand to establish the standard. This is
  declared because it bears on the interpretation of any result.
- **Moment density is not uniform**, ranging from one moment per 40 words of
  source to one per 1,808. Beats were selected for reach across titles rather
  than for plot coverage, so the map represents connective significance rather
  than screen time. The full table is given in [PROVENANCE.md](./PROVENANCE.md).
- **Precision is reported; recall is not.** The number of genuine connections in
  the corpus is unknown, so the proportion of proposals that are correct can be
  measured whilst the proportion of genuine connections recovered cannot.
- **The tile pyramid is unfinished.** The visual idiom holds across subjects and
  adjacent tiles join without a visible seam, but producing a tile that is
  simultaneously a distinct scene and continuous with its neighbour remains
  unsolved.

---

## Reproducing

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -r ml/requirements.txt

./.venv/bin/python ingest/registry.py     # derive the title list
./.venv/bin/python ingest/plots.py        # fetch the source text
./.venv/bin/python ml/embed.py            # meaning to vectors
./.venv/bin/python ml/relate.py           # propose typed edges
./.venv/bin/python ml/graph.py            # threads, centrality, layout
python3 -m http.server 8777               # then open localhost:8777
```

Every intermediate result is a JSON file that can be opened and read. There is
no database and no backend: the site is a static page reading `graph.json`,
which is why a merged contribution changes the published map.

---

## Structure

```
ingest/     Wikipedia to titles, plot text, provenance
ml/         embeddings, edge proposal, evaluation, graph construction
art/        image model experiments: style, terrain, scenes
data/       moments, connections, judgements, provenance
index.html  the canvas
```

---

## Credits, licensing and disclaimer

Source text is English Wikipedia, licensed CC BY-SA 4.0 and cited to exact
revision IDs in [PROVENANCE.md](./PROVENANCE.md). Moment descriptions are
original prose written from that source and released under the same licence.
Code is released under MIT.

Generated artwork uses Stable Diffusion XL, an open-source model. No model was
trained or fine-tuned for this project, and no Marvel artwork was used as
training data.

It should be acknowledged plainly that Marvel's house style is present in
SDXL's training data, and that prompting for the Silver Age comics idiom
therefore draws on it. Some generated imagery depicts recognisable characters,
costumes and insignia. This is a property of the base model rather than an
attempt to reproduce specific works, and it is disclosed here rather than
implied to be absent.

The visual idiom is that of Silver Age comics printing, whose visual language
was shaped above all by **Jack Kirby** and **Steve Ditko**, credited here as
influences.

This is an unofficial, non-commercial project. Nothing here is sold, licensed or
monetised in any form. It is not affiliated with, endorsed by, or sponsored by
Marvel, Marvel Studios, or The Walt Disney Company. Marvel, the Marvel
Cinematic Universe, and all associated characters, titles, logos and insignia
are the property of Marvel Entertainment and The Walt Disney Company, and appear
here only in the context of commentary and analysis of the works themselves.
Requests to remove specific material will be honoured.
