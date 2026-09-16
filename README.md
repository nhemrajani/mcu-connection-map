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
| Connections | 1,272 retained in the graph, filtered at a measured cutoff |
| Threads | 76, recovered by community detection without labels |
| Evaluation | 122 judged pairs (35 human, 87 model) |
| Imagery | 599 scenes, one per moment, generated with SDXL |
| Live map | [mcu.neeha.xyz](https://mcu.neeha.xyz) |

Both halves now run end to end. The graph is built and measured; every moment
carries a written scene, and the canvas places those scenes at their layout
coordinates and blends them where they overlap. What remains unresolved is
accuracy rather than completeness, and is set out in
[Limitations](#limitations).

---

## Findings

Three results that were not entered by hand, and that a timeline would not
surface. Each is stated with what supports it, because the interesting ones
here are not uniformly flattering.

**The Mandarin is reassembled across a one-shot and eight years.** *Iron Man 3*
(2013) reveals the Mandarin to be a hired actor, Trevor Slattery. The one-shot
*All Hail the King* (2014) revisits him in prison. *Shang-Chi* (2021) finds him
alive in a cell belonging to the real organisation. The graph links all three
directly, by shared object, with no instruction that they were related and no
human involvement in proposing them.

**Frigga's death reaches forward six years and then nine.** Her death in *The
Dark World* (2013) is linked as setup-payoff to Thor meeting her again in
*Endgame* (2019), and to Jane Foster's diagnosis in *Love and Thunder* (2022),
by way of the Aether she absorbs in the same 2013 film. This is the shape the
project was built to find: a consequence separated from its cause by most of a
decade and by three different films.

**The most connected moment in the corpus belongs to *What If...?*** The
Watcher breaking his oath sits at the top of the graph under both degree and
betweenness centrality. Part of that is real, since the Watcher is the one
character who observes every reality. Part of it is an artefact: the moment's
text names the Infinity Stones, and the Stones are named everywhere, so a
generic object inflates the count. The result is left in place rather than
tuned away, because the second half of it is the clearest statement of what
this method's weakness actually is.

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

Each edge source is a heuristic, so each was evaluated. 122 pairs have been
judged, 35 by hand and 87 by a language model, sampled across bands of evidence
strength rather than uniformly. Precision is not uniform across those bands.

| Evidence weight | n | Precision | 95% CI |
|---|---:|---:|---:|
| 0.25 and above | 53 | **92%** | 82%–97% |
| 0.167 to 0.25 | 3 | 100% | 44%–100% |
| 0.125 to 0.167 | 5 | 80% | 38%–96% |
| 0.100 to 0.125 | 5 | 20% | 4%–62% |
| below 0.100 | 7 | 43% | 16%–75% |

The intervals are the important column, and they are why the table is given
with its sample sizes rather than as a clean curve. Only the top band is
properly supported: 53 judgements, and an interval narrow enough to act on. The
middle bands rest on three to five judgements each and their intervals are
nearly the whole range, so they establish a direction and nothing finer.

The cutoff at 0.125 is therefore justified by the gap between the ends of the
distribution, not by the precise value of any middle row. Above it, nine in ten
proposals are sound. Below it, the two lowest bands together are 4 of 12, which
is not distinguishable from guessing. Everything below 0.125 is discarded,
amounting to roughly two-thirds of what the heuristics proposed.

A single aggregate figure would have concealed the distinction entirely, which
is the argument for stratified sampling. The honest reading is that the cutoff
is well placed and the curve is under-sampled; closing that gap needs a few
hundred more judgements, not a different method.

A further 49 of the 122 judged pairs can no longer be assigned to a band at
all, because the proposal step has been changed since they were judged and
those pairs are not in its current output. They are retained as training
signal, but they cannot contribute to this table.

### 4. Layout: position from meaning, drawn toward connection

Two criteria compete for the placement of each moment. Meaning groups moments
that read alike; connection is what constitutes a thread, and threads are what
the illustration must depict as contiguous regions. A blended compromise
performed worse against both criteria than either pure layout did against its
own, so the map retains both and lets the reader switch between them.

A third constraint arrived later, from the imagery rather than the analysis.
Scenes are drawn about 50 units across and must overlap in order to blend, but
the raw layout placed some pairs 0.7 units apart, which buried 57% of scenes
underneath a neighbour. A relaxation pass now pushes apart only those pairs
closer than 16 units. It is a small intervention with a measurable cost: 89% of
every moment's five nearest neighbours are still its five nearest neighbours
afterwards, the median confirmed-edge length moves from 38 units to 37, and the
average moment shifts 5 units in a 1000-unit world.

What it does remove is local density, which becomes near-uniform. That is a
smaller loss than it appears, because t-SNE does not preserve density in the
first place: the clumping the pass flattens was an artefact of the projection
rather than a fact about the corpus. Distance on this map means nearness, not
amount.

### 5. From graph to world

The image model functions as a stylist rather than an author. Asked to invent a
top-down map from a textual description it produced perspective landscapes with
horizons, twice, and a tile containing a horizon cannot be joined to the tile
above it. Composition is supplied by the data; the model supplies the idiom.

An intermediate version rendered threads as coloured territory with coastlines.
It was legible and it was wrong: it looked like an atlas, because it was one.
Nothing was depicted. The map named where things sat without showing what
happened there.

The canvas therefore draws moments rather than regions. Each of the 599 moments
carries a written scene, and each scene is generated independently at 512px
with a soft radial transparency baked into it. The browser places them at their
layout coordinates, where they overlap and their soft edges cross-dissolve.

That dissolve is the mechanism, and it is worth being precise about why it
carries meaning rather than merely looking attractive. Two scenes can only
blend if they are adjacent, they are only adjacent if the layout put them
there, and the layout only put them there because the graph connected them. A
blend between two scenes is an edge being drawn. Unconnected moments end up far
apart and never touch. Zooming in widens the blend bands, so detail and
connection become more legible together.

Nothing is composited into a single image. At a size where a scene is readable
the whole map would be roughly 42,000 pixels square, about 1.8 gigapixels,
which cannot be generated, stored or served. The scenes are served separately
and assembled per viewport, with 128px thumbnails standing in until the zoom
justifies fetching the full resolution. Below about 26 pixels a scene is an
expensive smudge, so none are loaded at all and the map remains a
constellation; between there and 92 pixels the two crossfade.

The scene width was measured rather than chosen. Sampling the inhabited part of
the map showed that at the original width 35% of the world was empty black and
the result read as a scatter of vignettes; at 50 units there are no gaps.

### 6. Writing for the image model

A composed prompt and a blunt one were rendered at identical seeds and settings
across four moments. The blunt version won three of the four, which was not the
expected result.

The failure that lost the fourth explained the rest. "A warrior's golden armour
breaking apart into sparks of light" produced abstract wallpaper, because a
transformation is not a thing. The prompts that worked each named something
solid that could be photographed. The rule for all 599 descriptions is
therefore to find the most concrete physical object in a moment and name that,
in one setting, and never to describe a transformation, an emotion, or a
relationship between people.

A second rule emerged from rendering a sample across phases, and it is specific
to a continuous canvas rather than to image quality: **put the object in a
room**. A prompt whose subject is a graphic, such as an insignia or a document,
returns a flat plate with the graphic centred and usually lettered. It has no
ground plane, no walls and no camera position, so it cannot bleed into a
neighbour and reads as a poster pasted onto the world. Thirty descriptions were
rewritten to name the space the graphic sits in instead.

The cost of both rules is stated in [Limitations](#limitations), because it is
real.

---

## Limitations

Stated directly, since a project that conceals its seams is worth less than one
that exhibits them.

- **The trained model did not beat the heuristics, and the honest result is
  the interesting one.** The embedding was fine-tuned on the project's own
  judgements, which is the step that closes the loop between labelling and
  prediction. Against easy negatives, meaning random pairs nobody proposed, ROC
  AUC rose from 0.885 to 0.899. Against hard negatives, meaning pairs a
  heuristic proposed and a reviewer rejected, it fell from 0.643 to 0.625.

  The second number is the one that matters, because it is the task the project
  actually has: every proposal in the graph already looks plausible, and the
  work is telling which of them are real. The model separates connected from
  unrelated, which was never the difficulty. It does not separate
  plausible-but-wrong from true. With 110 training pairs this demonstrates that
  the loop closes rather than that the method scales, and the direction of the
  gap is the finding.

- **Centrality rewards common nouns.** The most connected moment in the graph
  is inflated by naming the Infinity Stones, which appear throughout the
  corpus. The rarity weighting in edge proposal was designed to prevent exactly
  this and does not fully succeed on objects that are both rare per moment and
  ubiquitous across the corpus.

- **The thread structure is a few large groups and a long tail.** Community
  detection returns 76 threads, but 46 of them contain one or two moments, and
  the five largest hold 249 of the 599 between them. The graph has 62 connected
  components, the largest holding 460 moments. Thread membership is meaningful
  for the large threads and close to meaningless for the small ones.
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
- **Some moments cannot be depicted under the rules that make the rest work.**
  The rules that produce reliable images forbid naming transformations, and a
  transformation is precisely what several of the most consequential moments
  are. The Snap is a body becoming dust; the description renders as a man
  standing in a forest. The most important moment in the corpus is one the
  visual layer cannot draw. Across the whole set the image is reliably a scene
  *near* the moment rather than *of* it, and that is a real limit on what the
  visual layer can claim rather than a bug to be fixed with better prompting.

- **The imagery is a rendering of the graph, not evidence for it.** A blend
  between two scenes shows that the model connected them. It does not show that
  the connection is correct, and roughly one proposal in ten above the cutoff is
  not. The canvas inherits the graph's precision and displays it persuasively,
  which is a reason to read the two together rather than to trust the picture.

- **Text still appears in generated images.** Speech bubbles and lettering
  surface despite being listed in the negative prompt, because the Silver Age
  idiom is inseparable from them in the base model's training data. The text is
  not language.

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
./.venv/bin/python ml/evaluate.py         # precision by band, with intervals
./.venv/bin/python ml/finetune.py         # train on the judgements, both conditions
./.venv/bin/python art/atlas.py           # 599 scenes (~35 min on an M-series Mac)
python3 -m http.server 8777               # then open localhost:8777
```

`art/atlas.py` skips any scene already on disk, so an interrupted run resumes
where it stopped. Seeds are derived from the moment id, which means a rerun
reproduces the same world rather than a different one.

Every intermediate result is a JSON file that can be opened and read. There is
no database and no backend: the site is a static page reading `graph.json`,
which is why a merged contribution changes the published map.

---

## Structure

```
ingest/     Wikipedia to titles, plot text, provenance, scene descriptions
ml/         embeddings, edge proposal, evaluation, fine-tuning, graph, layout
art/        the scene renderer (atlas.py) and the experiments that preceded it
data/       moments, connections, judgements, provenance, schema
index.html  the canvas: World, Threads and Meaning
```

The three views share one layout and differ in what is drawn on it. **World**
places the generated scenes and lets them blend. **Threads** shows the same
positions as a constellation, where connection has pulled linked moments
together. **Meaning** drops the connection pull and shows the embedding alone.
Switching between World and Threads keeps the viewport, so the same region can
be read either way.

---

## Credits, licensing and disclaimer

Source text is English Wikipedia, licensed CC BY-SA 4.0 and cited to exact
revision IDs in [PROVENANCE.md](./PROVENANCE.md). Moment descriptions are
original prose written from that source and released under the same licence.
Code is released under MIT.

Generated artwork uses Stable Diffusion XL, an open-source model. No image
model was trained or fine-tuned for this project, and no Marvel artwork was
used as training data. A sentence-embedding model was fine-tuned on this
project's own judgements, as described in [Limitations](#limitations); it
touches the analysis only and plays no part in generating images.

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
