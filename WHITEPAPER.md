# Drawing a Story's Structure

### Recovering narrative connection from plot summaries, and rendering it as a continuous illustrated world

**Neeharika Hemrajani**
September 2026
[mcu.neeha.xyz](https://mcu.neeha.xyz) · [source](https://github.com/nhemrajani/mcu-connection-map)

---

## Abstract

The Marvel Cinematic Universe is conventionally represented as a timeline: 76
titles in an ordered list. A list is a poor description of a narrative whose
distinguishing property is that its parts refer to one another across decades.
This paper reports an attempt to recover that referential structure
automatically from 130,393 words of plot summary, and to render the recovered
structure as a continuous illustrated world rather than a diagram.

Two findings are of general interest. The first is that semantic similarity is
close to useless for this task and that entity overlap is not, which is
unsurprising in retrospect and was not anticipated. The second is that a model
fine-tuned on the project's own judgements improves at separating connected
pairs from random ones and gets *worse* at separating true connections from
plausible false ones, which is the only distinction that matters. The
difficulty is not recognising connection. It is recognising incorrect
connection.

The visual half of the work is reported with a comparable result: the
constraints that make image generation reliable also make certain classes of
moment undepictable, including the single most consequential moment in the
corpus.

---

## 1. The problem

A timeline answers the question "what happened next". The questions a large
narrative actually raises are of a different kind. What did this set up? What
was this paying off? Which apparently unrelated works are quietly the same
story?

These are questions about a graph, not a sequence. The MCU is an unusually good
subject for asking them because its scale makes the structure impossible to
hold in one's head, and because a great deal of it is explicitly constructed
around deferred payoff. An object introduced as set dressing in 2008 becomes a
plot in 2021.

The project had two questions:

1. Can a model predict narrative connection and causality from plot text?
2. Can the resulting structure be rendered as an image that is worth looking
   at, and that shows something a diagram would not?

The premise, and the reason the two were pursued together, is that a structure
nobody can see has not really been recovered.

---

## 2. Corpus

Titles are derived programmatically from Wikipedia's list articles rather than
entered by hand, so the corpus can be re-derived as the MCU grows. Plot text is
fetched for each title and cited to an exact revision ID, which makes every
statement in the corpus traceable to a specific version of a specific page.

The corpus is 76 titles and 130,393 words, reduced to **599 moments**: short
original descriptions of discrete narrative events. Moment density is
deliberately uneven, ranging from one moment per 40 words of source to one per
1,808, because beats were selected for reach across titles rather than for
plot coverage. The map therefore represents connective significance rather than
screen time, and this is a limitation as much as a design choice.

Three ingestion problems are worth recording because each was silent.
Wikipedia returns HTTP 403 to Python's default user agent. Rate limiting during
one run shrank the title registry from 84 to 66 without raising an error, which
was fixed by making the registry merge-safe so that a failed fetch can never
reduce it. And *Agents of S.H.I.E.L.D.* initially yielded 362 words, because
its plot summaries are transcluded through several levels of template
indirection; following those recursively yielded 24,148.

---

## 3. Recovering connection

### 3.1 Similarity is not causality

The first approach was the obvious one: embed each moment, compute cosine
similarity, propose the nearest pairs. It fails, and it fails in an instructive
direction.

Consider a drop of gamma-irradiated blood entering a wound in *The Incredible
Hulk* (2008), which pays off seventeen years later in *Brave New World* (2025).
The two descriptions share almost no vocabulary. Similarity cannot see the
link. Conversely, any two battle scenes are highly similar and usually
unrelated.

Semantic similarity measures whether two passages *sound alike*. Narrative
connection is a claim about consequence. These are different relations, and the
first is not an approximation of the second.

### 3.2 What works instead

Edges are proposed from three signals: shared rare entities, weighted by the
inverse of how often the entity occurs across the corpus; release chronology,
which supplies direction, since the earlier moment establishes what the later
one resolves; and stated references, where the source asserts the connection
explicitly.

Rarity weighting is the load-bearing part. "Tony Stark" occurs in 35 moments
and carries no information. "Quantum Realm" occurs in five, and those five
belong together.

Three bugs in this scheme were found by measurement rather than by reading the
code, which is the argument for measuring.

- **Place names pass the rarity filter and mean nothing.** Two moments both
  occurring in Wakanda are not thereby connected.
- **Person names across incompatible continuities produce false edges.** Three
  different Peter Parkers are not the same person. Edges built on shared people
  across universes are now dropped, while shared *objects* survive at half
  weight as thematic echo.
- **Weak-tail edges are close to noise**, which the precision curve in §3.3
  made visible.

### 3.3 Measurement

122 pairs have been judged, 35 by hand and 87 by a language model, sampled
across bands of evidence strength rather than uniformly. Uniform sampling would
have produced a single aggregate accuracy that concealed the structure
entirely.

| Evidence weight | n | Precision | 95% CI |
|---|---:|---:|---:|
| 0.25 and above | 53 | **92%** | 82%–97% |
| 0.167 to 0.25 | 3 | 100% | 44%–100% |
| 0.125 to 0.167 | 5 | 80% | 38%–96% |
| 0.100 to 0.125 | 5 | 20% | 4%–62% |
| below 0.100 | 7 | 43% | 16%–75% |

The confidence intervals are the honest content of this table. Only the top
band is properly supported. The middle bands rest on three to five judgements
each and establish a direction and nothing finer. The retained cutoff of 0.125
is justified by the gap between the ends of the distribution rather than by the
value of any middle row: above it, nine proposals in ten are sound; the two
lowest bands together are 4 of 12, which is not distinguishable from guessing.
Roughly two-thirds of proposed edges are discarded.

Precision is reported and **recall is not**, because the number of genuine
connections in the corpus is unknown. The proportion of proposals that are
correct is measurable; the proportion of real connections recovered is not.

### 3.4 A cost of the cutoff

The weight metric penalises exactly the structures the project was built to
find. The Tesseract recurs across many moments, and recurrence lowers its
rarity weight, so the Tesseract chain scores 0.100 and falls below the cutoff.
A threshold that raises precision severs the spine of the Infinity Saga.

This is not resolved. It is a genuine tension between a metric that rewards
rarity and a narrative whose most important objects are the ones that recur.

### 3.5 The trained model, and the result that matters

Every edge above is produced by a heuristic. The step that closes the loop is
to treat the accumulated judgements as labels and fine-tune the embedding on
them, so the model learns what this project means by a connection.

Pairs are split by **moment** rather than at random. A random split would place
the two halves of a connection on opposite sides, so a moment seen in training
would reappear at test time and the score would flatter itself.

Two conditions were run and both are reported, because the choice of negative
changes the result more than the training does.

| Negatives | ROC AUC before | after |
|---|---:|---:|
| Easy: random pairs nobody proposed | 0.885 | **0.899** |
| Hard: pairs proposed and then rejected | 0.643 | **0.625** |

Against easy negatives the model improves. Against hard negatives it gets
worse.

The second row is the project's task. Every proposal in the graph already looks
plausible, because a heuristic nominated it; the work is deciding which of them
are real. The model separates connected from unrelated, which was never the
difficulty, and does not separate plausible-but-wrong from true.

With 110 training pairs this demonstrates that the loop closes, not that the
method scales. But the direction is informative, and it suggests that
collecting more of the same kind of label will not be sufficient. The signal
that distinguishes a true connection from a plausible false one does not appear
to be present in the pair of descriptions considered alone.

---

## 4. Results

Three results that were not entered by hand.

**The Mandarin is reassembled across a one-shot and eight years.** *Iron Man 3*
(2013) reveals the Mandarin to be a hired actor. The one-shot *All Hail the
King* (2014) revisits him in prison. *Shang-Chi* (2021) finds him alive in a
cell belonging to the real organisation. The graph links all three directly by
shared object, unprompted.

**Frigga's death reaches forward six years and then nine.** Her death in *The
Dark World* (2013) is linked as setup-payoff to Thor meeting her again in
*Endgame* (2019), and to Jane Foster's diagnosis in *Love and Thunder* (2022)
by way of the Aether. This is the shape the project was built to find: a
consequence separated from its cause by most of a decade and three films.

**The most connected moment in the corpus belongs to *What If...?***, and the
reason is half wrong. The Watcher breaking his oath ranks first under both
degree and betweenness centrality. Part of this is real, since the Watcher is
the character who observes every reality. Part is artefact: the moment's text
names the Infinity Stones, which are named everywhere, so a generic object
inflates the count. The rarity weighting was designed to prevent this and does
not fully succeed against objects that are rare per moment and ubiquitous
across the corpus.

The result is left in place rather than tuned away, because the second half of
it is the clearest available statement of the method's weakness.

---

## 5. Rendering the structure

### 5.1 What the picture had to avoid being

An intermediate version rendered threads as coloured territory with coastlines.
It was legible and it was wrong: it looked like an atlas, because it was one.
Nothing was depicted. It named where things sat without showing what happened
there.

Asked to invent a map directly, the image model produced perspective landscapes
containing horizons, twice. A tile containing a horizon cannot be joined to the
tile above it. The conclusion was that composition must be supplied by the data
and the model restricted to supplying the idiom.

### 5.2 Blending as a representation of connection

Each of the 599 moments carries a written scene. Each scene is generated
independently at 512 pixels with a soft radial transparency baked into it. The
browser places the scenes at their layout coordinates, where they overlap and
their soft edges cross-dissolve.

The dissolve is the representational mechanism, and its meaning is inherited
rather than decorative. Two scenes can blend only if they are adjacent; they
are adjacent only because the layout placed them so; and the layout placed them
so only because the graph connected them. **A blend between two scenes is an
edge being drawn.** Unconnected moments are far apart and never touch. Zooming
in widens the blend bands, so detail and connection become legible together.

Nothing is composited into a single image. At a size where a scene is readable,
the whole map would be roughly 42,000 pixels square, about 1.8 gigapixels.
Scenes are served individually and assembled per viewport, with thumbnails
standing in until the zoom justifies full resolution, and no imagery loading at
all below the scale where a scene would be readable.

Two parameters were derived rather than chosen by eye. The scene width is
locked to the layout: the separation pass guarantees no two moments sit closer
than 16 units, and a scene is opaque out to 55% of its half-width, so setting
the opaque radius equal to that separation fixes the width at 58 units. It is
the largest size at which no scene can swallow a neighbour's centre, and it is
also past the point where gaps close, 35% of the map being empty black at 34
units and none at 58. And 57% of scenes had their own centre buried beneath a neighbour,
because the raw layout placed some pairs 0.7 units apart, so a relaxation pass
now separates only pairs closer than 16 units. Its cost is measurable: 89% of
every moment's five nearest neighbours survive it, median confirmed-edge length
moves from 38 units to 37, and the average moment shifts 5 units in a
1000-unit world.

### 5.3 Writing for an image model

A composed prompt and a blunt one were rendered at identical seeds across four
moments. The blunt version won three of four.

The failure that lost the fourth explains the rest. "A warrior's golden armour
breaking apart into sparks of light" produced abstract wallpaper, because **a
transformation is not a thing**. The prompts that worked each named something
solid that could be photographed. The resulting rule, applied to all 599
descriptions, is to name the most concrete physical object in the moment, in
one setting, and never to describe a transformation, an emotion, or a relation
between people.

A second rule is specific to a continuous canvas: **put the object in a room**.
A prompt whose subject is a graphic, such as an insignia or a document, returns
a flat plate with the graphic centred and usually lettered. It has no ground
plane, no walls and no camera position, so it cannot bleed into a neighbour and
reads as a poster pasted onto the world.

Two implementation faults are recorded because both were silent and both cost
several rounds of wrong output. CLIP accepts 77 tokens and discards the rest
without complaint, and the style string plus a composed description exceeded
that, so the model never saw the end of the prompt it was meant to draw.
Separately, `guidance_scale=0` disables classifier-free guidance and therefore
disables the negative prompt entirely, so every instruction about what not to
draw was ignored while appearing to be configured.

### 5.4 What cannot be drawn

The rules that make generation reliable forbid naming transformations. Several
of the most consequential moments in the corpus *are* transformations. The Snap
is a body becoming dust; the description renders as a man standing in a forest.

The most important moment in the corpus is one this visual layer cannot depict.
Across the whole set the image is reliably a scene *near* the moment rather
than *of* it. This is a limit on what the visual layer can claim, not a defect
to be fixed with better prompting.

---

## 6. Limitations

- **Recall is unmeasured** and unmeasurable without an exhaustive ground truth.
- **The thread structure is a few large groups and a long tail.** 76 threads are
  returned, but 46 contain one or two moments and the five largest hold 249 of
  599. The graph has 62 connected components. Thread membership is meaningful
  for large threads and close to meaningless for small ones.
- **590 of 599 moment descriptions were written by a language model** from the
  source text, with nine written by hand to establish the standard.
- **The imagery renders the graph; it is not evidence for it.** A blend shows
  that the model connected two moments. It does not show the connection is
  correct, and roughly one proposal in ten above the cutoff is not. The canvas
  inherits the graph's precision and displays it persuasively.
- **Text appears in generated images** despite being negatively prompted,
  because the idiom is inseparable from lettering in the base model's training
  data. The text is not language.

---

## 7. Conclusion

Narrative connection can be recovered from plot summaries well enough to be
worth looking at, provided the method is built on entity overlap rather than
semantic similarity, and provided the output is filtered at a threshold
established by stratified measurement rather than assumed.

The negative results are the more useful contribution. Fine-tuning on the
project's own judgements improved performance on the easy discrimination and
degraded it on the hard one, which suggests that distinguishing a true
connection from a plausible false one may not be possible from the pair of
descriptions alone. And a rarity-weighted metric systematically undervalues
recurring objects, which in this corpus are precisely the objects the story is
built around.

On the design question, the result is more positive. Making adjacency the
carrier of meaning, and blending the carrier of adjacency, produces an image
in which a visual property corresponds to a claim in the data. It is a
continuous illustrated world in which proximity is not decorative. That it
cannot depict a transformation is a real constraint, and it is stated here
rather than concealed, on the principle that a project which exhibits its seams
is worth more than one that hides them.

---

## Reproducibility, licensing and disclaimer

All intermediate results are JSON files. There is no database and no backend;
the published map is a static page reading one graph file. Source text is
English Wikipedia, CC BY-SA 4.0, cited to exact revision IDs. Moment
descriptions are original prose under the same licence. Code is MIT.

Imagery is generated with Stable Diffusion XL, an open-source model. No image
model was trained or fine-tuned for this project and no Marvel artwork was used
as training data. A sentence-embedding model was fine-tuned on this project's
own judgements for the analysis described in §3.5; it plays no part in
generating images.

Marvel's house style is present in SDXL's training data, and prompting for a
Silver Age comics idiom draws on it, so some generated imagery depicts
recognisable characters, costumes and insignia. This is a property of the base
model rather than an attempt to reproduce specific works, and it is disclosed
rather than implied to be absent. The idiom is that of Silver Age comics
printing, whose visual language was shaped above all by **Jack Kirby** and
**Steve Ditko**, credited here as influences.

This is an unofficial, non-commercial project. Nothing is sold, licensed or
monetised. It is not affiliated with, endorsed by, or sponsored by Marvel,
Marvel Studios, or The Walt Disney Company. Requests to remove specific
material will be honoured.
