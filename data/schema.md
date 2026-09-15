# Data schema

Two files hold the whole map. Keep them human-editable.

## `moments.json` — the nodes

| field         | type    | required | notes                                             |
|---------------|---------|----------|---------------------------------------------------|
| `id`          | string  | yes      | unique, lowercase-with-dashes, prefixed `m-`      |
| `title`       | string  | yes      | short, spoiler-light name                         |
| `film`        | string  | yes      | `Film Title (Year)`                               |
| `universe`    | string  | no       | `mcu`, `raimi`, `webb`, … which continuity it belongs to |
| `phase`       | number  | no       | MCU phase, if you track it (drives colour palette). `null` for non-MCU |
| `description` | string  | yes      | 1-2 original sentences (used for embeddings)      |
| `visual`      | string  | no       | a composed scene for the image model: subject, setting, framing |

The `description` is what the ML embeds, so write it to capture *meaning*, not
just a label. Name people, objects and places explicitly — proper nouns are
strong signal, and a description written to avoid spoilers embeds as generic
mush. Keep the `title` spoiler-light instead; that's the part humans read first.

### `visual` versus `description`

`description` is written for the ML: it names people and objects explicitly,
because proper nouns are what similarity and entity matching grip onto.

`visual` is written for the image model, and the two want opposite things. A
plot sentence clipped out of a description gives a diffusion model nothing to
stage; it needs a subject, a setting and a camera. `visual` also carries the
project's depiction rule, which cannot be enforced by a prompt written at
render time: figures may appear, but never a named character, a recognisable
costume or an insignia. People are described by silhouette and action.

Where `visual` is absent the renderer falls back to extracting imagery from
`description`, which works but reads as a fragment rather than a scene.

### Scope

The map covers the MCU **and the continuities the MCU has pulled into itself**
via the multiverse — the Raimi and Webb Spider-Man films are in scope because
*No Way Home* makes them canon-adjacent. `universe` is how they're kept
distinguishable.

## `connections.json` — the edges

| field         | type   | required | notes                                                  |
|---------------|--------|----------|--------------------------------------------------------|
| `source`      | string | yes      | an existing moment `id`                                |
| `target`      | string | yes      | an existing moment `id`                                |
| `verdict`     | string | yes      | `confirmed` or `rejected` — see below                  |
| `type`        | string | if confirmed | see types below (drives edge style on the canvas)  |
| `note`        | string | no       | short reason the two connect, or why they don't        |
| `proposed_by` | string | no       | `human`, or the model that suggested it                |
| `score`       | number | no       | similarity score, when a model proposed it             |
| `judged_at`   | string | no       | ISO date the call was made                             |

### Why rejections are stored

A rejected connection is not deleted, it is **kept with `verdict: "rejected"`**.

The graph and the canvas only ever read `confirmed` edges, so rejections are
invisible in the final map. They exist because they are training data. Every
judgement — yes and no — is a labelled example of what counts as a real
narrative connection *in this project's opinion*, and those labels are what let
the embedding model be fine-tuned to match that opinion rather than generic
semantic similarity.

Throw the rejections away and you keep only positive examples, which is not
enough to teach a model the boundary. Storing them costs nothing now and
cannot be reconstructed later.

### Connection types

- `setup-payoff` — one moment plants something a later one pays off
- `shared-character` — the same character links both moments
- `shared-object` — the same artifact/object appears in both
- `timeline-adjacent` — they sit next to each other in-universe
- `theme-echo` — they rhyme thematically without a hard causal link

## Derived fields (added by `ml/graph.py`, don't hand-edit)

- `community` — the thread this moment was clustered into
- `centrality` — 0-1 importance score, used for node size
