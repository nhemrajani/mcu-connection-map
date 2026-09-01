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

The `description` is what the ML embeds, so write it to capture *meaning*, not
just a label. Name people, objects and places explicitly — proper nouns are
strong signal, and a description written to avoid spoilers embeds as generic
mush. Keep the `title` spoiler-light instead; that's the part humans read first.

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
