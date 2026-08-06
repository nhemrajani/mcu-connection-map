# Data schema

Two files hold the whole map. Keep them human-editable.

## `moments.json` — the nodes

| field         | type    | required | notes                                             |
|---------------|---------|----------|---------------------------------------------------|
| `id`          | string  | yes      | unique, lowercase-with-dashes, prefixed `m-`      |
| `title`       | string  | yes      | short, spoiler-light name                         |
| `film`        | string  | yes      | `Film Title (Year)`                               |
| `phase`       | number  | no       | MCU phase, if you track it (drives colour palette)|
| `description` | string  | yes      | 1-2 original sentences (used for embeddings)      |

The `description` is what the ML embeds, so write it to capture *meaning*, not
just a label.

## `connections.json` — the edges

| field    | type   | required | notes                                                  |
|----------|--------|----------|--------------------------------------------------------|
| `source` | string | yes      | an existing moment `id`                                |
| `target` | string | yes      | an existing moment `id`                                |
| `type`   | string | yes      | see types below (drives edge style on the canvas)      |
| `note`   | string | no       | short reason the two connect                           |

### Connection types

- `setup-payoff` — one moment plants something a later one pays off
- `shared-character` — the same character links both moments
- `shared-object` — the same artifact/object appears in both
- `timeline-adjacent` — they sit next to each other in-universe
- `theme-echo` — they rhyme thematically without a hard causal link

## Derived fields (added by `ml/graph.py`, don't hand-edit)

- `community` — the thread this moment was clustered into
- `centrality` — 0-1 importance score, used for node size
