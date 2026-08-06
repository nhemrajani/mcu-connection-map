# Contributing

The best way to help is to grow the map. You can add a **moment** or a
**connection** without writing any app code — it's just JSON.

## Add a moment

Open `data/moments.json` and add an object:

```json
{
  "id": "m-short-unique-slug",
  "title": "A short, spoiler-light name for the moment",
  "film": "Film Title (Year)",
  "phase": 1,
  "description": "1-2 sentences, in your own words, on what happens and why it matters."
}
```

Guidelines:
- Write descriptions **in your own words** — no copied plot summaries.
- Keep `id` unique and lowercase-with-dashes.
- Describe the *moment*, not the whole film.

## Add a connection

Open `data/connections.json` and add an edge between two existing moment `id`s:

```json
{ "source": "m-a", "target": "m-b", "type": "setup-payoff", "note": "why they connect" }
```

`type` is one of: `setup-payoff`, `shared-character`, `shared-object`,
`timeline-adjacent`, `theme-echo`. Add a new type if you truly need one, and note
it in `data/schema.md`.

## Not sure a connection is real?

Run the ML step — it will *suggest* candidate connections it thinks exist:

```bash
cd ml && python embed.py   # look at ml/out/candidates.json
```

Suggestions are just a starting point. A human confirms them by moving the pair
into `data/connections.json`.

## Opening a pull request

1. Fork the repo and create a branch.
2. Make your edit to the JSON.
3. Run `python ml/graph.py` to confirm the data still loads.
4. Open a PR describing the moment/connection you added.

Be kind, keep it spoiler-aware in titles, and have fun.
