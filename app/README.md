# App (the canvas)

The frontend reads `ml/out/graph.json` and renders it on an infinite pan/zoom
surface.

## Plan

- **MVP:** React Flow (xyflow) — custom, CSS-styled nodes; pan/zoom for free.
- **Level-up (organic star-map feel):** d3-force layout, or Sigma.js (WebGL) if
  the graph grows large.

## The three things that make it feel alive

1. **Semantic zoom** — far out you see glowing thread-clusters; as you zoom in,
   individual moments resolve with their symbol + description.
2. **Data-driven style** — colour = `community` (thread), size = `centrality`,
   edge style = connection `type`.
3. **Thread tracing** — hovering a moment lights up the edges of its thread.

Nothing here yet — this is the next build step after the data and symbols exist.
