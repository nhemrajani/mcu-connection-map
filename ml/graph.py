"""graph.py - build the graph, find threads, place moments on the map.

The SECOND ML step. Four ideas:

  1. Graphs - moments become nodes, confirmed connections become edges.
  2. Community detection - an unsupervised algorithm groups tightly linked
     moments into "threads" with no labels required.
  3. Centrality - which moments are the load-bearing hubs.
  4. Layout - the embeddings from embed.py are squeezed from ~384 dimensions
     down to 2, so every moment gets an x and y computed from its MEANING.

That last one is the point of the whole map: position is not decoration.
Moments about similar things land near each other because the maths put them
there, so regions of the canvas become story threads.

Run:    .venv/bin/python ml/graph.py   (run embed.py first)
Output: ml/out/graph.json

Only edges with verdict "confirmed" reach the graph. Rejected judgements stay
in data/connections.json as training data but must never draw a line.
"""
import json
from pathlib import Path

import networkx as nx
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent / "out"

# Below this many edges the graph is too sparse for community detection to say
# anything, so we fall back to clustering the embeddings directly.
MIN_EDGES_FOR_COMMUNITIES = 40


def layout(vectors, seed=42):
    """Squeeze high-dimensional meaning down to a 2D position per moment."""
    n = len(vectors)
    perplexity = max(5, min(30, (n - 1) // 3))
    coords = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        random_state=seed,
    ).fit_transform(np.asarray(vectors))
    # Normalise into a friendly 0-1000 square for the canvas.
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    span = np.where(hi - lo == 0, 1, hi - lo)
    return (coords - lo) / span * 1000


def main():
    moments = json.loads((DATA / "moments.json").read_text())
    connections = json.loads((DATA / "connections.json").read_text())
    confirmed = [c for c in connections if c.get("verdict") == "confirmed"]

    graph = nx.Graph()
    for m in moments:
        graph.add_node(m["id"])
    for c in confirmed:
        graph.add_edge(c["source"], c["target"], type=c.get("type", "related"))

    # --- threads -----------------------------------------------------------
    if len(confirmed) >= MIN_EDGES_FOR_COMMUNITIES:
        groups = list(greedy_modularity_communities(graph))
        community_of = {n: i for i, g in enumerate(groups) for n in g}
        basis = "graph communities"
    else:
        groups = []
        community_of = {}
        basis = "embedding clusters (too few confirmed edges for communities)"

    # --- centrality --------------------------------------------------------
    centrality = nx.degree_centrality(graph) if confirmed else {}

    # --- layout from meaning ----------------------------------------------
    ids = json.loads((OUT / "moment_ids.json").read_text())
    vectors = np.load(OUT / "embeddings.npy")
    order = {mid: i for i, mid in enumerate(ids)}
    vectors = np.array([vectors[order[m["id"]]] for m in moments])
    positions = layout(vectors)

    if not community_of:
        k = max(2, min(12, len(moments) // 20))
        # Cluster the 2D positions, not the raw vectors. Clusters found in 384
        # dimensions do not survive the squeeze to 2D, so colouring by them
        # scatters every colour across the whole map. Clustering the layout
        # itself means a colour is a place, which is what a map should mean.
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(positions)
        community_of = {m["id"]: int(labels[i]) for i, m in enumerate(moments)}
        groups = [set() for _ in range(k)]
        for m, lab in zip(moments, labels):
            groups[lab].add(m["id"])

    nodes = []
    for i, m in enumerate(moments):
        nodes.append({
            **m,
            "community": community_of.get(m["id"], -1),
            "centrality": round(centrality.get(m["id"], 0.0), 4),
            "x": round(float(positions[i][0]), 2),
            "y": round(float(positions[i][1]), 2),
        })

    OUT.mkdir(exist_ok=True)
    (OUT / "graph.json").write_text(json.dumps({
        "nodes": nodes,
        "edges": confirmed,
        "meta": {
            "moments": len(nodes),
            "confirmed_edges": len(confirmed),
            "rejected": len(connections) - len(confirmed),
            "threads": len(groups),
            "thread_basis": basis,
        },
    }, indent=2) + "\n")

    print(f"{len(nodes)} moments, {len(confirmed)} confirmed edges")
    print(f"{len(groups)} threads from {basis}")
    print("wrote ml/out/graph.json")


if __name__ == "__main__":
    main()
