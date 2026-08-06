"""
graph.py - build the connection graph, find threads, and export it for the canvas.

This is the SECOND ml step. It teaches three ideas:

  1. Graphs - moments become nodes and confirmed connections become edges
     (using networkx).
  2. Community detection - an unsupervised algorithm groups tightly-linked
     moments into "threads" with no labels required.
  3. Centrality - which moments are the load-bearing hubs of the whole universe.

The two derived numbers drive the visuals downstream:
     community -> node colour,   centrality -> node size.

Run:    python graph.py
Output: ml/out/graph.json  (nodes with community + centrality, plus edges)
"""

import json
from pathlib import Path

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent / "out"


def main():
    moments = json.loads((DATA / "moments.json").read_text())
    connections = json.loads((DATA / "connections.json").read_text())

    # 1. build the graph
    G = nx.Graph()
    for m in moments:
        G.add_node(m["id"], title=m["title"], film=m.get("film"))
    for c in connections:
        G.add_edge(c["source"], c["target"], type=c.get("type", "related"))

    # 2. community detection - the unsupervised "threads"
    communities = list(greedy_modularity_communities(G))
    community_of = {}
    for idx, group in enumerate(communities):
        for node in group:
            community_of[node] = idx
    print(f"Found {len(communities)} threads across {G.number_of_nodes()} moments:")
    for idx, group in enumerate(communities):
        print(f"  thread {idx}: {sorted(group)}")

    # 3. centrality - how connected / important each moment is
    centrality = nx.degree_centrality(G)

    # 4. export enriched nodes + the edges, ready for the frontend
    nodes = []
    for m in moments:
        nid = m["id"]
        nodes.append({
            **m,
            "community": community_of.get(nid, -1),
            "centrality": round(centrality.get(nid, 0.0), 4),
        })

    graph = {"nodes": nodes, "edges": connections}
    OUT.mkdir(exist_ok=True)
    (OUT / "graph.json").write_text(json.dumps(graph, indent=2))
    print("\nWrote ml/out/graph.json - feed this straight into the canvas.")


if __name__ == "__main__":
    main()
