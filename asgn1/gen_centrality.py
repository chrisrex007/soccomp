"""Assignment 1 — Centrality metrics for the cit-HepTh subgraph.

Computes closeness, betweenness, eigenvector, and biased PageRank
centrality for every node.  All algorithms are hand-implemented
(no NetworkX centrality built-ins).
"""

from __future__ import annotations

import argparse
import math
from collections import deque
from pathlib import Path

import networkx as nx


# ---------------------------------------------------------------------------
# Argument parsing and setup
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse the subgraph filename from the command line."""

    parser = argparse.ArgumentParser()
    parser.add_argument("subgraph_name")
    return parser.parse_args()


def ensure_output_directory() -> None:
    """Create the centrality output directory."""

    Path("centralities").mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------

def load_subgraph(subgraph_name: str) -> nx.Graph:
    """Load the edge list from subgraphs/<subgraph_name> and return an
    undirected NetworkX graph."""

    graph = nx.Graph()
    filepath = Path("subgraphs") / subgraph_name

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                if u != v:
                    graph.add_edge(u, v)

    return graph


# ---------------------------------------------------------------------------
# BFS helper used by closeness and betweenness
# ---------------------------------------------------------------------------

def _bfs_shortest_paths(graph: nx.Graph, source):
    """BFS from *source*.

    Returns
    -------
    dist : dict[node, int]
        Shortest-path distance from *source* to every reachable node.
    num_paths : dict[node, int]
        Number of shortest paths from *source* to every reachable node.
    predecessors : dict[node, list[node]]
        Predecessor list on the shortest-path DAG.
    order : list[node]
        Nodes in BFS discovery order (used for back-propagation).
    """

    dist: dict = {source: 0}
    num_paths: dict = {source: 1}
    predecessors: dict = {source: []}
    order: list = []
    queue = deque([source])

    while queue:
        u = queue.popleft()
        order.append(u)
        for v in graph.neighbors(u):
            # First visit — tree edge.
            if v not in dist:
                dist[v] = dist[u] + 1
                num_paths[v] = num_paths[u]
                predecessors[v] = [u]
                queue.append(v)
            elif dist[v] == dist[u] + 1:
                # Another shortest path found.
                num_paths[v] += num_paths[u]
                predecessors[v].append(u)

    return dist, num_paths, predecessors, order


# ---------------------------------------------------------------------------
# 1. Closeness centrality
# ---------------------------------------------------------------------------

def compute_closeness(graph: nx.Graph) -> dict:
    """Compute closeness centrality C_i = 1 / sum(d_ij) for each node.

    Nodes that cannot reach any other node receive a centrality of 0.
    """

    closeness: dict = {}

    for node in graph.nodes():
        dist, _, _, _ = _bfs_shortest_paths(graph, node)
        total_dist = sum(dist.values())  # includes dist[node]=0
        if total_dist > 0:
            closeness[node] = 1.0 / total_dist
        else:
            closeness[node] = 0.0

    return closeness


# ---------------------------------------------------------------------------
# 2. Betweenness centrality  (Brandes' algorithm)
# ---------------------------------------------------------------------------

def compute_betweenness(graph: nx.Graph) -> dict:
    """Compute betweenness centrality using Brandes' algorithm.

    B_i = 2/((n-1)(n-2)) * sum_{s,t} (n_i^{st} / g_{st})
    """

    n = graph.number_of_nodes()
    betweenness: dict = {node: 0.0 for node in graph.nodes()}

    for s in graph.nodes():
        dist, num_paths, predecessors, order = _bfs_shortest_paths(graph, s)

        # Back-propagation of pair dependencies.
        delta: dict = {node: 0.0 for node in graph.nodes()}

        # Process nodes in reverse BFS order (farthest first).
        for w in reversed(order):
            for pred in predecessors.get(w, []):
                # Fraction of shortest paths through pred.
                delta[pred] += (num_paths[pred] / num_paths[w]) * (1.0 + delta[w])
            if w != s:
                betweenness[w] += delta[w]

    # Normalisation for undirected graph: each pair (s,t) counted twice,
    # and formula has factor 2/((n-1)(n-2)).
    # Brandes counts each pair once from each direction, so raw sum is
    # already double-counted.  Divide by 2 then apply the formula factor.
    norm = (n - 1) * (n - 2)
    if norm > 0:
        for node in betweenness:
            # Raw sum from Brandes is sum over all sources s of delta_s(v),
            # which double-counts because (s->t) and (t->s) are both visited.
            # For undirected graphs, divide by 2 first.
            betweenness[node] = (2.0 / norm) * (betweenness[node] / 2.0)

    return betweenness


# ---------------------------------------------------------------------------
# 3. Eigenvector centrality  (power iteration)
# ---------------------------------------------------------------------------

def compute_eigenvector(graph: nx.Graph) -> dict:
    """Compute eigenvector centrality via power iteration (up to 100 iters).

    x_i = (1/lambda) * sum_j A_ij * x_j
    """

    nodes = list(graph.nodes())
    n = len(nodes)
    if n == 0:
        return {}

    # Initialise with uniform values.
    x: dict = {node: 1.0 / n for node in nodes}

    for _ in range(100):
        x_new: dict = {}
        for node in nodes:
            x_new[node] = sum(x[nbr] for nbr in graph.neighbors(node))

        # Normalise by the largest value (approximates dividing by lambda).
        max_val = max(abs(v) for v in x_new.values())
        if max_val == 0:
            break
        for node in nodes:
            x_new[node] /= max_val

        # Check convergence.
        diff = sum(abs(x_new[node] - x[node]) for node in nodes)
        x = x_new
        if diff < 1e-10:
            break

    return x


# ---------------------------------------------------------------------------
# 4. Biased PageRank  (power iteration, alpha=0.8)
# ---------------------------------------------------------------------------

def compute_pagerank(graph: nx.Graph) -> dict:
    """Compute biased PageRank with damping alpha=0.8 and preference vector
    biased towards nodes with IDs divisible by 3.  Up to 100 iterations.
    """

    alpha = 0.8
    nodes = list(graph.nodes())
    n = len(nodes)
    if n == 0:
        return {}

    # Preference vector: higher weight for nodes whose ID is divisible by 3.
    pref: dict = {}
    for node in nodes:
        if node % 3 == 0:
            pref[node] = 1.0
        else:
            pref[node] = 0.0

    # If no nodes are divisible by 3, fall back to uniform.
    pref_sum = sum(pref.values())
    if pref_sum == 0:
        pref = {node: 1.0 / n for node in nodes}
    else:
        for node in nodes:
            pref[node] /= pref_sum

    # Initialise PageRank uniformly.
    pr: dict = {node: 1.0 / n for node in nodes}

    for _ in range(100):
        pr_new: dict = {}

        # Dangling node mass (nodes with no neighbours).
        dangling_sum = sum(pr[node] for node in nodes if graph.degree(node) == 0)

        for node in nodes:
            # Contribution from incoming neighbours.
            rank_sum = sum(pr[nbr] / graph.degree(nbr) for nbr in graph.neighbors(node))
            # Add dangling mass distributed proportionally to preference.
            rank_sum += dangling_sum * pref[node]
            pr_new[node] = alpha * rank_sum + (1.0 - alpha) * pref[node]

        # Normalise to ensure probabilities sum to 1.
        total = sum(pr_new.values())
        if total > 0:
            for node in nodes:
                pr_new[node] /= total

        # Check convergence.
        diff = sum(abs(pr_new[node] - pr[node]) for node in nodes)
        pr = pr_new
        if diff < 1e-10:
            break

    return pr


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_scores(filename: str, scores: dict) -> None:
    """Write centrality scores to centralities/<filename>.

    Nodes are sorted by centrality value (highest first).
    Values are rounded to 6 decimal places.
    """

    outpath = Path("centralities") / filename
    # Sort descending by score, break ties by node id ascending.
    sorted_scores = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))

    with open(outpath, "w") as f:
        for node, score in sorted_scores:
            f.write(f"{node} {score:.6f}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    ensure_output_directory()

    graph = load_subgraph(args.subgraph_name)

    print("Computing closeness centrality...")
    write_scores("closeness.txt", compute_closeness(graph))
    print("  closeness.txt written.")

    print("Computing betweenness centrality...")
    write_scores("betweenness.txt", compute_betweenness(graph))
    print("  betweenness.txt written.")

    print("Computing eigenvector centrality...")
    write_scores("eigenvector.txt", compute_eigenvector(graph))
    print("  eigenvector.txt written.")

    print("Computing biased PageRank...")
    write_scores("pagerank.txt", compute_pagerank(graph))
    print("  pagerank.txt written.")

    print("All centrality computations complete.")


if __name__ == "__main__":
    main()