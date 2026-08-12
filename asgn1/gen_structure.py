"""Assignment 1 — Structural metrics for the cit-HepTh subgraph.

Computes and prints network size, degree distribution, path metrics,
component statistics, and clustering coefficients.  Generates
corresponding distribution plots in the plots/ directory.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering.
import matplotlib.pyplot as plt
import networkx as nx


# ---------------------------------------------------------------------------
# Argument parsing and setup
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse the subgraph filename from the command line."""

    parser = argparse.ArgumentParser()
    parser.add_argument("subgraph_name")
    return parser.parse_args()


def ensure_output_directories() -> None:
    """Create the output directories expected by the assignment."""

    Path("subgraphs").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)


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
                # Skip self-loops.
                if u != v:
                    graph.add_edge(u, v)

    return graph


# ---------------------------------------------------------------------------
# 1. Network size
# ---------------------------------------------------------------------------

def print_network_size(graph: nx.Graph) -> None:
    """Print the node and edge counts."""

    print(f"Number of nodes:{graph.number_of_nodes()}")
    print(f"Number of edges:{graph.number_of_edges()}")


# ---------------------------------------------------------------------------
# 2. Degree metrics
# ---------------------------------------------------------------------------

def print_degree_metrics(graph: nx.Graph, subgraph_name: str) -> None:
    """Print highest-degree nodes and generate the degree distribution plot."""

    # Find max degree.
    degrees = dict(graph.degree())
    max_deg = max(degrees.values())

    # Collect all nodes with the maximum degree, sorted for determinism.
    max_nodes = sorted([n for n, d in degrees.items() if d == max_deg])
    print(f"Node id(s) with highest degree:{','.join(str(n) for n in max_nodes)}")

    # Degree distribution plot.
    deg_values = list(degrees.values())
    deg_count = Counter(deg_values)
    xs = sorted(deg_count.keys())
    ys = [deg_count[x] for x in xs]

    plt.figure()
    plt.bar(xs, ys, width=1.0, edgecolor="black", linewidth=0.3)
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.title("Degree Distribution")
    plt.tight_layout()
    plt.savefig(Path("plots") / f"deg_dist_{subgraph_name}.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 3. Path metrics (diameter estimates and shortest-path distribution)
# ---------------------------------------------------------------------------

def _bfs_eccentricity(graph: nx.Graph, source: int) -> tuple[int, dict[int, int]]:
    """Return (eccentricity, {node: distance}) via BFS from *source*.

    Only visits nodes in the same connected component as *source*.
    """

    dist: dict[int, int] = {source: 0}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in graph.neighbors(u):
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    ecc = max(dist.values()) if dist else 0
    return ecc, dist


def print_path_metrics(graph: nx.Graph, subgraph_name: str) -> None:
    """Compute approximate diameters and produce the shortest-path plot."""

    nodes = list(graph.nodes())
    sample_sizes = [10, 100, 1000]
    diameters: list[int] = []

    for n_samples in sample_sizes:
        sample = random.sample(nodes, min(n_samples, len(nodes)))
        max_ecc = 0
        for s in sample:
            ecc, _ = _bfs_eccentricity(graph, s)
            if ecc > max_ecc:
                max_ecc = ecc
        diameters.append(max_ecc)
        print(f"Approximate full diameter by sampling {n_samples} nodes:{max_ecc}")

    # Mean and variance of the three estimates.
    mean_d = sum(diameters) / len(diameters)
    var_d = sum((d - mean_d) ** 2 for d in diameters) / len(diameters)
    print(f"Approximate full diameter (mean and variance):{round(mean_d, 4)},{round(var_d, 4)}")

    # Shortest path length distribution — sample BFS from a set of nodes.
    sp_sample_size = min(1000, len(nodes))
    sp_sample = random.sample(nodes, sp_sample_size)
    path_len_counter: Counter[int] = Counter()

    for s in sp_sample:
        _, dists = _bfs_eccentricity(graph, s)
        for d in dists.values():
            if d > 0:
                path_len_counter[d] += 1

    xs = sorted(path_len_counter.keys())
    ys = [path_len_counter[x] for x in xs]

    plt.figure()
    plt.bar(xs, ys, width=1.0, edgecolor="black", linewidth=0.3)
    plt.xlabel("Shortest Path Length")
    plt.ylabel("Count")
    plt.title("Shortest Path Length Distribution")
    plt.tight_layout()
    plt.savefig(Path("plots") / f"shortest_path_{subgraph_name}.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 4. Component metrics
# ---------------------------------------------------------------------------

def print_component_metrics(graph: nx.Graph, subgraph_name: str) -> None:
    """Print component statistics and generate the component size plot."""

    components = list(nx.connected_components(graph))
    comp_sizes = sorted([len(c) for c in components], reverse=True)
    n = graph.number_of_nodes()

    # Largest connected component fraction.
    lcc_frac = round(comp_sizes[0] / n, 4)
    print(f"Fraction of nodes in largest connected component:{lcc_frac}")

    # Edge bridges.
    bridges = list(nx.bridges(graph))
    print(f"Number of edge bridges:{len(bridges)}")

    # Articulation points.
    aps = list(nx.articulation_points(graph))
    print(f"Number of articulation points:{len(aps)}")

    # Connected component size distribution plot.
    size_count = Counter(comp_sizes)
    xs = sorted(size_count.keys())
    ys = [size_count[x] for x in xs]

    plt.figure()
    plt.bar(xs, ys, width=1.0, edgecolor="black", linewidth=0.3)
    plt.xlabel("Component Size")
    plt.ylabel("Count")
    plt.title("Connected Component Size Distribution")
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(Path("plots") / f"connected_comp_{subgraph_name}.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 5. Clustering metrics
# ---------------------------------------------------------------------------

def print_clustering_metrics(graph: nx.Graph, subgraph_name: str) -> None:
    """Print clustering statistics and generate the clustering coeff plot."""

    # Average clustering coefficient.
    avg_cc = round(nx.average_clustering(graph), 4)
    print(f"Average clustering coefficient:{avg_cc}")

    # Number of triads (triangles).
    triangles = nx.triangles(graph)
    num_triads = sum(triangles.values()) // 3
    print(f"Number of triads:{num_triads}")

    # Random node: clustering coefficient and triad participation.
    nodes = list(graph.nodes())
    rand_node = random.choice(nodes)
    rand_cc = round(nx.clustering(graph, rand_node), 4)
    rand_triads = triangles[rand_node]

    print(f"Clustering coefficient of random node {rand_node}:{rand_cc}")
    print(f"Number of triads random node {rand_node} participates:{rand_triads}")

    # Edges participating in at least one triad (triangle).
    edges_in_triads = 0
    for u, v in graph.edges():
        # An edge (u,v) participates in a triangle if u and v share a neighbor.
        if set(graph.neighbors(u)) & set(graph.neighbors(v)):
            edges_in_triads += 1
    print(f"Number of edges that participate in at least one triad:{edges_in_triads}")

    # Clustering coefficient distribution plot.
    cc_values = list(nx.clustering(graph).values())

    plt.figure()
    plt.hist(cc_values, bins=50, edgecolor="black", linewidth=0.3)
    plt.xlabel("Clustering Coefficient")
    plt.ylabel("Count")
    plt.title("Clustering Coefficient Distribution")
    plt.tight_layout()
    plt.savefig(Path("plots") / f"clustering_coeff_{subgraph_name}.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    ensure_output_directories()

    graph = load_subgraph(args.subgraph_name)
    print_network_size(graph)
    print_degree_metrics(graph, args.subgraph_name)
    print_path_metrics(graph, args.subgraph_name)
    print_component_metrics(graph, args.subgraph_name)
    print_clustering_metrics(graph, args.subgraph_name)


if __name__ == "__main__":
    main()