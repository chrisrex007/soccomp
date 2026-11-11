import argparse
import random
import sys
from collections import Counter, deque
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


RANDOM_SEED = 42
DIAMETER_SAMPLE_SIZES = [10, 100, 1000]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("subgraph_name")
    return parser.parse_args()


def ensure_output_directories():
    Path("plots").mkdir(exist_ok=True)


# The edge list is directed, so reverse pairs collapse into one undirected edge; self loops are dropped as they mean nothing here.
def load_subgraph(subgraph_name):
    filepath = Path("subgraphs") / subgraph_name
    if not filepath.exists():
        sys.exit(f"error: edge list not found at {filepath}")

    graph = nx.Graph()
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


# Plain lists are a lot faster to walk than the NetworkX neighbour views, and we do one search per node below.
def build_adjacency(graph):
    return {node: list(graph.neighbors(node)) for node in graph.nodes()}


# Only reaches nodes in the same component as source, so the result also identifies that component.
def bfs_distances(adjacency, source):
    dist = {source: 0}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in adjacency[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def print_network_size(graph):
    print(f"Number of nodes:{graph.number_of_nodes()}")
    print(f"Number of edges:{graph.number_of_edges()}")


# Several nodes can tie at the maximum, so report all of them.  The plot is log-log because the distribution is heavy tailed.
def print_degree_metrics(graph, subgraph_name):
    degrees = dict(graph.degree())
    max_degree = max(degrees.values())

    top_nodes = sorted(node for node, deg in degrees.items() if deg == max_degree)
    print(f"Node id(s) with highest degree:{','.join(str(n) for n in top_nodes)}")

    degree_counts = Counter(degrees.values())
    xs = sorted(degree_counts)
    ys = [degree_counts[x] for x in xs]

    plt.figure()
    plt.scatter(xs, ys, s=12)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Degree")
    plt.ylabel("Number of nodes")
    plt.title("Degree Distribution")
    plt.grid(True, which="both", linewidth=0.3, alpha=0.5)
    plt.tight_layout()
    plt.savefig(Path("plots") / f"deg_dist_{subgraph_name}.png", dpi=150)
    plt.close()


# Diameters are sampled, but the length distribution is exact: one search per node sees every connected pair twice, so the counts are halved.
def print_path_metrics(adjacency, subgraph_name):
    nodes = list(adjacency)

    diameters = []
    for sample_size in DIAMETER_SAMPLE_SIZES:
        sample = random.sample(nodes, min(sample_size, len(nodes)))
        diameter = max(max(bfs_distances(adjacency, s).values()) for s in sample)
        diameters.append(diameter)
        print(f"Approximate full diameter by sampling {sample_size} nodes:{diameter}")

    mean_diameter = sum(diameters) / len(diameters)
    variance = sum((d - mean_diameter) ** 2 for d in diameters) / len(diameters)
    print(
        "Approximate full diameter (mean and variance):"
        f"{mean_diameter:.4f}, {variance:.4f}"
    )

    lc = Counter()
    for source in nodes:
        for d in bfs_distances(adjacency, source).values():
            if d > 0:
                lc[d] += 1

    xs = sorted(lc)
    ys = [lc[x] // 2 for x in xs]

    plt.figure()
    plt.bar(xs, ys, width=0.8, edgecolor="black", linewidth=0.3)
    plt.xlabel("Shortest path length")
    plt.ylabel("Number of node pairs")
    plt.title("Shortest Path Length Distribution")
    plt.tight_layout()
    plt.savefig(Path("plots") / f"shortest_path_{subgraph_name}.png", dpi=150)
    plt.close()


# Sizes run from 2 up to the giant component, so both axes are logarithmic and the counts are drawn as points rather than bars.
def print_component_metrics(graph, subgraph_name):
    component_sizes = [len(c) for c in nx.connected_components(graph)]

    lcc_fraction = max(component_sizes) / graph.number_of_nodes()
    print(f"Fraction of nodes in largest connected component:{lcc_fraction:.4f}")
    print(f"Number of edge bridges:{len(list(nx.bridges(graph)))}")
    print(f"Number of articulation points:{len(list(nx.articulation_points(graph)))}")

    size_counts = Counter(component_sizes)
    xs = sorted(size_counts)
    ys = [size_counts[x] for x in xs]

    plt.figure()
    plt.scatter(xs, ys, s=25)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Component size")
    plt.ylabel("Number of components")
    plt.title("Connected Component Size Distribution")
    plt.grid(True, which="both", linewidth=0.3, alpha=0.5)
    plt.tight_layout()
    plt.savefig(Path("plots") / f"connected_comp_{subgraph_name}.png", dpi=150)
    plt.close()


# Each triad is counted once by each of its three members, and an edge sits in a triad exactly when its two endpoints share a neighbour.
def print_clustering_metrics(graph, adjacency, subgraph_name):
    # nx.clustering would repeat the triangle count that nx.triangles already does, so C_i = 2*T_i / (d_i*(d_i-1)) is derived from it instead.
    triangles = nx.triangles(graph)
    clustering = {}
    for node, nbrs in adjacency.items():
        degree = len(nbrs)
        clustering[node] = (
            2.0 * triangles[node] / (degree * (degree - 1)) if degree > 1 else 0.0
        )

    avg_clustering = sum(clustering.values()) / len(adjacency)
    print(f"Average clustering coefficient:{avg_clustering:.4f}")
    print(f"Number of triads:{sum(triangles.values()) // 3}")

    random_node = random.choice(list(adjacency))
    print(
        f"Clustering coefficient of random node {random_node}:"
        f"{clustering[random_node]:.4f}"
    )
    print(
        f"Number of triads random node {random_node} participates:"
        f"{triangles[random_node]}"
    )

    neighbours = {node: set(nbrs) for node, nbrs in adjacency.items()}
    edges_in_triads = sum(
        1 for u, v in graph.edges() if neighbours[u] & neighbours[v]
    )
    print(f"Number of edges that participate in at least one triad:{edges_in_triads}")

    plt.figure()
    plt.hist(list(clustering.values()), bins=50, edgecolor="black", linewidth=0.3)
    plt.yscale("log")
    plt.xlabel("Clustering coefficient")
    plt.ylabel("Number of nodes")
    plt.title("Clustering Coefficient Distribution")
    plt.tight_layout()
    plt.savefig(Path("plots") / f"clustering_coeff_{subgraph_name}.png", dpi=150)
    plt.close()


def main():
    args = parse_args()
    random.seed(RANDOM_SEED)
    ensure_output_directories()

    graph = load_subgraph(args.subgraph_name)
    adjacency = build_adjacency(graph)

    print_network_size(graph)
    print_degree_metrics(graph, args.subgraph_name)
    print_path_metrics(adjacency, args.subgraph_name)
    print_component_metrics(graph, args.subgraph_name)
    print_clustering_metrics(graph, adjacency, args.subgraph_name)


if __name__ == "__main__":
    main()
