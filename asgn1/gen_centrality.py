import argparse
import math
import sys
from collections import deque
from pathlib import Path

import networkx as nx


DAMPING_FACTOR = 0.8
MAX_ITERATIONS = 100
PAGERANK_BIAS_DIVISOR = 3
CONVERGENCE_TOLERANCE = 1.0e-12


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("subgraph_name")
    return parser.parse_args()


def ensure_output_directory():
    Path("centralities").mkdir(exist_ok=True)


# The edge list is directed, so reverse pairs collapse into one undirected
# edge; self loops are dropped as they mean nothing here.
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


# Plain lists are a lot faster to walk than the NetworkX neighbour views,
# and we do one search per node below.
def build_adjacency(graph):
    return {node: list(graph.neighbors(node)) for node in graph.nodes()}


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


# Same as above but also records how many shortest paths reach each node,
# their parents on the shortest path DAG, and the visit order.
def bfs_shortest_path_dag(adjacency, source):
    dist = {source: 0}
    num_paths = {source: 1}
    predecessors = {source: []}
    order = []
    queue = deque([source])

    while queue:
        u = queue.popleft()
        order.append(u)
        for v in adjacency[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                num_paths[v] = num_paths[u]
                predecessors[v] = [u]
                queue.append(v)
            elif dist[v] == dist[u] + 1:
                num_paths[v] += num_paths[u]
                predecessors[v].append(u)

    return num_paths, predecessors, order


# C_i = (n - 1) / sum_j d_ij.  The graph is not connected, so pairs in
# different components simply do not contribute to the sum.
def compute_closeness(graph):
    adjacency = build_adjacency(graph)
    n = graph.number_of_nodes()
    closeness = {}

    for node in adjacency:
        total_distance = sum(bfs_distances(adjacency, node).values())
        if total_distance > 0:
            closeness[node] = (n - 1) / total_distance
        else:
            closeness[node] = 0.0

    return closeness


# Brandes' algorithm.  Every unordered pair is walked once from each end,
# so the accumulated sum is halved before the 2/((n-1)(n-2)) factor.
def compute_betweenness(graph):
    adjacency = build_adjacency(graph)
    n = graph.number_of_nodes()
    betweenness = {node: 0.0 for node in adjacency}

    for source in adjacency:
        num_paths, predecessors, order = bfs_shortest_path_dag(adjacency, source)

        delta = {node: 0.0 for node in order}
        for w in reversed(order):
            for pred in predecessors[w]:
                delta[pred] += (num_paths[pred] / num_paths[w]) * (1.0 + delta[w])
            if w != source:
                betweenness[w] += delta[w]

    pair_count = (n - 1) * (n - 2)
    if pair_count > 0:
        for node in betweenness:
            betweenness[node] = (2.0 / pair_count) * (betweenness[node] / 2.0)

    return betweenness


# Power iteration for x_i = (1/lambda) * sum_j A_ij x_j.  The assignment
# fixes no scale, so the result is normalised to unit Euclidean length.
def compute_eigenvector(graph):
    adjacency = build_adjacency(graph)
    n = len(adjacency)
    if n == 0:
        return {}

    x = {node: 1.0 / n for node in adjacency}

    for _ in range(MAX_ITERATIONS):
        x_next = {
            node: sum(x[neighbour] for neighbour in neighbours)
            for node, neighbours in adjacency.items()
        }

        norm = math.sqrt(sum(value * value for value in x_next.values()))
        if norm == 0:
            break
        for node in x_next:
            x_next[node] /= norm

        change = sum(abs(x_next[node] - x[node]) for node in adjacency)
        x = x_next
        if change < n * CONVERGENCE_TOLERANCE:
            break

    return x


# All of the teleport mass goes to the nodes whose id divides by 3, shared
# equally.  Falls back to uniform if no node qualifies.
def build_preference_vector(nodes):
    biased = [node for node in nodes if node % PAGERANK_BIAS_DIVISOR == 0]
    if not biased:
        return {node: 1.0 / len(nodes) for node in nodes}

    weight = 1.0 / len(biased)
    return {node: (weight if node % PAGERANK_BIAS_DIVISOR == 0 else 0.0)
            for node in nodes}


# Biased PageRank by power iteration.  Nothing loaded from an edge list is
# dangling, so no rank leaks away.
def compute_pagerank(graph):
    adjacency = build_adjacency(graph)
    nodes = list(adjacency)
    n = len(nodes)
    if n == 0:
        return {}

    preference = build_preference_vector(nodes)
    degree = {node: len(neighbours) for node, neighbours in adjacency.items()}
    rank = {node: 1.0 / n for node in nodes}

    for _ in range(MAX_ITERATIONS):
        rank_next = {}
        for node, neighbours in adjacency.items():
            incoming = sum(rank[nbr] / degree[nbr] for nbr in neighbours)
            rank_next[node] = (
                DAMPING_FACTOR * incoming
                + (1.0 - DAMPING_FACTOR) * preference[node]
            )

        change = sum(abs(rank_next[node] - rank[node]) for node in nodes)
        rank = rank_next
        if change < n * CONVERGENCE_TOLERANCE:
            break

    return rank


# Highest score first, ties broken by node id so the file is reproducible.
def write_scores(filename, scores):
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    with open(Path("centralities") / filename, "w") as f:
        for node, score in ordered:
            f.write(f"{node} {score:.6f}\n")


def main():
    args = parse_args()
    ensure_output_directory()

    graph = load_subgraph(args.subgraph_name)

    write_scores("closeness.txt", compute_closeness(graph))
    write_scores("betweenness.txt", compute_betweenness(graph))
    write_scores("eigenvector.txt", compute_eigenvector(graph))
    write_scores("pagerank.txt", compute_pagerank(graph))


if __name__ == "__main__":
    main()
