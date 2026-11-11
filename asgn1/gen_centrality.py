import argparse
import math
import sys
from collections import deque
from pathlib import Path

import networkx as nx

DAMPING_FACTOR = 0.8
MAX_ITR = 100

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("subgraph_name")
    return parser.parse_args()


PG_BIAS_DIVISOR = 3
TOLERANCE = 1.0e-12

def ensure_output_directory():
    Path("centralities").mkdir(exist_ok=True)


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


# Same as above but also records how many shortest paths reach each node, their parents on the shortest path DAG, and the visit order.
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


# C_i = (n - 1) / sum_j d_ij.
def compute_closeness(adjacency):
    n = len(adjacency)
    closeness = {}

    for node in adjacency:
        total_distance = sum(bfs_distances(adjacency, node).values())
        closeness[node] = (n - 1) / total_distance

    return closeness

def compute_betweenness(adjacency):
    n = len(adjacency)
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
            betweenness[node] /= pair_count

    return betweenness


# Power iteration for x_i = (1/lambda) * sum_j A_ij x_j.
def compute_eigenvector(adjacency):
    n = len(adjacency)
    if n == 0:
        return {}

    x = {node: 1.0 / n for node in adjacency}

    for _ in range(MAX_ITR):
        x_next = {
            node: sum(x[neighbour] for neighbour in neighbours)
            for node, neighbours in adjacency.items()
        }

        norm = math.sqrt(sum(value * value for value in x_next.values()))
        for node in x_next:
            x_next[node] /= norm

        change = sum(abs(x_next[node] - x[node]) for node in adjacency)
        x = x_next
        if change < n * TOLERANCE:
            break

    return x


def build_preference_vector(nodes):
    biased = {node for node in nodes if node % PG_BIAS_DIVISOR == 0}
    if not biased:
        return {node: 1.0 / len(nodes) for node in nodes}

    weight = 1.0 / len(biased)
    return {node: (weight if node in biased else 0.0) for node in nodes}


# Biased PageRank by power iteration.
def compute_pagerank(adjacency):
    nodes = list(adjacency)
    n = len(nodes)
    if n == 0:
        return {}

    preference = build_preference_vector(nodes)
    degree = {node: len(neighbours) for node, neighbours in adjacency.items()}
    rank = {node: 1.0 / n for node in nodes}

    for _ in range(MAX_ITR):
        # Each node's share is the same for all of its edges, so it is divided out once per node rather than once per edge.
        contribution = {node: rank[node] / degree[node] for node in nodes}
        rank_next = {}
        for node, neighbours in adjacency.items():
            incoming = sum(contribution[nbr] for nbr in neighbours)
            rank_next[node] = (
                DAMPING_FACTOR * incoming
                + (1.0 - DAMPING_FACTOR) * preference[node]
            )

        change = sum(abs(rank_next[node] - rank[node]) for node in nodes)
        rank = rank_next
        if change < n * TOLERANCE:
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
    adjacency = build_adjacency(graph)

    write_scores("closeness.txt", compute_closeness(adjacency))
    write_scores("betweenness.txt", compute_betweenness(adjacency))
    write_scores("eigenvector.txt", compute_eigenvector(adjacency))
    write_scores("pagerank.txt", compute_pagerank(adjacency))


if __name__ == "__main__":
    main()
