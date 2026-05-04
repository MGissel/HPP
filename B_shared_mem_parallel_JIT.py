from numba import jit
import networkx as nx
import time
import numpy as np


def tree(x, y):
    return nx.balanced_tree(r=x, h=y)


def graph_to_csr(G):
    n = G.number_of_nodes()
    adj_list = [list(G.neighbors(v)) for v in range(n)]

    offsets = np.zeros(n + 1, dtype=np.int32)
    for i, neighbors in enumerate(adj_list):
        offsets[i + 1] = offsets[i] + len(neighbors)

    neighbors_flat = np.array(
        [nb for nbrs in adj_list for nb in nbrs], dtype=np.int32
    )
    return offsets, neighbors_flat


@jit(nopython=True)
def BFS(offsets, neighbors_flat, source, n):
    distance = np.full(n, -1, dtype=np.int32)
    distance[source] = 0

    frontier = np.zeros(n, dtype=np.bool_)
    frontier[source] = True

    current_level = 0

    while True:
        frontier_nodes = np.where(frontier)[0]
        if len(frontier_nodes) == 0:
            break

        next_frontier = np.zeros(n, dtype=np.bool_)

        for i in range(len(frontier_nodes)):        # plain for loop
            node = frontier_nodes[i]
            for j in range(offsets[node], offsets[node + 1]):
                neighbor = neighbors_flat[j]
                if distance[neighbor] == -1:
                    distance[neighbor] = current_level + 1
                    next_frontier[neighbor] = True

        frontier = next_frontier
        current_level += 1

    return distance


if __name__ == "__main__":
    G = tree(2, 4)
    n = G.number_of_nodes()
    source = 0

    offsets, neighbors_flat = graph_to_csr(G)

    print("Warming up JIT...")
    _ = BFS(offsets, neighbors_flat, source, n)


    time_avg = np.zeros(1000)

    for i in range(1000):
        time_start = time.time()
        distance = BFS(offsets, neighbors_flat, source, n)
        time_end = time.time()
        time_avg[i] = time_end - time_start

    print(f"\nAverage time (excluding warmup): {np.mean(time_avg):.6f} seconds")