from numba import jit, prange
import numba as nb
import networkx as nx
import time
import numpy as np


def tree(x, y):
    G = nx.balanced_tree(r=x, h=y)
    return G


def graph_to_csr(G):
    """Convert NetworkX graph to CSR (Compressed Sparse Row) format for Numba."""
    n = G.number_of_nodes()
    adj_list = [list(G.neighbors(v)) for v in range(n)]
    
    # Build CSR arrays
    offsets = np.zeros(n + 1, dtype=np.int32)
    for i, neighbors in enumerate(adj_list):
        offsets[i + 1] = offsets[i] + len(neighbors)
    
    neighbors_flat = np.array(
        [nb for nbrs in adj_list for nb in nbrs], dtype=np.int32
    )
    return offsets, neighbors_flat


@jit(nopython=True, parallel=True)
def parallel_BFS(offsets, neighbors_flat, source, n):
    """
    Parallel BFS using Numba JIT with prange.
    Uses frontier-based level-synchronous BFS.
    """
    distance = np.full(n, -1, dtype=np.int32)   # -1 = unvisited
    distance[source] = 0

    # Frontier as a boolean array for this level
    current_frontier = np.zeros(n, dtype=nb.boolean)
    current_frontier[source] = True

    current_level = 0

    while True:
        # Collect nodes in the current frontier
        frontier_nodes = np.where(current_frontier)[0]
        if len(frontier_nodes) == 0:
            break

        next_frontier = np.zeros(n, dtype=nb.boolean)

        # Process all frontier nodes in parallel
        for i in prange(len(frontier_nodes)):
            node = frontier_nodes[i]
            start = offsets[node]
            end = offsets[node + 1]

            for j in range(start, end):
                neighbor = neighbors_flat[j]
                if distance[neighbor] == -1:  # Unvisited
                    distance[neighbor] = current_level + 1
                    next_frontier[neighbor] = True

        current_frontier = next_frontier
        current_level += 1

    return distance


if __name__ == "__main__":
    print("Generating tree graph...")
    G = tree(2, 4)
    n = G.number_of_nodes()
    source = 0

    offsets, neighbors_flat = graph_to_csr(G)

    # Warm up JIT (first call compiles)
    print("Warming up JIT compiler...")
    _ = parallel_BFS(offsets, neighbors_flat, source, n)

    time_arr = np.zeros(1000)

    for i in range(1000):
    # Timed run
        time_start = time.time()
        distance = parallel_BFS(offsets, neighbors_flat, source, n)
        time_end = time.time()
        time_arr[i] = time_end - time_start


    print(f'Average time: {time_arr.mean()}')

