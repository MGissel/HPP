"""
Parallel shared memory.
using numba vectorize to check unvisited neighbours in parallel.
"""

import numpy as np
import networkx as nx
from numba import vectorize, int32
import time

G = nx.balanced_tree(r=2, h=7)
n = G.number_of_nodes()

# Build adjacency list from edges
adj = {u: [] for u in range(n)}
for u, v in G.edges():
    adj[u].append(v)
    adj[v].append(u) 

# Numba vectorized unvisited check
@vectorize([int32(int32, int32)], nopython=True)
def is_unvisited(neighbour, dist_of_neighbour):
    return 1 if dist_of_neighbour == -1 else 0

# Breadth-First Search (BFS) algorithm 
def parallel_bfs(adj, source, n):
    dist = np.full(n, -1, dtype=np.int32)
    dist[source] = 0
    frontier = [source]

    while frontier:
        next_frontier = []

        for u in frontier:
            nbrs = np.array(adj[u], dtype=np.int32)
            mask = is_unvisited(nbrs, dist[nbrs])

            for i, v in enumerate(nbrs):
                if mask[i] and dist[v] == -1:
                    dist[v] = dist[u] + 1
                    next_frontier.append(v)

        frontier = next_frontier

    return dist

def iterator(iterations):
    times = []
    for _ in range(iterations):
        time_start = time.time()
        dist = parallel_bfs(adj, source=0, n=n)
        time_end = time.time()

        times.append(time_end - time_start)

    return sum(times) / len(times), dist
    
if __name__ == "__main__":
    iterations = 100
    avg_time, dist = iterator(iterations)

    print(f"Average time of {iterations} iterations: {avg_time:.4f} seconds")

    # print("Node  Distance")
    # for u in range(n):
    #    print(f"  {u:2}     {dist[u]}")