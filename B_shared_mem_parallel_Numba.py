"""
Parallel shared memory.
using numba vectorize to check unvisited neighbours in parallel.
"""

import numpy as np
import networkx as nx
from numba import vectorize, int32
import time

class Graph:
    def __init__(self, r, h):
        self.G = nx.balanced_tree(r, h)
        self.source = 0
        self.n = self.G.number_of_nodes()
        self.adj = {u: [] for u in range(self.n)}
        self.adjacency_list()
  
    def adjacency_list(self):
        for u, v in self.G.edges():
            self.adj[u].append(v)
            self.adj[v].append(u)

class SharedMemoryParallelBFS(Graph):
    def __init__(self, r, h):
        super().__init__(r, h)
        
    @vectorize([int32(int32, int32)], nopython=True)
    def is_unvisited(neighbour, dist_of_neighbour):
        return 1 if dist_of_neighbour == -1 else 0

    # Breadth-First Search (BFS) algorithm 
    def parallel_bfs(self):
        dist = np.full(self.n, -1, dtype=np.int32)
        dist[self.source] = 0
        frontier = [self.source]

        while frontier:
            next_frontier = []

            for u in frontier:
                nbrs = np.array(self.adj[u], dtype=np.int32)
                mask = self.is_unvisited(nbrs, dist[nbrs])

                for i, v in enumerate(nbrs):
                    if mask[i] and dist[v] == -1:
                        dist[v] = dist[u] + 1
                        next_frontier.append(v)

            frontier = next_frontier

        return dist

# Helper function to run multiple iterations and calculate average time taken for BFS
def iterator(iterations):
    times = []
    for _ in range(iterations):
        time_start = time.time()
        dist = graph.parallel_bfs()
        time_end = time.time()

        times.append(time_end - time_start)

    return sum(times) / len(times), dist, n
    
if __name__ == "__main__":

    r = 2 # branching factor (childs per node)
    h = 4 # height of the tree
    graph = SharedMemoryParallelBFS(r, h)
    n = graph.n
    iterations = 5000

    avg_time, dist, n = iterator(iterations)

    print(f"Average time of {iterations} iterations: {avg_time:.6f} seconds")

    # print("Node  Distance")
    # for u in range(n):
    #    print(f"  {u:2}     {dist[u]}")