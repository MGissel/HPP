
"""Simple MPI-based breadth-first search for a balanced tree graph.

The graph is created on every MPI rank, but the BFS frontier is processed in a
distributed way: each rank handles a slice of the current frontier, sends the
newly discovered vertices back to rank 0, and rank 0 builds the next frontier.

This keeps the example easy to follow while still using a distributed-memory
architecture that can run with multiple processors.

!!!!!!!!!!!!!!!!!!!!
USAGE INSTRUCTIONS:

Dependencies:
- OpenMPI: Install with `sudo apt install libopenmpi-dev openmpi-bin`
- mpi4py: Install with `pip install mpi4py`

Run with:
mpirun -n <num_processes> python C_dist_mem.py

example using 4 processes:
mpirun -n 4 python C_dist_mem.py

"""

import networkx as nx
import numpy as np
from mpi4py import MPI

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

class DistributedMemoryParallelBFS(Graph):
    def __init__(self, r, h):
        super().__init__(r, h)

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    # Breadth-First Search (BFS) algorithm
    def parallel_bfs(self):
        dist = np.full(self.n, -1, dtype=np.int32)
        frontier = [self.source] if self.rank == 0 else None
        level = 0

        if self.rank == 0:
            dist[self.source] = 0

        while True:
            frontier = self.comm.bcast(frontier, root=0)
            dist = self.comm.bcast(dist, root=0)

            if not frontier:
                break

            local_candidates = []

            for index, u in enumerate(frontier):
                if index % self.size != self.rank:
                    continue

                for v in self.adj[u]:
                    if dist[v] == -1:
                        local_candidates.append(v)

            gathered_candidates = self.comm.gather(local_candidates, root=0)

            if self.rank == 0:
                next_frontier = []
                seen = set()

                for candidate_list in gathered_candidates:
                    for v in candidate_list:
                        if v not in seen and dist[v] == -1:
                            seen.add(v)
                            dist[v] = level + 1
                            next_frontier.append(v)

                frontier = next_frontier
                level += 1
            else:
                frontier = None

        return dist if self.rank == 0 else None

def iterator(graph, iterations):
    times = []
    dist = None

    for _ in range(iterations):
        graph.comm.Barrier()
        time_start = MPI.Wtime() if graph.rank == 0 else None

        dist = graph.parallel_bfs()

        graph.comm.Barrier()
        time_end = MPI.Wtime() if graph.rank == 0 else None

        if graph.rank == 0:
            times.append(time_end - time_start)

    if graph.rank == 0:
        return sum(times) / len(times), dist, graph.n

    return None, None, graph.n


if __name__ == "__main__":

    r = 2 # branching factor (childs per node)
    h = 20 # height of the tree
    graph = DistributedMemoryParallelBFS(r, h)
    iterations = 1

    avg_time, dist, n = iterator(graph, iterations)

    if graph.rank == 0:
        print(f"MPI ranks: {graph.size}")
        print(f"Graph with {n} nodes (r={r}, h={h})")
        print(f"Average time of {iterations} iterations: {avg_time:.6f} seconds")

    # print("Node  Distance")
    # for u in range(n):
    #    print(f"  {u:2}     {dist[u]}")