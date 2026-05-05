from mpi4py import MPI
import numpy as np
import networkx as nx
import time
import argparse

argparse = argparse.ArgumentParser(description="MPI BFS")
argparse.add_argument("--r", type=int, default=2, help="Branching factor of the tree")
argparse.add_argument("--h", type=int, default=5, help="Height of the tree")
argparse.add_argument("--runs", type=int, default=1, help="Number of BFS runs to average over")
args = argparse.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # What process am I?
size = comm.Get_size()  # How many processes are there?


# All processes build the same graph
def make_graph(r, h):
    return nx.balanced_tree(r, h)

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

# Partition: which vertices to I own
def my_range(n):
    chunk = (n + size - 1) // size 
    start = rank * chunk
    end = min(start + chunk, n)
    return start, end

# MPI BFS
def bfs_mpi(offsets, neighbors_flat, n, source=0):
    local_start, local_end = my_range(n)
    distance = np.full(n, -1, dtype=np.int32)
    distance[source] = 0

    frontier = np.zeros(n, dtype=np.bool_)
    frontier[source] = True

    level = 0
    while True:
        # Each process expands only its own verrtices
        local_next = np.zeros(n, dtype=np.bool_)

        for v in range(local_start, local_end):
            if not frontier[v]:
                continue
            for j in range(offsets[v], offsets[v + 1]):
                nb = neighbors_flat[j]
                if distance[nb] == -1:
                    distance[nb] = level + 1
                    local_next[nb] = True
        
        # sync: combine every process's discoveries
        global_next = np.zeros(n, dtype=np.bool_)
        comm.Allreduce(local_next, global_next, op=MPI.LOR)

        global_dist = np.zeros(n, dtype=np.int32)
        comm.Allreduce(distance, global_dist, op=MPI.MAX)
        distance = global_dist

        if not global_next.any():
            break

        frontier = global_next
        level += 1
    return distance

def iterator(graph, runs):
    offsets, neighbors_flat = graph_to_csr(graph)
    n = graph.number_of_nodes()

    # Warmup
    bfs_mpi(offsets, neighbors_flat, n)
    
    
    comm.Barrier()
    t0 = time.perf_counter()
    for _ in range(runs):
        dist = bfs_mpi(offsets, neighbors_flat, n)
    comm.Barrier()
    t1 = time.perf_counter()
    
    avg_ms = (t1 - t0) / runs
    return avg_ms
        
if __name__ == "__main__":
    G = make_graph(r=args.r, h=args.h)
    avg_time = iterator(G, runs=args.runs)
    if rank == 0:
        print(f"{avg_time:.8f}")