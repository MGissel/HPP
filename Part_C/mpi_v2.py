from mpi4py import MPI
import numpy as np
import networkx as nx
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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

def BFS_mpi(offsets, neighbors_flat, source, n):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    distance = np.full(n, -1, dtype=np.int32)
    distance[source] = 0

    frontier = np.zeros(n, dtype=np.bool_)
    frontier[source] = True

    current_level = 0

    while True:
        # --- Rank 0 splits and scatters frontier chunks ---
        if rank == 0:
            frontier_nodes = np.where(frontier)[0]
            chunks = np.array_split(frontier_nodes, size)
            max_len = max(len(c) for c in chunks)
            chunks_padded = np.array([
                np.pad(c, (0, max_len - len(c)), constant_values=-1)
                for c in chunks
            ], dtype=np.int32)
        else:
            max_len = None
            chunks_padded = None

        # Broadcast chunk size so all processes can allocate
        max_len = comm.bcast(max_len, root=0)
        done = comm.bcast(len(np.where(frontier)[0]) == 0 if rank == 0 else None, root=0)
        if done:
            break

        my_chunk = np.empty(max_len, dtype=np.int32)
        comm.Scatter(chunks_padded, my_chunk, root=0)
        my_nodes = my_chunk[my_chunk != -1]  # strip padding

        # --- Each process explores its nodes ---
        local_next = np.zeros(n, dtype=np.bool_)
        for node in my_nodes:
            neighbors = neighbors_flat[offsets[node]:offsets[node + 1]]
            unvisited = neighbors[distance[neighbors] == -1]
            local_next[unvisited] = True

        # --- Gather all local_next arrays to rank 0 ---
        all_local_next = None
        if rank == 0:
            all_local_next = np.empty((size, n), dtype=np.bool_)
        comm.Gather(local_next, all_local_next, root=0)

        # --- Rank 0 merges and broadcasts the new frontier ---
        if rank == 0:
            global_next = np.any(all_local_next, axis=0)
            new_nodes = global_next & (distance == -1)
            distance[new_nodes] = current_level + 1
            frontier = global_next & (distance == current_level + 1)

        # Broadcast updated frontier and distance back to all processes
        comm.Bcast(frontier, root=0)
        comm.Bcast(distance, root=0)

        current_level += 1

    return distance if rank == 0 else None

if __name__ == "__main__":
    r = 2
    h = 25
    iterations = 10

    # rank 0 builds the graph, then ALL processes call bcast
    if rank == 0:
        print(f"Creating graph with branching factor={r}, height={h}...")
        G = make_graph(r, h)
        offsets, neighbors_flat = graph_to_csr(G)
        n = G.number_of_nodes()
    else:
        offsets = None
        neighbors_flat = None
        n = None

    # every process must call bcast — not just rank 0
    offsets = comm.bcast(offsets, root=0)
    neighbors_flat = comm.bcast(neighbors_flat, root=0)
    n = comm.bcast(n, root=0)  # broadcast n so all processes know graph size

    source = 0

    # Warmup
    BFS_mpi(offsets, neighbors_flat, source, n)

    # Timed runs
    time_arr = np.zeros(iterations)
    for i in range(iterations):
        comm.Barrier()  # sync all processes before each timed run
        t0 = time.time()
        distance = BFS_mpi(offsets, neighbors_flat, source, n)
        comm.Barrier()  # sync before stopping the clock
        time_arr[i] = time.time() - t0

    if rank == 0:
        print(f"Avg time over {iterations} runs: {np.mean(time_arr):.8f} seconds")