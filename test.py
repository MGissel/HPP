import numpy as np
import timeit

def bfs_vectorized(offsets, neighbors_flat, start, n):
    n_bytes = (n + 7) // 8
    visited  = np.zeros(n_bytes, dtype=np.uint8)
    frontier = np.zeros(n_bytes, dtype=np.uint8)

    def set_bit(vec, node):
        vec[node // 8] |= np.uint8(1 << (7 - node % 8))

    set_bit(visited, start)
    set_bit(frontier, start)

    level = 0
    while frontier.any():
        # all nodes in frontier
        frontier_nodes = np.where(np.unpackbits(frontier, count=n))[0]

        # gather ALL neighbors of ALL frontier nodes at once
        neighbor_slices = [neighbors_flat[offsets[v]:offsets[v+1]] 
                           for v in frontier_nodes]
        if not neighbor_slices:
            break
        all_neighbors = np.concatenate(neighbor_slices)
        all_neighbors = np.unique(all_neighbors)

        # which ones are unvisited?
        visited_bits = np.unpackbits(visited, count=n)
        new_nodes = all_neighbors[visited_bits[all_neighbors] == 0]

        # mark them all at once
        next_frontier = np.zeros(n_bytes, dtype=np.uint8)
        if new_nodes.size > 0:
            byte_idx = new_nodes // 8
            bit_mask = np.uint8(1) << (7 - new_nodes % 8).astype(np.uint8)
            np.bitwise_or.at(visited,       byte_idx, bit_mask)
            np.bitwise_or.at(next_frontier, byte_idx, bit_mask)

        frontier = next_frontier
        level += 1

    return visited

import networkx as nx

def tree(x, y):
    return nx.balanced_tree(r=x, h=y)

def graph_to_csr(G):
    n = G.number_of_nodes()
    adj_list = [list(G.neighbors(v)) for v in range(n)]
    offsets = np.zeros(n + 1, dtype=np.int32)
    for i, neighbors in enumerate(adj_list):
        offsets[i + 1] = offsets[i] + len(neighbors)
    neighbors_flat = np.array([nb for nbrs in adj_list for nb in nbrs], dtype=np.int32)
    return offsets, neighbors_flat

g = tree(3, 5)
offsets, neighbors_flat = graph_to_csr(g)
n = g.number_of_nodes()

visited = bfs_vectorized(offsets, neighbors_flat, start=0, n=n)
print(np.unpackbits(visited, count=n))  # all 1s if fully connected

runs = 1000
elapsed = timeit.timeit(lambda: bfs_vectorized(offsets, neighbors_flat, start=0, n=n), number=runs)
print(f"Average time over {runs} runs: {elapsed / runs:.6f} seconds")