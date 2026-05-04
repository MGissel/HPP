import networkx as nx
from time import time
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

g = tree(2, 4)
offsets, neighbors_flat = graph_to_csr(g)
print(offsets, neighbors_flat)