from wrappers import timer_func
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
from functools import lru_cache

def bfs():
    pass

def generate_graph(r, h):
    return nx.balanced_tree(r, h)

@lru_cache(maxsize=None)
def generate_graph_fast(r, h):
    """Returns adjacency list as dict. Node i's children: r*i+1 ... r*i+r"""
    n = (r ** (h + 1) - 1) // (r - 1)  # total nodes
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for k in range(1, r + 1):
            child = r * i + k
            if child < n:
                adj[i].append(child)
                adj[child].append(i)
    return adj

def loop_graphs(r, h_max):
    results = []
    for i in range(1, h_max + 1):
        h = i # height of the tree
        print(f"Creating graph, with {(2**i)-1} nodes...")
        G = generate_graph_fast(r, h)
        print(f"Running BFS for tree height {i}...")
        total = 0
        for i in range(1):
            _, time = bfs(G, start=list(G.keys())[0])
            total += time
        avg_time = total / 1
        results.append((avg_time))
    return results

@timer_func
def bfs(graph, start):
    distance = {v: float('inf') for v in graph} # Initialize distances to infinity
    visited = {v: False for v in graph} # Initialize visited status to False

    distance[start] = 0 # Distance to the start node is 0
    visited[start] = True # Mark the start node as visited

    Q = deque([start]) # Initialize the queue with the start node
    while Q: # While the queue is not empty
        u = Q.popleft() # Dequeue a node from the queue
        for w in graph[u]: # Iterate through the neighbors of the dequeued node
            if not visited[w]: # If the neighbor has not been visited
                visited[w] = True # Mark the neighbor as visited
                distance[w] = distance[u] + 1 # Update the distance to the neighbor
                Q.append(w) # Enqueue the neighbor to the queue
    
    return distance

import numpy as np

def plot_results(results, i):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    heights = np.array(range(1, len(results) + 1))
    times = np.array(results)

    ax.plot(heights, times, marker='o', label='BFS Time')

    # O(n) reference: n = 2^h - 1, scaled to first data point
    ns = 2**heights - 1
    ref = ns * (times[0] / ns[0])
    ax.plot(heights, ref, "k--", linewidth=0.8, label="O(n) reference")

    ax.set_xlabel('Height of the Tree')
    ax.set_ylabel('Average Time (seconds)')
    ax.set_title('Performance of BFS on Balanced Trees')
    ax.legend()
    ax.grid()

    if i == 0:
        ax.set_yscale('log')
        plt.savefig('bfs_time_log.png')
    else:
        plt.savefig('bfs_time.png')

    plt.show()

if __name__ == "__main__":
    results = loop_graphs(r=2, h_max=24)
    #results = [1.6927719116210937e-06, 3.0469894409179686e-06, 7.328987121582031e-06, 1.1496543884277343e-05, 3.031015396118164e-05, 4.9505233764648434e-05, 0.00011777877807617188, 0.00024958610534667967, 0.0004723238945007324, 0.000959162712097168, 0.002104747295379639, 0.004062471389770508, 0.008014488220214843, 0.017129714488983153, 0.0347056245803833, 0.07027392625808716, 0.14099830150604248, 0.29211068391799927, 0.608132803440094, 1.2748368763923645]
    print(results)
    plot_results(results, 0)