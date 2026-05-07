import networkx as nx
from collections import deque
from wrappers import timer_func
import matplotlib.pyplot as plt

#-------------#
#-----BFS-----#
#-------------#
@timer_func
def BFS(graph, start):
    distance = {v: float('inf') for v in graph.nodes} # Initialize distances to infinity
    visited = {v: False for v in graph.nodes} # Initialize visited status to False

    distance[start] = 0 # Distance to the start node is 0
    visited[start] = True # Mark the start node as visited

    Q = deque([start]) # Initialize the queue with the start node
    while Q: # While the queue is not empty
        u = Q.popleft() # Dequeue a node from the queue
        for w in graph.neighbors(u): # Iterate through the neighbors of the dequeued node
            if not visited[w]: # If the neighbor has not been visited
                visited[w] = True # Mark the neighbor as visited
                distance[w] = distance[u] + 1 # Update the distance to the neighbor
                Q.append(w) # Enqueue the neighbor to the queue
    
    return distance

def iterator(G, iterations):
    total_time = 0
    for i in range(iterations):
        distance, elapsed = BFS(G, start=0)
        total_time += elapsed
        # print(f"Iteration {i+1}/{iterations} completed in {elapsed:.4f} seconds.")
    return total_time / iterations


def plot_results(results, i):
    plt.figure(figsize=(10, 6))
    # Support two formats:
    # 1) results is an iterable of (label, data) where data is list of (h, t)
    # 2) results is a single list of (h, t) pairs (plot single series)
    try:
        first = next(iter(results))
    except StopIteration:
        return

    if isinstance(first, tuple) and len(first) == 2 and isinstance(first[1], (list, tuple)):
        # multiple labeled series
        for label, data in results:
            heights = [h for h, _ in data]
            times = [t for _, t in data]
            plt.plot(heights, times, marker='o', label=label)
    else:
        # single unlabeled series: treat `results` as sequence of (h, t)
        heights = [h for h, _ in results]
        times = [t for _, t in results]
        plt.plot(heights, times, marker='o', label='Series')
    plt.xlabel('Height of the Tree')
    plt.ylabel('Average Time (seconds)')
    plt.title('Performance Comparison of BFS Implementations')
    plt.legend()
    plt.grid()
    if i == 0:
        plt.yscale('log')
        plt.savefig('bfs_comparison_log.png')
    else:
        plt.savefig('bfs_comparison.png')
        
if __name__ == "__main__":
    for h in range(1, 10):
        r = 2 # branching factor (childs per node)
        graph = nx.balanced_tree(r, h)
        G = graph
        iterations = 100

        avg_time = iterator(G, iterations)
        plot_results([(f"Balanced Tree (r={r})", [(h, avg_time)])], i=1)

        print(f"Graph with {graph.number_of_nodes()} nodes (r={r}, h={h})")
        print(f"Average time of {iterations} iterations: {avg_time:.8f} seconds")