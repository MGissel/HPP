import networkx as nx
from collections import deque
from Part_A.wrappers import timer_func

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