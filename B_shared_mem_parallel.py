from numba import jit
import networkx as nx
import threading as th
import time
import numpy as np

def tree(x, y):
    G = nx.balanced_tree(r=x, h=y)
    #G = nx.watts_strogatz_graph(n=x, k=4, p=0.1)
    return G


def BFS(node, graph):
    frontier.remove(node) # Remove the node from the frontier
    for neighbor in graph.neighbors(node): # Iterate through neighbors of the current node
        if not visited[neighbor]: # If the neighbor has not been visited
            with th.Lock(): # Acquire a lock to ensure thread safety when updating shared data structures
                visited[neighbor] = True # Mark the neighbor as visited
                distance[neighbor] = distance[node] + 1 # Update the distance to the neighbor
                frontier.append(neighbor) # Add the neighbor to the frontier
                print(f"From node {node} Visited node {neighbor} with distance {distance[neighbor]}")

                

def parallel_BFS(graph):
    threads = []
    current_level = 0
    while frontier: # While there are nodes in the frontier
        current_frontier = frontier.copy() # Create a copy of the frontier to iterate over
        for node in current_frontier:
            thread = th.Thread(target=BFS, args=(node, graph)) # Create a thread for each node in the frontier
            threads.append(thread)
            print(f"Started thread {thread} for node {node}")
            thread.start() # Start the thread

        for thread in threads:
            thread.join() # Wait for all threads to finish
        print(f"Completed level {current_level}")
        current_level += 1 # Increment the current level after processing all nodes in the current frontier

if __name__ == "__main__":
    graph=tree(2, 4)
    distance = {v: float('inf') for v in graph.nodes} # Initialize distances to infinity
    visited = {v: False for v in graph.nodes} # Initialize visited status to False
    frontier = [0] # Initialize the frontier with the source node
    distance[0] = 0 # Set the distance to the source node to 0
    visited[0] = True # Mark the source node as visited

    time_start = time.time() # Start the timer
    parallel_BFS(graph)
    time_end = time.time() # End the timer
    #print(f"Total time taken: {time_end - time_start} seconds")
