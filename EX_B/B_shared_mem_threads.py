from numba import jit
import networkx as nx
import threading as th
import time
import numpy as np

distance = {} # Initialize distances to infinity
visited = {} # Initialize visited status to False
frontier = [] # Initialize the frontier as an empty list

def tree(x, y):
    G = nx.balanced_tree(r=x, h=y)
    #G = nx.watts_strogatz_graph(n=x, k=4, p=0.1)
    return G


def BFS(node, graph):
    global frontier, distance, visited
    frontier.remove(node) # Remove the node from the frontier
    for neighbor in graph.neighbors(node): # Iterate through neighbors of the current node
        if not visited[neighbor]: # If the neighbor has not been visited
                visited[neighbor] = True # Mark the neighbor as visited
                distance[neighbor] = distance[node] + 1 # Update the distance to the neighbor
                frontier.append(neighbor) # Add the neighbor to the frontier
                #print(f"From node {node} Visited node {neighbor} with distance {distance[neighbor]}")

                

def parallel_BFS(graph):
    global frontier
    threads = []
    while frontier: # While there are nodes in the frontier
        current_frontier = frontier.copy() # Create a copy of the frontier to iterate over
        for node in current_frontier:
            thread = th.Thread(target=BFS, args=(node, graph)) # Create a thread for each node in the frontier
            threads.append(thread)
            #print(f"Started thread {thread} for node {node}")
            thread.start() # Start the thread

        for thread in threads:
            thread.join() # Wait for all threads to finish

def iterator(graph, iterations):
    global distance, visited, frontier
    graph=graph
    distance = {v: float('inf') for v in graph.nodes} # Initialize distances to infinity
    visited = {v: False for v in graph.nodes} # Initialize visited status to False

    timer_array = np.zeros(iterations) # Initialize an array to store the time taken for each iteration

    for i in range(iterations):
        distance[0] = 0 # Set the distance to the source node to 0
        frontier = [0] # Initialize the frontier with the source node
        visited[0] = True # Mark the source node as visited
        time_start = time.time() # Start the timer
        parallel_BFS(graph)
        time_end = time.time() # End the timer
        timer_array[i] = time_end - time_start # Store the time taken for this iteration

    return np.mean(timer_array) # Return the average time taken across all iterations


if __name__ == "__main__":
    x = 3 # Branching factor
    y = 4 # Height of the tree
    iterations = 5 # Number of iterations to run
    graph = tree(x, y) # Create the graph
    average_time = iterator(graph, iterations) # Run the iterator and get the average time
    print(f"Average time taken for BFS: {average_time:.8f} seconds")