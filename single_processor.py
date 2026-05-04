import networkx as nx
from collections import deque

G = nx.erdos_renyi_graph(100, 0.3)

def BFS(graph, start):
    distance = {v: float('inf') for v in graph.nodes}
    visited = {v: False for v in graph.nodes}

    distance[start] = 0
    visited[start] = True

    Q = deque([start])
    while Q:
        u = Q.popleft()
        for w in graph.neighbors(u):
            if not visited[w]:
                visited[w] = True
                distance[w] = distance[u] + 1
                Q.append(w)
    
    return distance

distance = BFS(G, 0)
print(distance)