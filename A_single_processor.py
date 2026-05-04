import networkx as nx
from collections import deque
from time import time
import sqlite3
import json
import database
from wrappers import timer_func
from dataclasses import dataclass, field

#------------------------#
#-----Configurations-----#
#------------------------#
NUMBER_OF_NODES = 10000
RUNS            = 100        # Minimum of 20 runs

BA_M            = 2         # Barabasi-Albert: edges per new node
WS_K            = 4         # Watts-Strogatz: nearest neighbors
WS_P            = 0.1       # Watts-Strogatz: rewiring probability
ER_P            = 0.1       # Bipartite: edge probability

DP_PATH         = "results.db" # Database path for storing results

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

#--------------------------#
#-----Graph generation-----#
#--------------------------#
def generate_graph(graph_type: function, **kwargs):
    return graph_type(**kwargs)

#--------------------------#
#-----Benchmark runner-----#
#--------------------------#
def benchmark(con: sqlite3.Connection, name: str, graph_fn, runs=RUNS, **graph_kwargs):
    print(f"\n--{name} ({'  '.join(f'{k}={v}' for k, v in graph_kwargs.items())})--")
    parms = json.dumps(graph_kwargs)
    times = []
    
    for i in range(runs):
        bar = "#" * (i // (runs // 20) + 1)
        print(f"({bar:<20}) Run {i+1}/{runs}", end="\r")

        G = generate_graph(graph_fn, **graph_kwargs)
        e = G.number_of_edges()
        _, elapsed = BFS(G, start=0)
        times.append(elapsed)
    
        con.execute('''
        INSERT INTO bfs_runs (graph_type, n, edge_count, parms_json, run_index, elapsed)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (name, graph_kwargs.get('n', 0), e, parms, i, elapsed))
    con.commit()

    avg = sum(times) / len(times)
    print(f"\nAverage: {avg:.4f}s  Min: {min(times):.4f}s  Max: {max(times):.4f}s  Total: {sum(times):.4f}s")
    return avg

#-----------------------#
#-----Summary query-----#
#-----------------------#
def print_summary(con: sqlite3.Connection):
    print("\n─── Summary ───────────────────────────────────────────────────")
    rows = con.execute("""
        SELECT graph_type, n,
               COUNT(*)                            AS runs,
               ROUND(AVG(elapsed), 5)      AS avg_s,
               ROUND(MIN(elapsed), 5)      AS min_s,
               ROUND(MAX(elapsed), 5)      AS max_s
        FROM   bfs_runs
        GROUP  BY graph_type, n
        ORDER  BY graph_type, n
    """).fetchall()
    print(f"{'Graph':<20} {'n':>7} {'runs':>5} {'avg':>9} {'min':>9} {'max':>9}")
    print("─" * 65)
    for r in rows:
        print(f"{r[0]:<20} {r[1]:>7} {r[2]:>5} {r[3]:>9} {r[4]:>9} {r[5]:>9}")

#-------------------------#
#-----Run experiments-----#
#-------------------------#
database_table_columns = {
    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
    "graph_type": "TEXT NOT NULL",
    "n": "INTEGER NOT NULL",
    "edge_count": "INTEGER NOT NULL",
    "parms_json": "TEXT NOT NULL",
    "run_index": "INTEGER NOT NULL",
    "elapsed": "REAL NOT NULL",
    "timestamp": "DATETIME DEFAULT (datetime('now'))"
}
con = database.init_db(DP_PATH, "bfs_runs", database_table_columns)

N = NUMBER_OF_NODES
averages = {}
for nodes in [i for i in range(10, 17)]: # 1024 to 65536
    #benchmark(con, "Barbasi-Albert", nx.barabasi_albert_graph, n=nodes, m=BA_M)
    #benchmark(con, "Watts-Strogatz", nx.watts_strogatz_graph, n=nodes, k=WS_K, p=WS_P)
    avg = benchmark(con, "Balanced Tree", nx.balanced_tree, r=2, h=nodes)
    averages[nodes] = avg

print_summary(con)
con.close()

def iterator():
    return averages