from B_shared_mem_parallel_Numba import SharedMemoryParallelBFS, iterator
from B_shared_mem_parallel_JIT import BFS, graph_to_csr, tree


if __name__ == "__main__":

    r = 2 # branching factor (childs per node)
    iterations = 100
    source = 0
    numba_results = []

    for i in range(1, 15):
        
        h = i # height of the tree
        
        # NUMBA BFS ####################################
        graph = SharedMemoryParallelBFS(r, h)
        n = graph.n

        avg_time, dist, n = iterator(graph, iterations)

        numba_results.append((i, avg_time))

        #################################################


        # JIT BFS ######################################
        G = tree(r, h)

        offsets, neighbors_flat = graph_to_csr(G)

        print("Warming up JIT...")
        _ = BFS(offsets, neighbors_flat, source, n)


        time_avg = np.zeros(1000)

        for i in range(1000):
            time_start = time.time()
            distance = BFS(offsets, neighbors_flat, source, n)
            time_end = time.time()
            time_avg[i] = time_end - time_start

        print(f"\nAverage time (excluding warmup): {np.mean(time_avg):.8f} seconds")
        #################################################

    
    
    
    
    
    
    
    
    print(numba_results)