from B_shared_mem_parallel_Numba import SharedMemoryParallelBFS, iterator
from B_shared_mem_JIT import iterator as iterator_jit
from B_share_mem_parallel_JIT import iterator as iterator_parallel_jit
from B_shared_mem_threads import iterator as iterator_threading

import matplotlib.pyplot as plt


def plot_results(numba_results, JIT_results, parallel_JIT_results):

    heights = [res[0] for res in numba_results]
    numba_times = [res[1] for res in numba_results]
    JIT_times = [res[1] for res in JIT_results]
    parallel_JIT_times = [res[1] for res in parallel_JIT_results]

    for i in range(2):
        plt.figure(figsize=(10, 6))
        plt.plot(heights, numba_times, label='Shared Memory Parallel BFS (Numba)', marker='o')
        plt.plot(heights, JIT_times, label='BFS with JIT', marker='o')
        plt.plot(heights, parallel_JIT_times, label='Parallel BFS with JIT', marker='o')
        plt.xlabel('Height of the Tree')
        plt.ylabel('Average Time (seconds)')
        plt.title('Performance Comparison of BFS Implementations')
        if i == 0:
            plt.yscale('log')
        plt.legend()
        plt.grid()
        plt.savefig('graphs/bfs_comparison.png')
        plt.show()


if __name__ == "__main__":

    r = 2 # branching factor (childs per node)
    iterations = 30
    source = 0
    numba_results = []
    JIT_results = []
    parallel_JIT_results = []
    C_dist_mem_results = []

    for i in range(1, 20):
        print(f"Running Numba BFS for tree height {i}...")
        h = i # height of the tree
        
        # NUMBA BFS #####################################
        graph = SharedMemoryParallelBFS(r, h)
        n = graph.n
        avg_time, dist, n = iterator(graph, iterations)
        numba_results.append((i, float(avg_time)))
        #################################################
        
        print(f"Running JIT BFS for tree height {i}...")

        # JIT BFS #######################################
        G = graph.G
        time_avg = iterator_jit(G, iterations)
        JIT_results.append((i, float(time_avg)))
        #################################################

        print(f"Running Parallel JIT BFS for tree height {i}...")

        # Parallel JIT BFS ##############################
        time_avg_parallel = iterator_parallel_jit(G, iterations)
        parallel_JIT_results.append((i, float(time_avg_parallel)))
        #################################################

        # parallel by Threading BFS ##############################
        time_avg_parallel_threading = iterator_threading(G, iterations)
        C_dist_mem_results.append((i, float(time_avg_parallel_threading)))

    # save reasults to json file
    import json
    with open('results/bfs_comparison_results.json', 'w') as f:
        json.dump({
            'numba_results': numba_results,
            'JIT_results': JIT_results,
            'parallel_JIT_results': parallel_JIT_results,
            'C_dist_mem_results': C_dist_mem_results
        }, f, indent=4)
    
    print(f"numba_results: {numba_results} \n")
    print(f"JIT_results: {JIT_results} \n")
    print(f"parallel_JIT_results: {parallel_JIT_results}")
    
    plot_results(numba_results, JIT_results, parallel_JIT_results)