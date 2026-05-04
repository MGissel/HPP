from B_shared_mem_parallel_Numba import SharedMemoryParallelBFS, iterator
from B_shared_mem_JIT import iterator as iterator_jit
from B_share_mem_parallel_JIT import iterator as iterator_parallel_jit
from B_shared_mem_threads import iterator as iterator_threading

import matplotlib.pyplot as plt
import json
import time


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

def main(r, iterations, h_max):

    numba_results = []
    JIT_results = []
    parallel_JIT_results = []
    threading_results = []

    for i in range(1, h_max):
    
        h = i # height of the tree

        print("Creating graph...")
        graph = SharedMemoryParallelBFS(r, h)
        G = graph.G

        # NUMBA BFS #####################################
        print(f"Running Numba BFS for tree height {i}...")
        avg_time, dist, n = iterator(graph, iterations)
        numba_results.append((i, float(avg_time)))
        #################################################

        # JIT BFS #######################################
        print(f"Running JIT BFS for tree height {i}...")
        time_avg = iterator_jit(G, iterations)
        JIT_results.append((i, float(time_avg)))
        #################################################


        # Parallel JIT BFS ##############################
        print(f"Running Parallel JIT BFS for tree height {i}...")
        time_avg_parallel = iterator_parallel_jit(G, iterations)
        parallel_JIT_results.append((i, float(time_avg_parallel)))
        #################################################


        # parallel by Threading BFS #####################
        # print(f"Running Parallel Threading BFS for tree height {i}...")
        #time_avg_parallel_threading = iterator_threading(G, iterations)
        #threading_results.append((i, float(time_avg_parallel_threading)))
        #################################################
    
    return {
        'numba_results': numba_results,
        'JIT_results': JIT_results,
        'parallel_JIT_results': parallel_JIT_results,
        #'threading_results': threading_results
    }

def export(results, filename):
    # check if file exists, if not create it
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(filename + f"_{timestamp}.json" , 'w') as f:
        json.dump({
            'results': results
        }, f, indent=4)
        

if __name__ == "__main__":

    r = 2 # branching factor (childs per node)
    iterations = 30
    h_max = 15

    results = main(r, iterations, h_max)

    export(results, 'results/bfs_comparison_results')

    # print(f"numba_results: {results['numba_results']} \n")
    # print(f"JIT_results: {results['JIT_results']} \n")
    # print(f"parallel_JIT_results: {results['parallel_JIT_results']}")
    # print(f"threading_results: {results['threading_results']}")

    plot_results(results['numba_results'], results['JIT_results'], results['parallel_JIT_results'])