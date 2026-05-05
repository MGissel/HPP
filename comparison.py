from Part_B.B_shared_mem_parallel_Numba import SharedMemoryParallelBFS, iterator
from Part_B.B_shared_mem_JIT import iterator as iterator_jit
from Part_B.B_share_mem_parallel_JIT import iterator as iterator_parallel_jit
from Part_B.B_shared_mem_threads import iterator as iterator_threading
from Part_A.A_single_processor import iterator as iterator_sequential

import matplotlib.pyplot as plt
import json
import time
import subprocess


def plot_results(numba_results, JIT_results, parallel_JIT_results, threading_results, sequential_results, multi_process_results):

    heights = [res[0] for res in numba_results]
    numba_times = [res[1] for res in numba_results]
    JIT_times = [res[1] for res in JIT_results]
    parallel_JIT_times = [res[1] for res in parallel_JIT_results]
    threading_times = [res[1] for res in threading_results]
    sequential_times = [res[1] for res in sequential_results]
    multi_process_times = [res[1] for res in multi_process_results]

    for i in range(2):
        plt.figure(figsize=(10, 6))
        plt.plot(heights, numba_times, label='Shared Memory Parallel BFS (Numba)', marker='o')
        plt.plot(heights, JIT_times, label='BFS with JIT', marker='o')
        plt.plot(heights, parallel_JIT_times, label='Parallel BFS with JIT', marker='o')
        plt.plot(heights, threading_times, label='Parallel Threading BFS', marker='o')
        plt.plot(heights, sequential_times, label='Sequential BFS', marker='o')
        plt.plot(heights, multi_process_times, label='Multi-Process BFS', marker='o')

        plt
        plt.xlabel('Height of the Tree')
        plt.title('Performance Comparison of BFS Implementations')
        plt.legend()
        plt.grid()
        if i == 0:
            plt.ylabel('Average Time (seconds, log scale)')
            plt.yscale('log')
            plt.savefig('bfs_comparison_log1.png')
        else:
            plt.ylabel('Average Time (seconds)')
            plt.savefig('bfs_comparison1.png')
        plt.show()

def data_bringer(r, iterations, h_max, p):

    numba_results = []
    JIT_results = []
    parallel_JIT_results = []
    threading_results = []
    sequential_results = []
    multi_process_results = []

    for i in range(1, h_max):
    
        h = i # height of the tree

        print(f"Creating graph, with {(2**i)-1} nodes...")
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
        print(f"Running Parallel Threading BFS for tree height {i}...")
        time_avg_parallel_threading = iterator_threading(G, iterations)
        threading_results.append((i, float(time_avg_parallel_threading)))
        #################################################

        # Sequential BFS ################################
        print(f"Running Sequential BFS for tree height {i}...")
        time_avg_sequential = iterator_sequential(G, iterations)
        sequential_results.append((i, float(time_avg_sequential)))
        #################################################

        
        print(f"Running Multi-Process BFS for tree height {i}...")
        val = subprocess.check_output(["mpirun", "-np", str(p), "python3", "Part_C/mpi.py", "--r", str(r), "--h", str(i), "--runs", str(iterations)]).decode('utf-8').strip()
        multi_process_results.append((i, float(val)))
    
    return {
        'numba_results': numba_results,
        'JIT_results': JIT_results,
        'parallel_JIT_results': parallel_JIT_results,
        'threading_results': threading_results,
        'sequential_results': sequential_results,
        'multi_process_results': multi_process_results
    }

def export(results, filename):
    # check if file exists, if not create it
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(filename + f"_{timestamp}.json" , 'w') as f:
        json.dump({
            'results': results
        }, f, indent=4)

def main(r, iterations, h_max, p):
    results = data_bringer(r, iterations, h_max, p)
    export(results, 'bfs_comparison_results')

    # print(f"numba_results: {results['numba_results']} \n")
    # print(f"JIT_results: {results['JIT_results']} \n")
    # print(f"parallel_JIT_results: {results['parallel_JIT_results']}")
    # print(f"threading_results: {results['threading_results']}")
    # print(f"sequential_results: {results['sequential_results']}")
    plot_results(results['numba_results'], 
                 results['JIT_results'], 
                 results['parallel_JIT_results'], 
                 results['threading_results'], 
                 results['sequential_results'],
                 results['multi_process_results'])
        

if __name__ == "__main__":

    r = 2 # branching factor (childs per node)
    iterations = 30
    h_max = 10
    p = 4 # number of processes for multi-process BFS

    main(r, iterations, h_max, p)

    