from comparison import main

"""
This file serves as the main entry point for running the BFS performance comparison. 
It imports the main function from the comparison module, 
which orchestrates the execution of various BFS implementations and collects their performance data. 
The results are then exported to a JSON file as well as plotted for visual analysis.

It will run the BFS algorithms on balanced trees of varying heights from h=1 up to h_max.
for each height it will run the bfs algorithm for a specified number of iterations,
to get an average time taken for each implementation.

Usage:

- Ensure all necessary modules and dependencies are installed: 
    matplotlib==3.10.9
    mpi4py==4.1.1
    networkx==3.6.1
    numba==0.64.0
    numpy==2.4.4

- define the parameters: 
      r             (branching factor), 
      iterations    (number of runs for averaging),
      h_max         (the tests will run on tree heights from 1 to >h_max),
      p             (number of processes. Only for multi-process BFS).
- Run this script to execute the performance comparison and generate results and graphs.
"""

if __name__ == "__main__":
    r = 2
    h_max = 20
    iterations = 30
    p = 4 # number of processes (only for multi-process BFS)
    main(r, iterations, h_max, p)
