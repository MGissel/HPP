This is our High Performance Programming Mini Project

Hyg med det!

# USAGE


USE: main.py
It runs a comparison on the different algorithms, of both part A, B and C. 
it compares both sequential, different approaches of shared memory paralleilism, and multi process distributed memeory approach.

Just download the dependencies found in: requirements.txt
command: pip install -r requirements.txt

And fill in the variables of main.py
r:              spreading factor of balanced tree
iterations:     The number of iterations, that it will average the answer over, done for all tree heights
h_max:          the maximum tree depth (it will test for all tree depths up until this)
p:              Number of processors to be used at multi processing (part C)

It will ouptut two graphs and save them. both is the same comparison between the different methods, but one hase the y-axis scaled logarithmically.
the data is also saved to a .json format with timestamp.

