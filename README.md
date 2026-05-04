This is our High Performance Programming Mini Project

Hyg med det!




HOW TO USE C-dist_mem.py.
This is a shared memory parallel machine, and must be run with seperate runtime than pythons own.
We Use MPI4PY

## Setup

Before running the script, ensure you have the required dependencies installed:

```bash
sudo apt install libopenmpi-dev openmpi-bin
pip install mpi4py
```

## Running C_dist_mem.py

Execute the script using:

```bash
mpirun -n 4 python3 C_dist_mem.py
```

### Parameters

- **`-n 4`**: Specifies the number of processes to spawn. The `4` is the process count and can be varied based on your system's CPU cores or requirements.

You can adjust the number to match your hardware capabilities or testing needs (e.g., `-n 2`, `-n 8`, etc.).
