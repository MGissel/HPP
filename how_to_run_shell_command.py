import subprocess

val = subprocess.check_output(["mpirun", "-np", "4", "python3", "mpi.py", "--r", "2", "--h", "5", "--runs", "1"]).decode('utf-8').strip()

print(f"{val}")