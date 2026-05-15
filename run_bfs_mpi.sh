#!/bin/bash
# run_bfs.sh
# Runs the MPI BFS script with 1..MAX_PROCS processes and saves results to JSON.
# Usage: bash run_bfs.sh [path/to/bfs_mpi.py]

BFS_SCRIPT="${1:-bfs_mpi.py}"   # first argument, default: bfs_mpi.py
MAX_PROCS=15
R=2
H=17
RUNS=10
OUTPUT="bfs_results.json"

# ── sanity checks ────────────────────────────────────────────────────────────
if [ ! -f "$BFS_SCRIPT" ]; then
    echo "ERROR: BFS script '$BFS_SCRIPT' not found."
    echo "Usage: bash run_bfs.sh [path/to/bfs_mpi.py]"
    exit 1
fi

if ! command -v mpirun &>/dev/null; then
    echo "ERROR: mpirun not found. Please install OpenMPI / MPICH."
    exit 1
fi

# ── run ──────────────────────────────────────────────────────────────────────
echo "Running MPI BFS: r=$R  h=$H  runs=$RUNS  max_procs=$MAX_PROCS"
echo "Results will be saved to $OUTPUT"
echo ""

# Start JSON array
echo "[" > "$OUTPUT"

for NP in $(seq 1 $MAX_PROCS); do
    echo -n "  np=$NP ... "

    # Capture stdout (the printed avg time); suppress MPI noise on stderr
    AVG_TIME=$(mpirun -np "$NP" python3 "$BFS_SCRIPT" \
                   --r "$R" --h "$H" --runs "$RUNS" \
                   2>/dev/null)

    # Check mpirun succeeded and output looks like a number
    if [[ $? -ne 0 ]] || ! [[ "$AVG_TIME" =~ ^[0-9]+\.[0-9]+$ ]]; then
        echo "FAILED (got: '$AVG_TIME') — skipping"
        continue
    fi

    echo "avg_time=${AVG_TIME}s"

    # Append JSON object (trailing comma handled below)
    echo "  {\"processes\": $NP, \"avg_time_s\": $AVG_TIME}," >> "$OUTPUT"
done

# Remove last trailing comma and close array (valid JSON)
# Replace the last ',' before the closing bracket
sed -i '$ s/,$//' "$OUTPUT"
echo "]" >> "$OUTPUT"

echo ""
echo "Done. Results saved to $OUTPUT"