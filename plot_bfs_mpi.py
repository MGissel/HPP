#!/usr/bin/env python3
"""
plot_bfs.py
Reads bfs_results.json produced by run_bfs.sh and generates two plots:
  1. Average BFS time vs number of processes
  2. Speedup vs number of processes  (with ideal speedup reference line)
"""

import json
import argparse
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Plot MPI BFS benchmark results")
parser.add_argument(
    "--input", default="bfs_results.json",
    help="Path to JSON results file (default: bfs_results.json)"
)
parser.add_argument(
    "--output-dir", default=".",
    help="Directory to save the plots (default: current directory)"
)
args = parser.parse_args()

# ── load data ─────────────────────────────────────────────────────────────────
if not os.path.exists(args.input):
    sys.exit(f"ERROR: input file '{args.input}' not found. Run run_bfs.sh first.")

with open(args.input) as f:
    data = json.load(f)

if not data:
    sys.exit("ERROR: JSON file is empty.")

processes = np.array([d["processes"]  for d in data], dtype=int)
times     = np.array([d["avg_time_s"] for d in data], dtype=float)

# Convert to milliseconds for readability
times_ms  = times * 1000.0

# Speedup relative to single-process run
t_serial  = times[processes == 1]
if len(t_serial) == 0:
    sys.exit("ERROR: no single-process (np=1) result found — cannot compute speedup.")
t_serial = t_serial[0]

speedup       = t_serial / times
ideal_speedup = processes.astype(float)

# ── shared style ──────────────────────────────────────────────────────────────
BLUE   = "#2563EB"
RED    = "#DC2626"
GREY   = "#6B7280"
BG     = "#F9FAFB"

plt.rcParams.update({
    "font.family":  "serif",
    "font.size":    12,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

def style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_facecolor(BG)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# ── Plot 1: Time vs Processes ─────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(8, 5))

ax1.plot(processes, times_ms, marker="o", linewidth=2,
         color=BLUE, markersize=6, label="Measured time")
ax1.fill_between(processes, times_ms, alpha=0.12, color=BLUE)

style_ax(ax1,
         title="MPI BFS — Average execution time per run",
         xlabel="Number of MPI processes",
         ylabel="Average time (ms)")

ax1.legend(frameon=False)
fig1.tight_layout()

out1 = os.path.join(args.output_dir, "bfs_time.png")
fig1.savefig(out1, dpi=150)
print(f"Saved: {out1}")

# ── Plot 2: Speedup vs Processes ─────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))

ax2.plot(processes, ideal_speedup, linestyle="--", linewidth=1.5,
         color=GREY, label="Ideal (linear) speedup")
ax2.plot(processes, speedup, marker="o", linewidth=2,
         color=RED, markersize=6, label="Measured speedup")
ax2.fill_between(processes, speedup, alpha=0.12, color=RED)

style_ax(ax2,
         title="MPI BFS — Speedup relative to 1 process",
         xlabel="Number of MPI processes",
         ylabel="Speedup  $S(P) = T_1 / T_P$")

ax2.legend(frameon=False)
fig2.tight_layout()

out2 = os.path.join(args.output_dir, "bfs_speedup.png")
fig2.savefig(out2, dpi=150)
print(f"Saved: {out2}")

# ── summary table ─────────────────────────────────────────────────────────────
print("\n{:>10}  {:>14}  {:>10}".format("Processes", "Avg time (ms)", "Speedup"))
print("-" * 40)
for p, t, s in zip(processes, times_ms, speedup):
    print(f"{p:>10}  {t:>14.4f}  {s:>10.4f}")