import sqlite3
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# ── Load data ─────────────────────────────────────────────────
con  = sqlite3.connect("results.db")
rows = con.execute("""
    SELECT graph_type, n, edge_count, elapsed
    FROM bfs_runs
    ORDER BY graph_type, n
""").fetchall()
con.close()

# Organise into {graph_type: {n: {"times": [], "edges": int}}}
data = defaultdict(lambda: defaultdict(lambda: {"times": [], "edges": 0}))
for graph_type, n, edge_count, elapsed in rows:
    data[graph_type][n]["times"].append(elapsed)
    data[graph_type][n]["edges"] = edge_count

graph_types = sorted(data.keys())
colors      = {"Barbasi-Albert": "#E8593C", "Watts-Strogatz": "#1D9E75"}

# ── Plot 1: Average time vs n (log-log) ───────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for gt in graph_types:
    ns   = sorted(data[gt].keys())
    avgs = [np.mean(data[gt][n]["times"]) for n in ns]
    mins = [np.min(data[gt][n]["times"])  for n in ns]
    maxs = [np.max(data[gt][n]["times"])  for n in ns]
    ax.plot(ns, avgs, marker="o", label=gt, color=colors[gt])
    ax.fill_between(ns, mins, maxs, alpha=0.15, color=colors[gt])

ns_ref = np.array(sorted(data[graph_types[0]].keys()))
ref    = ns_ref * (np.mean(data[graph_types[0]][ns_ref[0]]["times"]) / ns_ref[0])
ax.plot(ns_ref, ref, "k--", linewidth=0.8, label="O(n) reference")

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("Number of nodes (n)")
ax.set_ylabel("BFS time (s)")
ax.set_title("BFS scaling — average ± min/max band")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("plot_scaling.png", dpi=300)
plt.show()

# ── Plot 2: Box plot of run distribution per n ────────────────
fig, axes = plt.subplots(1, len(graph_types), figsize=(14, 5), sharey=True)
for ax, gt in zip(axes, graph_types):
    ns       = sorted(data[gt].keys())
    box_data = [data[gt][n]["times"] for n in ns]
    bp       = ax.boxplot(box_data, patch_artist=True, labels=[str(n) for n in ns])
    for patch in bp["boxes"]:
        patch.set_facecolor(colors[gt])
        patch.set_alpha(0.6)
    ax.set_title(gt)
    ax.set_xlabel("n")
    ax.tick_params(axis="x", rotation=45)
axes[0].set_ylabel("BFS time (s)")
plt.suptitle("BFS run distribution per node count")
plt.tight_layout()
plt.savefig("plot_boxplot.png", dpi=300)
plt.show()

# ── Plot 3: Slowdown factor vs theoretical O(n) ───────────────
fig, ax = plt.subplots(figsize=(8, 5))
for gt in graph_types:
    ns       = sorted(data[gt].keys())
    avgs     = [np.mean(data[gt][n]["times"]) for n in ns]
    slowdown = [a / avgs[0] for a in avgs]
    ax.plot(ns, slowdown, marker="o", label=gt, color=colors[gt])

ax.plot(ns, [n / ns[0] for n in ns], "k--", linewidth=0.8, label="Perfect O(n)")
ax.set_xscale("log", base=2)
ax.set_xlabel("Number of nodes (n)")
ax.set_ylabel("Slowdown factor (relative to n=1024)")
ax.set_title("Empirical scaling factor vs theoretical O(n)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("plot_slowdown.png", dpi=300)
plt.show()

# ── Plot 4: BFS time vs V+E ───────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
all_ve, all_t = [], []
for gt in graph_types:
    ns   = sorted(data[gt].keys())
    ve   = [n + data[gt][n]["edges"]          for n in ns]
    avgs = [np.mean(data[gt][n]["times"])      for n in ns]
    ax.plot(ve, avgs, marker="o", label=gt, color=colors[gt])
    all_ve.extend(ve)
    all_t.extend(avgs)

slope = np.mean([t / ve for t, ve in zip(all_t, all_ve)])
ref_x = np.linspace(min(all_ve), max(all_ve), 200)
ax.plot(ref_x, slope * ref_x, "k--", linewidth=0.8, label="O(V+E) reference")

ax.set_xlabel("V + E")
ax.set_ylabel("BFS time (s)")
ax.set_title("BFS time vs V+E — confirming O(V+E) complexity")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("plot_ve.png", dpi=300)
plt.show()