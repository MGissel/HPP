import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Load data ──────────────────────────────────────────────────────────────────
with open("bfs_comparison_results_2026-05-05 13:44:20.json") as f:
    raw = json.load(f)["results"]

# ── Parse into {label: {depth: time}} ─────────────────────────────────────────
series = {
    "Numba (vectorized)":      raw["numba_results"],
    "JIT":                 raw["JIT_results"],
    "Parallel JIT":        raw["parallel_JIT_results"],
    "Threading":           raw["threading_results"],
    "Multi-process":       raw["multi_process_results"],
    # sequential is the baseline — we keep it for reference but don't plot speedup for it
}
baseline_raw = raw["sequential_results"]

baseline = {int(depth): t for depth, t in baseline_raw}

def to_dict(pairs):
    return {int(depth): t for depth, t in pairs}

# ── Compute speedup = baseline_time / method_time ─────────────────────────────
depths = sorted(baseline.keys())

fig, ax = plt.subplots(figsize=(11, 6.5))
fig.patch.set_facecolor("#ffffff")
ax.set_facecolor("#ffffff")

# colour palette
palette = {
    "Numba (vectorized)": "#e63946",
    "JIT":            "#2196a6",
    "Parallel JIT":   "#e6900a",
    "Threading":      "#6a4fcf",
    "Multi-process":  "#2a9d5c",
}
markers = {
    "Numba (vectorized)": "o",
    "JIT":            "s",
    "Parallel JIT":   "^",
    "Threading":      "D",
    "Multi-process":  "P",
}

for label, pairs in series.items():
    data = to_dict(pairs)
    speedups = []
    xs = []
    for d in depths:
        if d in data and data[d] > 0:
            speedups.append(baseline[d] / data[d])
            xs.append(d)
    ax.plot(
        xs, speedups,
        color=palette[label],
        marker=markers[label],
        markersize=6,
        linewidth=2,
        label=label,
    )

# ── Baseline reference line ────────────────────────────────────────────────────
ax.axhline(1.0, color="#888888", linewidth=1.2, linestyle="--", label="Sequential (baseline)")

# ── Grid & axes ───────────────────────────────────────────────────────────────
ax.set_yscale("log", base=2)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.3g}×"))
ax.yaxis.set_minor_formatter(ticker.NullFormatter())
ax.grid(color="#dddddd", which="major", linestyle="-", linewidth=0.7)
ax.grid(color="#eeeeee", which="minor", linestyle=":", linewidth=0.5)

ax.set_xticks(depths)
ax.set_xlabel("Binary Tree Depth", color="#222222", fontsize=12)
ax.set_ylabel("Speedup over Sequential", color="#222222", fontsize=12)
ax.tick_params(colors="#333333", which="both")
for spine in ax.spines.values():
    spine.set_edgecolor("#bbbbbb")

# ── Legend & title ─────────────────────────────────────────────────────────────
ax.set_title("BFS on Binary Tree — Speedup vs Sequential", color="#111111",
             fontsize=15, fontweight="bold", pad=16)

leg = ax.legend(
    framealpha=0.9,
    facecolor="#ffffff",
    edgecolor="#cccccc",
    labelcolor="#222222",
    fontsize=10,
    loc="upper left",
)

fig.tight_layout()

out = "bfs_speedup.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out}")
plt.show()