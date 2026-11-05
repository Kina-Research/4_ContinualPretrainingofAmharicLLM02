"""
plot_results.py
===============
Plotting script for:
  1. NER F1 retention curves across CL stages (Figure 1)
  2. Entity-level F1 bar chart at final stage (Figure 2)
  3. Data-efficiency curves: F1 vs training fraction (Figure 3)
  4. Pareto frontier: GPU memory vs Final F1 (Figure 4)
  5. Forgetting heatmap of the performance matrix (Figure 5)

Usage:
    python plot_results.py \
        --results_dir ./outputs \
        --output_dir  ./figures
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from evaluate_ner import aggregate_results, compute_cl_metrics

METHODS = ["full_ft", "lora", "dora"]
METHOD_LABELS = {"full_ft": "Full-FT", "lora": "LoRA", "dora": "DoRA"}
METHOD_COLORS = {"full_ft": "#e74c3c", "lora": "#2980b9", "dora": "#27ae60"}
METHOD_MARKERS = {"full_ft": "s", "lora": "^", "dora": "o"}


def load_all_results(results_dir: str) -> dict:
    out = {}
    for m in METHODS:
        path = os.path.join(results_dir, m, "results.json")
        if os.path.exists(path):
            with open(path) as f:
                out[m] = json.load(f)
        else:
            out[m] = None
    return out


# ---------------------------------------------------------------------------
# Figure 1: F1 retention curves across CL stages
# ---------------------------------------------------------------------------
def plot_retention_curves(all_results: dict, output_dir: str):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in METHODS:
        if all_results[m] is None:
            continue
        stage_f1s = []
        for seed_res in all_results[m]:
            pm = seed_res["performance_matrix"]
            T = len(pm) - 1
            # Mean F1 across all shards at each stage
            means = [np.mean(pm[t]) for t in range(T + 1)]
            stage_f1s.append(means)
        stage_f1s = np.array(stage_f1s)
        mean_curve = stage_f1s.mean(axis=0)
        std_curve  = stage_f1s.std(axis=0)
        stages = list(range(len(mean_curve)))
        ax.plot(stages, mean_curve,
                color=METHOD_COLORS[m], marker=METHOD_MARKERS[m],
                linewidth=2, markersize=7, label=METHOD_LABELS[m])
        ax.fill_between(stages,
                        mean_curve - std_curve,
                        mean_curve + std_curve,
                        alpha=0.15, color=METHOD_COLORS[m])
    ax.set_xlabel("Continual Learning Stage", fontsize=12)
    ax.set_ylabel("Mean Macro-F1 (across all shards)", fontsize=12)
    ax.set_title("NER Performance Retention Across CL Stages", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Stage 0\n(baseline)", "Stage 1\n($D_1$)",
                         "Stage 2\n($D_2$)", "Stage 3\n($D_3$)"])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig1_retention_curves.pdf")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Entity-level F1 bar chart (final stage)
# ---------------------------------------------------------------------------
ENTITIES = ["LOC", "ORG", "PER", "TIME", "TTL"]
# Values from manuscript Table 4 (actual experimental results)
ENTITY_F1 = {
    "full_ft": [0.522, 0.612, 0.860, 0.470, 0.857],
    "lora":    [0.630, 0.721, 0.963, 0.527, 0.914],
    "dora":    [0.630, 0.726, 0.950, 0.517, 0.914],
}

def plot_entity_f1(output_dir: str):
    x = np.arange(len(ENTITIES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, m in enumerate(METHODS):
        vals = ENTITY_F1[m]
        ax.bar(x + (i - 1) * width, vals, width,
               label=METHOD_LABELS[m], color=METHOD_COLORS[m],
               alpha=0.85, edgecolor="white")
    ax.set_xlabel("Entity Category", fontsize=12)
    ax.set_ylabel("Macro-F1", fontsize=12)
    ax.set_title("Entity-Level F1 at Final CL Stage", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(ENTITIES, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0.3, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig2_entity_f1.pdf")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Data-efficiency curves (requires data-fraction results)
# ---------------------------------------------------------------------------
FRACTIONS = [10, 25, 50, 100]

def plot_data_efficiency(results_dir: str, output_dir: str):
    """
    Expects results.json files in:
        results_dir/<method>/frac<N>/results.json
    Falls back to placeholder if files are missing.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in METHODS:
        f1s_per_frac = []
        found_any = False
        for frac in FRACTIONS:
            path = os.path.join(results_dir, m, f"frac{frac}", "results.json")
            if os.path.exists(path):
                with open(path) as f:
                    res = json.load(f)
                agg = aggregate_results(res)
                f1s_per_frac.append(agg["mean_final_f1"])
                found_any = True
            else:
                f1s_per_frac.append(None)
        if not found_any:
            print(f"  [WARN] No fraction results found for {m}; skipping curve.")
            continue
        valid_frac = [FRACTIONS[i] for i, v in enumerate(f1s_per_frac) if v is not None]
        valid_f1   = [v for v in f1s_per_frac if v is not None]
        ax.plot(valid_frac, valid_f1,
                color=METHOD_COLORS[m], marker=METHOD_MARKERS[m],
                linewidth=2, markersize=7, label=METHOD_LABELS[m])

    ax.set_xlabel("Training Data Fraction (%)", fontsize=12)
    ax.set_ylabel("Mean Final Macro-F1", fontsize=12)
    ax.set_title("Data-Efficiency Curves", fontsize=13)
    ax.set_xticks(FRACTIONS)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig3_data_efficiency.pdf")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: Pareto frontier — GPU memory vs Final F1
# ---------------------------------------------------------------------------
# Values from manuscript Table 3 (actual results)
EFFICIENCY_DATA = {
    "full_ft": {"gpu_gb": 3.604, "final_f1": 0.6616, "time_s": 4891.9},
    "lora":    {"gpu_gb": 2.669, "final_f1": 0.7461, "time_s": 5715.7},
    "dora":    {"gpu_gb": 2.348, "final_f1": 0.7458, "time_s": 10864.2},
}

def plot_pareto(output_dir: str):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for m in METHODS:
        d = EFFICIENCY_DATA[m]
        ax.scatter(d["gpu_gb"], d["final_f1"],
                   color=METHOD_COLORS[m], s=150,
                   marker=METHOD_MARKERS[m], zorder=3, label=METHOD_LABELS[m])
        ax.annotate(METHOD_LABELS[m],
                    (d["gpu_gb"], d["final_f1"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=10)
    ax.set_xlabel("Peak GPU Memory (GB)", fontsize=12)
    ax.set_ylabel("Mean Final Macro-F1", fontsize=12)
    ax.set_title("Pareto Frontier: Memory vs. NER Accuracy", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.invert_xaxis()   # lower memory is better (right on x-axis)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig4_pareto.pdf")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 5: Performance matrix heatmap (DoRA)
# ---------------------------------------------------------------------------
# R[i][j] values from the actual DoRA experiment
DORA_MATRIX = np.array([
    [0.61, 0.61, 0.62],   # Stage 0 (zero-shot) -- placeholder; update with actual
    [0.74, 0.68, 0.66],   # Stage 1 after D1
    [0.80, 0.75, 0.71],   # Stage 2 after D2
    [0.82, 0.82, 0.77],   # Stage 3 after D3  (R3,2=0.822, R3,3=0.770)
])

def plot_performance_matrix(output_dir: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(DORA_MATRIX, vmin=0.55, vmax=0.90, cmap="YlGn")
    ax.set_xticks(range(3))
    ax.set_yticks(range(4))
    ax.set_xticklabels(["Shard $D_1$", "Shard $D_2$", "Shard $D_3$"], fontsize=10)
    ax.set_yticklabels(["Stage 0", "Stage 1", "Stage 2", "Stage 3"], fontsize=10)
    ax.set_xlabel("Evaluation Shard", fontsize=11)
    ax.set_ylabel("Training Stage", fontsize=11)
    ax.set_title("DoRA Performance Matrix $R_{i,j}$", fontsize=12)
    for i in range(DORA_MATRIX.shape[0]):
        for j in range(DORA_MATRIX.shape[1]):
            ax.text(j, i, f"{DORA_MATRIX[i, j]:.2f}",
                    ha="center", va="center", fontsize=11,
                    color="black" if DORA_MATRIX[i, j] < 0.80 else "white")
    plt.colorbar(im, ax=ax, label="Macro-F1")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig5_performance_matrix.pdf")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_dir",  default="./figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = load_all_results(args.results_dir)

    print("\n[Fig 1] NER retention curves...")
    plot_retention_curves(all_results, args.output_dir)

    print("[Fig 2] Entity-level F1...")
    plot_entity_f1(args.output_dir)

    print("[Fig 3] Data-efficiency curves...")
    plot_data_efficiency(args.results_dir, args.output_dir)

    print("[Fig 4] Pareto frontier...")
    plot_pareto(args.output_dir)

    print("[Fig 5] Performance matrix heatmap...")
    plot_performance_matrix(args.output_dir)

    print("\nAll figures saved.")


if __name__ == "__main__":
    main()
