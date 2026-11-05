"""
compare_methods.py
==================
Cross-method statistical comparison and result table generation.

Compares Full-FT, LoRA, DoRA, LoRA+EWC results.json files, produces:
  - Table 2: CL Dynamics (BWT/FWT/SG)
  - Table 3: Computational Efficiency
  - Table 4: Entity-Level F1
  - Table 5: Statistical summary with Welch t-tests
  - Data-efficiency curve data (CSV)

Usage:
    python compare_methods.py \
        --results_dir ./outputs \
        --output_dir  ./tables
"""

import argparse
import csv
import json
import os
import numpy as np
from scipy import stats
from evaluate_ner import compute_cl_metrics, aggregate_results


METHODS = ["full_ft", "lora", "dora", "lora_ewc"]
METHOD_LABELS = {
    "full_ft":  "Full-FT",
    "lora":     "LoRA",
    "dora":     "DoRA",
    "lora_ewc": "LoRA+EWC",
}


def load_results(results_dir: str) -> dict:
    """Load results.json for each method."""
    loaded = {}
    for m in METHODS:
        path = os.path.join(results_dir, m, "results.json")
        if os.path.exists(path):
            with open(path) as f:
                loaded[m] = json.load(f)
        else:
            print(f"  [WARN] Missing results for method '{m}': {path}")
            loaded[m] = None
    return loaded


def table_cl_dynamics(all_results: dict) -> list:
    """Table 2: BWT, FWT, SG for each method."""
    rows = []
    for m in METHODS:
        if all_results[m] is None:
            rows.append({
                "Method": METHOD_LABELS[m],
                "BWT (mean)": "N/A", "BWT (std)": "N/A",
                "FWT (mean)": "N/A", "FWT (std)": "N/A",
                "SG (mean)":  "N/A", "SG (std)":  "N/A",
            })
            continue
        agg = aggregate_results(all_results[m])
        rows.append({
            "Method":    METHOD_LABELS[m],
            "BWT (mean)": f"{agg['mean_bwt']:+.4f}",
            "BWT (std)":  f"{agg['std_bwt']:.4f}",
            "FWT (mean)": f"{agg['mean_fwt']:+.4f}",
            "FWT (std)":  f"{agg['std_fwt']:.4f}",
            "SG (mean)":  f"{-agg['mean_bwt']:+.4f}",
            "SG (std)":   f"{agg['std_bwt']:.4f}",
        })
    return rows


def table_efficiency(all_results: dict) -> list:
    """Table 3: Trainable params %, wall-clock time, peak GPU, final F1."""
    rows = []
    for m in METHODS:
        if all_results[m] is None:
            rows.append({"Method": METHOD_LABELS[m],
                         "Params (%)": "N/A", "Time (s)": "N/A",
                         "Peak GPU (GB)": "N/A", "Final F1": "N/A"})
            continue
        pcts, times, gpus, f1s = [], [], [], []
        for seed_res in all_results[m]:
            pcts.append(seed_res["param_info"]["trainable_pct"])
            times.append(seed_res.get("total_wall_clock_s", float("nan")))
            gpus.append(seed_res.get("peak_gpu_gb", float("nan")))
            pm = seed_res["performance_matrix"]
            T = len(pm) - 1
            f1s.append(np.mean(pm[T]))
        rows.append({
            "Method":        METHOD_LABELS[m],
            "Params (%)":    f"{np.mean(pcts):.3f}",
            "Time (s)":      f"{np.nanmean(times):.1f}",
            "Peak GPU (GB)": f"{np.nanmean(gpus):.3f}",
            "Final F1":      f"{np.mean(f1s):.4f}",
        })
    return rows


def table_stats(all_results: dict, reference_method: str = "dora") -> list:
    """Table 5: Mean F1 ± 95%CI, Welch t-test vs reference."""
    ref_f1s = None
    if all_results.get(reference_method):
        ref_f1s = [
            np.mean(r["performance_matrix"][len(r["performance_matrix"])-1])
            for r in all_results[reference_method]
        ]

    rows = []
    for m in METHODS:
        if all_results[m] is None:
            rows.append({"Method": METHOD_LABELS[m],
                         "Mean F1": "N/A", "Std": "N/A",
                         "95% CI": "N/A", "p vs DoRA": "N/A", "t vs DoRA": "N/A"})
            continue
        agg = aggregate_results(all_results[m])
        f1s = [
            np.mean(r["performance_matrix"][len(r["performance_matrix"])-1])
            for r in all_results[m]
        ]
        if ref_f1s is not None and m != reference_method:
            t_stat, p_val = stats.ttest_ind(f1s, ref_f1s, equal_var=False)
        else:
            t_stat, p_val = float("nan"), float("nan")

        ci = agg["ci_95"]
        rows.append({
            "Method":     METHOD_LABELS[m],
            "Mean F1":    f"{agg['mean_final_f1']:.4f}",
            "Std":        f"{agg['std_final_f1']:.4f}",
            "95% CI":     f"[{ci[0]:.4f}, {ci[1]:.4f}]",
            "p vs DoRA":  f"{p_val:.3f}" if not np.isnan(p_val) else "---",
            "t vs DoRA":  f"{t_stat:.3f}" if not np.isnan(t_stat) else "---",
        })
    return rows


def save_csv(rows: list, filepath: str):
    if not rows:
        return
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {filepath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_dir",  default="./tables")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nLoading results from: {args.results_dir}")
    all_results = load_results(args.results_dir)

    print("\n[Table 2] CL Dynamics (BWT/FWT/SG)...")
    rows_cl = table_cl_dynamics(all_results)
    save_csv(rows_cl, os.path.join(args.output_dir, "table2_cl_dynamics.csv"))

    print("[Table 3] Computational Efficiency...")
    rows_eff = table_efficiency(all_results)
    save_csv(rows_eff, os.path.join(args.output_dir, "table3_efficiency.csv"))

    print("[Table 5] Statistical Summary...")
    rows_stats = table_stats(all_results, reference_method="dora")
    save_csv(rows_stats, os.path.join(args.output_dir, "table5_stats.csv"))

    print("\nAll tables saved.")


if __name__ == "__main__":
    main()
