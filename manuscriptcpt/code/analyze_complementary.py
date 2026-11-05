"""
analyze_complementary.py
========================
Aggregates the outputs of `complementary_experiments.py` (and of the original
runs) from their `results.json` files and emits LaTeX table fragments that can
be \\input{} into the manuscript.  Nothing is hard-coded: if a results file is
missing the corresponding table cell is written as "--" so that un-run
experiments can never be mistaken for results.

Usage:
    python analyze_complementary.py --exp_dir ./experiments/complementary \
                                    --output_dir ./experiments/complementary/tex
"""

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np
from scipy import stats

from evaluate_ner import compute_cl_metrics


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def load(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def final_f1(res: Dict) -> float:
    pm = res["performance_matrix"]
    return float(np.mean(pm[-1]))


def summarise(results: List[Dict]) -> Dict:
    if not results:
        return {}
    f1 = np.array([final_f1(r) for r in results])
    cl = [compute_cl_metrics(r["performance_matrix"]) for r in results]
    n = len(f1)
    sd = float(np.std(f1, ddof=1)) if n > 1 else float("nan")
    half = (stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else float("nan")
    return {
        "n": n,
        "f1": f1,
        "f1_mean": float(f1.mean()),
        "f1_sd": sd,
        "f1_ci": (float(f1.mean() - half), float(f1.mean() + half)),
        "bwt": np.array([c["bwt"] for c in cl]),
        "fwt": np.array([c["fwt"] for c in cl]),
        "gpu": float(np.mean([r.get("peak_gpu_gb", np.nan) for r in results])),
        "time": float(np.mean([r.get("total_wall_clock_s", np.nan) for r in results])),
        "pct": float(np.mean([r["param_info"]["trainable_pct"] for r in results])),
    }


def fmt(x, nd=4):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def paired_test(a: np.ndarray, b: np.ndarray) -> Dict:
    """Paired (same-seed) comparison: paired t, Wilcoxon, bootstrap CI."""
    if len(a) != len(b) or len(a) < 2:
        return {}
    d = a - b
    t, p = stats.ttest_rel(a, b)
    rng = np.random.default_rng(0)
    boots = [rng.choice(d, size=len(d), replace=True).mean() for _ in range(10000)]
    return {
        "mean_diff": float(d.mean()),
        "t": float(t), "p": float(p),
        "boot_ci": (float(np.percentile(boots, 2.5)),
                    float(np.percentile(boots, 97.5))),
        "cohen_dz": float(d.mean() / (d.std(ddof=1) + 1e-12)),
    }


def holm(pd: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pd.items(), key=lambda kv: kv[1])
    m, run, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def table_main_extended(exp: str, out: str):
    """E1: main comparison with n >= 5 seeds, same-seed pairing, Holm."""
    methods = {"Full-FT": "full_ft", "LoRA": "lora", "DoRA": "dora"}
    S = {m: summarise(load(f"{exp}/E1_seeds/{d}/results.json"))
         for m, d in methods.items()}
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Method & $n$ & F1 & 95\,\% CI & BWT & FWT \\", r"\midrule"]
    for m, s in S.items():
        if not s:
            lines.append(f"{m} & -- & -- & -- & -- & -- \\\\")
            continue
        lines.append(
            f"{m} & {s['n']} & {fmt(s['f1_mean'])} & "
            f"[{fmt(s['f1_ci'][0])}, {fmt(s['f1_ci'][1])}] & "
            f"{fmt(s['bwt'].mean())} & {fmt(s['fwt'].mean())} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/E1_main.tex", "w").write("\n".join(lines))

    # paired comparisons with Holm correction
    pairs = {"DoRA-FullFT": ("DoRA", "Full-FT"),
             "LoRA-FullFT": ("LoRA", "Full-FT"),
             "DoRA-LoRA": ("DoRA", "LoRA")}
    res, pvals = {}, {}
    for k, (a, b) in pairs.items():
        if S[a] and S[b]:
            res[k] = paired_test(S[a]["f1"], S[b]["f1"])
            if res[k]:
                pvals[k] = res[k]["p"]
    adj = holm(pvals) if pvals else {}
    for k in res:
        res[k]["p_holm"] = adj.get(k)
    json.dump(res, open(f"{out}/E1_paired.json", "w"), indent=2)

    # FWT / BWT DoRA vs LoRA (paired)
    if S["DoRA"] and S["LoRA"]:
        cl = {"FWT": paired_test(S["DoRA"]["fwt"], S["LoRA"]["fwt"]),
              "BWT": paired_test(S["DoRA"]["bwt"], S["LoRA"]["bwt"])}
        json.dump(cl, open(f"{out}/E1_cl_dora_vs_lora.json", "w"), indent=2)


def table_variants(exp: str, group: str, out: str, header: str):
    """Generic one-row-per-tag table (shard order, rank, fraction, lr ...)."""
    rows = []
    for d in sorted(glob.glob(f"{exp}/{group}/*/results.json")):
        tag = os.path.basename(os.path.dirname(d))
        s = summarise(load(d))
        if s:
            rows.append((tag, s))
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             f"{header} & $n$ & F1 & BWT & FWT & Mem (GB) & Time (s) \\\\",
             r"\midrule"]
    if not rows:
        lines.append(r"\multicolumn{7}{c}{\textit{not yet run}} \\")
    for tag, s in rows:
        lines.append(
            f"{tag.replace('_', ' ')} & {s['n']} & {fmt(s['f1_mean'])}"
            f" & {fmt(s['bwt'].mean())} & {fmt(s['fwt'].mean())}"
            f" & {fmt(s['gpu'], 3)} & {fmt(s['time'], 0)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/{group}.tex", "w").write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", default="./experiments/complementary")
    ap.add_argument("--output_dir", default="./experiments/complementary/tex")
    a = ap.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)

    table_main_extended(a.exp_dir, a.output_dir)
    for grp, hdr in [("E2_shard_order", "Order"),
                     ("E3_per_shard_test", "Config"),
                     ("E4_memory", "Method"),
                     ("E5_ewc", "Variant"),
                     ("E6_fraction", "Fraction"),
                     ("E7_rank", "Rank"),
                     ("E8_fullft_lr", "Full-FT LR")]:
        table_variants(a.exp_dir, grp, a.output_dir, hdr)
    print(f"LaTeX fragments written to {a.output_dir}")


if __name__ == "__main__":
    main()
