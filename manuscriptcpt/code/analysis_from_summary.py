"""
analysis_from_summary.py
========================
Re-analysis of the *already reported* summary statistics (mean, SD / 95% CI,
n = 3 seeds).  No new training is performed and no per-seed values are
needed: every quantity below is a deterministic function of numbers printed in
the manuscript tables.

It answers four questions that the original Welch-only analysis left open:

  Q1  What is the *absolute* (percentage-point) and *relative* F1 gain of the
      PEFT methods over Full-FT?
  Q2  Is the DoRA-vs-LoRA difference in FWT / BWT statistically supported?
  Q3  Is BWT / FWT significantly different from 0 for each method?
  Q4  With the observed variance, how many seeds are needed (80 % power)?

Usage:
    python analysis_from_summary.py --output_dir ./derived
"""

import argparse
import json
import math
import os

import numpy as np
from scipy import stats

N = 3  # seeds

# --------------------------------------------------------------------------
# Reported numbers (Tables: clDynamics, efficiency, stats of the manuscript)
# --------------------------------------------------------------------------
F1_MEAN = {"Full-FT": 0.6616, "LoRA": 0.7461, "DoRA": 0.7458}
F1_CI = {
    "Full-FT": (0.6258, 0.6975),
    "LoRA": (0.5994, 0.8929),
    "DoRA": (0.6035, 0.8881),
}
BWT = {  # mean, SD
    "Full-FT": (0.0041, 0.0058),
    "LoRA": (0.0505, 0.0125),
    "DoRA": (0.0513, 0.0163),
}
FWT = {
    "Full-FT": (-0.0072, 0.0036),
    "LoRA": (0.0237, 0.0179),
    "DoRA": (0.0325, 0.0242),
}
GPU = {"Full-FT": 3.604, "LoRA": 2.669, "DoRA": 2.348}
TIME = {"Full-FT": 4891.9, "LoRA": 5715.7, "DoRA": 10864.2}


def sd_from_ci(lo, hi, n=N):
    """Recover the sample SD from a t-based 95% CI."""
    half = (hi - lo) / 2.0
    return half * math.sqrt(n) / stats.t.ppf(0.975, n - 1)


def welch(m1, s1, m2, s2, n1=N, n2=N):
    """Welch t-test from summary statistics; returns t, df, p, CI of diff."""
    v1, v2 = s1 ** 2 / n1, s2 ** 2 / n2
    se = math.sqrt(v1 + v2)
    t = (m1 - m2) / se
    df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    p = 2 * stats.t.sf(abs(t), df)
    tc = stats.t.ppf(0.975, df)
    return {
        "diff": m1 - m2,
        "t": t,
        "df": df,
        "p": p,
        "ci95": (m1 - m2 - tc * se, m1 - m2 + tc * se),
    }


def hedges_g(m1, s1, m2, s2, n1=N, n2=N):
    sp = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    d = (m1 - m2) / sp
    j = 1 - 3 / (4 * (n1 + n2) - 9)
    return d * j


def one_sample_t(m, s, n=N):
    t = m / (s / math.sqrt(n))
    p = 2 * stats.t.sf(abs(t), n - 1)
    tc = stats.t.ppf(0.975, n - 1)
    half = tc * s / math.sqrt(n)
    return {"t": t, "p": p, "ci95": (m - half, m + half)}


def holm(pvals: dict):
    """Holm-Bonferroni step-down adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def n_per_group_for_power(delta, sd1, sd2, power=0.8, alpha=0.05, nmax=200):
    """Smallest n per group for a Welch t-test (non-central t approx)."""
    for n in range(2, nmax + 1):
        v = sd1 ** 2 / n + sd2 ** 2 / n
        se = math.sqrt(v)
        df = v ** 2 / ((sd1 ** 2 / n) ** 2 / (n - 1) + (sd2 ** 2 / n) ** 2 / (n - 1))
        tc = stats.t.ppf(1 - alpha / 2, df)
        ncp = delta / se
        pw = stats.nct.sf(tc, df, ncp) + stats.nct.cdf(-tc, df, ncp)
        if pw >= power:
            return n
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="./derived")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    out = {}

    # ---- SDs recovered from CIs ---------------------------------------
    f1_sd = {k: sd_from_ci(*F1_CI[k]) for k in F1_CI}
    out["f1_sd_recovered"] = f1_sd

    # ---- Q1: absolute vs relative gain --------------------------------
    gain = {}
    for k in ("LoRA", "DoRA"):
        g = F1_MEAN[k] - F1_MEAN["Full-FT"]
        gain[k] = {
            "absolute_pp": 100 * g,
            "relative_pct": 100 * g / F1_MEAN["Full-FT"],
            "welch_vs_fullft": welch(F1_MEAN[k], f1_sd[k],
                                     F1_MEAN["Full-FT"], f1_sd["Full-FT"]),
            "hedges_g": hedges_g(F1_MEAN[k], f1_sd[k],
                                 F1_MEAN["Full-FT"], f1_sd["Full-FT"]),
        }
    out["f1_gain_over_fullft"] = gain
    out["f1_dora_vs_lora"] = welch(F1_MEAN["DoRA"], f1_sd["DoRA"],
                                   F1_MEAN["LoRA"], f1_sd["LoRA"])

    # ---- Q2: DoRA vs LoRA on CL metrics -------------------------------
    out["fwt_dora_vs_lora"] = welch(*FWT["DoRA"], *FWT["LoRA"])
    out["bwt_dora_vs_lora"] = welch(*BWT["DoRA"], *BWT["LoRA"])
    out["fwt_relative_gain_dora_over_lora_pct"] = (
        100 * (FWT["DoRA"][0] / FWT["LoRA"][0] - 1))

    # ---- Q3: metric vs 0 ----------------------------------------------
    vs0 = {}
    for name, tab in (("BWT", BWT), ("FWT", FWT)):
        for k, (m, s) in tab.items():
            vs0[f"{name}:{k}"] = one_sample_t(m, s)
    adj = holm({k: v["p"] for k, v in vs0.items()})
    for k in vs0:
        vs0[k]["p_holm"] = adj[k]
    out["cl_metric_vs_zero"] = vs0

    # ---- Q4: power -----------------------------------------------------
    delta = F1_MEAN["DoRA"] - F1_MEAN["Full-FT"]
    out["seeds_needed_dora_vs_fullft_80pct_power"] = n_per_group_for_power(
        delta, f1_sd["DoRA"], f1_sd["Full-FT"])
    out["seeds_needed_fwt_dora_vs_lora_80pct_power"] = n_per_group_for_power(
        FWT["DoRA"][0] - FWT["LoRA"][0], FWT["DoRA"][1], FWT["LoRA"][1])

    # ---- Efficiency ratios --------------------------------------------
    out["efficiency"] = {
        "dora_gpu_vs_fullft_pct": 100 * (GPU["DoRA"] / GPU["Full-FT"] - 1),
        "dora_gpu_vs_lora_pct": 100 * (GPU["DoRA"] / GPU["LoRA"] - 1),
        "lora_gpu_vs_fullft_pct": 100 * (GPU["LoRA"] / GPU["Full-FT"] - 1),
        "dora_time_vs_lora_x": TIME["DoRA"] / TIME["LoRA"],
        "lora_time_vs_fullft_x": TIME["LoRA"] / TIME["Full-FT"],
        "dora_time_vs_fullft_x": TIME["DoRA"] / TIME["Full-FT"],
    }

    # ---- Print + save ---------------------------------------------------
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, tuple):
            return list(o)
        raise TypeError(type(o))

    path = os.path.join(args.output_dir, "derived_stats.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=conv)
    print(json.dumps(out, indent=2, default=conv))
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
