"""
complementary_experiments.py
============================
Runner for the *complementary* experiments that address the validity gaps
identified during the manuscript revision.  Dry-run by default: commands are
printed and written to `run_complementary.sh`; nothing is trained unless
`--execute` is passed.

  E1  More seeds (n = 5) for Full-FT / LoRA / DoRA  -> power, paired tests
  E2  Shard-order sensitivity (123, 231, 312)       -> order effects on BWT
  E3  Disjoint per-shard test partitions (+strict)  -> real forgetting
      measurement (the original protocol scored every column on one set)
  E4  Memory re-measurement with torch peak counters-> DoRA vs LoRA memory
  E5  EWC feasibility: timing, batch-grad vs per-sample Fisher
  E6  Data-efficiency fractions 10/25/50 %          -> H4 (real fraction runs)
  E7  Adapter rank r in {4,16,32} for LoRA and DoRA -> rank sensitivity
  E8  Full-FT learning-rate sweep {1e-5,3e-5,5e-5}  -> fair-baseline check
      (the shared 3e-4 is an adapter-style LR, far above the usual
      full-fine-tuning range for a 125M encoder)

Usage:
    python complementary_experiments.py --raw_dir ./data/raw \
        --base_dir ./experiments/complementary            # dry run
    python complementary_experiments.py ... --execute      # run all
    python complementary_experiments.py ... --execute --only E1 E4
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

PY = sys.executable
SEEDS3 = ["42", "123", "456"]
SEEDS5 = SEEDS3 + ["789", "1011"]
METHODS = ["full_ft", "lora", "dora"]


def train_cmd(method, data_dir, out_dir, seeds, extra=None):
    cmd = [PY, "train.py", "--method", method,
           "--data_dir", str(data_dir), "--output_dir", str(out_dir),
           "--seeds", *seeds]
    return cmd + (extra or [])


def build(base: Path, data: Path, data_pst: Path):
    jobs = []  # (experiment, tag, cmd, timeout_s)

    # E1 ------------------------------------------------------------------
    for m in METHODS:
        jobs.append(("E1", m, train_cmd(m, data, base / "E1_seeds" / m, SEEDS5),
                     None))

    # E2 ------------------------------------------------------------------
    for order in (["2", "3", "1"], ["3", "1", "2"]):
        for m in ("lora", "dora"):
            tag = f"{m}_order_{''.join(order)}"
            jobs.append(("E2", tag, train_cmd(
                m, data, base / "E2_shard_order" / tag, SEEDS3,
                ["--shard_order", *order]), None))

    # E3 ------------------------------------------------------------------
    for m in METHODS:
        jobs.append(("E3", f"{m}_pst", train_cmd(
            m, data_pst, base / "E3_per_shard_test" / f"{m}_pst", SEEDS3,
            ["--per_shard_test"]), None))
        jobs.append(("E3", f"{m}_pst_strict", train_cmd(
            m, data_pst, base / "E3_per_shard_test" / f"{m}_pst_strict", SEEDS3,
            ["--per_shard_test", "--strict_eval"]), None))

    # E4 (memory is independent of corpus size: padded to 256, batch 8) ----
    for m in METHODS:
        jobs.append(("E4", m, train_cmd(
            m, data, base / "E4_memory" / m, SEEDS3,
            ["--data_fraction", "0.1", "--epochs_per_shard", "1"]), None))

    # E5 ------------------------------------------------------------------
    for tag, extra in (("ewc_batchgrad", []),
                       ("ewc_persample", ["--ewc_per_sample"])):
        jobs.append(("E5", tag, train_cmd(
            "lora_ewc", data, base / "E5_ewc" / tag, ["42"],
            ["--data_fraction", "0.25", *extra]), 6 * 3600))
    jobs.append(("E5", "lora_reference", train_cmd(
        "lora", data, base / "E5_ewc" / "lora_reference", ["42"],
        ["--data_fraction", "0.25"]), None))

    # E6 ------------------------------------------------------------------
    for frac in ("0.1", "0.25", "0.5"):
        for m in METHODS:
            tag = f"{m}_f{int(float(frac) * 100)}"
            jobs.append(("E6", tag, train_cmd(
                m, data, base / "E6_fraction" / tag, SEEDS3,
                ["--data_fraction", frac]), None))

    # E7 ------------------------------------------------------------------
    for r in (4, 16, 32):
        for m in ("lora", "dora"):
            tag = f"{m}_r{r}"
            jobs.append(("E7", tag, train_cmd(
                m, data, base / "E7_rank" / tag, SEEDS3,
                ["--lora_rank", str(r), "--lora_alpha", str(2 * r)]), None))

    # E8 ------------------------------------------------------------------
    for lr in ("1e-5", "3e-5", "5e-5"):
        tag = f"full_ft_lr{lr}"
        jobs.append(("E8", tag, train_cmd(
            "full_ft", data, base / "E8_fullft_lr" / tag, SEEDS3,
            ["--lr", lr]), None))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True,
                    help="Dir with train.conll / dev.conll / test.conll")
    ap.add_argument("--base_dir", default="./experiments/complementary")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--only", nargs="+", default=None,
                    help="Subset of experiments, e.g. E1 E4")
    args = ap.parse_args()

    base = Path(args.base_dir)
    data = base / "data" / "processed"
    data_pst = base / "data" / "processed_pst"
    raw = Path(args.raw_dir)

    pre = []
    for out, extra in ((data, []), (data_pst, ["--per_shard_test"])):
        pre.append([PY, "preprocess.py",
                    "--train_file", str(raw / "train.conll"),
                    "--dev_file", str(raw / "dev.conll"),
                    "--test_file", str(raw / "test.conll"),
                    "--output_dir", str(out), "--n_shards", "3", *extra])

    jobs = build(base, data, data_pst)
    if args.only:
        jobs = [j for j in jobs if j[0] in args.only]

    base.mkdir(parents=True, exist_ok=True)
    script = base / "run_complementary.sh"
    with open(script, "w") as f:
        f.write("#!/bin/bash\nset -e\n")
        for c in pre:
            f.write(" ".join(shlex.quote(x) for x in c) + "\n")
        for exp, tag, c, _ in jobs:
            f.write(f"# {exp}/{tag}\n" + " ".join(shlex.quote(x) for x in c) + "\n")
    print(f"{len(jobs)} training jobs; script written to {script}")

    if not args.execute:
        for exp, tag, c, _ in jobs:
            print(f"[{exp}/{tag}] " + " ".join(c))
        print("\n(dry run -- pass --execute to run)")
        return

    for c in pre:
        subprocess.run(c, check=True)
    for exp, tag, c, timeout in jobs:
        print(f"\n=== {exp}/{tag} ===")
        try:
            subprocess.run(c, check=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            # E5 only: a timeout is itself a reportable outcome.
            print(f"TIMEOUT after {timeout}s: {exp}/{tag}")
            Path(c[c.index('--output_dir') + 1]).mkdir(parents=True, exist_ok=True)
            (Path(c[c.index('--output_dir') + 1]) / "TIMEOUT").write_text(
                str(timeout))

    subprocess.run([PY, "analyze_complementary.py",
                    "--exp_dir", str(base),
                    "--output_dir", str(base / "tex")], check=False)


if __name__ == "__main__":
    main()
