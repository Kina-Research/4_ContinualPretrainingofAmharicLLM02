"""
run_all_experiments.py
======================
Master experiment runner. Executes the complete experimental pipeline:

  1. Preprocess Amharic NER dataset into 3 sequential shards.
  2. Train all four methods (Full-FT, LoRA, DoRA, LoRA+EWC) across 3 seeds.
  3. Data-efficiency experiments (10%, 25%, 50%, 100% fractions).
  4. Aggregate results and compute CL metrics.
  5. Generate comparison tables (CSV).
  6. Generate all manuscript figures.

Usage:
    python run_all_experiments.py \
        --train_file  ./data/raw/amharic_ner_train.conll \
        --dev_file    ./data/raw/amharic_ner_dev.conll \
        --test_file   ./data/raw/amharic_ner_test.conll \
        --base_dir    ./experiments \
        --model_name  ibrarap/roberta-base-amharic \
        [--dry_run]

The --dry_run flag prints the commands without executing them.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list, dry_run: bool = False):
    """Run a subprocess command, or print it if dry_run is True."""
    cmd_str = " ".join(cmd)
    print(f"\n  $ {cmd_str}")
    if not dry_run:
        result = subprocess.run(cmd, check=True)
        return result
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Master experiment runner for Amharic CL-NER study."
    )
    parser.add_argument("--train_file",  required=True)
    parser.add_argument("--dev_file",    required=True)
    parser.add_argument("--test_file",   required=True)
    parser.add_argument("--base_dir",    default="./experiments")
    parser.add_argument("--model_name",
                        default="ibrarap/roberta-base-amharic")
    parser.add_argument("--n_shards",    type=int, default=3)
    parser.add_argument("--seeds",       nargs="+", type=int,
                        default=[42, 123, 456])
    parser.add_argument("--epochs",      type=int, default=3)
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--dry_run",     action="store_true")
    args = parser.parse_args()

    base   = Path(args.base_dir)
    data_d = base / "data" / "processed"
    out_d  = base / "outputs"
    fig_d  = base / "figures"
    tab_d  = base / "tables"

    for d in [data_d, out_d, fig_d, tab_d]:
        d.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    # -----------------------------------------------------------------------
    # STEP 1: Preprocessing
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 1: Preprocessing")
    print("="*60)
    run([
        py, "preprocess.py",
        "--train_file", args.train_file,
        "--dev_file",   args.dev_file,
        "--test_file",  args.test_file,
        "--output_dir", str(data_d),
        "--n_shards",   str(args.n_shards),
    ], dry_run=args.dry_run)

    # -----------------------------------------------------------------------
    # STEP 2: Train all methods
    # -----------------------------------------------------------------------
    methods = ["full_ft", "lora", "dora", "lora_ewc"]
    print("\n" + "="*60)
    print("STEP 2: Training all methods")
    print("="*60)

    for method in methods:
        method_out = out_d / method
        method_out.mkdir(parents=True, exist_ok=True)
        print(f"\n  --- Method: {method} ---")
        run([
            py, "train.py",
            "--method",            method,
            "--model_name_or_path", args.model_name,
            "--data_dir",          str(data_d),
            "--output_dir",        str(method_out),
            "--n_shards",          str(args.n_shards),
            "--seeds",             *[str(s) for s in args.seeds],
            "--epochs_per_shard",  str(args.epochs),
            "--batch_size",        str(args.batch_size),
            "--lr",                str(args.lr),
        ], dry_run=args.dry_run)

    # -----------------------------------------------------------------------
    # STEP 3: Data-efficiency experiments (full_ft, lora, dora only)
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 3: Data-efficiency experiments")
    print("="*60)

    for frac in [10, 25, 50]:
        for method in ["full_ft", "lora", "dora"]:
            frac_out = out_d / method / f"frac{frac}"
            frac_out.mkdir(parents=True, exist_ok=True)
            print(f"\n  --- Method: {method}  Fraction: {frac}% ---")
            # train.py subsamples every shard with --data_fraction (seeded).
            run([
                py, "train.py",
                "--method",            method,
                "--model_name_or_path", args.model_name,
                "--data_dir",          str(data_d),
                "--output_dir",        str(frac_out),
                "--n_shards",          str(args.n_shards),
                "--seeds",             str(args.seeds[0]),
                "--epochs_per_shard",  str(args.epochs),
                "--batch_size",        str(args.batch_size),
                "--lr",                str(args.lr),
                "--data_fraction",     str(frac / 100.0),
            ], dry_run=args.dry_run)

    # -----------------------------------------------------------------------
    # STEP 4: Aggregate and compare
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 4: Cross-method comparison and tables")
    print("="*60)
    run([
        py, "compare_methods.py",
        "--results_dir", str(out_d),
        "--output_dir",  str(tab_d),
    ], dry_run=args.dry_run)

    # -----------------------------------------------------------------------
    # STEP 5: Generate figures
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 5: Generating figures")
    print("="*60)
    run([
        py, "plot_results.py",
        "--results_dir", str(out_d),
        "--output_dir",  str(fig_d),
    ], dry_run=args.dry_run)

    print("\n" + "="*60)
    print("ALL STEPS COMPLETE.")
    print(f"  Tables:  {tab_d}")
    print(f"  Figures: {fig_d}")
    print(f"  Outputs: {out_d}")
    print("="*60)


if __name__ == "__main__":
    main()
