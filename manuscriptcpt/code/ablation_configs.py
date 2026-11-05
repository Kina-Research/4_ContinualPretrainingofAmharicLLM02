"""
ablation_configs.py
===================
Configuration generator for ablation experiments.

Ablation studies covered:
  A1. Weight decomposition: DoRA vs LoRA (same hyperparameters)
  A2. Explicit regularisation: LoRA vs LoRA+EWC
  A3. Shard size: vary number of shards (2, 3, 5)
  A4. Training data quantity: 10%, 25%, 50%, 100%
  A5. LoRA rank: r = 4, 8, 16, 32
  A6. Learning rate: 1e-4, 3e-4, 5e-4

Each ablation produces a config dict and a run command.
Run with:
    python ablation_configs.py --execute --base_dir ./experiments
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Ablation configuration definitions
# ---------------------------------------------------------------------------
ABLATIONS = {
    "A1_decomposition": [
        {"method": "lora", "tag": "lora_no_decomp"},
        {"method": "dora", "tag": "dora_decomp"},
    ],
    "A2_regularisation": [
        {"method": "lora",     "tag": "lora_no_ewc"},
        {"method": "lora_ewc", "tag": "lora_with_ewc"},
    ],
    "A3_shard_count": [
        {"n_shards": 2, "tag": "shards_2"},
        {"n_shards": 3, "tag": "shards_3"},
        {"n_shards": 5, "tag": "shards_5"},
    ],
    "A4_data_fraction": [
        {"data_fraction": 0.10, "tag": "frac_10"},
        {"data_fraction": 0.25, "tag": "frac_25"},
        {"data_fraction": 0.50, "tag": "frac_50"},
        {"data_fraction": 1.00, "tag": "frac_100"},
    ],
    "A5_lora_rank": [
        {"lora_rank": 4,  "tag": "rank_4"},
        {"lora_rank": 8,  "tag": "rank_8"},
        {"lora_rank": 16, "tag": "rank_16"},
        {"lora_rank": 32, "tag": "rank_32"},
    ],
    "A6_learning_rate": [
        {"lr": 1e-4, "tag": "lr_1e4"},
        {"lr": 3e-4, "tag": "lr_3e4"},
        {"lr": 5e-4, "tag": "lr_5e4"},
    ],
}

# Default arguments (shared across ablations)
DEFAULTS = {
    "method":           "dora",
    "n_shards":         3,
    "data_fraction":    1.00,
    "lora_rank":        8,
    "lora_alpha":       16,
    "lora_dropout":     0.05,
    "lr":               3e-4,
    "batch_size":       8,
    "epochs":           3,
    "seeds":            [42],          # single seed for ablations
    "model_name":       "ibrarap/roberta-base-amharic",
}

# Hypotheses for each ablation
HYPOTHESES = {
    "A1_decomposition":
        "H_A1: DoRA's weight decomposition yields higher FWT and lower GPU "
        "memory than LoRA under identical conditions.",
    "A2_regularisation":
        "H_A2: Explicit Fisher regularisation (EWC) provides additional "
        "forgetting mitigation over LoRA alone, at the cost of compute.",
    "A3_shard_count":
        "H_A3: More shards (finer increments) increase cumulative "
        "adaptation quality but may increase per-stage forgetting risk.",
    "A4_data_fraction":
        "H_A4: PEFT methods reach comparable performance at lower data "
        "fractions than Full-FT.",
    "A5_lora_rank":
        "H_A5: Higher rank r increases expressiveness and final accuracy "
        "but also increases trainable parameters and memory.",
    "A6_learning_rate":
        "H_A6: A learning rate of 3e-4 is near-optimal for DoRA on "
        "the Amharic NER task under the given batch and epoch budget.",
}


def build_train_command(
    config: dict,
    data_dir: str,
    output_dir: str,
    py: str = sys.executable,
) -> list:
    """Build a train.py command from a config dict."""
    seeds = config.get("seeds", DEFAULTS["seeds"])
    cmd = [
        py, "train.py",
        "--method",             config.get("method",      DEFAULTS["method"]),
        "--model_name_or_path", config.get("model_name",  DEFAULTS["model_name"]),
        "--data_dir",           data_dir,
        "--output_dir",         output_dir,
        "--n_shards",           str(config.get("n_shards", DEFAULTS["n_shards"])),
        "--seeds",              *[str(s) for s in seeds],
        "--epochs_per_shard",   str(config.get("epochs",    DEFAULTS["epochs"])),
        "--batch_size",         str(config.get("batch_size", DEFAULTS["batch_size"])),
        "--lr",                 str(config.get("lr",         DEFAULTS["lr"])),
        "--lora_rank",          str(config.get("lora_rank",  DEFAULTS["lora_rank"])),
        "--lora_alpha",         str(config.get("lora_alpha", DEFAULTS["lora_alpha"])),
        "--lora_dropout",       str(config.get("lora_dropout", DEFAULTS["lora_dropout"])),
    ]
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir",   default="./experiments")
    parser.add_argument("--data_dir",   default=None)
    parser.add_argument("--execute",    action="store_true",
                        help="Execute experiments; otherwise just print commands")
    parser.add_argument("--ablation",   default=None,
                        help="Run a specific ablation group (e.g. A1_decomposition)")
    args = parser.parse_args()

    base   = Path(args.base_dir)
    data_d = args.data_dir or str(base / "data" / "processed")

    ablations_to_run = (
        {args.ablation: ABLATIONS[args.ablation]}
        if args.ablation and args.ablation in ABLATIONS
        else ABLATIONS
    )

    all_cmds = []
    for group_name, variants in ablations_to_run.items():
        print(f"\n{'='*60}")
        print(f"  Ablation: {group_name}")
        print(f"  Hypothesis: {HYPOTHESES[group_name]}")
        print(f"{'='*60}")

        for variant in variants:
            # Merge with defaults
            cfg = {**DEFAULTS, **variant}
            tag = variant.get("tag", "run")
            out_dir = str(base / "ablations" / group_name / tag)
            os.makedirs(out_dir, exist_ok=True)

            # Save config
            cfg_path = os.path.join(out_dir, "config.json")
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2)

            cmd = build_train_command(cfg, data_d, out_dir)
            print(f"\n  Tag: {tag}")
            print(f"  Config: {cfg}")
            print(f"  Command: {' '.join(cmd)}")
            all_cmds.append({"tag": tag, "group": group_name, "cmd": cmd})

            if args.execute:
                subprocess.run(cmd, check=True)

    # Save all commands to a shell script
    script_path = str(base / "run_ablations.sh")
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated ablation runner\n\n")
        for item in all_cmds:
            f.write(f"# {item['group']} / {item['tag']}\n")
            f.write(" ".join(item["cmd"]) + "\n\n")
    print(f"\nAll ablation commands saved to: {script_path}")

    # Also produce result template CSVs
    template_dir = str(base / "ablations" / "_templates")
    os.makedirs(template_dir, exist_ok=True)
    import csv
    for group_name, variants in ablations_to_run.items():
        tags = [v.get("tag", "run") for v in variants]
        out_csv = os.path.join(template_dir, f"{group_name}_results.csv")
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "tag", "mean_final_f1", "std_final_f1", "mean_bwt", "mean_fwt",
                "peak_gpu_gb", "wall_clock_s", "trainable_pct"
            ])
            writer.writeheader()
            for tag in tags:
                writer.writerow({
                    "tag": tag,
                    "mean_final_f1": "PENDING",
                    "std_final_f1":  "PENDING",
                    "mean_bwt":      "PENDING",
                    "mean_fwt":      "PENDING",
                    "peak_gpu_gb":   "PENDING",
                    "wall_clock_s":  "PENDING",
                    "trainable_pct": "PENDING",
                })
        print(f"  Template: {out_csv}")


if __name__ == "__main__":
    main()
