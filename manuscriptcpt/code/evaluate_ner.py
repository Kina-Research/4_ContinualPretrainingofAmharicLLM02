"""
evaluate_ner.py
===============
NER evaluation module and standalone evaluation script.

Provides:
  - evaluate_ner_model()  -- called during training
  - compute_cl_metrics()  -- computes BWT, FWT, SG from performance matrix
  - Standalone CLI for evaluating a saved checkpoint

Usage (standalone):
    python evaluate_ner.py \
        --model_dir ./outputs/dora/seed_42 \
        --data_dir  ./data/processed \
        --results_json ./outputs/dora/results.json \
        --output_csv   ./outputs/dora/cl_metrics.csv
"""

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from seqeval.metrics import classification_report, f1_score
from seqeval.scheme import IOB2
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from peft import PeftModel


# ---------------------------------------------------------------------------
# Entity-level NER evaluation
# ---------------------------------------------------------------------------
def evaluate_ner_model(
    model,
    dataloader: DataLoader,
    id2label: Dict[int, str],
    device: torch.device,
    strict: bool = False,
) -> float:
    """
    Run inference and return entity-level macro-F1 (seqeval strict IOB2).

    Returns:
        Macro-F1 as a float.
    """
    model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits
            preds   = torch.argmax(logits, dim=-1)

            for i in range(labels.size(0)):
                true_seq, pred_seq = [], []
                for t, p in zip(labels[i].tolist(), preds[i].tolist()):
                    if t == -100:
                        continue
                    true_seq.append(id2label.get(t, "O"))
                    pred_seq.append(id2label.get(p, "O"))
                all_true.append(true_seq)
                all_pred.append(pred_seq)

    # strict=False: seqeval default (conlleval-compatible) matching, as used
    # in the original runs.  strict=True: strict IOB2 scheme.
    if strict:
        return f1_score(all_true, all_pred, average="macro", zero_division=0,
                        mode="strict", scheme=IOB2)
    score = f1_score(all_true, all_pred, average="macro", zero_division=0)
    return score


def evaluate_ner_full(
    model,
    dataloader: DataLoader,
    id2label: Dict[int, str],
    device: torch.device,
) -> Dict:
    """
    Return full seqeval classification report as dict.
    """
    model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = torch.argmax(outputs.logits, dim=-1)

            for i in range(labels.size(0)):
                true_seq, pred_seq = [], []
                for t, p in zip(labels[i].tolist(), preds[i].tolist()):
                    if t == -100:
                        continue
                    true_seq.append(id2label.get(t, "O"))
                    pred_seq.append(id2label.get(p, "O"))
                all_true.append(true_seq)
                all_pred.append(pred_seq)

    report = classification_report(
        all_true, all_pred, output_dict=True, zero_division=0
    )
    return report


# ---------------------------------------------------------------------------
# Continual Learning metrics from performance matrix
# ---------------------------------------------------------------------------
def compute_cl_metrics(
    R: List[List[float]],
) -> Dict[str, float]:
    """
    Compute BWT, FWT, and Stability Gap (SG) from a T×T performance matrix.

    Args:
        R: R[i][j] = macro-F1 on shard j after training through stage i.
           R[0][j] = zero-shot score before any training.
           Rows: stages 0..T; Cols: shards 1..T (0-indexed in the list).

    Returns:
        Dict with bwt, fwt, sg, and per-task forgetting F_k.
    """
    T = len(R) - 1  # number of training stages

    # BWT: average change on prior tasks after final stage
    bwt_vals = []
    for j in range(T - 1):             # shards 0..(T-2) in 0-indexed
        bwt_vals.append(R[T][j] - R[j + 1][j])   # R[T][j] - R[j+1][j]
    bwt = np.mean(bwt_vals) if bwt_vals else 0.0

    # FWT: average improvement on future shards before seeing them
    fwt_vals = []
    for j in range(1, T):             # shards 1..(T-1) in 0-indexed
        fwt_vals.append(R[j][j] - R[0][j])
    fwt = np.mean(fwt_vals) if fwt_vals else 0.0

    sg = -bwt

    # Per-task forgetting F_k
    forgetting = {}
    for j in range(T - 1):
        best_before_final = max(R[t][j] for t in range(j + 1, T + 1))
        fk = best_before_final - R[T][j]
        forgetting[f"F_{j+1}"] = fk

    return {
        "bwt": bwt,
        "fwt": fwt,
        "sg":  sg,
        "per_task_forgetting": forgetting,
    }


# ---------------------------------------------------------------------------
# Welch t-test across seeds
# ---------------------------------------------------------------------------
def welch_t_test_vs_reference(
    method_scores: List[float],
    reference_scores: List[float],
) -> Tuple[float, float]:
    """
    Perform a two-sample Welch's t-test.

    Returns:
        (t_statistic, p_value)
    """
    from scipy import stats
    t_stat, p_val = stats.ttest_ind(
        method_scores, reference_scores, equal_var=False
    )
    return float(t_stat), float(p_val)


# ---------------------------------------------------------------------------
# Aggregate results across seeds
# ---------------------------------------------------------------------------
def aggregate_results(results_list: List[Dict]) -> Dict:
    """
    Aggregate per-seed results into means, standard deviations,
    95% CIs, and CL metrics.
    """
    final_f1s = []
    bwts, fwts, sgs = [], [], []

    for res in results_list:
        pm = res["performance_matrix"]
        # Final F1 = average of R[T][0..T-1] (final stage, all shards)
        T = len(pm) - 1
        final_f1s.append(np.mean(pm[T]))
        cl = compute_cl_metrics(pm)
        bwts.append(cl["bwt"])
        fwts.append(cl["fwt"])
        sgs.append(cl["sg"])

    n = len(final_f1s)
    from scipy.stats import t as t_dist
    alpha = 0.05
    se_f1 = np.std(final_f1s, ddof=1) / (n ** 0.5) if n > 1 else 0.0
    t_crit = t_dist.ppf(1 - alpha / 2, df=max(n - 1, 1))
    ci_lo  = np.mean(final_f1s) - t_crit * se_f1
    ci_hi  = np.mean(final_f1s) + t_crit * se_f1

    return {
        "mean_final_f1":  float(np.mean(final_f1s)),
        "std_final_f1":   float(np.std(final_f1s, ddof=1)),
        "ci_95":          [float(ci_lo), float(ci_hi)],
        "mean_bwt":       float(np.mean(bwts)),
        "std_bwt":        float(np.std(bwts, ddof=1)),
        "mean_fwt":       float(np.mean(fwts)),
        "std_fwt":        float(np.std(fwts, ddof=1)),
        "mean_sg":        float(np.mean(sgs)),
        "n_seeds":        n,
    }


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved checkpoint on Amharic NER"
    )
    parser.add_argument("--model_dir",    required=True)
    parser.add_argument("--data_dir",     required=True)
    parser.add_argument("--results_json", required=True,
                        help="Path to results.json from train.py")
    parser.add_argument("--output_csv",   default="cl_metrics.csv")
    parser.add_argument("--batch_size",   type=int, default=8)
    parser.add_argument("--max_length",   type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_map_path = os.path.join(args.data_dir, "label_map.json")
    with open(label_map_path) as f:
        label_info = json.load(f)
    label2id = {k: int(v) for k, v in label_info["label2id"].items()}
    id2label  = {int(k): v for k, v in label_info["id2label"].items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    try:
        model = PeftModel.from_pretrained(
            AutoModelForTokenClassification.from_pretrained(
                args.model_dir,
                num_labels=len(label2id),
                id2label=id2label,
                label2id=label2id,
            ),
            args.model_dir,
        ).to(device)
    except Exception:
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_dir,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
        ).to(device)

    # Full entity-level report on shared test set
    from train import NERDataset
    test_path = os.path.join(args.data_dir, "test_shard_1.jsonl")
    test_ds   = NERDataset(test_path, tokenizer, label2id, args.max_length)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    report = evaluate_ner_full(model, test_loader, id2label, device)
    print("\n=== Entity-level Classification Report ===")
    for entity, metrics in report.items():
        if isinstance(metrics, dict):
            print(f"  {entity:15s}: P={metrics['precision']:.3f}  "
                  f"R={metrics['recall']:.3f}  F1={metrics['f1-score']:.3f}")

    # Aggregate CL metrics from results.json
    with open(args.results_json) as f:
        results_list = json.load(f)

    agg = aggregate_results(results_list)
    print("\n=== Aggregated CL Metrics ===")
    for k, v in agg.items():
        print(f"  {k}: {v}")

    # Statistical tests vs DoRA (when comparing multiple method result files)
    # Saved for cross-method comparison in compare_methods.py

    # Save to CSV
    rows = [{"metric": k, "value": v} for k, v in agg.items()]
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved metrics to {args.output_csv}")


if __name__ == "__main__":
    main()
