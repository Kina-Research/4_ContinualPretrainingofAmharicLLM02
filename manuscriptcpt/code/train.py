"""
train.py
========
Sequential continual-learning training script for Amharic NER.

Supports: Full-FT, LoRA, DoRA, LoRA+EWC.
Protocol: D1 -> D2 -> D3 (three-stage sequential fine-tuning).

Usage:
    python train.py \
        --method dora \
        --data_dir ./data/processed \
        --output_dir ./outputs/dora \
        --model_name_or_path ibrarap/roberta-base-amharic \
        --seeds 42 123 456

Requirements:
    pip install transformers peft datasets seqeval torch
"""

import argparse
import json
import os
import random
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)

# ── DoRA via PEFT ─────────────────────────────────────────────────────────────
# PEFT >= 0.10 supports use_dora=True inside LoraConfig.
# ─────────────────────────────────────────────────────────────────────────────


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class NERDataset(Dataset):
    """Loads a pre-processed JSONL shard and tokenises on-the-fly."""

    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        label2id: Dict[str, int],
        max_length: int = 256,
        fraction: float = 1.0,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.examples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.examples.append(json.loads(line.strip()))
        # Data-efficiency experiments: seeded random subsample of the shard.
        if fraction < 1.0:
            rng = random.Random(seed)
            n = max(1, int(len(self.examples) * fraction))
            idx = sorted(rng.sample(range(len(self.examples)), n))
            self.examples = [self.examples[i] for i in idx]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        words = ex["tokens"]
        tags  = ex["ner_tags"]
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        word_ids = encoding.word_ids(batch_index=0)
        labels = []
        prev_wid = None
        for wid in word_ids:
            if wid is None:
                labels.append(-100)
            elif wid != prev_wid:
                labels.append(self.label2id.get(tags[wid], -100))
            else:
                labels.append(-100)
            prev_wid = wid

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(labels, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# GPU memory monitor
# ---------------------------------------------------------------------------
class GPUMonitor:
    """Background thread that records peak GPU memory every 250ms."""

    def __init__(self, device_index: int = 0, interval: float = 0.25):
        self.device = device_index
        self.interval = interval
        self.peak_gb = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    def _run(self):
        while not self._stop.is_set():
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
                self.peak_gb = max(self.peak_gb, mem)
            time.sleep(self.interval)


# ---------------------------------------------------------------------------
# EWC regularisation
# ---------------------------------------------------------------------------
class EWCRegulariser:
    """
    Online EWC regulariser.

    After each task, compute the diagonal Fisher Information Matrix (FIM)
    via Monte-Carlo approximation over a small sample of training data.
    During subsequent tasks, penalise changes to important parameters.

    WARNING: This is computationally expensive for large models.
    """

    def __init__(self, model, lambda_ewc: float = 1000.0):
        self.lambda_ewc = lambda_ewc
        self.params = {}
        self.fisher = {}
        self._store_params(model)

    def _store_params(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.params[name] = param.data.clone()

    def update_fisher(self, model, dataloader, device, n_samples: int = 200,
                      per_sample: bool = False):
        """Estimate the diagonal FIM using log-likelihood gradients.

        per_sample=False (original behaviour): squared gradient of the
            *batch-mean* loss, which under-estimates the Fisher.
        per_sample=True: one backward pass per example (the textbook
            empirical Fisher); slower but unbiased.
        Wall-clock time of the estimation is stored in self.last_fisher_time_s.
        """
        t0 = time.time()
        model.eval()
        fisher_tmp = {
            n: torch.zeros_like(p)
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        total = 0
        for batch in dataloader:
            if total >= n_samples:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if per_sample:
                chunks = [(input_ids[i:i + 1], attention_mask[i:i + 1],
                           labels[i:i + 1]) for i in range(input_ids.size(0))]
            else:
                chunks = [(input_ids, attention_mask, labels)]

            for ids_c, mask_c, lab_c in chunks:
                outputs = model(
                    input_ids=ids_c,
                    attention_mask=mask_c,
                    labels=lab_c,
                )
                loss = outputs.loss
                model.zero_grad()
                loss.backward()
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher_tmp[name] += param.grad.data.clone() ** 2
                if per_sample:
                    total += 1
            if not per_sample:
                total += input_ids.size(0)

        for name in fisher_tmp:
            self.fisher[name] = fisher_tmp[name] / max(total, 1)
        self._store_params(model)
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize()
        self.last_fisher_time_s = time.time() - t0

    def penalty(self, model) -> torch.Tensor:
        """Compute the EWC penalty term."""
        loss = torch.tensor(0.0, requires_grad=True)
        for name, param in model.named_parameters():
            if name in self.fisher:
                diff = param - self.params[name]
                loss = loss + (self.fisher[name] * diff ** 2).sum()
        return self.lambda_ewc / 2.0 * loss


# ---------------------------------------------------------------------------
# Training one stage
# ---------------------------------------------------------------------------
def train_one_stage(
    model,
    train_loader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    epochs: int = 3,
    ewc: Optional[EWCRegulariser] = None,
) -> Dict:
    """Train for `epochs` on a single shard."""
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    monitor = GPUMonitor()
    monitor.start()
    start_time = time.time()
    total_steps = 0
    total_samples = 0

    for epoch in range(epochs):
        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            if ewc is not None:
                loss = loss + ewc.penalty(model)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_steps += 1
            total_samples += input_ids.size(0)

    elapsed = time.time() - start_time
    monitor.stop()
    # `peak_gpu_gb` is the original 250 ms *sampled* value of allocated memory
    # (can miss short activation peaks).  `peak_gpu_gb_torch` is the exact
    # allocator high-water mark and should be preferred for memory claims.
    peak_torch = (torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                  if device.type == "cuda" else 0.0)

    return {
        "wall_clock_time_s": elapsed,
        "peak_gpu_gb": monitor.peak_gb,
        "peak_gpu_gb_torch": peak_torch,
        "throughput_sps": total_samples / max(elapsed, 1e-6),
        "steps": total_steps,
    }


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------
def build_model(
    method: str,
    model_name: str,
    num_labels: int,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
) -> torch.nn.Module:
    """Load backbone and optionally wrap with PEFT adapters."""
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    if method in ("lora", "lora_ewc", "dora"):
        use_dora = (method == "dora")
        config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value", "dense"],
            use_dora=use_dora,
            bias="none",
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    return model


# ---------------------------------------------------------------------------
# Count trainable parameters
# ---------------------------------------------------------------------------
def count_trainable_params(model) -> Dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": 100.0 * trainable / max(total, 1),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["full_ft", "lora", "dora", "lora_ewc"],
                        default="dora")
    parser.add_argument("--model_name_or_path", default="ibrarap/roberta-base-amharic")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_shards", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--epochs_per_shard", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--ewc_lambda", type=float, default=1000.0)
    parser.add_argument("--ewc_samples", type=int, default=200)
    # ---- complementary-experiment options (defaults = original protocol) ----
    parser.add_argument("--data_fraction", type=float, default=1.0,
                        help="Fraction of each training shard to use (E6).")
    parser.add_argument("--shard_order", type=int, nargs="+", default=None,
                        help="Permutation of shard ids, e.g. 3 1 2 (E2).")
    parser.add_argument("--per_shard_test", action="store_true",
                        help="Evaluate on disjoint test_part_{j}.jsonl files "
                             "instead of the single shared test set (E3).")
    parser.add_argument("--ewc_per_sample", action="store_true",
                        help="Textbook per-example empirical Fisher (E5).")
    parser.add_argument("--strict_eval", action="store_true",
                        help="Score with seqeval strict IOB2 (default: lenient).")
    args = parser.parse_args()
    order = args.shard_order or list(range(1, args.n_shards + 1))
    assert sorted(order) == list(range(1, args.n_shards + 1)), \
        "--shard_order must be a permutation of 1..n_shards"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load label map
    label_map_path = os.path.join(args.data_dir, "label_map.json")
    with open(label_map_path) as f:
        label_info = json.load(f)
    label2id = {k: int(v) for k, v in label_info["label2id"].items()}
    id2label  = {int(k): v for k, v in label_info["id2label"].items()}
    num_labels = len(label2id)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Load test shard (shared across all stages)
    test_path = os.path.join(args.data_dir, "test_shard_1.jsonl")
    test_dataset = NERDataset(test_path, tokenizer, label2id, args.max_length)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size)

    all_results = []

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"  SEED: {seed}  |  METHOD: {args.method}")
        print(f"{'='*60}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model = build_model(
            method=args.method,
            model_name=args.model_name_or_path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        ).to(device)

        param_info = count_trainable_params(model)
        print(f"  Trainable params: {param_info['trainable_params']:,} "
              f"({param_info['trainable_pct']:.3f}%)")

        ewc = None
        if args.method == "lora_ewc":
            ewc = EWCRegulariser(model, lambda_ewc=args.ewc_lambda)

        seed_results = {
            "seed": seed,
            "method": args.method,
            "param_info": param_info,
            "performance_matrix": [],      # R[i][k] = F1 on test part of the
                                           # k-th *trained* shard after stage i
            "stage_efficiency": [],
            "shard_order": order,
            "data_fraction": args.data_fraction,
            "shared_test_fallback": False,
            "ewc_fisher_time_s": [],
        }

        def test_loader_for_position(k: int):
            """Test loader for the shard trained at position k (1-based)."""
            sid = order[k - 1]
            if args.per_shard_test:
                p = os.path.join(args.data_dir, f"test_part_{sid}.jsonl")
                if not os.path.exists(p):
                    raise FileNotFoundError(
                        f"{p} missing: run preprocess.py --per_shard_test")
            else:
                p = os.path.join(args.data_dir, f"test_shard_{sid}.jsonl")
                if not os.path.exists(p):
                    p = test_path  # original behaviour: shared test set
                    seed_results["shared_test_fallback"] = True
            ds = NERDataset(p, tokenizer, label2id, args.max_length)
            return DataLoader(ds, batch_size=args.batch_size)

        # Evaluate before any training (stage 0)
        from functools import partial
        from evaluate_ner import evaluate_ner_model as _eval  # imported below
        evaluate_ner_model = partial(_eval, strict=args.strict_eval)
        stage0_scores = []
        for j in range(1, args.n_shards + 1):
            loader_j = test_loader_for_position(j)
            score = evaluate_ner_model(model, loader_j, id2label, device)
            stage0_scores.append(score)
            print(f"  Stage 0 | Shard {j} | F1: {score:.4f}")
        seed_results["performance_matrix"].append(stage0_scores)

        total_wall_clock = 0.0
        total_peak_gpu   = 0.0
        total_peak_gpu_torch = 0.0

        for stage in range(1, args.n_shards + 1):
            print(f"\n  --- Stage {stage}: Training on D{order[stage - 1]} ---")
            shard_path = os.path.join(args.data_dir,
                                      f"train_shard_{order[stage - 1]}.jsonl")
            train_ds = NERDataset(shard_path, tokenizer, label2id,
                                  args.max_length,
                                  fraction=args.data_fraction, seed=seed)
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True
            )

            # Optimizer and scheduler
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            total_steps = args.epochs_per_shard * len(train_loader)
            warmup_steps = int(0.10 * total_steps)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )

            # Train
            eff = train_one_stage(
                model, train_loader, optimizer, scheduler, device,
                epochs=args.epochs_per_shard, ewc=ewc,
            )
            total_wall_clock += eff["wall_clock_time_s"]
            total_peak_gpu = max(total_peak_gpu, eff["peak_gpu_gb"])
            total_peak_gpu_torch = max(total_peak_gpu_torch,
                                       eff["peak_gpu_gb_torch"])
            seed_results["stage_efficiency"].append(eff)

            # Update EWC Fisher after each stage
            if ewc is not None:
                print("  Updating EWC Fisher...")
                ewc.update_fisher(model, train_loader, device,
                                  n_samples=args.ewc_samples,
                                  per_sample=args.ewc_per_sample)
                seed_results["ewc_fisher_time_s"].append(ewc.last_fisher_time_s)
                print(f"  Fisher estimation: {ewc.last_fisher_time_s:.1f}s")

            # Evaluate on all shards
            stage_scores = []
            for j in range(1, args.n_shards + 1):
                loader_j = test_loader_for_position(j)
                score = evaluate_ner_model(model, loader_j, id2label, device)
                stage_scores.append(score)
                print(f"  Stage {stage} | Shard {j} | F1: {score:.4f}")
            seed_results["performance_matrix"].append(stage_scores)

        seed_results["total_wall_clock_s"] = total_wall_clock
        seed_results["peak_gpu_gb"] = total_peak_gpu
        seed_results["peak_gpu_gb_torch"] = total_peak_gpu_torch
        all_results.append(seed_results)

        # Save per-seed checkpoint
        out_seed_dir = os.path.join(args.output_dir, f"seed_{seed}")
        os.makedirs(out_seed_dir, exist_ok=True)
        model.save_pretrained(out_seed_dir)
        tokenizer.save_pretrained(out_seed_dir)

    # Save all results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
