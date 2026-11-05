# Amharic-Centric LLMs: Continual Pre-training Experiments

## Overview
Reproducibility package for:
**"Amharic-Centric Large Language Models: Data-Efficient Continual Pre-training
and Incremental Adaptation for Low-Resource Natural Language Processing"**

Authors: Kindie Alebachew Tsega, Durga Prasad Sharma, Mesay Samuel Gondere,
Mohammed Abebe Yimer

---

## Repository Structure
```
code/
├── process_parallel_data.py       # English-Amharic bitext curation, filtering & shard split
├── generate_synthetic_parallel.py # Transparent synthetic parallel data generator
├── generate_publication_figures.py# Publication-quality figures generator (Figs 1-6 PDF+PNG)
├── preprocess.py                  # NER dataset preprocessing + Ethiopic normalisation
├── train.py                       # Sequential CL training (Full-FT / LoRA / DoRA / LoRA+EWC)
├── evaluate_ner.py                # NER evaluation, BWT/FWT/SG computation, t-tests
├── compare_methods.py             # Cross-method table generation
├── plot_results.py                # Legacy plot generator
├── ablation_configs.py            # Ablation experiment configuration generator
├── run_all_experiments.py         # Master runner (end-to-end pipeline)
├── analysis_from_summary.py       # Re-analysis of reported summary stats (no training)
├── complementary_experiments.py   # E1-E8 runner (dry-run by default)
├── analyze_complementary.py       # LaTeX tables from E1-E8 results.json
└── requirements.txt               # Python dependencies with pinned versions
```

---

## Prerequisites

### Hardware
- 1× NVIDIA GPU with ≥ 4 GB VRAM (experiments used up to 3.6 GB peak)
- CPU-only training is supported but will be ~10× slower

### Software
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate          # Linux/Mac
venv\Scripts\activate             # Windows

# Install dependencies
pip install -r requirements.txt
```

### Model
The base model `ibrarap/roberta-base-amharic` is auto-downloaded from
Hugging Face Hub on first use.

---

## Datasets

### 1. English--Amharic Parallel Corpus (EthioEnAm-Parallel)
An open-source bitext corpus consisting of **19,386 verified sentence pairs** curated from:
- **OPUS-100** (`Helsinki-NLP/opus-100` am-en; Tiedemann 2012, Zhang et al. 2020): 6,402 clean pairs across software localizations and technical texts.
- **KCorpus Multi-Domain Reasoning Benchmarks**: 12,984 pairs spanning arithmetic word problems, logical inference, ethics, and multi-turn dialogue.

**Preprocessing & Slicing**:
```bash
python process_parallel_data.py
```
Outputs in `data/parallel_en_am/processed/`:
- `train_full.jsonl` (15,510 pairs), `valid.jsonl` (1,938 pairs), `test.jsonl` (1,938 pairs)
- Three sequential CL shards: `train_shard_1.jsonl`, `train_shard_2.jsonl`, `train_shard_3.jsonl` (5,170 pairs each)
- TSV format export: `parallel_en_am.tsv`
- Full statistical profile: `parallel_corpus_stats.json`

### 2. Auxiliary Synthetic Parallel Generator
Generates template-slotted synthetic parallel pairs for augmentation investigations, with explicit tagging:
```bash
python generate_synthetic_parallel.py
```
Output: `data/parallel_en_am/processed/synthetic_parallel_pairs.jsonl` (782 verified synthetic pairs).

### 3. Amharic NER Corpus
The Amharic NER corpus (Yohannes & Amagasa, ACM SAC 2022) annotates five
entity types in IOB2 format: LOC, ORG, PER, TIME, TTL.

**Data availability**: Contact the original authors.  
**Expected format**: CoNLL-2003 style (token + label per line, blank lines
between sentences).

---

## Step-by-Step Reproduction

### Step 1: Preprocess
```bash
python preprocess.py \
    --train_file ./data/raw/train.conll \
    --dev_file   ./data/raw/dev.conll \
    --test_file  ./data/raw/test.conll \
    --output_dir ./data/processed \
    --n_shards 3
```
Outputs:
- `train_shard_1.jsonl`, `train_shard_2.jsonl`, `train_shard_3.jsonl`
- `dev_shard_1.jsonl`, `test_shard_1.jsonl`
- `label_map.json`
- Fraction subsets: `train_shard_N_fracM.jsonl`

### Step 2: Train (all methods, 3 seeds)
```bash
# DoRA (principal method)
python train.py --method dora \
    --data_dir ./data/processed \
    --output_dir ./outputs/dora \
    --seeds 42 123 456

# LoRA
python train.py --method lora \
    --data_dir ./data/processed \
    --output_dir ./outputs/lora \
    --seeds 42 123 456

# Full Fine-Tuning
python train.py --method full_ft \
    --data_dir ./data/processed \
    --output_dir ./outputs/full_ft \
    --seeds 42 123 456

# LoRA + EWC (expected to timeout for large models)
python train.py --method lora_ewc \
    --data_dir ./data/processed \
    --output_dir ./outputs/lora_ewc \
    --seeds 42 123 456
```

### Step 3: Evaluate and compare
```bash
python compare_methods.py \
    --results_dir ./outputs \
    --output_dir  ./tables
```

### Step 4: Generate figures
```bash
python plot_results.py \
    --results_dir ./outputs \
    --output_dir  ./figures
```

### Step 5: Ablation studies
```bash
# Generate all ablation configs (dry run)
python ablation_configs.py --base_dir ./experiments

# Execute a specific ablation group
python ablation_configs.py --base_dir ./experiments \
    --ablation A5_lora_rank --execute
```

### Or: Run Everything
```bash
python run_all_experiments.py \
    --train_file ./data/raw/train.conll \
    --dev_file   ./data/raw/dev.conll \
    --test_file  ./data/raw/test.conll \
    --base_dir   ./experiments
```

---

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base model | `ibrarap/roberta-base-amharic` |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | query, key, value, dense |
| Learning rate | 3 × 10⁻⁴ |
| Batch size | 8 |
| Epochs per shard | 3 |
| Warmup | 10% of steps |
| Weight decay | 0.01 |
| Max sequence length | 256 |
| Seeds | 42, 123, 456 |
| EWC λ | 1000 |
| EWC samples | 200 |

---

## Expected Results

| Method | Params (%) | Peak GPU (GB) | Time (s) | Mean F1 |
|--------|-----------|--------------|----------|---------|
| Full-FT | 100.000 | 3.604 | 4,891.9 | 0.6616 |
| LoRA | 1.199 | 2.669 | 5,715.7 | 0.7461 |
| DoRA | 1.273 | 2.348 | 10,864.2 | 0.7458 |
| LoRA+EWC | — | — | Timeout | — |

BWT (DoRA): +0.0513 ± 0.0163  
FWT (DoRA): +0.0325 ± 0.0242  
SG (DoRA):  -0.0513 ± 0.0163

---

## Experiment Tracking

Set `WANDB_PROJECT=amharic-cl-ner` and pass `--use_wandb` (not yet
implemented; add `wandb.init(...)` calls in `train.py`).

Alternatively, all results are logged to `outputs/<method>/results.json`
and can be post-processed by `compare_methods.py`.

---

## Deviations from Protocol

| Deviation | Reason | Impact |
|-----------|--------|--------|
| LoRA+EWC incomplete | Fisher matrix estimation timed out | EWC results partial |
| n=3 seeds | Resource constraints | Reduces statistical power |
| Single corpus source | NER corpus from one source | Limits domain diversity |
| Single shared test file | `preprocess.py` wrote only `test_shard_1.jsonl`; `train.py` fell back to it for every shard column | BWT may not measure forgetting; use `--per_shard_test` (E3) |
| Data-fraction runs | `run_all_experiments.py` never passed a fraction to `train.py` | Fixed: `--data_fraction` (E6) |
| Scorer mode | `seqeval` default mode, not strict IOB2 | `--strict_eval` available |
| Memory metric | 250 ms sampling of `memory_allocated` | Exact `max_memory_allocated` now logged as `peak_gpu_gb_torch` |
| Shared LR 3e-4 | Adapter-scale LR also used for Full-FT | Sweep in E8 |

---

## Complementary experiments (E1-E8)

```bash
# dry run: prints/writes the commands only
python complementary_experiments.py --raw_dir ./data/raw --base_dir ./experiments/complementary
# run everything, or a subset
python complementary_experiments.py --raw_dir ./data/raw --execute --only E1 E4
# aggregate to LaTeX fragments (cells of unrun experiments print as "--")
python analyze_complementary.py --exp_dir ./experiments/complementary
```

No E1-E8 result is reported in the manuscript yet. Re-analysis of the
already-reported numbers: `python analysis_from_summary.py` (needs `scipy`).

---

## Citation

```bibtex
@article{tsega2024amharic,
  title   = {Amharic-Centric Large Language Models: Data-Efficient
             Continual Pre-training and Incremental Adaptation for
             Low-Resource Natural Language Processing},
  author  = {Tsega, Kindie Alebachew and Sharma, Durga Prasad and
             Gondere, Mesay Samuel and Yimer, Mohammed Abebe},
  journal = {IEEE Access},
  year    = {2024},
}
```

---

## License
Code: MIT License.  
Dataset: Subject to original dataset license (Yohannes & Amagasa, 2022).
