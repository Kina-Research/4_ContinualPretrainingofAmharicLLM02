"""
preprocess.py
=============
Dataset preprocessing and Ethiopic script normalisation pipeline for
Amharic NER continual-learning experiments.

Usage:
    python preprocess.py --input_dir <path> --output_dir <path> [--n_shards 3]

Requirements:
    pip install datasets transformers seqeval
"""

import argparse
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Tuple

# ---------------------------------------------------------------------------
# 1. Ethiopic script normalisation
# ---------------------------------------------------------------------------
# Mapping of homophonous Ethiopic characters to a canonical form.
# Only characters that are fully interchangeable in standard Amharic are
# merged; characters with distinct phonological status are preserved.
ETHIOPIC_NORMALISATION_MAP = {
    # Group 1: he-variants
    "\u1200": "\u1200",  # ሀ (reference)
    "\u1210": "\u1200",  # ሐ -> ሀ
    "\u1280": "\u1200",  # ሠ — keep separate (different sound in some dialects)
    "\u1203": "\u1203",  # ሃ (reference)
    "\u1213": "\u1203",  # ሓ -> ሃ
    # Group 2: ae-variants
    "\u1226": "\u1226",  # ሦ (reference, rare)
    # Group 3: ts-variants
    "\u1338": "\u1338",  # ፸ reference
    "\u1340": "\u1338",  # ፠ -> ፸ (edge case)
}

# Lamalign / ጸ vs ፀ are distinct and must NOT be merged.
# The mapping above is conservative; expand with linguistic validation.

def normalise_ethiopic(text: str) -> str:
    """Apply Ethiopic-script character normalisation."""
    # Unicode NFC first
    text = unicodedata.normalize("NFC", text)
    result = []
    for ch in text:
        result.append(ETHIOPIC_NORMALISATION_MAP.get(ch, ch))
    return "".join(result)


# ---------------------------------------------------------------------------
# 2. Text cleaning
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Remove HTML markup, control characters, and excessive whitespace."""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove non-printable control characters (keep Ethiopic Unicode range)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 3. IOB2 label alignment (subword -> word mapping)
# ---------------------------------------------------------------------------
def align_labels_to_tokens(
    tokenizer,
    words: List[str],
    labels: List[str],
    max_length: int = 256,
    label2id: Dict[str, int] = None,
) -> Dict:
    """
    Tokenise a word-labelled sentence and align IOB2 labels to subwords.

    Rules:
      - The first subword of each word receives the word's label.
      - Subsequent subwords receive the -100 ignore index.
      - Special tokens ([CLS], [SEP]) receive the -100 ignore index.

    Returns a dict compatible with Hugging Face datasets.
    """
    if label2id is None:
        label2id = {}

    tokenized = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_offsets_mapping=False,
    )

    word_ids = tokenized.word_ids()
    aligned_labels = []
    previous_word_id = None
    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(-100)
        elif word_id != previous_word_id:
            # First subword: assign the true label
            label_str = labels[word_id]
            aligned_labels.append(label2id.get(label_str, -100))
        else:
            # Subsequent subwords: ignore
            aligned_labels.append(-100)
        previous_word_id = word_id

    tokenized["labels"] = aligned_labels
    return tokenized


# ---------------------------------------------------------------------------
# 4. Deduplication
# ---------------------------------------------------------------------------
def deduplicate(examples: List[Dict], text_key: str = "tokens") -> List[Dict]:
    """Exact-match deduplication based on the token sequence."""
    seen = set()
    unique = []
    for ex in examples:
        key = " ".join(ex[text_key])
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    return unique


# ---------------------------------------------------------------------------
# 5. Sequential shard partitioning
# ---------------------------------------------------------------------------
def partition_into_shards(
    examples: List[Dict],
    n_shards: int = 3,
    seed: int = 42,
) -> List[List[Dict]]:
    """
    Split the training corpus into n_shards sequential (non-shuffled) partitions.
    The partition is deterministic: first n_shards-1 shards are equal-sized;
    the last shard absorbs any remainder.
    """
    import math
    n = len(examples)
    shard_size = math.ceil(n / n_shards)
    shards = []
    for i in range(n_shards):
        start = i * shard_size
        end = min(start + shard_size, n)
        shards.append(examples[start:end])
    return shards


# ---------------------------------------------------------------------------
# 6. Data fraction subsampling
# ---------------------------------------------------------------------------
def subsample(
    examples: List[Dict],
    fraction: float,
    seed: int = 42,
) -> List[Dict]:
    """
    Return a fraction of examples for data-efficiency experiments.

    Args:
        examples: Full list of examples.
        fraction: Float in (0, 1].
        seed: Random seed for reproducibility.
    Returns:
        Subsampled list of examples.
    """
    import random
    rng = random.Random(seed)
    n = max(1, int(len(examples) * fraction))
    indices = sorted(rng.sample(range(len(examples)), n))
    return [examples[i] for i in indices]


# ---------------------------------------------------------------------------
# 7. Main pipeline
# ---------------------------------------------------------------------------
def load_conll_ner(filepath: str) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Load a CoNLL-format NER file.

    Expected format (space-separated):
        TOKEN  LABEL
        (blank line between sentences)

    Returns:
        sentences: list of token lists
        labels: list of label lists
    """
    sentences, labels = [], []
    current_tokens, current_labels = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("-DOCSTART-"):
                continue
            if not line:
                if current_tokens:
                    sentences.append(current_tokens)
                    labels.append(current_labels)
                    current_tokens, current_labels = [], []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    token = normalise_ethiopic(clean_text(parts[0]))
                    label = parts[-1]
                    current_tokens.append(token)
                    current_labels.append(label)
    if current_tokens:
        sentences.append(current_tokens)
        labels.append(current_labels)
    return sentences, labels


def save_shards(shards, output_dir: str, split_name: str = "train"):
    """Save shards as JSON lines files."""
    os.makedirs(output_dir, exist_ok=True)
    for i, shard in enumerate(shards):
        out_path = os.path.join(output_dir, f"{split_name}_shard_{i+1}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in shard:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  Saved {len(shard)} examples to {out_path}")


def validate_no_test_contamination(
    train_shards: List[List[Dict]],
    test_examples: List[Dict],
) -> None:
    """Assert that no test sentence appears in any training shard."""
    test_keys = {" ".join(ex["tokens"]) for ex in test_examples}
    for i, shard in enumerate(train_shards):
        for ex in shard:
            key = " ".join(ex["tokens"])
            assert key not in test_keys, (
                f"Test contamination detected in shard {i+1}: '{key[:60]}...'"
            )
    print("  No test contamination detected.")


def main():
    parser = argparse.ArgumentParser(description="Amharic NER preprocessing pipeline")
    parser.add_argument("--train_file", required=True, help="CoNLL train file")
    parser.add_argument("--dev_file", required=True, help="CoNLL validation file")
    parser.add_argument("--test_file", required=True, help="CoNLL test file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--n_shards", type=int, default=3, help="Number of shards")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per_shard_test", action="store_true",
                        help="Also write disjoint test_part_{j}.jsonl files "
                             "(contiguous n_shards-way split of the test set).")
    args = parser.parse_args()

    print("=== Amharic NER Preprocessing Pipeline ===")

    # Load
    print("\n[1] Loading CoNLL files...")
    train_sentences, train_labels = load_conll_ner(args.train_file)
    dev_sentences, dev_labels = load_conll_ner(args.dev_file)
    test_sentences, test_labels = load_conll_ner(args.test_file)
    print(f"  Train: {len(train_sentences)} sentences")
    print(f"  Dev:   {len(dev_sentences)} sentences")
    print(f"  Test:  {len(test_sentences)} sentences")

    # Build example dicts
    train_examples = [
        {"tokens": t, "ner_tags": l}
        for t, l in zip(train_sentences, train_labels)
    ]
    dev_examples = [
        {"tokens": t, "ner_tags": l}
        for t, l in zip(dev_sentences, dev_labels)
    ]
    test_examples = [
        {"tokens": t, "ner_tags": l}
        for t, l in zip(test_sentences, test_labels)
    ]

    # Deduplicate train
    print("\n[2] Deduplicating training set...")
    before = len(train_examples)
    train_examples = deduplicate(train_examples, text_key="tokens")
    print(f"  Removed {before - len(train_examples)} duplicates. "
          f"Remaining: {len(train_examples)}")

    # Partition into shards
    print(f"\n[3] Partitioning into {args.n_shards} shards...")
    shards = partition_into_shards(train_examples, n_shards=args.n_shards,
                                   seed=args.seed)
    for i, shard in enumerate(shards):
        print(f"  Shard {i+1}: {len(shard)} sentences")

    # Validate no contamination
    print("\n[4] Checking test contamination...")
    validate_no_test_contamination(shards, test_examples)

    # Save
    print("\n[5] Saving outputs...")
    save_shards(shards, args.output_dir, split_name="train")
    save_shards([dev_examples], args.output_dir, split_name="dev")
    save_shards([test_examples], args.output_dir, split_name="test")
    if args.per_shard_test:
        # NOTE: the original release wrote only test_shard_1.jsonl (the whole
        # test set), so every R[i][j] column was scored on the same sentences.
        parts = partition_into_shards(test_examples, n_shards=args.n_shards)
        save_shards(parts, args.output_dir, split_name="test_part")

    # Data-efficiency subsets
    print("\n[6] Creating data-efficiency subsets (10%, 25%, 50%)...")
    for frac in [0.10, 0.25, 0.50]:
        for i, shard in enumerate(shards):
            sub = subsample(shard, fraction=frac, seed=args.seed)
            out_path = os.path.join(
                args.output_dir,
                f"train_shard_{i+1}_frac{int(frac*100)}.jsonl"
            )
            with open(out_path, "w", encoding="utf-8") as f:
                for ex in sub:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            print(f"  {frac*100:.0f}% shard {i+1}: {len(sub)} sentences -> {out_path}")

    # Save label map
    all_labels = sorted(
        {tag for ex in train_examples + dev_examples + test_examples
              for tag in ex["ner_tags"]}
    )
    label2id = {l: i for i, l in enumerate(all_labels)}
    id2label = {i: l for l, i in label2id.items()}
    with open(os.path.join(args.output_dir, "label_map.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)
    print(f"\n  Label map ({len(label2id)} labels): {list(label2id.keys())}")
    print("\n=== Preprocessing complete. ===")


if __name__ == "__main__":
    main()
