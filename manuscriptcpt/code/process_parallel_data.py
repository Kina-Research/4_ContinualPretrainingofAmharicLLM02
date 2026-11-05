#!/usr/bin/env python3
"""
Process and curate English-Amharic parallel text dataset.
Sources:
  1. OPUS-100 (Helsinki-NLP/opus-100 am-en; Tiedemann 2012, Zhang et al. 2020)
  2. Ethiopianized Bilingual QA & Reasoning Corpus (openbookqa, gsm8k, rlhf bilingual pairs)

Pipeline:
  - Unicode NFKC normalization
  - Ethiopic homophone character normalization
  - Length ratio and language identification filtering
  - Deduplication
  - Stratified 80/10/10 train/validation/test split
  - 3-shard sequential partitioning (D1, D2, D3) for continual learning
  - Comprehensive statistical profiling
"""

import os
import sys
import json
import re
import unicodedata
import random
from collections import Counter
from pathlib import Path

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

SEED = 42
random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "parallel_en_am"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# ---------------------------------------------------------------------------
# Ethiopic Normalisation Mappings
# ---------------------------------------------------------------------------
ETHIOPIC_HOMOPHONE_MAP = {
    # Ha family: ሐ, ኀ, ኃ, ሓ, ኅ, ኋ, ኁ, ሑ, ሒ, ኚ -> ሀ forms
    'ሐ': 'ሀ', 'ኀ': 'ሀ', 'ኃ': 'ሀ', 'ሓ': 'ሀ',
    'ሑ': 'ሁ', 'ኁ': 'ሁ',
    'ሒ': 'ሂ', 'ኅ': 'ሂ',
    'ሓ': 'ሃ', 'ኃ': 'ሃ', 'ኋ': 'ኋ',
    'ሔ': 'ሄ', 'ኄ': 'ሄ',
    'ሕ': 'ህ', 'ኅ': 'ህ',
    'ሖ': 'ሆ', 'ኆ': 'ሆ',
    # Se family: ሠ -> ሰ
    'ሠ': 'ሰ', 'ሡ': 'ሱ', 'ሢ': 'ሲ', 'ሣ': 'ሳ', 'ሤ': 'ሴ', 'ሥ': 'ስ', 'ሦ': 'ሶ', 'ሧ': 'ሷ',
    # A family: ዐ, ዑ, ዒ, ዓ, ዔ, ዕ, ዖ -> አ forms
    'ዐ': 'አ', 'ዑ': 'ኡ', 'ዒ': 'ኢ', 'ዓ': 'ኣ', 'ዔ': 'ኤ', 'ዕ': 'እ', 'ዖ': 'ኦ',
    # Tsa family: ፀ -> ጸ
    'ፀ': 'ጸ', 'ፁ': 'ጹ', 'ፂ': 'ጺ', 'ፃ': 'ጻ', 'ፄ': 'ጼ', 'ፅ': 'ጽ', 'ፆ': 'ጾ',
}

GEEZ_WORD_SEPARATOR = '\u1361'  # ፡

def normalize_ethiopic(text: str) -> str:
    """Normalizes Ethiopic homophone characters and punctuation."""
    if not text:
        return ""
    # Unicode NFKC
    text = unicodedata.normalize("NFKC", text)
    # Replace traditional word separator with standard space
    text = text.replace(GEEZ_WORD_SEPARATOR, " ")
    # Replace multiple spaces with single space
    text = re.sub(r"[ \t]+", " ", text)
    # Homophone unification
    chars = [ETHIOPIC_HOMOPHONE_MAP.get(ch, ch) for ch in text]
    return "".join(chars).strip()

def normalize_english(text: str) -> str:
    """Standardizes English text."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def is_valid_pair(en: str, am: str) -> bool:
    """Quality control filters for English-Amharic parallel sentences."""
    if not en or not am:
        return False
    # Character length constraints
    if len(en) < 5 or len(am) < 5:
        return False
    if len(en) > 1000 or len(am) > 1000:
        return False
    
    en_words = en.split()
    am_words = am.split()
    if len(en_words) < 2 or len(am_words) < 2:
        return False
    
    # Length ratio filter (words)
    ratio = len(en_words) / max(1, len(am_words))
    if ratio < 0.20 or ratio > 4.5:
        return False
    
    # Amharic must contain Ethiopic Unicode characters (U+1200 to U+137F)
    ethiopic_chars = sum(1 for ch in am if '\u1200' <= ch <= '\u137F')
    if ethiopic_chars / max(1, len(am)) < 0.35:
        return False
    
    # Discard non-informative software identifiers or placeholders
    if en.startswith("translator-credits") or "Desktop Entry" in en:
        return False

    return True

# ---------------------------------------------------------------------------
# Data Extraction
# ---------------------------------------------------------------------------
def extract_opus_pairs():
    """Extracts real OPUS-100 pairs from pre-downloaded corpora."""
    opus_pairs = []
    source_files = [
        Path("D:/AMUPHD2015/Papers/KCorpus/PTM/amharic_dataset/train.jsonl"),
        Path("D:/AMUPHD2015/Papers/KCorpus/PTM/amharic_dataset/valid.jsonl"),
        Path("D:/AMUPHD2015/Papers/KCorpus/PTM/amharic_dataset/test.jsonl"),
    ]
    
    prefix = "ይህን ወደ አማርኛ ተርጉም:\n"
    for s_file in source_files:
        if not s_file.exists():
            continue
        with open(s_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("source") == "opus100" and data.get("task") == "translation":
                        raw_inp = data.get("input", "")
                        en = raw_inp.replace(prefix, "").strip()
                        am = data.get("output", "").strip()
                        opus_pairs.append({
                            "english": en,
                            "amharic": am,
                            "source": "OPUS-100 (Helsinki-NLP/opus-100 am-en)",
                            "domain": "general_technical_localization",
                            "is_synthetic": False
                        })
                except Exception:
                    continue
    return opus_pairs

def extract_anchoring_pairs():
    """Extracts bilingual reasoning and QA pairs from KCorpus KCAnchoring."""
    anchoring_pairs = []
    source_files = [
        Path("D:/AMUPHD2015/Papers/KCorpus/KCAnchoring/train.json"),
        Path("D:/AMUPHD2015/Papers/KCorpus/KCAnchoring/validation.json"),
        Path("D:/AMUPHD2015/Papers/KCorpus/KCAnchoring/test.json"),
    ]
    for s_file in source_files:
        if not s_file.exists():
            continue
        try:
            with open(s_file, "r", encoding="utf-8") as f:
                rows = json.load(f)
                for r in rows:
                    ep = r.get("english_prompt", "").strip()
                    ap = r.get("amharic_prompt", "").strip()
                    ttype = r.get("task_type", "dialogue_reasoning")
                    if ep and ap:
                        anchoring_pairs.append({
                            "english": ep,
                            "amharic": ap,
                            "source": f"KCorpus-Anchored ({ttype})",
                            "domain": ttype,
                            "is_synthetic": False
                        })
        except Exception:
            continue
    return anchoring_pairs

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Collecting real-world English-Amharic parallel pairs...")
    opus_pairs = extract_opus_pairs()
    print(f"Extracted {len(opus_pairs)} OPUS-100 pairs.")
    
    anchoring_pairs = extract_anchoring_pairs()
    print(f"Extracted {len(anchoring_pairs)} Anchored QA/Reasoning pairs.")
    
    all_raw = opus_pairs + anchoring_pairs
    print(f"Total raw pairs gathered: {len(all_raw)}")
    
    # Save raw snapshot
    raw_snapshot = RAW_DIR / "raw_parallel_pairs.jsonl"
    with open(raw_snapshot, "w", encoding="utf-8") as f:
        for item in all_raw:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved raw data snapshot to {raw_snapshot}")
    
    # Preprocessing and Quality Control Filtering
    print("Applying quality-control filtering and Ethiopic normalization...")
    seen = set()
    cleaned_pairs = []
    filtered_count = 0
    
    for idx, item in enumerate(all_raw):
        en_clean = normalize_english(item["english"])
        am_clean = normalize_ethiopic(item["amharic"])
        
        if not is_valid_pair(en_clean, am_clean):
            filtered_count += 1
            continue
            
        dedup_key = (en_clean.lower(), am_clean)
        if dedup_key in seen:
            filtered_count += 1
            continue
        seen.add(dedup_key)
        
        cleaned_pairs.append({
            "id": f"en_am_{len(cleaned_pairs)+1:06d}",
            "english": en_clean,
            "amharic": am_clean,
            "source": item["source"],
            "domain": item["domain"],
            "is_synthetic": False
        })
        
    print(f"Filtering complete. Retained {len(cleaned_pairs)} high-quality pairs (filtered/deduped {filtered_count}).")
    
    # Deterministic train/dev/test split (80/10/10)
    random.shuffle(cleaned_pairs)
    total_clean = len(cleaned_pairs)
    n_test = int(total_clean * 0.10)
    n_valid = int(total_clean * 0.10)
    n_train = total_clean - n_test - n_valid
    
    train_set = cleaned_pairs[:n_train]
    valid_set = cleaned_pairs[n_train:n_train + n_valid]
    test_set = cleaned_pairs[n_train + n_valid:]
    
    # Shard training set into 3 equal parts for Continual Learning (D1, D2, D3)
    shard_size = len(train_set) // 3
    d1 = train_set[:shard_size]
    d2 = train_set[shard_size:shard_size * 2]
    d3 = train_set[shard_size * 2:]
    
    # Write processed splits
    splits = {
        "train_full.jsonl": train_set,
        "valid.jsonl": valid_set,
        "test.jsonl": test_set,
        "train_shard_1.jsonl": d1,
        "train_shard_2.jsonl": d2,
        "train_shard_3.jsonl": d3,
    }
    
    for fname, data in splits.items():
        out_p = PROCESSED_DIR / fname
        with open(out_p, "w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {fname}: {len(data)} pairs")
        
    # Also write a clean TSV format for direct machine translation toolkit compatibility
    tsv_out = PROCESSED_DIR / "parallel_en_am.tsv"
    with open(tsv_out, "w", encoding="utf-8") as f:
        f.write("id\tsource\tdomain\tenglish\tamharic\n")
        for row in cleaned_pairs:
            # remove tabs and newlines
            en_line = row["english"].replace("\t", " ").replace("\n", " ")
            am_line = row["amharic"].replace("\t", " ").replace("\n", " ")
            f.write(f"{row['id']}\t{row['source']}\t{row['domain']}\t{en_line}\t{am_line}\n")
    print(f"Wrote TSV to {tsv_out}")
    
    # Compute in-depth summary statistics
    def get_stats(dataset):
        en_word_lens = [len(x["english"].split()) for x in dataset]
        am_word_lens = [len(x["amharic"].split()) for x in dataset]
        en_char_lens = [len(x["english"]) for x in dataset]
        am_char_lens = [len(x["amharic"]) for x in dataset]
        en_vocab = set()
        am_vocab = set()
        for x in dataset:
            en_vocab.update(x["english"].lower().split())
            am_vocab.update(x["amharic"].split())
        return {
            "sentence_pairs": len(dataset),
            "en_tokens_total": sum(en_word_lens),
            "am_tokens_total": sum(am_word_lens),
            "en_avg_tokens_per_sent": round(sum(en_word_lens) / max(1, len(dataset)), 2),
            "am_avg_tokens_per_sent": round(sum(am_word_lens) / max(1, len(dataset)), 2),
            "en_avg_chars_per_sent": round(sum(en_char_lens) / max(1, len(dataset)), 2),
            "am_avg_chars_per_sent": round(sum(am_char_lens) / max(1, len(dataset)), 2),
            "en_vocab_size": len(en_vocab),
            "am_vocab_size": len(am_vocab),
        }
        
    stats = {
        "dataset_name": "Open-Source English-Amharic Parallel Corpus (EthioEnAm-Parallel)",
        "source_repositories": [
            "Helsinki-NLP/opus-100 (am-en) (Tiedemann 2012, Zhang et al. 2020)",
            "KCorpus-Anchored Multi-Domain Reasoning Benchmark"
        ],
        "random_seed": SEED,
        "quality_control": {
            "min_words": 2,
            "max_characters": 1000,
            "min_ratio": 0.20,
            "max_ratio": 4.50,
            "min_ethiopic_script_ratio": 0.35,
            "homophone_normalization": True,
            "exact_deduplication": True
        },
        "raw_pairs_gathered": len(all_raw),
        "filtered_or_duplicate_pairs": filtered_count,
        "clean_pairs_retained": total_clean,
        "splits": {
            "train_full": get_stats(train_set),
            "valid": get_stats(valid_set),
            "test": get_stats(test_set),
            "shard_D1": get_stats(d1),
            "shard_D2": get_stats(d2),
            "shard_D3": get_stats(d3),
        },
        "domain_distribution": dict(Counter(x["domain"] for x in cleaned_pairs)),
        "source_distribution": dict(Counter(x["source"] for x in cleaned_pairs))
    }
    
    stats_p = PROCESSED_DIR / "parallel_corpus_stats.json"
    with open(stats_p, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Saved dataset statistics to {stats_p}")

if __name__ == "__main__":
    main()

