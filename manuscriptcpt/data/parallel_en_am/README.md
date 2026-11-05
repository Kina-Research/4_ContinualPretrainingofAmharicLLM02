# EthioEnAm-Parallel: English-Amharic Parallel Text Corpus

## 1. Overview & Data Provenance

This directory contains the **EthioEnAm-Parallel** dataset, an open-source parallel text corpus consisting of **19,386 curated English-Amharic sentence pairs** designed for bilingual continual pre-training, parameter-efficient fine-tuning (PEFT), and cross-lingual retention evaluation under sequential domain shifts.

The dataset is divided into two transparently distinguished categories:
1. **Empirical Real-World Corpus** (19,386 pairs): Derived from standard, reliable open-source multilingual repositories (OPUS-100 and localized multi-domain reasoning benchmarks).
2. **Auxiliary Synthetic Corpus** (782 template-generated pairs): Specifically marked and isolated for synthetic data augmentation and structural stress-testing, following the precedent established in the I-GANH reference paper (Syiemlieh et al., 2026).

---

## 2. Source Repositories & Data Collection Methodology

### 2.1 Empirical Real-World Bitext
- **OPUS-100 (`Helsinki-NLP/opus-100`, English-Amharic split)**:
  - *Reference*: Tiedemann (2012); Zhang et al. (2020).
  - *Content*: Community and professional open-source software localizations, technical manuals, GNOME/Ubuntu translations, and open-domain text.
  - *Retained Pairs*: 6,402 clean pairs.
- **KCorpus Bilingual Domain Adaptations**:
  - *Reference*: KCorpus Multi-Task Corpus (Alebachew et al., 2024/2025).
  - *Content*: Ethiopianized parallel text pairs across four key domains:
    - *Mathematics & Arithmetic Word Problems* (4,168 pairs; based on GSM8K)
    - *Logical Reasoning & Commonsense Science* (3,732 pairs; based on OpenBookQA)
    - *Ethical Dilemmas & Professional Conduct* (2,941 pairs; based on Anthropic RLHF)
    - *Multi-turn Conversational Exchanges* (2,143 pairs)

### 2.2 Auxiliary Synthetic Corpus
- *Generator Script*: `code/generate_synthetic_parallel.py`
- *Methodology*: Combinatorial slot-filling across 10 bilingual grammatical templates spanning civic news, health, climate reasoning, agriculture, and academic discourse.
- *Tagging*: Every entry carries `"is_synthetic": true` and a dedicated `limitations_note`.

> **Academic Publication Integrity Disclaimer**: Synthetic data are explicitly segregated from empirical evaluation metrics. Synthetic samples are provided solely to enable reproducible investigations into data augmentation and pseudo-parallel regularization (analogous to the MC-GAN pseudo-parallel stream in I-GANH). Synthetic outputs are not presented as human linguistic behavior.

---

## 3. Dataset Size & Structural Split

The empirical real-world corpus is split into a **80% Train / 10% Validation / 10% Test** partition using a deterministic random seed (`seed = 42`). The training set is further sliced into **three equal sequential shards** ($D_1 \to D_2 \to D_3$) to establish a rigorous continual learning (CL) benchmark.

| Split | Pairs | English Tokens | Amharic Tokens | Mean EN Length (Tokens) | Mean AM Length (Tokens) | EN Vocab Size | AM Vocab Size |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train Full** | **15,510** | 555,680 | 420,277 | 35.83 | 27.10 | 44,371 | 70,911 |
| ├─ *Shard $D_1$* | 5,170 | 184,184 | 139,112 | 35.63 | 26.91 | 23,054 | 34,199 |
| ├─ *Shard $D_2$* | 5,170 | 185,922 | 141,046 | 35.96 | 27.28 | 22,968 | 34,087 |
| └─ *Shard $D_3$* | 5,170 | 185,574 | 140,119 | 35.89 | 27.10 | 22,937 | 33,938 |
| **Validation** | **1,938** | 69,159 | 52,659 | 35.69 | 27.17 | 12,036 | 16,935 |
| **Test** | **1,938** | 70,085 | 53,166 | 36.16 | 27.43 | 12,177 | 17,103 |
| **Total Real** | **19,386** | **694,924** | **526,102** | **35.85** | **27.14** | **50,422** | **78,650** |
| *Synthetic Auxiliary* | *782* | *11,730* | *9,384* | *15.00* | *12.00* | *280* | *320* |

---

## 4. Language Distribution & Typological Profile

- **Source Language**: English (`en`), Latin alphabet, analytic syntax (SVO).
- **Target Language**: Amharic (`am`), Ge'ez (Ethiopic) syllabary (`U+1200` to `U+137F`), morphologically rich Semitic language with root-and-pattern inflection, clitic affixation, and Subject-Object-Verb (SOV) canonical word order.
- **Morphological Agglutination Effect**: While the English text contains 50,422 unique word types, the Amharic text contains 78,650 unique word types across the same sentences. This 1.56× vocabulary expansion reflects Amharic's extensive prefixation, suffixation, and pronominal cliticization.

---

## 5. Preprocessing Pipeline

The preprocessing script (`code/process_parallel_data.py`) executes a 6-stage standardization pipeline:

1. **Unicode NFKC Normalization**: Resolves composite characters, full-width variants, and encoding inconsistencies.
2. **Ge'ez Word Separator Normalization**: Converts archaic two-dot word boundaries (`፡`, `U+1361`) to standard spaces while preserving punctuation boundaries (`።`, `፣`, `፤`, `፦`).
3. **Ethiopic Homophone Unification**: Systematically maps phonetically identical character classes to their standard base representations:
   - $h$-phonemes: ሐ, ኀ, ኃ, ሓ, ኅ, ኋ, ኁ, ሑ, ሒ $\to$ ሀ
   - $s$-phonemes: ሠ, ሡ, ሢ, ሣ, ሤ, ሥ, ሦ, ሧ $\to$ ሰ
   - Glottal/vocalic $a$-phonemes: ዐ, ዑ, ዒ, ዓ, ዔ, ዕ, ዖ $\to$ አ
   - $ts$-phonemes: ፀ, ፁ, ፂ, ፃ, ፄ, ፅ, ፆ $\to$ ጸ
4. **Length and Ratio Filtering**:
   - Minimum length: $\ge 2$ words, $\ge 5$ characters.
   - Maximum length: $\le 1000$ characters.
   - Word length ratio: $0.20 \le \frac{|W_{\text{en}}|}{|W_{\text{am}}|} \le 4.50$.
5. **Script Validity Check**:
   - Rejection of any Amharic string where Ethiopic Unicode characters account for $< 35\%$ of non-whitespace characters.
6. **Exact Deduplication**:
   - Filtered across normalized `(lower(english), amharic)` tuples. A total of 3,934 duplicate, malformed, or out-of-ratio pairs were removed from the initial pool of 23,320 pairs.

---

## 6. Directory Structure & File Manifest

```
data/parallel_en_am/
├── README.md                          # Comprehensive documentation (this file)
├── raw/
│   └── raw_parallel_pairs.jsonl       # Unfiltered collection of raw extracted pairs
└── processed/
    ├── train_full.jsonl               # Complete training partition (15,510 pairs)
    ├── train_shard_1.jsonl            # Continual Learning Stage 1 (D1, 5,170 pairs)
    ├── train_shard_2.jsonl            # Continual Learning Stage 2 (D2, 5,170 pairs)
    ├── train_shard_3.jsonl            # Continual Learning Stage 3 (D3, 5,170 pairs)
    ├── valid.jsonl                    # Validation partition (1,938 pairs)
    ├── test.jsonl                     # Held-out evaluation partition (1,938 pairs)
    ├── parallel_en_am.tsv             # Tab-separated export for MT tooling
    ├── parallel_corpus_stats.json     # Detailed numerical statistics in JSON
    └── synthetic_parallel_pairs.jsonl # Auxiliary synthetic pairs (782 pairs)
```

### JSONL Sample Schema
```json
{
  "id": "en_am_000001",
  "english": "Web Addresses",
  "amharic": "የዌብ አድራሻ ዎች",
  "source": "OPUS-100 (Helsinki-NLP/opus-100 am-en)",
  "domain": "general_technical_localization",
  "is_synthetic": false
}
```

---

## 7. Quality-Control Measures & Validation Protocol

- **Script Integrity**: Automated assertion that each Amharic entry passes regular expression matching on the Ethiopic syllabary range `[\u1200-\u137F]`.
- **Zero Test Contamination**: Test sentences are strictly excluded from all training shards.
- **Reproducibility**: The complete raw-to-processed curation pipeline is fully deterministic and re-executable via:
  ```bash
  python code/process_parallel_data.py
  python code/generate_synthetic_parallel.py
  ```

---

## 8. Primary Citations

- Tiedemann, J. (2012). Parallel Data, Tools and Interfaces in OPUS. In *Proceedings of LREC 2012*, pp. 2214–2218.
- Zhang, B., Williams, P., Titov, I., & Sennrich, R. (2020). Improving Massively Multilingual Neural Machine Translation and Zero-Shot Translation. In *Proceedings of ACL 2020*, pp. 1628–1639.
- Syiemlieh, I., Ahmed, Z., Maji, A. K., & Das, S. (2026). GAN-based Incremental Learning for English to Khasi Machine Translation: A Low-Resource Language Model. *Computación y Sistemas*, 30(3), 1991–2007. DOI: 10.13053/CyS-30-3-6641.

