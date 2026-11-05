# Amharic NER Dataset

This directory is a placeholder for the Amharic Named Entity Recognition dataset
used in this study.

## Source

The NER dataset is from:

> H. M. Yohannes and T. Amagasa, "Named-entity recognition for a morphologically
> rich language: A case of Amharic NER," *Information*, vol. 13, no. 2, p. 94, 2022.
> DOI: 10.3390/info13020094

## How to Obtain

Please contact the original authors or access the dataset via their published
repository. The dataset is not redistributed here due to licensing terms.

## Expected Format

Place the following files in this directory:
- `train.txt` — Training split in CoNLL-2003 BIO format
- `dev.txt` — Development/validation split
- `test.txt` — Test split

**CoNLL-2003 BIO format example:**
```
አቶ  B-PER
አብዱ  I-PER
አዲስ  B-LOC
አበባ  I-LOC
ሄደ  O
```

## Dataset Statistics

| Split | Sentences | Tokens | Named Entities |
|-------|-----------|--------|----------------|
| Train | ~8,500 | ~95,000 | ~18,500 |
| Dev | ~1,100 | ~12,000 | ~2,300 |
| Test | ~1,100 | ~12,000 | ~2,300 |

*(Approximate values; exact counts depend on the version of the dataset obtained.)*

## Entity Types

The dataset includes 5 entity types:
- **PER** (Person names)
- **LOC** (Location names)
- **ORG** (Organization names)
- **TIME** (Time expressions including Ethiopic calendar dates)
- **TTL** (Titles and honorifics)

