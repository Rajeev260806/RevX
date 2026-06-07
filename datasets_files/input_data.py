import re
import os
import string
import json
from pathlib import Path
from collections import Counter

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
MIN_FREQ  = 5

VOCAB_PATH = Path(__file__).parent / "vocab.json"

STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "he", "him", "his", "she", "her", "hers", "it", "its",
    "they", "them", "their", "what", "which", "who", "this", "that", "these",
    "those", "am", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "shall", "can", "need", "dare", "ought", "used", "to",
    "and", "but", "if", "or", "because", "as", "until", "while", "of", "at",
    "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "both", "each",
    "few", "more", "other", "some", "such", "no", "not", "only", "same",
    "so", "than", "too", "very", "just", "a", "an", "the",
}

def to_lowercase(text):
    return text.lower()

def remove_html_tags(text):
    return re.sub(r"<[^>]+>", " ", text)

def remove_urls(text):
    return re.sub(r"http\S+|www\.\S+", "", text)

def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))

def remove_extra_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text):
    return text.split()

def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS]

def clean_text(text):
    text = to_lowercase(text)
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_punctuation(text)
    text = remove_extra_whitespace(text)
    return text

def clean_and_tokenize(text, remove_sw=True):
    text   = clean_text(text)
    tokens = tokenize(text)
    if remove_sw:
        tokens = remove_stopwords(tokens)
    return tokens

def encode(tokens, word_to_idx):
    unk_idx = word_to_idx[UNK_TOKEN]
    return [word_to_idx.get(t, unk_idx) for t in tokens]


def decode(indices, idx_to_word):
    return [idx_to_word.get(i, UNK_TOKEN) for i in indices]

def load_vocab(vocab_path=None):
    
    path = vocab_path or VOCAB_PATH

    if not Path(path).exists():
        raise FileNotFoundError(
            f"\n  vocab.json not found at: {path}\n"
            f"  Fix: run `python week1_data_loading.py` once to generate it.\n"
            f"  This only needs to be done once."
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    word_to_idx = data["word_to_idx"]

    idx_to_word = {int(idx): word for word, idx in word_to_idx.items()}

    return word_to_idx, idx_to_word


def get_mock_data():
    
    reviews = [
        "This movie was absolutely great, loved every moment of it",
        "I hated this film, complete waste of time",
        "Brilliant acting and a wonderful storyline throughout",
        "Terrible plot and very poor acting, do not watch",
        "One of the best films I have ever seen in my life",
        "Boring and predictable, fell asleep halfway through",
        "A masterpiece of cinema, truly unforgettable experience",
        "Awful script and the characters were completely unconvincing",
    ]
    labels = [1, 0, 1, 0, 1, 0, 1, 0]
    return reviews, labels


def get_real_data(split="train"):
    
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got: '{split}'")

    from datasets import load_dataset
    dataset = load_dataset("stanfordnlp/imdb")

    reviews = [sample["text"]  for sample in dataset[split]]
    labels  = [sample["label"] for sample in dataset[split]]
    return reviews, labels

if __name__ == "__main__":

    from datasets import load_dataset

    print("=" * 60)
    print("STEP 1 — LOADING DATASET")
    print("=" * 60)

    dataset = load_dataset("stanfordnlp/imdb")

    print(f"  Splits available : {list(dataset.keys())}")
    print(f"  Training samples : {len(dataset['train']):,}")
    print(f"  Test samples     : {len(dataset['test']):,}")
    print(f"  Columns          : {dataset['train'].column_names}")

    print("\n" + "=" * 60)
    print("STEP 2 — DATASET EXPLORATION REPORT")
    print("=" * 60)

    train_labels  = [s["label"] for s in dataset["train"]]
    train_pos     = train_labels.count(1)
    train_neg     = train_labels.count(0)
    print(f"\n  [CLASS BALANCE]")
    print(f"  Positive reviews : {train_pos:,}")
    print(f"  Negative reviews : {train_neg:,}")
    print(f"  Balance ratio    : {train_pos / len(train_labels):.2%} positive")
    if abs(train_pos - train_neg) < 0.05 * len(train_labels):
        print(f"  ✅ Well-balanced. No resampling needed.")
    else:
        print(f"  ⚠️  Imbalanced. Consider oversampling minority class.")

    train_lengths = [len(s["text"].split()) for s in dataset["train"]]
    ls = sorted(train_lengths)
    n  = len(ls)
    print(f"\n  [REVIEW LENGTHS — word count before cleaning]")
    print(f"  Min    : {min(train_lengths):>6,} words")
    print(f"  Max    : {max(train_lengths):>6,} words")
    print(f"  Mean   : {sum(train_lengths)/n:>9.1f} words")
    print(f"  Median : {ls[n//2]:>6,} words")
    print(f"  p90    : {ls[int(n*0.90)]:>6,} words")
    print(f"  p95    : {ls[int(n*0.95)]:>6,} words  ← recommended MAX_LEN")
    print(f"  p99    : {ls[int(n*0.99)]:>6,} words")

    html_count = sum(1 for s in dataset["train"] if re.search(r"<[^>]+>", s["text"]))
    url_count  = sum(1 for s in dataset["train"] if re.search(r"http\S+|www\.\S+", s["text"]))
    print(f"\n  [NOISE DETECTION]")
    print(f"  Reviews with HTML tags : {html_count:,}  ({html_count/len(dataset['train']):.2%})")
    print(f"  Reviews with URLs      : {url_count:,}  ({url_count/len(dataset['train']):.2%})")

    print(f"\n  [RAW SAMPLE INSPECTION]")
    for i in range(3):
        s   = dataset["train"][i]
        tag = "POSITIVE" if s["label"] == 1 else "NEGATIVE"
        print(f"  Sample {i+1} [{tag}]: {s['text'][:120]!r}")

    print("\n" + "=" * 60)
    print("STEP 3 — CLEANING PIPELINE VERIFICATION")
    print("=" * 60)

    test_cases = [
        dataset["train"][5]["text"],
        "<br />This film was <b>amazing</b>! Visit http://www.filmreview.com for more.",
    ]
    for i, raw in enumerate(test_cases):
        cleaned = clean_text(raw)
        tokens  = clean_and_tokenize(raw, remove_sw=True)
        print(f"\n  Case {i+1}:")
        print(f"  RAW     ({len(raw.split()):>4}w): {raw[:100]!r}")
        print(f"  CLEANED ({len(cleaned.split()):>4}w): {cleaned[:100]!r}")
        print(f"  TOKENS  ({len(tokens):>4}t): {tokens[:12]}")

    print("\n" + "=" * 60)
    print("STEP 4 — BUILDING VOCABULARY")
    print("=" * 60)
    print(f"  Scanning {len(dataset['train']):,} reviews (MIN_FREQ={MIN_FREQ}) ...")

    word_counter = Counter()
    for sample in dataset["train"]:
        tokens = clean_and_tokenize(sample["text"], remove_sw=True)
        word_counter.update(tokens)

    filtered_words = [w for w, freq in word_counter.items() if freq >= MIN_FREQ]
    sorted_words   = sorted(filtered_words, key=lambda w: -word_counter[w])

    word_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word in sorted_words:
        word_to_idx[word] = len(word_to_idx)

    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    coverage_u = len(filtered_words) / len(word_counter) * 100
    coverage_c = sum(c for w,c in word_counter.items() if c>=MIN_FREQ) / sum(word_counter.values()) * 100

    print(f"  Total unique tokens         : {len(word_counter):>8,}")
    print(f"  Tokens with freq >= {MIN_FREQ}       : {len(filtered_words):>8,}")
    print(f"  Final vocab size (+ specials): {len(word_to_idx):>8,}")
    print(f"  Coverage (unique words)      : {coverage_u:.1f}%")
    print(f"  Coverage (by token count)    : {coverage_c:.1f}%")
    print(f"  PAD  → {word_to_idx[PAD_TOKEN]}  |  UNK → {word_to_idx[UNK_TOKEN]}")
    print(f"  'film'  → {word_to_idx.get('film',  'not in vocab')}")
    print(f"  'great' → {word_to_idx.get('great', 'not in vocab')}")

    print("\n" + "=" * 60)
    print("STEP 5 — ENCODE / DECODE VERIFICATION")
    print("=" * 60)

    sample_text  = dataset["train"][0]["text"]
    tokens       = clean_and_tokenize(sample_text, remove_sw=True)
    indices      = encode(tokens, word_to_idx)
    decoded      = decode(indices, idx_to_word)
    mismatches   = [(t, d) for t, d in zip(tokens, decoded) if t != d]
    unk_count    = indices.count(word_to_idx[UNK_TOKEN])

    print(f"  Tokens  (first 12): {tokens[:12]}")
    print(f"  Indices (first 12): {indices[:12]}")
    print(f"  Decoded (first 12): {decoded[:12]}")
    print(f"  Round-trip mismatches : {len(mismatches)}  {'✅' if not mismatches else '❌'}")
    print(f"  UNK tokens in sample  : {unk_count}/{len(tokens)}  ({unk_count/len(tokens):.1%})")
    print("\n" + "=" * 60)
    print("STEP 6 — SAVING VOCABULARY")
    print("=" * 60)

    vocab_data = {
        "metadata": {
            "min_freq"   : MIN_FREQ,
            "vocab_size" : len(word_to_idx),
            "pad_token"  : PAD_TOKEN,
            "unk_token"  : UNK_TOKEN,
            "pad_index"  : word_to_idx[PAD_TOKEN],
            "unk_index"  : word_to_idx[UNK_TOKEN],
        },
        "word_to_idx": word_to_idx
    }

    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)

    print(f"  Saved to : {VOCAB_PATH}")
    print(f"  Contains : metadata + word_to_idx")
    print(f"  Note     : idx_to_word is rebuilt on load (avoids JSON int-key bug)")

    w2i_reloaded, i2w_reloaded = load_vocab()
    assert w2i_reloaded[PAD_TOKEN] == 0, "PAD index mismatch after reload"
    assert w2i_reloaded[UNK_TOKEN] == 1, "UNK index mismatch after reload"
    assert i2w_reloaded[0] == PAD_TOKEN,  "PAD word mismatch after reload"
    print(f"  ✅ Reload verified: PAD={i2w_reloaded[0]!r}, UNK={i2w_reloaded[1]!r}")

    print("\n" + "=" * 60)
    print("STEP 7 — DATA INTERFACE VERIFICATION")
    print("=" * 60)

    mock_reviews, mock_labels = get_mock_data()
    print(f"\n  [MOCK DATA — {len(mock_reviews)} reviews]")
    for rev, lbl in zip(mock_reviews, mock_labels):
        print(f"  [{'POS' if lbl else 'NEG'}] {rev}")

    real_reviews, real_labels = get_real_data("train")
    print(f"\n  [REAL DATA] {len(real_reviews):,} reviews loaded from train split")
    print(f"  First : {real_reviews[0][:80]!r} → label={real_labels[0]}")
    print(f"  ✅ Both return list[str], list[int] — identical interface")

    print("\n" + "=" * 60)
    print("WEEK 1 COMPLETE — SUMMARY")
    print("=" * 60)
    print(f"  ✅  Dataset loaded    : {len(dataset['train']):,} train + {len(dataset['test']):,} test")
    print(f"  ✅  Explored          : balance, lengths, noise")
    print(f"  ✅  Cleaning pipeline : lowercase→HTML→URL→punct→whitespace→tokenize→stopwords")
    print(f"  ✅  Vocabulary built  : {len(word_to_idx):,} words  (MIN_FREQ={MIN_FREQ})")
    print(f"  ✅  Encode/decode     : round-trip verified")
    print(f"  ✅  Vocab saved       : {VOCAB_PATH}")
    print(f"  ✅  Data interface    : get_mock_data() and get_real_data() ready")
    print()
    print("  YOUR FRIEND IMPORTS LIKE THIS:")
    print("    from week1_data_loading import get_mock_data, get_real_data")
    print("    from week1_data_loading import clean_and_tokenize, encode")
    print("    from week1_data_loading import load_vocab")
    print()
    print("    word_to_idx, idx_to_word = load_vocab()")
    print("    reviews, labels = get_mock_data()   # test first")
    print("    reviews, labels = get_real_data('train')  # then swap in")
    print()
    print("  NEXT (Week 2):")
    print("  - Padding and truncation (MAX_LEN sequences)")
    print("  - Train / validation / test split (80/10/10)")
    print("=" * 60)