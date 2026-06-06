import re
import string
import json
from collections import Counter
from datasets import load_dataset

dataset = load_dataset("stanfordnlp/imdb")

print(f"  Dataset splits available : {list(dataset.keys())}")
print(f"  Training samples         : {len(dataset['train']):,}")
print(f"  Test samples             : {len(dataset['test']):,}")
print(f"  Columns in each sample   : {dataset['train'].column_names}")

def explore_dataset(dataset):
    train = dataset["train"]
    test = dataset["test"]

    train_labels = [s["label"] for s in train]
    test_labels = [s["label"] for s in test]

    train_pos = train_labels.count(1)
    train_neg = train_labels.count(0)
    test_pos  = test_labels.count(1)
    test_neg  = test_labels.count(0)

    print("\n  [CLASS BALANCE]")
    print(f"  Train — Positive (label=1): {train_pos:,}  |  Negative (label=0): {train_neg:,}")
    print(f"  Test  — Positive (label=1): {test_pos:,}   |  Negative (label=0): {test_neg:,}")
    print(f"  Balance ratio (train)      : {train_pos / len(train_labels):.2%} positive")

    print("Dataset is clean and need not be resampled!") if abs(train_pos-train_neg)<0.05*len(train_labels) else print("Dataset is imbalanced!")


    print("\n  [REVIEW LENGTH DISTRIBUTION — word count before cleaning]")
    train_lengths = [len(s["text"].split()) for s in train]
 
    lengths_sorted = sorted(train_lengths)
    n = len(lengths_sorted)
 
    print(f"  Min length    : {min(train_lengths):>6,} words")
    print(f"  Max length    : {max(train_lengths):>6,} words")
    print(f"  Mean length   : {sum(train_lengths) / n:>9.1f} words")
    print(f"  Median (p50)  : {lengths_sorted[n // 2]:>6,} words")
    print(f"  p75           : {lengths_sorted[int(n * 0.75)]:>6,} words")
    print(f"  p90           : {lengths_sorted[int(n * 0.90)]:>6,} words")
    print(f"  p95           : {lengths_sorted[int(n * 0.95)]:>6,} words")
    print(f"  p99           : {lengths_sorted[int(n * 0.99)]:>6,} words")
 
    very_short = sum(1 for l in train_lengths if l < 10)
    print(f"\n  Reviews < 10 words : {very_short:,}  ({very_short/n:.2%} of train set)")
    print(f"  Recommended MAX_LEN: {lengths_sorted[int(n * 0.95)]} words (covers 95% of data)")

    print("\n  [RAW SAMPLE INSPECTION — first 5 training reviews]")
    for i in range(5):
        sample = train[i]
        raw    = sample["text"]
        label  = "POSITIVE" if sample["label"] == 1 else "NEGATIVE"
        print(f"\n  Sample {i+1} [{label}] — {len(raw.split())} words")
        print(f"  First 200 chars: {raw[:200]!r}")
 
    print("\n  [NOISE DETECTION]")
    html_count = sum(1 for s in train if re.search(r"<[^>]+>", s["text"]))
    url_count  = sum(1 for s in train if re.search(r"http\S+|www\.\S+", s["text"]))
    print(f"  Reviews with HTML tags : {html_count:,}  ({html_count/len(train):.2%})")
    print(f"  Reviews with URLs      : {url_count:,}  ({url_count/len(train):.2%})")
 
 
explore_dataset(dataset)


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

def remove_htmltags(text):
    return re.sub(r"<[^>]+>"," ",text)

def remove_urlheaders(text):
    return re.sub(r"http\S+|www\.\S+", "", text)

def remove_punctuations(text):
    return text.translate(str.maketrans("", "", string.punctuation))

def remove_extrawhitespace(text):
    return re.sub(r"\s+", " ", text).strip()

def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS]

def tokenize(text):
    return text.split()

def clean_text(text):
    text = to_lowercase(text)
    text = remove_htmltags(text)
    text = remove_urlheaders(text)
    text = remove_punctuations(text)
    text = remove_extrawhitespace(text)
    return text

def clean_and_tokenize(text,removeSW=True):
    text = clean_text(text)
    tokens = tokenize(text)
    if removeSW:
        tokens = remove_stopwords(tokens)
    return tokens

def test_cleaning_pipeline(dataset):
    print("\n" + "=" * 60)
    print("STEP 3 — CLEANING PIPELINE VERIFICATION")
    print("=" * 60)
 
    test_cases = [
        # Reviews with known noise types
        dataset["train"][5]["text"],    # typical review
        dataset["train"][10]["text"],   # may have HTML
        "<br />This film was <b>amazing</b>! Visit http://www.filmreview.com for more.",
    ]
 
    for i, raw in enumerate(test_cases):
        cleaned = clean_text(raw)
        tokens  = clean_and_tokenize(raw, removeSW=True)
 
        print(f"\n  Test case {i+1}:")
        print(f"  RAW     ({len(raw.split()):>4} words): {raw[:150]!r}")
        print(f"  CLEANED ({len(cleaned.split()):>4} words): {cleaned[:150]!r}")
        print(f"  TOKENS  ({len(tokens):>4} tokens): {tokens[:15]}")
 
 
test_cleaning_pipeline(dataset)

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
MIN_FREQ = 5

word_counter = Counter()

for sample in dataset["train"]:
    tokens = clean_and_tokenize(sample["text"],True)
    word_counter.update(tokens)


filter_small_words = [words for words,freq in word_counter.items() if freq>=MIN_FREQ]
sorted_filtered_words = sorted(filter_small_words,key=lambda w: -word_counter[w])

wordtoidx = {PAD_TOKEN:0,UNK_TOKEN:1}
for word in sorted_filtered_words:
    wordtoidx[word] = len(wordtoidx)

idxtoword = {idx:word for word,idx in wordtoidx.items()}

print(f"\n  Total unique raw tokens         : {len(word_counter):>8,}")
print(f"  Tokens with freq >= {MIN_FREQ}           : {len(filter_small_words):>8,}")
print(f"  Final vocab size (incl specials): {len(wordtoidx):>8,}")
coverage_unique = len(filter_small_words) / len(word_counter) * 100
coverage_count  = sum(c for w, c in word_counter.items() if c >= MIN_FREQ) / sum(word_counter.values()) * 100
print(f"\n  Coverage (unique words kept)    : {coverage_unique:.1f}%")
print(f"  Coverage (by token occurrences) : {coverage_count:.1f}%")
 
print(f"\n  PAD  → index {wordtoidx[PAD_TOKEN]}")
print(f"  UNK  → index {wordtoidx[UNK_TOKEN]}")
print(f"  'film'  → index {wordtoidx.get('film',  'not in vocab')}")
print(f"  'movie' → index {wordtoidx.get('movie', 'not in vocab')}")
print(f"  'great' → index {wordtoidx.get('great', 'not in vocab')}")
print(f"  'awful' → index {wordtoidx.get('awful', 'not in vocab')}")


def encode(tokens,wordtoidx):
    unk_idx = wordtoidx[UNK_TOKEN]
    return [wordtoidx.get(t,unk_idx) for t in tokens]

def decode(indices,idxtoword):
    return [idxtoword.get(i,UNK_TOKEN) for i in indices]

def test_encode_decode(dataset, wordtoidx, idxtoword):
    print("\n" + "=" * 60)
    print("STEP 5 — ENCODE / DECODE VERIFICATION")
    print("=" * 60)
 
    sample_text   = dataset["train"][0]["text"]
    sample_label  = "POSITIVE" if dataset["train"][0]["label"] == 1 else "NEGATIVE"
 
    tokens  = clean_and_tokenize(sample_text, removeSW=True)
    indices = encode(tokens, wordtoidx)
    decoded = decode(indices, idxtoword)
 
    print(f"\n  Label    : {sample_label}")
    print(f"  Raw text (first 100 chars): {sample_text[:100]!r}")
    print(f"\n  Tokens  (first 15): {tokens[:15]}")
    print(f"  Indices (first 15): {indices[:15]}")
    print(f"  Decoded (first 15): {decoded[:15]}")
 
    mismatch = [(t, d) for t, d in zip(tokens, decoded) if t != d]
    if not mismatch:
        print("\n  ✅ Round-trip check passed: tokens == decoded")
    else:
        print(f"\n  ❌ Mismatches found: {mismatch[:5]}")
 
    unk_count = indices.count(wordtoidx[UNK_TOKEN])
    print(f"\n  UNK tokens in this review : {unk_count} / {len(tokens)}  ({unk_count/len(tokens):.1%})")
 
 
test_encode_decode(dataset, wordtoidx, idxtoword)

VOCAB_PATH = "vocab.json"
vocab_data = {
    "metadata": {
        "min_freq"   : MIN_FREQ,
        "vocab_size" : len(wordtoidx),
        "pad_token"  : PAD_TOKEN,
        "unk_token"  : UNK_TOKEN,
        "pad_index"  : wordtoidx[PAD_TOKEN],
        "unk_index"  : wordtoidx[UNK_TOKEN],
    },
    "wordtoidx": wordtoidx
}
 
with open(VOCAB_PATH, "w", encoding="utf-8") as f:
    json.dump(vocab_data, f, ensure_ascii=False, indent=2)

def test_vocab_reload(vocab_path):
    print("\n  [RELOAD TEST]")
    with open(vocab_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
 
    loaded_w2i = loaded["wordtoidx"]
    loaded_i2w = {int(idx): w for w, idx in loaded_w2i.items()}
 
    pad_idx = loaded_w2i[PAD_TOKEN]
    unk_idx = loaded_w2i[UNK_TOKEN]
 
    print(f"  Reloaded vocab size : {len(loaded_w2i):,}")
    print(f"  PAD idx after reload: {pad_idx}  → word: {loaded_i2w[pad_idx]!r}  ✅")
    print(f"  UNK idx after reload: {unk_idx}  → word: {loaded_i2w[unk_idx]!r}  ✅")
    print(f"  'film' after reload : {loaded_w2i.get('film', 'not found')}")
 
 
test_vocab_reload(VOCAB_PATH)

def get_real_data(split="train"):
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got '{split}'")
 
    reviews = [sample["text"]  for sample in dataset[split]]
    labels  = [sample["label"] for sample in dataset[split]]
 
    return reviews, labels


    