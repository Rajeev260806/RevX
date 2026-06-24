import re
import random
import pandas as pd
from pathlib import Path
from collections import Counter
 
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
 
from dataset_helpers.dataset_explore import clean_and_tokenize, encode, load_vocab
from dataset_helpers.data_tokenize import pad_or_truncate, MAX_LEN, RANDOM_SEED, VAL_RATIO

CSV_PATH = Path(__file__).parent / "fake reviews dataset.csv"

def load_raw():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"\n  Dataset not found at: {CSV_PATH}\n"
            f"  Download it first:\n"
            f"    kaggle datasets download -d mexwell/fake-reviews-dataset\n"
            f"  Then unzip into: {CSV_PATH.parent}\n"
        )
    return pd.read_csv(CSV_PATH)

def explore_raw(df):
    print("=" * 60)
    print("STEP 1 — RAW DATASET EXPLORATION")
    print("=" * 60)
 
    print(f"\n  Shape          : {df.shape}")
    print(f"  Columns        : {df.columns.tolist()}")
 
    label_col = "label" if "label" in df.columns else df.columns[-2]
    text_col  = "text_" if "text_" in df.columns else df.columns[-1]
 
    print(f"\n  [CLASS BALANCE] (column: '{label_col}')")
    counts = df[label_col].value_counts()
    print(counts)
    total = len(df)
    for label, count in counts.items():
        print(f"  {label}: {count:,}  ({count/total:.1%})")
 
    print(f"\n  [TEXT LENGTH] (column: '{text_col}')")
    lengths = df[text_col].astype(str).apply(lambda t: len(t.split()))
    print(f"  Min    : {lengths.min()}")
    print(f"  Max    : {lengths.max()}")
    print(f"  Mean   : {lengths.mean():.1f}")
    print(f"  Median : {lengths.median():.0f}")
 
    print(f"\n  [SAMPLE ROWS]")
    for i in range(3):
        print(f"\n  Row {i+1} [{df.iloc[i][label_col]}]:")
        print(f"  {str(df.iloc[i][text_col])[:150]!r}")
 
    if "category" in df.columns:
        print(f"\n  [CATEGORY BREAKDOWN]")
        print(df["category"].value_counts())
 
    return label_col, text_col

def get_fake_review_data():
    df = load_raw()
    label_col = "label" if "label" in df.columns else df.columns[-2]
    text_col  = "text_" if "text_" in df.columns else df.columns[-1]
    df = df.dropna(subset=[text_col, label_col])
    reviews = df[text_col].astype(str).tolist()
    labels  = [1 if lbl == "CG" else 0 for lbl in df[label_col]]
 
    return reviews, labels

def get_mock_fake_review_data():
    reviews = [
        #CG reviews
        "Amazing product exceeded my expectations highly recommend to everyone",
        "Best purchase ever five stars perfect quality fast shipping amazing",
        "Great value for money works exactly as described very happy customer",
        "Excellent product amazing quality will buy again highly recommended",
        #OR reviews
        "The strap broke after two weeks but customer service replaced it free",
        "Battery life is shorter than advertised, around 5 hours not 8 as claimed",
        "Good sound quality for the price, bass is a bit weak for my taste",
        "Took 3 days longer than expected to arrive but works fine now that it's here",
    ]
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    return reviews, labels

def split_fake_review_data(reviews, labels, val_ratio=VAL_RATIO, seed=RANDOM_SEED):
    assert len(reviews) == len(labels)
 
    paired = list(zip(reviews, labels))
    random.seed(seed)
    random.shuffle(paired)
    reviews_shuffled, labels_shuffled = zip(*paired)
    reviews_shuffled, labels_shuffled = list(reviews_shuffled), list(labels_shuffled)
 
    n = len(reviews_shuffled)
    test_size  = int(n * 0.1)
    val_size   = int(n * val_ratio)
    train_size = n - val_size - test_size
 
    train_r = reviews_shuffled[:train_size]
    train_l = labels_shuffled[:train_size]
    val_r   = reviews_shuffled[train_size : train_size + val_size]
    val_l   = labels_shuffled[train_size : train_size + val_size]
    test_r  = reviews_shuffled[train_size + val_size :]
    test_l  = labels_shuffled[train_size + val_size :]
 
    return train_r, train_l, val_r, val_l, test_r, test_l

def get_fake_review_splits(use_mock=False, max_len=MAX_LEN):
    word_to_idx, _ = load_vocab()   
 
    if use_mock:
        reviews, labels = get_mock_fake_review_data()
        random.seed(RANDOM_SEED)
        paired = list(zip(reviews, labels))
        random.shuffle(paired)
        reviews, labels = zip(*paired)
        reviews, labels = list(reviews), list(labels)
        train_r, train_l = reviews[:6], labels[:6]
        val_r,   val_l   = reviews[6:7], labels[6:7]
        test_r,  test_l  = reviews[7:],  labels[7:]
    else:
        reviews, labels = get_fake_review_data()
        train_r, train_l, val_r, val_l, test_r, test_l = split_fake_review_data(
            reviews, labels
        )
 
    def encode_batch(texts):
        out = []
        for t in texts:
            tokens  = clean_and_tokenize(t, remove_sw=True)
            indices = encode(tokens, word_to_idx)
            out.append(pad_or_truncate(indices, max_len))
        return out
 
    return (
        encode_batch(train_r), train_l,
        encode_batch(val_r),   val_l,
        encode_batch(test_r),  test_l,
    )

if __name__ == "__main__":
 
    df = load_raw()
    label_col, text_col = explore_raw(df)

    print("\n" + "=" * 60)
    print("STEP 2 — CONVERTING TO PROJECT INTERFACE")
    print("=" * 60)
 
    reviews, labels = get_fake_review_data()
    print(f"\n  Total reviews : {len(reviews):,}")
    print(f"  Fake (1)      : {sum(labels):,}  ({sum(labels)/len(labels):.1%})")
    print(f"  Real (0)      : {len(labels)-sum(labels):,}  ({1-sum(labels)/len(labels):.1%})")
 
    print(f"\n  Sample fake (label=1): {reviews[labels.index(1)][:120]!r}")
    print(f"  Sample real (label=0): {reviews[labels.index(0)][:120]!r}")

    print("=" * 60)
    print("STEP 3 — VERIFYING SPLIT")
    print("=" * 60)
 
    tr_enc, tr_lbl, va_enc, va_lbl, te_enc, te_lbl = get_fake_review_splits(use_mock=False)
 
    print(f"\n  Train : {len(tr_enc):,} reviews  ({sum(tr_lbl)/len(tr_lbl):.1%} fake)")
    print(f"  Val   : {len(va_enc):,} reviews  ({sum(va_lbl)/len(va_lbl):.1%} fake)")
    print(f"  Test  : {len(te_enc):,} reviews  ({sum(te_lbl)/len(te_lbl):.1%} fake)")
 
    for split_name, encodings in [("train", tr_enc), ("val", va_enc), ("test", te_enc)]:
        for enc in encodings[:5]:
            assert len(enc) == MAX_LEN, f"{split_name} encoding length mismatch"
    print(f"\n  ✅ All encodings padded/truncated to MAX_LEN={MAX_LEN}")

    print("\n" + "=" * 60)
    print("STEP 4 — MOCK DATA TEST (fast sanity check)")
    print("=" * 60)
 
    mock_reviews, mock_labels = get_mock_fake_review_data()
    for r, l in zip(mock_reviews, mock_labels):
        print(f"  [{'FAKE' if l else 'REAL'}] {r}")