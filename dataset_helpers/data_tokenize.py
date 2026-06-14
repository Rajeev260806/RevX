import json
import random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset_explore import(clean_and_tokenize,encode,load_vocab,get_mock_data,get_real_data,PAD_TOKEN,UNK_TOKEN,)

MAX_LEN = 512
RANDOM_SEED = 42
VAL_RATIO = 0.1  #10% of training set = 2500 reviews

def pad_or_truncate(indices,MAX_LEN,pad_index=0):
    if len(indices)<=MAX_LEN:
        pad = [pad_index]*(MAX_LEN-len(indices))
        return indices+pad
    else:
        return indices[:MAX_LEN]
    
def tokenize_and_encode(text,word_to_idx,max_len,removeSW=True):
    tokens = clean_and_tokenize(text,removeSW)
    encoded_tokens = encode(tokens,word_to_idx)
    return pad_or_truncate(encoded_tokens,max_len)

def split_data(reviews,labels,val_ratio=VAL_RATIO,seed=RANDOM_SEED):
    assert len(reviews) == len(labels), \
        f"reviews and labels must be same length: {len(reviews)} vs {len(labels)}"
    paired = list(zip(reviews,labels))
    random.seed(seed)
    random.shuffle(paired)

    reviews_shuffled, labels_shuffled = zip(*paired)
    reviews_shuffled = list(reviews_shuffled)
    labels_shuffled = list(labels_shuffled)
    val_size    = int(len(reviews_shuffled) * val_ratio)
    train_size  = len(reviews_shuffled) - val_size
    train_reviews = reviews_shuffled[:train_size]
    train_labels  = labels_shuffled[:train_size]
    val_reviews   = reviews_shuffled[train_size:]
    val_labels    = labels_shuffled[train_size:]

    return train_reviews,train_labels,val_reviews,val_labels

def get_splits(mock_data=False):
    word_to_idx,idx_to_word = load_vocab()
    if mock_data:
        reviews,labels = get_mock_data()
        random.seed(RANDOM_SEED)
        paired = list(zip(reviews, labels))
        random.shuffle(paired)
        reviews, labels = zip(*paired)
        reviews, labels = list(reviews), list(labels)
        train_r, train_l = reviews[:6], labels[:6]
        val_r,   val_l   = reviews[6:7], labels[6:7]
        test_r,  test_l  = reviews[7:],  labels[7:]
    else:
        all_reviews, all_labels = get_real_data("train")
        train_r, train_l, val_r, val_l = split_data(
            all_reviews, all_labels, val_ratio=VAL_RATIO, seed=RANDOM_SEED
        )
        test_r,test_l = get_real_data("test")

    train_encodings = [tokenize_and_encode(r, word_to_idx, MAX_LEN) for r in train_r]
    val_encodings   = [tokenize_and_encode(r, word_to_idx, MAX_LEN) for r in val_r]
    test_encodings  = [tokenize_and_encode(r, word_to_idx, MAX_LEN) for r in test_r]

    return (
        train_encodings, list(train_l),
        val_encodings,   list(val_l),
        test_encodings,  list(test_l),
    )