import re
import string
import json
from collections import Counter
from datasets import load_dataset

dataset = load_dataset("stanfordnlp/imdb")

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

#This is just a check for above cleaning process - Status: Success
""" sample_raw = dataset["train"][5]["text"]
 print(f"\nRaw review (first 300 chars):\n{sample_raw[:300]}")

 sample_cleaned = clean_text(sample_raw)
 print(f"\nCleaned (first 300 chars):\n{sample_cleaned[:300]}")

 sample_tokens = clean_and_tokenize(sample_raw,True)
 print(f"\nTokens (first 20): ",sample_tokens[:20])
 print(f"Total tokens after cleaning: {len(sample_tokens)}")"""

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

word_counter = Counter()

for sample in dataset["train"]:
    tokens = clean_and_tokenize(sample["text"],True)
    word_counter.update(tokens)

MIN_FREQ = 5

filter_small_words = [words for words,freq in word_counter.items() if freq>=MIN_FREQ]
sorted_filtered_words = sorted(filter_small_words,key=lambda w: -word_counter[w])

wordtoidx = {PAD_TOKEN:0,UNK_TOKEN:1}
for word in sorted_filtered_words:
    wordtoidx[word] = len(wordtoidx)

idxtoword = {idx:word for word,idx in wordtoidx.items()}

#Checking for the index to word and word to index dictionary data - Status: Success
""" VOCAB_SIZE = len(wordtoidx)
print(f"\nFinal vocabulary size (including PAD & UNK): {VOCAB_SIZE:,}")
print(f"  PAD → index {wordtoidx[PAD_TOKEN]}")
print(f"  UNK → index {wordtoidx[UNK_TOKEN]}")
print(f"  'film' → index {wordtoidx.get('film', 'not in vocab')}")
print(f"  'movie' → index {wordtoidx.get('movie', 'not in vocab')}")"""

def encode(tokens,wordtoidx):
    unk_idx = wordtoidx[UNK_TOKEN]
    return [wordtoidx.get(t,unk_idx) for t in tokens]

def decode(indices,idxtoword):
    return [idxtoword.get(i,UNK_TOKEN) for i in indices]


#Test for encode and decode - Status: Success
"""demo_tokens  = clean_and_tokenize(dataset["train"][0]["text"],True)
demo_indices = encode(demo_tokens, wordtoidx)
demo_decoded = decode(demo_indices, idxtoword)
print(f"\nDemo encode/decode:")
print(f"  Tokens : {demo_tokens}")
print(f"  Indices: {demo_indices}")
print(f"  Decoded: {demo_decoded}")"""

print("\n" + "=" * 60)
print("VOCABULARY SUMMARY")
print("=" * 60)
print(f"  Total unique raw tokens    : {len(word_counter):>8,}")
print(f"  Tokens with freq >= {MIN_FREQ:>2}     : {len(filter_small_words):>8,}")
print(f"  Final vocab size (+ specials): {len(wordtoidx):>8,}")
print(f"  Coverage: {len(filter_small_words)/len(word_counter)*100:.1f}% of unique words kept")
 
coverage_by_tokens = sum(c for w, c in word_counter.items() if c >= MIN_FREQ)
total_tokens       = sum(word_counter.values())
print(f"  Token coverage (by count)  : {coverage_by_tokens/total_tokens*100:.1f}% of all token occurrences")

VOCAB_PATH = "vocab.json"
with open(VOCAB_PATH, "w", encoding="utf-8") as f:
    json.dump({"word_to_idx": wordtoidx, "idx_to_word": idxtoword}, f, ensure_ascii=False)