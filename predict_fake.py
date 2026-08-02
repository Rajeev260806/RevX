import sys
import os
import torch
from pathlib import Path

# Set up project directory paths
PROJECT_ROOT = Path(__file__).parent
DATASET_DIR = PROJECT_ROOT / "dataset_fake_reviews"

# Ensure runtime import paths are resolved
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DATASET_DIR))

# Import exact model, features, and tokenizers from project modules
from dataset_fake_reviews.FRD_BC import FakeReviewDetector, extract_features
from dataset_fake_reviews.ensemble import statistical_fake_score, ensemble_fake_score
from dataset_helpers.dataset_explore import clean_and_tokenize, encode, load_vocab
from dataset_helpers.data_tokenize import pad_or_truncate, MAX_LEN

MODEL_PATH = DATASET_DIR / "fake_detector.pth"

_MODEL_CACHE = None
_VOCAB_CACHE = None

def get_pipeline():
    global _MODEL_CACHE, _VOCAB_CACHE
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if _VOCAB_CACHE is None:
        try:
            _VOCAB_CACHE, _ = load_vocab()
        except Exception:
            _VOCAB_CACHE = {"<PAD>": 0}

    vocab_size = len(_VOCAB_CACHE)

    if _MODEL_CACHE is None:
        model = FakeReviewDetector(vocab_size=vocab_size).to(device)
        if MODEL_PATH.exists():
            try:
                model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            except Exception:
                pass
        model.eval()
        _MODEL_CACHE = (model, device)

    return _MODEL_CACHE[0], _MODEL_CACHE[1], _VOCAB_CACHE

def calculate_real_fake_score(text):
    try:
        model, device, word_to_idx = get_pipeline()

        # 1. Tokenize & Encode
        tokens = clean_and_tokenize(text, remove_sw=True)
        indices = encode(tokens, word_to_idx)
        padded = pad_or_truncate(indices, MAX_LEN)

        # 2. Extract PyTorch Tensors & Features
        X = torch.tensor([padded], dtype=torch.long).to(device)
        feats = extract_features(text).unsqueeze(0).to(device)

        # 3. Model Forward Pass
        with torch.no_grad():
            logit = model(X, feats)
            ml_prob = torch.sigmoid(logit).item()

        # 4. Extract Statistical Anomaly Score
        stat_score, _ = statistical_fake_score(text)

        # 5. Blend using Ensemble Formula (w_ml=0.10, w_stat=0.90)
        final_score = ensemble_fake_score(ml_prob, stat_score, w_ml=0.10, w_stat=0.90)
        return round(float(final_score), 4)

    except Exception:
        return 0.15

if __name__ == "__main__":
    if len(sys.argv) > 1:
        review_text = sys.argv[1]
        score = calculate_real_fake_score(review_text)
        print(score)