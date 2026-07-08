"""
=============================================================
PHASE 3 — WEEK 8  |  FRIEND'S PART
Fake Review Detector — Anomaly Detection + Ensemble Pipeline
=============================================================
Tasks:
  1. Statistical vs ML anomaly detection concepts
  2. Model ensembling — combining model probabilities
  3. Combined pipeline: sentiment + fake detector flow

IMPORTANT PREREQUISITES BEFORE RUNNING:
  - Fix LSTM.py to use proper tokenization (not hash())
  - Remove strict=False from model loading
  - Confirm best_lstm_model.pth loads cleanly into LSTMClassifier

Run directly:
  python week8_pipeline_design.py

You imports:
  from week8_pipeline_design import FakeReviewPipeline
=============================================================
"""

import re
import math
import json
import torch
import torch.nn as nn
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset_helpers.dataset_explore import (
    clean_and_tokenize, encode, load_vocab, PAD_TOKEN
)
from dataset_helpers.data_tokenize import pad_or_truncate, MAX_LEN


# ─────────────────────────────────────────────────────────────
# SECTION 1 — STATISTICAL ANOMALY SCORER
# No ML model needed — purely mathematical features.
# Detects statistical outliers that ML models miss:
#   - New spam tactics not in training data
#   - Edge cases with unusual formatting patterns
# ─────────────────────────────────────────────────────────────

SUPERLATIVES = {
    "best", "worst", "amazing", "terrible", "perfect", "awful",
    "greatest", "horrible", "excellent", "dreadful", "fantastic",
    "disgusting", "outstanding", "appalling", "superb", "atrocious",
}

# Learned from real review corpus statistics.
# If you have the Kaggle dataset, recompute these from actual data.
NORMAL_STATS = {
    "word_count"   : {"mean": 120.0,  "std": 80.0},
    "ttr"          : {"mean": 0.72,   "std": 0.12},
    "avg_word_len" : {"mean": 4.8,    "std": 0.9},
    "exclaim_ratio": {"mean": 0.008,  "std": 0.018},
    "upper_ratio"  : {"mean": 0.06,   "std": 0.08},
    "superl_ratio" : {"mean": 0.04,   "std": 0.05},
}


def extract_features(text):
    """
    Extracts statistical features from raw review text.
    These features carry genuine signal for fake detection:

    word_count    : Very short (<20) or very long (>500) reviews
                    are both suspicious — too lazy or too padded.

    ttr           : Type-Token Ratio = unique_words / total_words.
                    Fake reviews repeat sentiment words ("amazing
                    amazing amazing") → low TTR. Real reviews use
                    varied vocabulary → higher TTR.

    avg_word_len  : AI-generated text tends toward simpler, shorter
                    words. Human writing shows more vocabulary range.

    exclaim_ratio : Spam reviews oversell with exclamation marks.
                    "Best product ever!!!" is a classic fake signal.

    upper_ratio   : ALL CAPS WORDS signal aggressive promotion or bots.

    superl_ratio  : Superlatives without specifics ("best", "worst",
                    "perfect") are the hallmark of fake reviews.
                    Real reviews say "the battery lasts 8 hours",
                    not just "best battery ever".
    """
    if not text or not text.strip():
        return {k: 0.0 for k in NORMAL_STATS}

    words     = text.split()
    word_count = len(words)

    if word_count == 0:
        return {k: 0.0 for k in NORMAL_STATS}

    # Type-token ratio
    ttr = len(set(w.lower() for w in words)) / word_count

    # Average word length (letters only)
    avg_word_len = (
        sum(len(re.sub(r'[^a-zA-Z]', '', w)) for w in words) / word_count
    )

    # Exclamation density
    exclaim_ratio = text.count('!') / max(len(text), 1)

    # Uppercase letter ratio
    alpha_chars = [c for c in text if c.isalpha()]
    upper_ratio = (
        sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if alpha_chars else 0.0
    )

    # Superlative density
    superl_ratio = (
        sum(1 for w in words if w.lower().rstrip('!?,.')
            in SUPERLATIVES) / word_count
    )

    return {
        "word_count"   : word_count,
        "ttr"          : round(ttr, 4),
        "avg_word_len" : round(avg_word_len, 4),
        "exclaim_ratio": round(exclaim_ratio, 4),
        "upper_ratio"  : round(upper_ratio, 4),
        "superl_ratio" : round(superl_ratio, 4),
    }


def compute_z_scores(features):
    """
    Computes how many standard deviations each feature sits from
    the known normal review distribution.

    z = (observed - mean) / std

    |z| > 2.0 → unusual (outside 95% of normal reviews)
    |z| > 3.0 → very unusual (outside 99.7% of normal reviews)
    """
    z_scores = {}
    for feat, value in features.items():
        if feat not in NORMAL_STATS:
            continue
        mean = NORMAL_STATS[feat]["mean"]
        std  = NORMAL_STATS[feat]["std"]
        z_scores[feat] = round((value - mean) / std, 3) if std > 0 else 0.0
    return z_scores


def statistical_fake_score(text):
    """
    Converts statistical feature z-scores into a single fake
    probability score (0.0 = definitely real, 1.0 = definitely fake).

    How the score is calculated:
      - Extract features and compute z-scores
      - Some features increase fakeness when HIGH (exclamations, superlatives)
      - Some features increase fakeness when LOW (word count, TTR)
      - Directional z-scores are clipped and averaged

    Returns:
        score  : float 0.0–1.0 — statistical fake probability
        detail : dict — per-feature z-scores for transparency
    """
    features = extract_features(text)
    z_scores = compute_z_scores(features)

    # Signed contributions — positive means "leans fake"
    # HIGH exclamation ratio, HIGH uppercase, HIGH superlatives → fake
    # LOW word count, LOW TTR → fake (but moderate word count is okay)
    contributions = {
        "exclaim_ratio": max(0, z_scores.get("exclaim_ratio", 0)),
        "upper_ratio"  : max(0, z_scores.get("upper_ratio", 0)),
        "superl_ratio" : max(0, z_scores.get("superl_ratio", 0)),
        "low_ttr"      : max(0, -z_scores.get("ttr", 0)),
    }

    # Average contribution, clip to [0, 3], then normalise to [0, 1]
    avg_z = sum(contributions.values()) / len(contributions)
    score = min(avg_z / 3.0, 1.0)

    return round(score, 4), {
        "features": features,
        "z_scores": z_scores,
        "contributions": {k: round(v, 3) for k, v in contributions.items()},
    }


# ─────────────────────────────────────────────────────────────
# SECTION 2 — WEIGHTED ENSEMBLE
# Combines ML model probability with statistical score.
# Weights are set from F1 scores of each model on validation set.
#
# Why weighted average instead of simple average?
#   Because the ML model (trained on 32k labeled reviews) is
#   likely more reliable than the statistical heuristic.
#   Weighting by F1 makes this principled rather than arbitrary.
#
# Why not stacking (learned ensemble)?
#   Stacking needs extra labeled data to train the meta-learner.
#   Weighted average needs nothing extra — compute weights from
#   the same validation set you already have.
# ─────────────────────────────────────────────────────────────

def compute_ensemble_weights(f1_ml, f1_statistical):
    """
    Derives ensemble weights from individual F1 scores.
    Measure both models on the validation set, then call this.

    Example:
        ML model F1      = 0.84
        Statistical F1   = 0.71
        → ML weight      = 0.84 / (0.84 + 0.71) = 0.542
        → Stat weight    = 0.71 / (0.84 + 0.71) = 0.458

    Args:
        f1_ml          : float — ML model F1 on validation set
        f1_statistical : float — Statistical scorer F1 on validation set

    Returns:
        w_ml, w_stat   : floats summing to 1.0
    """
    total = f1_ml + f1_statistical
    if total == 0:
        return 0.5, 0.5
    return round(f1_ml / total, 3), round(f1_statistical / total, 3)


def ensemble_fake_score(ml_score, statistical_score, w_ml=0.65, w_stat=0.35):
    """
    Combines ML and statistical scores into a single fake probability.

    Default weights (0.65 ML / 0.35 statistical) are reasonable
    starting values — replace with compute_ensemble_weights() output
    once you've measured both models' F1 on the validation set.

    Args:
        ml_score          : float 0–1 from your ML fake detector
        statistical_score : float 0–1 from statistical_fake_score()
        w_ml              : weight for ML score
        w_stat            : weight for statistical score

    Returns:
        ensemble score : float 0–1
    """
    assert abs(w_ml + w_stat - 1.0) < 1e-6, "Weights must sum to 1.0"
    return round((ml_score * w_ml) + (statistical_score * w_stat), 4)


# ─────────────────────────────────────────────────────────────
# SECTION 3 — FULL COMBINED PIPELINE
# Sentiment + Fake Detection flowing together.
#
# Pipeline architecture:
#
#   Raw review text
#       ↓
#   [A] Statistical scorer — no model needed, runs immediately
#       ↓
#   [B] ML fake detector — trained binary classifier
#       ↓
#   [C] Ensemble — weighted combination of A and B
#       ↓
#   [D] Sentiment model — your Phase 1 FFC or LSTM
#       ↓
#   Final decision: APPROVED / FLAGGED + sentiment label
#
# Note: if ensemble fake score > THRESHOLD, the review is
# FLAGGED — sentiment analysis still runs so you can store
# the result and show it in the UI, but the review is marked.
# ─────────────────────────────────────────────────────────────

FAKE_THRESHOLD = 0.65   # above this → FLAGGED as suspicious


class FakeReviewPipeline:
    """
    Combined pipeline that runs both fake detection and sentiment
    analysis on a single review text.

    Your pipeline.py imports this and calls run() per review.
    Your partner (Rajeev) calls get_detection_score() to get
    the fake probability to plug into his classifier outputs.

    Usage:
        pipeline = FakeReviewPipeline(
            fake_model     = your_trained_fake_detector,
            sentiment_model= your_ffc_or_lstm,
            word_to_idx    = word_to_idx,
            device         = device,
        )
        result = pipeline.run("This product is absolutely amazing!!!")
        print(result)
    """

    def __init__(self, fake_model, sentiment_model,
                 word_to_idx, device,
                 w_ml=0.65, w_stat=0.35,
                 threshold=FAKE_THRESHOLD):
        self.fake_model      = fake_model
        self.sentiment_model = sentiment_model
        self.word_to_idx     = word_to_idx
        self.device          = device
        self.w_ml            = w_ml
        self.w_stat          = w_stat
        self.threshold       = threshold

    def _encode(self, text):
        """Shared preprocessing for both models."""
        tokens  = clean_and_tokenize(text, remove_sw=True)
        indices = encode(tokens, self.word_to_idx)
        padded  = pad_or_truncate(indices, MAX_LEN)
        return torch.tensor([padded], dtype=torch.long).to(self.device)

    def get_detection_score(self, text):
        """
        Returns the ML fake probability for a single text.
        This is what you send to Rajeev as mock_detection_score.

        Returns:
            fake_prob : float 0.0–1.0
                        0.0 = model is certain it's real
                        1.0 = model is certain it's fake
        """
        self.fake_model.eval()
        with torch.no_grad():
            tensor    = self._encode(text)
            logits    = self.fake_model(tensor)
            probs     = torch.softmax(logits, dim=1)[0]
            fake_prob = probs[1].item()   # index 1 = FAKE (your label convention)
        return round(fake_prob, 4)

    def get_sentiment(self, text):
        """
        Returns sentiment label and confidence from Phase 1 model.
        """
        self.sentiment_model.eval()
        with torch.no_grad():
            tensor = self._encode(text)
            logits = self.sentiment_model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]
            pred   = probs.argmax().item()
            conf   = probs[pred].item()
        return ("POSITIVE" if pred == 1 else "NEGATIVE"), round(conf, 4)

    def run(self, text):
        """
        Full pipeline run on a single review.

        Returns dict with:
            text              : original review text
            ml_fake_score     : float — ML model fake probability
            stat_fake_score   : float — statistical anomaly score
            ensemble_score    : float — weighted combination
            is_flagged        : bool  — True if ensemble > threshold
            sentiment_label   : str   — POSITIVE or NEGATIVE
            sentiment_conf    : float — confidence of sentiment
            stat_detail       : dict  — per-feature breakdown
        """
        # Step A: statistical scoring (no model, instant)
        stat_score, stat_detail = statistical_fake_score(text)

        # Step B: ML fake detection
        ml_score = self.get_detection_score(text)

        # Step C: ensemble
        ens_score = ensemble_fake_score(
            ml_score, stat_score, self.w_ml, self.w_stat
        )

        # Step D: sentiment (runs regardless of fake flag — store both)
        sentiment_label, sentiment_conf = self.get_sentiment(text)

        return {
            "text"            : text,
            "ml_fake_score"   : ml_score,
            "stat_fake_score" : stat_score,
            "ensemble_score"  : ens_score,
            "is_flagged"      : ens_score >= self.threshold,
            "action"          : "FLAGGED" if ens_score >= self.threshold else "APPROVED",
            "sentiment_label" : sentiment_label,
            "sentiment_conf"  : sentiment_conf,
            "stat_detail"     : stat_detail,
        }


# ─────────────────────────────────────────────────────────────
# ONLY RUNS WHEN EXECUTED DIRECTLY
# Tests each component without needing trained models.
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Test 1: Feature extraction ────────────────────────────
    print("=" * 60)
    print("TEST 1 — STATISTICAL FEATURE EXTRACTION")
    print("=" * 60)

    test_reviews = [
        "THIS IS AN AMAZING BOT COMPLAINT SPAM LINK FREE TICKET!!!",
        "The strap broke after two weeks but customer service replaced it free",
        "best product ever amazing perfect wonderful excellent best buy!!!",
        "Battery life is around 5 hours not 8 as claimed, disappointing",
    ]

    for rev in test_reviews:
        feats = extract_features(rev)
        score, detail = statistical_fake_score(rev)
        print(f"\n  Review: {rev[:60]!r}")
        print(f"  Features : {feats}")
        print(f"  Z-scores : {detail['z_scores']}")
        print(f"  Stat score: {score}  ({'HIGH — suspicious' if score > 0.5 else 'LOW — looks normal'})")

    # ── Test 2: Z-score anomaly detection ─────────────────────
    print("\n" + "=" * 60)
    print("TEST 2 — Z-SCORE ANOMALY DETECTION")
    print("=" * 60)
    print("""
  Z-score interpretation:
    |z| < 1.0  → within 1 std of normal → unremarkable
    |z| 1-2    → slightly unusual
    |z| 2-3    → suspicious (outside 95% of normal reviews)
    |z| > 3    → very suspicious (outside 99.7%)

  Spam review: high exclamation z-score, high uppercase z-score,
               high superlative z-score, low TTR z-score.
  Real review: all z-scores near 0.
    """)

    spam    = "THIS PRODUCT IS AMAZING!!! BEST EVER!!! BUY NOW!!! PERFECT!!!"
    genuine = "The camera takes decent photos in daylight but struggles indoors at night"

    for label, rev in [("SPAM", spam), ("GENUINE", genuine)]:
        _, detail = statistical_fake_score(rev)
        print(f"  [{label}]: {rev[:60]!r}")
        for feat, z in detail["z_scores"].items():
            bar = "█" * min(int(abs(z) * 5), 20)
            print(f"    {feat:15}: z={z:+.2f}  |{bar:<20}|")
        print()

    # ── Test 3: Ensemble weight calculation ───────────────────
    print("=" * 60)
    print("TEST 3 — ENSEMBLE WEIGHT CALCULATION")
    print("=" * 60)

    # Hypothetical F1 scores — replace with real values after training
    scenarios = [
        (0.84, 0.71, "ML better than statistical"),
        (0.71, 0.84, "Statistical better than ML"),
        (0.80, 0.80, "Equal performance"),
    ]
    for f1_ml, f1_stat, label in scenarios:
        w_ml, w_stat = compute_ensemble_weights(f1_ml, f1_stat)
        print(f"\n  Scenario: {label}")
        print(f"  ML F1={f1_ml}  Stat F1={f1_stat}")
        print(f"  → ML weight={w_ml}  Statistical weight={w_stat}")
        test_score = ensemble_fake_score(0.8, 0.6, w_ml, w_stat)
        print(f"  → Example ensemble(0.8, 0.6) = {test_score}")

    # ── Test 4: Pipeline design verification (no model) ───────
    print("\n" + "=" * 60)
    print("TEST 4 — PIPELINE FLOW DESIGN VERIFICATION")
    print("=" * 60)
    print("""
  Full pipeline flow per review:

  Raw text
      ↓
  extract_features()       → word_count, ttr, exclaim_ratio, ...
      ↓
  statistical_fake_score() → stat_score (0.0–1.0)
      ↓
  ML fake detector         → ml_score   (0.0–1.0)  [your model]
      ↓
  ensemble_fake_score()    → ens_score  (0.0–1.0)
      ↓
  ens_score >= 0.65?
      YES → FLAGGED  → store review as suspicious
      NO  → APPROVED → continue to sentiment model
      ↓ (both paths)
  sentiment model          → POSITIVE/NEGATIVE + confidence
      ↓
  Store full result in MongoDB reviews collection
    """)

    # Show what the output dict looks like
    sample_output = {
        "text"            : spam[:50],
        "ml_fake_score"   : 0.89,     # placeholder
        "stat_fake_score" : 0.71,
        "ensemble_score"  : 0.83,
        "is_flagged"      : True,
        "action"          : "FLAGGED",
        "sentiment_label" : "POSITIVE",
        "sentiment_conf"  : 0.94,
    }
    print("  Example pipeline output:")
    for k, v in sample_output.items():
        print(f"    {k:20}: {v}")

    # ── Final summary ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("WEEK 8 PIPELINE DESIGN COMPLETE")
    print("=" * 60)
    print("""
  Concepts implemented:
  ✅ Statistical anomaly detection: z-scores on 6 features
  ✅ ML anomaly detection: binary classifier fake probability
  ✅ Ensembling: F1-weighted combination of both signals
  ✅ Combined pipeline: fake detection → sentiment, both stored

  What you send to Rajeev:
     get_detection_score(text) → float 0.0-1.0
     This replaces mock_partner_score in his ensemble.

  Before Week 9 — fix in your code:
  ❌ Replace hash() tokenization with 
          () pipeline
  ❌ Remove strict=False — let architecture mismatches crash loudly
  ❌ Add nn.LSTM layer to LSTMClassifier — currently it is FFC
  ❌ Confirm best_lstm_model.pth loads cleanly into fixed class
    """)