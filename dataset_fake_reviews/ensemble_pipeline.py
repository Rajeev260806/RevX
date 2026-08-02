import os
import json
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Maintain clean directory structural insertion
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import your friend's Week 9 binary model architecture and feature extractor
from FRD_BC import FakeReviewDetector, extract_features

# Import your Week 8 mathematical blending and statistical scoring functions
from ensemble import statistical_fake_score, ensemble_fake_score

from dataset_helpers.dataset_explore import clean_and_tokenize, encode, load_vocab
from dataset_helpers.data_tokenize import pad_or_truncate, MAX_LEN

MODEL_PATH = Path(__file__).parent / "fake_detector.pth"

# ─────────────────────────────────────────────────────────────
# 1. DATABASE ROUTING MANAGER (INTERCEPTION & TRANSPARENCY LOGGING)
# ─────────────────────────────────────────────────────────────
class ProductionDatabase:
    """Handles operational routing by blocking or approving review commits."""
    def __init__(self):
        self.live_reviews = []          # Main public review feed
        self.rejected_audit_logs = []   # Transparency/Audit logs for intercepted items

    def save_review(self, pipeline_output):
        """Intercepts reviews before saving to the live site based on the pipeline decision."""
        if pipeline_output["action"] == "REJECTED":
            self.rejected_audit_logs.append(pipeline_output)
            return " REJECTED: Review blocked and routed to Transparency Audit Logs."
        else:
            self.live_reviews.append(pipeline_output)
            return " APPROVED: Review committed safely to Public Production Feed."


# ─────────────────────────────────────────────────────────────
# 2. OPERATIONAL PIPELINE PIPING
# ─────────────────────────────────────────────────────────────
class OperationalReviewPipeline:
    def __init__(self, fake_model_path, word_to_idx, device, threshold=0.50, w_ml=0.10, w_stat=0.90):
        self.device = device
        self.word_to_idx = word_to_idx
        self.threshold = threshold  # Tuned threshold floor evaluated from validation tuning
        self.w_ml = w_ml            # Ensemble weights for ML
        self.w_stat = w_stat        # Ensemble weights for Statistical Score
        
        # Initialize and load your friend's PyTorch Binary Model
        vocab_size = len(word_to_idx)
        self.fake_model = FakeReviewDetector(vocab_size=vocab_size).to(device)
        if os.path.exists(fake_model_path):
            self.fake_model.load_state_dict(torch.load(fake_model_path, map_location=device))
        self.fake_model.eval()

    def get_ml_score(self, text):
        """Processes text tokens and features through your friend's single-logit binary classifier."""
        with torch.no_grad():
            tokens = clean_and_tokenize(text, remove_sw=True)
            indices = encode(tokens, self.word_to_idx)
            padded = pad_or_truncate(indices, MAX_LEN)
            
            X = torch.tensor([padded], dtype=torch.long).to(self.device)
            feats = extract_features(text).unsqueeze(0).to(self.device)
            
            # Aligned to your friend's single-logit layout: calculate probability via Sigmoid
            logit = self.fake_model(X, feats)
            prob = torch.sigmoid(logit).item()
        return round(prob, 4)

    def evaluate_review(self, text):
        """Blends scores via your ensemble.py functions and applies systematic action rules."""
        # 1. Fetch ML score from friend's model framework
        ml_score = self.get_ml_score(text)
        
        # 2. Fetch Statistical anomaly score from your ensemble.py logic
        stat_score, stat_detail = statistical_fake_score(text)
        
        # 3. Calculate blended probability using your ensemble.py calculation rules
        final_score = ensemble_fake_score(ml_score, stat_score, self.w_ml, self.w_stat)
        # Core Week 9 Interception Filtering Rule
        action = "REJECTED" if final_score > self.threshold else "APPROVED"
        
        return {
            "text": text,
            "ml_model_probability": ml_score,
            "statistical_anomaly_score": stat_score,
            "ensemble_combined_score": final_score,
            "action": action,
            "stat_breakdown": stat_detail.get("z_scores", {}),
            "rejection_log_reason": "High anomaly/spam metric threshold breach" if action == "REJECTED" else None
        }


# ─────────────────────────────────────────────────────────────
# 3. COMPREHENSIVE VALIDATION SUITE (10 TEST REVIEWS)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    db = ProductionDatabase()
    
    # Load application vocabulary baseline
    try:
        word_to_idx, _ = load_vocab()
    except:
        word_to_idx = {"<PAD>": 0, "best": 1, "amazing": 2} # Resilient fallback context

    w_ml   = 0.60
    w_stat = 0.40

    pipeline = OperationalReviewPipeline(
        fake_model_path=MODEL_PATH,
        word_to_idx=word_to_idx,
        device=device,
        threshold=0.50,
        w_ml = w_ml,
        w_stat= w_stat  # Aligned with verified validation optimal targets
    )
    
    # Validation suite containing a variety of reviews (Targeted bots, humans, complaints)
    test_reviews = [
        "CLICK HERE NOW TO WIN A FREE CASH GIFT CARD!!! BEST DEALS!!! LINK SPAM!!!",
        "AMAZING PERFECT EXCELLENT FANTASTIC OUTSTANDING WONDERFUL BEST APP EVER!!!",
        "The battery lasts roughly five hours, which is much lower than the eight hours advertised.",
        "FREE MOVIE TICKETS FREE DISCOUNTS CLICK THIS LINK IMMEDIATELY FOR CASH CODE!!!",
        "The screen housing came with a minor scratch on the lower right border frame.",
        "BEST DEAL QUALITY AMAZING AWESOME SPECTACULAR BUY NOW CHEAP DEAL!!!",
        "The database interface drops connection frequently whenever executing large multi-row updates.",
        "CLAIM REWARD NOW EXCELLENT PROMO CODE DISCOUNTS AVAILABLE AT THIS SITE NOW!!!",
        "It arrived four days late, but customer service answered my email in ten minutes.",
        "The packaging was crushed completely during shipping transit, ruining the item box."
    ]
    
    print("=" * 75)
    print("RUNNING OPERATIONAL ENSEMBLE SYSTEM CHECKPOINT (10 TEST CASES)")
    print("=" * 75)
    
    for idx, review in enumerate(test_reviews, 1):
        evaluation_result = pipeline.evaluate_review(review)
        db_feedback = db.save_review(evaluation_result)
        
        print(f"\n[Review #{idx}]: {review[:65]}...")
        print(f"   Ensemble Score: {evaluation_result['ensemble_combined_score']:.4f} : Action: {evaluation_result['action']}")
        print(f"   Data Storage Route: {db_feedback}")

    print("\n" + "=" * 75)  
    print("OPERATIONAL INFRASTRUCTURE TELEMETRY METRICS")
    print("=" * 75)
    print(f" Total Live Product Table Rows Written: {len(db.live_reviews)}")
    print(f" Total Audit Log Transparency Triggers: {len(db.rejected_audit_logs)}")