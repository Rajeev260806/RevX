import json
import torch
import torch.nn as nn
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
 
from dataset_helpers.dataset_explore import load_vocab, PAD_TOKEN
from dataset_helpers.data_tokenize   import MAX_LEN
from fake_review_detector import (
    get_fake_review_splits,
    get_mock_fake_review_data,
    get_fake_review_data,
    split_fake_review_data,
)
from dataset_helpers.dataset_explore import clean_and_tokenize, encode
from dataset_helpers.data_tokenize   import pad_or_truncate

EMBED_DIM = 128
HIDDEN_DIM = 64
DROPOUT_P = 0.3
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
BATCH_SIZE = 32
PATIENCE = 3

DEFAULT_THRESHOLD = 0.5
 
MODEL_PATH  = Path(__file__).parent / "fake_detector.pth"
CONFIG_PATH = Path(__file__).parent / "fake_detector_config.json"

SUPERLATIVES = {
    "best","worst","amazing","terrible","perfect","awful",
    "greatest","horrible","excellent","dreadful","fantastic",
    "outstanding","superb","incredible","unbelievable",
}

def extract_features(text):
    if not text or not text.strip():
        return torch.zeros(6)
 
    words = text.split()
    n     = max(len(words), 1)
 
    word_count_norm = min(n / 500.0, 1.0)            #Normalised word count
    ttr = len(set(w.lower() for w in words)) / n      #type-token ratio
    exclaim = min(text.count('!') / max(len(text), 1) * 50, 1.0)  # Exclamatory density
    alpha = [c for c in text if c.isalpha()]
    upper = sum(1 for c in alpha if c.isupper()) / max(len(alpha), 1) #Uppercase ratio
 
    superl = sum(1 for w in words if w.lower().rstrip('!?,.')
                 in SUPERLATIVES) / n                                #Superlative density
 
    avg_wl = sum(len(w) for w in words) / n / 10.0   # Avg word length
 
    return torch.tensor(
        [word_count_norm, ttr, exclaim, upper, superl, avg_wl],
        dtype=torch.float32
    )
 
 
NUM_EXTRA_FEATURES = 6

class FakeReviewDetector(nn.Module):

    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout_p=DROPOUT_P, pad_idx=0):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embed_dim,padding_idx=pad_idx)
        combined_dim = embed_dim + NUM_EXTRA_FEATURES
        self.hidden = nn.Linear(combined_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self,token_ids,extra_features=None):
        embedded = self.embedding(token_ids)      
        pooled   = embedded.mean(dim=1)

        if extra_features is not None:
            combined = torch.cat([pooled, extra_features], dim=1)
        else:
            zeros    = torch.zeros(
                pooled.size(0), NUM_EXTRA_FEATURES,
                device=pooled.device
            )
            combined = torch.cat([pooled, zeros], dim=1)

        hidden = self.relu(self.hidden(combined)) 
        drop   = self.dropout(hidden)
        logit  = self.classifier(drop)             
        return logit.squeeze(1) 
    
def build_dataloaders(use_mock=False, batch_size=BATCH_SIZE):
    from torch.utils.data import TensorDataset, DataLoader
 
    tr_enc, tr_lbl, va_enc, va_lbl, te_enc, te_lbl = get_fake_review_splits(use_mock=use_mock)

    if use_mock:
        raw_reviews, raw_labels = get_mock_fake_review_data()
        import random
        random.seed(42)
        paired = list(zip(raw_reviews, raw_labels))
        random.shuffle(paired)
        raw_reviews, raw_labels = zip(*paired)
        raw_reviews = list(raw_reviews)
        tr_raw = raw_reviews[:6]
        va_raw = raw_reviews[6:7]
        te_raw = raw_reviews[7:]
    else:
        raw_reviews, raw_labels = get_fake_review_data()
        tr_raw, _, va_raw, _, te_raw, _ = split_fake_review_data(raw_reviews, raw_labels)

    def make_feature_tensor(raw_texts):
        return torch.stack([extract_features(t) for t in raw_texts])
 
    def to_dataset(encodings, labels, raw_texts):
        X = torch.tensor(encodings, dtype=torch.long)
        feats = make_feature_tensor(raw_texts)
        y = torch.tensor(labels, dtype=torch.float32)  
        return TensorDataset(X, feats, y)
 
    train_ds = to_dataset(tr_enc, tr_lbl, tr_raw)
    val_ds   = to_dataset(va_enc, va_lbl, va_raw)
    test_ds  = to_dataset(te_enc, te_lbl, te_raw)
 
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
 
    word_to_idx, _ = load_vocab()
    return train_loader, val_loader, test_loader, len(word_to_idx)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
 
    for X, feats, y in loader:
        X, feats, y = X.to(device), feats.to(device), y.to(device)
 
        logits = model(X, feats)          
        loss   = criterion(logits, y)     
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        preds          = (torch.sigmoid(logits) >= 0.5).float()
        total_correct += (preds == y).sum().item()
        total_loss    += loss.item() * X.size(0)
        total_samples += X.size(0)
 
    return total_loss / total_samples, total_correct / total_samples

def evaluate(model, loader, criterion, device, threshold=DEFAULT_THRESHOLD):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
 
    with torch.no_grad():
        for X, feats, y in loader:
            X, feats, y = X.to(device), feats.to(device), y.to(device)
            logits       = model(X, feats)
            loss         = criterion(logits, y)
            preds        = (torch.sigmoid(logits) >= threshold).float()
 
            total_loss    += loss.item() * X.size(0)
            total_correct += (preds == y).sum().item()
            total_samples += X.size(0)
 
    return total_loss / total_samples, total_correct / total_samples

def collect_predictions(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
 
    with torch.no_grad():
        for X, feats, y in loader:
            X, feats = X.to(device), feats.to(device)
            logits   = model(X, feats)
            probs    = torch.sigmoid(logits).cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(y.tolist())
 
    return all_probs, all_labels

def train(model, train_loader, val_loader, device,lr=LEARNING_RATE, num_epochs=NUM_EPOCHS, patience=PATIENCE):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
 
    best_val_loss    = float("inf")
    epochs_no_improve= 0
    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>8}  {'Note':>12}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*12}")
 
    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
 
        note = ""
        if va_loss < best_val_loss:
            best_val_loss     = va_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
            note = "saved"
        else:
            epochs_no_improve += 1
            note = f"no improve {epochs_no_improve}/{patience}"
 
        print(f"  {epoch:>5}  {tr_loss:>10.4f}  {tr_acc:>8.2%}  "
              f"{va_loss:>8.4f}  {va_acc:>8.2%}  {note:>12}")
 
        if epochs_no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch}.")
            break
 
    print(f"\n  Best val loss : {best_val_loss:.4f}")
    print(f"  Model saved   : {MODEL_PATH}")

def compute_metrics_at_threshold(probs, labels, threshold):
    tp = sum(p >= threshold and l == 1 for p, l in zip(probs, labels))
    tn = sum(p <  threshold and l == 0 for p, l in zip(probs, labels))
    fp = sum(p >= threshold and l == 0 for p, l in zip(probs, labels))
    fn = sum(p <  threshold and l == 1 for p, l in zip(probs, labels))
 
    total     = tp + tn + fp + fn
    accuracy  = (tp + tn) / total    if total > 0         else 0.0
    precision = tp / (tp + fp)       if (tp + fp) > 0     else 0.0
    recall    = tp / (tp + fn)       if (tp + fn) > 0     else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
 
    return dict(threshold=threshold, accuracy=accuracy,
                precision=precision, recall=recall, f1=f1,
                tp=tp, tn=tn, fp=fp, fn=fn)

def tune_threshold(probs, labels, thresholds=None):
    """
    Evaluates multiple thresholds and prints a comparison table.
    Returns the threshold with the best F1 score.
 
    This is the core of Week 9's threshold tuning task.
    Run this on the VALIDATION set, never the test set.
    """
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8]
 
    print(f"\n  {'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  "
          f"{'F1':>8}  {'Accuracy':>9}  {'TP':>5}  {'FP':>5}  {'FN':>5}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*5}  {'-'*5}  {'-'*5}")
 
    best_f1, best_threshold, best_metrics = -1.0, DEFAULT_THRESHOLD, {}
 
    for t in thresholds:
        m = compute_metrics_at_threshold(probs, labels, t)
        marker = " ← best F1" if m["f1"] > best_f1 else ""
 
        if m["f1"] > best_f1:
            best_f1, best_threshold, best_metrics = m["f1"], t, m
 
        print(f"  {t:>10.2f}  {m['precision']:>10.4f}  {m['recall']:>8.4f}  "
              f"{m['f1']:>8.4f}  {m['accuracy']:>9.4f}  "
              f"{m['tp']:>5}  {m['fp']:>5}  {m['fn']:>5}{marker}")
 
    print(f"\n  Best threshold : {best_threshold}")
    print(f"  Best F1        : {best_f1:.4f}")
    print(f"\n  Interpretation:")
    print(f"  At threshold={best_threshold}:")
    print(f"    {best_metrics['tp']} real fake reviews CAUGHT")
    print(f"    {best_metrics['fp']} real reviews WRONGLY flagged")
    print(f"    {best_metrics['fn']} fake reviews MISSED")
 
    return best_threshold, best_metrics

def predict_fake(text, model, word_to_idx, device,threshold=DEFAULT_THRESHOLD, max_len=MAX_LEN):
    model.eval()
    with torch.no_grad():
        tokens  = clean_and_tokenize(text, remove_sw=True)
        indices = encode(tokens, word_to_idx)
        padded  = pad_or_truncate(indices, max_len)
        X       = torch.tensor([padded], dtype=torch.long).to(device)
        feats = extract_features(text).unsqueeze(0).to(device)  
        logit     = model(X, feats)
        fake_prob = torch.sigmoid(logit).item()
 
    label = "FAKE" if fake_prob >= threshold else "REAL"
    return label, round(fake_prob, 4), extract_features(text).tolist() 
if __name__ == "__main__":
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    word_to_idx, _ = load_vocab()
    vocab_size      = len(word_to_idx)
    pad_idx         = word_to_idx[PAD_TOKEN]
 
    # ── Test 1: Feature engineering ──────────────────────────
    print("=" * 60)
    print("TEST 1 — FEATURE ENGINEERING")
    print("=" * 60)
 
    test_reviews = [
        ("FAKE", "THIS IS AN AMAZING BOT SPAM LINK FREE TICKET AMAZING!!!"),
        ("FAKE", "best product ever amazing perfect wonderful excellent best!!!"),
        ("REAL", "The strap broke after two weeks, customer service replaced it"),
        ("REAL", "Battery life is 5 hours not 8 as advertised, disappointing"),
    ]
 
    feature_names = [
        "word_count_norm", "ttr", "exclaim", "upper", "superl", "avg_wl"
    ]
 
    print(f"\n  {'Label':>5}  {feature_names[0]:>15}  {feature_names[1]:>5}  "
          f"{feature_names[2]:>8}  {feature_names[3]:>6}  "
          f"{feature_names[4]:>6}  {feature_names[5]:>6}")
    print(f"  {'-'*5}  {'-'*15}  {'-'*5}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*6}")
 
    for label, rev in test_reviews:
        f = extract_features(rev).tolist()
        print(f"  {label:>5}  {f[0]:>15.3f}  {f[1]:>5.3f}  "
              f"{f[2]:>8.4f}  {f[3]:>6.3f}  {f[4]:>6.3f}  {f[5]:>6.3f}")
        print(f"         {rev[:60]!r}")
 
    # ── Test 2: Model architecture ────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 2 — MODEL ARCHITECTURE")
    print("=" * 60)
 
    model = FakeReviewDetector(
        vocab_size = vocab_size,
        embed_dim  = EMBED_DIM,
        hidden_dim = HIDDEN_DIM,
        dropout_p  = DROPOUT_P,
        pad_idx    = pad_idx,
    ).to(device)
 
    print(f"\n  Vocab size   : {vocab_size:,}")
    print(f"  Architecture : {model}")
    print(f"\n  Total params : {sum(p.numel() for p in model.parameters()):,}")
    print(f"\n  KEY DIFFERENCES from Phase 1 SentimentClassifier:")
    print(f"  - Output: 1 logit (not 2) → apply sigmoid for probability")
    print(f"  - Loss: BCEWithLogitsLoss (not CrossEntropyLoss)")
    print(f"  - Extra input: 6 hand-crafted features concatenated to embedding")
 
    # Forward pass shape test
    dummy_X     = torch.randint(0, vocab_size, (4, MAX_LEN)).to(device)
    dummy_feats = torch.rand(4, NUM_EXTRA_FEATURES).to(device)
 
    with torch.no_grad():
        dummy_logits = model(dummy_X, dummy_feats)
        dummy_probs  = torch.sigmoid(dummy_logits)
 
    print(f"\n  Input tokens shape : {dummy_X.shape}")
    print(f"  Input feats shape  : {dummy_feats.shape}")
    print(f"  Output logits shape: {dummy_logits.shape}  ← (batch,) not (batch, 2)")
    print(f"  After sigmoid      : {dummy_probs.tolist()}")
 
    # ── Test 3: Single training step ─────────────────────────
    print("\n" + "=" * 60)
    print("TEST 3 — SINGLE TRAINING STEP")
    print("=" * 60)
 
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
 
    # Float labels required for BCEWithLogitsLoss (not long)
    dummy_labels = torch.tensor([1.0, 0.0, 1.0, 0.0]).to(device)
 
    model.train()
    w_before = model.classifier.weight.data.clone()
 
    logits = model(dummy_X, dummy_feats)
    loss   = criterion(logits, dummy_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 
    changed = not torch.equal(w_before, model.classifier.weight.data)
    print(f"\n  Loss value          : {loss.item():.4f}")
    print(f"  Weights updated     : {'✅ yes' if changed else '❌ no'}")
    print(f"\n  IMPORTANT: Labels are float32 for BCEWithLogitsLoss")
    print(f"  Phase 1 used long (int64) for CrossEntropyLoss")
    print(f"  This is the most common crash bug in binary classifiers")
 
    # ── Test 4: Full training on mock data ────────────────────
    print("\n" + "=" * 60)
    print("TEST 4 — FULL TRAINING LOOP (mock data)")
    print("=" * 60)
    print("  Using 8 mock reviews. Switch use_mock=False for Kaggle data.")
 
    model = FakeReviewDetector(
        vocab_size=vocab_size, embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM, dropout_p=DROPOUT_P, pad_idx=pad_idx,
    ).to(device)
 
    train_loader, val_loader, test_loader, _ = build_dataloaders(
        use_mock=False, batch_size=32
    )
 
    train(model, train_loader, val_loader, device)
 
    # ── Test 5: Threshold tuning on validation set ────────────
    print("\n" + "=" * 60)
    print("TEST 5 — THRESHOLD TUNING")
    print("=" * 60)
    print("  (Tuning on mock data — not meaningful, just verifying the logic)")
    print("  Run with use_mock=False for real tuning numbers\n")
 
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    criterion_eval = nn.BCEWithLogitsLoss()
 
    val_probs, val_labels = collect_predictions(model, val_loader, device)
    best_threshold, best_metrics = tune_threshold(val_probs, val_labels)
 
    # ── Test 6: Save model config ─────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 6 — SAVING MODEL + CONFIG")
    print("=" * 60)
 
    config = {
        "architecture"     : "FakeReviewDetector",
        "vocab_size"       : vocab_size,
        "embed_dim"        : EMBED_DIM,
        "hidden_dim"       : HIDDEN_DIM,
        "dropout_p"        : DROPOUT_P,
        "pad_idx"          : pad_idx,
        "best_threshold"   : best_threshold,
        "num_extra_features": NUM_EXTRA_FEATURES,
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n  Model  : {MODEL_PATH}")
    print(f"  Config : {CONFIG_PATH}")
    for k, v in config.items():
        print(f"    {k:22}: {v}")
 
    # ── Test 7: Inference on custom reviews ───────────────────
    print("\n" + "=" * 60)
    print("TEST 7 — INFERENCE WITH TUNED THRESHOLD")
    print("=" * 60)
 
    inference_reviews = [
        "THIS PRODUCT IS AMAZING!!! BEST EVER!!! BUY NOW!!!",
        "The battery lasts about 6 hours, less than advertised but acceptable",
        "perfect excellent amazing wonderful best quality highly recommend!!!",
        "Arrived 3 days late but packaging was fine and product works as described",
        "CLICK HERE FREE GIFT AMAZING DEAL LIMITED TIME OFFER BEST PRICE!!!",
    ]
 
    print()
    for rev in inference_reviews:
        label, prob, feats = predict_fake(
            rev, model, word_to_idx, device, threshold=best_threshold
        )
        bar = "█" * int(prob * 30)
        print(f"  Review   : {rev[:60]!r}")
        print(f"  Fake prob: {prob:.4f}  |{bar:<30}|  → {label}")
        print()