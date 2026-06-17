import torch
import torch.nn as nn
from pathlib import Path
import sys
import json
 
sys.path.insert(0, str(Path(__file__).parent.parent))
 
from dataset_helpers.dataset_explore import load_vocab, PAD_TOKEN, UNK_TOKEN
from dataset_helpers.data_tokenize import get_splits, MAX_LEN, tokenize_and_encode
from model.FeedForwardClassifier import (SentimentClassifier,build_dataloader,train,evaluate,EMBED_DIM,NUM_CLASSES,LEARNING_RATE,NUM_EPOCHS,BATCH_SIZE,PATIENCE,)

FINAL_MODEL_PATH = Path(__file__).parent / "final_model.pth"
TUNING_RESULTS_PATH = Path(__file__).parent / "tuning_results.json"

def collect_predictions(model,data_loader,device):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch_X,batch_Y in data_loader:
            batch_X = batch_X.to(device)
            logits  = model(batch_X)
            preds   = logits.argmax(dim=1).cpu().tolist()
            labels  = batch_Y.tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)
    return all_preds,all_labels

def compute_metrics(all_preds,all_labels):
    assert len(all_labels)==len(all_preds)

    tp = sum(p == 1 and l == 1 for p, l in zip(all_preds, all_labels))
    tn = sum(p == 0 and l == 0 for p, l in zip(all_preds, all_labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(all_preds, all_labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(all_preds, all_labels))

    total = tp+fp+fn+tn
    accuracy = (tn+tp)/total if total>0 else 0.0
    precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
    recall = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return dict(accuracy=accuracy, precision=precision,
                recall=recall, f1=f1,
                tp=tp, tn=tn, fp=fp, fn=fn, total=total)

def print_metrics(metrics, split_name="Validation"):
    print(f"\n  [{split_name.upper()} METRICS]")
    print(f"  {'─'*44}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}  ({metrics['accuracy']:.2%})")
    print(f"  Precision : {metrics['precision']:.4f}  ({metrics['precision']:.2%})")
    print(f"  Recall    : {metrics['recall']:.4f}  ({metrics['recall']:.2%})")
    print(f"  F1 Score  : {metrics['f1']:.4f}  ({metrics['f1']:.2%})")
    print(f"  {'─'*44}")
    print(f"  True  Positives (TP) : {metrics['tp']:>6,}  correctly called POSITIVE")
    print(f"  True  Negatives (TN) : {metrics['tn']:>6,}  correctly called NEGATIVE")
    print(f"  False Positives (FP) : {metrics['fp']:>6,}  negative reviews wrongly called POSITIVE")
    print(f"  False Negatives (FN) : {metrics['fn']:>6,}  positive reviews wrongly called NEGATIVE")
    print(f"  Total                : {metrics['total']:>6,}")


def print_confusion_matrix(metrics):
    tn, fp = metrics["tn"], metrics["fp"]
    fn, tp = metrics["fn"], metrics["tp"]
    col_w  = 14
 
    print(f"\n  CONFUSION MATRIX")
    print(f"  {'':20}  {'Predicted':^{col_w*2+3}}")
    print(f"  {'':20}  {'NEG':^{col_w}}  {'POS':^{col_w}}")
    print(f"  {'─'*52}")
    print(f"  {'Actual NEG':20}  {tn:^{col_w},}  {fp:^{col_w},}")
    print(f"  {'Actual POS':20}  {fn:^{col_w},}  {tp:^{col_w},}")
    print(f"  {'─'*52}")
 
    fp_pct = fp / (tn + fp) * 100 if (tn + fp) > 0 else 0
    fn_pct = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0
    print(f"\n  Your model wrongly calls {fp_pct:.1f}% of negatives positive  (FP rate)")
    print(f"  Your model misses {fn_pct:.1f}% of positives entirely          (FN rate)")
 
    if fp > fn * 1.5:
        print(f"\n  Model leans POSITIVE — too many false positives")
    elif fn > fp * 1.5:
        print(f"\n  Model leans NEGATIVE — too many false negatives")
    else:
        print(f"\n  Model is roughly balanced between FP and FN")

class DropoutSentimentClassifier(nn.Module):
    
    def __init__(self, vocab_size, embed_dim, hidden_dim,num_classes, pad_idx=0, dropout_p=0.3):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.hidden     = nn.Linear(embed_dim, hidden_dim)
        self.relu       = nn.ReLU()
        self.dropout    = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x).mean(dim=1)   # (batch, embed_dim)
        hidden   = self.relu(self.hidden(embedded)) # (batch, hidden_dim)
        dropped  = self.dropout(hidden)             # dropout applied here
        return self.classifier(dropped)
    
def run_hyperparameter_search(train_loader, val_loader,vocab_size, pad_idx, device):
    search_grid = [
        {"embed_dim": 128, "hidden_dim": 64,  "dropout_p": 0.3, "lr": 1e-3},
        {"embed_dim": 128, "hidden_dim": 64,  "dropout_p": 0.5, "lr": 1e-3},
        {"embed_dim": 128, "hidden_dim": 128, "dropout_p": 0.3, "lr": 1e-3},
        {"embed_dim": 256, "hidden_dim": 128, "dropout_p": 0.3, "lr": 1e-3},
        {"embed_dim": 128, "hidden_dim": 64,  "dropout_p": 0.3, "lr": 5e-4},
    ]
    print(f"\n  Running {len(search_grid)} configurations...")
    print(f"  {'Cfg':>4}  {'embed':>5}  {'hidden':>6}  "
          f"{'drop':>4}  {'lr':>6}  {'val_f1':>8}  {'val_acc':>8}")
    print(f"  {'─'*58}")
 
    best_f1, best_config, all_results = -1.0, None, []
 
    for i, cfg in enumerate(search_grid, start=1):
        model = DropoutSentimentClassifier(
            vocab_size  = vocab_size,
            embed_dim   = cfg["embed_dim"],
            hidden_dim  = cfg["hidden_dim"],
            dropout_p   = cfg["dropout_p"],
            num_classes = NUM_CLASSES,
            pad_idx     = pad_idx,
        ).to(device)
 
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
 
        best_val_loss, epochs_no_improve, best_state = float("inf"), 0, None
 
        for epoch in range(1, 6):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                loss = criterion(model(X_batch), y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
 
            val_loss, _ = evaluate(model, val_loader, criterion, device)
 
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= 2:
                    break

        if best_state:
            model.load_state_dict(best_state)
 
        preds, lbls = collect_predictions(model, val_loader, device)
        metrics = compute_metrics(preds, lbls)
        all_results.append({**cfg, "val_f1": metrics["f1"],
                             "val_acc": metrics["accuracy"]})
 
        print(f"  {i:>4}  {cfg['embed_dim']:>5}  {cfg['hidden_dim']:>6}  "
              f"{cfg['dropout_p']:>4.1f}  {cfg['lr']:>6.4f}  "
              f"{metrics['f1']:>8.4f}  {metrics['accuracy']:>8.2%}")
 
        if metrics["f1"] > best_f1:
            best_f1, best_config = metrics["f1"], cfg
 
    print(f"\n  Best config : {best_config}")
    print(f"  Best val F1 : {best_f1:.4f}")
    return best_config, best_f1, all_results

def analyse_errors(model, word_to_idx, reviews, labels, device, n=10):
    model.eval()
    errors = []
 
    with torch.no_grad():
        for review, true_label in zip(reviews, labels):
            encoded = tokenize_and_encode(review, word_to_idx, MAX_LEN)
            tensor  = torch.tensor([encoded], dtype=torch.long).to(device)
            probs   = torch.softmax(model(tensor), dim=1)[0]
            pred    = probs.argmax().item()
            conf    = probs[pred].item()
 
            if pred != true_label:
                errors.append({
                    "review"    : review,
                    "predicted" : "POSITIVE" if pred == 1 else "NEGATIVE",
                    "actual"    : "POSITIVE" if true_label == 1 else "NEGATIVE",
                    "confidence": conf,
                })
 
    errors.sort(key=lambda e: e["confidence"], reverse=True)
 
    print(f"\n  TOP {n} MOST CONFIDENTLY WRONG PREDICTIONS")
    print(f"  Total errors : {len(errors)} out of {len(reviews)}")
 
    for i, err in enumerate(errors[:n], start=1):
        print(f"\n  Error {i}:")
        print(f"  Review    : {err['review'][:120]!r}")
        print(f"  Predicted : {err['predicted']}  ({err['confidence']:.2%} confidence)")
        print(f"  Actual    : {err['actual']}")


def load_best_model(model_path, vocab_size, embed_dim, hidden_dim,num_classes, pad_idx, device, **kwargs):
    model = DropoutSentimentClassifier(
        vocab_size=vocab_size, embed_dim=embed_dim,
        hidden_dim=hidden_dim, num_classes=num_classes, pad_idx=pad_idx,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict(text, model, word_to_idx, device, max_len=MAX_LEN):
    model.eval()
    with torch.no_grad():
        encoded = tokenize_and_encode(text, word_to_idx, max_len)
        tensor  = torch.tensor([encoded], dtype=torch.long).to(device)
        probs   = torch.softmax(model(tensor), dim=1)[0]
        pred    = probs.argmax().item()
 
    label = "POSITIVE" if pred == 1 else "NEGATIVE"
    conf = probs[pred].item()
    prob_dict = {"POSITIVE": probs[1].item(), "NEGATIVE": probs[0].item()}
    return label, conf, prob_dict

if __name__ == "__main__":
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    word_to_idx, idx_to_word = load_vocab()
    vocab_size = len(word_to_idx)
    pad_idx    = word_to_idx[PAD_TOKEN]
 
    USE_MOCK = False   
 
    train_loader, val_loader, test_loader, _ = build_dataloader(mock=USE_MOCK, batch_size=BATCH_SIZE)
    
    print("=" * 60)
    print("STEP 1 — BASELINE EVALUATION (Week 3 model)")
    print("=" * 60)
 
    baseline_path = Path(__file__).parent / "best_model.pth"
    if not baseline_path.exists():
        print("  best_model.pth not found — training fresh baseline...")
        baseline = SentimentClassifier(
            vocab_size=vocab_size, emb_size=EMBED_DIM,
            num_classes=NUM_CLASSES, pad_token=pad_idx,
        ).to(device)
        train(baseline, train_loader, val_loader,
              num_epochs=NUM_EPOCHS, lr=LEARNING_RATE,
              patience=PATIENCE, device=device)
        torch.save(baseline.state_dict(), baseline_path)
    else:
        baseline = SentimentClassifier(
            vocab_size=vocab_size, emb_size=EMBED_DIM,
            num_classes=NUM_CLASSES, pad_token=pad_idx,
        ).to(device)
        baseline.load_state_dict(
            torch.load(baseline_path, map_location=device)
        )

    base_preds, base_labels = collect_predictions(baseline, val_loader, device)
    baseline_metrics = compute_metrics(base_preds, base_labels)
    print_metrics(baseline_metrics, "Baseline — Validation")
    print_confusion_matrix(baseline_metrics)

    print("\n" + "=" * 60)
    print("STEP 2 — ERROR ANALYSIS")
    print("=" * 60)
 
    if not USE_MOCK:
        from dataset_helpers.dataset_explore import get_real_data
        import random

        all_reviews, all_labels = get_real_data("train")

        # Reproduce the exact same shuffle your get_splits() uses
        paired = list(zip(all_reviews, all_labels))
        random.seed(42)                          # must match RANDOM_SEED in data_tokenize.py
        random.shuffle(paired)

        val_size    = int(len(paired) * 0.1)     # must match VAL_RATIO in data_tokenize.py
        val_pairs   = paired[-val_size:]         # last 10% after shuffle = validation set

        val_txt_raw = [r for r, l in val_pairs]
        val_lb_raw  = [l for r, l in val_pairs]

        analyse_errors(baseline, word_to_idx, val_txt_raw, val_lb_raw, device, n=10)
    else:
        print("  Skipping on mock data — run with USE_MOCK=False for real errors")

    print("\n" + "=" * 60)
    print("STEP 3 — HYPERPARAMETER SEARCH")
    print("=" * 60)
 
    best_config, best_f1, all_results = run_hyperparameter_search(
        train_loader, val_loader, vocab_size, pad_idx, device
    )
 
    with open(TUNING_RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Tuning results saved: {TUNING_RESULTS_PATH}")
    

    print("\n" + "=" * 60)
    print("STEP 4 — TRAINING FINAL MODEL WITH BEST CONFIG")
    print("=" * 60)
 
    final_model = DropoutSentimentClassifier(
        vocab_size  = vocab_size,
        embed_dim   = best_config["embed_dim"],
        hidden_dim  = best_config["hidden_dim"],
        dropout_p   = best_config["dropout_p"],
        num_classes = NUM_CLASSES,
        pad_idx     = pad_idx,
    ).to(device)
 
    print(f"\n  Architecture: {final_model}")
    print(f"  Parameters  : {sum(p.numel() for p in final_model.parameters()):,}")
 
    # Patch MODEL_PATH in FeedForwardClassifier so train() saves the improved
    # model's best checkpoint to a separate file — not over the Week 3 baseline.
    import model.FeedForwardClassifier as ffc_module
    IMPROVED_MODEL_PATH = Path(__file__).parent / "best_improved_model.pth"
    _original_model_path  = ffc_module.MODEL_PATH       # save original
    ffc_module.MODEL_PATH = IMPROVED_MODEL_PATH         # redirect saves here

    train(final_model, train_loader, val_loader,
          num_epochs=NUM_EPOCHS, lr=best_config["lr"],
          patience=PATIENCE, device=device)

    ffc_module.MODEL_PATH = _original_model_path        # restore original

    # Load the best checkpoint train() saved (not last-epoch weights)
    final_model.load_state_dict(
        torch.load(IMPROVED_MODEL_PATH, map_location=device)
    )
    print(f"\n  Best improved checkpoint saved to : {IMPROVED_MODEL_PATH}")
    
    print("\n" + "=" * 60)
    print("STEP 5 — BASELINE vs IMPROVED COMPARISON")
    print("=" * 60)
 
    final_model.load_state_dict(
        torch.load(IMPROVED_MODEL_PATH, map_location=device)
    )
    final_preds, final_labels = collect_predictions(final_model, val_loader, device)
    final_metrics = compute_metrics(final_preds, final_labels)
 
    print(f"\n  {'Metric':12}  {'Baseline':>10}  {'Improved':>10}  {'Change':>10}")
    print(f"  {'─'*46}")
    for key in ("accuracy", "precision", "recall", "f1"):
        b = baseline_metrics[key]
        f = final_metrics[key]
        d = f - b
        arrow = "↑" if d > 0 else ("↓" if d < 0 else "→")
        print(f"  {key.capitalize():12}  {b:>10.4f}  {f:>10.4f}  {arrow} {abs(d):.4f}")
 
    print_confusion_matrix(final_metrics)


    print("\n" + "=" * 60)
    print("STEP 6 — SAVING FINAL MODEL FOR PHASE 2")
    print("=" * 60)
 
    torch.save(final_model.state_dict(), FINAL_MODEL_PATH)
 
    config = {
        "architecture": "ImprovedSentimentClassifier",
        "vocab_size"  : vocab_size,
        "embed_dim"   : best_config["embed_dim"],
        "hidden_dim"  : best_config["hidden_dim"],
        "dropout_p"   : best_config["dropout_p"],
        "num_classes" : NUM_CLASSES,
        "pad_idx"     : pad_idx,
        "max_len"     : MAX_LEN,
        "val_accuracy": final_metrics["accuracy"],
        "val_f1"      : final_metrics["f1"],
    }
 
    config_path = Path(__file__).parent / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
 
    print(f"\n  Model saved  : {FINAL_MODEL_PATH}")
    print(f"  Config saved : {config_path}")
    for k, v in config.items():
        print(f"    {k:16}: {v}")

    print("\n" + "=" * 60)
    print("STEP 7 — FINAL TEST SET EVALUATION")
    print("  Run this ONCE. Do not tune after seeing this number.")
    print("=" * 60)

    test_preds, test_lbls = collect_predictions(final_model, test_loader, device)
    test_metrics = compute_metrics(test_preds, test_lbls)
    print_metrics(test_metrics, "TEST SET — final honest evaluation")
    print_confusion_matrix(test_metrics)

    print("\n" + "=" * 60)
    print("STEP 8 — INFERENCE ON CUSTOM REVIEWS")
    print("=" * 60)
 
    test_sentences = [
        "This movie was absolutely brilliant, loved every scene",
        "Terrible film, complete waste of two hours",
        "Not bad at all, surprisingly enjoyable",
        "I expected to hate it but loved every minute",
        "Brilliant acting but the plot was awful",
    ]
 
    print()
    for sentence in test_sentences:
        label, conf, probs = predict(sentence, final_model, word_to_idx, device)
        pos = "█" * int(probs["POSITIVE"] * 20)
        neg = "█" * int(probs["NEGATIVE"] * 20)
        print(f"  Review : {sentence!r}")
        print(f"  Result : {label}  ({conf:.2%} confidence)")
        print(f"  POS {probs['POSITIVE']:>5.2%} |{pos:<20}|")
        print(f"  NEG {probs['NEGATIVE']:>5.2%} |{neg:<20}|")
        print()