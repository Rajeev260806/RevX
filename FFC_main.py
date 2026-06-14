import torch
import torch.nn as nn
from pathlib import Path
import sys
import json
 
sys.path.insert(0, str(Path(__file__).parent))
 
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
    accuracy = tn+tp/total if total>0 else 0.0
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
    
     
