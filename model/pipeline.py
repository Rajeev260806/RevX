import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataset_helpers.data_tokenize import get_splits
from model.LSTM import LSTMClassifier

MODEL_DIR = Path(__file__).parent

# 1. Hardware Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. PyTorch Dataset Management
class ReviewDataset(Dataset):
    def __init__(self, reviews, labels):
        self.reviews = torch.tensor(reviews, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.reviews)
        
    def __getitem__(self, idx):
        return self.reviews[idx], self.labels[idx]

# 3. Data Pipelines
train_x, train_y, val_x, val_y, test_x, test_y = get_splits(mock_data=False)

train_dataset = ReviewDataset(train_x, train_y)
val_dataset   = ReviewDataset(val_x, val_y)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

from dataset_helpers.dataset_explore import load_vocab
word_to_idx, _ = load_vocab()
VOCAB_SIZE = len(word_to_idx)

# =============================================================
#         WEEK 4: HYPERPARAMETER TUNING ENGINE
# =============================================================

def train_and_evaluate(lr, hidden_dim, dropout_rate, epochs=5):
    """
    Trains the LSTM and returns the best checkpoint's val_loss and model.
    Saves best weights during training, not last-epoch weights.
    """
    local_model = LSTMClassifier(
        vocab_size=VOCAB_SIZE,
        embedding_dim=128,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate
    ).to(device)

    local_criterion = nn.CrossEntropyLoss()
    local_optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)

    best_val_loss  = float("inf")
    best_state     = None

    for epoch in range(epochs):
        # Training phase
        local_model.train()
        for batch_reviews, batch_labels in train_loader:
            batch_reviews = batch_reviews.to(device)
            batch_labels  = batch_labels.to(device).long()
            local_optimizer.zero_grad()
            loss = local_criterion(local_model(batch_reviews), batch_labels)
            loss.backward()
            local_optimizer.step()

        # Validation phase
        local_model.eval()
        running_val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for batch_reviews, batch_labels in val_loader:
                batch_reviews = batch_reviews.to(device)
                batch_labels  = batch_labels.to(device).long()
                preds         = local_model(batch_reviews)
                running_val_loss += local_criterion(preds, batch_labels).item() * batch_reviews.size(0)
                correct += (preds.argmax(dim=1) == batch_labels).sum().item()

        epoch_val_loss = running_val_loss / len(val_dataset)
        val_acc        = correct / len(val_dataset)

        # Save best checkpoint — not last-epoch weights
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state    = {k: v.clone() for k, v in local_model.state_dict().items()}

        print(f"  Epoch {epoch+1}/{epochs} — val_loss={epoch_val_loss:.4f}  val_acc={val_acc:.2%}")

    # Load best checkpoint before returning
    local_model.load_state_dict(best_state)
    return best_val_loss, local_model

# Define the tuning search space matrix
learning_rates = [0.001, 0.0005]
hidden_sizes = [64, 128]
dropout_rates = [0.3, 0.5]

best_val_loss = float('inf')
best_hyperparameters = {}
best_model = None

print("\n--- Starting Grid Search Hyperparameter Tuning ---")

for lr in learning_rates:
    for hidden_dim in hidden_sizes:
        for dropout in dropout_rates:
            print(f"Testing Config: LR={lr} | Hidden Size={hidden_dim} | Dropout={dropout}")
            
            val_loss, trained_model = train_and_evaluate(lr, hidden_dim, dropout, epochs=5)
            print(f"--> Resulting Validation Loss: {val_loss:.4f}\n")
            
            # Save the configurations if they outperform previous trials
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = trained_model
                best_hyperparameters = {
                    'learning_rate': lr,
                    'hidden_dim': hidden_dim,
                    'dropout_rate': dropout
                }
import json
lstm_config = {
    "vocab_size"   : VOCAB_SIZE,
    "embedding_dim": 128,
    "hidden_dim"   : best_hyperparameters["hidden_dim"],
    "output_dim"   : 2,
    "dropout_rate" : best_hyperparameters["dropout_rate"],
}
with open(MODEL_DIR / "lstm_config.json", "w") as f:
    json.dump(lstm_config, f, indent=2)

print("TUNING COMPLETE: BEST EXPERIMENTAL CONFIGURATION")
print(f"Optimal Parameters: {best_hyperparameters}")
print(f"Top Validation Loss Achieved: {best_val_loss:.4f}\n")

# Save the absolute optimal weights to disk as requested by the Wk 4 rubric
torch.save(best_model.state_dict(), MODEL_DIR / "best_lstm_model.pth")
print("Saved top-tier model weights to 'best_lstm_model.pth'")

# INFERENCE PIPELINE FOR CUSTOM REVIEWS

def predict_sentiment(text, model, word_idx_map, max_len=512):
    """
    Uses the same tokenize_and_encode pipeline as training.
    Model now outputs 2 logits (NEG, POS) — use softmax + argmax,
    not sigmoid which only works for single-scalar binary output.
    """
    from dataset_helpers.data_tokenize import tokenize_and_encode
    model.eval()

    tokens = tokenize_and_encode(text, word_idx_map, max_len)
    input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)

    with torch.no_grad():
        logits      = model(input_tensor)              # shape: (1, 2)
        probs       = torch.softmax(logits, dim=1)[0]  # shape: (2,)
        pred        = probs.argmax().item()            # 0=NEG, 1=POS
        confidence  = probs[pred].item() * 100

    label = "Positive" if pred == 1 else "Negative"
    return label, confidence

custom_reviews = [
    "This film was an absolute masterpiece with incredible writing.",
    "A total waste of money. The acting was completely unwatchable.",
    "I expected this movie to be garbage, but it was surprisingly great.",
    "The cinematography was beautiful, but the story line was very weak.",
    "Honestly, the worst cinematic experience of my entire life."
]

print("\n--- Running Final Inference Using Best Tuned Model ---")
for i, review_text in enumerate(custom_reviews):
    label, conf = predict_sentiment(review_text, best_model, word_to_idx)
    print(f"Review {i+1}: '{review_text}' -> Prediction: {label} ({conf:.2f}%)")