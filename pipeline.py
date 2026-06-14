import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataset_helpers.data_tokenize import get_splits
from model.LSTM import LSTMClassifier

# 1. Hardware Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. PyTorch Dataset Management
class ReviewDataset(Dataset):
    def __init__(self, reviews, labels):
        self.reviews = torch.tensor(reviews, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.reviews)
        
    def __getitem__(self, idx):
        return self.reviews[idx], self.labels[idx]

# 3. Data Pipelines (Loading Full Real Splits)
train_x, train_y, val_x, val_y, test_x, test_y = get_splits(mock_data=False)

train_dataset = ReviewDataset(train_x, train_y)
val_dataset = ReviewDataset(val_x, val_y)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 4. Dynamic Shape & Architectural Constraints
word_to_idx = get_splits.__globals__['load_vocab']()[0]
VOCAB_SIZE = len(word_to_idx)  
EMBEDDING_DIM = 128            

# 5. Model Initialization
model = LSTMClassifier(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM).to(device)
print(f"LSTM Model Initialized on {device}.")

# 6. Loss Optimization Functions
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 7. History Storage for Evaluation
train_loss_history = []
val_loss_history = []

# 8. Training and Validation Loop
EPOCHS = 3
print("\nCommencing Full Week 3 Training Loop...")

for epoch in range(EPOCHS):
    # --- Phase 1: Training ---
    model.train()
    running_train_loss = 0.0
    
    for batch_reviews, batch_labels in train_loader:
        batch_reviews = batch_reviews.to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(batch_reviews).squeeze(1)
        loss = criterion(predictions, batch_labels)
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item() * batch_reviews.size(0)
        
    epoch_train_loss = running_train_loss / len(train_dataset)
    train_loss_history.append(epoch_train_loss)
    
    # --- Phase 2: Validation ---
    model.eval()
    running_val_loss = 0.0
    
    with torch.no_grad():
        for batch_reviews, batch_labels in val_loader:
            batch_reviews = batch_reviews.to(device)
            batch_labels = batch_labels.to(device)
            
            predictions = model(batch_reviews).squeeze(1)
            loss = criterion(predictions, batch_labels)
            
            running_val_loss += loss.item() * batch_reviews.size(0)
            
    epoch_val_loss = running_val_loss / len(val_dataset)
    val_loss_history.append(epoch_val_loss)
    
    print(f"Epoch {epoch+1}/{EPOCHS} Finished | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

print("\nWeek 3 execution complete.")
print(f"Train History: {train_loss_history}")
print(f"Val History:   {val_loss_history}")