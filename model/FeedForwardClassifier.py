import torch
import torch.nn as nn
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset_helpers.dataset_explore import load_vocab,UNK_TOKEN,PAD_TOKEN
from dataset_helpers.data_tokenize import MAX_LEN,get_splits

EMBED_DIM = 128
NUM_CLASSES = 2  # for neg and pos
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
BATCH_SIZE = 32
PATIENCE = 3
MODEL_PATH = Path(__file__).parent / "best_model.pth"

class SentimentClassifier(nn.Module):
    
    def __init__(self,vocab_size,emb_size,num_classes,pad_token=0):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=emb_size,padding_idx=pad_token) #vocab_size x emb_size
        self.classifier = nn.Linear(emb_size, num_classes)

    def forward(self,x):
        embedded = self.embedding(x)
        pooled = embedded.mean(dim=1)
        logits = self.classifier(pooled)
        return logits
    
def build_dataloader(mock=False,batch_size=BATCH_SIZE):
    from torch.utils.data import TensorDataset, DataLoader
    tr_txt,tr_lb,val_txt,val_lb,te_txt,te_lb = get_splits(mock_data=mock)

    def to_tensors(encodings, labels):
        X = torch.tensor(encodings,dtype=torch.long)   
        Y = torch.tensor(labels,dtype=torch.long)
        return X,Y
    
    tr_X,tr_Y = to_tensors(tr_txt,tr_lb)       #python to tensor
    val_X,val_Y = to_tensors(val_txt,val_lb)
    te_X,te_Y = to_tensors(te_txt,te_lb)

    train_ds = TensorDataset(tr_X, tr_Y)   #pairs text and label
    val_ds   = TensorDataset(val_X, val_Y)
    test_ds  = TensorDataset(te_X, te_Y)

    train_loader = DataLoader(train_ds,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_ds,batch_size=batch_size,shuffle=False)
    test_loader = DataLoader(test_ds,batch_size=batch_size,shuffle=False)

    word_to_idx, _ = load_vocab()
    vocab_size = len(word_to_idx)

    return train_loader,val_loader,test_loader,vocab_size

def train_one_epoch(model,train_loader,criterion,optimizer,device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch_idx, (X_batch, Y_batch) in enumerate(train_loader):
        X_batch = X_batch.to(device)   
        Y_batch = Y_batch.to(device)
        #5 steps begin
        logits = model(X_batch)          #1-calls for forward pass from the class
        loss = criterion(logits,Y_batch) #2-measure how wrong the predictions are
        optimizer.zero_grad()            #3-clears gradient of previous calculations
        loss.backward()                  #4-calculates loss gradient for this iteration
        optimizer.step()                 #5-update the weights

        total_loss += loss.item() * X_batch.size(0)
        predictions = logits.argmax(dim=1)           # (batch_size,)
        total_correct += (predictions == Y_batch).sum().item()
        total_samples += X_batch.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def evaluate(model,data_loader,criterion,device):
    model.eval()    
 
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0
 
    with torch.no_grad():   
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
 
            logits      = model(X_batch)
            loss        = criterion(logits, y_batch)
            predictions = logits.argmax(dim=1)
 
            total_loss    += loss.item() * X_batch.size(0)
            total_correct += (predictions == y_batch).sum().item()
            total_samples += X_batch.size(0)
 
    avg_loss = total_loss    / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def train(model, train_loader, val_loader, num_epochs=NUM_EPOCHS,lr=LEARNING_RATE, patience=PATIENCE, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }

    best_val_loss = float("inf")
    epochs_no_improve  = 0

    print(f"\n  Device      : {device}")
    print(f"  Parameters  : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Epochs max  : {num_epochs}  |  Patience: {patience}")
    print(f"  Batch size  : {train_loader.batch_size}")
    print(f"  Learning rate: {lr}")
    print()
    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}  {'Note':>12}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*9}  {'-'*8}  {'-'*7}  {'-'*12}")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        note = ""
        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
            note = "saved"
        else:
            epochs_no_improve += 1
            note = f"no improve {epochs_no_improve}/{patience}"
 
        print(f"  {epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.2%}  "
              f"{val_loss:>8.4f}  {val_acc:>7.2%}  {note:>12}")
 
        if epochs_no_improve >= patience:
            print(f"\n  Early stopping triggered at epoch {epoch}.")
            print(f"  Val loss did not improve for {patience} consecutive epochs.")
            break
    print(f"\n  Best val loss : {best_val_loss:.4f}")
    print(f"  Best model : {MODEL_PATH}")
    return history

def print_loss_curves(history):
    print("\n  LOSS CURVE (train=·  val=○)")
    print(f"  {'Epoch':>5}  {'Train':>8}  {'Val':>8}  {'Overfit gap':>12}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*12}")
 
    for i, (tl, vl) in enumerate(
        zip(history["train_loss"], history["val_loss"]), start=1
    ):
        gap  = vl - tl
        flag = "  ⚠️  overfitting" if gap > 0.1 else ""
        print(f"  {i:>5}  {tl:>8.4f}  {vl:>8.4f}  {gap:>+12.4f}{flag}")
 
    print("\n  ACCURACY CURVE")
    print(f"  {'Epoch':>5}  {'Train Acc':>9}  {'Val Acc':>8}")
    print(f"  {'-'*5}  {'-'*9}  {'-'*8}")
    for i, (ta, va) in enumerate(
        zip(history["train_acc"], history["val_acc"]), start=1
    ):
        print(f"  {i:>5}  {ta:>8.2%}  {va:>8.2%}")


if __name__ == "__main__":
 
    # ── Device setup ─────────────────────────────────────────
    # Use GPU if available, otherwise CPU.
    # All tensors and the model must be on the same device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    # ── Test 1: Model architecture ───────────────────────────
    print("=" * 60)
    print("TEST 1 — MODEL ARCHITECTURE")
    print("=" * 60)
 
    word_to_idx, _ = load_vocab()
    vocab_size     = len(word_to_idx)
    pad_token        = word_to_idx[PAD_TOKEN]
 
    model = SentimentClassifier(
        vocab_size  = vocab_size,
        emb_size   = EMBED_DIM,
        num_classes = NUM_CLASSES,
        pad_token     = pad_token,
    ).to(device)
 
    print(f"\n  Vocabulary size : {vocab_size:,}")
    print(f"  Embed dim       : {EMBED_DIM}")
    print(f"  Num classes     : {NUM_CLASSES}")
    print(f"  Pad index       : {pad_token}")
    print(f"\n  Model architecture:")
    print(f"  {model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total trainable parameters: {total_params:,}")
 
    # ── Test 2: Forward pass with dummy data ─────────────────
    print("\n" + "=" * 60)
    print("TEST 2 — FORWARD PASS (shapes)")
    print("=" * 60)
 
    batch_size  = 32
    dummy_input = torch.randint(0, vocab_size, (batch_size, MAX_LEN)).to(device)
    dummy_input[:, 100:] = 0   # simulate PAD tokens in second half
 
    with torch.no_grad():
        logits = model(dummy_input)
 
    print(f"\n  Input shape  : {dummy_input.shape}   ← (batch, MAX_LEN)")
 
    # Inspect intermediate shapes manually
    embedded = model.embedding(dummy_input)
    pooled   = embedded.mean(dim=1)
    print(f"  After embedding : {embedded.shape}   ← (batch, MAX_LEN, emb_size)")
    print(f"  After pooling   : {pooled.shape}   ← (batch, emb_size)")
    print(f"  Output logits   : {logits.shape}    ← (batch, num_classes)")
 
    # Verify PAD embedding is all zeros
    pad_embed = model.embedding(torch.tensor([0]).to(device))
    all_zero  = pad_embed.abs().sum().item() == 0.0
    print(f"\n  PAD embedding is all zeros : {'✅' if all_zero else '❌'}")
 
    # Softmax to show as probabilities
    probs = torch.softmax(logits, dim=1)
    print(f"\n  Sample logits : {logits[0].tolist()}")
    print(f"  After softmax : {probs[0].tolist()}   ← sum = {probs[0].sum().item():.4f}")
 
    # ── Test 3: Single training step ─────────────────────────
    print("\n" + "=" * 60)
    print("TEST 3 — SINGLE TRAINING STEP (5-step loop)")
    print("=" * 60)
 
    criterion_test = nn.CrossEntropyLoss()
    optimizer_test = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    dummy_labels = torch.randint(0, 2, (batch_size,), dtype=torch.long).to(device)
 
    model.train()
    logits_before = model(dummy_input).detach().clone()
 
    # Save weights before step
    w_before = model.classifier.weight.data.clone()
 
    # Run one training step
    logits_step = model(dummy_input)
    loss_step   = criterion_test(logits_step, dummy_labels)
    optimizer_test.zero_grad()
    loss_step.backward()
    optimizer_test.step()
 
    # Verify weights changed
    w_after = model.classifier.weight.data.clone()
    weights_changed = not torch.equal(w_before, w_after)
 
    print(f"\n  Loss value          : {loss_step.item():.4f}")
    print(f"  Weights updated     : {'✅ yes' if weights_changed else '❌ no change'}")
    print(f"  Gradient on embed   : {model.embedding.weight.grad is not None}")
 
    # ── Test 4: Full training with mock data ─────────────────
    print("\n" + "=" * 60)
    print("TEST 4 — FULL TRAINING LOOP (mock data)")
    print("=" * 60)
    print("  (Using 8 mock reviews — loss values are not meaningful)")
    print("  (Run with use_mock=False for real training on IMDb)")
 
    # Re-initialise model with fresh weights for clean training test
    model = SentimentClassifier(
        vocab_size  = vocab_size,
        emb_size   = EMBED_DIM,
        num_classes = NUM_CLASSES,
        pad_token     = pad_token,
    ).to(device)
 
    train_loader, val_loader, test_loader, _ = build_dataloader(
        mock=False, batch_size=32
    )
 
    history = train(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        num_epochs   = NUM_EPOCHS,
        lr           = LEARNING_RATE,
        patience     = PATIENCE,
        device       = device,
    )
 
    print_loss_curves(history)
 
    # ── Test 5: Load best model and evaluate on test set ─────
    print("\n" + "=" * 60)
    print("TEST 5 — LOAD BEST MODEL AND EVALUATE")
    print("=" * 60)
 
    best_model = SentimentClassifier(
        vocab_size  = vocab_size,
        emb_size   = EMBED_DIM,
        num_classes = NUM_CLASSES,
        pad_token     = pad_token,
    ).to(device)
 
    best_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    criterion_final = nn.CrossEntropyLoss()
 
    test_loss, test_acc = evaluate(
        best_model, test_loader, criterion_final, device
    )
 
    print(f"\n  Test loss     : {test_loss:.4f}")
    print(f"  Test accuracy : {test_acc:.2%}")
    print(f"\n  ⚠️  This is mock data — expect ~50-100% random accuracy")
    print(f"  ⚠️  Run with use_mock=False for meaningful numbers")
 
    # ── Test 6: Inference on raw text ────────────────────────
    print("\n" + "=" * 60)
    print("TEST 6 — INFERENCE ON RAW TEXT")
    print("=" * 60)
 
    from dataset_helpers.dataset_explore import clean_and_tokenize, encode
    from dataset_helpers.data_tokenize import pad_or_truncate, tokenize_and_encode
 
    def predict(text, model, word_to_idx, device, max_len=MAX_LEN):
        """
        Predicts sentiment for a single raw review string.
        This is what your web app will call in Phase 2.
 
        Args:
            text        : raw review string
            model       : trained SentimentClassifier
            word_to_idx : vocabulary dict
            device      : "cuda" or "cpu"
            max_len     : sequence length (must match training)
 
        Returns:
            label       : "POSITIVE" or "NEGATIVE"
            confidence  : float — probability of predicted class
        """
        model.eval()
        with torch.no_grad():
            encoded = tokenize_and_encode(text, word_to_idx, max_len)
            tensor  = torch.tensor([encoded], dtype=torch.long).to(device)
            logits  = model(tensor)                    # (1, 2)
            probs   = torch.softmax(logits, dim=1)     # (1, 2)
            pred    = probs.argmax(dim=1).item()       # 0 or 1
            conf    = probs[0][pred].item()            # confidence
 
        label = "POSITIVE" if pred == 1 else "NEGATIVE"
        return label, conf
 
    test_sentences = [
        "This movie was absolutely brilliant, loved every scene",
        "Terrible film, complete waste of two hours",
        "It was okay, nothing special but watchable",
        "Not bad but not great as well"
    ]
 
    print()
    for sentence in test_sentences:
        label, conf = predict(sentence, best_model, word_to_idx, device)
        print(f"  Review   : {sentence!r}")
        print(f"  Predicted: {label}  ({conf:.2%} confidence)")
        print()
 

