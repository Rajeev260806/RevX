import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, output_dim=1, dropout_rate=0.3):
        super(LSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, 
                            hidden_size=hidden_dim, 
                            batch_first=True)
        
        #Overfitting Safety Net (Dropout Layer)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        #The Final Decider (Linear Layer)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text_tokens):
        # Pass tokens through your map to get coordinates
        # Shape changes from: (Batch Size, 512) -> (Batch Size, 512, 128)
        embedded = self.embedding(text_tokens)
        
        # Feed coordinates into the LSTM sequential engine
        # hidden holds the final memory vectors after reading word 512
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Grab only the absolute final hidden memory state vector
        # Shape: (Batch Size, hidden_dim)
        final_memory = hidden[-1]
        
        # Run through dropout to randomly deactivate neurons (Prevents Memorization)
        dropped_memory = self.dropout(final_memory)
        
        # Compress the final memory down to a single prediction value (0 to 1)
        predictions = self.fc(dropped_memory)
        
        return predictions