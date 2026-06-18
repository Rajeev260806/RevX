import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, output_dim=2, dropout_rate=0.3):
        super(LSTMClassifier, self).__init__()

        # padding_idx=0 keeps PAD embedding as all-zeros and stops
        # its gradient from updating — PAD tokens carry no meaning
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim  = embedding_dim,
            padding_idx    = 0,
        )
        self.lstm    = nn.LSTM(input_size=embedding_dim,
                               hidden_size=hidden_dim,
                               batch_first=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc      = nn.Linear(hidden_dim, output_dim)

    def forward(self, text_tokens):
        # Compute real (non-PAD) length of each sequence in the batch.
        # text_tokens != 0 is True for every real token, False for PAD.
        # Summing across dim=1 gives the count of real tokens per review.
        # .clamp(min=1) prevents zero-length sequences from crashing pack.
        lengths = (text_tokens != 0).sum(dim=1).clamp(min=1).cpu()

        embedded = self.embedding(text_tokens)   # (batch, seq_len, embed_dim)

        # pack_padded_sequence tells the LSTM to skip PAD positions entirely.
        # enforce_sorted=False means we do not need to pre-sort by length.
        # Without packing, the LSTM reads all 512 positions including PAD —
        # hundreds of zero vectors dilute the hidden state and kill signal.
        packed   = pack_padded_sequence(embedded, lengths,
                                        batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)

        # hidden[-1] is the final hidden state after the last REAL token
        # (not the last PAD token, which is what you get without packing)
        final_memory   = self.dropout(hidden[-1])   # (batch, hidden_dim)
        return self.fc(final_memory)                # (batch, output_dim)