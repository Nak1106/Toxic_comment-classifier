import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    """
    Simple BiLSTM based classifier for multi label toxicity.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_labels: int,
        pad_idx: int = 0,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids):
        """
        input_ids: tensor of shape (batch, seq_len)
        returns logits of shape (batch, num_labels)
        """
        emb = self.embedding(input_ids)             # (batch, seq_len, embed_dim)
        out, _ = self.lstm(emb)                     # (batch, seq_len, 2*hidden)
        # simple mean pooling over time
        pooled = out.mean(dim=1)                    # (batch, 2*hidden)
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits


class Attention(nn.Module):
    """Attention mechanism for BiLSTM."""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.w = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, h):
        # h: (batch, seq, 2*hidden)
        scores = torch.tanh(self.w(h))           # (batch, seq, hidden)
        attn = torch.softmax(self.v(scores).squeeze(-1), dim=-1)  # (batch, seq)
        context = (h * attn.unsqueeze(-1)).sum(dim=1)             # (batch, 2*hidden)
        return context, attn


class BiLSTMAttentionClassifier(nn.Module):
    """
    BiLSTM with attention mechanism for multi-label toxicity classification.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_labels: int,
        pad_idx: int = 0,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attn = Attention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)
    
    def forward(self, input_ids):
        """
        input_ids: tensor of shape (batch, seq_len)
        returns logits of shape (batch, num_labels) and attention weights
        """
        emb = self.embedding(input_ids)             # (batch, seq_len, embed_dim)
        h, _ = self.lstm(emb)                       # (batch, seq_len, 2*hidden)
        ctx, attn = self.attn(h)                    # ctx: (batch, 2*hidden), attn: (batch, seq)
        ctx = self.dropout(ctx)
        logits = self.fc(ctx)
        return logits, attn

