"""
Reusable training and evaluation utilities for BiLSTM models.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import DATA_DIR, MODELS_DIR, REPORTS_DIR, LABELS, EPOCHS_RNN, LR_RNN
from ..data_utils import load_raw_jigsaw, train_valid_test_split, build_dataloaders_rnn
from ..metrics import compute_classification_metrics
from ..models.rnn_models import BiLSTMClassifier


def train_epoch_bilstm(model, loader, criterion, optimizer, device):
    """Train BiLSTM for one epoch."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids, labels = [x.to(device) for x in batch]
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(loader.dataset)


def eval_epoch_bilstm(model, loader, device):
    """Evaluate BiLSTM on a dataset."""
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids, labels = [x.to(device) for x in batch]
            logits = model(input_ids)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    return y_true, y_prob


def train_bilstm_model(
    csv_path=None,
    embed_dim=128,
    hidden_dim=128,
    epochs=None,
    lr=None,
    save_name="bilstm_baseline",
):
    """
    Complete training pipeline for BiLSTM model.
    
    Returns:
        model: trained model
        vocab: vocabulary dict
        metrics: evaluation metrics
    """
    if csv_path is None:
        csv_path = DATA_DIR / "jigsaw_train.csv"
    if epochs is None:
        epochs = EPOCHS_RNN
    if lr is None:
        lr = LR_RNN
        
    df = load_raw_jigsaw(csv_path)
    train_df, valid_df, _ = train_valid_test_split(df)
    train_loader, valid_loader, vocab = build_dataloaders_rnn(train_df, valid_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = len(vocab)
    num_labels = len(LABELS)
    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_labels=num_labels,
        pad_idx=vocab["<pad>"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_macro_f1 = 0.0
    best_path = MODELS_DIR / f"{save_name}.pt"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    best_metrics = None
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch_bilstm(model, train_loader, criterion, optimizer, device)
        y_true, y_prob = eval_epoch_bilstm(model, valid_loader, device)
        metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5, label_names=LABELS)
        macro_f1 = metrics["macro_f1"]
        print(f"Epoch {epoch} train_loss={train_loss:.4f} macro_f1={macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_metrics = metrics
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                },
                best_path,
            )
            with open(REPORTS_DIR / f"{save_name}_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

    print(f"Best macro F1: {best_macro_f1:.4f}")
    print(f"Saved best model to {best_path}")
    
    return model, vocab, best_metrics


def load_bilstm_model(model_path, device=None):
    """Load a trained BiLSTM model from checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    checkpoint = torch.load(model_path, map_location=device)
    vocab = checkpoint["vocab"]
    
    vocab_size = len(vocab)
    num_labels = len(LABELS)
    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=128,
        hidden_dim=128,
        num_labels=num_labels,
        pad_idx=vocab["<pad>"],
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, vocab

