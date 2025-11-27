System RAM
1.1 / 12.7 GB
 
GPU RAM
0.0 / 15.0 GB
 
Disk
38.1 / 112.6 GB
"""
Reusable training and evaluation utilities for transformer models (BERT, DistilBERT).
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from ..config import (
    DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    LABELS,
    MAX_SEQ_LEN,
    TRAIN_BATCH_SIZE_BERT,
    VAL_BATCH_SIZE_BERT,
    EPOCHS_BERT,
    LR_BERT,
)
from ..data_utils import load_raw_jigsaw, train_valid_test_split
from ..metrics import compute_classification_metrics
from ..models.transformer_models import create_bert_base, create_distilbert


class JigsawBertDataset(Dataset):
    """Dataset for BERT/DistilBERT models."""
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels.astype("float32")
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        txt = str(self.texts[idx])
        encoded = self.tokenizer(
            txt,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }
        return item


def train_epoch_transformer(model, data_loader, optimizer, scheduler, device):
    """Train transformer for one epoch."""
    model.train()
    total_loss = 0.0
    criterion = torch.nn.BCEWithLogitsLoss()
    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(data_loader.dataset)


def eval_epoch_transformer(model, data_loader, device):
    """Evaluate transformer on a dataset."""
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    return y_true, y_prob


def train_transformer_model(
    model_type="bert",
    csv_path=None,
    epochs=None,
    lr=None,
    batch_size_train=None,
    batch_size_val=None,
):
    """
    Complete training pipeline for BERT or DistilBERT.
    
    Args:
        model_type: "bert" or "distilbert"
        
    Returns:
        model: trained model
        tokenizer: tokenizer
        metrics: evaluation metrics
    """
    if csv_path is None:
        csv_path = DATA_DIR / "jigsaw_train.csv"
    if epochs is None:
        epochs = EPOCHS_BERT
    if lr is None:
        lr = LR_BERT
    if batch_size_train is None:
        batch_size_train = TRAIN_BATCH_SIZE_BERT
    if batch_size_val is None:
        batch_size_val = VAL_BATCH_SIZE_BERT
        
    df = load_raw_jigsaw(csv_path)
    train_df, valid_df, _ = train_valid_test_split(df)

    model_name = "bert-base-uncased" if model_type == "bert" else "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_texts = train_df["comment_text"].tolist()
    valid_texts = valid_df["comment_text"].tolist()

    y_train = train_df[LABELS].values
    y_valid = valid_df[LABELS].values

    train_ds = JigsawBertDataset(train_texts, y_train, tokenizer, MAX_SEQ_LEN)
    valid_ds = JigsawBertDataset(valid_texts, y_valid, tokenizer, MAX_SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size_val, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_labels = len(LABELS)
    
    if model_type == "bert":
        model = create_bert_base(num_labels)
        save_name = "bert_toxic"
    else:
        model = create_distilbert(num_labels)
        save_name = "distilbert_toxic"

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    best_macro_f1 = 0.0
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    best_path = MODELS_DIR / f"{save_name}.pt"

    best_metrics = None
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch_transformer(model, train_loader, optimizer, scheduler, device)
        y_true, y_prob = eval_epoch_transformer(model, valid_loader, device)
        metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5, label_names=LABELS)
        macro_f1 = metrics["macro_f1"]
        print(f"{model_type} epoch {epoch} train_loss={train_loss:.4f} macro_f1={macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_metrics = metrics
            torch.save(model.state_dict(), best_path)
            with open(REPORTS_DIR / f"{save_name}_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

    print(f"Saved best {model_type} model to {best_path}")
    
    return model, tokenizer, best_metrics

