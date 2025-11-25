"""
Reusable training and evaluation utilities for contrastive learning models.
"""
import json

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
from ..contrastive_dataset import PairDataset
from ..models.contrastive_model import ContrastiveBertEncoder, ContrastiveLoss


class JigsawEncoderDataset(Dataset):
    """Dataset for contrastive classifier."""
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
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class ContrastiveClassifier(torch.nn.Module):
    """Classification head on top of a pretrained contrastive encoder."""

    def __init__(self, encoder: ContrastiveBertEncoder, num_labels: int):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.hidden_size
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        emb = self.encoder.encode(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(emb)
        return logits


def train_contrastive_encoder(
    csv_path=None,
    num_pairs=80000,
    epochs=None,
    lr=None,
    batch_size=None,
):
    """
    Train a contrastive encoder using pair-wise similarities.
    
    Returns:
        encoder: trained encoder model
        tokenizer: tokenizer
    """
    if csv_path is None:
        csv_path = DATA_DIR / "jigsaw_train.csv"
    if epochs is None:
        epochs = EPOCHS_BERT
    if lr is None:
        lr = LR_BERT
    if batch_size is None:
        batch_size = TRAIN_BATCH_SIZE_BERT
        
    df = load_raw_jigsaw(csv_path)
    train_df, _, _ = train_valid_test_split(df)

    texts = train_df["comment_text"].tolist()
    labels = train_df[LABELS].values
    toxic_any = (labels.sum(axis=1) > 0).astype("int32")

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    pair_ds = PairDataset(texts, toxic_any, num_pairs=num_pairs)
    pair_loader = DataLoader(pair_ds, batch_size=batch_size, shuffle=True, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = ContrastiveBertEncoder(model_name=model_name).to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr)

    total_steps = len(pair_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_DIR / "contrastive_encoder.pt"

    for epoch in range(1, epochs + 1):
        encoder.train()
        total_loss = 0.0
        for batch in pair_loader:
            text_a, text_b, label = batch
            enc_a = tokenizer(
                list(text_a),
                truncation=True,
                padding="max_length",
                max_length=MAX_SEQ_LEN,
                return_tensors="pt",
            )
            enc_b = tokenizer(
                list(text_b),
                truncation=True,
                padding="max_length",
                max_length=MAX_SEQ_LEN,
                return_tensors="pt",
            )

            input_ids_a = enc_a["input_ids"].to(device)
            attention_mask_a = enc_a["attention_mask"].to(device)
            input_ids_b = enc_b["input_ids"].to(device)
            attention_mask_b = enc_b["attention_mask"].to(device)
            labels_tensor = torch.tensor(label, dtype=torch.float32, device=device)

            optimizer.zero_grad()
            emb_a, emb_b = encoder(
                input_ids_a=input_ids_a,
                attention_mask_a=attention_mask_a,
                input_ids_b=input_ids_b,
                attention_mask_b=attention_mask_b,
            )
            loss = criterion(emb_a, emb_b, labels_tensor)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * input_ids_a.size(0)

        avg_loss = total_loss / len(pair_loader.dataset)
        print(f"contrastive epoch {epoch} loss={avg_loss:.4f}")

    torch.save(encoder.state_dict(), save_path)
    print(f"Saved contrastive encoder to {save_path}")
    
    return encoder, tokenizer


def train_contrastive_classifier(
    encoder_path=None,
    csv_path=None,
    epochs=None,
    lr=None,
    batch_size_train=None,
    batch_size_val=None,
):
    """
    Train a classifier on top of a pretrained contrastive encoder.
    
    Returns:
        model: trained classifier
        tokenizer: tokenizer
        metrics: evaluation metrics
    """
    if encoder_path is None:
        encoder_path = MODELS_DIR / "contrastive_encoder.pt"
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

    texts_train = train_df["comment_text"].tolist()
    texts_valid = valid_df["comment_text"].tolist()

    y_train = train_df[LABELS].values
    y_valid = valid_df[LABELS].values

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = JigsawEncoderDataset(texts_train, y_train, tokenizer, MAX_SEQ_LEN)
    valid_ds = JigsawEncoderDataset(texts_valid, y_valid, tokenizer, MAX_SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size_val, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = ContrastiveBertEncoder(model_name=model_name)
    state = torch.load(encoder_path, map_location="cpu")
    encoder.load_state_dict(state)
    encoder.to(device)

    num_labels = len(LABELS)
    model = ContrastiveClassifier(encoder, num_labels=num_labels).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    def train_epoch(model, data_loader, optimizer, scheduler, device):
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

    def eval_epoch(model, data_loader, device):
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

    best_macro_f1 = 0.0
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_name = "contrastive_bert_classifier"
    best_path = MODELS_DIR / f"{save_name}.pt"

    best_metrics = None
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        y_true, y_prob = eval_epoch(model, valid_loader, device)
        metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5, label_names=LABELS)
        macro_f1 = metrics["macro_f1"]
        print(f"contrastive classifier epoch {epoch} train_loss={train_loss:.4f} macro_f1={macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_metrics = metrics
            torch.save(model.state_dict(), best_path)
            with open(REPORTS_DIR / f"{save_name}_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

    print(f"Saved best contrastive classifier to {best_path}")
    
    return model, tokenizer, best_metrics

