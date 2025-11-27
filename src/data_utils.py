import re
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

from .config import LABELS, RANDOM_SEED, MAX_SEQ_LEN, TRAIN_BATCH_SIZE_RNN, VAL_BATCH_SIZE_RNN


def load_raw_jigsaw(csv_path: Path) -> pd.DataFrame:
    """
    Load the Jigsaw toxic comment dataset from a csv file.

    Expected columns: id, comment_text, and the six label columns.
    """
    df = pd.read_csv(csv_path)
    expected = {"comment_text", *LABELS}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    # Keep only the needed columns for now
    cols = ["comment_text"] + LABELS
    return df[cols].copy()


def basic_text_clean(text: str) -> str:
    """
    Simple text cleaning.

    Lowercase, remove extra spaces, and strip basic html artifacts.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def train_valid_test_split(
    df: pd.DataFrame,
    valid_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataframe into train, validation, and test sets.

    We stratify on the main 'toxic' label to keep some balance.
    """
    assert "toxic" in df.columns, "toxic column not found"

    train_df, temp_df = train_test_split(
        df,
        test_size=valid_size + test_size,
        random_state=random_state,
        stratify=df["toxic"],
    )
    proportion_valid = valid_size / (valid_size + test_size)
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=1.0 - proportion_valid,
        random_state=random_state,
        stratify=temp_df["toxic"],
    )
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _build_vocab(texts: List[str], min_freq: int = 2) -> dict:
    """
    Build a simple word to index mapping from a list of texts.
    """
    counts = {}
    for t in texts:
        for tok in t.split():
            counts[tok] = counts.get(tok, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, c in counts.items():
        if c >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


def _encode_text(text: str, vocab: dict, max_len: int) -> List[int]:
    """
    Convert text into a list of token ids with padding or truncation.
    """
    tokens = text.split()
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens][:max_len]
    if len(ids) < max_len:
        ids = ids + [vocab["<pad>"]] * (max_len - len(ids))
    return ids


class JigsawRNNDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, vocab: dict, max_len: int):
        self.texts = texts
        self.labels = labels.astype("float32")
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x_ids = _encode_text(self.texts[idx], self.vocab, self.max_len)
        y = self.labels[idx]
        return torch.tensor(x_ids, dtype=torch.long), torch.tensor(y, dtype=torch.float32)


def build_dataloaders_rnn(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    max_len: int = 100,
) -> Tuple[DataLoader, DataLoader, dict]:
    """
    Build PyTorch DataLoaders for the BiLSTM model.

    Args:
        train_df: Training dataframe
        valid_df: Validation dataframe
        max_len: Maximum sequence length (default: 100)

    Returns train_loader, valid_loader, and the vocab dictionary.
    """
    # Clean text
    train_texts = [basic_text_clean(t) for t in train_df["comment_text"].tolist()]
    valid_texts = [basic_text_clean(t) for t in valid_df["comment_text"].tolist()]

    vocab = _build_vocab(train_texts)

    y_train = train_df[LABELS].values
    y_valid = valid_df[LABELS].values

    train_ds = JigsawRNNDataset(train_texts, y_train, vocab, max_len)
    valid_ds = JigsawRNNDataset(valid_texts, y_valid, vocab, max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE_RNN,
        shuffle=True,
        num_workers=2,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=VAL_BATCH_SIZE_RNN,
        shuffle=False,
        num_workers=2,
    )
    return train_loader, valid_loader, vocab


def get_label_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute positive counts and ratios for each label.
    """
    stats = []
    n = len(df)
    for label in LABELS:
        cnt = df[label].sum()
        stats.append({"label": label, "count": int(cnt), "ratio": float(cnt / n)})
    return pd.DataFrame(stats)

