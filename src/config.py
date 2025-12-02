from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"

# Labels for the Jigsaw toxic comment dataset
LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

# Basic hyper parameters
RANDOM_SEED = 42
MAX_SEQ_LEN = 128

TRAIN_BATCH_SIZE_RNN = 64
VAL_BATCH_SIZE_RNN = 128

TRAIN_BATCH_SIZE_BERT = 16
VAL_BATCH_SIZE_BERT = 32

LR_RNN = 1e-3
LR_BERT = 2e-5

EPOCHS_RNN = 5
EPOCHS_BERT = 3

