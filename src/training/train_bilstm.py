"""
Thin wrapper script for training BiLSTM model.

This script simply calls the reusable training utility.
"""
from .bilstm_utils import train_bilstm_model


def main():
    """Train BiLSTM model using default configuration."""
    train_bilstm_model()


if __name__ == "__main__":
    main()
