"""
Thin wrapper script for training lexicon hybrid model.

This script simply calls the reusable training utility.
"""
from .lexicon_hybrid_utils import train_lexicon_hybrid


def main():
    """Train lexicon hybrid BERT model using default configuration."""
    train_lexicon_hybrid()


if __name__ == "__main__":
    main()

