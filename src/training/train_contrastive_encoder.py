"""
Thin wrapper script for training contrastive encoder.

This script simply calls the reusable training utility.
"""
from .contrastive_utils import train_contrastive_encoder


def main():
    """Train contrastive encoder using default configuration."""
    train_contrastive_encoder()


if __name__ == "__main__":
    main()

