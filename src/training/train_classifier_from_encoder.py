"""
Thin wrapper script for training classifier from pretrained contrastive encoder.

This script simply calls the reusable training utility.
"""
from .contrastive_utils import train_contrastive_classifier


def main():
    """Train classifier from pretrained contrastive encoder using default configuration."""
    train_contrastive_classifier()


if __name__ == "__main__":
    main()

