"""
Thin wrapper script for training transformer models (BERT/DistilBERT).

This script simply calls the reusable training utility.
"""
import argparse
from .transformer_utils import train_transformer_model


def main():
    """Train transformer model using command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["bert", "distilbert"], default="bert")
    args = parser.parse_args()
    
    train_transformer_model(model_type=args.model_type)


if __name__ == "__main__":
    main()
