# ToxiFlow: Toxic Comment Classification

A research-grade implementation of multi-label toxic comment classification using the Jigsaw dataset.

## Project Structure

```
toxiflow/
├── data/                           # Place jigsaw_train.csv here
├── models/                         # Saved model checkpoints
├── reports/                        # Metrics and results
├── notebooks/                      # Research notebooks
│   ├── 00_eda_preprocessing.ipynb           ✅ COMPLETE
│   ├── 01_baseline_logreg_bilstm.ipynb      ✅ COMPLETE
│   ├── 02_transformers_bert_distilbert_lexicon.ipynb  (TODO)
│   ├── 03_contrastive_learning.ipynb        (TODO)
│   ├── 04_calibration_thresholds.ipynb      (TODO)
│   ├── 05_fairness_bias_analysis.ipynb      (TODO)
│   ├── 06_ensembles_ablation.ipynb          (TODO)
│   └── 07_qualitative_cases.ipynb           (TODO)
├── src/
│   ├── config.py                   # Configuration and hyperparameters
│   ├── data_utils.py               # Data loading and preprocessing
│   ├── metrics.py                  # Evaluation metrics
│   ├── lexicon_utils.py            # Lexicon-based features
│   ├── contrastive_dataset.py      # Contrastive learning dataset
│   ├── calibration.py              # ✨ Temperature scaling, ECE
│   ├── thresholds.py               # ✨ Threshold optimization
│   ├── fairness_eval.py            # ✨ Bias evaluation
│   ├── ensemble_eval.py            # ✨ Ensemble methods
│   ├── models/
│   │   ├── rnn_models.py           # BiLSTM
│   │   ├── transformer_models.py   # BERT, DistilBERT, LexiconHybrid
│   │   └── contrastive_model.py    # Contrastive encoder
│   └── training/
│       ├── bilstm_utils.py         # ✨ Reusable BiLSTM training
│       ├── transformer_utils.py    # ✨ Reusable transformer training
│       ├── lexicon_hybrid_utils.py # ✨ Lexicon hybrid training
│       ├── contrastive_utils.py    # ✨ Contrastive learning training
│       ├── train_bilstm.py         # CLI wrapper
│       ├── train_transformer.py    # CLI wrapper
│       ├── train_lexicon_hybrid.py # CLI wrapper
│       ├── train_contrastive_encoder.py # CLI wrapper
│       └── train_classifier_from_encoder.py # CLI wrapper
├── demo/
│   └── streamlit_app.py            # Interactive demo (TODO)
├── requirements.txt
└── README.md
```

## Features

### ✅ Completed

**Core Infrastructure:**
- Modular data loading and preprocessing
- Comprehensive evaluation metrics (F1, PR-AUC, ROC-AUC)
- Label statistics and analysis utilities

**Models:**
- BiLSTM classifier
- BERT and DistilBERT fine-tuning
- Lexicon-hybrid BERT (combines transformer + lexicon features)
- Contrastive learning encoder

**Research-Grade Utilities:**
- **Calibration**: Temperature scaling, Expected Calibration Error (ECE), Brier score
- **Thresholds**: Per-label threshold optimization
- **Fairness**: Identity bias evaluation with template-based testing
- **Ensembles**: Multiple ensemble strategies (average, max, min, voting, weighted)

**Notebooks:**
- `00_eda_preprocessing.ipynb`: Comprehensive EDA with label distributions, co-occurrence analysis
- `01_baseline_logreg_bilstm.ipynb`: Logistic regression and BiLSTM baselines with comparison

### 🚧 TODO

**Remaining Notebooks:**
- Transformer models (BERT, DistilBERT, Lexicon Hybrid)
- Contrastive learning with embedding visualization
- Calibration and threshold tuning analysis
- Fairness and bias evaluation
- Ensemble ablation studies
- Qualitative error analysis

**Demo:**
- Streamlit web interface

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place your `train.csv` from Kaggle Jigsaw dataset in `data/jigsaw_train.csv`.

### 3. Run Notebooks

Start with EDA and baselines:
```bash
jupyter notebook notebooks/00_eda_preprocessing.ipynb
```

### 4. Train Models

**From Command Line:**
```bash
# BiLSTM
python -m src.training.train_bilstm

# BERT
python -m src.training.train_transformer --model_type bert

# DistilBERT
python -m src.training.train_transformer --model_type distilbert

# Lexicon Hybrid
python -m src.training.train_lexicon_hybrid

# Contrastive Learning
python -m src.training.train_contrastive_encoder
python -m src.training.train_classifier_from_encoder
```

**From Notebooks:**
```python
from src.training.bilstm_utils import train_bilstm_model
model, vocab, metrics = train_bilstm_model(epochs=5)
```

## Key Design Principles

1. **Modular**: Core logic in `src/`, experiments in notebooks
2. **Reusable**: Training functions can be called from scripts or notebooks
3. **Research-Grade**: Calibration, fairness, ensembles built-in
4. **Zero Duplication**: Utilities shared across all experiments
5. **Professional**: Clean structure suitable for graduate-level projects

## Labels

The Jigsaw dataset includes 6 binary labels:
- `toxic`: General toxicity
- `severe_toxic`: Severe toxicity
- `obscene`: Obscene language
- `threat`: Threats
- `insult`: Insults
- `identity_hate`: Identity-based hate

## Citation

Based on the Jigsaw Toxic Comment Classification Challenge dataset from Kaggle.

## License

For educational purposes.
