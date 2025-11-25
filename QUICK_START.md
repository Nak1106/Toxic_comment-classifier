# Quick Start Guide

## 🚀 Get Running in 5 Minutes

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Place your data
# Copy train.csv from Kaggle to: data/jigsaw_train.csv
```

### 2. Run EDA Notebook

```bash
jupyter notebook notebooks/00_eda_preprocessing.ipynb
```

This will show you:
- Label distributions (how imbalanced the dataset is)
- Text length statistics
- Label co-occurrence patterns
- Sample toxic comments

### 3. Train Your First Model (BiLSTM)

**Option A: Quick notebook demo (recommended for learning)**
```bash
jupyter notebook notebooks/01_baseline_logreg_bilstm.ipynb
```

**Option B: Full CLI training**
```bash
python -m src.training.train_bilstm
```

This saves:
- Model: `models/bilstm_baseline.pt`
- Metrics: `reports/bilstm_baseline_metrics.json`

### 4. Train Transformer Models

```bash
# BERT
python -m src.training.train_transformer --model_type bert

# DistilBERT (faster)
python -m src.training.train_transformer --model_type distilbert

# Lexicon Hybrid (BERT + lexicon features)
python -m src.training.train_lexicon_hybrid
```

### 5. Advanced: Contrastive Learning

```bash
# Step 1: Train encoder
python -m src.training.train_contrastive_encoder

# Step 2: Train classifier from encoder
python -m src.training.train_classifier_from_encoder
```

---

## 📚 Using Utilities in Your Own Code

### Train a Model Programmatically

```python
from src.training.bilstm_utils import train_bilstm_model
from src.training.transformer_utils import train_transformer_model

# Train BiLSTM
model, vocab, metrics = train_bilstm_model(epochs=5)
print(f"BiLSTM Macro F1: {metrics['macro_f1']:.4f}")

# Train BERT
model, tokenizer, metrics = train_transformer_model(model_type="bert", epochs=3)
print(f"BERT Macro F1: {metrics['macro_f1']:.4f}")
```

### Load Data

```python
from src.config import DATA_DIR, LABELS
from src.data_utils import load_raw_jigsaw, train_valid_test_split

# Load and split
df = load_raw_jigsaw(DATA_DIR / "jigsaw_train.csv")
train_df, valid_df, test_df = train_valid_test_split(df)

print(f"Labels: {LABELS}")
print(f"Train size: {len(train_df)}")
```

### Compute Metrics

```python
from src.metrics import compute_classification_metrics
import numpy as np

# y_true: shape (N, 6) - true binary labels
# y_prob: shape (N, 6) - predicted probabilities

metrics = compute_classification_metrics(y_true, y_prob, label_names=LABELS)

print(f"Macro F1: {metrics['macro_f1']:.4f}")
print(f"Micro F1: {metrics['micro_f1']:.4f}")

# Per-label metrics
for label in LABELS:
    print(f"{label}: F1={metrics['per_label'][label]['f1']:.4f}")
```

### Calibration

```python
from src.calibration import expected_calibration_error, temperature_scale_predictions

# Compute ECE
ece = expected_calibration_error(y_true, y_prob)
print(f"ECE: {ece:.4f}")

# Apply temperature scaling
calibrated_probs = temperature_scale_predictions(
    logits_train=train_logits,  # for fitting temperature
    labels_train=train_labels,
    logits_test=test_logits,    # to calibrate
)
```

### Threshold Optimization

```python
from src.thresholds import find_optimal_threshold_per_label, apply_thresholds

# Find optimal thresholds
optimal_thresholds = find_optimal_threshold_per_label(y_true, y_prob, metric="f1")
print(f"Optimal thresholds: {optimal_thresholds}")

# Apply thresholds
y_pred = apply_thresholds(y_prob, optimal_thresholds)
```

### Fairness Evaluation

```python
from src.fairness_eval import generate_identity_templates, evaluate_identity_bias

# Generate test templates
templates = generate_identity_templates()
print(f"Generated {len(templates)} identity templates")

# Evaluate bias
def predict_fn(text):
    # Your model prediction function
    # Should return probabilities of shape (1, 6)
    pass

results = evaluate_identity_bias(predict_fn, templates, LABELS)
```

### Ensemble Models

```python
from src.ensemble_eval import ensemble_average, evaluate_ensemble

# Combine predictions from multiple models
predictions = [bert_probs, distilbert_probs, bilstm_probs]
model_names = ["BERT", "DistilBERT", "BiLSTM"]

# Simple average
ensemble_probs = ensemble_average(predictions)

# Full evaluation
results = evaluate_ensemble(y_true, predictions, model_names, LABELS)
print(f"Ensemble Macro F1: {results['ensemble_average']['macro_f1']:.4f}")
```

---

## 🔍 Inspecting Results

### Load Saved Metrics

```python
import json
from pathlib import Path

# Load metrics from a training run
with open(Path("reports/bert_toxic_metrics.json")) as f:
    metrics = json.load(f)
    
print(f"BERT Macro F1: {metrics['macro_f1']:.4f}")

# Per-label results
for label, stats in metrics["per_label"].items():
    print(f"{label:15s} F1: {stats['f1']:.4f}  Precision: {stats['precision']:.4f}")
```

### Load Saved Model

```python
import torch
from src.models.rnn_models import BiLSTMClassifier
from src.config import LABELS

# Load BiLSTM
checkpoint = torch.load("models/bilstm_baseline.pt")
vocab = checkpoint["vocab"]

model = BiLSTMClassifier(
    vocab_size=len(vocab),
    embed_dim=128,
    hidden_dim=128,
    num_labels=len(LABELS),
    pad_idx=vocab["<pad>"],
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

---

## 💡 Pro Tips

1. **Start Small**: Train for 1-2 epochs in notebooks to test, then run full training via CLI
2. **Save Everything**: Models and metrics are automatically saved to `models/` and `reports/`
3. **GPU vs CPU**: The code automatically uses GPU if available
4. **Quick Iteration**: Load saved metrics in notebooks instead of retraining
5. **Modular**: Mix and match utilities - everything is designed to work together

---

## 🐛 Common Issues

**Issue**: `FileNotFoundError: data/jigsaw_train.csv`
**Solution**: Make sure you've placed the Kaggle dataset in the correct location

**Issue**: `CUDA out of memory`
**Solution**: Reduce batch size in `src/config.py` or use CPU

**Issue**: `ModuleNotFoundError: No module named 'src'`
**Solution**: Run notebooks from project root or add `sys.path.append('..')`

**Issue**: Low performance on rare labels
**Solution**: This is expected! Try lexicon hybrid model or threshold optimization

---

## 📊 Expected Performance (Approximate)

| Model | Macro F1 | Training Time |
|-------|----------|---------------|
| Logistic Regression | 0.45-0.55 | 2-5 min |
| BiLSTM | 0.50-0.60 | 10-20 min |
| DistilBERT | 0.65-0.75 | 20-40 min |
| BERT | 0.70-0.80 | 40-60 min |
| Lexicon Hybrid | 0.72-0.82 | 45-70 min |
| Contrastive | 0.68-0.78 | 50-80 min |

*Times are for 3 epochs on CPU. GPU is 5-10x faster.*

---

## 🎯 Next Steps After Quick Start

1. Complete notebook 02 (transformers comparison)
2. Try calibration techniques
3. Evaluate fairness/bias
4. Build ensembles
5. Create Streamlit demo

---

Happy Coding! 🚀

