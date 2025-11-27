# 🚀 Google Colab Training Guide

## Quick Start on Colab

### Step 1: Open the Notebook
1. Go to: https://colab.research.google.com/
2. Click **File → Open Notebook → GitHub**
3. Enter: `https://github.com/Nak1106/Toxic_comment-classifier`
4. Select: `notebooks/COLAB_Full_Training.ipynb`

### Step 2: Enable GPU
1. Click **Runtime → Change runtime type**
2. Select **T4 GPU** or **A100 GPU** (if available)
3. Click **Save**

### Step 3: Run All Cells
1. Click **Runtime → Run all** (or press `Ctrl+F9`)
2. When prompted, upload your `train.csv` file
3. Wait ~2-3 hours for complete training

---

## What Gets Trained

The notebook trains **5 models** for your Phase 1-3:

| Model | Training Time | GPU Memory | Expected Macro F1 |
|-------|--------------|------------|-------------------|
| **Logistic Regression** | ~5 min (CPU) | 0 GB | 0.45-0.55 |
| **BiLSTM** | ~10 min | ~2 GB | 0.50-0.60 |
| **BiLSTM + Attention** | ~10 min | ~2 GB | 0.52-0.62 |
| **DistilBERT** | ~40 min | ~6 GB | 0.65-0.75 |
| **BERT-base** | ~60 min | ~8 GB | 0.70-0.80 |

**Total Runtime**: ~2-3 hours on Colab GPU

---

## GPU Optimizations Included

✅ **Mixed Precision Training (AMP)**
- 2x faster training on GPU
- 50% less memory usage
- Automatically enabled for BERT models

✅ **Optimized Batch Sizes**
- BiLSTM: 128 samples/batch
- DistilBERT: 16 samples/batch
- BERT: 8 samples/batch (larger model)

✅ **Memory Management**
- GPU cache clearing between models
- Gradient accumulation if needed
- Efficient dataloaders

✅ **Fast Tokenization**
- Sequence length: 128 tokens (vs 512)
- Pre-computed TF-IDF for LogReg
- Parallel data loading

---

## Expected Output

After training completes, you'll see:

### 1. Console Output
```
============================================================
FINAL RESULTS COMPARISON
============================================================
                  Model  Macro F1  Micro F1
  Logistic Regression    0.5234    0.8456
               BiLSTM    0.5678    0.8712
   BiLSTM + Attention    0.5834    0.8798
          DistilBERT    0.7145    0.9123
           BERT-base    0.7456    0.9234
```

### 2. Visualizations
- Overall F1 comparison bar chart
- Per-label F1 heatmap
- Saved as: `reports/model_comparison.png`

### 3. Saved Files
```
models/
  ├── bilstm_baseline.pt          (~50 MB)
  ├── bilstm_attention.pt         (~50 MB)
  ├── distilbert_toxic.pt         (~260 MB)
  └── bert_toxic.pt               (~440 MB)

reports/
  ├── logreg_baseline_metrics.json
  ├── bilstm_baseline_metrics.json
  ├── bilstm_attention_metrics.json
  ├── distilbert_toxic_metrics.json
  ├── bert_toxic_metrics.json
  └── model_comparison.png
```

### 4. Download Results
The final cell will create `results.zip` containing all models and metrics.
Download it and extract to your local project folder.

---

## Troubleshooting

### ❌ CUDA Out of Memory

**Solution**: Reduce batch sizes in Cell 6:
```python
BATCH_SIZE_BERT = 8   # Reduce from 16
BATCH_SIZE_BERT_LARGE = 4  # Reduce from 8 for BERT-base
```

### ❌ Colab Disconnects (Timeout)

Colab free tier disconnects after ~90 minutes of inactivity.

**Solution 1**: Use Colab Pro ($10/month) for longer sessions

**Solution 2**: Run models separately:
- Comment out models you don't need
- Save/download after each model
- Resume from where you left off

### ❌ Slow Training

Check GPU is enabled:
```python
import torch
print(torch.cuda.is_available())  # Should print: True
print(torch.cuda.get_device_name(0))  # Should show: Tesla T4 or A100
```

### ❌ Dataset Not Found

Make sure you uploaded `train.csv` in Cell 4.
Or use Kaggle API (uncomment Option 2 in Cell 4).

---

## Advanced: Running Individual Models

You can run models separately by commenting out sections:

### Train Only BiLSTM Models
1. Run cells 1-8 (setup and data loading)
2. Run cells 9-14 (LogReg, BiLSTM, BiLSTM+Attention)
3. Skip cells 15-18 (DistilBERT and BERT)
4. Run cells 19-22 (results and download)

### Train Only Transformer Models
1. Run cells 1-8 (setup and data loading)
2. Run cell 9 (LogReg for comparison)
3. Skip cells 11-14 (BiLSTM models)
4. Run cells 15-18 (DistilBERT and BERT)
5. Run cells 19-22 (results and download)

---

## Tips for Best Results

1. **Use Colab Pro** if available
   - More GPU time
   - Faster GPUs (A100)
   - Longer runtime limits

2. **Monitor GPU Usage**
   ```python
   !nvidia-smi
   ```

3. **Save Intermediate Results**
   After each model, manually download metrics:
   ```python
   from google.colab import files
   files.download('reports/bert_toxic_metrics.json')
   ```

4. **Clear Outputs**
   If notebook gets slow, clear outputs:
   **Edit → Clear all outputs**

---

## Performance Benchmarks

On Colab T4 GPU:

| Model | Training Speed | Memory Usage | Time per Epoch |
|-------|----------------|--------------|----------------|
| BiLSTM | ~5000 samples/sec | 2 GB | ~2 min |
| BiLSTM+Attn | ~4800 samples/sec | 2 GB | ~2 min |
| DistilBERT | ~300 samples/sec | 6 GB | ~12 min |
| BERT | ~150 samples/sec | 8 GB | ~20 min |

---

## Next Steps After Training

1. **Download results.zip** from last cell
2. **Extract to your local project**
3. **Commit to GitHub**:
   ```bash
   git add reports/ models/
   git commit -m "Add training results from Colab"
   git push
   ```

4. **Create analysis notebooks** using saved metrics
5. **Compare with teammate's results**

---

## Need Help?

- **Colab Issues**: https://research.google.com/colaboratory/faq.html
- **GitHub Repo**: https://github.com/Nak1106/Toxic_comment-classifier
- **Your PROGRESS.md**: Check project status

---

**Happy Training! 🚀**

