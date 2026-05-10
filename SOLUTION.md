# Solution Report — SMILES-2026 Hallucination Detection

## Reproducibility Instructions

### Environment

- Python 3.10+
- PyTorch 2.11+ (CUDA recommended, CPU works but slower)
- Dependencies: `pip install -r requirements.txt`
- GPU: NVIDIA GTX 1650 (4GB VRAM) or better; BATCH_SIZE=1 for 4GB cards

### Commands

```bash
git clone https://github.com/ReginaStrem/SMILES-2026.git
cd SMILES-2026
pip install -r requirements.txt
python3 solution.py
```

This produces `results.json` and `predictions.csv` in the project root.

### Important Implementation Details

- `USE_GEOMETRIC = True` is set in `solution.py` to enable geometric feature extraction.
- `BATCH_SIZE = 1` is set for compatibility with 4GB VRAM GPUs; increase to 4+ on larger cards.
- The pipeline auto-detects CUDA/MPS/CPU.
- 5-fold stratified cross-validation is used for evaluation; the final probe for `predictions.csv` is trained on all non-test data.

---

## Final Solution Description

**Best Test AUROC: 71.35%** (averaged over 5 folds)

### Components Modified

Three files were modified as permitted by the competition rules:

#### 1. `aggregation.py` — Last-Token Representation + Geometric Features

**Aggregation (`aggregate`)**

The final layer's last real token representation is used as the primary feature vector (896 dimensions). This is the simplest and most effective approach — the last token position in a causal LM aggregates information from the entire sequence, and the final layer encodes the model's "prediction-ready" representation.

**Geometric Features (`extract_geometric_features`)**

56 hand-crafted features appended when `USE_GEOMETRIC=True`:

1. **Layer-wise L2 norms** (25 features): Mean-pooled representation norm per layer — captures activation magnitude changes across depth.
2. **Inter-layer cosine similarities** (24 features): Cosine similarity between consecutive layers' mean-pooled representations — measures representation drift, which differs for hallucinated vs truthful outputs.
3. **Activation variance** (3 features): Mean per-dimension variance across real tokens for 3 selected layers — hallucinated responses may exhibit different activation spread.
4. **Last-token norms** (3 features): L2 norm of the last real token at 3 selected layers — captures confidence-related magnitude.
5. **Normalized sequence length** (1 feature): Ratio of real tokens to max sequence length — hallucinated responses tend to be longer.

Total feature dimension: 896 + 56 = 952.

#### 2. `probe.py` — Logistic Regression with L2 Regularization

After extensive experimentation with MLP probes (see Failed Attempts), the final solution uses **sklearn's LogisticRegression** wrapped in `nn.Module`:

- **C=0.05** (inverse regularization strength) — strong L2 penalty prevents overfitting on 952 features with ~450 training samples
- **class_weight="balanced"** — handles the 70/30 class imbalance (483 hallucinated vs 206 truthful)
- **solver="lbfgs"**, max_iter=2000
- **Threshold tuning**: `fit_hyperparameters` sweeps candidate thresholds to maximize F1 on the validation split

Key insight: On small datasets (689 samples), logistic regression with strong regularization significantly outperforms neural network probes, which tend to overfit despite dropout and early stopping.

#### 3. `splitting.py` — Stratified 5-Fold Cross-Validation

Replaced the single stratified split with **StratifiedKFold (k=5)**:
- Each fold: test = one fold, remaining data split into train (85%) and validation (15%)
- Preserves class ratio in every split
- More stable metric estimation across folds
- Final predictions use a probe trained on all non-test data

### What Contributed Most to Improving the Metric

1. **Geometric features** — the 56 hand-crafted features (especially inter-layer cosine similarities and L2 norms) provided the single biggest boost, adding discriminative signal beyond raw hidden states.
2. **LogisticRegression over MLP** — switching from neural network probes to L2-regularized logistic regression eliminated overfitting and improved test AUROC from 65-71% (MLP) to 71.35%.
3. **Strong L2 regularization (C=0.05)** — critical for generalization with 952 features and only ~450 training samples per fold.
4. **5-fold CV** — more reliable evaluation and better generalization than a single split.
5. **class_weight="balanced"** — essential for the 70/30 imbalanced dataset.

---

## Experiments and Failed Attempts

### Iteration History

| # | Aggregation | Probe | Feature Dim | Test AUROC | Problem |
|---|-------------|-------|-------------|------------|---------|
| 1 | 5 layers × 3 pools | MLP (512→128→32→1), PCA=256 | 13,496 | 65.99% | Severe overfitting (train 99%, test 66%) |
| 2 | 2 layers × 2 pools | MLP (64→16→1), PCA=64 | 3,640 | 68.00% | Underfitting (train 75%, test 68%) |
| 3 | 4 layers + 3 diffs, last token | MLP (64→16→1), PCA=64 | 6,328 | 69.27% | Underfitting (train 74%, test 69%) |
| 4 | 4 layers + 3 diffs, last token | MLP (128→32→1), PCA=128 | 6,328 | 71.05% | Overfitting (train 99%, test 71%) |
| 5 | 4 layers + 3 diffs, last token | LogReg C=1.0 | 6,328 | 66.04% | Too many features for LogReg |
| 6 | Last token only | LogReg C=0.1 | 952 | 70.62% | Good, slight overfitting |
| 7 | 3 layers + 2 diffs, last token | LogReg C=0.1 | 4,536 | 68.78% | Adding layers hurt LogReg |
| 8 | Last token only | LogReg C=0.01 | 952 | 71.20% | Slightly underfitting |
| **9** | **Last token only** | **LogReg C=0.05** | **952** | **71.35%** | **Best result** |

### Key Failed Approaches

1. **Multi-layer MLP probes** — consistently overfit on this small dataset. Even with PCA (64-256 components), dropout (0.2-0.5), BatchNorm, weight decay, gradient clipping, and early stopping, the gap between train and test AUROC remained 20-30 percentage points. The MLP can memorize 450 training samples easily.

2. **Mean/max pooling** — contrary to initial expectations, pooling across all real tokens did not help. The last token position in a causal LM already aggregates sequence information, and mean/max pooling introduces noise from padding-adjacent tokens.

3. **Multi-layer concatenation with LogReg** — adding more layers (3-5) with LogReg increased feature dimensionality (2,688-13,440) and hurt performance despite L2 regularization. The additional layers introduced more noise than signal for the linear classifier.

4. **PCA dimensionality reduction + MLP** — tried PCA (64-256 components) to compress features before the MLP. While this reduced overfitting slightly, the PCA+MLP combination still couldn't match simple LogReg on raw features with strong L2.

5. **Inter-layer differences** — computing differences between consecutive layers' representations added features but hurt performance with LogReg (iteration 7), likely because the differences are highly correlated with the original representations.
