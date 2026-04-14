# LBSP Transformer v1

Per-residue ligand binding site prediction using a Transformer encoder trained on pre-computed ProtT5 residue embeddings.

---

## Overview

The model takes per-residue ProtT5 embeddings as input and predicts a binary binding/non-binding label for each residue. Three model architectures are supported, all configured via `config/config.yaml`.

| Model type | Description |
|---|---|
| `transformer_mlp` | Transformer encoder → MLP head (default) |
| `bilstm_mlp` | Bidirectional LSTM → MLP head |
| `mlp` | Mean-pooled protein embedding → MLP head |

---

## Files

| File | Purpose |
|---|---|
| `transformer_v1.py` | Self-contained training script for HPC (no local imports needed) |
| `eval_threshold.py` | Re-run threshold sweep on a saved checkpoint (no retraining) |
| `benchmark.py` | Evaluate a saved checkpoint on an external benchmark dataset |
| `config/config.yaml` | All hyperparameters and paths |
| `transformer_v1_requirements.txt` | Python dependencies |
| `src/` | Modular source package (dataset, model, loss, train, evaluate) |
| `inference/` | Inference pipeline for unlabelled proteins |

---

## Setup

```bash
pip install -r transformer_v1_requirements.txt
```

---

## Configuration

Edit `config/config.yaml` before running. Key fields:

```yaml
data:
  train_df:       # path to training DataFrame (.pkl)
  val_df:         # path to validation DataFrame (.pkl)
  test_df:        # path to test DataFrame (.pkl)
  h5_embeddings:  # path to HDF5 file of per-residue ProtT5 embeddings

model:
  model_type: "transformer_mlp"   # "transformer_mlp" | "bilstm_mlp" | "mlp"
  residue_emb_dim: 1024           # ProtT5 embedding dimension
  d_model: 256                    # Transformer internal dimension
  nhead: 8                        # Attention heads
  num_transformer_layers: 2
  transformer_ff_dim: 512
  pos_encoding_type: "sinusoidal" # "sinusoidal" | "learned" | "rope"
  global_pool: "mean"             # "mean" | "attention"

training:
  batch_size: 32
  learning_rate: 0.0001
  num_epochs: 100
  patience: 25
  loss_fn: "focal"                # "focal" | "bce"
  use_amp: false                  # Enable AMP (recommended on A40/RTX 3090+)
  amp_dtype: "auto"               # "auto" | "bf16" | "fp16" | "none"

paths:
  checkpoint:  # path to save best model (.pt)
  results:     # path to save training results (.joblib)
```

---

## Training

### Local
```bash
python transformer_v1.py --config config/config.yaml
```

### HPC (SLURM)
```bash
python transformer_v1.py --config config/config.yaml
```

Config values can be overridden from the command line without editing the file:
```bash
python transformer_v1.py --config config/config.yaml \
    --override training.batch_size=16 training.use_amp=true
```

---

## Threshold Sweep

The optimal classification threshold is found during training by sweeping MCC on the validation set. To re-run the sweep on a saved checkpoint (e.g. with an extended threshold range):

```bash
python eval_threshold.py --config config/config.yaml
```

Custom threshold range:
```bash
python eval_threshold.py --config config/config.yaml \
    --thresholds 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5
```

Override checkpoint path:
```bash
python eval_threshold.py --config config/config.yaml \
    --checkpoint /path/to/best_model.pt
```

---

## Benchmarking

Evaluate a saved checkpoint on any labelled external dataset (e.g. COACH420):

```bash
python benchmark.py \
    --config config/config.yaml \
    --df /path/to/benchmark.json \
    --h5 /path/to/benchmark_embeddings.h5 \
    --threshold 0.30
```

Use `--sweep` to find the optimal threshold on the benchmark set directly:
```bash
python benchmark.py \
    --config config/config.yaml \
    --df /path/to/benchmark.json \
    --h5 /path/to/benchmark_embeddings.h5 \
    --sweep
```

The benchmark DataFrame must have two columns:
- `dataset_key` — protein identifier matching the HDF5 key
- `binding_vector` — list of per-residue binary labels (0/1)

Accepts `.pkl` and `.json` formats. To convert a `.pkl` saved with a newer NumPy version for use on an older environment:

```python
import joblib
df = joblib.load("benchmark.pkl")
df.to_json("benchmark.json", orient="records")
```

---

## Metrics

All evaluation reports the following metrics over valid (non-padded) residues:

| Metric | Description |
|---|---|
| ROC-AUC | Area under the ROC curve |
| AU-PRC | Area under the precision-recall curve (primary metric for imbalanced data) |
| MCC | Matthews Correlation Coefficient (used for early stopping and threshold selection) |
| Precision | Positive predictive value at the selected threshold |
| Recall | Sensitivity at the selected threshold |

---

## Model Architecture (`transformer_mlp`)

```
ProtT5 residue embeddings [B, L, 1024]
        ↓
Linear projection [B, L, 256]
        ↓
Positional encoding (sinusoidal / learned / RoPE)
        ↓
Transformer encoder (pre-LN, 2 layers, 8 heads)
        ↓
Global pooling (mean or attention)
        ↓
MLP head → per-residue logits [B, L]
        ↓
Binary prediction (binding / non-binding)
```

Trainable parameters: ~1.38M (default config)
