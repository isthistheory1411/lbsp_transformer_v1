"""
train_hpc.py — Self-contained training script for HPC submission.

Combines: dataset, model, loss, training loop, evaluation, and main pipeline.
No local package imports required — just this file, config.yaml, and requirements.txt.

Usage:
    python train_hpc.py --config config/config.yaml
    python train_hpc.py --config config/config.yaml --override training.batch_size=16
"""

import copy
import json
import math
import os
import random
import argparse

import h5py
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Dict, List, Optional, Tuple


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across PyTorch, NumPy, and Python."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_results(results, save_path):
    joblib.dump(results, save_path, compress=3)
    print(f"Training results saved to {save_path}")


# =============================================================================
# Dataset
# =============================================================================

class ProteinDataset(Dataset):
    """
    PyTorch Dataset for protein ligand binding site prediction.

    Supports optional inference mode (no labels).

    Args:
        df (pd.DataFrame): DataFrame with 'dataset_key' and optionally 'binding_vector'.
        h5_path (str): Path to HDF5 file storing protein embeddings.
        max_len (int): Maximum sequence length; shorter sequences are padded.
        inference (bool): If True, dataset will ignore labels for inference.
    """
    def __init__(self, df, h5_path, max_len=1022, inference=False):
        self.df = df.reset_index(drop=True)
        self.h5_path = h5_path
        self.max_len = max_len
        self.inference = inference

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        protein_key = row['dataset_key']
        labels = None if self.inference else np.asarray(row['binding_vector'], dtype=np.float32)

        # Load embeddings from HDF5.
        # File is opened lazily on first access per worker so that each
        # DataLoader worker gets its own handle after forking — required for
        # safe use with num_workers > 0.
        if not hasattr(self, '_h5f'):
            self._h5f = h5py.File(self.h5_path, 'r')
        if protein_key not in self._h5f:
            raise KeyError(f"Protein key '{protein_key}' not found in {self.h5_path}")
        emb = self._h5f[protein_key][:]  # [L_i, D]

        L_i, D = emb.shape

        if L_i < self.max_len:
            pad_len = self.max_len - L_i
            emb = np.vstack([emb, np.zeros((pad_len, D), dtype=np.float32)])
            if not self.inference:
                labels = np.hstack([labels.astype(np.float32),
                                    np.zeros(pad_len, dtype=np.float32)])
        else:
            emb = emb[:self.max_len]
            if not self.inference:
                labels = labels[:self.max_len]

        mask = np.zeros(self.max_len, dtype=np.float32)
        mask[:min(L_i, self.max_len)] = 1.0
        pos = (np.arange(self.max_len) / self.max_len).astype(np.float32)

        sample = {
            "embeddings": torch.tensor(emb,  dtype=torch.float32),
            "mask":       torch.tensor(mask, dtype=torch.float32),
            "position":   torch.tensor(pos,  dtype=torch.float32),
        }
        if not self.inference:
            sample["labels"] = torch.tensor(labels, dtype=torch.float32)
        return sample


def collate_fn(batch):
    """Collate samples from ProteinDataset into batched tensors."""
    emb  = torch.stack([b['embeddings'] for b in batch], dim=0)          # [B,L,D]
    mask = torch.stack([b['mask']       for b in batch], dim=0)          # [B,L]
    pos  = torch.stack([b['position']   for b in batch], dim=0).unsqueeze(-1)  # [B,L,1]

    labels = None
    if 'labels' in batch[0]:
        labels = torch.stack([b['labels'] for b in batch], dim=0)        # [B,L]

    return {"embeddings": emb, "mask": mask, "position": pos, "labels": labels}


def get_protein_dataloader(df, h5_path, batch_size=32, shuffle=True,
                           max_len=1022, num_workers=0, inference=False):
    dataset = ProteinDataset(df, h5_path, max_len=max_len, inference=inference)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_fn, num_workers=num_workers)


# =============================================================================
# Positional encodings
# =============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al. 2017)."""
    def __init__(self, d_model: int, max_len: int = 1022, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embedding (nn.Embedding over position indices)."""
    def __init__(self, d_model: int, max_len: int = 1022, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device)
        return self.dropout(x + self.embedding(positions))


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) — Su et al. 2021 (RoFormer).

    Applied to Q and K inside each attention layer so the dot-product Q·Kᵀ
    depends only on the relative distance (m - n) between positions.

    dim should equal d_model // nhead (per-head dimension).
    """
    def __init__(self, dim: int, max_len: int = 1022):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos', emb.cos())  # [max_len, dim]
        self.register_buffer('sin', emb.sin())  # [max_len, dim]

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor, nhead: int) -> torch.Tensor:
        """x: [B, L, d_model] — applies per-head rotation."""
        B, L, d_model = x.shape
        d_head = d_model // nhead
        cos = self.cos[:L].unsqueeze(0)   # [1, L, d_head]
        sin = self.sin[:L].unsqueeze(0)
        x = x.view(B, L, nhead, d_head)
        x = x * cos.unsqueeze(2) + self._rotate_half(x) * sin.unsqueeze(2)
        return x.view(B, L, d_model)


# =============================================================================
# RoPE-aware Transformer encoder
# =============================================================================

class _RoPEEncoderLayer(nn.Module):
    """Pre-LN Transformer encoder layer with RoPE applied to Q and K."""
    def __init__(self, d_model, nhead, dim_feedforward, dropout, rope):
        super().__init__()
        self.nhead = nhead
        self.rope  = rope
        self.self_attn = nn.MultiheadAttention(d_model, nhead,
                                               dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, src, src_key_padding_mask=None):
        normed = self.norm1(src)
        q = self.rope(normed, self.nhead)
        k = self.rope(normed, self.nhead)
        attn_out, _ = self.self_attn(q, k, normed,
                                     key_padding_mask=src_key_padding_mask)
        src = src + self.drop1(attn_out)
        src = src + self.drop2(self.ff(self.norm2(src)))
        return src


class _RoPEEncoder(nn.Module):
    """Stack of _RoPEEncoderLayer, API-compatible with nn.TransformerEncoder."""
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(num_layers)]
        )

    def forward(self, src, src_key_padding_mask=None):
        x = src
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x


# =============================================================================
# Models
# =============================================================================

class ResidueMLP(nn.Module):
    """
    Per-residue MLP baseline for ligand binding site prediction.

    Combines per-residue embeddings, a projected mean-pooled protein embedding,
    and a normalised position scalar.
    """
    def __init__(self, residue_emb_dim, protein_emb_dim=256,
                 hidden_dims=None, dropout=0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        self.protein_proj = nn.Linear(residue_emb_dim, protein_emb_dim)
        layers = []
        last_dim = residue_emb_dim + protein_emb_dim + 1
        for h in hidden_dims:
            layers.extend([nn.Linear(last_dim, h), nn.LayerNorm(h),
                           nn.ReLU(), nn.Dropout(dropout)])
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, embeddings, mask, position):
        mask_exp = mask.unsqueeze(-1)
        mean_emb = (embeddings * mask_exp).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        protein_feats = self.protein_proj(mean_emb)
        protein_feats_exp = protein_feats.unsqueeze(1).expand(-1, embeddings.size(1), -1)
        x = torch.cat([embeddings, protein_feats_exp, position], dim=-1)
        return self.mlp(x).squeeze(-1)


class ResidueTransformerMLP(nn.Module):
    """
    Transformer encoder for per-residue binding site prediction.

    Architecture:
        1. Project residue embeddings to d_model
        2. Positional encoding: sinusoidal (fixed) | learned (nn.Embedding) |
           RoPE (no additive encoding; rotation applied to Q and K inside attention)
        3. Transformer encoder (Pre-LN) with padding mask
        4. Global context: mean pool or attention pool (learnable query)
        5. Concatenate per-residue features + global context
        6. MLP head -> per-residue logit
    """
    def __init__(self, residue_emb_dim, d_model=256, nhead=8,
                 num_transformer_layers=2, transformer_ff_dim=512,
                 dropout=0.1, max_len=1022,
                 pos_encoding_type="sinusoidal", global_pool="mean"):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(residue_emb_dim, d_model),
            nn.LayerNorm(d_model)
        )

        self.pos_encoding_type = pos_encoding_type
        if pos_encoding_type == "learned":
            self.pos_enc = LearnedPositionalEncoding(d_model, max_len, dropout)
        elif pos_encoding_type == "rope":
            self.pos_enc = None
        else:
            self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        if pos_encoding_type == "rope":
            assert d_model % nhead == 0, "d_model must be divisible by nhead for RoPE"
            rope = RotaryEmbedding(dim=d_model // nhead, max_len=max_len)
            self.transformer = _RoPEEncoder(
                _RoPEEncoderLayer(d_model, nhead, transformer_ff_dim, dropout, rope),
                num_layers=num_transformer_layers
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=transformer_ff_dim,
                dropout=dropout, batch_first=True, norm_first=True
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_transformer_layers
            )

        self.global_pool = global_pool
        if global_pool == "attention":
            self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))
            self.pool_attn  = nn.MultiheadAttention(d_model, nhead,
                                                    dropout=dropout, batch_first=True)

        self.mlp_head = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, embeddings, mask, position):
        x = self.input_proj(embeddings)
        if self.pos_enc is not None:
            x = self.pos_enc(x)

        pad_mask = ~mask.bool()  # True = padded
        x = self.transformer(x, src_key_padding_mask=pad_mask)

        if self.global_pool == "attention":
            B = x.size(0)
            query = self.pool_query.expand(B, -1, -1)
            global_ctx, _ = self.pool_attn(query, x, x,
                                           key_padding_mask=pad_mask)
            global_ctx = global_ctx.squeeze(1)
        else:
            mask_exp = mask.unsqueeze(-1)
            global_ctx = (x * mask_exp).sum(dim=1) / mask.sum(dim=1, keepdim=True)

        global_ctx_exp = global_ctx.unsqueeze(1).expand(-1, x.size(1), -1)
        combined = torch.cat([x, global_ctx_exp], dim=-1)
        return self.mlp_head(combined).squeeze(-1)


class ResidueBiLSTMMLP(nn.Module):
    """
    Bidirectional LSTM encoder for per-residue binding site prediction.

    Captures sequential dependencies at lower memory cost than the Transformer.
    Useful for ablations or compute-constrained settings.
    """
    def __init__(self, residue_emb_dim, d_model=256, lstm_hidden_size=128,
                 lstm_num_layers=2, dropout=0.1,
                 global_pool="mean", pool_nhead=8):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(residue_emb_dim, d_model),
            nn.LayerNorm(d_model)
        )
        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers, bidirectional=True, batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0
        )
        lstm_out_dim = lstm_hidden_size * 2

        self.global_pool = global_pool
        if global_pool == "attention":
            assert lstm_out_dim % pool_nhead == 0, (
                f"lstm_hidden_size*2 ({lstm_out_dim}) must be divisible by "
                f"pool_nhead ({pool_nhead})"
            )
            self.pool_query = nn.Parameter(torch.randn(1, 1, lstm_out_dim))
            self.pool_attn  = nn.MultiheadAttention(lstm_out_dim, pool_nhead,
                                                    dropout=dropout, batch_first=True)

        self.mlp_head = nn.Sequential(
            nn.Linear(lstm_out_dim * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, embeddings, mask, position):
        x = self.input_proj(embeddings)

        lengths = mask.sum(dim=1).long().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=embeddings.size(1)
        )
        x = x * mask.unsqueeze(-1)

        pad_mask = ~mask.bool()
        if self.global_pool == "attention":
            B = x.size(0)
            query = self.pool_query.expand(B, -1, -1)
            global_ctx, _ = self.pool_attn(query, x, x,
                                           key_padding_mask=pad_mask)
            global_ctx = global_ctx.squeeze(1)
        else:
            mask_exp = mask.unsqueeze(-1)
            global_ctx = (x * mask_exp).sum(dim=1) / mask.sum(dim=1, keepdim=True)

        global_ctx_exp = global_ctx.unsqueeze(1).expand(-1, x.size(1), -1)
        combined = torch.cat([x, global_ctx_exp], dim=-1)
        return self.mlp_head(combined).squeeze(-1)


def build_model(cfg) -> nn.Module:
    """
    Instantiate the correct model from OmegaConf config.

    model_type:
        "mlp"             -> ResidueMLP
        "transformer_mlp" -> ResidueTransformerMLP
        "bilstm_mlp"      -> ResidueBiLSTMMLP
    """
    model_type = OmegaConf.select(cfg, "model.model_type", default="mlp")

    if model_type == "mlp":
        return ResidueMLP(
            residue_emb_dim=cfg.model.residue_emb_dim,
            protein_emb_dim=OmegaConf.select(cfg, "model.protein_emb_dim", default=256),
            hidden_dims=list(OmegaConf.select(cfg, "model.hidden_dims",
                                              default=[512, 256, 128])),
            dropout=cfg.model.dropout
        )

    elif model_type == "transformer_mlp":
        pos_encoding_type = OmegaConf.select(cfg, "model.pos_encoding_type", default=None)
        if pos_encoding_type is None:
            use_learned = OmegaConf.select(cfg, "model.use_learned_pos_encoding", default=False)
            pos_encoding_type = "learned" if use_learned else "sinusoidal"

        return ResidueTransformerMLP(
            residue_emb_dim=cfg.model.residue_emb_dim,
            d_model=OmegaConf.select(cfg, "model.d_model", default=256),
            nhead=OmegaConf.select(cfg, "model.nhead", default=8),
            num_transformer_layers=OmegaConf.select(cfg, "model.num_transformer_layers", default=2),
            transformer_ff_dim=OmegaConf.select(cfg, "model.transformer_ff_dim", default=512),
            dropout=cfg.model.dropout,
            max_len=cfg.model.max_len,
            pos_encoding_type=pos_encoding_type,
            global_pool=OmegaConf.select(cfg, "model.global_pool", default="mean")
        )

    elif model_type == "bilstm_mlp":
        return ResidueBiLSTMMLP(
            residue_emb_dim=cfg.model.residue_emb_dim,
            d_model=OmegaConf.select(cfg, "model.d_model", default=256),
            lstm_hidden_size=OmegaConf.select(cfg, "model.lstm_hidden_size", default=128),
            lstm_num_layers=OmegaConf.select(cfg, "model.lstm_num_layers", default=2),
            dropout=cfg.model.dropout,
            global_pool=OmegaConf.select(cfg, "model.global_pool", default="mean"),
            pool_nhead=OmegaConf.select(cfg, "model.lstm_pool_nhead", default=8)
        )

    raise ValueError(
        f"Unknown model_type '{model_type}'. "
        "Choose from: 'mlp', 'transformer_mlp', 'bilstm_mlp'."
    )


# =============================================================================
# Loss functions
# =============================================================================

def masked_bce_loss(logits, labels, mask, pos_weight=None):
    """BCE loss over valid (unpadded) residues only."""
    criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    loss = criterion(logits, labels) * mask
    return loss.sum() / mask.sum()


def masked_focal_loss(logits, labels, mask, alpha=0.25, gamma=2.0):
    """Focal loss (Lin et al. 2017) over valid residues — handles class imbalance."""
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    p_t = torch.exp(-bce)
    focal_weight = alpha * labels + (1.0 - alpha) * (1.0 - labels)
    loss = focal_weight * ((1.0 - p_t) ** gamma) * bce * mask
    return loss.sum() / mask.sum()


def build_loss_fn(cfg):
    """
    Return a loss callable (logits, labels, mask) -> scalar from config.

    loss_fn: "bce" | "focal"
    """
    loss_fn_name = OmegaConf.select(cfg, "training.loss_fn", default="bce")

    if loss_fn_name == "bce":
        pos_weight_val = OmegaConf.select(cfg, "training.pos_weight", default=9.0)
        _pw = torch.tensor([pos_weight_val])

        def _bce(logits, labels, mask, device=None):
            return masked_bce_loss(logits, labels, mask,
                                   pos_weight=_pw.to(logits.device))
        return _bce

    elif loss_fn_name == "focal":
        alpha = float(OmegaConf.select(cfg, "training.focal_alpha", default=0.25))
        gamma = float(OmegaConf.select(cfg, "training.focal_gamma", default=2.0))

        def _focal(logits, labels, mask, device=None):
            return masked_focal_loss(logits, labels, mask, alpha=alpha, gamma=gamma)
        return _focal

    raise ValueError(f"Unknown loss_fn '{loss_fn_name}'. Choose from: 'bce', 'focal'.")


# =============================================================================
# AMP helper
# =============================================================================

def _resolve_amp(amp_dtype: str):
    """
    Resolve amp_dtype string to (use_amp, dtype, use_scaler).

        "auto" -> BF16 if supported (Ampere/Ada), else FP16
        "bf16" -> BF16 (stable range, no GradScaler needed)
        "fp16" -> FP16 (needs GradScaler to prevent overflow)
        "none" -> AMP disabled
    """
    if amp_dtype == "none" or not torch.cuda.is_available():
        return False, None, False
    if amp_dtype == "auto":
        amp_dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if amp_dtype == "bf16":
        return True, torch.bfloat16, False
    return True, torch.float16, True


# =============================================================================
# Training loop
# =============================================================================

def train_model_hpc(
    model, train_loader, val_loader, optimizer, loss_fn, device,
    num_epochs=100, patience=25, save_path="best_model.pt",
    use_amp=False, amp_dtype="auto", verbose=True,
    scheduler=None, early_stopping_metric="mcc",
    mcc_thresholds=None
):
    """
    Training loop with focal/BCE loss, MCC or loss early stopping,
    optional cosine LR scheduling, and AMP support.
    """
    if mcc_thresholds is None:
        mcc_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    model.to(device)
    best_val_loss = float('inf')
    best_val_mcc  = -1.0
    counter = 0
    train_loss_history, val_loss_history = [], []

    _use_amp, _amp_dtype, _use_scaler = _resolve_amp(amp_dtype if use_amp else "none")
    scaler = torch.cuda.amp.GradScaler(enabled=_use_scaler)

    for epoch in range(1, num_epochs + 1):

        # ---- Training -------------------------------------------------------
        model.train()
        train_loss_accum, total_masked = 0.0, 0

        for batch_idx, batch in enumerate(train_loader):
            embeddings = batch['embeddings'].to(device)
            mask       = batch['mask'].to(device)
            position   = batch['position'].to(device)
            labels     = batch['labels'].to(device)

            optimizer.zero_grad()

            if _use_amp:
                with torch.cuda.amp.autocast(dtype=_amp_dtype):
                    logits = model(embeddings, mask, position)
                    loss   = loss_fn(logits, labels, mask)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(embeddings, mask, position)
                loss   = loss_fn(logits, labels, mask)
                loss.backward()
                optimizer.step()

            if scheduler is not None:
                scheduler.step(epoch - 1 + batch_idx / len(train_loader))

            train_loss_accum += loss.item() * mask.sum().item()
            total_masked     += mask.sum().item()

        train_loss = train_loss_accum / total_masked
        train_loss_history.append(train_loss)
        if verbose:
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")

        # ---- Validation -----------------------------------------------------
        model.eval()
        val_loss_accum, total_masked_val = 0.0, 0
        all_logits, all_labels, all_mask = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embeddings'].to(device)
                mask       = batch['mask'].to(device)
                position   = batch['position'].to(device)
                labels     = batch['labels'].to(device)

                logits = model(embeddings, mask, position)
                loss   = loss_fn(logits, labels, mask)

                val_loss_accum   += loss.item() * mask.sum().item()
                total_masked_val += mask.sum().item()
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
                all_mask.append(mask.cpu())

        val_loss = val_loss_accum / total_masked_val
        val_loss_history.append(val_loss)

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_mask   = torch.cat(all_mask,   dim=0)
        valid_logits = all_logits[all_mask.bool()]
        valid_labels = all_labels[all_mask.bool()]
        probs = torch.sigmoid(valid_logits).numpy()
        true  = valid_labels.numpy()

        if verbose:
            auc   = roc_auc_score(true, probs)
            auprc = average_precision_score(true, probs)
            mcc   = matthews_corrcoef(true, (probs >= 0.5).astype(int))
            print(f"Validation | Loss: {val_loss:.4f} | ROC-AUC: {auc:.4f} | "
                  f"AU-PRC: {auprc:.4f} | MCC@0.5: {mcc:.4f}")

        # ---- Early stopping -------------------------------------------------
        if early_stopping_metric == "mcc":
            epoch_best_mcc = max(
                matthews_corrcoef(true, (probs >= t).astype(int))
                for t in mcc_thresholds
            )
            if verbose:
                print(f"Best val MCC (sweep): {epoch_best_mcc:.4f} "
                      f"(prev best: {best_val_mcc:.4f})")

            if epoch_best_mcc > best_val_mcc:
                best_val_mcc = epoch_best_mcc
                counter = 0
                torch.save(model.state_dict(), save_path)
                if verbose:
                    print(f"Val MCC improved. Saved to {save_path}")
            else:
                counter += 1
                if verbose:
                    print(f"No improvement. Patience: {counter}/{patience}")
                if counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

        else:  # loss-based
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                torch.save(model.state_dict(), save_path)
                if verbose:
                    print(f"Val loss improved. Saved to {save_path}")
            else:
                counter += 1
                if verbose:
                    print(f"No improvement. Patience: {counter}/{patience}")
                if counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

    return train_loss_history, val_loss_history, save_path


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_on_test_hpc(model, test_loader, device, threshold=0.5, verbose=True):
    """Evaluate on test set and return ROC-AUC, AU-PRC, MCC, Precision, Recall."""
    model.to(device)
    model.eval()
    all_logits, all_labels, all_mask = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            embeddings = batch['embeddings'].to(device)
            mask       = batch['mask'].to(device)
            position   = batch['position'].to(device)
            labels     = batch['labels'].to(device)
            logits = model(embeddings, mask, position)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_mask.append(mask.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_mask   = torch.cat(all_mask,   dim=0)
    valid_logits = all_logits[all_mask.bool()]
    valid_labels = all_labels[all_mask.bool()]

    probs = torch.sigmoid(valid_logits).numpy()
    pred  = (probs >= threshold).astype(int)
    true  = valid_labels.numpy()

    metrics = {
        'ROC-AUC':   roc_auc_score(true, probs),
        'AU-PRC':    average_precision_score(true, probs),
        'MCC':       matthews_corrcoef(true, pred),
        'Precision': precision_score(true, pred, zero_division=0),
        'Recall':    recall_score(true, pred, zero_division=0),
    }
    if verbose:
        print("Test metrics:", metrics)
    return metrics


def find_optimal_threshold(model, val_loader, device,
                           thresholds=None, verbose=True):
    """Sweep thresholds on the validation set; return the one maximising MCC."""
    if thresholds is None:
        thresholds = np.arange(0.5, 1.0, 0.1)

    model.to(device)
    model.eval()
    all_logits, all_labels, all_mask = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            embeddings = batch['embeddings'].to(device)
            mask       = batch['mask'].to(device)
            position   = batch['position'].to(device)
            labels     = batch['labels'].to(device)
            logits = model(embeddings, mask, position)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_mask.append(mask.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_mask   = torch.cat(all_mask,   dim=0)
    valid_logits = all_logits[all_mask.bool()]
    valid_labels = all_labels[all_mask.bool()]

    probs = torch.sigmoid(valid_logits).numpy()
    true  = valid_labels.numpy()

    best_mcc, best_threshold, results = -1.0, 0.5, {}
    for t in thresholds:
        pred = (probs >= t).astype(int)
        mcc  = matthews_corrcoef(true, pred)
        results[t] = {
            'MCC':       mcc,
            'Precision': precision_score(true, pred, zero_division=0),
            'Recall':    recall_score(true, pred, zero_division=0),
        }
        if mcc > best_mcc:
            best_mcc, best_threshold = mcc, t

    if verbose:
        print(f"Optimal threshold: {best_threshold:.2f} | MCC: {best_mcc:.4f}")
    return best_threshold, results


# =============================================================================
# Pipeline
# =============================================================================

def build_scheduler(optimizer, cfg):
    """Construct optional LR scheduler from config."""
    scheduler_name = OmegaConf.select(cfg, "training.lr_scheduler", default=None)
    if scheduler_name == "cosine_warm_restart":
        T0     = int(OmegaConf.select(cfg, "training.lr_T0",     default=10))
        T_mult = int(OmegaConf.select(cfg, "training.lr_T_mult", default=2))
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T0, T_mult=T_mult, eta_min=1e-6
        )
    return None


def execute_training_pipeline_hpc(
    model, train_loader, val_loader, test_loader,
    optimizer, loss_fn, device, cfg,
    save_path=None, thresholds=None, use_amp=False, amp_dtype="auto"
):
    """Full training pipeline: train → load best → threshold sweep → test eval."""
    if save_path is None:
        raise ValueError("save_path must be specified.")

    scheduler = build_scheduler(optimizer, cfg)
    early_stopping_metric = OmegaConf.select(cfg, "training.early_stopping_metric",
                                             default="mcc")
    mcc_thresholds = list(OmegaConf.select(
        cfg, "training.mcc_sweep_thresholds",
        default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ))

    train_loss_history, val_loss_history, _ = train_model_hpc(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, loss_fn=loss_fn, device=device,
        num_epochs=cfg.training.num_epochs, patience=cfg.training.patience,
        save_path=save_path, use_amp=use_amp, amp_dtype=amp_dtype,
        scheduler=scheduler, early_stopping_metric=early_stopping_metric,
        mcc_thresholds=mcc_thresholds
    )

    model.load_state_dict(torch.load(save_path, map_location=device,
                                     weights_only=True))
    model.to(device)

    if thresholds is None:
        thresholds = np.arange(0.5, 1.0, 0.1)
    optimal_threshold, threshold_results = find_optimal_threshold(
        model, val_loader, device, thresholds=thresholds
    )
    test_metrics = evaluate_on_test_hpc(
        model, test_loader, device, threshold=optimal_threshold
    )
    return test_metrics, optimal_threshold, train_loss_history, val_loss_history, threshold_results


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LBSP Training Pipeline")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config.yaml")
    parser.add_argument("--override", nargs="*", default=[],
                        help="OmegaConf overrides, e.g. training.batch_size=16")
    args = parser.parse_args()

    cfg = OmegaConf.load(os.path.expanduser(args.config))
    for override in args.override:
        key, value = override.split("=", 1)
        OmegaConf.update(cfg, key, value)

    set_seed(cfg.training.seed)

    train_pkl       = os.path.expanduser(cfg.data.train_df)
    val_pkl         = os.path.expanduser(cfg.data.val_df)
    test_pkl        = os.path.expanduser(cfg.data.test_df)
    h5_path         = os.path.expanduser(cfg.data.h5_embeddings)
    checkpoint_path = os.path.expanduser(cfg.paths.checkpoint)
    results_path    = os.path.expanduser(cfg.paths.results)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(results_path),    exist_ok=True)

    num_workers = int(OmegaConf.select(cfg, "training.num_workers", default=0))

    train_df = joblib.load(train_pkl)
    val_df   = joblib.load(val_pkl)
    test_df  = joblib.load(test_pkl)

    train_loader = get_protein_dataloader(
        train_df, h5_path, batch_size=cfg.training.batch_size,
        shuffle=True, max_len=cfg.model.max_len, num_workers=num_workers
    )
    val_loader = get_protein_dataloader(
        val_df, h5_path, batch_size=cfg.training.batch_size,
        shuffle=False, max_len=cfg.model.max_len, num_workers=num_workers
    )
    test_loader = get_protein_dataloader(
        test_df, h5_path, batch_size=cfg.training.batch_size,
        shuffle=False, max_len=cfg.model.max_len, num_workers=num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(cfg)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {OmegaConf.select(cfg, 'model.model_type', default='mlp')} "
          f"| Trainable parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay
    )
    loss_fn = build_loss_fn(cfg)

    eval_thresholds = list(OmegaConf.select(
        cfg, "evaluation.thresholds", default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ))

    test_metrics, optimal_threshold, train_loss_history, val_loss_history, \
        threshold_results = execute_training_pipeline_hpc(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            cfg=cfg,
            save_path=checkpoint_path,
            thresholds=eval_thresholds,
            use_amp=OmegaConf.select(cfg, "training.use_amp", default=False),
            amp_dtype=OmegaConf.select(cfg, "training.amp_dtype", default="auto")
        )

    save_results(
        {
            "test_metrics":       test_metrics,
            "optimal_threshold":  optimal_threshold,
            "train_loss_history": train_loss_history,
            "val_loss_history":   val_loss_history,
            "threshold_results":  threshold_results,
            "model_type":         OmegaConf.select(cfg, "model.model_type", default="mlp"),
        },
        save_path=results_path
    )
