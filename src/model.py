import copy
import math
import torch
import torch.nn as nn
from typing import List


# ---------------------------------------------------------------------------
# Positional encodings
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani et al. 2017).
    Added to the projected input embeddings before the Transformer encoder.
    """
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
        """x: [B, L, d_model]"""
        return self.dropout(x + self.pe[:, :x.size(1)])


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional embedding (nn.Embedding over position indices).
    """
    def __init__(self, d_model: int, max_len: int = 1022, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d_model]"""
        L = x.size(1)
        positions = torch.arange(L, device=x.device)
        return self.dropout(x + self.embedding(positions))


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) — Su et al. 2021 (RoFormer).

    Unlike sinusoidal/learned encodings, RoPE is NOT added to the input
    embeddings. Instead it is applied directly to Q and K inside each
    attention layer so that the dot-product Q·Kᵀ depends only on the
    relative distance (m - n) between positions m and n.

    dim should equal d_model // nhead (the per-head dimension). The same
    rotation is applied to every head, which is equivalent to the original
    formulation.
    """
    def __init__(self, dim: int, max_len: int = 1022):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)           # [max_len, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)    # [max_len, dim]
        self.register_buffer('cos', emb.cos())     # [max_len, dim]
        self.register_buffer('sin', emb.sin())     # [max_len, dim]

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor, nhead: int) -> torch.Tensor:
        """
        Apply RoPE per attention head.

        x:     [B, L, d_model]
        nhead: number of attention heads (d_model must be divisible by nhead)

        Returns [B, L, d_model] with rotation applied to each head's dims.
        """
        B, L, d_model = x.shape
        d_head = d_model // nhead

        cos = self.cos[:L].unsqueeze(0)            # [1, L, d_head]
        sin = self.sin[:L].unsqueeze(0)            # [1, L, d_head]

        x = x.view(B, L, nhead, d_head)
        x = x * cos.unsqueeze(2) + self._rotate_half(x) * sin.unsqueeze(2)
        return x.view(B, L, d_model)


# ---------------------------------------------------------------------------
# RoPE-aware Transformer encoder (mirrors nn.TransformerEncoder + Pre-LN
# but hooks into Q and K before self-attention)
# ---------------------------------------------------------------------------

class _RoPEEncoderLayer(nn.Module):
    """
    Single Pre-LN Transformer encoder layer with RoPE applied to Q and K.
    Not intended to be used directly; instantiated by ResidueTransformerMLP.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        rope: RotaryEmbedding
    ):
        super().__init__()
        self.nhead = nhead
        self.rope = rope
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
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

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask=None
    ) -> torch.Tensor:
        # Pre-LN: normalise first, then attend
        normed = self.norm1(src)
        q = self.rope(normed, self.nhead)
        k = self.rope(normed, self.nhead)
        v = normed
        attn_out, _ = self.self_attn(q, k, v, key_padding_mask=src_key_padding_mask)
        src = src + self.drop1(attn_out)
        src = src + self.drop2(self.ff(self.norm2(src)))
        return src


class _RoPEEncoder(nn.Module):
    """Stack of _RoPEEncoderLayer, API-compatible with nn.TransformerEncoder."""
    def __init__(self, layer: _RoPEEncoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(num_layers)]
        )

    def forward(self, src: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
        x = src
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x


# ---------------------------------------------------------------------------
# Baseline MLP (unchanged from lbsp_baseline_mlp_v2)
# ---------------------------------------------------------------------------

class ResidueMLP(nn.Module):
    """
    Per-residue MLP for ligand binding site prediction.

    Combines:
        - per-residue embeddings
        - projected per-protein embedding (masked mean pool)
        - normalized residue position scalar

    Supports configurable hidden layers, dropout, and LayerNorm.
    """
    def __init__(
        self,
        residue_emb_dim: int,
        protein_emb_dim: int = 256,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.1
    ):
        super().__init__()
        self.protein_proj = nn.Linear(residue_emb_dim, protein_emb_dim)

        layers = []
        input_dim = residue_emb_dim + protein_emb_dim + 1
        last_dim = input_dim

        for h in hidden_dims:
            layers.extend([
                nn.Linear(last_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            last_dim = h

        layers.append(nn.Linear(last_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        embeddings: torch.Tensor,  # [B, L, D_res]
        mask: torch.Tensor,        # [B, L]
        position: torch.Tensor     # [B, L, 1]
    ) -> torch.Tensor:
        mask_exp = mask.unsqueeze(-1)
        sum_emb = (embeddings * mask_exp).sum(dim=1)
        lengths = mask.sum(dim=1, keepdim=True)
        mean_emb = sum_emb / lengths
        protein_feats = self.protein_proj(mean_emb)
        protein_feats_exp = protein_feats.unsqueeze(1).expand(-1, embeddings.size(1), -1)
        x = torch.cat([embeddings, protein_feats_exp, position], dim=-1)
        logits = self.mlp(x)
        return logits.squeeze(-1)


# ---------------------------------------------------------------------------
# Transformer-based model
# ---------------------------------------------------------------------------

class ResidueTransformerMLP(nn.Module):
    """
    Transformer encoder head for per-residue binding site prediction.

    Architecture:
        1. Project residue embeddings to d_model
        2. Positional encoding: sinusoidal (fixed) | learned (nn.Embedding) |
           RoPE (no additive encoding; rotation applied to Q and K inside attention)
        3. Transformer encoder (Pre-LN, batch_first) with padding mask
        4. Global context: masked mean pool or attention pool (learnable query)
        5. Concatenate contextualised residue features + global context
        6. MLP head -> per-residue logit

    The forward signature (embeddings, mask, position) is identical to
    ResidueMLP so all existing training, evaluation, and inference code
    works without modification.
    """
    def __init__(
        self,
        residue_emb_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_transformer_layers: int = 2,
        transformer_ff_dim: int = 512,
        dropout: float = 0.1,
        max_len: int = 1022,
        pos_encoding_type: str = "sinusoidal",  # "sinusoidal" | "learned" | "rope"
        global_pool: str = "mean"               # "mean" | "attention"
    ):
        super().__init__()

        # 1. Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(residue_emb_dim, d_model),
            nn.LayerNorm(d_model)
        )

        # 2. Positional encoding
        # RoPE: no additive encoding on the input; rotation is applied inside
        # each attention layer to Q and K.
        # Sinusoidal / learned: added to projected embeddings before the encoder.
        self.pos_encoding_type = pos_encoding_type
        if pos_encoding_type == "learned":
            self.pos_enc = LearnedPositionalEncoding(d_model, max_len, dropout)
        elif pos_encoding_type == "rope":
            self.pos_enc = None
        else:  # default: sinusoidal
            self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        # 3. Transformer encoder (Pre-LN)
        if pos_encoding_type == "rope":
            d_head = d_model // nhead
            assert d_model % nhead == 0, "d_model must be divisible by nhead for RoPE"
            rope = RotaryEmbedding(dim=d_head, max_len=max_len)
            rope_layer = _RoPEEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=transformer_ff_dim,
                dropout=dropout,
                rope=rope
            )
            self.transformer = _RoPEEncoder(rope_layer, num_layers=num_transformer_layers)
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=transformer_ff_dim,
                dropout=dropout,
                batch_first=True,
                norm_first=True   # Pre-LN: more stable for shallow Transformers
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_transformer_layers
            )

        # 4. Global context pooling
        self.global_pool = global_pool
        if global_pool == "attention":
            self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))
            self.pool_attn  = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True
            )

        # 5. MLP head over [transformer_out || global_ctx]
        mlp_input_dim = d_model * 2
        self.mlp_head = nn.Sequential(
            nn.Linear(mlp_input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(
        self,
        embeddings: torch.Tensor,  # [B, L, D_res]
        mask: torch.Tensor,        # [B, L]  1=valid, 0=pad
        position: torch.Tensor     # [B, L, 1]  (accepted but unused; pos enc is internal)
    ) -> torch.Tensor:
        # Project + positional encoding
        x = self.input_proj(embeddings)                     # [B, L, d_model]
        if self.pos_enc is not None:
            x = self.pos_enc(x)                             # [B, L, d_model]
        # RoPE: no additive encoding here; rotation applied inside attention

        # Transformer: mask out padded positions
        # TransformerEncoder expects True = ignore; our mask is 1=valid so invert
        pad_mask = ~mask.bool()              # [B, L]  True = padded
        x = self.transformer(x, src_key_padding_mask=pad_mask)  # [B, L, d_model]

        # Global context: aggregate over all residues into a single protein vector
        if self.global_pool == "attention":
            # Learnable query attends over Transformer output; reuse pad_mask
            B = x.size(0)
            query = self.pool_query.expand(B, -1, -1)                            # [B, 1, d_model]
            global_ctx, _ = self.pool_attn(query, x, x,
                                           key_padding_mask=pad_mask)            # [B, 1, d_model]
            global_ctx = global_ctx.squeeze(1)                                   # [B, d_model]
        else:
            mask_exp   = mask.unsqueeze(-1)
            global_ctx = (x * mask_exp).sum(dim=1) / mask.sum(dim=1, keepdim=True)  # [B, d_model]
        global_ctx_exp = global_ctx.unsqueeze(1).expand(-1, x.size(1), -1)      # [B, L, d_model]

        # Concatenate and project to logit
        combined = torch.cat([x, global_ctx_exp], dim=-1)  # [B, L, 2*d_model]
        logits = self.mlp_head(combined)                    # [B, L, 1]
        return logits.squeeze(-1)                           # [B, L]


# ---------------------------------------------------------------------------
# BiLSTM-based model (efficient alternative for ablations)
# ---------------------------------------------------------------------------

class ResidueBiLSTMMLP(nn.Module):
    """
    Bidirectional LSTM encoder for per-residue binding site prediction.

    Captures sequential dependencies at lower memory cost than the Transformer.
    Useful for ablations or compute-constrained settings.

    Forward signature identical to ResidueMLP and ResidueTransformerMLP.
    """
    def __init__(
        self,
        residue_emb_dim: int,
        d_model: int = 256,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        dropout: float = 0.1,
        global_pool: str = "mean",   # "mean" | "attention"
        pool_nhead: int = 8          # attention heads for pooling (lstm_hidden_size*2 must be divisible)
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(residue_emb_dim, d_model),
            nn.LayerNorm(d_model)
        )

        # BiLSTM: output dim = lstm_hidden_size * 2
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0
        )
        lstm_out_dim = lstm_hidden_size * 2

        # Global context pooling
        self.global_pool = global_pool
        if global_pool == "attention":
            assert lstm_out_dim % pool_nhead == 0, (
                f"lstm_hidden_size*2 ({lstm_out_dim}) must be divisible by pool_nhead ({pool_nhead})"
            )
            self.pool_query = nn.Parameter(torch.randn(1, 1, lstm_out_dim))
            self.pool_attn  = nn.MultiheadAttention(
                lstm_out_dim, pool_nhead, dropout=dropout, batch_first=True
            )

        # MLP head over [lstm_out || global_ctx]
        mlp_input_dim = lstm_out_dim * 2
        self.mlp_head = nn.Sequential(
            nn.Linear(mlp_input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(
        self,
        embeddings: torch.Tensor,  # [B, L, D_res]
        mask: torch.Tensor,        # [B, L]
        position: torch.Tensor     # [B, L, 1]  (accepted but unused)
    ) -> torch.Tensor:
        x = self.input_proj(embeddings)       # [B, L, d_model]

        # Pack for efficient LSTM over variable-length sequences
        lengths = mask.sum(dim=1).long().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=embeddings.size(1)
        )                                     # [B, L, lstm_hidden*2]

        # Zero out any residual values at padded positions
        x = x * mask.unsqueeze(-1)

        # Global context: aggregate over all residues into a single protein vector
        pad_mask = ~mask.bool()   # [B, L]  True = padded (MultiheadAttention convention)
        if self.global_pool == "attention":
            B = x.size(0)
            query = self.pool_query.expand(B, -1, -1)                        # [B, 1, lstm_out_dim]
            global_ctx, _ = self.pool_attn(query, x, x,
                                           key_padding_mask=pad_mask)        # [B, 1, lstm_out_dim]
            global_ctx = global_ctx.squeeze(1)                               # [B, lstm_out_dim]
        else:
            mask_exp   = mask.unsqueeze(-1)
            global_ctx = (x * mask_exp).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        global_ctx_exp = global_ctx.unsqueeze(1).expand(-1, x.size(1), -1)

        combined = torch.cat([x, global_ctx_exp], dim=-1)
        logits = self.mlp_head(combined)
        return logits.squeeze(-1)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg) -> nn.Module:
    """
    Instantiate the correct model class from OmegaConf config.

    Supported model_type values:
        "mlp"              -> ResidueMLP  (baseline)
        "transformer_mlp"  -> ResidueTransformerMLP
        "bilstm_mlp"       -> ResidueBiLSTMMLP

    All new keys are read with OmegaConf.select so old configs that lack
    the new fields remain backward compatible.
    """
    from omegaconf import OmegaConf

    model_type = OmegaConf.select(cfg, "model.model_type", default="mlp")

    if model_type == "mlp":
        return ResidueMLP(
            residue_emb_dim=cfg.model.residue_emb_dim,
            protein_emb_dim=OmegaConf.select(cfg, "model.protein_emb_dim", default=256),
            hidden_dims=list(OmegaConf.select(cfg, "model.hidden_dims", default=[512, 256, 128])),
            dropout=cfg.model.dropout
        )

    elif model_type == "transformer_mlp":
        # Resolve pos_encoding_type with backward compatibility:
        # New configs set model.pos_encoding_type directly.
        # Old configs may only have model.use_learned_pos_encoding (bool).
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

    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            "Choose from: 'mlp', 'transformer_mlp', 'bilstm_mlp'."
        )
