import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor = None
) -> torch.Tensor:
    """
    Binary Cross-Entropy loss per residue, ignoring masked (padded) residues.

    Args:
        logits:     [B, L] model outputs before sigmoid
        labels:     [B, L] ground truth labels (0/1)
        mask:       [B, L] 1 for valid residues, 0 for padded
        pos_weight: optional scalar tensor weighting the positive class

    Returns:
        Scalar: mean BCE over valid residues
    """
    criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    loss = criterion(logits, labels)
    loss = loss * mask
    return loss.sum() / mask.sum()


def masked_focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0
) -> torch.Tensor:
    """
    Focal loss for residue-level binary classification (Lin et al. 2017).

    Down-weights easy-to-classify negatives so training focuses on the
    hard residues near the decision boundary — more effective than
    pos_weight BCE under severe class imbalance.

    Args:
        logits: [B, L] model outputs before sigmoid
        labels: [B, L] ground truth labels (0/1)
        mask:   [B, L] 1 for valid residues, 0 for padded
        alpha:  weight for the positive class (~positive fraction, e.g. 0.1)
        gamma:  focusing exponent (0 = standard BCE; 2.0 is the standard default)

    Returns:
        Scalar: mean focal loss over valid residues
    """
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    p_t = torch.exp(-bce)                                        # probability of the correct class
    focal_weight = alpha * labels + (1.0 - alpha) * (1.0 - labels)
    loss = focal_weight * ((1.0 - p_t) ** gamma) * bce
    loss = loss * mask
    return loss.sum() / mask.sum()


def build_loss_fn(cfg):
    """
    Return a loss callable (logits, labels, mask) -> scalar from OmegaConf config.

    Supported cfg.training.loss_fn values:
        "bce"   -> masked_bce_loss  with pos_weight from cfg.training.pos_weight
        "focal" -> masked_focal_loss with alpha/gamma from cfg.training.*

    The returned function has no external dependencies so it can be passed
    directly into train_model_hpc as a drop-in replacement for the old
    positional pos_weight argument.
    """
    from omegaconf import OmegaConf

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

    else:
        raise ValueError(
            f"Unknown loss_fn '{loss_fn_name}'. Choose from: 'bce', 'focal'."
        )
