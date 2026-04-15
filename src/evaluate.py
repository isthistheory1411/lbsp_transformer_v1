import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    precision_score,
    recall_score
)
import numpy as np
from typing import Dict, Tuple


def evaluate_on_test_hpc(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str,
    threshold: float = 0.5,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate trained model on test set (HPC-ready).

    Args:
        model: trained model
        test_loader: DataLoader for test dataset
        device: 'cuda' or 'cpu'
        threshold: probability threshold for binary classification
        verbose: whether to print metrics

    Returns:
        metrics: dict containing ROC-AUC, AU-PRC, MCC, precision, recall
    """
    model.to(device)
    model.eval()
    all_logits, all_labels, all_mask = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            embeddings = batch['embeddings'].to(device)
            mask = batch['mask'].to(device)
            position = batch['position'].to(device)
            labels = batch['labels'].to(device)

            logits = model(embeddings, mask, position)  # [B,L]
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_mask.append(mask.cpu())

    # Concatenate batches
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_mask = torch.cat(all_mask, dim=0)

    # Only valid residues
    valid_logits = all_logits[all_mask.bool()]
    valid_labels = all_labels[all_mask.bool()]

    probs = torch.sigmoid(valid_logits).numpy()
    pred = (probs >= threshold).astype(int)
    true = valid_labels.numpy()

    metrics = {
        'ROC-AUC': roc_auc_score(true, probs),
        'AU-PRC': average_precision_score(true, probs),
        'MCC': matthews_corrcoef(true, pred),
        'Precision': precision_score(true, pred, zero_division=0),
        'Recall': recall_score(true, pred, zero_division=0)
    }

    if verbose:
        print("Test set metrics:", metrics)

    return metrics


def find_optimal_threshold(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: str,
    thresholds: np.ndarray = np.arange(0.5, 1.0, 0.1),
    verbose: bool = True
) -> Tuple[float, Dict[float, Dict[str, float]]]:
    """
    Sweep thresholds on validation set to find the threshold that maximizes MCC.

    Args:
        model: trained model
        val_loader: validation DataLoader
        device: 'cuda' or 'cpu'
        thresholds: array of probability thresholds to test
        verbose: whether to print best threshold

    Returns:
        best_threshold: threshold maximizing MCC
        results: dict mapping threshold -> metrics
    """
    model.to(device)
    model.eval()
    all_logits, all_labels, all_mask = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            embeddings = batch['embeddings'].to(device)
            mask = batch['mask'].to(device)
            position = batch['position'].to(device)
            labels = batch['labels'].to(device)

            logits = model(embeddings, mask, position)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_mask.append(mask.cpu())

    # Concatenate all batches
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_mask = torch.cat(all_mask, dim=0)

    # Flatten only valid residues
    valid_logits = all_logits[all_mask.bool()]
    valid_labels = all_labels[all_mask.bool()]

    probs = torch.sigmoid(valid_logits).numpy()
    true = valid_labels.numpy()

    best_mcc = -1.0
    best_threshold = 0.5
    results = {}

    for t in thresholds:
        pred = (probs >= t).astype(int)
        mcc = matthews_corrcoef(true, pred)
        precision = precision_score(true, pred, zero_division=0)
        recall = recall_score(true, pred, zero_division=0)
        results[t] = {'MCC': mcc, 'Precision': precision, 'Recall': recall}

        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = t

    if verbose:
        print(f"Optimal threshold: {best_threshold:.2f} with MCC: {best_mcc:.4f}")

    return best_threshold, results
