import torch
import pandas as pd
import numpy as np
from src.dataset import get_protein_dataloader
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, precision_score, recall_score

def run_inference(model, df, h5_path, device='cpu', batch_size=32, max_len=1022,
                  threshold=0.5, save_csv=None, compute_metrics_if_labels=True):
    """
    Run per-residue binding site prediction for a given protein dataset.
    Optionally compute metrics if 'binding_vector' is present.

    Args:
        model: trained ResidueMLP
        df: pd.DataFrame with 'dataset_key' (optional 'binding_vector')
        h5_path: path to per-residue embeddings HDF5
        device: 'cuda' or 'cpu'
        batch_size: DataLoader batch size
        max_len: max sequence length
        threshold: probability threshold for binary predictions
        save_csv: optional path to save predictions as CSV
        compute_metrics_if_labels: compute ROC-AUC, MCC, etc., if binding_vector exists

    Returns:
        results: dict with keys:
            'logits': tensor [N,L] raw logits
            'probs': tensor [N,L] sigmoid probabilities
            'preds': tensor [N,L] binary predictions
            'mask': tensor [N,L] valid residue mask
            'metrics' (optional): dict with ROC-AUC, AU-PRC, MCC, Precision, Recall
    """
    model.eval()
    model.to(device)

    loader = get_protein_dataloader(df, h5_path, batch_size=batch_size, shuffle=False,
                                    max_len=max_len, inference=True)

    all_logits, all_mask = [], []
    protein_keys = []

    # If labels exist, collect them
    has_labels = 'binding_vector' in df.columns and compute_metrics_if_labels
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            embeddings = batch['embeddings'].to(device)
            mask = batch['mask'].to(device)
            position = batch['position'].to(device)

            logits = model(embeddings, mask, position)

            all_logits.append(logits.cpu())
            all_mask.append(mask.cpu())

            start_idx = batch_idx * batch_size
            end_idx = start_idx + embeddings.size(0)
            protein_keys.extend(df['dataset_key'].iloc[start_idx:end_idx].tolist())

            if has_labels:
                labels = []
                for i in range(start_idx, end_idx):
                    vec = df['binding_vector'].iloc[i]
                    # pad/truncate to max_len
                    if len(vec) < max_len:
                        vec = np.hstack([vec, np.zeros(max_len - len(vec))])
                    else:
                        vec = vec[:max_len]
                    labels.append(vec)
                all_labels.append(torch.tensor(labels, dtype=torch.float32))

    all_logits = torch.cat(all_logits, dim=0)
    all_mask = torch.cat(all_mask, dim=0)
    probs = torch.sigmoid(all_logits)
    preds = (probs >= threshold).int()

    # -------------------------------
    # Save CSV if requested
    # -------------------------------
    if save_csv is not None:
        rows = []
        for i, protein_key in enumerate(protein_keys):
            valid_len = int(all_mask[i].sum().item())
            for j in range(valid_len):
                rows.append({
                    "protein_key": protein_key,
                    "residue_index": j,
                    "probability": probs[i, j].item(),
                    "prediction": preds[i, j].item()
                })
        df_out = pd.DataFrame(rows)
        df_out.to_csv(save_csv, index=False)
        print(f"Predictions saved to {save_csv}")

    results = {
        "logits": all_logits,
        "probs": probs,
        "preds": preds,
        "mask": all_mask
    }

    # -------------------------------
    # Compute metrics if labels exist
    # -------------------------------
    if has_labels:
        all_labels = torch.cat(all_labels, dim=0)
        valid_logits = all_logits[all_mask.bool()]
        valid_labels = all_labels[all_mask.bool()]

        probs_flat = torch.sigmoid(valid_logits).detach().numpy()
        pred_flat = (probs_flat >= threshold).astype(int)
        true_flat = valid_labels.numpy()

        metrics = {
            'ROC-AUC': roc_auc_score(true_flat, probs_flat),
            'AU-PRC': average_precision_score(true_flat, probs_flat),
            'MCC': matthews_corrcoef(true_flat, pred_flat),
            'Precision': precision_score(true_flat, pred_flat),
            'Recall': recall_score(true_flat, pred_flat)
        }
        results['metrics'] = metrics
        print("Inference metrics (computed from available labels):", metrics)

    return results
