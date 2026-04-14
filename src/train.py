import torch
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
from typing import Callable, List, Optional, Tuple


def _resolve_amp(amp_dtype: str):
    """
    Resolve amp_dtype config string to (use_amp, dtype, use_scaler).

    amp_dtype values:
        "auto"  -> BF16 if supported (Ampere/Ada GPUs), else FP16
        "bf16"  -> force BF16 (no GradScaler needed — same range as FP32)
        "fp16"  -> force FP16 (GradScaler required to prevent overflow)
        "none"  -> disable AMP

    Returns:
        use_amp (bool), dtype (torch.dtype or None), use_scaler (bool)
    """
    if amp_dtype == "none" or not torch.cuda.is_available():
        return False, None, False
    if amp_dtype == "auto":
        amp_dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if amp_dtype == "bf16":
        return True, torch.bfloat16, False
    return True, torch.float16, True


def train_model_hpc(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    device: str,
    num_epochs: int = 100,
    patience: int = 25,
    save_path: str = "best_model.pt",
    use_amp: bool = False,
    amp_dtype: str = "auto",
    verbose: bool = True,
    scheduler=None,
    early_stopping_metric: str = "mcc",
    mcc_thresholds: Optional[List[float]] = None
) -> Tuple[List[float], List[float], str]:
    """
    Training loop with:
        - Configurable loss function (BCE or focal, via loss_fn callable)
        - MCC-based or loss-based early stopping
        - Optional cosine LR scheduling
        - AMP (mixed precision) support

    Args:
        model:                   PyTorch model with forward(embeddings, mask, position)
        train_loader:            Training DataLoader
        val_loader:              Validation DataLoader
        optimizer:               Instantiated optimizer
        loss_fn:                 Callable (logits, labels, mask) -> scalar loss
        device:                  'cuda' or 'cpu'
        num_epochs:              Maximum training epochs
        patience:                Early stopping patience (epochs without improvement)
        save_path:               Path to save the best model checkpoint
        use_amp:                 Enable automatic mixed precision (CUDA only)
        verbose:                 Print per-epoch metrics
        scheduler:               Optional LR scheduler (stepped per batch)
        early_stopping_metric:   'mcc' (recommended) or 'loss'
        mcc_thresholds:          Threshold sweep for MCC early stopping;
                                 defaults to [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    Returns:
        train_loss_history, val_loss_history, best_model_path
    """
    if mcc_thresholds is None:
        mcc_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    model.to(device)

    # Tracking variables depend on the chosen early-stopping metric
    best_val_loss = float('inf')
    best_val_mcc = -1.0
    counter = 0

    train_loss_history = []
    val_loss_history = []

    # Resolve AMP settings — use_amp flag overrides amp_dtype to "none" if False
    _use_amp, _amp_dtype, _use_scaler = _resolve_amp(amp_dtype if use_amp else "none")
    scaler = torch.amp.GradScaler('cuda', enabled=_use_scaler)

    for epoch in range(1, num_epochs + 1):

        # ----------------------------------------------------------------
        # Training step
        # ----------------------------------------------------------------
        model.train()
        train_loss_accum = 0.0
        total_masked = 0

        for batch_idx, batch in enumerate(train_loader):
            embeddings = batch['embeddings'].to(device)
            mask       = batch['mask'].to(device)
            position   = batch['position'].to(device)
            labels     = batch['labels'].to(device)

            optimizer.zero_grad()

            if _use_amp:
                with torch.amp.autocast('cuda', dtype=_amp_dtype):
                    logits = model(embeddings, mask, position)
                    loss = loss_fn(logits, labels, mask)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(embeddings, mask, position)
                loss = loss_fn(logits, labels, mask)
                loss.backward()
                optimizer.step()

            # Step per-batch scheduler (CosineAnnealingWarmRestarts convention)
            if scheduler is not None:
                scheduler.step(epoch - 1 + batch_idx / len(train_loader))

            train_loss_accum += loss.item() * mask.sum().item()
            total_masked += mask.sum().item()

        train_loss = train_loss_accum / total_masked
        train_loss_history.append(train_loss)

        if verbose:
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")

        # ----------------------------------------------------------------
        # Validation step
        # ----------------------------------------------------------------
        model.eval()
        val_loss_accum = 0.0
        total_masked_val = 0
        all_logits, all_labels, all_mask = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embeddings'].to(device)
                mask       = batch['mask'].to(device)
                position   = batch['position'].to(device)
                labels     = batch['labels'].to(device)

                logits = model(embeddings, mask, position)
                loss = loss_fn(logits, labels, mask)

                val_loss_accum += loss.item() * mask.sum().item()
                total_masked_val += mask.sum().item()

                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
                all_mask.append(mask.cpu())

        val_loss = val_loss_accum / total_masked_val
        val_loss_history.append(val_loss)

        # Flatten valid residues for metrics
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
            pred  = (probs >= 0.5).astype(int)
            mcc   = matthews_corrcoef(true, pred)
            print(f"Validation | Loss: {val_loss:.4f} | ROC-AUC: {auc:.4f} | "
                  f"AU-PRC: {auprc:.4f} | MCC@0.5: {mcc:.4f}")

        # ----------------------------------------------------------------
        # Early stopping
        # ----------------------------------------------------------------
        if early_stopping_metric == "mcc":
            # Sweep thresholds to find best MCC at this epoch
            epoch_best_mcc = max(
                matthews_corrcoef(true, (probs >= t).astype(int))
                for t in mcc_thresholds
            )

            if verbose:
                print(f"Best val MCC (threshold sweep): {epoch_best_mcc:.4f} "
                      f"(prev best: {best_val_mcc:.4f})")

            if epoch_best_mcc > best_val_mcc:
                best_val_mcc = epoch_best_mcc
                counter = 0
                torch.save(model.state_dict(), save_path)
                if verbose:
                    print(f"Val MCC improved. Model saved to {save_path}")
            else:
                counter += 1
                if verbose:
                    print(f"No improvement. Patience counter: {counter}/{patience}")
                if counter >= patience:
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch}")
                    break

        else:  # early_stopping_metric == "loss"
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                torch.save(model.state_dict(), save_path)
                if verbose:
                    print(f"Val loss improved. Model saved to {save_path}")
            else:
                counter += 1
                if verbose:
                    print(f"No improvement. Patience counter: {counter}/{patience}")
                if counter >= patience:
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch}")
                    break

    return train_loss_history, val_loss_history, save_path
