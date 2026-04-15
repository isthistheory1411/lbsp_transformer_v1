import os
import torch
import joblib
import argparse
import numpy as np
from omegaconf import OmegaConf

from src.dataset import get_protein_dataloader
from src.model import build_model
from src.loss import build_loss_fn
from src.train import train_model_hpc
from src.evaluate import evaluate_on_test_hpc, find_optimal_threshold
from src.utils import set_seed, save_results


def build_scheduler(optimizer, cfg):
    """
    Construct an optional LR scheduler from config.

    Supported cfg.training.lr_scheduler values:
        "cosine_warm_restart" -> CosineAnnealingWarmRestarts
        null / omitted        -> None (constant LR)
    """
    scheduler_name = OmegaConf.select(cfg, "training.lr_scheduler", default=None)

    if scheduler_name == "cosine_warm_restart":
        T0     = int(OmegaConf.select(cfg, "training.lr_T0",   default=10))
        T_mult = int(OmegaConf.select(cfg, "training.lr_T_mult", default=2))
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T0, T_mult=T_mult, eta_min=1e-6
        )

    return None


def execute_training_pipeline_hpc(
    model,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    loss_fn,
    device,
    cfg,
    save_path=None,
    thresholds=None,
    use_amp=False,
    amp_dtype="auto"
):
    """
    Full HPC-ready training pipeline.

    Args:
        model:        Instantiated model
        train_loader: Training DataLoader
        val_loader:   Validation DataLoader
        test_loader:  Test DataLoader
        optimizer:    Instantiated optimizer
        loss_fn:      Callable (logits, labels, mask) -> scalar
        device:       torch.device
        cfg:          OmegaConf config (used for scheduler and early-stopping params)
        save_path:    Path for best model checkpoint
        thresholds:   Threshold array for post-training MCC sweep
    """
    if save_path is None:
        raise ValueError("You must specify a full path for the model checkpoint (save_path).")

    scheduler = build_scheduler(optimizer, cfg)

    early_stopping_metric = OmegaConf.select(cfg, "training.early_stopping_metric", default="mcc")
    mcc_thresholds = list(OmegaConf.select(
        cfg, "training.mcc_sweep_thresholds",
        default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ))

    # 1. Train with early stopping
    train_loss_history, val_loss_history, _ = train_model_hpc(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        num_epochs=cfg.training.num_epochs,
        patience=cfg.training.patience,
        save_path=save_path,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        scheduler=scheduler,
        early_stopping_metric=early_stopping_metric,
        mcc_thresholds=mcc_thresholds
    )

    # 2. Load best model checkpoint
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    model.to(device)

    # 3. Find optimal threshold on validation set
    if thresholds is None:
        thresholds = np.arange(0.5, 1.0, 0.1)
    optimal_threshold, threshold_results = find_optimal_threshold(
        model, val_loader, device, thresholds=thresholds
    )

    # 4. Evaluate on test set
    test_metrics = evaluate_on_test_hpc(
        model, test_loader, device, threshold=optimal_threshold
    )

    return test_metrics, optimal_threshold, train_loss_history, val_loss_history, threshold_results


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # Command-line arguments
    # -----------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Ligand Binding Site Prediction Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="OmegaConf key=value overrides, e.g. training.batch_size=16"
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load config
    # -----------------------------------------------------------------------
    cfg = OmegaConf.load(os.path.expanduser(args.config))
    for override in args.override:
        key, value = override.split("=", 1)
        OmegaConf.update(cfg, key, value)

    # -----------------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------------
    set_seed(cfg.training.seed)

    # -----------------------------------------------------------------------
    # Paths
    # -----------------------------------------------------------------------
    train_pkl      = os.path.expanduser(cfg.data.train_df)
    val_pkl        = os.path.expanduser(cfg.data.val_df)
    test_pkl       = os.path.expanduser(cfg.data.test_df)
    h5_path        = os.path.expanduser(cfg.data.h5_embeddings)
    checkpoint_path = os.path.expanduser(cfg.paths.checkpoint)
    results_path   = os.path.expanduser(cfg.paths.results)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(results_path),    exist_ok=True)

    # -----------------------------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------------------------
    train_df = joblib.load(train_pkl)
    val_df   = joblib.load(val_pkl)
    test_df  = joblib.load(test_pkl)

    train_loader = get_protein_dataloader(
        train_df, h5_path,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        max_len=cfg.model.max_len
    )
    val_loader = get_protein_dataloader(
        val_df, h5_path,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        max_len=cfg.model.max_len
    )
    test_loader = get_protein_dataloader(
        test_df, h5_path,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        max_len=cfg.model.max_len
    )

    # -----------------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model = build_model(cfg)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {OmegaConf.select(cfg, 'model.model_type', default='mlp')} "
          f"| Trainable parameters: {total_params:,}")

    # -----------------------------------------------------------------------
    # Optimizer and loss
    # -----------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay
    )
    loss_fn = build_loss_fn(cfg)

    # -----------------------------------------------------------------------
    # Training pipeline
    # -----------------------------------------------------------------------
    eval_thresholds = list(OmegaConf.select(
        cfg, "evaluation.thresholds", default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ))

    test_metrics, optimal_threshold, train_loss_history, val_loss_history, threshold_results = \
        execute_training_pipeline_hpc(
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

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    results_to_save = {
        "test_metrics":        test_metrics,
        "optimal_threshold":   optimal_threshold,
        "train_loss_history":  train_loss_history,
        "val_loss_history":    val_loss_history,
        "threshold_results":   threshold_results,
        "model_type":          OmegaConf.select(cfg, "model.model_type", default="mlp")
    }
    save_results(results_to_save, save_path=results_path)
