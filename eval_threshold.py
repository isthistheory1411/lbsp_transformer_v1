"""
eval_threshold.py — Re-run threshold sweep + test evaluation on a saved checkpoint.

Usage:
    python eval_threshold.py --config config/config.yaml

    # Override thresholds:
    python eval_threshold.py --config config/config.yaml \
        --thresholds 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5

    # Point to a different checkpoint:
    python eval_threshold.py --config config/config.yaml \
        --checkpoint /path/to/best_model.pt
"""

import os
import sys
import argparse
import joblib
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformer_v1 import (
    set_seed,
    build_model,
    get_protein_dataloader,
    find_optimal_threshold,
    evaluate_on_test_hpc,
)


def main():
    parser = argparse.ArgumentParser(description="Threshold sweep + test eval on saved checkpoint")
    parser.add_argument("--config",     type=str, required=True,
                        help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override checkpoint path from config")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        help="Thresholds to sweep (space-separated floats)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    cfg = OmegaConf.load(os.path.expanduser(args.config))
    set_seed(cfg.training.seed)

    val_pkl         = os.path.expanduser(cfg.data.val_df)
    test_pkl        = os.path.expanduser(cfg.data.test_df)
    h5_path         = os.path.expanduser(cfg.data.h5_embeddings)
    checkpoint_path = os.path.expanduser(
        args.checkpoint if args.checkpoint else cfg.paths.checkpoint
    )

    print(f"Checkpoint : {checkpoint_path}")
    print(f"Thresholds : {args.thresholds}")

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")

    # ------------------------------------------------------------------
    # DataLoaders (val + test only)
    # ------------------------------------------------------------------
    val_df  = joblib.load(val_pkl)
    test_df = joblib.load(test_pkl)

    num_workers = int(OmegaConf.select(cfg, "training.num_workers", default=0))

    val_loader = get_protein_dataloader(
        val_df, h5_path,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        max_len=cfg.model.max_len,
        num_workers=num_workers,
    )
    test_loader = get_protein_dataloader(
        test_df, h5_path,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        max_len=cfg.model.max_len,
        num_workers=num_workers,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(cfg)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print(f"Model      : {OmegaConf.select(cfg, 'model.model_type', default='mlp')}")

    # ------------------------------------------------------------------
    # Threshold sweep on validation set
    # ------------------------------------------------------------------
    print("\n--- Threshold sweep (validation set) ---")
    optimal_threshold, threshold_results = find_optimal_threshold(
        model, val_loader, device, thresholds=args.thresholds, verbose=True
    )

    print("\nFull sweep results:")
    print(f"{'Threshold':>10} {'MCC':>8} {'Precision':>10} {'Recall':>8}")
    for t, m in sorted(threshold_results.items()):
        marker = " <-- optimal" if abs(t - optimal_threshold) < 1e-9 else ""
        print(f"{t:>10.2f} {m['MCC']:>8.4f} {m['Precision']:>10.4f} {m['Recall']:>8.4f}{marker}")

    # ------------------------------------------------------------------
    # Test evaluation with optimal threshold
    # ------------------------------------------------------------------
    print(f"\n--- Test evaluation (threshold = {optimal_threshold:.2f}) ---")
    test_metrics = evaluate_on_test_hpc(model, test_loader, device,
                                        threshold=optimal_threshold, verbose=False)
    print(f"ROC-AUC   : {test_metrics['ROC-AUC']:.4f}")
    print(f"AU-PRC    : {test_metrics['AU-PRC']:.4f}")
    print(f"MCC       : {test_metrics['MCC']:.4f}")
    print(f"Precision : {test_metrics['Precision']:.4f}")
    print(f"Recall    : {test_metrics['Recall']:.4f}")


if __name__ == "__main__":
    main()
