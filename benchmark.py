"""
benchmark.py — Evaluate a saved checkpoint on any labelled dataset (e.g. COACH420).

Usage:
    python benchmark.py \
        --config config/config.yaml \
        --df /path/to/coach420.pkl \
        --h5 /path/to/coach420_embeddings.h5 \
        --threshold 0.30

    # Sweep thresholds instead of using a fixed one:
    python benchmark.py \
        --config config/config.yaml \
        --df /path/to/coach420.pkl \
        --h5 /path/to/coach420_embeddings.h5 \
        --sweep

    # Override checkpoint path:
    python benchmark.py \
        --config config/config.yaml \
        --df /path/to/coach420.pkl \
        --h5 /path/to/coach420_embeddings.h5 \
        --threshold 0.30 \
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
    parser = argparse.ArgumentParser(description="Benchmark a saved checkpoint on a labelled dataset")
    parser.add_argument("--config",     type=str, required=True,
                        help="Path to config.yaml")
    parser.add_argument("--df",         type=str, required=True,
                        help="Path to benchmark DataFrame (.pkl)")
    parser.add_argument("--h5",         type=str, required=True,
                        help="Path to benchmark HDF5 embeddings (.h5)")
    parser.add_argument("--threshold",  type=float, default=None,
                        help="Fixed threshold to evaluate at (e.g. 0.30)")
    parser.add_argument("--sweep",      action="store_true",
                        help="Sweep thresholds on the benchmark set to find optimal MCC "
                             "(use when no validation threshold is available)")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        help="Thresholds to sweep when --sweep is set")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override checkpoint path from config")
    args = parser.parse_args()

    if args.threshold is None and not args.sweep:
        parser.error("Provide either --threshold <value> or --sweep")

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    cfg = OmegaConf.load(os.path.expanduser(args.config))
    set_seed(cfg.training.seed)

    checkpoint_path = os.path.expanduser(
        args.checkpoint if args.checkpoint else cfg.paths.checkpoint
    )

    print(f"Checkpoint : {checkpoint_path}")
    print(f"Dataset    : {args.df}")

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------
    df_path = os.path.expanduser(args.df)
    if df_path.endswith(".json"):
        import pandas as pd
        df = pd.read_json(df_path, orient="records")
    else:
        df = joblib.load(df_path)
    print(f"Proteins   : {len(df)}")

    loader = get_protein_dataloader(
        df,
        os.path.expanduser(args.h5),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        max_len=cfg.model.max_len,
        num_workers=0,  # CPU-safe
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
    # Threshold sweep or fixed evaluation
    # ------------------------------------------------------------------
    if args.sweep:
        print("\n--- Threshold sweep ---")
        optimal_threshold, threshold_results = find_optimal_threshold(
            model, loader, device, thresholds=args.thresholds, verbose=True
        )
        print("\nFull sweep results:")
        print(f"{'Threshold':>10} {'MCC':>8} {'Precision':>10} {'Recall':>8}")
        for t, m in sorted(threshold_results.items()):
            marker = " <-- optimal" if abs(t - optimal_threshold) < 1e-9 else ""
            print(f"{t:>10.2f} {m['MCC']:>8.4f} {m['Precision']:>10.4f} {m['Recall']:>8.4f}{marker}")
        threshold = optimal_threshold
    else:
        threshold = args.threshold

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------
    print(f"\n--- Benchmark results (threshold = {threshold:.2f}) ---")
    metrics = evaluate_on_test_hpc(model, loader, device, threshold=threshold, verbose=False)
    print(f"ROC-AUC   : {metrics['ROC-AUC']:.4f}")
    print(f"AU-PRC    : {metrics['AU-PRC']:.4f}")
    print(f"MCC       : {metrics['MCC']:.4f}")
    print(f"Precision : {metrics['Precision']:.4f}")
    print(f"Recall    : {metrics['Recall']:.4f}")


if __name__ == "__main__":
    main()
