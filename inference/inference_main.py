import os
import torch
import joblib
import argparse
import json
from omegaconf import OmegaConf

from src.model import build_model
from src.utils import set_seed
from inference.inference import run_inference


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # Command-line arguments
    # -----------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Run per-residue binding site inference")
    parser.add_argument("--config", type=str, required=True, help="Path to inference_config.yaml")
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="OmegaConf key=value overrides, e.g. inference.threshold=0.7"
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
    if hasattr(cfg.inference, "seed"):
        set_seed(cfg.inference.seed)

    # -----------------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------------
    requested_device = cfg.inference.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model = build_model(cfg)
    checkpoint_path = os.path.expanduser(cfg.paths.checkpoint)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------
    df     = joblib.load(os.path.expanduser(cfg.data.test_df))
    h5_path = os.path.expanduser(cfg.data.h5_embeddings)

    # -----------------------------------------------------------------------
    # Output path
    # -----------------------------------------------------------------------
    save_csv = os.path.expanduser(cfg.paths.inference_csv)
    os.makedirs(os.path.dirname(save_csv), exist_ok=True)

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------
    results = run_inference(
        model=model,
        df=df,
        h5_path=h5_path,
        device=device,
        batch_size=cfg.inference.batch_size,
        max_len=cfg.model.max_len,
        threshold=cfg.inference.threshold,
        save_csv=save_csv
    )

    print(f"Inference complete. Predictions saved to {save_csv}")

    # -----------------------------------------------------------------------
    # Save metrics if labels were present
    # -----------------------------------------------------------------------
    if "metrics" in results:
        metrics_path = os.path.splitext(save_csv)[0] + "_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(results["metrics"], f, indent=4)
        print(f"Inference metrics saved to {metrics_path}")
