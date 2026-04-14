import torch
import numpy as np
import random
import joblib 

def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across PyTorch, NumPy, and Python.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_results(results, save_path):
    joblib.dump(results, save_path, compress=3)
    print(f"Training results saved to {save_path}")
