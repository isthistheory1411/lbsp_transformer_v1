import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import pandas as pd


class ProteinDataset(Dataset):
    """
    PyTorch Dataset for protein ligand binding site prediction.
    
    Supports optional inference mode (no labels).
    
    Args:
        df (pd.DataFrame): DataFrame containing at least 'dataset_key' and optionally 'binding_vector'.
        h5_path (str): Path to HDF5 file storing protein embeddings.
        max_len (int): Maximum sequence length; shorter sequences are padded.
        inference (bool): If True, dataset will ignore labels for inference.
    """
    def __init__(self, df, h5_path, max_len=1022, inference=False):
        self.df = df.reset_index(drop=True)
        self.h5_path = h5_path
        self.max_len = max_len
        self.inference = inference

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        protein_key = row['dataset_key']
        labels = None if self.inference else row['binding_vector']

        # Load embeddings from HDF5.
        # File is opened lazily on first access per worker so that each
        # DataLoader worker gets its own handle after forking — required for
        # safe use with num_workers > 0.
        if not hasattr(self, '_h5f'):
            self._h5f = h5py.File(self.h5_path, 'r')
        if protein_key not in self._h5f:
            raise KeyError(f"Protein key '{protein_key}' not found in {self.h5_path} file")
        emb = self._h5f[protein_key][:]  # shape [L_i, D]

        L_i, D = emb.shape

        # Pad embeddings and labels if sequence shorter than max_len
        if L_i < self.max_len:
            pad_len = self.max_len - L_i
            emb = np.vstack([emb, np.zeros((pad_len, D), dtype=np.float32)])
            if not self.inference:
                labels = np.hstack([labels.astype(np.float32), np.zeros(pad_len, dtype=np.float32)])
        else:
            emb = emb[:self.max_len]
            if not self.inference:
                labels = labels[:self.max_len]

        # Create mask and position encodings
        mask = np.zeros(self.max_len, dtype=np.float32)
        mask[:min(L_i, self.max_len)] = 1.0
        pos = (np.arange(self.max_len) / self.max_len).astype(np.float32)

        sample = {
            "embeddings": torch.tensor(emb, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "position": torch.tensor(pos, dtype=torch.float32)
        }
        if not self.inference:
            sample["labels"] = torch.tensor(labels, dtype=torch.float32)

        return sample


def collate_fn(batch):
    """
    Collate function to batch samples from ProteinDataset.
    
    Returns:
        dict: batched embeddings, labels (if present), masks, positions
    """
    B = len(batch)
    L = batch[0]['embeddings'].shape[0]

    # Stack embeddings, masks, positions
    emb = torch.stack([b['embeddings'] for b in batch], dim=0)  # [B,L,D]
    mask = torch.stack([b['mask'] for b in batch], dim=0)       # [B,L]
    pos = torch.stack([b['position'] for b in batch], dim=0).unsqueeze(-1)  # [B,L,1]

    # Stack labels if present
    labels = None
    if 'labels' in batch[0]:
        labels = torch.stack([b['labels'] for b in batch], dim=0)  # [B,L]

    return {
        "embeddings": emb,
        "mask": mask,
        "position": pos,
        "labels": labels
    }


def get_protein_dataloader(df, h5_path, batch_size=32, shuffle=True, max_len=1022, inference=False):
    """
    Helper function to create a DataLoader from ProteinDataset.

    Args:
        df (pd.DataFrame): DataFrame containing dataset keys and labels.
        h5_path (str): Path to HDF5 file with embeddings.
        batch_size (int): Batch size for DataLoader.
        shuffle (bool): Whether to shuffle the data.
        max_len (int): Maximum sequence length.
        inference (bool): If True, DataLoader will ignore labels.

    Returns:
        DataLoader: PyTorch DataLoader ready for training or inference.
    """
    dataset = ProteinDataset(df, h5_path, max_len=max_len, inference=inference)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
