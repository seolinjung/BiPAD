import os
import yaml
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class AdsDataset(Dataset):
    def __init__(self, root, gt_csv):
        self.data = pd.read_csv(gt_csv)
        self.root = root
        self.adblock_path = self.root / "data" / "adblock" / "adshots"
        self.control_path = self.root / "data" / "control" / "adshots"
        self.data = self.fuse_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx].to_dict()
        row["path"] = self.root / "data" / row["subcategory"] / self.data["path"]
        return row