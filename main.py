import os
import yaml
from pathlib import Path
import torch
import json
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from processing import process_data
from processing.dataset import AdsDataset

def main():
    root = Path.cwd()
    full_dataset = AdsDataset(root, root / "data" / "final_ground_truth.csv")
    train_dataset, test_dataset = random_split(full_dataset, [len(full_dataset) - 100, 100])
    print(test_dataset.indices)
    # test_dataset.indices
    # process_data.sync_data(Path.cwd() / "data")

if __name__ == "__main__":
    main()