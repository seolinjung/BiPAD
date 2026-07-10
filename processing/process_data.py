import os
import yaml
from pathlib import Path
import torch
import ast
import json
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, Subset

from .dataset import AdsDataset

def update_region_based_data_split(root_path):
    data = pd.read_csv(root_path / "data" / "cleaned_data.csv")
    # de_indices = data[data['region'] == 'DE'].index
    # us_indices = data[data['region'] == 'US'].index
    de_test_indices = data[data['region'] == 'DE'].sample(n=50, random_state=42).index
    us_test_indices = data[data['region'] == 'US'].sample(n=50, random_state=42).index
    test_indices = {
        'de': de_test_indices.to_list(),
        'us': us_test_indices.to_list()
    }
    with open(root_path / "config.yaml", "w") as f:
            yaml.dump(test_indices, f, default_flow_style=False, sort_keys=False)

def get_region_based_data_split(root_path, dataset_length):
    with open(root_path / "config.yaml", "r") as f:
        indices = yaml.safe_load(f)
    merged = indices['de'] + indices['us']
    complete_indices = np.arange(dataset_length)
    train_indices = complete_indices[~np.isin(complete_indices, merged)]
    return train_indices, merged

def get_default_data_split(root_path, dataset_length):
    with open(root_path / "config.yaml", "r") as f:
        test_indices = yaml.safe_load(f)["test"]
    complete_indices = np.arange(dataset_length)
    train_indices = complete_indices[~np.isin(complete_indices, test_indices)]
    return train_indices, test_indices

def update_data_split(root_path, full_dataset: Dataset):
    train_dataset, test_dataset = random_split(full_dataset, [len(full_dataset) - 100, 100])
    test_indices = { "test": test_dataset.indices }
    with open(root_path / "config.yaml", "w") as f:
        yaml.dump(test_indices, f, default_flow_style=False, sort_keys=False)
    return train_dataset.indices, test_indices

def get_ground_truth(root_path, dataset_length, test_indices):
    train_indices, test_indices = get_default_data_split(root_path, dataset_length)
    # train_labels = 
