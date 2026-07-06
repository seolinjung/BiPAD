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

def sync_data(root_path):

    def clean_format(value):
        first = ast.literal_eval(value)
        return next(iter(first))

    original = pd.read_csv(root_path / "merged_ground_truth.csv")
    original = original.rename(
        columns = {
            "Filename": "path",
            "JSON Filename": "subcategory",
            "Ground Truth": "gt"
        }
    )
    original["path"] = original["path"].apply(clean_format)
    original["subcategory"] = original["subcategory"].apply(clean_format)
    original = original.sort_values("subcategory").reset_index(drop=True)
    original = original.drop_duplicates(subset="path", keep="first")
    with open(root_path / "ocr_adblock.json", "r") as f:
        ocr_adblock = json.load(f)
    with open(root_path / "ocr_control.json", "r") as f:
        ocr_control = json.load(f)
    final = []
    for idx, img in original["path"].items():
        print(idx)
        subcategory = original.loc[idx, "subcategory"].split("_")[0]
        print(subcategory)
        final.append(
            {
                "path": img,
                "subcategory": subcategory,
                "gt": 0 if pd.isna(original.loc[idx, "gt"]) else 1,
                "ocr": ocr_adblock[img] if subcategory == "adblock" else ocr_control[img]
            }
        )
    final_df = pd.DataFrame(final)
    print(final_df)
    print(final_df['gt'].value_counts())
    print(final_df['subcategory'].value_counts())
    final_df.to_csv(root_path / "final_ground_truth.csv", index=False)

def fuse_data(root_path):
    adblock_path = root_path / "data" / "adblock" / "adshots"
    control_path = root_path / "data" / "control" / "adshots"
    adblock_data = [i for i in adblock_path.iterdir()]
    control_data = [i for i in control_path.iterdir()]
    data = []
    data.extend(adblock_data)
    data.extend(control_data)
    return data