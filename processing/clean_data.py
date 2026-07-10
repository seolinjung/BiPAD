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

def sync_data(root_path):

    def clean_format(value):
        first = ast.literal_eval(value)
        return next(iter(first))

    original = pd.read_csv(root_path / "data" / "merged_ground_truth.csv")
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
    with open(root_path / "data" / "ocr_adblock.json", "r") as f:
        ocr_adblock = json.load(f)
    with open(root_path / "data" / "ocr_control.json", "r") as f:
        ocr_control = json.load(f)
    final = []
    for idx, img in original["path"].items():
        print(idx)
        subcategory_arr = original.loc[idx, "subcategory"].split("_")
        subcategory = subcategory_arr[0]
        region = "DE" if subcategory_arr[1] == "germany" else "US"
        final.append(
            {
                "path": img,
                "subcategory": subcategory,
                "region": region,
                "gt": 0 if pd.isna(original.loc[idx, "gt"]) else 1,
                "ocr": ocr_adblock[img] if subcategory == "adblock" else ocr_control[img]
            }
        )
    final_df = pd.DataFrame(final)
    final_df['ocr'] = final_df['ocr'].fillna('')
    print(final_df)
    print(final_df['gt'].value_counts())
    print(final_df['subcategory'].value_counts())
    final_df.to_csv(root_path / "data" / "cleaned_data.csv", index=False)

def fuse_data(root_path):
    adblock_path = root_path / "data" / "adblock" / "adshots"
    control_path = root_path / "data" / "control" / "adshots"
    adblock_data = [i for i in adblock_path.iterdir()]
    control_data = [i for i in control_path.iterdir()]
    data = []
    data.extend(adblock_data)
    data.extend(control_data)
    return data