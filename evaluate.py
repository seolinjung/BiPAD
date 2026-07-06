import os
import yaml
from pathlib import Path
import torch
import json
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from processing import process_data

def evaluate_sota(root_path):
    final_gt = pd.read_csv(root_path / "data" / "final_ground_truth.csv")
    sota_results = pd.read_csv(root_path / "prior" / "llm_annotations.csv")
    # filename/ type/ path/ llm_annotation
    original = original.rename(
        columns = {
            "Filename": "path",
            "JSON Filename": "subcategory",
            "Ground Truth": "gt"
        }
    )
    cleaned_sota = []

# def evaluate()