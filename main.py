import yaml
import argparse
from pathlib import Path
import json

import yaml
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

from processing.dataset import AdsDataset
from processing.process_data import get_region_based_data_split, get_default_data_split, update_data_split
from processing.clean_data import sync_data
import decision_tree

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", required=False, action="store_true")
    parser.add_argument("--svm", required=False, action="store_true")
    parser.add_argument("--nouveau", required=False, action="store_true")
    return parser.parse_args()

def main(args):
    root = Path.cwd()
    # sync_data(root)
    full_dataset = AdsDataset(root, root / "data" / "final_ground_truth.csv")
    tokens = 
    train_indices, test_indices = get_region_based_data_split(root, len(full_dataset))
    if args.dt:
        decision_tree.train(train_indices, full_dataset)
        decision_tree.evaluate(test_indices, full_dataset)



if __name__ == "__main__":
    args = get_args()
    main(args)