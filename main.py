import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from processing.process_data import get_region_indices_test, get_region_indices_train, get_data_count

import decision_tree
import support_vector_machine
import bipad
import sota
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dt",
        action='store_true',
        required=False,
        help="run decision tree for both train and eval"
    )
    parser.add_argument(
        "--svm",
        action='store_true',
        required=False,
        help="run svm for both train and eval"
    )
    parser.add_argument(
        "--sota",
        action='store_true',
        required=False,
        help="run eval for sota"
    )
    parser.add_argument(
        "--bipad",
        type=str,
        nargs="+",
        choices=["train", "evaluate"],
        required=False,
        help="must specify train, evaluate, or both"
    )    
    return parser.parse_args()

def main(args):
    root = Path.cwd()
    data = pd.read_csv(root / "data" / "cleaned_data.csv")
    data["ocr"] = data["ocr"].replace(np.nan, "")

    test_indices_de, test_indices_us = get_region_indices_test(root, data)
    train_indices_de, train_indices_us = get_region_indices_train(data, test_indices_de, test_indices_us)
    test_indices = test_indices_de + test_indices_us
    train_indices = train_indices_de + train_indices_us

    get_data_count(test_indices_de, test_indices_us, train_indices_de, train_indices_us)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.sota:
        sota.run(root, test_indices, test_indices_de, test_indices_us, data)
    if args.dt:
        decision_tree.run(root, data, train_indices, test_indices_de, test_indices_us)
    if args.svm:
        support_vector_machine.run(root, data, train_indices, test_indices_de, test_indices_us)
    if args.bipad:
        for param in args.bipad:
            bipad.run(root, data, train_indices, test_indices_de, test_indices_us, param, device)

if __name__ == "__main__":
    args = get_args()
    main(args)