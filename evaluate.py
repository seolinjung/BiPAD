import os
import yaml
from pathlib import Path
import torch
import json
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from processing.process_data import get_sota_indices
from sklearn.metrics import f1_score, precision_score, recall_score

def evaluate_sota_de(test_indices_de, sota_results, data):
    gt_de = data.loc[test_indices_de, 'gt']
    sota_indices_de = get_sota_indices(test_indices_de, sota_results, data)
    sota_predictions = sota_results.loc[sota_indices_de, 'llm_annotation'].notna().astype(int)
    results = [
        f1_score(gt_de, sota_predictions),
        precision_score(gt_de, sota_predictions),
        recall_score(gt_de, sota_predictions)
    ]
    return results

def evaluate_sota_us(test_indices_us, sota_results, data):
    gt_us = data.loc[test_indices_us, 'gt']
    sota_indices_us = get_sota_indices(test_indices_us, sota_results, data)
    sota_predictions = sota_results.loc[sota_indices_us, 'llm_annotation'].notna().astype(int)
    results = [
        f1_score(gt_us, sota_predictions),
        precision_score(gt_us, sota_predictions),
        recall_score(gt_us, sota_predictions)
    ]
    return results

def evaluate_all(test_indices, sota_results, data):
    gt = data.loc[test_indices, 'gt']
    sota_indices = get_sota_indices(test_indices, sota_results, data)
    sota_predictions = sota_results.loc[sota_indices, 'llm_annotation'].notna().astype(int)
    results = [
        f1_score(gt, sota_predictions),
        precision_score(gt, sota_predictions),
        recall_score(gt, sota_predictions)
    ]
    return results

# def evaluate()