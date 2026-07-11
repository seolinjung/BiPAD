import os
import yaml
from pathlib import Path
import torch
import json
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from processing.process_data import get_sota_indices_de, get_sota_indices_us
from sklearn.metrics import accuracy_score

def evaluate_sota_de(test_indices_de, sota_results, data):
    gt_de = data.loc[test_indices_de, 'gt']
    sota_indices_de = get_sota_indices_de(test_indices_de, sota_results, data)
    sota_predictions = sota_results.loc[sota_indices_de, 'llm_annotation'].notna().astype(int)
    return accuracy_score(gt_de, sota_predictions)

def evaluate_sota_us(test_indices_us, sota_results, data):
    gt_us = data.loc[test_indices_us, 'gt']
    sota_indices_us = get_sota_indices_us(test_indices_us, sota_results, data)
    sota_predictions = sota_results.loc[sota_indices_us, 'llm_annotation'].notna().astype(int)
    return accuracy_score(gt_us, sota_predictions)

# def evaluate()