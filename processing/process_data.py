import yaml
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, precision_score, recall_score

def get_metrics(gt, predictions):
    return {
        "F1": f1_score(gt, predictions),
        "Precision": precision_score(gt, predictions),
        "Recall": recall_score(gt, predictions)
    }

def print_metrics(results_all, results_de, results_us):
    print_data = {
         "Combined": results_all,
         "DE": results_de,
         "US": results_us
    }
    df = pd.DataFrame.from_dict(print_data, orient="index").round(3)
    print("\n")
    print(df)
    print("\n")

def get_region_indices_test(root_path, data):
    config_path = Path(root_path / "test_indices.yaml")
    if not config_path.is_file():
         print("\ntest_indices.yaml not found.")
         config_path.touch()
         return update_region_indices_test(root_path, data)
    with open(config_path, "r") as f:
        test_indices = yaml.safe_load(f)
    if not test_indices:
         print("\ntest_indices.yaml is invalid or corrupt.")
         return update_region_indices_test(root_path, data)
    test_indices_de = test_indices.get('de', {})
    test_indices_us = test_indices.get('us', {})
    if not test_indices_de or not test_indices_us:
        print("\ntest_indices.yaml is invalid or corrupt.")
        return update_region_indices_test(root_path, data)
    print("\nFound valid test_indices.yaml. Loading test indices...\n")
    return test_indices_de, test_indices_us

def update_region_indices_test(root_path, data):
    print("Updating test indices...")
    test_indices_de = data[data['region'] == 'DE'].sample(n=50, random_state=42).index
    test_indices_us = data[data['region'] == 'US'].sample(n=50, random_state=42).index
    test_indices = {
        'de': test_indices_de.to_list(),
        'us': test_indices_us.to_list()
    }
    with open(root_path / "test_indices.yaml", "w") as f:
            yaml.dump(test_indices, f, default_flow_style=False, sort_keys=False)
    print(f"Saved new test indices to test_indices.yaml.\n")
    return test_indices_de.to_list(), test_indices_us.to_list()

def get_region_indices_train(data, test_indices_de, test_indices_us):
    indices_de = data[data['region'] == 'DE'].index.to_numpy()
    indices_us = data[data['region'] == 'US'].index.to_numpy()
    train_indices_de = indices_de[~np.isin(indices_de, test_indices_de)]
    train_indices_us = indices_us[~np.isin(indices_us, test_indices_us)]
    return train_indices_de.tolist(), train_indices_us.tolist()

def get_data_count(test_indices_de, test_indices_us, train_indices_de, train_indices_us):
    train_len_de, train_len_us = len(train_indices_de), len(train_indices_us)
    train_len_all = train_len_de + train_len_us
    test_len_de, test_len_us = len(test_indices_de), len(test_indices_us)
    test_len_all = test_len_de + test_len_us
    data_count = {
         "Train": [train_len_all, train_len_de, train_len_us],
         "Test": [test_len_all, test_len_de, test_len_us]
    }
    df = pd.DataFrame.from_dict(data_count)
    df.index = ["Combined", "DE", "US"]
    print("\n~Dataset Statistics~")
    print("\n")
    print(df)
    print("\n")