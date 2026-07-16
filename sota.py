import pandas as pd

from processing.process_data import get_metrics, print_metrics

def get_sota_indices(test_indices, sota_results, data):
    sota_indices = []
    for idx in test_indices:
        row = data.iloc[idx]
        path, subcategory = row['path'], row['subcategory']
        matches = sota_results[sota_results['filename'] == path]
        sota_idx = matches[matches['type'] == subcategory].index.tolist()
        sota_indices.extend(sota_idx)
    return sota_indices

def evaluate(root, test_indices, data):
    sota_results = pd.read_csv(root / "prior" / "llm_annotation.csv")
    gt = data.loc[test_indices, 'gt']
    sota_indices = get_sota_indices(test_indices, sota_results, data)
    sota_predictions = sota_results.loc[sota_indices, 'llm_annotation'].notna().astype(int)
    return get_metrics(gt, sota_predictions)

def run(root, test_indices, test_indices_de, test_indices_us, data):
    results_all = evaluate(root, test_indices, data)
    results_de = evaluate(root, test_indices_de, data)
    results_us = evaluate(root, test_indices_us, data)
    print("\n~SOTA Results~")
    print_metrics(results_all, results_de, results_us)
