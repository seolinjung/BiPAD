import yaml
from pathlib import Path
import numpy as np

def get_sota_indices(test_indices, sota_results, data):
    sota_indices = []
    for idx in test_indices:
        row = data.iloc[idx]
        path, subcategory = row['path'], row['subcategory']
        matches = sota_results[sota_results['filename'] == path]
        sota_idx = matches[matches['type'] == subcategory].index.tolist()
        sota_indices.extend(sota_idx)
    return sota_indices

def get_region_indices_test(root_path, data):
    config_path = Path(root_path / "test_indices.yaml")
    if not config_path.is_file():
         return update_region_indices_test(root_path, data)
    with open(config_path, "r") as f:
        test_indices = yaml.safe_load(f)
    if not test_indices:
         return update_region_indices_test(root_path, data)
    test_indices_de = test_indices.get('de', {})
    test_indices_us = test_indices.get('us', {})
    if not test_indices_de or not test_indices_us:
        return update_region_indices_test(root_path, data)
    print(len(test_indices_de) + len(test_indices_us))
    return test_indices_de, test_indices_us

def update_region_indices_test(root_path, data):
    test_indices_de = data[data['region'] == 'DE'].sample(n=50, random_state=42).index
    test_indices_us = data[data['region'] == 'US'].sample(n=50, random_state=42).index
    test_indices = {
        'de': test_indices_de.to_list(),
        'us': test_indices_us.to_list()
    }
    with open(root_path / "test_indices.yaml", "w") as f:
            yaml.dump(test_indices, f, default_flow_style=False, sort_keys=False)
    return test_indices_de, test_indices_us

def get_region_indices_train(data, test_indices_de, test_indices_us):
    indices_de = data[data['region'] == 'DE'].index.to_numpy()
    indices_us = data[data['region'] == 'US'].index.to_numpy()
    train_indices_de = indices_de[~np.isin(indices_de, test_indices_de)]
    train_indices_us = indices_us[~np.isin(indices_us, test_indices_us)]
    return train_indices_de.tolist(), train_indices_us.tolist()