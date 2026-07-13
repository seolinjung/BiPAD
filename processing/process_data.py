import yaml
from pathlib import Path
import numpy as np

def get_region_indices_test(root_path, data):
    config_path = Path(root_path / "test_indices.yaml")
    if not config_path.is_file():
         print("\ntest_indices.yaml not found.")
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
    print("\nFound valid test_indices.yaml. Loading test indices...")
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
    print(f"\nSaved new test indices to {root_path}/test_indices.yaml.")
    return test_indices_de, test_indices_us

def get_region_indices_train(data, test_indices_de, test_indices_us):
    indices_de = data[data['region'] == 'DE'].index.to_numpy()
    indices_us = data[data['region'] == 'US'].index.to_numpy()
    train_indices_de = indices_de[~np.isin(indices_de, test_indices_de)]
    train_indices_us = indices_us[~np.isin(indices_us, test_indices_us)]
    return train_indices_de.tolist(), train_indices_us.tolist()