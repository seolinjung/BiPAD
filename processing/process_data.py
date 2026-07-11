import yaml
from pathlib import Path
import numpy as np

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

# def get_region_based_indices_all(root_path, de_indices, us_indices):
#         with open(root_path / "config.yaml", "r") as f:
#         test_indices = yaml.safe_load(f)
#     de_indices = test_indices['de']
#     us_indices = test_indices['us']
#     merged_test_indices = de_indices + us_indices
#     complete_indices = np.arange(dataset_length)
#     train_indices = complete_indices[~np.isin(complete_indices, merged_test_indices)]
#     return train_indices, merged_test_indices, de_indices, us_indices

# def get_region_based_data_split(root_path, dataset_length):
#     with open(root_path / "config.yaml", "r") as f:
#         test_indices = yaml.safe_load(f)
#     de_indices = test_indices['de']
#     us_indices = test_indices['us']
#     merged_test_indices = de_indices + us_indices
#     complete_indices = np.arange(dataset_length)
#     train_indices = complete_indices[~np.isin(complete_indices, merged_test_indices)]
#     return train_indices, merged_test_indices, de_indices, us_indices

# def get_test_indices(root_path):
#     with open(root_path / "config.yaml", "r") as f:
#         all_indices = yaml.safe_load(f)
#     de_indices = all_indices['de']
#     us_indices = all_indices['us']
#     return de_indices, us_indices

# def get_default_data_split(root_path, dataset_length):
#     with open(root_path / "config.yaml", "r") as f:
#         test_indices = yaml.safe_load(f)["test"]
#     complete_indices = np.arange(dataset_length)
#     train_indices = complete_indices[~np.isin(complete_indices, test_indices)]
#     return train_indices, test_indices

# def update_data_split(root_path, full_dataset: Dataset):
#     train_dataset, test_dataset = random_split(full_dataset, [len(full_dataset) - 100, 100])
#     test_indices = { "test": test_dataset.indices }
#     with open(root_path / "config.yaml", "w") as f:
#         yaml.dump(test_indices, f, default_flow_style=False, sort_keys=False)
#     return train_dataset.indices, test_indices

# def get_ground_truth(root_path, dataset_length, test_indices):
#     train_indices, test_indices = get_default_data_split(root_path, dataset_length)
#     # train_labels = 
