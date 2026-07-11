from torch.utils.data import Dataset
import torch
class AdsDataset(Dataset):
    def __init__(self, root, data, features):
        self.data = data
        self.root = root
        self.features = features
        self.adblock_path = self.root / "data" / "adblock" / "adshots"
        self.control_path = self.root / "data" / "control" / "adshots"
        self.de_indices = self.get_de_indices
        self.us_indices = self.get_us_indices

    def get_de_indices(self):
         return self.data.index[self.data['region'] == 'DE']
    
    def get_us_indices(self):
         return self.data.index[self.data['region'] == 'US']
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        row["path"] = self.root / "data" / row["subcategory"] / "adshots" / row["path"]
        return row
    
class RobertaAdsDataset(AdsDataset):
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        encoded = self.features[idx]
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(row["gt"], dtype=torch.int64)
        }