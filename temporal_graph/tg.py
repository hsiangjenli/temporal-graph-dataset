from torch_geometric.data import TemporalData
import torch
import gdown
import numpy as np
import os
import toml
import zipfile
import pickle

class TemporalGraphDataset:
    def __init__(self, root: str=None):
        self._root = root
        self.available_datasets_dict = toml.load(os.path.join(self.pkg_dir, "available_datasets.toml"))
    
    def __call__(self, dataset_name: str) -> TemporalData:

        # ---- check if the dataset has already been downloaded ----------------
        if not(os.path.exists(self._data_folder(dataset_name))):
            self.download(dataset_name)

        data = torch.load(f"{self._data_folder(dataset_name)}/pyg_{dataset_name}.pt")
        x = torch.load(f"{self._data_folder(dataset_name)}/pyg_{dataset_name}_node_feat.pt")

        train_mask = pickle.load(open(f"{self._data_folder(dataset_name)}/mask_train.pkl", "rb"))
        val_mask = pickle.load(open(f"{self._data_folder(dataset_name)}/mask_val.pkl", "rb"))
        test_mask = pickle.load(open(f"{self._data_folder(dataset_name)}/mask_test.pkl", "rb"))

        # ---- convert numpy array to torch tensor ------------------------------
        data.src = torch.from_numpy(data.src).to(torch.long)
        data.dst = torch.from_numpy(data.dst).to(torch.long)
        data.t = torch.from_numpy(data.t).to(torch.long)

        data.msg = torch.from_numpy(data.msg).to(torch.float32)

        return data, x, train_mask, val_mask, test_mask

    @property
    def pkg_dir(self) -> str:
        return os.path.dirname(os.path.abspath(__file__))

    @property
    def available_datasets(self) -> list:
        return [self.available_datasets_dict[k]["name"] for k in self.available_datasets_dict]

    @property
    def root(self) -> str:
        if self._root is None:
            return os.path.join(self.pkg_dir, "data")
        else:
            return self._root
    
    def _url(self, dataset_name) -> str:
        return self.available_datasets_dict[dataset_name]["href"]
    
    def _data_folder(self, dataset_name) -> str:
        return os.path.join(self.root, dataset_name)

    def download(self, dataset_name: str):
        # -- Create the directory ----------------------------------------------
        os.makedirs(self.root, exist_ok=True)

        gdrive_id = self.available_datasets_dict[dataset_name]["href"]

        print(f"Downloading {dataset_name} from Google Drive...")
        gdown.download(id=gdrive_id, output=f"{self.root}/{dataset_name}.zip", quiet=False)
        zipfile.ZipFile(f"{self.root}/{dataset_name}.zip").extractall(f"{self.root}/{dataset_name}")
        os.remove(f"{self.root}/{dataset_name}.zip")