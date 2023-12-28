from torch_geometric.data import TemporalData
import gdown
from temporal_graph.flight import padding_callsign, padding_typecode, convert_str2int
import pandas as pd
import numpy as np
import tqdm
import os
import toml
import zipfile
import requests
import io

class TemporalGraph:
    def __init__(self, root: str=None):
        self._root = root
        self.available_datasets_dict = toml.load(os.path.join(self.pkg_dir, "available_datasets.toml"))
    
    def __call__(self, dataset_name: str) -> TemporalData:
        self.download(dataset_name)
        data = self._return_data(dataset_name)
        return TemporalData(**data)

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

    def _return_data(self, dataset_name) -> dict:

        if self.available_datasets_dict[dataset_name]["src"] == "zenodo":
            return self._return_data_zenodo(dataset_name)
        
        elif self.available_datasets_dict[dataset_name]["src"] == "tgb":
            return self._return_data_tgb(dataset_name)
        
    def _return_data_zenodo(self, dataset_name) -> dict:

        graph = pd.read_csv(f"{self._data_folder(dataset_name)}/ml_{dataset_name}.csv")

        edge_features = np.load(f"{self._data_folder(dataset_name)}/ml_{dataset_name}.npy")[1:]
        node_features = np.load(f"{self._data_folder(dataset_name)}/ml_{dataset_name}_node.npy")[1:]
        
        if dataset_name.lower() in ['enron', 'socialevolve', 'uci']:
            node_zero_padding = np.zeros((node_features.shape[0], 172 - node_features.shape[1]))
            node_features = np.concatenate([node_features, node_zero_padding], axis=1)
            edge_zero_padding = np.zeros((edge_features.shape[0], 172 - edge_features.shape[1]))
            edge_features = np.concatenate([edge_features, edge_zero_padding], axis=1)
        
        return {
            "src": graph["u"].to_numpy(),
            "dst": graph["i"].to_numpy(),
            "t": graph["ts"].to_numpy(),
            "msg": edge_features,
        }
    
    def _return_data_tgb(self, dataset_name) -> dict:
        df = pd.read_csv(f"{self._data_folder(dataset_name)}/{dataset_name}_edgelist_v2.csv")

        if dataset_name == "tgbl-coin":

            data = {
                "src": df["src"].to_numpy(),
                "dst": df["dst"].to_numpy(),
                "t": df["day"].to_numpy(),
                "msg": np.zeros((df.shape[0], 1)),
            }
        
        elif dataset_name == "tgbl-flight":
            df['callsign'] = df['callsign'].apply(padding_callsign)
            df['typecode'] = df['typecode'].apply(padding_typecode)
            df['msg'] = df['callsign'] + df['typecode']
            df['msg'] = df['msg'].apply(convert_str2int)

            data = {
                "src": df["src"].to_numpy(),
                "dst": df["dst"].to_numpy(),
                "t": df["day"].to_numpy(),
                "msg": df["msg"].to_numpy(),
            }
            
        return data
    
    def download(self, dataset_name: str):
        # -- Create the directory ----------------------------------------------
        os.makedirs(self.root, exist_ok=True)

        # -- Download the zip file with progress bar ---------------------------
        # ---- check if the dataset has already been downloaded ----------------
        if not(os.path.exists(self._data_folder(dataset_name))):
            # with requests.get(self._url(dataset_name), stream=True) as r:

            gdrive_id = self.available_datasets_dict[dataset_name]["href"]

            gdown.download(id=gdrive_id, output=f"{self.root}/{dataset_name}.zip", quiet=False)
            zipfile.ZipFile(f"{self.root}/{dataset_name}.zip").extractall(self.root)
            