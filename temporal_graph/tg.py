# -- for dataset --------------------------------------------
from torch_geometric.data import TemporalData
import torch

# -- for preprocessing --------------------------------------
import pandas as pd
import numpy as np
import tqdm
import os
import toml
import zipfile

import temporal_graph.p_flight as p_flight

class TemporalGraph:
    def __init__(self, root: str=None):
        self._root = root
        self.available_datasets_dict = toml.load(os.path.join(self.pkg_dir, "available_datasets.toml"))
    
    def __call__(self, dataset_name: str) -> TemporalData:

        # ---- check if the dataset has already been downloaded ----------------
        if not(os.path.exists(self._data_folder(dataset_name))):
            self.download(dataset_name)

        # ---- check if the dataset has already been preprocessed --------------
        # -------- if data_folder contains the file ending with .pt, then the dataset has already been preprocessed ------
        if not(os.path.exists(f"{self._data_folder(dataset_name)}/pyg_{dataset_name}.pt")): 
            data = self._return_data(dataset_name)
            tg = TemporalData(**data)
            torch.save(tg, f"{self._data_folder(dataset_name)}/pyg_{dataset_name}.pt")

        return torch.load(f"{self._data_folder(dataset_name)}/pyg_{dataset_name}.pt")

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

    def _preprocessed(self, dataset_name) -> dict:

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
    
    def _src_dst_to_idx(self, src, dst):
        src_dst = np.concatenate([src, dst])
        src_dst_unique = np.unique(src_dst)
        src_dst_unique_dict = {k: v for v, k in enumerate(src_dst_unique)}
        src_idx = np.array([src_dst_unique_dict[k] for k in src])
        dst_idx = np.array([src_dst_unique_dict[k] for k in dst])
        return src_idx, dst_idx

    def _return_data_tgb(self, dataset_name) -> dict:
        df = pd.read_csv(f"{self._data_folder(dataset_name)}/{dataset_name}_edgelist_v2.csv")

        if dataset_name == "tgbl-coin":

            data = {
                "t": df["day"].to_numpy(),
                "msg": np.zeros((df.shape[0], 1)),
                "src_x": np.ones((df.shape[0], 1)),
                "dst_x": np.ones((df.shape[0], 1)),
            }
        
        elif dataset_name == "tgbl-flight":
            
            # -- node features --------------------------------------------------
            df_node_feat = pd.read_csv(f"{self._data_folder(dataset_name)}/airport_node_feat_v2.csv")
            df_node_feat['iso_region'] = df_node_feat['iso_region'].apply(p_flight.padding_iso_region)
            df_node_feat['iso_region'] = df_node_feat['iso_region'].apply(p_flight.convert_str2int)
            df_node_feat["continent"] = df_node_feat["continent"].apply(p_flight.convert_continent)
            df_node_feat['type'] = df_node_feat['type'].apply(p_flight.convert_str2int)
            df_node_feat["longitude"] = df_node_feat["longitude"].apply(lambda x: [x])
            df_node_feat["latitude"] = df_node_feat["latitude"].apply(lambda x: [x])

            df_node_feat['combined'] = df_node_feat['iso_region'] + df_node_feat['continent'] + df_node_feat['type'] + df_node_feat['longitude'] + df_node_feat['latitude']
            df_node_feat['combined'] = df_node_feat['combined'].apply(lambda x: np.array(x, dtype=np.float32))

            df_src = pd.merge(df["src"], df_node_feat, left_on='src', right_on='airport_code')
            df_dst = pd.merge(df["dst"], df_node_feat, left_on='dst', right_on='airport_code')

            # -- edge data ------------------------------------------------------
            df['callsign'] = df['callsign'].apply(p_flight.padding_callsign)
            df['typecode'] = df['typecode'].apply(p_flight.padding_typecode)
            df['msg'] = df['callsign'] + df['typecode']
            df['msg'] = df['msg'].apply(p_flight.convert_str2int)

            data = {
                "t": df["timestamp"].to_numpy(),
                "msg": df["msg"].to_numpy(),
                "src_x": df_src["combined"].to_numpy(),
                "dst_x": df_dst["combined"].to_numpy(),
            }
        
        src_idx, dst_idx = self._src_dst_to_idx(df["src"], df["dst"])
        
        data.update({"src": src_idx, "dst": dst_idx})
            
        return data
    
    def download(self, dataset_name: str):
        # -- Create the directory ----------------------------------------------
        os.makedirs(self.root, exist_ok=True)

        gdrive_id = self.available_datasets_dict[dataset_name]["href"]

        gdown.download(id=gdrive_id, output=f"{self.root}/{dataset_name}.zip", quiet=False)
        zipfile.ZipFile(f"{self.root}/{dataset_name}.zip").extractall(self.root)
        os.remove(f"{self.root}/{dataset_name}.zip")