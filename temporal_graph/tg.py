# -- for dataset --------------------------------------------
from torch_geometric.data import TemporalData
import torch
import gdown

# -- for preprocessing --------------------------------------
import pandas as pd
import numpy as np
import tqdm
import os
import toml
import zipfile
import pickle

import temporal_graph.p_flight as p_flight
import temporal_graph.utils as utils

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
            data, x = self._preprocessed(dataset_name)
            data = TemporalData(**data)
            # x = torch.from_numpy(x).to(torch.float32)
            torch.save(data, f"{self._data_folder(dataset_name)}/pyg_{dataset_name}.pt", pickle_protocol=4)
            torch.save(x, f"{self._data_folder(dataset_name)}/pyg_{dataset_name}_node_feat.pt", pickle_protocol=4)
        else:
            data = torch.load(f"{self._data_folder(dataset_name)}/pyg_{dataset_name}.pt")
            x = torch.load(f"{self._data_folder(dataset_name)}/pyg_{dataset_name}_node_feat.pt")

        # ---- train/val/test split -----------------------------------------
        if not(os.path.exists(f"{self._data_folder(dataset_name)}/mask_train.pkl")):
            train_mask, val_mask, test_mask = utils.generate_splits(data)
            pickle.dump(train_mask, open(f"{self._data_folder(dataset_name)}/mask_train.pkl", "wb"))
            pickle.dump(val_mask, open(f"{self._data_folder(dataset_name)}/mask_val.pkl", "wb"))
            pickle.dump(test_mask, open(f"{self._data_folder(dataset_name)}/mask_test.pkl", "wb"))
        
        else:
            train_mask = pickle.load(open(f"{self._data_folder(dataset_name)}/mask_train.pkl", "rb"))
            val_mask = pickle.load(open(f"{self._data_folder(dataset_name)}/mask_val.pkl", "rb"))
            test_mask = pickle.load(open(f"{self._data_folder(dataset_name)}/mask_test.pkl", "rb"))

        # ---- convert numpy array to torch tensor ------------------------------
        data.src = torch.from_numpy(data.src).to(torch.long)
        data.dst = torch.from_numpy(data.dst).to(torch.long)
        data.t = torch.from_numpy(data.t).to(torch.long)

        # data.src_x = torch.from_numpy(data.src_x).to(torch.float32)
        # data.dst_x = torch.from_numpy(data.dst_x).to(torch.float32)

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

    def _preprocessed(self, dataset_name) -> dict:
        if self.available_datasets_dict[dataset_name]["src"] == "zenodo":
            return self._return_data_zenodo(dataset_name)
        
        elif self.available_datasets_dict[dataset_name]["src"] == "tgb":
            return self._return_data_tgb(dataset_name)
    
    def _src_dst_to_idx(self, src, dst):

        nodes = set(src.unique().tolist() + dst.unique().tolist())
        num_nodes = len(nodes)
        id_map = {}
        for i, node in enumerate(nodes):
            id_map[node] = i
        
        src_idx = src.apply(lambda x: id_map[x]).to_numpy()
        dst_idx = dst.apply(lambda x: id_map[x]).to_numpy()

        return num_nodes, id_map, src_idx, dst_idx 
    
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
        num_nodes, id_map, src_idx, dst_idx = self._src_dst_to_idx(df["src"], df["dst"])
        
        if dataset_name == "tgbl-coin":
            data = {
                "t": df["day"].to_numpy(),
                "msg": np.zeros((df.shape[0], 1)),
            }

            x = np.ones((num_nodes, 1))
        
        elif dataset_name == "tgbl-flight":
            
            # -- node features --------------------------------------------------
            df_node_feat = pd.read_csv(f"{self._data_folder(dataset_name)}/airport_node_feat_v2.csv")
            df_node_feat = df_node_feat[df_node_feat['airport_code'].isin(id_map.keys())]

            df_node_feat['iso_region'] = df_node_feat['iso_region'].apply(p_flight.padding_iso_region)
            df_node_feat['iso_region'] = df_node_feat['iso_region'].apply(p_flight.convert_str2int)
            df_node_feat["continent"] = df_node_feat["continent"].apply(p_flight.convert_continent)
            df_node_feat['type'] = df_node_feat['type'].apply(p_flight.convert_type)

            df_node_feat["type"] = df_node_feat["type"].apply(lambda x: [x])
            df_node_feat["longitude"] = df_node_feat["longitude"].apply(lambda x: [x])
            df_node_feat["latitude"] = df_node_feat["latitude"].apply(lambda x: [x])

            df_node_feat['combined'] = df_node_feat['iso_region'] + df_node_feat['continent'] + df_node_feat['type'] + df_node_feat['longitude'] + df_node_feat['latitude']
            # df_node_feat['combined'] = df_node_feat['type']# + df_node_feat['longitude'] + df_node_feat['latitude']
            # df_node_feat['combined'] = df_node_feat['combined'].apply(lambda x: np.array(x, dtype=np.float32))

            df_node_feat['idx'] = df_node_feat['airport_code'].apply(lambda x: id_map[x])
            df_node_feat = df_node_feat.set_index('idx')
            df_node_feat = df_node_feat.sort_index()

            x = df_node_feat['combined'].to_numpy()

            # df_src = pd.merge(df["src"], df_node_feat, left_on='src', right_on='airport_code', how='left')
            # df_dst = pd.merge(df["dst"], df_node_feat, left_on='dst', right_on='airport_code', how='left')

            # -- edge data ------------------------------------------------------
            df['callsign'] = df['callsign'].apply(p_flight.padding_callsign)
            df['typecode'] = df['typecode'].apply(p_flight.padding_typecode)
            df['msg'] = df['callsign'] + df['typecode']
            df['msg'] = df['msg'].apply(p_flight.convert_str2int)

            msg = np.array(df['msg'].to_list(), dtype=np.float32)
            msg = msg.astype(float)
            
            # src_x = np.array(df['combined'].to_list(), dtype=np.float32).reshape(-1, 1)
            # src_x = src_x.astype(float)

            # dst_x = np.array(df_dst['combined'].to_list(), dtype=np.float32).reshape(-1, 1)
            # dst_x = dst_x.astype(float)

            # print(src_x)
            # print(src_x.shape)

            data = {
                "t": df["timestamp"].to_numpy(),
                "msg": msg,
            }
                
        data.update({"src": src_idx, "dst": dst_idx})
        
            
        return data, x
    
    def download(self, dataset_name: str):
        # -- Create the directory ----------------------------------------------
        os.makedirs(self.root, exist_ok=True)

        gdrive_id = self.available_datasets_dict[dataset_name]["href"]

        gdown.download(id=gdrive_id, output=f"{self.root}/{dataset_name}.zip", quiet=False)
        zipfile.ZipFile(f"{self.root}/{dataset_name}.zip").extractall(f"{self.root}/{dataset_name}")
        os.remove(f"{self.root}/{dataset_name}.zip")