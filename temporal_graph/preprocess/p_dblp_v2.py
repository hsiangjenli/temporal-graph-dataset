import re
import numpy as np
import pandas as pd

from temporal_graph.base import Preprocessing


class DBLPPreprocessing(Preprocessing):
    def __init__(self, input, output, dataset_name) -> None:
        super().__init__(input=input, output=output, dataset_name=dataset_name)
    
    @staticmethod
    def author_to_list_dir(row):
        if isinstance(row, list):
            return [[row[i], row[i+1]] for i in range(len(row)-1)]
        else:
            return row
    
    @staticmethod
    def clean_venue(venue):
        venue = venue.lower()
        venue = re.sub(r'[^a-z0-9]', '', venue)
        return venue
    
    @staticmethod
    def padding_msg(msg):
        msg = msg.lower()
        if len(msg) < 20:
            msg += "!" * (20 - len(msg))
        elif len(msg) >= 20:
            msg = msg[:20]
        return msg
    
    @staticmethod
    def clean_org(row):
        row = row.lower() # lowercase
        row = row.split(",")[0] # only keep the first part
        row = re.sub(r'[^a-zA-Z\s]', '', row)
        row = row.replace("department", "")
        row = row.replace("departament", "")
        row = row.replace("dept", "")
        row = row.replace("university", "")
        row = row.replace("institute", "")
        row = row.replace("school", "")
        row = row.replace("college", "")
        row = row.replace("of", "")
        row = row.strip()
        row = row.replace(" ", "")
        row = row[:20]
        return row 
    
    @staticmethod
    def padding_org(org):
        min_len = 20

        if isinstance(org, float):
            return "!" * min_len
        if len(org) < min_len:
            org += "!" * (min_len - len(org))
        elif len(org) > min_len:
            org = org[:min_len]
        
        return org
    
    @staticmethod
    def convert_str2int(in_str: str):
        out = []
        for element in in_str:
            try:
                if element.isnumeric():
                    out.append(int(element))
                elif element == "!":
                    out.append(0)
                else:
                    out.append(ord(element.upper()) - 44 + 9)
            except:
                print(element)
        return out
    
    def processing(self, dataset_name):
        df_edge_feat = pd.read_csv(f"{self._input_folder(dataset_name)}/dblp-citation-network-v14.csv", sep="|")
        df_edge_feat.dropna(inplace=True)
        
        df_edge_feat = df_edge_feat.dropna(subset=['authors'])

        df_edge_feat["authors"] = df_edge_feat["authors"].apply(lambda x: eval(x))
        df_edge_feat["num_authors"] = df_edge_feat["authors"].apply(lambda x: len(x))

        df_edge_feat = df_edge_feat[df_edge_feat["num_authors"] > 1]
        df_edge_feat = df_edge_feat[df_edge_feat["num_authors"] <= 10]

        # -- node feature
        # df_node_feat = pd.DataFrame(list(df_edge_feat["authors"].explode().to_dict().values()))
        # df_node_feat["org"] = df_node_feat["org"].apply(DBLPPreprocessing.clean_org)
        # df_node_feat["org"] = df_node_feat["org"].apply(DBLPPreprocessing.padding_org)
        # df_node_feat["org"] = df_node_feat["org"].apply(DBLPPreprocessing.convert_str2int)
        # df_node_feat.drop_duplicates(subset=["id"], inplace=True)

        # -- pair author
        df_edge_feat["author_to_list_dir"] = df_edge_feat["authors"].apply(DBLPPreprocessing.author_to_list_dir)
        df_edge_feat = df_edge_feat.explode("author_to_list_dir")
        df_edge_feat = df_edge_feat[["author_to_list_dir", "year", "venue", "doc_type"]]
        df_edge_feat["t"] = df_edge_feat["year"].astype(int)
        
        df_edge_feat["venue"] = df_edge_feat["venue"].apply(lambda x: eval(x)) 
        df_edge_feat["venue"] = df_edge_feat["venue"].apply(lambda x: x["raw"])
        df_edge_feat["venue"] = df_edge_feat["venue"].apply(DBLPPreprocessing.clean_venue)
        
        df_edge_feat["msg"] = df_edge_feat["doc_type"].map({ "Journal": "J", "Conference": "C" }) + df_edge_feat["venue"]
        df_edge_feat["msg"] = df_edge_feat["msg"].apply(DBLPPreprocessing.padding_msg)
        df_edge_feat["msg"] = df_edge_feat["msg"].apply(DBLPPreprocessing.convert_str2int)

        df_edge_feat = df_edge_feat.reset_index()

        df_edge_feat = pd.concat(
        [
            pd.DataFrame(
                df_edge_feat["author_to_list_dir"].to_list(), columns=["src", "dst"]),
                df_edge_feat["msg"],
                df_edge_feat["t"]
        ], axis=1
        )

        df_node_feat = pd.concat([df_edge_feat["src"], df_edge_feat["dst"]]).to_frame()

        df_node_feat["id"] = df_node_feat[0].apply(lambda x: x['id'])
        df_node_feat["org"] = df_node_feat[0].apply(lambda x: x['org'])
        df_node_feat["org"] = df_node_feat["org"].apply(DBLPPreprocessing.clean_org)
        df_node_feat["org"] = df_node_feat["org"].apply(DBLPPreprocessing.padding_org)
        df_node_feat["org"] = df_node_feat["org"].apply(DBLPPreprocessing.convert_str2int)
    
        df_edge_feat["src"] = df_edge_feat["src"].apply(lambda x: x["id"])
        df_edge_feat["dst"] = df_edge_feat["dst"].apply(lambda x: x["id"])
        
        num_nodes, id_map, src_idx, dst_idx = self._src_dst_to_idx(df_edge_feat["src"], df_edge_feat["dst"])

        msg = np.array(df_edge_feat['msg'].to_list(), dtype=np.float32)
        msg = msg.astype(float)

        df_node_feat["idx"] = df_node_feat["id"].apply(lambda x: id_map[x])
        df_node_feat = df_node_feat.sort_values(by=["idx"], ascending=True)
        df_node_feat = df_node_feat.drop_duplicates(subset=["idx"], keep="first")
        df_node_feat = df_node_feat.reset_index()

        x = np.array(df_node_feat['org'].to_list(), dtype=np.float32)
        x = x.astype(float)

        data = {
            "t": df_edge_feat["t"].to_numpy(),
            "msg": msg,
            "src": src_idx, 
            "dst": dst_idx
        }

        return data, x

if __name__ == "__main__":
    dblp = DBLPPreprocessing(input="raw", output="data", dataset_name="dblp_v2")
    dblp.save()