import re
import pandas as pd
import numpy as np
from dateutil import parser
from dateutil.tz import gettz
from temporal_graph.base import Preprocessing

class EEDPreprocessing(Preprocessing):
    def __init__(self, input, output, dataset_name) -> None:
        super().__init__(input=input, output=output, dataset_name=dataset_name)

    @staticmethod
    def split_file(raw):
        return raw.split('/')[0]

    @staticmethod
    def extract_to(raw):
        matches = re.findall(r'([a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,})', raw)
        if matches:
            return [email.strip() for email in matches]
        else:
            return None

    @staticmethod
    def extract_date(raw):
        match = re.search(r'Date: (.+)', raw)
        if match:
            return match.group(1)
        else:
            return None

    @staticmethod
    def to_utc(date):
        date = parser.parse(date)
        date_utc = date.astimezone(gettz('UTC'))
        return date_utc.timestamp()

    @staticmethod
    def extract_to_org(email):
        return email.split('@')[-1]
    
    @staticmethod
    def padding_org(org):
        if len(org) <= 10:
            return f"{'!'* (10 - len(org))}{org}"
        
        elif len(org) > 10:
            return org[:10]
    
    @staticmethod
    def convert_str2int(in_str: str):
        out = []
        for element in in_str:
            if element.isnumeric():
                out.append(element)
            elif element == "!":
                out.append(-1)
            else:
                out.append(ord(element.upper()) - 44 + 9)
        return out
    
    def processing(self, dataset_name):
        df_edge_feat = pd.read_csv(f"{self._input_folder(dataset_name)}/emails.csv")

        df_edge_feat["From"] = df_edge_feat['file'].apply(EEDPreprocessing.split_file)
        df_edge_feat["To"] = df_edge_feat['message'].apply(EEDPreprocessing.extract_to)

        df_edge_feat["t"] = df_edge_feat['message'].apply(EEDPreprocessing.extract_date)
        df_edge_feat["t"] = df_edge_feat["t"].apply(EEDPreprocessing.to_utc)

        # -- Remove self-loop ---------------------------------------------
        df_edge_feat[["From", "To"]].dropna(inplace=True)
        df_edge_feat = df_edge_feat.explode("To")
        duplicate_rows = df_edge_feat[df_edge_feat['From'] == df_edge_feat['To']].index
        df_edge_feat = df_edge_feat.drop(duplicate_rows)

        df_edge_feat.sort_values(by=["t"], inplace=True)

        num_nodes, id_map, src_idx, dst_idx = self._src_dst_to_idx(df_edge_feat["From"], df_edge_feat["To"])
        df_edge_feat["src"] = df_edge_feat["From"].apply(lambda x: id_map[x])
        df_edge_feat["dst"] = df_edge_feat["To"].apply(lambda x: id_map[x])

        # -- node features ------------------------------------------------------------------------------------------------------
        df_node_feat = pd.DataFrame()
        _f = pd.DataFrame(df_edge_feat["From"].unique())
        _t = pd.DataFrame(df_edge_feat["To"].unique())

        df_node_feat["node"] = pd.concat([_f, _t])
        df_node_feat = df_node_feat.drop_duplicates()

        df_node_feat["idx"] = df_node_feat["node"].apply(lambda x: id_map[x])
        
        df_node_feat["x"] = df_node_feat["node"].apply(EEDPreprocessing.extract_to_org)
        df_node_feat["x"] = df_node_feat["x"].apply(EEDPreprocessing.padding_org)
        df_node_feat["x"] = df_node_feat["x"].apply(EEDPreprocessing.convert_str2int)

        df_node_feat = df_node_feat.sort_values(by=["idx"], ascending=True)

        # -- edge features ------------------------------------------------------------------------------------------------------
        df_edge_feat["msg_src"] = pd.merge(df_edge_feat, df_node_feat, left_on="src", right_on="idx", how="left")["x"]
        df_edge_feat["msg_dst"] = pd.merge(df_edge_feat, df_node_feat, left_on="dst", right_on="idx", how="left")["x"]

        df_edge_feat["msg"] = df_edge_feat["msg_src"] + df_edge_feat["msg_dst"]
        msg = np.array(df_edge_feat['msg'].to_list(), dtype=np.float32)
        msg = msg.astype(float)

        data = {
            "t": df_edge_feat["t"].to_numpy(),
            "src": src_idx, 
            "dst": dst_idx,
            "msg": msg
        }

        x = np.array(df_node_feat['x'].to_list(), dtype=np.float32)
        x = x.astype(float)
        return data, x
        

if __name__ == '__main__':
    eed = EEDPreprocessing(input="raw", output="data", dataset_name="eed")
    eed.save()