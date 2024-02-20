import re
import pandas as pd
import numpy as np
from datetime import datetime
from temporal_graph.base import Preprocessing

class EEDPreprocessing(Preprocessing):
    def __init__(self, input, output, dataset_name) -> None:
        super().__init__(input=input, output=output, dataset_name=dataset_name)
    
    @staticmethod
    def missing_to_nan(mail_address):
        if mail_address == "missing":
            return np.nan
        elif not(re.findall(r'[\w\.-]+@[\w\.-]+', mail_address)):
            return np.nan
        else:
            return mail_address

    @staticmethod
    def string_to_timestamp(date_string):
        return datetime.strptime(date_string, "%Y-%m-%d").timestamp()
    
    @staticmethod
    def re_mail_address(mail_address):
        return re.findall(r'[\w\.-]+@[\w\.-]+', mail_address)
    
    @staticmethod
    def re_remove_mail_address_in_body(text):
        return re.sub(r'[\w\.-]+@[\w\.-]+', "", text)
    
    @staticmethod
    def node_feat(mail_address):
        mail_address = EEDPreprocessing.padding_mail_address(mail_address)
        return EEDPreprocessing.convert_str2int(mail_address)

    @staticmethod
    def padding_mail_address(mail_address):
        mail_address = mail_address.split("@")[1].replace(".", "")
        min_len = 10

        if len(mail_address) < min_len:
            mail_address += "!" * (min_len - len(mail_address))
        
        elif len(mail_address) > min_len:
            mail_address = mail_address[:min_len]
        
        return mail_address
    
    @staticmethod
    def convert_str2int(in_str: str):
        out = []
        for element in in_str:
            if element.isnumeric():
                out.append(int(element))
            elif element == "!":
                out.append(-1)
            else:
                out.append(ord(element.upper()) - 44 + 9)
        return out
    
    @staticmethod
    def body_to_edge_feat(text):
        special_kw = ["original message from", "forwarded by"]
        special_kw_ohc = np.eye(len(special_kw))
        text = text.lower()
        
        text = re.sub(r'\s+', ' ', text) # remove multiple spaces
        text = f"{text} {' !' * 10}" # padding text to 10 words
        
        # check if text contains special_kw and return one hot encoded
        for i, kw in enumerate(special_kw):
            if kw in text and isinstance(text, str):
                text = text.replace(kw, "").strip()
                text = text.split(" ")[:10-len(special_kw)]
                text = [sum(EEDPreprocessing.convert_str2int(w)) for w in text]
                return special_kw_ohc[i].tolist() + text
            
            elif isinstance(text, str):
                text = text.strip().split(" ")[:10]
                text = [sum(EEDPreprocessing.convert_str2int(w)) for w in text]
                return text
    
    def processing(self, dataset_name):
        df_edge_feat = pd.read_csv(f"{self._input_folder(dataset_name)}/clean_emails.csv")

        df_edge_feat["sender"] = df_edge_feat["sender"].apply(EEDPreprocessing.missing_to_nan)
        df_edge_feat["recipient"] = df_edge_feat["recipient"].apply(EEDPreprocessing.missing_to_nan)

        df_edge_feat.dropna(inplace=True)
        df_edge_feat["t"] = df_edge_feat["date"].apply(EEDPreprocessing.string_to_timestamp)

        # combine recipient and body
        df_edge_feat["content"] = df_edge_feat["recipient"] + df_edge_feat["body"]
        df_edge_feat["recipient_2"] = df_edge_feat["recipient"].apply(EEDPreprocessing.re_mail_address)
        df_edge_feat["body_2"] = df_edge_feat["body"].apply(EEDPreprocessing.re_remove_mail_address_in_body)

        df_edge_feat = df_edge_feat.explode("recipient_2")
        # df_edge_feat.sort_values(by=["date"], inplace=True)
        num_nodes, id_map, src_idx, dst_idx = self._src_dst_to_idx(df_edge_feat["sender"], df_edge_feat["recipient_2"])
        df_edge_feat["src"] = df_edge_feat["sender"].apply(lambda x: id_map[x])
        df_edge_feat["dst"] = df_edge_feat["recipient_2"].apply(lambda x: id_map[x])

        # -- node features ------------------------------------------------------------------------------------------------------
        
        _src_node_feat = df_edge_feat["sender"].drop_duplicates().to_frame()
        _src_node_feat = _src_node_feat.rename(columns={"sender": "node"})

        _dst_node_feat = df_edge_feat["recipient_2"].drop_duplicates().to_frame()
        _dst_node_feat = _dst_node_feat.rename(columns={"recipient_2": "node"})

        df_node_feat = pd.concat([_src_node_feat, _dst_node_feat], axis=0)
        df_node_feat = df_node_feat.drop_duplicates()
        
        df_node_feat["x"] = df_node_feat["node"].apply(EEDPreprocessing.node_feat)
        # df_node_feat.to_csv(f"{self._output_folder(dataset_name)}/___node_feat.csv", index=False)
        
        df_node_feat["idx"] = df_node_feat["node"].apply(lambda x: id_map[x])

        df_node_feat = df_node_feat.sort_values(by=["idx"], ascending=True)

        # # -- edge features ------------------------------------------------------------------------------------------------------
        df_edge_feat["msg"] = df_edge_feat["body_2"].apply(EEDPreprocessing.body_to_edge_feat)

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
    eed.processing("eed")
    eed.save()