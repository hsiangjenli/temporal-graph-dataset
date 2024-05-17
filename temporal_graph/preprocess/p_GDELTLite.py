from torch_geometric.datasets import GDELTLite
from temporal_graph.base import Preprocessing
import numpy as np


class GDELTLitePreprocessing(Preprocessing):
    def __init__(self, input, output, dataset_name) -> None:
        super().__init__(input=input, output=output, dataset_name=dataset_name)
		
    def processing(self, dataset_name: str=None):
        raw = GDELTLite(root='raw/GDELTLite')[0]
        data = {"t": raw.time.numpy(), "src": raw.edge_index[0].numpy(), "dst": raw.edge_index[1].numpy(), "msg": raw.edge_attr.numpy()}
        
        x = np.array(raw.x.numpy(), dtype=np.float32)
        x = x.astype(float)
        return data, x
    
if __name__ == "__main__":
    gdel = GDELTLitePreprocessing(input="raw", output="data_v3", dataset_name="GDELTLite")
    gdel.processing("GDELTLite")
    gdel.save()