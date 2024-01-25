import numpy as np
import pandas as pd
from temporal_graph.base import Preprocessing

class PseudoData(Preprocessing):
    def __init__(self, input, output, dataset_name, num_nodes, num_edges, node_feat_dim, edge_type, t) -> None:
        super().__init__(input=input, output=output, dataset_name=dataset_name)
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.node_feat_dim = node_feat_dim
        self.edge_type = edge_type
        self.t = t


    def processing(self, dataset_name):
        # -- generate pseudo node features
        x = np.random.rand(self.num_nodes, self.node_feat_dim).astype(np.float32)

        # -- generate pseudo edge features in different time
        t = np.random.randint(0, self.t, size=self.num_edges)
        t = np.sort(t)

        # -- generate pseudo edge features(edge type, one-hot encoding)
        msg = np.random.randint(0, self.edge_type, size=self.num_edges)
        msg = np.eye(self.edge_type)[msg]

        # -- generate pseudo src and dst nodes
        src = np.random.randint(0, self.num_nodes, size=self.num_edges)
        dst = np.random.randint(0, self.num_nodes, size=self.num_edges)

        # -- check src and dst nodes are not the same
        while np.sum(src == dst) > 0:
            dst[src == dst] = np.random.randint(0, self.num_nodes, size=np.sum(src == dst))
        
        
        data = {
            "t": t,
            "msg": msg,
            "src": src,
            "dst": dst,
        }

        return data, x
    

if __name__ == "__main__":
    pseudo = PseudoData(
        input="raw", 
        output="data", 
        dataset_name="pseudo_data",
        num_nodes=50,
        num_edges=1000,
        node_feat_dim=10,
        edge_type=10,
        t=50
    )
    
    pseudo.save()