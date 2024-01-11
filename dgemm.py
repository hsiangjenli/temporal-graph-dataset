
from temporal_graph import TemporalGraph

from typing import Callable

import torch
from torch.nn import Linear, GRUCell

from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import MessagePassing

class KHopAggregator(MessagePassing):
    def __init__(self, k):
        super(KHopAggregator, self).__init__(aggr='add')
        self.k = k
    
    def forward(self, x, edge_index):
        for _ in range(self.k):
            x = self.propagate(edge_index, x=x)
        return x

class Net(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, embedd_dim, time_dim, his_loader):
        super(Net, self).__init__()

        edge_dim = edge_dim + 2 * (node_dim + time_dim)

        self.aggr = KHopAggregator(k=3)
        self.attn = TransformerConv(in_channels=embedd_dim, out_channels=embedd_dim, edge_dim=edge_dim)

        self.his_loader = his_loader

    def forward(self, x, n_id, src_n_id, dst_n_id, edge_index, edge_attr, t):

        m_edge_index = torch.cat([edge_index, self.his_loader.edge_index(n_id)], dim=1)
        x = self.aggr(x, m_edge_index)

        src_enc_t, dst_enc_t = self.his_loader.enc_t(src_n_id), self.his_loader.enc_t(dst_n_id)
        t = t.view(-1, 1).expand_as(src_enc_t)
        src_rel_t = t - src_enc_t
        dst_rel_t = t - dst_enc_t

        edge_attr = torch.cat([edge_attr, src_rel_t, dst_rel_t, x[src_n_id], x[dst_n_id]], dim=-1)

        return self.attn(self.his_loader.z, edge_index, edge_attr)

class HistoryLoader:
    def __init__(self, num_nodes, memory_dim, embedd_dim, time_dim):
        """
        Args:
            memory_dim (int): how many edges to remember
        """
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim

        self.memory_z = torch.zeros((num_nodes, embedd_dim), dtype=torch.float32)

        self.memory_t = torch.zeros((num_nodes, memory_dim), dtype=torch.float32)
        self.memory_edge_index = torch.zeros((num_nodes, memory_dim), dtype=torch.long)
        self.memory_edge_attr = torch.zeros((num_nodes, memory_dim), dtype=torch.float32)

        self.next_insert_position = torch.zeros(num_nodes, dtype=torch.long)
        self.cur_e_id = 0

        self.t_encoder = TimeEncoder(memory_dim=memory_dim, time_dim=time_dim)
        self.m_module = GRUCell(embedd_dim, memory_dim)
    
    def retrieve(self, n_id):
        edge_index = self.edge_index(n_id)
        t = self.t(n_id)
        self.memory_z[n_id] = self.m_module(self.memory_z[n_id])

        return self.cur_e_id, edge_index, t, self.memory_z
    
    def insert(self, src_n_id, dst_n_id, t, edge_attr, z):
        self.cur_e_id += src_n_id.shape[0]
        for src, dst, node_t, msg in zip(src_n_id, dst_n_id, t, edge_attr):
            position = self.next_insert_position[src].item()

            self.memory_z[src] = self.m_module(z[src])
            self.memory_edge_index[src, position] = dst
            #self.memory_edge_attr[src, position] = msg
            self.memory_t[src, position] = node_t

            self.next_insert_position[src] = (position + 1) % self.memory_dim
    
    @property
    def z(self):
        return self.memory_z

    def t(self, n_id):
        return self.memory_t[n_id]#.float()

    def enc_t(self, n_id):
        return self.t_encoder(self.t(n_id))
                        
    def edge_index(self, n_id):
        memory_edge = self.memory_edge_index[n_id]
        valid_edge_index = torch.nonzero(memory_edge, as_tuple=False)
        # -- replace the dst edge index with the correct node id --
        valid_edge_index[:, 1] = memory_edge[valid_edge_index[:, 0], valid_edge_index[:, 1]]
        # -- replace the src edge index with the correct node id --
        valid_edge_index[:, 0] = n_id[valid_edge_index[:, 0]]
        return valid_edge_index.reshape(-1, 2).t().contiguous()


class TimeEncoder(torch.nn.Module):
    def __init__(self, memory_dim, time_dim):
        super(TimeEncoder, self).__init__()
        self.time_dim = time_dim
        self.lin = Linear(memory_dim, time_dim)
    
    def reset_parameters(self):
        self.lin.reset_parameters()
    
    def forward(self, t):
        return self.lin(t).cos()

# ========= setup dataset ====================================================================
dataset = TemporalGraph(root="data2")
data, x, train_mask, val_mask, test_mask = dataset("tgbl-flight")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
x = x.to(device)

loader = TemporalDataLoader(data[train_mask], batch_size=5, shuffle=False)
# ========= setup parameters =================================================================
node_dim = 10       
edge_dim = 16
embedd_dim = 20     # embedding dimension for each node
memory_dim = 20     # how many edges to remember
time_dim = 10       # using linear transformation to encode memory time into smaller dimension
# ============================================================================================

history_loader = HistoryLoader(num_nodes=data.num_nodes, memory_dim=memory_dim, embedd_dim=embedd_dim, time_dim=time_dim)
model = Net(node_dim=node_dim, edge_dim=edge_dim, embedd_dim=embedd_dim, time_dim=time_dim, his_loader=history_loader)

for i, batch in enumerate(loader):

    # cur_e_id, m_edge_index, m_t, m_z = history_loader.retrieve(batch.n_id)

    z = model(x=x, n_id=batch.n_id, edge_index=batch.edge_index, edge_attr=batch.msg, t=batch.t, src_n_id=batch.src, dst_n_id=batch.dst)
    history_loader.insert(src_n_id=batch.src, dst_n_id=batch.dst, t=batch.t, edge_attr=batch.msg, z=z)

    # print(batch.edge_index.shape)
    # print(history_loader.edge_index(batch.n_id).shape)
    # break
    
    if i == 100:
        print(z[batch.n_id])
        break