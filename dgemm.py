
from temporal_graph import TemporalGraphDataset

from typing import Callable

import torch
from torch.nn import Linear, GRUCell, GRU
from torch.functional import F

from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import MessagePassing

import torch_geometric.utils as utils

class KHopAggregator(MessagePassing):
    def __init__(self, k):
        super(KHopAggregator, self).__init__(aggr='add')
        self.k = k
    
    def forward(self, x, edge_index):
        for _ in range(self.k):
            x = self.propagate(edge_index, x=x)
        return x

class KHopAggregators(MessagePassing):
    def __init__(self, k):
        super(KHopAggregators, self).__init__(aggr='add')
        self.k = k
    
    def forward(self, x, edge_index):
        # create an empty tensor to store different hop aggregation results
        k_hop_x = torch.zeros((self.k, x.shape[0], x.shape[1]), dtype=torch.float32)
        for k in range(self.k):
            k_hop_x[k] = self.propagate(edge_index, x=x, k=k)
        
        return k_hop_x

class kHopSelector(torch.nn.Module):
    def __init__(self, max_k, his_loader, num_nodes):
        super(kHopSelector, self).__init__()
        self.n_id = torch.arange(num_nodes)
        self.k_for_each_node = torch.ones(num_nodes, dtype=torch.long)
        
        self.his_loader = his_loader
        
        self.gru = GRU(input_size=his_loader.memory_dim, hidden_size=1, num_layers=1, batch_first=True)
        self.lin = Linear(1, max_k)

    def forward(self, n_id):
        # retrieve the memory information
        memory_edge_index = self.his_loader.edge_index(self.n_id)
        memory_degree = self.his_loader.degree(self.n_id)
        
        latest_degree = memory_degree[:, -1]
        latest_degree[:int(memory_edge_index[0].max())+1] = utils.degree(memory_edge_index[0], dtype=torch.float)
        
        latest_degree = latest_degree.view(-1, 1).expand_as(memory_degree)

        # compute the relative degree
        rel_degrees = latest_degree - memory_degree

        x = self.gru(rel_degrees)[0]
        x = self.lin(x).softmax(dim=-1)

        # output the k for each node
        _k_for_each_node = torch.argmax(x, dim=-1) + 1
        self.k_for_each_node[n_id] = _k_for_each_node[n_id]

        return self.k_for_each_node

class Net(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, embedd_dim, time_dim, his_loader, h_edge_attr_dim, k):
        super(Net, self).__init__()

        edge_dim = edge_dim + 2 * (node_dim + time_dim) + h_edge_attr_dim

        self.aggr = KHopAggregators(k=k)
        self.attn = TransformerConv(in_channels=embedd_dim, out_channels=embedd_dim, edge_dim=edge_dim)

        self.his_loader = his_loader

    def forward(self, x, n_id, src_n_id, dst_n_id, edge_index, edge_attr, t, k):

        m_edge_index = torch.cat([edge_index, self.his_loader.edge_index(n_id)], dim=1)
        k_hop_x = self.aggr(x, m_edge_index)
        x = k_hop_x[k, torch.arange(1), :]

        src_enc_t, dst_enc_t = self.his_loader.enc_t(src_n_id), self.his_loader.enc_t(dst_n_id)
        t = t.view(-1, 1).expand_as(src_enc_t)
        src_rel_t = t - src_enc_t
        dst_rel_t = t - dst_enc_t

        edge_attr = torch.cat([edge_attr, src_rel_t, dst_rel_t, x[src_n_id], x[dst_n_id], self.his_loader.h_edge_attr(src_n_id)], dim=-1)

        return self.attn(self.his_loader.z, edge_index, edge_attr)

class HistoryLoader:
    def __init__(self, num_nodes, edge_dim, memory_dim, embedd_dim, time_dim, h_edge_attr_dim):
        """
        Args:
            memory_dim (int): how many edges to remember
        """
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.time_dim = time_dim

        self.memory_z = torch.zeros((num_nodes, embedd_dim), dtype=torch.float32)

        self.memory_t = torch.zeros((num_nodes, memory_dim), dtype=torch.float32)
        self.memory_edge_index = torch.zeros((num_nodes, memory_dim), dtype=torch.long)
        self.memory_edge_attr = torch.zeros((memory_dim, num_nodes, edge_dim), dtype=torch.float32)
        self.memory_h_edge_attr = torch.zeros((num_nodes, h_edge_attr_dim), dtype=torch.float32)
        self.memory_degree = torch.zeros((num_nodes, memory_dim), dtype=torch.float32)

        self.next_insert_position = torch.zeros(num_nodes, dtype=torch.long)
        self.cur_e_id = 0

        self.t_encoder = TimeEncoder(memory_dim=memory_dim, time_dim=time_dim)
        self.m_module = GRUCell(edge_dim, h_edge_attr_dim)
    
    def reset_parameters(self):
        # TODO
        pass
    
    def retrieve(self, n_id):
        edge_index = self.edge_index(n_id)
        t = self.t(n_id)
        self.memory_z[n_id] = self.m_module(self.memory_z[n_id])

        return self.cur_e_id, edge_index, t, self.memory_z
    
    def insert(self, src_n_id, dst_n_id, t, edge_attr, z):
        self.cur_e_id += src_n_id.shape[0]
        position = self.memory_dim - 1

        for src, dst, node_t, msg in zip(src_n_id, dst_n_id, t, edge_attr):
            self.memory_z[src] = z[src]
            
            # move the information from the old position to the new position
            self.memory_edge_index[src, 0: position] = self.memory_edge_index[src, 1:].clone()
            self.memory_edge_attr[0: position, src] = self.memory_edge_attr[1:, position].clone()
            self.memory_t[src, 0: position] = self.memory_t[src, 1:].clone()
            self.memory_degree[src, 0: position] = self.memory_degree[src, 1:].clone()

            # insert the new information
            self.memory_edge_index[src, -1] = dst
            self.memory_edge_attr[-1, position] = msg
            self.memory_t[src, -1] = node_t
            self.memory_degree[src, -1] = utils.degree(self.edge_index(torch.arange(self.num_nodes))[0], dtype=torch.float)[src]
    
    @property
    def z(self):
        return self.memory_z

    def t(self, n_id):
        return self.memory_t[n_id]

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
    
    def edge_attr(self, n_id):
        return self.memory_edge_attr[:, n_id, :]
    
    def h_edge_attr(self, n_id):
        edge_attr = self.edge_attr(n_id)

        for t in range(self.memory_dim):
            print(self.memory_h_edge_attr[n_id])

            hx = self.m_module(edge_attr[t], self.memory_h_edge_attr[n_id])
            self.memory_h_edge_attr[n_id] = hx

        return self.memory_h_edge_attr[n_id]
    
    def degree(self, n_id):
        return self.memory_degree[n_id]


class TimeEncoder(torch.nn.Module):
    def __init__(self, memory_dim, time_dim):
        super(TimeEncoder, self).__init__()
        self.time_dim = time_dim
        self.lin = Linear(memory_dim, time_dim)
    
    def reset_parameters(self):
        self.lin.reset_parameters()
    
    def forward(self, t):
        return self.lin(t).cos()

# ========= setup dataset ===========================================================================================
dataset = TemporalGraphDataset(root="data")
data, x, train_mask, val_mask, test_mask = dataset("eed")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
x = x.to(device)

loader = TemporalDataLoader(data[train_mask], batch_size=10, shuffle=False)
# ========= setup parameters ========================================================================================
k = 3               # how many hops to aggregate
node_dim = x.shape[1]       
edge_dim = data.msg.shape[1]
embedd_dim = 20     # embedding dimension for each node
memory_dim = 100    # how many edges to remember
time_dim = 5        # using linear transformation to encode memory time into smaller dimension
h_edge_attr_dim = 3 # hidden state dimension for edge message/edge attr to remember the past interaction
# ====================================================================================================================

history_loader = HistoryLoader(num_nodes=data.num_nodes, memory_dim=memory_dim, embedd_dim=embedd_dim, time_dim=time_dim, edge_dim=edge_dim, h_edge_attr_dim=h_edge_attr_dim)
model = Net(node_dim=node_dim, edge_dim=edge_dim, embedd_dim=embedd_dim, time_dim=time_dim, his_loader=history_loader, h_edge_attr_dim=h_edge_attr_dim, k=k)
k_hop_sel = kHopSelector(max_k=k, his_loader=history_loader, num_nodes=data.num_nodes)


for i, batch in enumerate(loader):

    if i == 0:
        k_for_each_node = k_hop_sel.k_for_each_node
    else:
        k_for_each_node = k_hop_sel(n_id=batch.src)
        
    z = model(x=x, n_id=torch.arange(data.num_nodes), edge_index=batch.edge_index, edge_attr=batch.msg, t=batch.t, src_n_id=batch.src, dst_n_id=batch.dst, k=k_for_each_node)
    history_loader.insert(src_n_id=batch.src, dst_n_id=batch.dst, t=batch.t, edge_attr=batch.msg, z=z)

    if i == 100:
        print(z[batch.n_id])
        print(k_for_each_node)
        break