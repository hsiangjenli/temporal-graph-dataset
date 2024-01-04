import torch
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader

from temporal_graph import TemporalGraph

class TemporalGraphLoader:
    def __init__(self, device: torch.device = torch.device('cpu'), root: str = 'data', loader: callable = TemporalDataLoader):
        self.root = root
        self.device = device
        self.loader = loader
        self.dataset = TemporalGraph(root=self.root)
    
    def train_loader(self, dataset_name: str, batch_size: int=100, neg_sampling_ratio=1.0):
        kwargs = {"batch_size": batch_size, "neg_sampling_ratio": neg_sampling_ratio}

        self.data, self.train_mask, self.val_mask, self.test_mask = self.dataset(dataset_name)
        self.data = self.data.to(self.device)
        
        return self.loader(self.data[self.train_mask], **kwargs)
    
    def val_loader(self, dataset_name: str, batch_size: int=100, neg_sampling_ratio=1.0):
        kwargs = {"batch_size": batch_size, "neg_sampling_ratio": neg_sampling_ratio}
        return self.loader(self.data[self.val_mask], **kwargs)