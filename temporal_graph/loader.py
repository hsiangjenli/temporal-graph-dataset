import torch
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader

from temporal_graph import TemporalGraph

class TemporalGraphLoader:
    def __init__(self, device: torch.device = torch.device('cpu'), root: str = 'data', loader: callable = TemporalDataLoader):
        self.root = root
        self.device = device
        self.loader = loader
    
    
    def load(self, name: str, batch_size: int=100, neg_sampling_ratio=1.0) -> TemporalData:
        kwargs = {"batch_size": batch_size, "neg_sampling_ratio": neg_sampling_ratio}
        name = name.upper()
        self.data, self.train_mask, self.val_mask, self.test_mask = self._load(name)
        self.data = self.data.to(self.device)
        
        return self.loader(self.train_data, **kwargs)#, self.loader(self.val_data, **kwargs), self.loader(self.test_data, **kwargs)

    @property
    def train_data(self) -> TemporalData:
        return self.data[self.train_mask]
    
    @property
    def val_data(self) -> TemporalData:
        return self.data[self.val_mask]
    
    @property
    def test_data(self) -> TemporalData:
        return self.data[self.test_mask]

    def _load(self, name: str) -> TemporalData:

        if name == 'DBLP':
            dataset = DBLPDataset(root=self.root, name=name)
            return dataset[0], torch.from_numpy(dataset.train_mask).to(torch.bool), torch.from_numpy(dataset.val_mask).to(torch.bool), torch.from_numpy(dataset.test_mask).to(torch.bool)

        elif name == 'EED':
            dataset = EnronMailDataset(root=self.root, name=name)
            return dataset[0], torch.from_numpy(dataset.train_mask).to(torch.bool), torch.from_numpy(dataset.val_mask).to(torch.bool), torch.from_numpy(dataset.test_mask).to(torch.bool)
        
        elif name.startswith('TGBL'):
            dataset = PyGLinkPropPredDataset(name=name.lower(), root=self.root)
            return dataset.get_TemporalData(), dataset.train_mask, dataset.val_mask, dataset.test_mask

        else:
            raise ValueError(f'Unknown dataset: {name}')