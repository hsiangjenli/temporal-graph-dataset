import torch
from torch.nn import ReLU
from torch.nn import Linear

class SimpleLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)

        self.relu = ReLU(inplace=False)
        
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = self.relu(h)
        out = self.lin_final(h).clone()
        return out