from temporal_graph import TemporalGraphDataset
from torch_geometric.loader import TemporalDataLoader
import numpy as np

tgd = TemporalGraphDataset(root="data_demo")

for dataset_name in tgd.available_datasets:
    print(f"Downloading {dataset_name}...")
    tgd.download(dataset_name)
    print(f"Downloaded {dataset_name}!")

data, x, train_mask, val_mask, test_mask = tgd("eed")
print(data)
print(x)
print(data[train_mask])

loader = TemporalDataLoader(data, batch_size=100, shuffle=False)

for batch in loader:
    print(batch)
    break