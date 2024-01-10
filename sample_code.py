from temporal_graph import TemporalGraph
from torch_geometric.loader import TemporalDataLoader
import numpy as np

tg_dataset = TemporalGraph(root="data2")
# print(tg_dataset.available_datasets)
# data, x, train_mask, val_mask, test_mask = tg_dataset("tgbl-coin")
# print(data)
# print(x)
# # print(data[train_mask])

# loader = TemporalDataLoader(data, batch_size=1, shuffle=False)

# for batch in loader:
#     print(batch)
#     break

data, x, train_mask, val_mask, test_mask = tg_dataset("tgbl-flight")
print(data)
# print(data[train_mask])

loader = TemporalDataLoader(data, batch_size=1, shuffle=False)

for batch in loader:
    print(batch)
    break

# for dataset_name in tg_data.available_datasets:
#     print(f"Downloading {dataset_name}...")
#     data = tg_data(dataset_name)
#     print(data)
#     print(f"Downloaded {dataset_name}!")