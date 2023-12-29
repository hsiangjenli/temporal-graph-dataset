from temporal_graph import TemporalGraph
import numpy as np

tg_data = TemporalGraph(root="data")
data, train_mask, val_mask, test_mask = tg_data("enron")

print(data)
print(data[train_mask])
print(data[val_mask])
print(data[test_mask])

# for dataset_name in tg_data.available_datasets:
#     print(f"Downloading {dataset_name}...")
#     data = tg_data(dataset_name)
#     print(data)
#     print(f"Downloaded {dataset_name}!")