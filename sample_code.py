from temporal_graph import TemporalGraph
import numpy as np

tg_dataset = TemporalGraph(root="data2")
# print(tg_dataset.available_datasets)
# data, train_mask, val_mask, test_mask = tg_dataset("tgbl-coin")
# print(data)
# print(data.src_x)
# print(data[train_mask])

data, train_mask, val_mask, test_mask = tg_dataset("tgbl-flight")
print(data)
print(data.src_x)
print(data[train_mask])

# for dataset_name in tg_data.available_datasets:
#     print(f"Downloading {dataset_name}...")
#     data = tg_data(dataset_name)
#     print(data)
#     print(f"Downloaded {dataset_name}!")