from temporal_graph import TemporalGraph

tg_data = TemporalGraph(root="data")

for dataset_name in tg_data.available_datasets:
    print(f"Downloading {dataset_name}...")
    data = tg_data(dataset_name)
    print(data)
    print(f"Downloaded {dataset_name}!")