import numpy as np
from torch_geometric.data import TemporalData


def generate_splits(data: TemporalData, val_ratio: float = 0.15, test_ratio: float = 0.15):
	train_mask = np.zeros(data.num_nodes, dtype=bool)
	val_mask = np.zeros(data.num_nodes, dtype=bool)
	test_mask = np.zeros(data.num_nodes, dtype=bool)
	train_mask[:10000] = True
	val_mask[10000:11000] = True
	test_mask[11000:] = True
	return train_mask, val_mask, test_mask
