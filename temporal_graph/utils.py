import numpy as np
from torch_geometric.data import TemporalData

def generate_splits(data: TemporalData, val_ratio: float = 0.15, test_ratio: float = 0.15):

    full_timestamps = data.t

    val_time, test_time = list(np.quantile(full_timestamps,[(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    train_mask = full_timestamps <= val_time
    val_mask = np.logical_and(full_timestamps <= test_time, full_timestamps > val_time)
    test_mask = full_timestamps > test_time

    return list(train_mask), list(val_mask), list(test_mask)