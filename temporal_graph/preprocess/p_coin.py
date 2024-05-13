import numpy as np
import pandas as pd
from temporal_graph.base import Preprocessing


class CoinPreprocessing(Preprocessing):
	def __init__(self, input, output, dataset_name) -> None:
		super().__init__(input=input, output=output, dataset_name=dataset_name)

	def processing(self, dataset_name):
		df_edge_feat = pd.read_csv(f"{self._input_folder(dataset_name)}/{dataset_name}_edgelist_v2.csv")
		num_nodes, id_map, src_idx, dst_idx = self._src_dst_to_idx(df_edge_feat["src"], df_edge_feat["dst"])
		data = {"t": df_edge_feat["day"].to_numpy(), "msg": np.ones((df_edge_feat.shape[0], 1), dtype=np.float32), "src": src_idx, "dst": dst_idx}
		x = np.ones((num_nodes, 1), dtype=np.float32)
		return data, x


if __name__ == "__main__":
	coin = CoinPreprocessing(input="raw", output="data", dataset_name="tgbl-coin")
	coin.save()
