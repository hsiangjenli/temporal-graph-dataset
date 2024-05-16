import numpy as np
import pandas as pd
from temporal_graph.base import Preprocessing


class FlightPreprocessing(Preprocessing):
	def __init__(self, input, output, dataset_name) -> None:
		super().__init__(input=input, output=output, dataset_name=dataset_name)

	@staticmethod
	def convert_type(i):
		unique_type = ["heliport", "small_airport", "closed", "seaplane_base", "balloonport", "medium_airport", "large_airport"]
		return unique_type.index(i)

	@staticmethod
	def convert_str2int(in_str: str):
		out = []
		for element in in_str:
			if element.isnumeric():
				out.append(element)
			elif element == "!":
				out.append(0)
			else:
				out.append(ord(element.upper()) - 44 + 9)
		# out = np.array(out, dtype=np.float32)
		return out

	@staticmethod
	def convert_str2int_list(in_str: str):
		out = []
		for element in in_str:
			if element.isnumeric():
				out.append(element)
			elif element == "!":
				out.append(0)
			else:
				out.append(ord(element.upper()) - 44 + 9)
		out = np.array(out, dtype=np.float32)
		return out

	@staticmethod
	def padding_callsign(i):
		i = str(i)
		if len(i) == 0:
			i = "!!!!!!!!"
		while len(i) < 8:
			i += "!"
		return i

	@staticmethod
	def padding_typecode(i):
		i = str(i)
		if len(i) == 0 or i == "nan":
			i = "!!!!!!!!"
		while len(i) < 8:
			i = f"{'!'*(8-len(i))}{i}"
		if len(i) > 8:
			i = "!!!!!!!!"
		return i

	@staticmethod
	def padding_iso_region(i):
		i = i.replace("-", "")
		if len(i) != 6:
			i = f"{'!'*(6-len(i))}{i}"
		return i

	@staticmethod
	def convert_continent(i):
		unique_continent = ["OC", "AF", "AN", "EU", "AS", "SA"]
		if isinstance(i, float):
			return [0]
		else:
			return [unique_continent.index(i) + 1]

	def processing(self, dataset_name):
		df_edge_feat = pd.read_csv(f"{self._input_folder(dataset_name)}/{dataset_name}_edgelist_v2.csv")
		df_node_feat = pd.read_csv(f"{self._input_folder(dataset_name)}/airport_node_feat_v2.csv")

		num_nodes, id_map, src_idx, dst_idx = self._src_dst_to_idx(df_edge_feat["src"], df_edge_feat["dst"])

		# -- node features ------------------------------------------------------------------------------------------------------
		df_node_feat = df_node_feat[df_node_feat["airport_code"].isin(id_map.keys())]

		df_node_feat["iso_region"] = df_node_feat["iso_region"].apply(FlightPreprocessing.padding_iso_region)
		df_node_feat["iso_region"] = df_node_feat["iso_region"].apply(FlightPreprocessing.convert_str2int)
		df_node_feat["continent"] = df_node_feat["continent"].apply(FlightPreprocessing.convert_continent)
		df_node_feat["type"] = df_node_feat["type"].apply(FlightPreprocessing.convert_type)
		df_node_feat["type"] = df_node_feat["type"].apply(lambda x: [x])
		df_node_feat["longitude"] = df_node_feat["longitude"].apply(lambda x: [x])
		df_node_feat["latitude"] = df_node_feat["latitude"].apply(lambda x: [x])

		df_node_feat["combined"] = df_node_feat["iso_region"] + df_node_feat["continent"] + df_node_feat["type"] + df_node_feat["longitude"] + df_node_feat["latitude"]
		df_node_feat["idx"] = df_node_feat["airport_code"].apply(lambda x: id_map[x])
		df_node_feat = df_node_feat.set_index("idx")
		df_node_feat = df_node_feat.sort_index()

		x = np.array(df_node_feat["combined"].to_list(), dtype=np.float32)
		x = x.astype(float)

		# -- edge features ------------------------------------------------------------------------------------------------------
		df_edge_feat["callsign"] = df_edge_feat["callsign"].apply(FlightPreprocessing.padding_callsign)
		df_edge_feat["typecode"] = df_edge_feat["typecode"].apply(FlightPreprocessing.padding_typecode)
		df_edge_feat["msg"] = df_edge_feat["callsign"] + df_edge_feat["typecode"]
		df_edge_feat["msg"] = df_edge_feat["msg"].apply(FlightPreprocessing.convert_str2int)

		msg = np.array(df_edge_feat["msg"].to_list(), dtype=np.float32)
		msg = msg.astype(float)

		data = {"t": df_edge_feat["timestamp"].to_numpy(), "msg": msg, "src": src_idx, "dst": dst_idx}

		return data, x


if __name__ == "__main__":
	flight = FlightPreprocessing(input="raw", output="data_v3", dataset_name="tgbl-flight")
	flight.save()
