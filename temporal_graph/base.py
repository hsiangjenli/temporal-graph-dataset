import os
import torch
import pickle
import temporal_graph.utils as utils
from torch_geometric.data import TemporalData


class Preprocessing:
	def __init__(self, input, output, dataset_name, **kwargs) -> None:
		self._input = input
		self._output = output
		self.dataset_name = dataset_name

		for k, v in kwargs.items():
			setattr(self, k, v)

	@property
	def pkg_dir(self) -> str:
		return os.path.dirname(os.path.abspath(__file__))

	@property
	def root(self) -> str:
		if self._root is None:
			return os.path.join(self.pkg_dir, "data")
		else:
			return self._root

	def processing(self):
		raise NotImplementedError

	def save(self):
		# -- mkdir ------------------------------------------------------------------------------
		os.makedirs(self._output_folder(self.dataset_name), exist_ok=True)

		data, x = self.processing(self.dataset_name)
		data = TemporalData(**data)
		x = torch.from_numpy(x).to(torch.float32)

		# -- save the data -------------------------------------------------------------------
		torch.save(data, self._pyg_path(self.dataset_name), pickle_protocol=4)
		torch.save(x, self._node_feat_path(self.dataset_name), pickle_protocol=4)

		# -- save the train/val/test split ---------------------------------------------------
		train_mask, val_mask, test_mask = utils.generate_splits(data)
		pickle.dump(train_mask, open(self._mask_path(self.dataset_name, "train"), "wb"))
		pickle.dump(val_mask, open(self._mask_path(self.dataset_name, "val"), "wb"))
		pickle.dump(test_mask, open(self._mask_path(self.dataset_name, "test"), "wb"))

	def _input_folder(self, dataset_name) -> str:
		return os.path.join(self._input, dataset_name)

	def _output_folder(self, dataset_name) -> str:
		return os.path.join(self._output, dataset_name)

	def _mask_path(self, dataset_name, mask_type) -> str:
		return os.path.join(self._output_folder(dataset_name), f"mask_{mask_type}.pkl")

	def _pyg_path(self, dataset_name) -> str:
		return os.path.join(self._output_folder(dataset_name), f"pyg_{dataset_name}.pt")

	def _node_feat_path(self, dataset_name) -> str:
		return os.path.join(self._output_folder(dataset_name), f"pyg_{dataset_name}_node_feat.pt")

	def _src_dst_to_idx(self, src, dst):
		nodes = set(src.unique().tolist() + dst.unique().tolist())
		num_nodes = len(nodes)
		id_map = {}
		for i, node in enumerate(nodes):
			id_map[node] = i

		src_idx = src.apply(lambda x: id_map[x]).to_numpy()
		dst_idx = dst.apply(lambda x: id_map[x]).to_numpy()

		return num_nodes, id_map, src_idx, dst_idx
