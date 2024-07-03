from torch_geometric.data import TemporalData
import tqdm
import torch

import seaborn as sns
import matplotlib.pyplot as plt


class TemporalGraphInfo:
	def __init__(self, x, data: TemporalData, train_mask: list, val_mask: list, test_mask: list, batch_size: int) -> None:
		self.data = data
		self.x = x

		self.train_mask = train_mask
		self.val_mask = val_mask
		self.test_mask = test_mask

		self.batch_size = batch_size

		# -- basic properties --
		self.num_nodes = self.data.num_nodes
		self.num_edges = self.data.num_edges

		self._already_prepare = None
		self._num_unique_edges = None

		# -- features properties --
		self.num_node_features = self.x.size(1)
		self.num_edge_features = self.data.msg.size(1)

		# -- unique properties --
		self.unique_time_steps = self.data.t.unique().size(0)

		# -- inductive / transductive links --
		self.inductive_links_mask = None

		# -- set style --
		plt.style.use("fivethirtyeight")

		plt.rcParams["figure.facecolor"] = "white"
		plt.rcParams["axes.facecolor"] = "white"
		plt.rcParams["savefig.facecolor"] = "white"
		plt.rcParams["savefig.dpi"] = 300

	def report(self, to_frame=False, to_latex=False, name="default"):
		report_dict = {
			"Number of Nodes ($N$)": self.num_nodes,
			"Dimensionality of Node Features ($d_{v}$)": self.num_node_features,
			"Number of Edges ($E$)": self.num_edges,
			"Number of Unique Pairs of Vertices": self.num_unique_edges,
			"Dimensionality of Node Features ($d_{e}$)": self.num_edge_features,
			"Number of Unique Time Steps": self.unique_time_steps,
			"Number of inductive links": self.inductive_links,
			"Inductive links (\%)": self.inductive_links_pct,
			"Number of transductive links": self.tranductive_links,
			"Transductive links (\%)": self.tranductive_links_pct,
		}

		if to_frame or to_latex:
			import pandas as pd

			df = pd.DataFrame(report_dict, index=[name]).T

			def float_or_int(x):
				if x.is_integer():
					return f"{int(x):,}"
				else:
					return f"{x:.2f}"

			df = df.applymap(float_or_int)

			if to_frame:
				return df
			elif to_latex:
				return df.to_latex(escape=False)

		return report_dict

	def _prepare_for_process_num_unique_edges(self):
		self._src_n_id = self.data.src.clone()
		self._dst_n_id = self.data.dst.clone()

		self._src_n_id, self._src_perm = self._src_n_id.sort()
		self._dst_n_id, self._dst_perm = self._dst_n_id.sort()

		self._u_src, self._count_con_src = self._src_n_id.unique_consecutive(return_counts=True)
		self._u_dst, self._count_con_dst = self._dst_n_id.unique_consecutive(return_counts=True)

		_, self._count_src = self._src_n_id.unique(return_counts=True)
		_, self._count_dst = self._dst_n_id.unique(return_counts=True)

		self._num_u_src = self._u_src.size(0)
		self._num_u_dst = self._u_dst.size(0)

		self._already_prepare = True

	def _process_num_unique_edges(self):
		if self._num_u_src < self._num_u_dst:
			src_count = self._count_con_src
			src_perm = self._src_perm
			dst_nodes = self.data.dst
		else:
			src_count = self._count_con_dst
			src_perm = self._dst_perm
			dst_nodes = self.data.src

		num_unique_edges = 0

		for idx in tqdm.tqdm(src_perm.split(src_count.tolist()), desc="Counting unique edges"):
			num_unique_edges += dst_nodes[idx].unique().size(0)

		self._num_unique_edges = num_unique_edges

	def check_num_unique_edges_is_valid(func):
		def wrapper(self, *args, **kwargs):
			if self._num_unique_edges is None:
				print("Calculating unique edges")
				self._prepare_for_process_num_unique_edges()
				self._process_num_unique_edges()

			return func(self, *args, **kwargs)

		return wrapper

	def check_prepare_for_process_num_unique_edges_is_valid(func):
		def wrapper(self, *args, **kwargs):
			if self._already_prepare is None:
				print("Preparing for process unique edges")
				self._prepare_for_process_num_unique_edges()

			return func(self, *args, **kwargs)

		return wrapper

	@property
	@check_num_unique_edges_is_valid
	def num_unique_edges(self):
		return self._num_unique_edges

	# -- average properties --
	@property
	@check_prepare_for_process_num_unique_edges_is_valid
	def average_in_degree(self):
		return self._count_dst.float().mean().item()

	@property
	@check_prepare_for_process_num_unique_edges_is_valid
	def average_out_degree(self):
		return self._count_src.float().mean().item()

	# -- std properties --
	@property
	@check_prepare_for_process_num_unique_edges_is_valid
	def std_in_degree(self):
		return self._count_dst.float().std().item()

	@property
	@check_prepare_for_process_num_unique_edges_is_valid
	def std_out_degree(self):
		return self._count_src.float().std().item()

	@property
	@check_prepare_for_process_num_unique_edges_is_valid
	def max_in_degree(self):
		return self._count_dst.max().item()

	@property
	@check_prepare_for_process_num_unique_edges_is_valid
	def max_out_degree(self):
		return self._count_src.max().item()

	@property
	@check_prepare_for_process_num_unique_edges_is_valid
	def min_in_degree(self):
		return self._count_dst.min().item()

	@property
	@check_prepare_for_process_num_unique_edges_is_valid
	def min_out_degree(self):
		return self._count_src.min().item()

	# -- plot --
	def in_degree_distribution(self, **kwargs):
		plt.suptitle("In-degree distribution")
		return sns.histplot(self._count_dst, **kwargs)

	def out_degree_distribution(self, **kwargs):
		plt.suptitle("Out-degree distribution")
		return sns.histplot(self._count_src, **kwargs)

	# -- outliers --
	def in_degree_outliers(self):
		return NotImplementedError

	# -- transductive / inductive links --
	def check_inductive_mask_is_valid(func):
		def wrapper(self, *args, **kwargs):
			if self.inductive_links_mask is None:
				print("Calculating inductive links mask")
				self._process_inductive_links_mask()

			return func(self, *args, **kwargs)

		return wrapper

	def _process_inductive_links_mask(self):
		"""
		The definition of inductive links is the nodes that never seen in training set but in test set will be considered as inductive links
		"""
		# -- the nodes that appeared in training set --
		self.train_src_n_id = self.data.src[self.train_mask].unique()
		self.train_dst_n_id = self.data.dst[self.train_mask].unique()

		# -- the nodes that appeared in test set --
		self.test_src_n_id = self.data.src[self.test_mask]
		self.test_dst_n_id = self.data.dst[self.test_mask]

		# -- inductive links --
		src_inductive_mask = ~torch.isin(self.test_src_n_id, self.train_src_n_id)
		dst_inductive_mask = ~torch.isin(self.test_dst_n_id, self.train_dst_n_id)

		self.inductive_links_mask = torch.logical_or(src_inductive_mask, dst_inductive_mask)

	@property
	@check_inductive_mask_is_valid
	def inductive_links(self):
		return self.inductive_links_mask.sum().item()

	@property
	@check_inductive_mask_is_valid
	def inductive_links_pct(self):
		return self.inductive_links / sum(self.test_mask)

	@property
	@check_inductive_mask_is_valid
	def tranductive_links(self):
		return sum(self.test_mask) - self.inductive_links

	@property
	@check_inductive_mask_is_valid
	def tranductive_links_pct(self):
		return self.tranductive_links / sum(self.test_mask)


if __name__ == "__main__":
	from temporal_graph import TemporalGraphDataset
	import pandas as pd

	tgd = TemporalGraphDataset(root="data")
	dataset = ["eed", "dblp", "GDELTLite"]
	batch_size = 1000
	reports = []

	for d_name in dataset:
		data, x, train_mask, val_mask, test_mask = tgd(d_name)
		tgi = TemporalGraphInfo(x, data, train_mask, val_mask, test_mask, batch_size)
		report = tgi.report(to_frame=True, name=d_name.upper())
		reports.append(report)

	df = pd.concat(reports, axis=1)
	print(df.to_latex(escape=False))
