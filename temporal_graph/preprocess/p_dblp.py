import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from temporal_graph.base import Preprocessing

Seperator = ["-", "/", "&", "+"]
Month = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
Season = ["spring", "summer", "fall", "winter"]
Quarter = ["first", "second", "third", "fourth"]


class DBLPPreprocessing(Preprocessing):
	def __init__(self, input, output, dataset_name, top_n_journal) -> None:
		super().__init__(input=input, output=output, dataset_name=dataset_name, top_n_journal=top_n_journal)

	@staticmethod
	def month_clean(row):
		if isinstance(row, str):
			row = row.lower()
			for s in Seperator:
				if s in row:
					return row.split(s)[-1].strip().lower()

			for m in Month:
				if m[:3] in row:
					return m

			for s in Season:
				if s in row:
					return Month[Season.index(s) * 3 + 2]

			for q in Quarter:
				if q in row:
					return Month[Quarter.index(q) * 3]

			return row.lower()
		else:
			return random.choice(Month)

	@staticmethod
	def split_author(row):
		try:
			return row.split("|")
		except AttributeError:
			return row

	@staticmethod
	def author_to_list_dir(row):
		if isinstance(row, list):
			return [[row[i], row[i + 1]] for i in range(len(row) - 1)]
		else:
			return row

	def processing(self, dataset_name):
		df_edge_feat = pd.read_csv(f"{self._input_folder(dataset_name)}/raw_article.csv", sep=";")
		df_edge_feat = df_edge_feat[["author", "title", "year", "month", "journal"]]
		df_edge_feat.dropna(inplace=True)

		ohc = OneHotEncoder(handle_unknown="ignore", sparse=False, drop="first")
		top_journal = df_edge_feat["journal"].value_counts().head(self.top_n_journal).index.tolist()

		df_edge_feat["journal"] = df_edge_feat["journal"].apply(lambda x: x if x in top_journal else "Other")
		ohc.fit(df_edge_feat[["journal"]])

		df_edge_feat["msg"] = ohc.transform(df_edge_feat[["journal"]]).tolist()
		df_edge_feat["month"] = df_edge_feat["month"].apply(DBLPPreprocessing.month_clean)
		df_edge_feat["author"] = df_edge_feat["author"].apply(DBLPPreprocessing.split_author)

		df_edge_feat["year"] = df_edge_feat["year"].apply(lambda x: int(x))
		df_edge_feat["month"] = df_edge_feat["month"].apply(lambda x: Month.index(x.lower()) + 1)

		df_edge_feat["t"] = df_edge_feat["year"] * 1000 + round(df_edge_feat["month"] / 12 * 100)
		df_edge_feat["t"] = df_edge_feat["t"].apply(lambda x: int(x))

		df_edge_feat["author_to_list_dir"] = df_edge_feat["author"].apply(DBLPPreprocessing.author_to_list_dir)

		df_edge_feat = df_edge_feat.explode("author_to_list_dir")

		df_edge_feat.dropna(inplace=True)
		df_edge_feat.sort_values(by=["t"], inplace=True)
		df_edge_feat = df_edge_feat.reset_index(drop=True)

		df_edge_feat = pd.concat([pd.DataFrame(df_edge_feat["author_to_list_dir"].to_list(), columns=["src", "dst"]), df_edge_feat["msg"], df_edge_feat["t"]], axis=1)

		num_nodes, id_map, src_idx, dst_idx = self._src_dst_to_idx(df_edge_feat["src"], df_edge_feat["dst"])

		msg = np.array(df_edge_feat["msg"].to_list(), dtype=np.float32)
		msg = msg.astype(float)

		data = {"t": df_edge_feat["t"].to_numpy(), "msg": msg, "src": src_idx, "dst": dst_idx}

		return data, np.ones((num_nodes, 1), dtype=np.float32)


if __name__ == "__main__":
	dblp = DBLPPreprocessing(input="raw", output="data", dataset_name="dblp", top_n_journal=15)
	dblp.save()
