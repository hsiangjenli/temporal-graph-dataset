<div align="center"><center><b>

# Temporal Graph Dataset

</b></center></div>


A collection of temporal graph datasets from [***Towards Better Evaluation for Dynamic Link Prediction***]() with pytorch-geometric `TemporalData` class and `TemporalDataloader`. 

***BUT***, this repository is not official, some of the datasets are not included, and the preprocessing is not the same as the original paper.

## Installation

```
pip install git+https://github.com/hsiangjenli/temporal-graph-dataset.git
```

## How to use
```python
from temporal_graph import TemporalGraphDataset

tgd = TemporalGraphDataset(root="data")
```

### List available datasets
```python
tgd.available_datasets
```

### Get a dataset
```python
data, x, train_mask, val_mask, test_mask = tgd("tgbl-flight")
```

### TemporalGraphLoader
- torch_geometric.loader.TemporalDataLoader
  
```python
from torch_geometric.loader import TemporalDataLoader

loader = TemporalDataLoader(data[train_mask], batch_size=1000, neg_sampling_ratio=1.0)
```

## How I preprocess the dataset
- `temporal_graph/preprocess/p_{dataset_name}.py` is a Python script to preprocess the dataset. To see the details, please check the **Makefile**.


## Credit to ..
- Dataset for "Towards Better Evaluation for Dynamic Link Prediction"
  - https://zenodo.org/records/7213796#.Y1cO6y8r30o
  - Poursafaei, Farimah and Huang, Shenyang and Pelrine, Kellin and and Rabbany, Reihaneh

<blockquote>

## **⚠️ NOTE ⚠️**  
This project is part of my master's thesis. Feel free to use or modify the code, but please keep in mind that I'm just a student, so there might be mistakes. If you notice any bugs or have suggestions for improvement, please feel free to open an issue or pull request. Thanks for your understanding and support!"

<div align="right"><b><i>RN Lee @ 2024-05</i></b> 😊 </div></blockquote>