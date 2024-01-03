# temporal-graph-dataset

A collection of temporal graph datasets from [***Towards Better Evaluation for Dynamic Link Prediction***]() with pytorch-geometric `TemporalData` class and `TemporalDataloader`. 

***BUT***, this repository is not official, some of the datasets are not included, and the preprocessing is not the same as the original paper.

## Installation

### From source
```
git clone https://github.com/hsiangjenli/temporal-graph-dataset.git
cd temporal-graph-dataset
pip install -e .
```

### From PyPI
```
pip install [TODO]
```

## How to use
```python
from temporal_graph import TemporalGraph

tg_dataset = TemporalGraph(root="data")
```

### List available datasets
```python
print(tg_dataset.available_datasets)
```


## How I preprocess the dataset
- `p_{dataset_name}.py` is a Python script to preprocess the dataset.

## Credit to ..
- Dataset for "Towards Better Evaluation for Dynamic Link Prediction"
  - https://zenodo.org/records/7213796#.Y1cO6y8r30o
  - Poursafaei, Farimah and Huang, Shenyang and Pelrine, Kellin and and Rabbany, Reihaneh