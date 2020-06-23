<h2 align="center">Hierarchical Inter-Message Passing for Learning on Molecular Graphs</h2>

<img width="100%" src="https://raw.githubusercontent.com/rusty1s/himp-gnn/master/overview.png" />

--------------------------------------------------------------------------------

This is a PyTorch implementation of **Hierarchical Inter-Message Passing for Learning on Molecular Graphs**, as described in our paper:

Matthias Fey, Jan-Gin Yuen, Frank Weichert: [Hierarchical Inter-Message Passing for Learning on Molecular Graphs](https://arxiv.org/abs/2006.12179) *(GRL+ 2020)*

## Requirements

* **[PyTorch](https://pytorch.org/get-started/locally/)** (>=1.4.0)
* **[PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric)** (>=1.5.0)
* **[OGB](https://ogb.stanford.edu/)** (>=1.1.0)

## Experiments

Experiments can be run via:

```
$ python train_zinc_subset.py
$ python train_zinc_full.py
$ python train_hiv.py
$ python train_muv.py
$ python train_tox21.py
$ python train_ogbhiv.py
$ python train_ogbpcba.py
```

## Cite

Please cite [our paper](https://arxiv.org/abs/2006.12179) if you use this code in your own work:

```
@inproceedings{Fey/etal/2020,
  title={Hierarchical Inter-Message Passing for Learning on Molecular Graphs},
  author={Fey, M. and Yuen, J. G. and Weichert, F.},
  booktitle={ICML Graph Representation Learning and Beyond (GRL+) Workhop},
  year={2020},
}
```
