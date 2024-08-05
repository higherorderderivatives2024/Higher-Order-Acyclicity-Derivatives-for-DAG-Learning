# $\beta$-th order Acyclicity Derivatives for DAG Learning

This repository is the official implementation of $\beta$-th order Acyclicity Derivatives for DAG Learning.

## Requirements
- Python 3.6+
- numpy
- scipy
- python-igraph
- scikit-learn
- copy
- random
- time
- torch: only for neural network score function
- networkx: only for neural network score function

## Contents

- betaLS.py: implementation of $\beta$-LS for least squares loss.
- 2LSopt.py: implementation of $2$-LSopt for least squares loss.
- 2LSopt_NN.py: implementation of $2$-LSopt for neural network score function.

## Acknowledgements

We thank the authors of the [Optimizing NOTEARS Objectives via Topological Swaps repo](https://github.com/Duntrain/TOPO) for making their code available, whose code we have adapted in our implementation.
