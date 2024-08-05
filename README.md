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

## Contents

- betaLS.py: implementation of $\beta$-LS for least squares loss.
- 2LSopt.py: implementation of $2$-LSopt for least squares loss.

## Running $\beta$-LS and $2#-LSopt

Use requirements.txt to install the dependencies. The simplest way to try out $\beta$-LS is as follows:
```
$ git clone https://github.com/higherorderderivatives2024/Higher-Order-Acyclicity-Derivatives-for-DAG-Learning.git
$ cd Higher-Order-Acyclicity-Derivatives-for-DAG-Learning/
$ pip3 install -r requirements.txt
$ python3 beta_LS.py
```
Similarly,  the simplest way to try out $2$-LSopt is as follows:
```
$ git clone https://github.com/higherorderderivatives2024/Higher-Order-Acyclicity-Derivatives-for-DAG-Learning.git
$ cd Higher-Order-Acyclicity-Derivatives-for-DAG-Learning/
$ pip3 install -r requirements.txt
$ python3 2_LSopt.py
```

## Acknowledgements

We thank the authors of the [Optimizing NOTEARS Objectives via Topological Swaps repo](https://github.com/Duntrain/TOPO) for making their code available, whose code we have adapted in our implementation.
