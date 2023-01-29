# Non-IID Quantum Federated Learning with One-shot Communication Complexity

This repo contains the code for [this paper](https://link.springer.com/article/10.1007/s42484-022-00091-z) [(arXiv)](https://arxiv.org/abs/2209.00768). All experiments are implemented in [JAX](https://github.com/google/jax) and [TensorCircuit](https://github.com/tencent-quantum-lab/tensorcircuit).

# Abstract

Federated learning refers to the task of machine learning based on decentralized data from multiple clients with secured data privacy. Recent studies show that quantum algorithms can be exploited to boost its performance. However, when the clients’ data are not independent and identically distributed (IID), the performance of conventional federated algorithms is known to deteriorate. In this work, we explore the non-IID issue in quantum federated learning with both theoretical and numerical analysis. We further prove that a global quantum channel can be exactly decomposed into local channels trained by each client with the help of local density estimators. This observation leads to a general framework for quantum federated learning on non-IID data with one-shot communication complexity. Numerical simulations show that the proposed algorithm outperforms the conventional ones significantly under non-IID settings.

# Files
Files with prefix `centralized`, `qFedAvg` and `qFedInf` implement the benchmark with centalized data, the quantum federated averaging algorithm and the quantum federated inference algorithm respectively. The jupyter notebooks are for illustration and the python scripts are for mass production.  `plot.ipynb` reproduces the plots in the paper.

# Citation
If you find our work useful, please give us credit by citing our paper:

```bibtex
@article{zhao2023qfedinf,
  title={Non-IID quantum federated learning with one-shot communication complexity},
  author={Zhao, Haimeng},
  journal={Quantum Machine Intelligence},
  volume={5},
  number={1},
  pages={3},
  year={2023},
  publisher={Springer}
}
```
