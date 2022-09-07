# Exact Decomposition of Quantum Channels for Non-IID Quantum Federated Learning

This repo contains the code for [this paper](https://arxiv.org/abs/2209.00768). All experiments are implemented in [JAX](https://github.com/google/jax) and [TensorCircuit](https://github.com/tencent-quantum-lab/tensorcircuit).

# Abstract

Federated learning refers to the task of performing machine learning with decentralized data from multiple clients while protecting data security and privacy. Works have been done to incorporate quantum advantage in such scenarios. However, when the clients' data are not independent and identically distributed (IID), the performance of conventional federated algorithms deteriorates. In this work, we explore this phenomenon in the quantum regime with both theoretical and numerical analysis. We further prove that a global quantum channel can be exactly decomposed into channels trained by each client with the help of local density estimators. It leads to a general framework for quantum federated learning on non-IID data with one-shot communication complexity. We demonstrate it on classification tasks with numerical simulations.

# Files
Files with prefix `centralized`, `qFedAvg` and `qFedInf` implement the benchmark with centalized data, the quantum federated averaging algorithm and the quantum federated inference algorithm respectively. The jupyter notebooks are for illustration and the python scripts are for mass production.  `plot.ipynb` reproduces the plots in the paper.

# Citation
If you find our work useful, please give us credit by citing our paper:

```bibtex
@misc{zhao2022qfedinf,
    title={{Exact Decomposition of Quantum Channels for Non-IID Quantum Federated Learning}}, 
    author={Haimeng Zhao},
    year={2022},
    eprint={2209.00768},
    archivePrefix={arXiv},
    primaryClass={quant-ph},
}
```
