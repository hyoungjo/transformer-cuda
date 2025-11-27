# Transformer with CUDA

## Getting Started

The environmental configurations are done on top of [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install).

```sh
# Use an appropriate version for your machine, channel pinned to avoid loose dependencies
conda create -n cuda -c nvidia/label/cuda-12.1.0 -c pytorch -c conda-forge \
    pytorch pytorch-cuda=12.1 cuda-toolkit gxx_linux-64
```
