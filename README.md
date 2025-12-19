# Transformer with CUDA

## Getting Started

The environmental configurations are done on top of [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install).

```sh
# Use an appropriate version for your machine, channel pinned to avoid loose dependencies
conda create -n cuda --override-channels \
    -c nvidia/label/cuda-12.1.0 -c pytorch -c conda-forge \
    pytorch pytorch-cuda=12.1 cuda-toolkit gxx_linux-64
```

```sh
conda activate cuda
pip3 install numpy transformers
```

### Python

```sh
sbatch python/main.sh
sbatch python/timer.sh
```

### C++

```sh
sbatch cpp/main.sh
sbatch cpp/timer.sh
```

### CUDA

```sh
sbatch cuda/main.sh
sbatch cuda/timer.sh
```
