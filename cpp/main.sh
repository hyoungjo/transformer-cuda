#!/bin/bash

# ------------------------------------------- #
# ------------------ SLURM ------------------ #
# ------------------------------------------- #

# ---------- DEVICE (GPU) CONFIGS ----------- #
# #SBATCH --nodes=1
# #SBATCH --partition=baram
# #SBATCH --gres=gpu:1

# Do not request GPU for CPU runs.

# --------------- JOB CONFIGS --------------- #
#SBATCH --job-name=default
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x/cpp-%j.out
#SBATCH --error=logs/%x/cpp-%j.err

# The job takes about 2 hours for GPT-2 and ~72 hours for LLaMA-3-8B.
# Due to the time limits of sbatch, C++ version is only run for GPT-2.

# ------------ RESOURCE CONFIGS ------------- #
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G

# The code runs on a single thread and uses 6 ~ 7GB of memory for GPT-2 and 32 ~ 48GB for LLaMA-3-8B.
# The resource configs are to ensure the job runs with maximum capabilities.

# GPT-2 weights hold 124M * 4B ~= 500MB, but CUDA contexts, PyTorch Caching Allocator, .. etc takes up the rest.
# LLaMA-3-8B is a much larger model. The weights hold 8G * 4B ~= 32GB and KV caches become a large source of additional memory usage.
#  - 2 x 32 (layers) x 8 (num_heads) x 128 (head_dim) x 4 (bytes) x 8,192 (max. seq_len) ~= 2,147,483,648 ~= 2GB
#  - While the full context length is 128K for LLaMA-3-8B, 8K is the standard context length.

# ------------------------------------------- #
# ---------------- EXECUTION ---------------- #
# ------------------------------------------- #

# 1. Environment
source /home/hyoungjo/miniconda3/etc/profile.d/conda.sh
conda activate cuda

# 2. Build
make cpp

# 3. Run
echo "[SLURM][INFO] Run started on $(hostname) at $(date)"
./cpp/main "$@"
echo "[SLURM][INFO] Run finished at $(date)"
