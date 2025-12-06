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
#SBATCH --job-name=transformer-cpp
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# The time limit has been set to 3 hours, as the job takes about 2 hours.

# ------------ RESOURCE CONFIGS ------------- #
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# The code runs with a single thread (for now), and memory with 6 ~ 7GB.
# The resource configs are to ensure the job runs with maximum capabilities.

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
./cpp/main
echo "[SLURM][INFO] Run finished at $(date)"
