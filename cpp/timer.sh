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
#SBATCH --time=480:00:00
#SBATCH --output=logs/%x/cpp-timer-%j.out
#SBATCH --error=logs/%x/cpp-timer-%j.err

# ------------ RESOURCE CONFIGS ------------- #
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G

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
./cpp/timer "$@"
echo "[SLURM][INFO] Run finished at $(date)"
