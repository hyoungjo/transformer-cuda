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
#SBATCH --job-name=transformer-cpp-timer
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ------------ RESOURCE CONFIGS ------------- #
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

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
./cpp/timer
echo "[SLURM][INFO] Run finished at $(date)"
