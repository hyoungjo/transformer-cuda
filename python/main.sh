#!/bin/bash

# ------------------------------------------- #
# ------------------ SLURM ------------------ #
# ------------------------------------------- #

# ---------- DEVICE (GPU) CONFIGS ----------- #
#SBATCH --nodes=1
#SBATCH --partition=baram
#SBATCH --gres=gpu:1

# --------------- JOB CONFIGS --------------- #
#SBATCH --job-name=default
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x/python-%j.out
#SBATCH --error=logs/%x/python-%j.err

# ------------ RESOURCE CONFIGS ------------- #
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G


# ------------------------------------------- #
# ---------------- EXECUTION ---------------- #
# ------------------------------------------- #

# 1. Environment
source /home/hyoungjo/miniconda3/etc/profile.d/conda.sh
conda activate cuda

# 2. Run
echo "[SLURM][INFO] Run started on $(hostname) at $(date)"
python3 python/main.py "$@"
echo "[SLURM][INFO] Run finished at $(date)"
