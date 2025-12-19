#!/bin/bash

# ------------------------------------------- #
# ------------------ SLURM ------------------ #
# ------------------------------------------- #

# ---------- DEVICE (GPU) CONFIGS ----------- #
#SBATCH --nodes=1
#SBATCH --partition=baram
#SBATCH --gres=gpu:1

# --------------- JOB CONFIGS --------------- #
#SBATCH --job-name=transformer-cuda-timer
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ------------ RESOURCE CONFIGS ------------- #
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G


# ------------------------------------------- #
# ---------------- EXECUTION ---------------- #
# ------------------------------------------- #

# 1. Environment
source /home/hyoungjo/miniconda3/etc/profile.d/conda.sh
conda activate cuda

# 2. Build
make cuda

# 3. Run
echo "[SLURM][INFO] Run started on $(hostname) at $(date)"
nsys profile --stats=true --force-overwrite true -o profiles/report ./cuda/timer
echo "[SLURM][INFO] Run finished at $(date)"
