#!/bin/bash
#SBATCH --job-name=1027_sparse_v2_two_points_[space_7]_0_75
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --partition=monakhova
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-high
#SBATCH -N 1
#SBATCH --ntasks=16
#SBATCH --time=4-8:00:00
#SBATCH --mem=64G

set -euo pipefail

mkdir -p slurm_logs

# Use your current shell setup
source ~/.bashrc

# Mirror your `train()` function environment
conda activate mevent
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PREPROCESSED_DATASETS_FOLDER="/home/yl3836/DENSE/"
ulimit -n 4096

# Go to your project directory
cd ~/mono_event/rpg_e2depth/e2depth

# Allow overriding the config path via $CONFIG or $1
CONFIG_PATH="${CONFIG:-${1:-configs/example_psf_off_brownian.json}}"

echo "Starting training with config: ${CONFIG_PATH}"
python train.py -c "${CONFIG_PATH}"
