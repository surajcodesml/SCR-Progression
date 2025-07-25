#!/bin/bash
#SBATCH --job-name=cnn_layer_regression
#SBATCH --output=logs/cnn_train_%j.out
#SBATCH --error=logs/cnn_train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Set CUDA to use GPU 0
export CUDA_VISIBLE_DEVICES=0

# Activate your conda environment
source /home/skumar/miniconda3/etc/profile.d/conda.sh
conda activate vision

# Change to your project directory
cd /home/skumar/Git/SCR-Progression/Swin-Regression-Model

# Run your script
python CNN_train.py