#!/bin/bash
#SBATCH --job-name=CNN_regression_train      # Job name (change as needed)
#SBATCH --output=logs/regression_train_%j.out   # Standard output file (%j = job ID)
#SBATCH --error=logs/regression_train_%j.err    # Standard error file
#SBATCH --time=12:00:00                  # Time limit (HH:MM:SS)
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks per node
#SBATCH --cpus-per-task=4                # Number of CPU cores per task
#SBATCH --mem=16G                        # Memory per node

# Load necessary modules (e.g., for Python environment)
module load anaconda3/2023.09

# Activate a Conda environment
source /home/skumar/miniconda3/bin/activate /home/skumar/miniconda3/envs/alfa

export CUDA_VISIBLE_DEVICES=0

mkdir -p logs

# Execute your Python script
python /home/skumar/Git/SCR-Progression/CNN-Model/CNN_pytorch.py