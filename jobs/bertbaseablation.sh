#!/bin/bash

#SBATCH --partition=gpu_titanrtx_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=TrainDPRBaseline
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=20:00:00
#SBATCH --mem=32000M
#SBATCH --output=trainDPRBaseline.out

module purge
module load 2021
module load Anaconda3/2021.05

cd $HOME/asymdpr/asymDPR
eval "$(conda shell.bash hook)"
conda activate asymdpr
# Experiment with static embedding. Removing all 12 layers
python3 train_dpr.py --ablation True --query_layers 12

# Test by removing 8 layers
python3 train_dpr.py --ablation True --query_layers 12

# For by removing 4 layers
python3 train_dpr.py --ablation True --query_layers 12


