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
python3 train_dpr.py --query_model huawei-noah/TinyBERT_General_4L_312D --passage_model bert-large-uncased