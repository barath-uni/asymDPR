#!/bin/bash

#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=Trainmetatitan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=1:00:00
#SBATCH --mem=32000M
#SBATCH --output=downloaddatajob.out

module purge
module load 2021
module load Anaconda3/2021.05

cd $HOME/asymdpr/asymDPR
sh downloaddata.sh

