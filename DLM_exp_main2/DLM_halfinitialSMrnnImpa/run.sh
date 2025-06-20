#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=04:00:00
#SBATCH --output=out.out


source activate comm

python train.py