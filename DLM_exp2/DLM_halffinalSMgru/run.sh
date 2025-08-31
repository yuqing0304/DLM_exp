#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --output=output.out


source activate comm

python train.py