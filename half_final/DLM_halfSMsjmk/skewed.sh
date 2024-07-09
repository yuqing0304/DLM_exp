#!/bin/bash
#SBATCH --job-name=skewed0
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=skewed0.out

module purge
module load Miniconda3

source activate comm

python train.py