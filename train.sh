#!/bin/bash -l

#SBATCH --gres=gpu:4
#SBATCH --partition=tesla
#SBATCH -N 1
#SBATCH --mem 64000
#SBATCH --time=100:00:00
#SBATCH -c 8
#SBATCH -o ./slurm/output.%A.out # STDOUT

WANDB_API_KEY= python3 train.py