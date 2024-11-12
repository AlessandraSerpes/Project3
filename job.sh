#!/usr/bin/env bash
#SBATCH -A cs525
#SBATCH -p academic
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -C A30
#SBATCH -t 48:00:00
#SBATCH --mem 24G
#SBATCH --job-name="P3"

source activate RLProject
srun --unbuffered python main.py --train_dqn