#!/bin/bash
#SBATCH --job-name=epitopenet
#SBATCH -N 1
#SBATCH --partition=gpu-testq
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=gpu_users
#SBATCH --qos=gpu
#SBATCH -o /data/shared/groups/IA4Covid/millimono/epitonet/results/epitopenet-%j.out
#SBATCH -e /data/shared/groups/IA4Covid/millimono/epitonet/results/epitopenet-%j.err

module load pytorch-py39-cuda11.2-gcc9/1.9.1
module load cuda11.2/toolkit/11.2.2
module load cudnn8.1-cuda11.2/8.1.1.33

cd "/data/shared/groups/IA4Covid/millimono/epitonet/Architecture PopulationB"

python run_experiment_cli.py \
    --mode seed \
    --seed 42 \
    --epochs 5 \
    --patch 18 \
    --theta 0.2 \
    --lr 0.001 \
    --num_cells 2133 \
    --K 1 \
    --intensity false \
    --name test_hpc_seed42