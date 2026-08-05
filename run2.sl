#!/bin/bash
#SBATCH --job-name=myGPUjob
#SBATCH -N 1
#SBATCH --partition=gpu-prodq #partition de test limitée a 2h, changer à gpu-prodq 
#SBATCH --gres=gpu:2
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

module load cuda10.1/toolkit/10.1.243
module load cudnn
export CONDA_ENVS_PATH=envsf-gpu 
module load anaconda3
source activate my_envf_3.7
unset PYTHONPATH
python code_AI_new_architecture.py
