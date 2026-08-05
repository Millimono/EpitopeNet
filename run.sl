#!/bin/bash
#SBATCH --job-name=myPythonjob
#SBATCH --partition=longq  #partition de test limitée a 2h, changer à shortq mediumq ou longq selon durée estimée  
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

export CONDA_ENVS_PATH=/data/shared/groups/IA4Covid/millimono/envs-gpu

source activate my_env_3.7
unset PYTHONPATH
python code_AI_new_architecture.py
