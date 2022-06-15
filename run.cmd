#!/bin/bash
#SBATCH -t 08:00:00
#SBATCH -N 1
##SBATCH -p hsw_t4

cd $SLURM_SUBMIT_DIR
nvidia-smi
source ../.bashrc
conda init bash
conda activate cupy-env
python test.py > output2.txt 
