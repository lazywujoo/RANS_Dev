#!/bin/bash
#SBATCH -t 08:00:00
#SBATCH -N 1

cd $SLURM_SUBMIT_DIR
conda activate cupy-env
python test.py > output.txt 
