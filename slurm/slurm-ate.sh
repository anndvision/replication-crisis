#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --mem=16gb
#SBATCH --job-name="ate-test"
#SBATCH --output=logs/job-%j.out
#SBATCH --array=0-99
 
source /burg/stats/users/$USER/miniconda3/bin/activate ml-field-experiments

srun "$@" --seed $SLURM_ARRAY_TASK_ID
