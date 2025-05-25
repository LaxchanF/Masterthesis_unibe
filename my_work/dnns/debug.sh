#!/bin/bash

# You must specify a valid email address!
#SBATCH --mail-user=laxchan.florenceangelo@students.unibe.ch

# Mail on NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-type=fail,end
#SBATCH --job-name="optuna_test"
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --output=optuna_test.txt
#SBATCH --error=optuna_error_test.txt
#SBATCH --partition=gpu
#SBATCH --qos=job_gpu_debug
#SBATCH --gres=gpu:h100:1
#SBATCH --array=1


module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate lax

python lax.optuna.py
