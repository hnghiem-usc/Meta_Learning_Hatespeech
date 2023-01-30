#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --account=<project_id>
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --time=5:00:00

declare -a arr=("trac" "davidson")

for h in "${arr[@]}"; do \
	echo $h
	python run_test.py --config_key MAML_test --holdout_set "$h"
done;

