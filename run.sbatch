#!/bin/bash -l

#SBATCH --job-name=halu
#SBATCH --account=domainacs
#SBATCH --partition=tier3
#SBATCH --time=23:59:00
#
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=slack:@gs3260
#SBATCH --mail-type=ALL
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=a100:1
#
#SBATCH --mem=20g

source ~/miniconda3/etc/profile.d/conda.sh
conda activate halu
sh run.sh
