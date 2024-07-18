#!/bin/bash
#SBATCH -J Halu
#SBATCH -c 8
#SBATCH -t 32:00:00
#SBATCH -p gpu_dia
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-user=george.sun.academic@gmail.com

module purge
module load dgx
module load miniconda3/24.1.2
module load cuda12.1/blas/12.1.1

cd /n/lw_groups/hms/dbmi/yu/lab/jz290/hallucination-agent
sh run.sh