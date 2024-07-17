#!/bin/bash
#SBATCH -c 8
#SBATCH -t 32:00:00
#SBATCH -p gpu_yu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --account=yu_ky98_contrib


module load gcc/6.2.0
module load cuda/12.1
cd /n/scratch/users/j/jz290/hallucination-agent

sh run.sh