#!/bin/bash -l
#SBATCH --output slurm_logs/paper_shredderer.out
#SBATCH --error slurm_logs/paper_shredderer_error.out
#SBATCH --job-name run_notebooks
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#
# Wall clock limit (max is 24 hours):
#SBATCH --time=23:00:00

module load anaconda/3/2021.11
source ~/.condasetup_bash
conda activate sh_finetuning

srun bash -c scripts/paper_shredder.bash