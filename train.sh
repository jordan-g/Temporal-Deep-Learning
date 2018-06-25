#!/usr/bin/env bash
#SBATCH --time=0-12
#SBATCH --account=def-tyrell
#SBATCH --gres=gpu:1
#SBATCH --job-name=Temporal-Deep-Learning
#SBATCH --mem=8000M
#SBATCH --output=Logs/%j.out

module load miniconda3
source activate multiplexing
python $HOME/projects/jordang/Multiplexing/train.py $SLURM_JOB_ID -info $1 -n_layers $2 -n_units $3 -W_range $4 -Z_range $5 -Y_range $6 -f_etas $7 -r_etas $8 -b_etas $9 -desired_u ${10} -hard_m ${11} -hard_v ${12}
source deactivate
