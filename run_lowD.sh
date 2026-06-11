#!/bin/bash
#SBATCH --job-name=K2_141b_lowD
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --partition=wzhctdnormal
#SBATCH -o log/%j.loop
#SBATCH -e log/%j.loop

. /public/software/apps/anaconda3/5.2.0/etc/profile.d/conda.sh

module load apps/anaconda3/5.2.0
conda activate MCMC
echo MCMC

python core_lowD/main.py

echo DONE
