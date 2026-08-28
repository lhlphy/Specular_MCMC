#!/bin/bash
#SBATCH --job-name=K78b_limb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --partition=wzhcnormal
#SBATCH -o log/%j.loop
#SBATCH -e log/%j.loop

. /public/software/apps/anaconda3/5.2.0/etc/profile.d/conda.sh

module load apps/anaconda3/5.2.0
conda activate MCMC
echo MCMC

export MCMC_RESUME=1
export MCMC_EXTRA_STEPS=10000
# unset MCMC_EXTRA_STEPS
export MCMC_OBSERVATION_NAME=Kepler-78b_savgol
echo "Start a fresh MCMC chain"
echo "Observation data: ${MCMC_OBSERVATION_NAME}.txt"

python core_limb/main.py

echo DONE
