#!/bin/bash
#SBATCH --job-name=mcmc_cmp1000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --partition=wzhctdnormal
#SBATCH -o log/%j.compare1000.out
#SBATCH -e log/%j.compare1000.err

set -euo pipefail

module load apps/anaconda3/2023.09
. /public/software/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate MCMC

cd /work/home/haolinli/project/MCMC

export MPLBACKEND=Agg
export MCMC_COMPARE_MODES=map,mle
export MCMC_COMPARE_MAX_CANDIDATES=1000
export MCMC_COMPARE_RANDOM_SEED=20260606
export MCMC_COMPARE_SCORE_WORKERS=32

python codex_compare_scripts/Model_compare.py
