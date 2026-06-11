#!/bin/bash
#SBATCH --job-name=k2141b_lam
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=wzhctdnormal
#SBATCH -o results/slurm_lam_%j.out
#SBATCH -e results/slurm_lam_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-/work/home/haolinli/project/Nested_Sampling}"
if [ ! -d "${PROJECT_DIR}/nested_sampler" ]; then
    PROJECT_DIR="/work/home/haolinli/project/Nested_Sampling"
fi
cd "${PROJECT_DIR}"
# mkdir -p results
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

if [ ! -d "${PROJECT_DIR}/nested_sampler" ]; then
    echo "ERROR: nested_sampler package not found under ${PROJECT_DIR}" >&2
    exit 2
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

. /public/software/apps/anaconda3/5.2.0/etc/profile.d/conda.sh
module load apps/anaconda3/5.2.0
conda activate MCMC

WORKERS="${SLURM_CPUS_PER_TASK:-32}"
if [ "${WORKERS}" -gt 32 ]; then
    WORKERS=32
fi

python - <<'PY'
import os
import sys
import dynesty
print("dynesty", dynesty.__version__)
print("cwd", os.getcwd())
print("python", sys.executable)
PY

python -m nested_sampler.run \
    --model lambert_limb \
    --workers "${WORKERS}" \
    --nlive-init 500 \
    --nlive-batch 250 \
    --dlogz-init 0.1 \
    --sample rwalk \
    --bound multi \
    --outdir "results/server_${SLURM_JOB_ID:-manual}_lambert_limb"

echo DONE
