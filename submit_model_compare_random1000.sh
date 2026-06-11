#!/bin/bash
set -euo pipefail

cd /work/home/haolinli/project/MCMC
sbatch codex_compare_scripts/run_model_compare_random1000_ntasks32.sh
