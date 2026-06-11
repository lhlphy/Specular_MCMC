#!/bin/bash
#SBATCH --job-name=mcmc_submit_scripts
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=wzhctdnormal

set -euo pipefail

write_submit_script() {
    local path="$1"
    local job_name="$2"
    local main_path="$3"

    cat > "${path}" <<SH
#!/bin/bash
#SBATCH --job-name=${job_name}
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

python ${main_path}

echo DONE
SH
}

update_repo() {
    local repo="$1"
    local label="$2"

    mkdir -p "${repo}/log"
    write_submit_script "${repo}/run.sh" "${label}_limb" "core_limb/main.py"
    write_submit_script "${repo}/run_lam.sh" "${label}_lam_limb" "core_lambert_limb/main.py"
    write_submit_script "${repo}/run_atm.sh" "${label}_atm_limb" "core_atm_limb/main.py"
    chmod +x "${repo}/run.sh" "${repo}/run_lam.sh" "${repo}/run_atm.sh"

    echo "Updated submit scripts in ${repo}"
    grep -H -E "#SBATCH --nodes|#SBATCH --ntasks-per-node|python " \
        "${repo}/run.sh" "${repo}/run_lam.sh" "${repo}/run_atm.sh"
}

update_repo "/work/home/haolinli/project/MCMC_Kepler10b_new" "K10b"
update_repo "/work/home/haolinli/project/MCMC_Kepler78b_new" "K78b"

echo "DONE"
