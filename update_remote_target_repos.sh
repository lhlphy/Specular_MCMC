#!/bin/bash
#SBATCH --job-name=mcmc_target_update
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=wzhcnormal

set -euo pipefail

PROJECT=/work/home/haolinli/project
UPDATED="${PROJECT}/MCMC"

write_parameters() {
    local repo="$1"
    local coeffs="$2"
    local pps_expr="$3"
    local pps_label="$4"

    cat > "${repo}/parameters.py" <<PY
import numpy as np
import pandas as pd
import os

AU = 149_597_870.7
Sigma_const = 5.67e-8


class Planet_parameters:
    def __init__(self, Nline):
        data_base = pd.read_csv('PS.csv', header=96)
        row_data = data_base.iloc[Nline]

        print("Target name: ", row_data['pl_name'])
        self.Rs = row_data['st_rad'] * 696340
        self.Rp = row_data['pl_rade'] * 6371.4
        self.eccentricity = 0
        self.semi_axis = row_data['pl_orbsmax'] * AU

        self.Stellar_T = row_data['st_teff']
        self.pl_eqT = self.Stellar_T * np.sqrt(self.Rs / 2 / self.semi_axis)
        self.Period = row_data['pl_orbper'] * 24
        self.Mp_J = row_data['pl_bmassj']
        self.Ms_S = row_data['st_mass']
        self.Rs_S = row_data['st_rad']
        self.Tss = self.Stellar_T / np.sqrt(self.semi_axis / self.Rs)
        self.Rp2Rs = self.Rp / self.Rs
        self.alpha = np.arcsin(self.Rs / self.semi_axis)
        print("alpha: ", self.Rs / self.semi_axis)
        print("Tss: ", self.Tss)
        self.Coefficents = ${coeffs}


PPs = Planet_parameters(${pps_expr}) # ${pps_label}
PY
}

write_main() {
    local path="$1"
    local target="$2"
    local sigma="$3"
    local ndim="$4"
    local nsteps="$5"
    local burnin="$6"

    cat > "${path}" <<PY
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

from Class_MCMC import MCMC


if __name__ == '__main__':
    mcmc = MCMC('${target}', 'Kepler', sigma=${sigma}, ndim=${ndim}, nwalkers=64, nsteps=${nsteps}, burnin=${burnin})
    mcmc.sample()
    mcmc.plot_trace()
    mcmc.plot_corner()
    mcmc.compute_rhat()
    mcmc.estimate_parameters()
    mcmc.plot_fit()

    samples = mcmc.load_samples()
    print("sample shape:", samples.shape)
PY
}

ensure_target_copy() {
    local repo="$1"
    local src_name="$2"
    local dst_name="$3"
    local fallback_name="$4"

    local target_root="${repo}/Target"
    if [ -d "${target_root}/${dst_name}" ]; then
        return
    fi

    if [ -d "${target_root}/${src_name}" ]; then
        cp -a "${target_root}/${src_name}" "${target_root}/${dst_name}"
    else
        cp -a "${target_root}/${fallback_name}" "${target_root}/${dst_name}"
    fi
}

update_one() {
    local src="$1"
    local dst="$2"
    local base_target="$3"
    local sigma="$4"
    local coeffs="$5"
    local pps_expr="$6"
    local pps_label="$7"

    if [ -e "${dst}" ]; then
        echo "Refusing to overwrite existing ${dst}" >&2
        exit 2
    fi

    cp -a "${src}" "${dst}"

    for item in core core_atm core_lambert core_limb core_atm_limb core_lambert_limb; do
        rm -rf "${dst}/${item}"
        cp -a "${UPDATED}/${item}" "${dst}/${item}"
    done

    for item in Sampling.py color.py Model_compare.py ABS_compare.py plot_paper.py plot_TOI2431b.py uncertainty.py check_list.txt README.md run.sh; do
        if [ -e "${UPDATED}/${item}" ]; then
            cp -a "${UPDATED}/${item}" "${dst}/${item}"
        fi
    done

    write_parameters "${dst}" "${coeffs}" "${pps_expr}" "${pps_label}"

    ensure_target_copy "${dst}" "${base_target}" "${base_target}_limb" "${base_target}"
    ensure_target_copy "${dst}" "${base_target}_lambert" "${base_target}_lambert_limb" "${base_target}"
    ensure_target_copy "${dst}" "${base_target}_atm" "${base_target}_atm" "${base_target}"
    ensure_target_copy "${dst}" "${base_target}_atm" "${base_target}_atm_limb" "${base_target}"

    write_main "${dst}/core/main.py" "${base_target}" "${sigma}" 7 4000 1500
    write_main "${dst}/core_limb/main.py" "${base_target}_limb" "${sigma}" 9 4000 1500
    write_main "${dst}/core_lambert/main.py" "${base_target}_lambert" "${sigma}" 7 4000 1500
    write_main "${dst}/core_lambert_limb/main.py" "${base_target}_lambert_limb" "${sigma}" 9 4000 1500
    write_main "${dst}/core_atm/main.py" "${base_target}_atm" "${sigma}" 8 5000 2500
    write_main "${dst}/core_atm_limb/main.py" "${base_target}_atm_limb" "${sigma}" 10 5000 2500

    echo "Created ${dst}"
    grep -E "self.Coefficents|PPs = Planet_parameters" "${dst}/parameters.py"
    grep -E "MCMC\\('" "${dst}/core/main.py" "${dst}/core_limb/main.py" "${dst}/core_atm/main.py" "${dst}/core_atm_limb/main.py" "${dst}/core_lambert/main.py" "${dst}/core_lambert_limb/main.py"
}

update_one "${PROJECT}/MCMC_Kepler10b" "${PROJECT}/MCMC_Kepler10b_new" "Kepler-10b" 2.5 "[0.403, 0.256]" "6427 - 98" "Kepler-10 b"
update_one "${PROJECT}/MCMC_Kepler78b" "${PROJECT}/MCMC_Kepler78b_new" "Kepler-78b" 3.0 "[0.519, 0.178]" "30080 - 98" "Kepler-78 b"

echo "DONE"
