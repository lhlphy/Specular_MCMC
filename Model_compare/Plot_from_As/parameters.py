from pathlib import Path

import numpy as np


AU = 149_597_870.7  # km
Sigma_const = 5.67e-8  # W/m^2/K^4


FALLBACK_K2_141_B = {
    "pl_name": "K2-141 b",
    "st_rad": 0.67,
    "pl_rade": 1.54,
    "pl_orbsmax": 0.00716,
    "st_teff": 4373.0,
    "pl_orbper": 0.2803226,
    "pl_bmassj": 0.01671,
    "st_mass": 0.66,
}


def _load_planet_row(nline):
    try:
        import pandas as pd
    except ImportError:
        return FALLBACK_K2_141_B

    module_dir = Path(__file__).resolve().parent
    candidate_paths = [
        module_dir / "PS.csv",
        module_dir.parent / "PS.csv",
    ]
    for csv_path in candidate_paths:
        if csv_path.exists():
            data_base = pd.read_csv(csv_path, header=96)
            return data_base.iloc[nline]

    return FALLBACK_K2_141_B


class Planet_parameters:
    def __init__(self, nline):
        row_data = _load_planet_row(nline)
        self.name = row_data["pl_name"]
        self.Rs = row_data["st_rad"] * 696340
        self.Rp = row_data["pl_rade"] * 6371.4
        self.eccentricity = 0
        self.semi_axis = row_data["pl_orbsmax"] * AU
        self.Stellar_T = row_data["st_teff"]
        self.pl_eqT = self.Stellar_T * np.sqrt(self.Rs / 2 / self.semi_axis)
        self.Period = row_data["pl_orbper"]
        self.Mp_J = row_data["pl_bmassj"]
        self.Ms_S = row_data["st_mass"]
        self.Rs_S = row_data["st_rad"]
        self.Tss = self.Stellar_T / np.sqrt(self.semi_axis / self.Rs)
        self.Rp2Rs = self.Rp / self.Rs
        self.alpha = np.arcsin(self.Rs / self.semi_axis)


PPs = Planet_parameters(4170)
