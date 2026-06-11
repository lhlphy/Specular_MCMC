import numpy as np
import pandas as pd
import os

# Constants List
AU = 149_597_870.7  # km, 1 Astronomical Unit 149597870.7
Sigma_const = 5.67e-8  # W/m^2/K^4, Stefan-Boltzmann constant


PLANET_PRESETS = {
    "K2-141b": {
        "row": 4266 - 98,
        "coefficients": [0.666, 0.062],
        "rp2rs": 0.02037,
        "a2rs": 2.292,
    },
    "Kepler-10b": {
        "row": 6427 - 98,
        "coefficients": [0.403, 0.256],
    },
    "Kepler-78b": {
        "row": 30080 - 98,
        "coefficients": [0.519, 0.178],
    },
}


def selected_planet_preset():
    target = os.environ.get("MCMC_PARAMETER_TARGET", "K2-141b")
    if target not in PLANET_PRESETS:
        valid = ", ".join(sorted(PLANET_PRESETS))
        raise ValueError(f"Unknown MCMC_PARAMETER_TARGET={target!r}. Expected one of: {valid}")
    return target, PLANET_PRESETS[target]


class Planet_parameters:
    def __init__(self, Nline, coefficients=None, rp2rs=None, a2rs=None):
        data_base = pd.read_csv('PS.csv', header = 96)
        row_data = data_base.iloc[Nline]
        ### orbital parameters
        print("Target name: ", row_data['pl_name'])
        self.Rs = row_data['st_rad'] * 696340  # km, radius of the Star
        self.Rp = row_data['pl_rade'] * 6371.4  # km, radius of the Planet
        self.eccentricity = 0 # row_data['pl_orbeccen'] # Eccentricity of the planet's orbit
        self.semi_axis = row_data['pl_orbsmax'] * AU  # km, semi-major axis of the planet's orbit
        self.Rp2Rs = rp2rs if rp2rs is not None else self.Rp / self.Rs
        self.a2Rs = a2rs if a2rs is not None else self.semi_axis / self.Rs
        self.alpha = np.arcsin(1/self.a2Rs)
        ### Thermal and optical parameters
        self.Stellar_T = row_data['st_teff'] # K, temperature of the Star
        # self.pl_eqT = row_data['pl_eqt']  # K, fully redistribution, planet equilibrium Temperature [K] (from database)
        self.pl_eqT = self.Stellar_T * np.sqrt(1/self.a2Rs / 2)  # from theoretical calculation
        self.Period = row_data['pl_orbper'] *24 # hours, orbital period of the planet
        self.Mp_J = row_data['pl_bmassj'] # mass of the planet/Jupiter mass
        self.Ms_S = row_data['st_mass'] # mass of the star/Solar mass
        self.Rs_S = row_data['st_rad'] # radius of the star/Solar radius
        self.Tss = self.Stellar_T / np.sqrt(self.a2Rs)
        print("alpha: ", self.alpha)
        print("Tss: ", self.Tss)
        self.Coefficents = coefficients
        print("Reference: ", row_data['pl_refname'])


_TARGET_NAME, _PRESET = selected_planet_preset()
PPs = Planet_parameters(
    _PRESET["row"],
    coefficients=_PRESET["coefficients"],
    rp2rs=_PRESET.get("rp2rs"),
    a2rs=_PRESET.get("a2rs"),
) # selected by MCMC_PARAMETER_TARGET
# PPs = Planet_parameters(733 - 98) # GJ 367b
# PPs = Planet_parameters(6427 - 98) # Kepler-10 b
# PPs = Planet_parameters(6432 - 98) # Kepler-10 b
# PPs = Planet_parameters(30080 - 98) # Kepler-78 b
