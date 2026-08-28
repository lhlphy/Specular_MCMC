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
        self.Coefficents = [0.519, 0.178]


PPs = Planet_parameters(30080 - 98) # Kepler-78 b
