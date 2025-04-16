import numpy as np
import pandas as pd
import os
# Constants List
AU = 149_597_870.7  # km, 1 Astronomical Unit 149597870.7
Sigma_const = 5.67e-8  # W/m^2/K^4, Stefan-Boltzmann constant

class Planet_parameters:
    def __init__(self, Nline):
        data_base = pd.read_csv('PS.csv', header = 96)
        row_data = data_base.iloc[Nline]
        ### orbital parameters
        print("Target name: ", row_data['pl_name'])
        self.Rs = row_data['st_rad'] * 696340  # km, radius of the Star
        # self.Rp = row_data['pl_rade'] * 6371.4  # km, radius of the Planet
        self.eccentricity = 0 # row_data['pl_orbeccen'] # Eccentricity of the planet's orbit
        # self.semi_axis = row_data['pl_orbsmax'] * AU  # km, semi-major axis of the planet's orbit
        self.Rp2Rs = 0.02037
        self.a2Rs = 2.292
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
        # self.Coefficents = [0.619, 0.106] # K2-141 b
        self.Coefficents = [0.666, 0.062] # K2-141 b
        # self.Coefficents = [0.403, 0.256] # Kepler-10b
        print("Reference: ", row_data['pl_refname'])


PPs = Planet_parameters(4266 - 98) # K2-141 b Mala
# PPs = Planet_parameters(733 - 98) # GJ 367b
# PPs = Planet_parameters(6427 - 98) # Kepler-10 b
# PPs = Planet_parameters(6432 - 98) # Kepler-10 b
# PPs = Planet_parameters(30080 - 98) # Kepler-78 b
    