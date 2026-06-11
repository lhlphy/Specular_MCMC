import numpy as np


AU = 149_597_870.7
Sigma_const = 5.67e-8


class PlanetParameters:
    def __init__(self):
        self.Rs = 0.651 * 696340
        self.eccentricity = 0
        self.Rp2Rs = 0.02133
        self.a2Rs = 2.00079
        self.alpha = np.arcsin(1.0 / self.a2Rs)
        self.Stellar_T = 4109
        self.pl_eqT = self.Stellar_T * np.sqrt(1.0 / self.a2Rs / 2.0)
        self.Period = 0.2236 * 24
        self.Mp_J = 0.0195031
        self.Ms_S = 0.661
        self.Rs_S = 0.651
        self.Tss = self.Stellar_T / np.sqrt(self.a2Rs)
        self.inc = 74
        self.Coefficents = [0.229, 0.225]


PPs = PlanetParameters()
