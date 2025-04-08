import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from analytical_comb import *
import time
from parameters import PPs
from Class_MCMC import MCMC
import cProfile
import pstats
import io


# test2
if __name__ == '__main__':
    
    t1 = time.time()
    print('test')
    # import MCMC class
    mcmc = MCMC('Kepler-10b', 'Kepler', sigma=2.5, ndim=7, nwalkers=64, nsteps=2000, burnin=1000)
    # set parameters
    Theta_array = np.linspace(0, 2*np.pi, 70)
    offset = 0
    AB =  0.0475
    alpha_ellipse = 3.1739
    alpha_Doppler = 0
    F = 0
    Tss = PPs.Tss
    Rp2Rs = PPs.Rp2Rs
    inc = 90
    Coefficents = [0.1, 0.1]
    print("Tss standard is: ", Tss)
    print("Rp/Rs standard is: ", Rp2Rs)
    
    
    # test Fp2Fs() function
    plt.subplots()
    plt.plot(Theta_array, Fp2Fs(Theta_array, 0.5, 0.1, 0, alpha_ellipse, *Coefficents, 0, Tss, Rp2Rs, inc))
    plt.plot(Theta_array, Fp2Fs(Theta_array, 0, 0.1, 0, alpha_ellipse, *Coefficents, 0, Tss, Rp2Rs, inc), '--')
    plt.plot(Theta_array, Fp2Fs(Theta_array, 0.5, 0, 0, alpha_ellipse, *Coefficents, 0, Tss, Rp2Rs, inc), ':')

    # plt.ylim([0, max(F_thermal + F_specular + F_Doppler + F_ellip)*1.1])
    # plt.xlim([0, 0.5])
    # plt.ylim([-200, -130])
    plt.ylim([0, 40])
    plt.show()
    
# # test quadratic function
# # conclusion: two quadratic functions are the same
# if __name__ == '__main__':
#     t1 = time.time()
#     print('test')
#     # import MCMC class
#     mcmc = MCMC('Kepler-10b', 'Kepler', sigma=2.5, ndim=7, nwalkers=64, nsteps=2000, burnin=1000)
#     # set parameters
#     Theta_array = np.linspace(0, 2*np.pi, 70)
#     offset = 0
#     AB =  0.0475
#     alpha_ellipse = 3.1739
#     alpha_Doppler = 0
#     F = 0
#     Tss = PPs.Tss
#     Rp2Rs = PPs.Rp2Rs
#     inc = 90
#     Coefficents = [0, 0]
#     print("Tss standard is: ", Tss)
#     print("Rp/Rs standard is: ", Rp2Rs)
    
#     # plot each function

#     F_tr = F_Transit(Theta_array, Rp2Rs, *Coefficents, inc)
#     F_tr_comp = F_Transit_comp(Theta_array, Rp2Rs, *Coefficents, inc)
#     print(f'time1: {time.time() - t1}')
    
#     plt.plot(Theta_array, F_tr, label='F_Transit')
#     plt.plot(Theta_array, F_tr_comp,'--', label='F_Transit_comp')
#     plt.legend()
#     plt.show()

# # test inc of thermal
# if __name__ == '__main__':
#     t1 = time.time()
#     print('test')
#     # import MCMC class
#     mcmc = MCMC('Kepler-10b', 'Kepler', sigma=2.5, ndim=7, nwalkers=64, nsteps=2000, burnin=1000)
#     # set parameters
#     Theta_array = np.linspace(0, 2*np.pi, 70)
#     offset = 0
#     AB =  0.0475
#     alpha_ellipse = 3.1739
#     alpha_Doppler = 0
#     F = 0
#     Tss = PPs.Tss
#     Rp2Rs = PPs.Rp2Rs
#     inc = 70
#     Coefficents = PPs.Coefficents
#     print("Tss standard is: ", Tss)
#     print("Rp/Rs standard is: ", Rp2Rs)
#     for inc in [90, 80]:
#     # plot each function
#         F_thermal1 = F_thermal(Theta_array, AB, F, Tss, Rp2Rs, offset, inc)
#         F_thermal_comp1 = F_thermal_comp(Theta_array, AB, F, Tss, Rp2Rs, offset, inc)
#         print(f'time1: {time.time() - t1}')

#         plt.plot(Theta_array, F_thermal1,'b', label='F_thermal')
#         plt.plot(Theta_array, F_thermal_comp1,'r--', label='F_thermal_comp')
#     plt.legend()
#     plt.show()
    
    