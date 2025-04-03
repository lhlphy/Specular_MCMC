import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from analytical_model import *
import time
from parameters import PPs
from Class_MCMC import MCMC
import cProfile
import pstats
import io

#  # test1
# if __name__ == '__main__':
#     print('test')
#     # import MCMC class
#     mcmc = MCMC('Kepler-10b', 'Kepler', sigma=2.5, ndim=7, nwalkers=64, nsteps=2000, burnin=1000)
#     # set parameters
#     Theta_array = np.linspace(0, 2*np.pi, 100)
#     AB = 0.1
#     alpha_ellipse = 3.2351
#     alpha_Doppler = 2.52 *20
#     F = 0.5
#     Tss = 3552
#     Rp2Rs = 0.0132
#     offset = 0.2
#     print("Tss standard is: ", Tss)
#     print("Rp/Rs standard is: ", Rp2Rs)
    
#     F_tr1 = F_Transit(Theta_array, Rp2Rs, 0, 0)
#     F_tr2 = F_Transit(Theta_array, Rp2Rs, 0.3, 0)
#     F_tr3 = F_Transit(Theta_array, Rp2Rs, 0, 0.3)
#     F_tr4 = F_Transit(Theta_array, Rp2Rs, 0.3, 0.3)
    
#     plt.figure()
#     plt.plot(Theta_array, F_tr1, label='0, 0')
#     plt.plot(Theta_array, F_tr2, label='0.3, 0')
#     plt.plot(Theta_array, F_tr3, label='0, 0.3')
#     plt.plot(Theta_array, F_tr4, label='0.3, 0.3')
#     plt.legend()
#     plt.show()


# test2
if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()
    
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
    Coefficents = PPs.Coefficents
    print("Tss standard is: ", Tss)
    print("Rp/Rs standard is: ", Rp2Rs)
    
    # plot each function
    F_thermal = F_thermal(Theta_array, AB, F, Tss, Rp2Rs, offset, inc)
    F_specular = F_specular(Theta_array, AB, Rp2Rs, offset, inc)
    F_Doppler = F_Doppler(Theta_array, alpha_Doppler)
    F_ellip = F_ellip(Theta_array, alpha_ellipse)
    F_tr = F_Transit(Theta_array, Rp2Rs, *Coefficents, inc)
    print(f'time1: {time.time() - t1}')
    
    pr.disable()
    # s = io.StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())
    
    # subplot, plot each function as a subplot, so 4 subplots in total
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].plot(Theta_array, F_thermal)
    axs[0, 0].set_title("F_thermal")
    axs[0, 0].set_ylim([-2, max(F_thermal)*1.2])
    axs[0, 1].plot(Theta_array, F_specular)
    axs[0, 1].set_title("F_specular")
    axs[1, 0].plot(Theta_array, F_tr)
    axs[1, 0].set_title("F_Transit")
    axs[1, 1].plot(Theta_array, F_ellip)
    axs[1, 1].set_title("F_ellip")
    plt.show()
    
    # test Fp2Fs() function
    plt.subplots()
    plt.plot(Theta_array, Fp2Fs(Theta_array, AB, 0, alpha_ellipse, *Coefficents, 0, Tss, Rp2Rs, inc))
    plt.plot(Theta_array, F_tr)
    plt.plot(Theta_array, F_thermal + F_specular + F_ellip + F_tr, '--b')
    # plt.ylim([0, max(F_thermal + F_specular + F_Doppler + F_ellip)*1.1])
    # plt.xlim([0, 0.5])
    # plt.ylim([-200, -130])
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


    
    