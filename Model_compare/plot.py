import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from analytical_model import *
from analytical_model_Lambert import F_lambert
import time
from parameters import PPs
# from Class_MCMC import MCMC
import cProfile
import pstats
import io
import hengmorris_analytic_phasecurves
import hengmorris_analytic_phasecurves_SingleScattering
from parameters import PPs

# Example: K2-141b
L_STAR = 0.18 * 3.828e26  # Watts (Total luminosity of the star)
STAR_RADIUS = 0.68 * 6.957e8   # meters
PLANET_RADIUS = 1.51 * 6.371e6 # meters
ORBIT_RADIUS = 0.00716 * 1.496e11
# --- 2. Define Atmosphere ---
H_SCALE_HEIGHT = 10e3   # 10e3 # meters

G_ASYMMETRY = 0.95
tau_inf = 100./(1-G_ASYMMETRY)            # optical depth of atmosphere on a perpendicular path, at sec.eclipse
KAPPA_SCAT_0 = tau_inf / (H_SCALE_HEIGHT) # Scattering coeff (1/m)
KAPPA_ABS_0 = 0.                          # Absorption coeff (1/m)
cutoff_scale_heights=15.0
# Show phase curve Fp/Fstar as fn of orbit angle. linear vs log y-axis
x = np.linspace(0,np.pi,num=1000)
y,A_g,q = hengmorris_analytic_phasecurves_SingleScattering.reflected_phase_curve( x, 1., G_ASYMMETRY, ORBIT_RADIUS/PLANET_RADIUS )
y_SS,A_g_SS,q_SS = hengmorris_analytic_phasecurves_SingleScattering.reflected_phase_curve_SS( x, 1., G_ASYMMETRY, ORBIT_RADIUS/PLANET_RADIUS )
F_atmos = np.array([y_SS[::-1], y_SS]).flatten()

if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()
    
    Theta_array = np.linspace(0, 2*np.pi, 2000)
    AB = 0.1  # Bond albedo
    alpha_ellipse = 0 #3.2351
    alpha_Doppler = 0 # 2.52 *20
    H1 = 20  # km
    H2 = 100  # km
    # H = 8.31/0.028 * 1000 / 20
    # print(H)
    I0 = (PPs.Rp/PPs.semi_axis)**2 * 1e6
    
    plt.figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    plt.plot(Theta_array/(2*np.pi), F_specular(Theta_array, AB, PPs.Rp2Rs) *I0,color=colors[1], label='Specular (lava ocean)', linewidth=2)
    plt.plot(Theta_array/(2*np.pi), F_lambert(Theta_array, AB, PPs.Rp2Rs) *I0,color='black', label='Diffuse (rocky surface)', linewidth=2)
    plt.plot(Theta_array/(2*np.pi),  F_atmos *I0,color=colors[0], label='Atmospheric Forward scattering', linewidth=2)
    plt.legend(loc='upper center', fontsize=15, frameon=False) # "upper center"
    plt.ylabel("Normalized intensity", fontsize=16)
    plt.xlabel("Orbital Phase", fontsize=16)
    plt.xlim(0,1)
    plt.tick_params(axis='both', labelsize=14)  # 设置x和y轴ticks字号
    # plt.title("Reflection", fontsize=17)
    ax = plt.gca()  # 获取当前轴
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)  # 设置边框线宽为1
    plt.tight_layout()
    plt.savefig('./figures/reflection.pdf', format='pdf')
    plt.show()
    
    
    # plt.figure()
    # plt.plot(Theta_array/(2*np.pi), F_thermal(Theta_array, AB, F=0.5, Rp2Rs=PPs.Rp2Rs), 'k-', label='Thick atmosphere', linewidth=2)
    # plt.plot(Theta_array/(2*np.pi), F_thermal(Theta_array, AB, F=0, Rp2Rs=PPs.Rp2Rs), 'k--', label='Bare surface', linewidth=2)
    # plt.legend(loc='upper center', fontsize=15, frameon=False)
    # plt.tick_params(axis='both', labelsize=14)  # 设置x和y轴ticks字号
    # plt.title("Thermal emission", fontsize=17)
    # plt.ylim(0, 22)
    # plt.xlim(0,1)
    # plt.xlabel("Orbital Phase", fontsize=19)
    # ax = plt.gca()  # 获取当前轴
    # for spine in ax.spines.values():
    #     spine.set_linewidth(1.5)  # 设置边框线宽为1
    # plt.tight_layout()
    # plt.savefig('./figures/thermal_emission.pdf', format='pdf')
    # plt.show()
    
    # plt.figure()
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    # plt.plot(Theta_array/(2*np.pi), F_specular(Theta_array, AB, PPs.Rp2Rs) + F_thermal(Theta_array, AB, F=0, Rp2Rs=PPs.Rp2Rs),'--',  linewidth=2, color = colors[1])
    # plt.plot(Theta_array/(2*np.pi), F_lambert(Theta_array, AB, PPs.Rp2Rs) + F_thermal(Theta_array, AB, F=0.5, Rp2Rs=PPs.Rp2Rs),  linewidth=2, color = colors[0])
    # plt.plot(Theta_array/(2*np.pi), F_atmos + F_thermal(Theta_array, AB, F=0.5, Rp2Rs=PPs.Rp2Rs), linewidth=2, color = colors[2])
    # plt.plot(Theta_array/(2*np.pi), F_atmos + F_thermal(Theta_array, AB, F=0.5, Rp2Rs=PPs.Rp2Rs), linewidth=2, color = colors[3])
    # plt.tick_params(axis='both', labelsize=14)  # 设置x和y轴ticks字号
    # plt.legend(loc='upper center', fontsize=15, frameon=False)
    # plt.title("Total flux", fontsize=17)
    # plt.xlabel(" ", fontsize=19)
    # plt.ylim(0, 20)
    # plt.xlim(0,1)
    # ax = plt.gca()  # 获取当前轴
    # for spine in ax.spines.values():
    #     spine.set_linewidth(1.5)  # 设置边框线宽为1
    # plt.tight_layout()
    # plt.savefig('./figures/total_flux.pdf', format='pdf')
    # plt.show()

    



