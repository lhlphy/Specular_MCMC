import numpy as np
from scipy.integrate import dblquad, quad, tplquad
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from parameters import PPs
import os
from pytransit import RoadRunnerModel
from Sampling import supersample_decorator
from thermal_lookup import band_integral, blackbody, response_values

# Constants List
Ts = PPs.Stellar_T
P = PPs.Period
Mp_J = PPs.Mp_J
Rs_S = PPs.Rs_S
Ms_S = PPs.Ms_S
# alpha = np.arcsin(Rs / a)
lam1 = 0.43e-6
lam2 = 0.89e-6
Tss_ref = PPs.Tss
Co1, Co2 = PPs.Coefficents

def F_Transit(Theta_array, Rp2Rs, co1, co2, inc, alpha = PPs.alpha):
    time = Theta_array / (2 * np.pi) * P
    tm = RoadRunnerModel('quadratic')
    tm.set_data(time, exptimes=29.4/60, nsamples=11)
    
    a_sc = 1/np.sin(alpha)   # a / Rs
    flux1 = tm.evaluate(k=Rp2Rs, ldc=[co1, co2], t0=0.0, p=P, a=a_sc, i= inc/180 *np.pi, e=0.0, w=0.0)
    return (flux1 - 1) *1e6

def Eclipse(Theta_array, Rp2Rs, inc, alpha = PPs.alpha):
    time = (Theta_array + np.pi) / (2 * np.pi) * P
    tm = RoadRunnerModel('uniform')
    tm.set_data(time, exptimes=29.4/60, nsamples=11)
    
    a_sc = 1/np.sin(alpha)/Rp2Rs # a / Rp
    flux1 = tm.evaluate(k= 1/Rp2Rs, ldc=[0, 0], t0=0.0, p=P, a=a_sc, i= inc/180 *np.pi, e=0.0, w=0.0)
    return (flux1 - np.min(flux1)) / np.max(flux1)

def Toy_model(cos_zenith, AB, F=0, Tss = Tss_ref):
    # Surface temperature model: Toy Model
    condition = cos_zenith < 0
    branch_true = (F / 2)**(1/4)  * Tss
    branch_false = (F / 2 + (1 - 2 * F) * cos_zenith)**(1/4) * Tss
    return np.where(condition, branch_true, branch_false)
    

def B(lam, T):
    return blackbody(lam, T)
    # 定义物理常数
    h = 6.626e-34  # Planck's constant
    c = 3.0e8      # Speed of light
    k = 1.38e-23   # Boltzmann constant
    
    # 条件：T < 10
    condition = T < 10
    
    # 计算黑体辐射公式
    A = np.exp(h * c / lam / k / T) - 1
    blackbody_result = 2 * h * c**2 / lam**5 / A
    
    # 根据条件选择返回�?    return np.where(condition, 0, blackbody_result)

def Response(lam):
    return response_values(lam)
    # 如果环境变量中存�?FOLDER_PATH，则使用环境变量中的路径
    try:
        folder_path = os.environ['FOLDER_PATH']
        # 读取文件
        Response_data = np.loadtxt(os.path.join(folder_path, 'Response.txt'), delimiter=',')
    except KeyError:
        try:
            Response_data = np.loadtxt('Response.txt', delimiter=',')
        except FileNotFoundError:
            # print('Not using response function of any telescope.')
            return 1

    # 插�?    spl = interp1d(Response_data[:, 0], Response_data[:, 1], kind='linear')
    return spl(lam *1e6)
    
@supersample_decorator()
def F_thermal(Theta_array, AB, F=0, Tss=Tss_ref, Rp2Rs=PPs.Rp2Rs, inc=90, lam1=lam1, lam2=lam2):
    results = []
    int_result = band_integral(Ts, lam1, lam2, n_lam=100)
    cor = Rp2Rs**2 / (np.pi * int_result)
    theta_view = np.acos(np.cos(Theta_array) * np.sin(inc / 180 * np.pi))

    phi_list = np.linspace(-np.pi / 2, np.pi / 2, 180)
    dphi = phi_list[1] - phi_list[0]
    phi_array = phi_list[np.newaxis, :]

    for Theta in theta_view:
        theta_list = np.linspace(np.pi / 2 - Theta, 3 * np.pi / 2 - Theta, 180)
        dtheta = theta_list[1] - theta_list[0]
        theta_array = theta_list[:, np.newaxis]

        cos_phi = np.cos(phi_array)
        cos_zenith = np.cos(theta_array) * cos_phi
        thermal_band = band_integral(Toy_model(cos_zenith, AB, F, Tss), lam1, lam2, n_lam=8)
        integrand = -thermal_band * cos_phi**2 * np.cos(Theta + theta_array) * (1 - AB)
        results.append(np.sum(integrand) * dphi * dtheta * cor)

    return np.array(results) * 1e6
@supersample_decorator()
def F_lambert(Theta_array, AB, Rp2Rs=PPs.Rp2Rs, inc=90, alpha=PPs.alpha):
    zt = np.acos(- np.sin(inc/180 *np.pi)* np.cos(Theta_array))
    Pt = AB * 2/3*(np.sin(zt) + (np.pi - zt) * np.cos(zt)) / np.pi
    # condition = np.abs(Theta_array - np.pi) < alpha
    # Pt = np.where(condition, 0, Pt)
    return Rp2Rs**2 *alpha**2 * Pt *1e6

    
@supersample_decorator()
def F_ellip(Theta_array, alpha_ellip):
    A_ellip = alpha_ellip /0.077 *Mp_J* Rs_S**3 *Ms_S**-2 *P**-2
    return A_ellip *(1 - np.cos(2* Theta_array - 2*np.pi)) 

@supersample_decorator()
def F_Doppler(Theta_array, alpha_Doppler):
    A_Doppler = alpha_Doppler/0.37 *Mp_J *Ms_S**(-2/3) *P**(-1/3)
    return A_Doppler *np.sin(Theta_array)

def Fp2Fs(Theta_array, AB=0, F=0, alpha_ellip=0, co1=Co1, co2=Co2, Tss = Tss_ref, Rp2Rs = PPs.Rp2Rs, inc = 90, alpha = PPs.alpha, x_offset=0, params = []):
    if len(params) != 0:
        AB, Tss, Rp2Rs, F, inc, alpha, co1, co2, delta, x_offset = params
    if inc == 0:
        print('Warning: inc is 0, set to 90.')
        inc = 90
    shifted_theta = Theta_array + 2 * np.pi * x_offset
    return (F_thermal(shifted_theta, AB, F, Tss, Rp2Rs, inc) + F_lambert(shifted_theta, AB, Rp2Rs, inc, alpha)) * Eclipse(shifted_theta, Rp2Rs, inc, alpha) + F_Transit(shifted_theta, Rp2Rs, co1, co2, inc, alpha) + delta
