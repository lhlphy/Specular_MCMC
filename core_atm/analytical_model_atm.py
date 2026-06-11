import numpy as np
from scipy.integrate import dblquad, quad, tplquad
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from parameters import PPs
import os
from pytransit import RoadRunnerModel
from Sampling import supersample_decorator

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
    # 定义物理常数
    h = 6.626e-34  # Planck's constant
    c = 3.0e8      # Speed of light
    k = 1.38e-23   # Boltzmann constant
    
    # 条件：T < 10
    condition = T < 10
    
    # 计算黑体辐射公式
    A = np.exp(h * c / lam / k / T) - 1
    blackbody_result = 2 * h * c**2 / lam**5 / A
    
    # 根据条件选择返回值
    return np.where(condition, 0, blackbody_result)

def Response(lam):
    # 如果环境变量中存在 FOLDER_PATH，则使用环境变量中的路径
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

    # 插值
    spl = interp1d(Response_data[:, 0], Response_data[:, 1], kind='linear', fill_value="extrapolate", bounds_error=False)
    res = spl(lam *1e6)
    res = np.where(res < 0, 0, res)
    return res
    
@supersample_decorator()
def F_thermal(Theta_array, AB, F=0, Tss = Tss_ref, Rp2Rs = PPs.Rp2Rs, inc = 90, lam1 = lam1, lam2 = lam2):
    # print('1')
    results = []
    # manual calculate "quad(lambda lam: B(lam, Ts)* Response(lam), lam1, lam2, limit=100)[0]"
    lam_array = np.linspace(lam1, lam2, 100)
    int_result = np.sum(B(lam_array, Ts) * Response(lam_array)) * (lam_array[1] - lam_array[0])
    
    Cor = Rp2Rs**2 / (np.pi *  int_result)
    Theta_array = np.arccos(np.cos(Theta_array) * np.sin(inc/180 *np.pi)) # 计算入射角
    for i, Theta in enumerate(Theta_array):
        # print(Theta)
        # if Theta > np.pi + 0.01:  # 关于np.pi对称 
        #     results.append(results[len(Theta_array) - i - 1])
        #     continue
            
        # if Theta < alpha or Theta > 2*np.pi - alpha: # transit
        #     results.append(-Rp2Rs**2)
        #     # print((Rp/Rs)**2)
        #     continue
        # if np.abs(Theta - np.pi) < alpha: # eclipse

        #     results.append(0)
        #     continue

        def int_func(lam, theta, phi):
            cos_phi = np.cos(phi)
            cos_zenith = np.cos(theta)* np.cos(phi)
            return -B(lam, Toy_model(cos_zenith, AB, F, Tss)) * cos_phi**2 * np.cos(Theta + theta) *(1 - AB) * Response(lam)

        # 定义采样点
        phi_list = np.linspace(-np.pi / 2, np.pi / 2, 180)
        theta_list = np.linspace(np.pi/2 - Theta, 3*np.pi/2 - Theta, 180)
        lam_list = np.linspace(lam1, lam2, 8)

        # 构造广播数组
        theta_array = theta_list[:, np.newaxis, np.newaxis]  # 形状 (180, 1, 1)
        phi_array = phi_list[np.newaxis, :, np.newaxis]      # 形状 (1, 180, 1)
        lam_array = lam_list[np.newaxis, np.newaxis, :]      # 形状 (1, 1, 10)

        # # 矢量化计算 I_matrix
        I_matrix = int_func(lam_array, theta_array, phi_array)

        # 计算结果
        result = np.sum(I_matrix) * (phi_list[1] - phi_list[0]) * (theta_list[1] - theta_list[0]) * (lam_list[1] - lam_list[0])
        # result, _ = tplquad(
        #     int_func,
        #     -np.pi / 2, np.pi / 2,  # phi limits
        #     lambda phi: np.pi/2 - Theta ,
        #     lambda phi: 3* np.pi/2 - Theta ,  # theta limits -> 3* np.pi/2 - Theta ||if F=0 use np.pi/2
        #     lambda phi, theta: lam1,
        #     lambda phi, theta: lam2,  # lam limits
        #     epsabs=1e-3,       # Increase absolute tolerance
        #     epsrel=1e-3       # Increase relative tolerance
        # )
        results.append(result* Cor)

    results = np.array(results) *1e6
    return results

def heng_reflected_phase_curve(xi, omega, g, a_rp):
    xi = np.asarray(xi, dtype=float)
    phases = (xi + np.pi) / (2.0 * np.pi)
    alpha_hg = 2.0 * np.pi * phases - np.pi
    abs_alpha = np.abs(alpha_hg)

    gamma = np.sqrt(1.0 - omega)
    eps = (1.0 - gamma) / (1.0 + gamma)

    p_star = (1.0 - g**2) / (1.0 + g**2 + 2.0 * g * np.cos(alpha_hg)) ** 1.5
    p_0 = (1.0 - g) / (1.0 + g) ** 2

    rho_s = p_star - 1.0 + 0.25 * ((1.0 + eps) * (2.0 - eps)) ** 2
    rho_s_0 = p_0 - 1.0 + 0.25 * ((1.0 + eps) * (2.0 - eps)) ** 2
    rho_l = 0.5 * eps * (2.0 - eps) * (1.0 + eps) ** 2
    rho_c = eps**2 * (1.0 + eps) ** 2

    alpha_plus = np.sin(abs_alpha / 2.0) + np.cos(abs_alpha / 2.0)
    alpha_minus = np.sin(abs_alpha / 2.0) - np.cos(abs_alpha / 2.0)

    valid = (
        (alpha_minus != -1.0)
        & (alpha_plus != 1.0)
        & (alpha_plus != -1.0)
        & (alpha_minus != 1.0)
    )
    num1 = np.where(valid, 1.0 + alpha_minus, 1.0)
    num2 = np.where(valid, alpha_plus - 1.0, 1.0)
    den1 = np.where(valid, 1.0 + alpha_plus, 1.0)
    den2 = np.where(valid, 1.0 - alpha_minus, 1.0)

    psi_0 = np.where(valid, np.log(num1 * num2 / den1 / den2), 0.0)
    psi_s = 1.0 - 0.5 * (np.cos(abs_alpha / 2.0) - 1.0 / np.cos(abs_alpha / 2.0)) * psi_0
    psi_l = (np.sin(abs_alpha) + (np.pi - abs_alpha) * np.cos(abs_alpha)) / np.pi
    psi_c = (
        -1.0
        + 5.0 / 3.0 * np.cos(abs_alpha / 2.0) ** 2
        - 0.5 * np.tan(abs_alpha / 2.0) * np.sin(abs_alpha / 2.0) ** 3 * psi_0
    )

    psi_s = np.where(abs_alpha == 0.0, 1.0, psi_s)
    psi_c = np.where(abs_alpha == 0.0, 2.0 / 3.0, psi_c)
    psi_s = np.where(abs_alpha == np.pi, 0.0, psi_s)
    psi_c = np.where(abs_alpha == np.pi, 0.0, psi_c)

    a_g = omega / 8.0 * (p_0 - 1.0) + eps / 2.0 + eps**2 / 6.0 + eps**3 / 24.0
    psi = (
        12.0 * rho_s * psi_s + 16.0 * rho_l * psi_l + 9.0 * rho_c * psi_c
    ) / (12.0 * rho_s_0 + 16.0 * rho_l + 6.0 * rho_c)
    return 1e6 * (a_rp**-2 * a_g * psi)


def hg_point_star_phase_function(alpha_hg, g):
    return (1.0 - g**2) / (
        4.0 * np.pi * (1.0 + g**2 - 2.0 * g * np.cos(alpha_hg)) ** 1.5
    )


def garcia_munoz_limb_flux(alpha_hg, h, rp, semi_major_axis, omega, g):
    return (
        2.0
        * np.pi
        * h
        * rp
        / semi_major_axis**2
        * omega
        * hg_point_star_phase_function(alpha_hg, g)
        * 1e6
    )


@supersample_decorator()
def F_atmospheric(Theta_array, omega, g, Rp2Rs=PPs.Rp2Rs, inc=90, alpha=PPs.alpha):
    theta = np.asarray(Theta_array, dtype=float)
    phase_angle = np.arccos(
        np.clip(-np.sin(inc / 180.0 * np.pi) * np.cos(theta), -1.0, 1.0)
    )
    orbital_angle = np.pi - phase_angle
    rp = Rp2Rs
    semi_major_axis = 1.0 / np.sin(alpha)
    h = 100.0 / PPs.Rs

    heng_hg = heng_reflected_phase_curve(
        np.pi - orbital_angle,
        omega,
        g,
        semi_major_axis / rp,
    )
    limb = garcia_munoz_limb_flux(
        orbital_angle,
        h=h,
        rp=rp,
        semi_major_axis=semi_major_axis,
        omega=omega,
        g=g,
    )
    return heng_hg + limb

    
@supersample_decorator()
def F_ellip(Theta_array, alpha_ellip):
    A_ellip = alpha_ellip /0.077 *Mp_J* Rs_S**3 *Ms_S**-2 *(P/24)**-2
    return A_ellip *(1 - np.cos(2* Theta_array - 2*np.pi)) 

@supersample_decorator()
def F_Doppler(Theta_array, alpha_Doppler):
    A_Doppler = alpha_Doppler/0.37 *Mp_J *Ms_S**(-2/3) *(P/24)**(-1/3)
    return A_Doppler *np.sin(Theta_array)

def Fp2Fs(Theta_array, omega=0.8, g=0.77, F=0, alpha_ellip=0, co1=Co1, co2=Co2, Tss = Tss_ref, Rp2Rs = PPs.Rp2Rs, inc = 90, alpha = PPs.alpha, params = []):
    if len(params) != 0:
        omega, g, Tss, Rp2Rs, F, inc, alpha, delta  = params
    if inc == 0:
        print('Warning: inc is 0, set to 90.')
        inc = 90
    return (F_thermal(Theta_array, omega, F, Tss, Rp2Rs, inc) + F_atmospheric(Theta_array, omega, g, Rp2Rs, inc, alpha)) *Eclipse(Theta_array, Rp2Rs, inc, alpha)  + F_Transit(Theta_array, Rp2Rs, co1, co2, inc, alpha) + F_ellip(Theta_array, alpha_ellip) + delta

# if __name__ == '__main__':
#     F_Doppler(0,1)
#     F_ellip(0,1)
