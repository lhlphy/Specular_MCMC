import numpy as np
from scipy.integrate import dblquad, quad, tplquad
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from parameters import PPs
import os
from pytransit import RoadRunnerModel

# Constants List
# Rs = PPs.Rs
# Rp = PPs.Rp
# e = PPs.eccentricity
# a = PPs.semi_axis
Ts = PPs.Stellar_T
P = PPs.Period
Mp_J = PPs.Mp_J
Rs_S = PPs.Rs_S
Ms_S = PPs.Ms_S
# alpha = np.arcsin(Rs / a)
lam1 = 0.43e-6
lam2 = 0.89e-6
Tss_ref = PPs.Tss

def F_Transit(Theta_array, Rp2Rs, co1, co2, inc, alpha = PPs.alpha):
    time = Theta_array / (2 * np.pi) * P
    tm = RoadRunnerModel('quadratic')
    tm.set_data(time)
    
    a_sc = 1/np.sin(alpha)   # a / Rs
    flux1 = tm.evaluate(k=Rp2Rs, ldc=[co1, co2], t0=0.0, p=P, a=a_sc, i= inc/180 *np.pi, e=0.0, w=0.0)
    return (flux1 - 1) *1e6

def Eclipse(Theta_array, Rp2Rs, inc, alpha = PPs.alpha):
    time = (Theta_array + np.pi) / (2 * np.pi) * P
    tm = RoadRunnerModel('uniform')
    tm.set_data(time)
    
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
    spl = interp1d(Response_data[:, 0], Response_data[:, 1], kind='linear')
    return spl(lam *1e6)
    
def F_thermal(Theta_array, AB_D, AB_S, F=0, Tss = Tss_ref, Rp2Rs = PPs.Rp2Rs, inc = 90, lam1 = lam1, lam2 = lam2, alpha = PPs.alpha):
    # print('1')
    results = []
    # manual calculate "quad(lambda lam: B(lam, Ts)* Response(lam), lam1, lam2, limit=100)[0]"
    lam_array = np.linspace(lam1, lam2, 100)
    int_result = np.sum(B(lam_array, Ts) * Response(lam_array)) * (lam_array[1] - lam_array[0])
    
    Cor = Rp2Rs**2 / (np.pi *  int_result)
    Theta_array = np.acos(np.cos(Theta_array) * np.sin(inc/180 *np.pi)) # 计算入射角
    for i, Theta in enumerate(Theta_array):
        # if np.abs(Theta - np.pi) < alpha: # eclipse

        #     results.append(0)
        #     continue

        def int_func(lam, theta, phi):
            cos_phi = np.cos(phi)
            cos_zenith = np.cos(theta)* np.cos(phi)
            zenith_obs = np.arccos(- np.cos(theta + Theta)* cos_phi)
            return -B(lam, Toy_model(cos_zenith, AB_D + AB_S, F, Tss)) * cos_phi**2 * np.cos(Theta + theta) *(1 - AB_D - A_Fresnel(A_normal=AB_S, I_angle = zenith_obs)) * Response(lam)

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
        results.append(result* Cor)

    results = np.array(results) *1e6
    return results

def F_lambert(Theta_array, AB, Rp2Rs=PPs.Rp2Rs, inc=90, alpha=PPs.alpha):
    zt = np.acos(- np.sin(inc/180 *np.pi)* np.cos(Theta_array))
    Pt = AB * 3/2*(np.sin(zt) + (np.pi - zt) * np.cos(zt)) / np.pi
    # condition = np.abs(Theta_array - np.pi) < alpha
    # Pt = np.where(condition, 0, Pt)
    return Rp2Rs**2 *alpha**2 * Pt *1e6

def A_Fresnel(Theta = 0, A_normal = 0, I_angle = -1, inc = 90):
    
    Theta = np.acos(np.cos(Theta) * np.sin(inc/180 *np.pi))
    Ang = np.abs((np.pi - Theta) / 2) 
    I_angle = np.where(I_angle == -1, Ang, I_angle)
    # 将I_angle中大于pi/2的值转换为pi/2
    I_angle = np.where(I_angle > np.pi / 2, np.pi / 2, I_angle)
    SINI = np.sin(I_angle)
    COSI = np.cos(I_angle)  
    n = 2/(1- np.sqrt(A_normal)) -1
    Co1 = np.sqrt(n**2 - SINI**2)

    Rs = ((COSI - Co1) / (COSI + Co1)) **2
    Rp = ((Co1 - n**2 *COSI)/ (Co1 + n**2 *COSI))**2
    return (Rs+Rp)/2

def F_specular(Theta_array, An, Rp2Rs=PPs.Rp2Rs, inc = 90, alpha=PPs.alpha): 
    F = Rp2Rs**2 * np.sin(alpha/2)**2
    Theta_array = np.acos(np.cos(Theta_array) * np.sin(inc/180 *np.pi))
    Tx = np.abs(np.pi-np.abs(Theta_array))/2
    F = F * np.where(Tx > np.pi/2 - alpha/2, (A_Fresnel(I_angle= Tx , A_normal= An, inc= inc) *(np.pi - 2*Tx)/alpha +  A_Fresnel(I_angle= Tx-alpha/3 , A_normal= An, inc=inc) *(2*Tx - np.pi + alpha)/alpha), A_Fresnel(I_angle= Tx , A_normal= An, inc=inc))
    
    # F[np.abs(Theta_array - np.pi) < alpha] = 0 # eclipse
    return F *1e6

    
def F_ellip(Theta_array, alpha_ellip):
    A_ellip = alpha_ellip /0.077 *Mp_J* Rs_S**3 *Ms_S**-2 *P**-2
    return A_ellip *(1 - np.cos(2* Theta_array - 2*np.pi)) 

def F_Doppler(Theta_array, alpha_Doppler):
    A_Doppler = alpha_Doppler/0.37 *Mp_J *Ms_S**(-2/3) *P**(-1/3)
    return A_Doppler *np.sin(Theta_array)

from Sampling import supersample_decorator
@supersample_decorator()
def Fp2Fs(Theta_array, AB_D=0, AB_S=0, F=0, alpha_ellip=0, co1=0, co2=0, delta =0, Tss = Tss_ref, Rp2Rs = PPs.Rp2Rs, inc = 90, alpha =PPs.alpha, params = []):
    if len(params) != 0:
        AB_D, AB_S, alpha_ellip, delta, Tss, Rp2Rs, F, inc, alpha  = params
    if inc == 0:
        print('Warning: inc is 0, set to 90.')
        inc = 90
    return (F_thermal(Theta_array, AB_D, AB_S, F, Tss, Rp2Rs, inc, alpha) + F_lambert(Theta_array, AB_D, Rp2Rs, inc, alpha) + F_specular(Theta_array, AB_S, Rp2Rs, inc, alpha)) *Eclipse(Theta_array, Rp2Rs, inc, alpha) + F_ellip(Theta_array, alpha_ellip) + delta + F_Transit(Theta_array, Rp2Rs, co1, co2, inc, alpha)
