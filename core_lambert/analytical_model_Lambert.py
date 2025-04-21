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
    spl = interp1d(Response_data[:, 0], Response_data[:, 1], kind='linear')
    return spl(lam *1e6)
    
@supersample_decorator()
def F_thermal(Theta_array, AB, F=0, Tss = Tss_ref, Rp2Rs = PPs.Rp2Rs, inc = 90, lam1 = lam1, lam2 = lam2):
    # print('1')
    results = []
    # manual calculate "quad(lambda lam: B(lam, Ts)* Response(lam), lam1, lam2, limit=100)[0]"
    lam_array = np.linspace(lam1, lam2, 100)
    int_result = np.sum(B(lam_array, Ts) * Response(lam_array)) * (lam_array[1] - lam_array[0])
    
    Cor = Rp2Rs**2 / (np.pi *  int_result)
    Theta_array = np.acos(np.cos(Theta_array) * np.sin(inc/180 *np.pi)) # 计算入射角
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

@supersample_decorator()
def F_lambert(Theta_array, AB, Rp2Rs=PPs.Rp2Rs, inc=90, alpha=PPs.alpha):
    zt = np.acos(- np.sin(inc/180 *np.pi)* np.cos(Theta_array))
    Pt = AB * 2/3*(np.sin(zt) + (np.pi - zt) * np.cos(zt)) / np.pi
    # condition = np.abs(Theta_array - np.pi) < alpha
    # Pt = np.where(condition, 0, Pt)
    return Rp2Rs**2 *alpha**2 * Pt *1e6

    
@supersample_decorator()
def F_ellip(Theta_array, alpha_ellip):
    A_ellip = alpha_ellip /0.077 *Mp_J* Rs_S**3 *Ms_S**-2 *(P/24)**-2
    print('A_ellip:', A_ellip)
    return A_ellip *(1 - np.cos(2* Theta_array - 2*np.pi)) 

@supersample_decorator()
def F_Doppler(Theta_array, alpha_Doppler):
    A_Doppler = alpha_Doppler/0.37 *Mp_J *Ms_S**(-2/3) *(P/24)**(-1/3)
    print('A_Doppler:', A_Doppler)
    return A_Doppler *np.sin(Theta_array)

def Fp2Fs(Theta_array, AB=0, F=0, alpha_ellip=0, co1=Co1, co2=Co2, Tss = Tss_ref, Rp2Rs = PPs.Rp2Rs, inc = 90, alpha = PPs.alpha, params = []):
    if len(params) != 0:
        AB, Tss, Rp2Rs, F, inc, alpha  = params
    if inc == 0:
        print('Warning: inc is 0, set to 90.')
        inc = 90
    return (F_thermal(Theta_array, AB, F, Tss, Rp2Rs, inc) + F_lambert(Theta_array, AB, Rp2Rs, inc, alpha)) *Eclipse(Theta_array, Rp2Rs, inc, alpha)  + F_Transit(Theta_array, Rp2Rs, co1, co2, inc, alpha)

# if __name__ == '__main__':
#     F_Doppler(0,1)
#     F_ellip(0,1)