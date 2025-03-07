import numpy as np
from scipy.integrate import dblquad, quad, tplquad
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from core.parameters import PPs
import os

# Constants List
Rs = PPs.Rs
e = PPs.eccentricity
a = PPs.semi_axis
Ts = PPs.Stellar_T
P = PPs.Period
Mp_J = PPs.Mp_J
Rs_S = PPs.Rs_S
Ms_S = PPs.Ms_S
alpha = np.arcsin(Rs / a)
lam1 = 0.43e-6
lam2 = 0.89e-6
Tss_ref = PPs.Tss

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
    except KeyError:
        print('Not using response function of any telescope.')
        return 1
    
    # 读取文件
    Response_data = np.loadtxt(os.path.join(folder_path, 'Response.txt'), delimiter=',')
    # 插值
    spl = interp1d(Response_data[:, 0], Response_data[:, 1], kind='linear')
    return spl(lam *1e6)
    
def A_Fresnel(Theta = 0, A_normal = 0, I_angle = -1):
    I_angle = np.where(I_angle == -1, np.abs((np.pi - Theta) / 2) + alpha/2, I_angle)
    # 将I_angle中大于pi/2的值转换为pi/2
    I_angle = np.where(I_angle > np.pi / 2, np.pi / 2, I_angle)
    SINI = np.sin(I_angle)
    COSI = np.cos(I_angle)  
    n = 2/(1- np.sqrt(A_normal)) -1
    Co1 = np.sqrt(n**2 - SINI**2)

    Rs = ((COSI - Co1) / (COSI + Co1)) **2
    Rp = ((Co1 - n**2 *COSI)/ (Co1 + n**2 *COSI))**2
    return (Rs+Rp)/2
    
import numpy as np
    
def F_thermal_compressed(Theta_array, AB, F=0, Tss=Tss_ref, Rp2Rs=0):
    # 定义采样点
    phi_list = np.linspace(-np.pi / 2, np.pi / 2, 180)  # phi 网格
    theta_list = np.linspace(-np.pi, np.pi, 360)        # theta 网格，覆盖所有可能的范围
    lam_list = np.linspace(lam1, lam2, 20)              # lambda 网格

    # 计算步长
    dphi = phi_list[1] - phi_list[0]
    dtheta = theta_list[1] - theta_list[0]
    dlam = lam_list[1] - lam_list[0]

    # 构造广播数组，增加 Theta 维度
    Theta_array_bc = Theta_array[:, np.newaxis, np.newaxis, np.newaxis]  # 形状 (N_Theta, 1, 1, 1)
    theta_array = theta_list[np.newaxis, :, np.newaxis, np.newaxis]      # 形状 (1, 360, 1, 1)
    phi_array = phi_list[np.newaxis, np.newaxis, :, np.newaxis]          # 形状 (1, 1, 180, 1)
    lam_array = lam_list[np.newaxis, np.newaxis, np.newaxis, :]          # 形状 (1, 1, 1, 20)

    # 计算 cos_zenith 和 zenith_obs
    cos_phi = np.cos(phi_array)
    cos_zenith = np.cos(theta_array) * cos_phi
    zenith_obs = np.arccos(-np.cos(theta_array + Theta_array_bc) * cos_phi)

    # 计算被积函数的各个分量
    T = Toy_model(cos_zenith, AB, F, Tss)  # 温度模型
    B_lam_T = B(lam_array, T)              # 黑体辐射
    A_fresnel = A_Fresnel(A_normal=AB, I_angle=zenith_obs)  # 菲涅耳反射率
    response = Response(lam_array)         # 响应函数

    # 计算被积函数
    integrand = -B_lam_T * cos_phi**2 * np.cos(Theta_array_bc + theta_array) * (1 - A_fresnel) * response

    # 创建掩码，限制 theta 的积分范围
    theta_min = np.pi / 2 - Theta_array_bc
    theta_max = 3 * np.pi / 2 - Theta_array_bc
    mask = (theta_array >= theta_min) & (theta_array <= theta_max)

    # 应用掩码并积分
    integrand_masked = integrand * mask
    results = np.sum(integrand_masked, axis=(1, 2, 3)) * dtheta * dphi * dlam

    # 处理 transit 和 eclipse 情况
    transit_mask = (Theta_array < alpha) | (Theta_array > 2 * np.pi - alpha)
    eclipse_mask = np.abs(Theta_array - np.pi) < alpha
    results[transit_mask] = -Rp2Rs**2
    results[eclipse_mask] = 0

    # 计算归一化常数 Cor
    lam_array_norm = np.linspace(lam1, lam2, 100)
    int_result = np.sum(B(lam_array_norm, Ts) * Response(lam_array_norm)) * (lam_array_norm[1] - lam_array_norm[0])
    Cor = Rp2Rs**2 / (np.pi * int_result)

    # 应用归一化并转换为 ppm 单位
    results = results * Cor * 1e6

    return results

def F_thermal(Theta_array, AB, F=0, Tss=Tss_ref, Rp2Rs=0):
    '''
    该函数是对 F_thermal_compressed 的优化封装
    通过掩码的方式，将 transit 和 eclipse 的情况剔除，减少计算量
    F_thermal_compressed 可直接用于替代 F_thermal, 但计算效率较低(低大约20%, 具体为 4 alpha / 2pi)
    '''
    #1. 识别 transit 和 eclipse 的索引, 使用布尔掩码来找出 Theta_array 中属于 transit 和 eclipse 的部分
    transit_mask = (Theta_array < alpha) | (Theta_array > 2 * np.pi - alpha)
    eclipse_mask = np.abs(Theta_array - np.pi) < alpha
    compute_mask = ~(transit_mask | eclipse_mask)  # 需要计算的 Theta
    
    #2. 压缩 Theta_array, 利用 compute_mask 提取需要计算的 Theta 值，生成一个压缩后的数组
    Theta_array_compressed = Theta_array[compute_mask]
    
    #3. 对压缩后的数组进行矢量化计算
    results_compressed = F_thermal_compressed(Theta_array_compressed, AB, F, Tss, Rp2Rs)
    
    #4. 创建并填充最终结果数组
    results = np.zeros_like(Theta_array)
    results[compute_mask] = results_compressed  # 填入计算结果
    results[transit_mask] = -Rp2Rs**2  *1e6     # 填入 transit 的结果
    results[eclipse_mask] = 0                  # 填入 eclipse 的结果
    return results

def F_specular(Theta_array, AB, Rp2Rs = 0):
    SI = A_Fresnel(Theta_array, AB)* Rp2Rs**2 * alpha**2 /4 * (1-alpha**2 /24 *(2-np.cos(Theta_array))/ np.sin(Theta_array/2)**2)
    SI[(Theta_array < alpha) | (Theta_array > 2*np.pi - alpha) | (np.abs(Theta_array - np.pi) < alpha)] = 0
    return SI *1e6
    
def F_ellip(Theta_array, alpha_ellip):
    A_ellip = alpha_ellip /0.077 *Mp_J* Rs_S**3 *Ms_S**-2 *P**-2
    return A_ellip *(1 - np.cos(2* Theta_array - 2*np.pi)) 

def F_Doppler(Theta_array, alpha_Doppler):
    A_Doppler = alpha_Doppler/0.37 *Mp_J *Ms_S**(-2/3) *P**(-1/3)
    return A_Doppler *np.sin(Theta_array)

def Fp2Fs(Theta_array, AB, alpha_ellip, alpha_Doppler, F=0, delta =0, Tss = Tss_ref, Rp2Rs = 0):
    return F_thermal(Theta_array, AB, F, Tss, Rp2Rs) + F_specular(Theta_array, AB, Rp2Rs) + F_ellip(Theta_array, alpha_ellip) + F_Doppler(Theta_array, alpha_Doppler) + delta
