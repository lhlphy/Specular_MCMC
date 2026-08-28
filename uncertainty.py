import numpy as np
from scipy.constants import h, c, k

# 定义辅助函数 A_Fresnel 和 B_func（保持不变）
def A_Fresnel(I_angle=0, A_normal=0, Polarization='None'):
    I_angle = np.where(I_angle > np.pi/2, np.pi/2, I_angle)
    SINI = np.abs(np.sin(I_angle))
    COSI = np.abs(np.cos(I_angle))
    An = A_normal
    n = 2 / (1 - np.sqrt(An)) - 1
    Co1 = np.sqrt(n**2 - SINI**2)
    if Polarization == 'None':
        Rs = ((COSI - Co1) / (COSI + Co1))**2
        Rp = ((Co1 - n**2 * COSI) / (Co1 + n**2 * COSI))**2
        return (Rs + Rp) / 2
    else:
        raise ValueError("Polarization must be 'None' for this context")

def B_func(T, lam, B=1):
    return 2 * B * h * c**2 / lam**5 / (np.exp(h * c / (lam * k * T)) - 1)

# 中间变量计算函数
def compute_intermediates(a2Rs, Ts):
    """计算中间变量 alpha 和 Tss
    参数：
        a2Rs: a / Rs
        Ts: 恒星表面温度
    返回：
        alpha: 角度
        Tss: 次恒星点温度
    """
    alpha = np.arcsin(1 / a2Rs)
    Tss = Ts * np.sqrt(1 / a2Rs)
    return alpha, Tss

# 定义目标函数（基于新参数）
def T_substellar_func(a2Rs, Ts):
    """计算次恒星点温度 Tss"""
    return Ts * np.sqrt(1 / a2Rs)

def alpha_func(a2Rs):
    """计算角度 alpha"""
    return np.arcsin(1 / a2Rs)

def Rp2Rs_func(Rp2Rs):
    """直接返回 Rp2Rs（输入参数）"""
    return Rp2Rs  # 可选函数，因为 Rp2Rs 已作为输入

def RSM_func(Rp2Rs, a2Rs, Ts, Mag_G):
    """计算 RSM 值"""
    alpha = alpha_func(a2Rs)
    sort_key = Rp2Rs**2 * np.sin(alpha / 2)**2 * 10**(-Mag_G / 5)
    return sort_key / 1.394368e-07 * 100

def peak_func(Rp2Rs, a2Rs, Ts):
    """计算 peak 值"""
    alpha = alpha_func(a2Rs)
    I_angle = np.pi / 2 - alpha / 3
    A_fresnel = A_Fresnel(I_angle=I_angle, A_normal=0.1)
    return Rp2Rs**2 * np.sin(alpha / 2)**2 * A_fresnel * 1e6

def R_star_func(a2Rs, Ts, lam_spec=1e-6):
    """计算 R_star 值"""
    alpha = alpha_func(a2Rs)
    Tss = T_substellar_func(a2Rs, Ts)
    B_Ts = B_func(Ts, lam_spec)
    B_Tss = B_func(Tss * (2/3)**0.25, lam_spec)
    return 0.1 / 0.9 * (B_Ts / B_Tss) * np.sin(alpha / 2)**2

# 数值导数函数（保持不变）
def numerical_derivative(func, params, param_idx, delta=1e-6):
    """计算函数对某一参数的数值导数"""
    params_plus = params.copy()
    params_minus = params.copy()
    params_plus[param_idx] += delta
    params_minus[param_idx] -= delta
    return (func(*params_plus) - func(*params_minus)) / (2 * delta)

# 不确定度计算函数（保持不变）
def calculate_uncertainty(params, uncertainties, func):
    """根据误差传播公式计算总不确定度"""
    derivatives = []
    for i, (param, unc) in enumerate(zip(params, uncertainties)):
        derivative = numerical_derivative(func, params, i)
        derivatives.append(derivative * unc)
    total_uncertainty = np.sqrt(np.sum(np.square(derivatives)))
    return total_uncertainty

# 主函数：计算 alpha, Rp2Rs, Tss, RSM, peak, R_star 及其不确定度
def compute_uncertainties(a2Rs, Rp2Rs, Ts, Mag_G, 
                         a2Rs_err, Rp2Rs_err, Ts_err, Mag_G_err, 
                         lam_spec=1e-6):
    """
    计算 'alpha', 'Rp2Rs', 'Tss', 'RSM', 'peak', 'R_star' 的值和不确定度
    参数：
        a2Rs, Rp2Rs, Ts, Mag_G: 输入参数值
        a2Rs_err, Rp2Rs_err, Ts_err, Mag_G_err: 输入参数的不确定度
        lam_spec: 波长（默认 1e-6 m）
    返回：
        dict: 包含各变量值及其不确定度的字典
    """
    # 计算中间变量和目标值
    alpha = alpha_func(a2Rs)
    Tss = T_substellar_func(a2Rs, Ts)
    Rp2Rs_value = Rp2Rs_func(Rp2Rs)  # 直接使用输入值
    RSM = RSM_func(Rp2Rs, a2Rs, Ts, Mag_G)
    peak = peak_func(Rp2Rs, a2Rs, Ts)
    R_star = R_star_func(a2Rs, Ts, lam_spec)

    # alpha 不确定度
    params_alpha = [a2Rs]
    uncertainties_alpha = [a2Rs_err]
    alpha_uncertainty = calculate_uncertainty(params_alpha, uncertainties_alpha, 
                                              lambda a2Rs: alpha_func(a2Rs))

    # Tss 不确定度
    params_Tss = [a2Rs, Ts]
    uncertainties_Tss = [a2Rs_err, Ts_err]
    Tss_uncertainty = calculate_uncertainty(params_Tss, uncertainties_Tss, 
                                            lambda a2Rs, Ts: T_substellar_func(a2Rs, Ts))

    # Rp2Rs 不确定度（直接使用输入的不确定度）
    Rp2Rs_uncertainty = Rp2Rs_err

    # RSM 不确定度
    params_RSM = [Rp2Rs, a2Rs, Ts, Mag_G]
    uncertainties_RSM = [Rp2Rs_err, a2Rs_err, Ts_err, Mag_G_err]
    RSM_uncertainty = calculate_uncertainty(params_RSM, uncertainties_RSM, 
                                            lambda Rp2Rs, a2Rs, Ts, Mag_G: RSM_func(Rp2Rs, a2Rs, Ts, Mag_G))

    # peak 不确定度
    params_peak = [Rp2Rs, a2Rs, Ts]
    uncertainties_peak = [Rp2Rs_err, a2Rs_err, Ts_err]
    peak_uncertainty = calculate_uncertainty(params_peak, uncertainties_peak, 
                                             lambda Rp2Rs, a2Rs, Ts: peak_func(Rp2Rs, a2Rs, Ts))

    # R_star 不确定度
    params_R_star = [a2Rs, Ts]
    uncertainties_R_star = [a2Rs_err, Ts_err]
    R_star_uncertainty = calculate_uncertainty(params_R_star, uncertainties_R_star, 
                                               lambda a2Rs, Ts: R_star_func(a2Rs, Ts, lam_spec))

    # 返回结果字典
    return {
        'alpha': alpha,
        'Tss': Tss,
        'Rp2Rs': Rp2Rs_value,
        'RSM': RSM,
        'peak': peak,
        'R_star': R_star,
        'alpha_uncertainty': alpha_uncertainty,
        'Tss_uncertainty': Tss_uncertainty,
        'Rp2Rs_uncertainty': Rp2Rs_uncertainty,
        'RSM_uncertainty': RSM_uncertainty,
        'peak_uncertainty': peak_uncertainty,
        'R_star_uncertainty': R_star_uncertainty
    }

# 测试代码（可选）
if __name__ == "__main__":
    # 示例输入
    a2Rs, Rp2Rs, Ts, Mag_G = 2.292, 0.02037, 4599, 10.619
    a2Rs_err, Rp2Rs_err, Ts_err, Mag_G_err = 0.0565, 0.00046, 79, 0

    # 调用主函数
    results = compute_uncertainties(a2Rs, Rp2Rs, Ts, Mag_G, 
                                   a2Rs_err, Rp2Rs_err, Ts_err, Mag_G_err)
    
    # 输出结果
    for key, value in results.items():
        print(f"{key}: {value}")