import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from parameters import PPs

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

def kelp_atmos_pc(omega = 1, g = 0, a_rp = 96.309):
    from kelp.jax import reflected_phase_curve, thermal_phase_curve
    phase = np.linspace(0, 1, 100)
    flux, Ag, q = reflected_phase_curve(phase, omega, g, a_rp)
    return phase, flux

def int_transit(F, Theta_list, alpha = PPs.alpha):
    # 计算transit的平均值
    mask = Theta_list < alpha/2/np.pi 
    F = F[mask]
    return np.mean(F)

def detect_tr(F, Theta_list, alpha = PPs.alpha):
    # 计算积分后的 F_tr
    F_tr = int_transit(F, Theta_list)
    # 计算阈值
    threshold = 1 - 1.5 *alpha/2/np.pi  
    # 筛选出满足条件的索引
    mask = Theta_list < threshold
    F_subset = F[mask]
    Theta_subset = Theta_list[mask]
    
    # 找到最小值的索引
    idx_min = np.argmin(F_subset)
    F_min = F_subset[idx_min]
    Theta_min = Theta_subset[idx_min]
    
    # # 计算最小值的积分值
    # mask = (Theta_list > alpha/2/np.pi) & (Theta_list < 3 *alpha/2/np.pi)
    # F_subset = F[mask]
    # F_min = np.mean(F_subset)
    
    print(f"F_tr is {F_tr}, F_min is {F_min}.")
    return F_tr - F_min
    

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

def Toy_model(cos_zenith, AB, F=0, Tss = Tss_ref):
    # Surface temperature model: Toy Model
    condition = cos_zenith < 0
    branch_true = (F / 2)**(1/4)  * Tss
    branch_false = (F / 2 + (1 - 2 * F) * cos_zenith)**(1/4) * Tss
    return np.where(condition, branch_true, branch_false)
    
    
def Response(lam):
    return 1
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

def F_thermal(Theta_array, AB, F=0, Tss = Tss_ref, Rp2Rs = PPs.Rp2Rs, inc = PPs.inc, lam1 = lam1, lam2 = lam2):
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


def F_lambert(Theta_array, AB, Rp2Rs=PPs.Rp2Rs, inc=PPs.inc, alpha=PPs.alpha):
    zt = np.acos(- np.sin(inc/180 *np.pi)* np.cos(Theta_array))
    Pt = AB * 2/3*(np.sin(zt) + (np.pi - zt) * np.cos(zt)) / np.pi
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

def F_specular(Theta_array, An, Rp2Rs=PPs.Rp2Rs, inc = PPs.inc, alpha=PPs.alpha): 
    F = Rp2Rs**2 * np.sin(alpha/2)**2
    Theta_array = np.acos(np.cos(Theta_array) * np.sin(inc/180 *np.pi))
    Tx = np.abs(np.pi-np.abs(Theta_array))/2
    F = F * np.where(Tx > np.pi/2 - alpha/2, (A_Fresnel(I_angle= Tx , A_normal= An, inc= inc) *(np.pi - 2*Tx)/alpha +  A_Fresnel(I_angle= Tx-alpha/3 , A_normal= An, inc=inc) *(2*Tx - np.pi + alpha)/alpha), A_Fresnel(I_angle= Tx , A_normal= An, inc=inc))
    
    # F[np.abs(Theta_array - np.pi) < alpha] = 0 # eclipse
    return F *1e6
    
def compare_phase_curve_plot_transit(An_list, wave_range, instrument = '  ', legend = 'below', xlabel = 'on', ylabel = 'on', errorbar = 0):
    '''
    该函数绘图transit为off状态, 即不考虑transit存在
    绘制主要full phase curve, 默认情况下name_list只有两个name, 分别代表low albedo和high albedo, 绘制4条曲线
    分别是: low albedo & Lambert, low albedo & Specular, high albedo & Lambert, high albedo & Specular
    可绘制 OpticalFrame == 'Full_cal' or 'Non_Fresnel' 两种情况下的phase_curve_comp plot
    
    legend: 'below', 'insert', 'off'
        'below' means legend below center the plot
        'insert' means legend in the plot
        'off' means no legend
        
    ylabel, xlabel: 'on', 'off': on or off the y and x label, because when four pics merge together, only the boundary axis label is needed
    
    errorbar: if errorbar == 0, no errorbar; Otherwise, draw a errorbar
    '''
    # load data, I_diffuse,I_specular is contrast ratio 
    
    pallet = ['b','r','k']
    plotarr  = [0] * 8
    fig, ax = plt.subplots()
    
    # Period: 7.72614 h
    Theta_list = np.linspace(0, 2*np.pi, 500) / (2 * np.pi)
    Nt = np.size(Theta_list)
    data = np.zeros([np.size(Theta_list), 6])
    lam1, lam2 = wave_range
    
    up_bound = 0  # control xlim up_lim
    # 计算in transit的phase span
    half_in_transit = np.arcsin(1/PPs.a2Rs) / (2 * np.pi)
    
    for i, An in enumerate(An_list):
        # 计算specular(core) and diffuse (core_lambert) 的phase curve
        F_t = F_thermal(Theta_list *2*np.pi, An, Tss= PPs.Tss *(1-An)**0.25, lam1=lam1, lam2=lam2)
        F_S = F_specular(Theta_list *2*np.pi, An)
        
        F_tl = F_thermal(Theta_list *2*np.pi, An, Tss= PPs.Tss *(1-An)**0.25, lam1=lam1, lam2=lam2)
        F_D = F_lambert(Theta_list *2*np.pi, An)

        CR_S = F_t + F_S  # full phase curve
        CR_D = F_tl + F_D # full phase curve

        if i == 0:
            CR_S0 = CR_S
        # 绘制phase curve
        plotarr[i*2], = plt.plot(Theta_list, CR_D, '-', color = 'b', linewidth = 2)
        plotarr[i*2+1], = plt.plot(Theta_list, CR_S, '-', color = 'r', linewidth = 2)
        data[:,0] = Theta_list
        up_bound = np.max([np.max(CR_D), np.max(CR_S), up_bound]) # find the max value for ylim
        if i != 2:
            data[:,2 + i*2] = CR_S
            data[:,3 + i*2] = CR_D
        elif i == 2:
            data[:,1] = CR_S
        
        # 需要放置errorbar的x坐标
        xloc = np.array([0.003,0.145, 0.32, 0.5, 0.68, 0.855, 0.997])
        spl = interp1d(Theta_list, CR_S, kind='linear')
        yloc = spl(xloc)
        for k, xl in enumerate(xloc):
            # 不带random drop
            plt.errorbar(xloc[k], yloc[k], yerr = errorbar, xerr=half_in_transit, fmt='o', color = 'k', linestyle='None', markersize=4)
            # 带random drop
            # plt.errorbar(xloc[k], yloc[k] + errorbar * np.random.random(), yerr = errorbar, xerr=half_in_transit, fmt='o', color = pallet[i], ecolor=pallet[i], linestyle='None')
        
    # 调整布局以便为图例腾出空间  
    fig.subplots_adjust(bottom=0.25) 
    plt.ylim([0, up_bound * 1.2])
    plt.xlim([0,1])

    # using gray background to sign "in transit"
    # 绘制in transit, eclipse区域的灰色背景 并标注
    ax.axvspan(0.5 - half_in_transit, 0.5 + half_in_transit, color='gray', alpha=0.2) 
    ax.axvspan(0, half_in_transit, color='gray', alpha=0.2) 
    ax.axvspan(1-half_in_transit, 1, color='gray', alpha=0.2) 
    if legend == 'insert': # insert为第一个图，仅第一个图标注，save your ink!
        plt.text(0.5, up_bound * 0.4, 'Eclipse', fontsize=11, fontweight='bold', color='gray', ha='center')
        plt.text(0, up_bound * 0.4, 'Transit', fontsize=11, fontweight='bold', color='gray', ha = 'left')
        plt.text(1, up_bound * 0.4, 'Transit', fontsize=11, fontweight='bold', color='gray', ha='right')
    
    # # 当 errorbar != 0 时，在坐标系左上角添加一个带误差棒的点
    # if errorbar != 0:
    #     if legend == 'insert': # insert为第一个图，要特别处理
    #         ax.errorbar(0.5, up_bound *0.9, xerr=half_in_transit, yerr=errorbar, fmt='o', color='k', markersize=4)
    #     else:
    #         ax.errorbar(0.8, up_bound *0.75, xerr=half_in_transit, yerr=errorbar, fmt='o', color='k', markersize=4)
    
    # 设置图例位置, insert:插入到图中；below：图下方；off:不显示图例
    if legend == 'below':  
        plt.legend([plotarr[0],plotarr[2],plotarr[4],plotarr[1],plotarr[3]], ['Low albedo & Lambert', 'High albedo & Lambert', 'Blackbody', 'Low albedo & Specular', 'High albedo & Specular'], loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)
    elif legend == 'insert':
        # plt.legend([plotarr[0],plotarr[2],plotarr[4],plotarr[1],plotarr[3]], [r'$A_B$='+f'{An_list[0]:.2f} & Lambert', r'$A_B$='+f'{An_list[1]:.2f} & Lambert', 'Blackbody', r'$A_B$='+f'{An_list[0]:.2f} & Specular', r'$A_B$='+f'{An_list[1]:.2f} & Specular'], loc='upper left', bbox_to_anchor=(0, 1.01), fontsize=9, frameon=False)
        plt.legend([plotarr[1],plotarr[0]], [f'Specular', f'Lambert'], loc='upper left', bbox_to_anchor=(0, 1.01), fontsize=12, frameon=False)
    elif legend == 'off':
        pass
    else:
        raise ValueError("Invalid legend position specified.")

    # 设置坐标轴标签 
    if xlabel == 'on':
        plt.xlabel('Orbital Phase', fontsize = 13)
    if ylabel == 'on':
        plt.ylabel(r'$F_p/F_*$ (ppm)', fontsize = 13)

    # 标注仪器名称和波长范围
    plt.text(0.75, up_bound *0.95 , instrument+'\n'+f'{wave_range[0]*1e6 :.2f}-{wave_range[1]*1e6 :.2f} μm', fontsize = 10, ha='center', fontweight='bold') # label instrument name and wavelength range

    # plt.savefig(f"temp/{name}/phase_curve_comp1.pdf", format = 'pdf')
    ins_name = instrument.replace('/','_').replace('\n', '')
    plt.savefig(f"temp/phase_curve_comp_{ins_name}_nt.pdf", format = 'pdf', bbox_inches='tight')
    plt.savefig(f"temp/phase_curve_comp_{ins_name}_nt.pdf", format = 'pdf', bbox_inches='tight')
    plt.show()
    plt.close()
    
    # 计算transit深度
    Transit_depth = PPs.Rp2Rs**2 * 1e6
    print(f'Transit depth: {Transit_depth:.2f} ppm')
    
    # 计算glint depth
    glint = detect_tr(CR_S, Theta_list)
    glint_sig = glint/errorbar
    print(f'Glint depth: {glint:.2f}, detect as {glint_sig:.2f}')