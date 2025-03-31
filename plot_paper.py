import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from parameters import PPs
import sys
sys.path.append('./core')
sys.path.append('./core_Lambert')
from core.Class_MCMC import MCMC
import core.analytical_model
import core_lambert.Class_MCMC
import core_lambert.analytical_model_Lambert
import core
import core_lambert
import warnings
warnings.filterwarnings("ignore")


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
    
    pallet = ['r','b','k']
    plotarr  = [0] * 8
    fig, ax = plt.subplots()
    
    xloc = np.array([0.3948, 0.5, 0.6052])
    # Period: 7.72614 h
    Theta_list = np.linspace(0, 2*np.pi, 200) / (2 * np.pi)
    Nt = np.size(Theta_list)
    data = np.zeros([np.size(Theta_list), 6])
    lam1, lam2 = wave_range
    
    up_bound = 0  # control xlim up_lim
    for i, An in enumerate(An_list):
        F_t = core.analytical_model.F_thermal(Theta_list *2*np.pi, An, Tss= PPs.Tss *(1-An)**0.25, lam1=lam1, lam2=lam2)
        F_S = core.analytical_model.F_specular(Theta_list *2*np.pi, An)
        
        F_tl = core_lambert.analytical_model_Lambert.F_thermal(Theta_list *2*np.pi, An, Tss= PPs.Tss *(1-An)**0.25, lam1=lam1, lam2=lam2)
        F_D = core_lambert.analytical_model_Lambert.F_lambert(Theta_list *2*np.pi, An)

        CR_S = F_t + F_S
        CR_D = F_tl + F_D
        if i == 0:
            CR_S0 = CR_S

        # # 计算最大偏差并输出 low-high
        # if i == 0:
        #     CR_Dl = CR_D
        #     CR_Sl = CR_S
        #     print(f"Max difference list of {instrument}") # 提示仪器名称
            
        # # 计算最大偏差并输出 specular-diffuse
        # print(f"Max difference|specular-diffuse|{name}: ", np.max(np.abs(CR_S - CR_D)))
        # if i == 1:
        #     print(f"Max difference|low-high|diffuse: ", np.max(np.abs(CR_Dl - CR_D)))
        #     print(f"Max difference|low-high|specular: ", np.max(np.abs(CR_Sl - CR_S)))
            
        # # CR_D, CR_S 第一个元素和最后一个元素是0， 需要矫正
        # CR_D[0] = CR_D[1]
        # CR_S[0] = CR_S[1]
        # CR_D[-1] = CR_D[-2]
        # CR_S[-1] = CR_S[-2]
        
        plotarr[i*2], = plt.plot(Theta_list, CR_D, '--', color = pallet[i], linewidth = 2)
        plotarr[i*2+1], = plt.plot(Theta_list, CR_S, '-', color = pallet[i], linewidth = 2)
        data[:,0] = Theta_list
        up_bound = np.max([np.max(CR_D), np.max(CR_S), up_bound]) # find the max value for ylim
        if i != 2:
            data[:,2 + i*2] = CR_S
            data[:,3 + i*2] = CR_D
        elif i == 2:
            data[:,1] = CR_S
        
        spl = interp1d(Theta_list, CR_D, kind='linear')
        yloc = spl(xloc)
        for k, xl in enumerate(xloc):
            if i != 2:
                print(' ')
                # plt.errorbar(xloc[k], yloc[k], yerr = ebar[k], xerr=tbar[k], fmt='o', color = pallet[i], ecolor=pallet[i], linestyle='None')
            # plt.errorbar(xloc[k], yloc[k] + ebar[k] * np.random.random(), yerr = ebar[k], xerr=tbar[k], fmt='o', color = pallet[i], ecolor=pallet[i], linestyle='None')
        
        # 调整布局以便为图例腾出空间  
    fig.subplots_adjust(bottom=0.25) 
 
    plt.ylim([0, up_bound * 1.1])
    plt.xlim([0,1])
    # 设置背景颜色  
    # ax.axvspan(0.4593, 0.5407, color='gray', alpha=0.5)
    # ax.axvspan(0.3303, 0.4593, color='gray', alpha=0.2)
    # ax.axvspan(0.5407, 0.6697, color='gray', alpha=0.2)
    
    # 计算in transit的phase span
    half_in_transit = np.arcsin(PPs.Rs/PPs.semi_axis) / (2 * np.pi)
    # using gray background to sign "in transit"
    ax.axvspan(0.5 - half_in_transit, 0.5 + half_in_transit, color='gray', alpha=0.2) 
    ax.axvspan(0, half_in_transit, color='gray', alpha=0.2) 
    ax.axvspan(1-half_in_transit, 1, color='gray', alpha=0.2) 
    if legend == 'insert':
        plt.text(0.5, up_bound * 0.4, 'Eclipse', fontsize=11, fontweight='bold', color='gray', ha='center')
        plt.text(0, up_bound * 0.4, 'Transit', fontsize=11, fontweight='bold', color='gray', ha = 'left')
        plt.text(1, up_bound * 0.4, 'Transit', fontsize=11, fontweight='bold', color='gray', ha='right')
    
    # ax.axvline(x= half_in_transit, color='gray', linestyle='--', alpha=0.3)
    # ax.axvline(x= 1-half_in_transit, color='gray', linestyle='--', alpha=0.3)

    # 当 errorbar != 0 时，在坐标系左上角添加一个带误差棒的点
    if errorbar != 0:
        if legend == 'insert': # insert
            ax.errorbar(0.5, up_bound *0.9, xerr=half_in_transit, yerr=errorbar, fmt='o', color='k', markersize=4)
        else:
            # 找到最接近 2 * half_in_transit 的值的索引
            # target_value = 2 * half_in_transit
            # closest_index = np.abs(Theta_list - target_value).argmin()
            # ax.errorbar(Theta_list[closest_index-1], CR_S0[closest_index-1], xerr=half_in_transit, yerr=errorbar, fmt='o', color='k', markersize=4)
            ax.errorbar(0.8, up_bound *0.75, xerr=half_in_transit, yerr=errorbar, fmt='o', color='k', markersize=4)
    
    # 设置图例位置
    if legend == 'below':  
        plt.legend([plotarr[0],plotarr[2],plotarr[4],plotarr[1],plotarr[3]], ['Low albedo & Lambert', 'High albedo & Lambert', 'Blackbody', 'Low albedo & Specular', 'High albedo & Specular'], loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)
    elif legend == 'insert':
        # plt.legend([plotarr[0],plotarr[2],plotarr[4],plotarr[1],plotarr[3]], [r'$A_B$='+f'{An_list[0]:.2f} & Lambert', r'$A_B$='+f'{An_list[1]:.2f} & Lambert', 'Blackbody', r'$A_B$='+f'{An_list[0]:.2f} & Specular', r'$A_B$='+f'{An_list[1]:.2f} & Specular'], loc='upper left', bbox_to_anchor=(0, 1.01), fontsize=9, frameon=False)
        plt.legend([plotarr[1],plotarr[0]], [f'Specular', f'Lambert'], loc='upper left', bbox_to_anchor=(0, 1.01), fontsize=12, frameon=False)
    elif legend == 'off':
        pass
    else:
        raise ValueError("Invalid legend position specified.")

    # plt.legend(plotarr, ['Low albedo & Lambert', 'Low albedo & Specular',  'Mid albedo & Lambert', 'Mid albedo & Specular',
    #             'High albedo & Lambert', 'High albedo & Specular'], loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)  
    if xlabel == 'on':
        plt.xlabel('Orbital Phase', fontsize = 13)
    if ylabel == 'on':
        plt.ylabel(r'$F_p/F_*$ (ppm)', fontsize = 13)
    # 添加文字和箭头，设置字体透明度  
    # plt.text(0.3948, 0, '1 hour', fontsize=10,fontweight='bold', color='black', ha='center') 
    # # plt.text(0.3948, -5, '', fontsize=12, color='black', ha='center')
    # plt.text(0.5, -4, '0.63 h', fontsize=10,fontweight='bold', color='black', ha='center')
    # # plt.text(0.3948, -3, '1 hour', fontsize=12, color='black', ha='center')
    # plt.text(0.6052, 0, '1 hour', fontsize=10,fontweight='bold', color='black', ha='center')
    # # plt.text(0.3948, -3, '1 hour', fontsize=12, color='black', ha='center')
    
    # plt.text(0.5,-100, 'NIRCam/F444W', fontsize = 10, ha='center', fontweight='bold')
    # plt.text(0.5,-120, '3.9-5 μm', fontsize = 10, ha='center', fontweight='bold')
    plt.text(0.79, up_bound *0.9 , instrument+'\n'+f'{wave_range[0]*1e6 :.2f}-{wave_range[1]*1e6 :.2f} μm', fontsize = 10, ha='center', fontweight='bold') # label instrument name and wavelength range
    # plt.text(0.81, up_bound *0.92, f'{wave_range[0]*1e6 :.2f}-{wave_range[1]*1e6 :.2f} μm', fontsize = 10, ha='center', fontweight='bold')
    # plt.axis([0, 1, -5, 40])
    Transit_depth = PPs.Rp**2 / PPs.Rs**2 * 1e6
    print(f'Transit depth: {Transit_depth:.2f} ppm')

    # plt.savefig(f"temp/{name}/phase_curve_comp1.pdf", format = 'pdf')
    ins_name = instrument.replace('/','_').replace('\n', '')
    plt.savefig(f"temp/phase_curve_comp_{ins_name}_nt.pdf", format = 'pdf', bbox_inches='tight')
    plt.savefig(f"temp/phase_curve_comp_{ins_name}_nt.pdf", format = 'pdf', bbox_inches='tight')
    plt.show()
    plt.close()
    
    
if __name__ == '__main__':
    # mcmc = MCMC('K2-141b', 'Kepler', sigma=2.5, ndim=7, nwalkers=64, nsteps=2000, burnin=1000)
    
    An_list = [0.05]
    compare_phase_curve_plot_transit(An_list, np.array([0.33, 1.1])* 1e-6, instrument = 'CHEOPS', legend = 'insert', xlabel = 'on', ylabel='on', errorbar=38.73)
    compare_phase_curve_plot_transit(An_list, np.array([0.80, 1.15])* 1e-6, instrument = 'HST/WFC3/G102', legend = 'off', xlabel='on', ylabel='on', errorbar=5.9287)
    # compare_phase_curve_plot_transit(An_list, np.array([1.075, 1.70])* 1e-6, instrument = 'HST/WFC3/G141', legend = 'off', xlabel = 'on', ylabel='on', errorbar=6.85)
    # compare_phase_curve_plot_transit(An_list, np.array([2.7, 4.0])* 1e-6, instrument = 'JWST/NIRCam/F322W2', legend = 'off', xlabel = 'on', ylabel='off', errorbar=5.14)
    # compare_phase_curve_plot_transit(An_list, np.array([1.5, 2])* 1e-6, instrument = 'JWST/NIRISS/F150W', legend = 'off', xlabel = 'on', ylabel='on', errorbar=288.1/np.sqrt(0.97*3600))
    
    compare_phase_curve_plot_transit(An_list, np.array([0.70, 1.27])* 1e-6, instrument = 'JWST/NIRSpec\nG140M/F070LP', legend = 'off', xlabel = 'on', ylabel='on', errorbar=2.985)
    # compare_phase_curve_plot_transit(An_list, np.array([1.66, 3.07])* 1e-6, instrument = 'JWST/NIRSpec\nG235M/F170LP', legend = 'off', xlabel='on', ylabel='on', errorbar=7.40)
    # compare_phase_curve_plot_transit(An_list, np.array([2.87, 5.10])* 1e-6, instrument = 'JWST/NIRSpec\nG395M/F290LP', legend = 'off', xlabel = 'on', ylabel='on', errorbar=6.85)
    compare_phase_curve_plot_transit(An_list, np.array([0.6, 5.30])* 1e-6, instrument = 'JWST/NIRSpec\nPRISM', legend = 'off', xlabel = 'on', ylabel='on', errorbar=1.269)