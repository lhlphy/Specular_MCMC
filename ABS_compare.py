import numpy as np
import matplotlib.pyplot as plt
# 将./core和./core_Lambert加入到系统路径中, 注意跨平台兼容性
import sys
sys.path.append('./core_comb')
sys.path.append('./core_Lambert')
import core_comb.Class_MCMC
import core_comb.analytical_comb
import core.Class_MCMC
import core.analytical_model
import warnings
warnings.filterwarnings("ignore")

# plot the data and model
fig, ax = plt.subplots(figsize=(8, 6))
AB_S_list = [0.00, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

# generate color map
from color import get_color
color_map = get_color(n = len(AB_S_list))
for i, AB_S in enumerate(AB_S_list):
    mcmc = core_comb.Class_MCMC.MCMC(f'K2-141b_comb{AB_S:.2f}', 'Kepler', sigma=7.5, ndim=9, nwalkers=64, nsteps=3000, burnin=1200)
    params, lower, upper = mcmc.estimate_parameters() # estimate the parameters and print them
    # load transit parameters
    Co1, Co2 = mcmc.Co1, mcmc.Co2
    # calculate the model, upper and lower limits
    dataX_2 = np.linspace(0, 2* np.pi, 300)
    data_model = core_comb.analytical_comb.Fp2Fs(dataX_2, co1=Co1, co2=Co2, params=params, AA = AB_S)
    # data_model_lower = core_comb.analytical_comb.Fp2Fs(dataX_2, co1=Co1, co2=Co2, params=lower, AA = AB_S)
    # data_model_upper = core_comb.analytical_comb.Fp2Fs(dataX_2, co1=Co1, co2=Co2, params=upper, AA = AB_S)
    
    # load Kepler data
    dataX = mcmc.data_X
    dataY = mcmc.data_Y
    # plot the data and model
    ax.errorbar(dataX/(2 *np.pi), dataY, yerr=mcmc.sigma, fmt='o', color='k',markersize=3)
    # 绘制Specular模型的拟合曲线
    ax.plot(dataX_2/(2 *np.pi), data_model, '-', color=color_map[i], linewidth=2, label=f'$A_S$ = {AB_S:.2f}, $\chi^2$ = {mcmc.chi2:.3f}')
    # 绘制95%置信区间
    # ax.fill_between(dataX_2/(2 *np.pi), data_model_lower, data_model_upper, alpha=0.3, color=color_map[i], edgecolor=None)


# # 绘制一个纯specular模型的拟合曲线
# mcmc = core.Class_MCMC.MCMC('K2-141b_specular', 'Kepler', sigma=7.5, ndim=8, nwalkers=64, nsteps=3000, burnin=1200)
# params, lower, upper = mcmc.estimate_parameters() # estimate the parameters and print them
# Co1, Co2 = mcmc.Co1, mcmc.Co2
# dataX_2 = np.linspace(0, 2* np.pi, 300)
# data_model = core.analytical_model.Fp2Fs(dataX_2, co1=Co1, co2=Co2)
# ax.plot(dataX_2/(2 *np.pi), data_model, '-', color='k', linewidth=2, label=f'Specular, $\chi^2$ = {mcmc.chi2:.3f}')
# # load Kepler data
# dataX = mcmc.data_X
# dataY = mcmc.data_Y
# # plot the data and model
# ax.errorbar(dataX/(2 *np.pi), dataY, yerr=mcmc.sigma, fmt='o', color='k',markersize=3)

# 绘图后处理
ax.set_xlabel('Orbital phase')
ax.set_ylabel('Fp/Fs (ppm)')
ax.legend(frameon=False)

plt.savefig('./output/ABS_compare.pdf')
ax.set_ylim(-20, (np.max(dataY)+mcmc.sigma) *1.1)  # 设置下限为-10，上限自动调整
plt.savefig('./output/ABS_compare_nT.pdf')
plt.show()
plt.close()

