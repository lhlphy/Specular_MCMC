import numpy as np
import matplotlib.pyplot as plt
# 将./core和./core_Lambert加入到系统路径中, 注意跨平台兼容性
import sys
sys.path.append('./core')
sys.path.append('./core_Lambert')
import core.Class_MCMC
import core.analytical_model
import core_lambert.Class_MCMC
import core_lambert.analytical_model_Lambert
import core
import core_lambert
import warnings
warnings.filterwarnings("ignore")

# load the parameters from the MCMC class
mcmc = core.Class_MCMC.MCMC('K2-141b', 'Kepler', sigma=7.05, ndim=6, nwalkers=64, nsteps=3000, burnin=1500)
mcmc_lambert = core_lambert.Class_MCMC.MCMC('K2-141b_lambert', 'Kepler', sigma=7.05, ndim=6, nwalkers=64, nsteps=3000, burnin=1500)
print("\nThe parameters of the Specular model: ")
params, lower, upper = mcmc.estimate_parameters() # estimate the parameters and print them
print("\nThe parameters of the Diffuse model: ")
params_l, lower_l, upper_l = mcmc_lambert.estimate_parameters()

# load transit parameters
Co1, Co2 = mcmc.Co1, mcmc.Co2
Co1_l, Co2_l = mcmc_lambert.Co1, mcmc_lambert.Co2

# load Kepler data
dataX = mcmc.data_X
dataY = mcmc.data_Y

# calculate the model
data_model = core.analytical_model.Fp2Fs(dataX, co1=Co1, co2=Co2, params=params)
data_model_l = core_lambert.analytical_model_Lambert.Fp2Fs(dataX, co1=Co1_l, co2=Co2_l, params=params_l)

# calculate the chi2 value
def chi2(dataY, data_Model, errorbar):
    return np.sum(((dataY - data_Model) / errorbar)**2)

chi2_value = chi2(dataY, data_model, mcmc.sigma)
chi2_value_lambert = chi2(dataY, data_model_l, mcmc_lambert.sigma)

# print the chi2 results
print("\nchi2 for Specular model is ", chi2_value)
print("chi2 for Diffuse  model is ", chi2_value_lambert)
if chi2_value < chi2_value_lambert:
    print("Specular model is better than Diffuse model")
else:
    print("Diffuse model is better than Specular model")

# plot the boundary of the 95% confidence interval
data_model_lower = core.analytical_model.Fp2Fs(dataX, co1=Co1, co2=Co2, params=lower)
data_model_upper = core.analytical_model.Fp2Fs(dataX, co1=Co1, co2=Co2, params=upper)
data_model_l_lower = core_lambert.analytical_model_Lambert.Fp2Fs(dataX, co1=Co1_l, co2=Co2_l, params=lower_l)
data_model_l_upper = core_lambert.analytical_model_Lambert.Fp2Fs(dataX, co1=Co1_l, co2=Co2_l, params=upper_l)

# plot the data and model
fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(dataX/(2*np.pi), dataY, yerr=mcmc.sigma, fmt='o', color='k', markersize=3)
# 绘制Specular和Lambert模型的拟合曲线
ax.plot(dataX/(2*np.pi), data_model, '-', color='red', linewidth=2.5, label=f'Specular: $\chi^2$={chi2_value:.2f}')
ax.plot(dataX/(2*np.pi), data_model_l, '-', color='blue', linewidth=2.5, label=f'Diffuse: $\chi^2$={chi2_value_lambert:.2f}')
# 绘制95%置信区间
# ax.fill_between(dataX, data_model_lower, data_model_upper,
#                 alpha=0.5, color='red', label='Specular Model 95% CI')
# ax.fill_between(dataX, data_model_l_lower, data_model_l_upper,
#                 alpha=0.5, color='blue', label='Diffuse Model 95% CI')
ax.set_xlabel('Orbital phase', fontsize=15)
ax.set_ylabel('Fp/Fs (ppm)', fontsize=15)
ax.tick_params(axis='both', labelsize=12)
ax.legend(fontsize=13, frameon=False)

plt.savefig('./output/Kepler_specular_vs_lambert.pdf')
ax.set_ylim(-20, (np.max(dataY)+mcmc.sigma) *1.15)  # 设置下限为-10，上限自动调整
ax.set_xlim(0, 1)
plt.savefig('./output/Kepler_specular_vs_lambert_nT.pdf')
plt.show()
plt.close()

#########################################
# plot A_lambda distribution
# 从两个 MCMC 对象中各取出第 0 维 (A_λ) 的采样
samples_specular = mcmc.load_samples()[:, 0]
samples_lambert = mcmc_lambert.load_samples()[:, 0]

# corner.corner 需要输入 shape=(N, D)，所以对一维数据做 reshape
spec_data = samples_specular.reshape(-1, 1)
lamb_data = samples_lambert.reshape(-1, 1)

# 绘制corner plot
import corner
fig = corner.corner(spec_data, labels=[r"$A_{\lambda}$"], color="red", hist_kwargs={'histtype': 'step', 'linewidth': 1.3})
corner.corner(lamb_data, fig=fig, labels=[r"$A_{\lambda}$"], color="blue", hist_kwargs={'histtype': 'step', 'linewidth': 1.3})

## 手动添加图例（corner 不会自动生成图例）
# import matplotlib.patches as mpatches
# patch_spec = mpatches.Patch(color="red", label="Specular")
# patch_lamb = mpatches.Patch(color="blue", label="Lambert")
# plt.legend(handles=[patch_spec, patch_lamb], loc="best", fontsize=12)

# corner.corner 返回的 axes，若只有一个参数，就只有 1 个轴
ax = fig.axes[0]
# 调整刻度与横轴标签大小
ax.tick_params(axis='both', labelsize=8)
plt.setp(ax.get_xticklabels(), rotation=0)
ax.set_ylabel("Probability density", fontsize=10)  # 为 y 轴添加标签
ax.set_xlabel(r"$A_{\lambda}$", fontsize=10)


# # ax.xaxis.label.set_size(8)
# # 设置 x 轴标签居中，并将其纵向位置调整到更靠近 xticks（具体数值可调）
ax.xaxis.set_label_coords(0.5, -0.12)

# 手动设置 y 轴刻度与标签（此处以 5 个刻度为例）
y_min, y_max = ax.get_ylim()  # 获取当前 y 轴范围
yticks = np.linspace(y_min, y_max, 5)
ax.set_yticks(yticks)
ax.set_yticklabels([f"{(y/115200):.2f}" for y in yticks], fontsize=8) #获取的y轴刻度是频数，不是频率
    
# save the figure
plt.savefig("A_lambda_compare.pdf", bbox_inches="tight")
plt.show()
plt.close()