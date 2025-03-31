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
mcmc = core.Class_MCMC.MCMC('K2-141b', 'Kepler', sigma=2.5, ndim=7, nwalkers=64, nsteps=3000, burnin=1500)
mcmc_lambert = core_lambert.Class_MCMC.MCMC('K2-141b_lambert', 'Kepler', sigma=2.5, ndim=7, nwalkers=64, nsteps=3000, burnin=1500)
print("\nThe parameters of the Specular model: ")
params, lower, upper = mcmc.estimate_parameters() # estimate the parameters and print them
print("\nThe parameters of the Lambert model: ")
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
    return np.sum(((dataY - data_Model) / errorbar)**2) / len(dataY)

chi2_value = chi2(dataY, data_model, mcmc.sigma)
chi2_value_lambert = chi2(dataY, data_model_l, mcmc_lambert.sigma)

# print the chi2 results
print("\nchi2 for Specular model is ", chi2_value)
print("chi2 for Lambert  model is ", chi2_value_lambert)
if chi2_value < chi2_value_lambert:
    print("Specular model is better than Lambert model")
else:
    print("Lambert model is better than Specular model")

# plot the boundary of the 95% confidence interval
data_model_lower = core.analytical_model.Fp2Fs(dataX, co1=Co1, co2=Co2, params=lower)
data_model_upper = core.analytical_model.Fp2Fs(dataX, co1=Co1, co2=Co2, params=upper)
data_model_l_lower = core_lambert.analytical_model_Lambert.Fp2Fs(dataX, co1=Co1_l, co2=Co2_l, params=lower_l)
data_model_l_upper = core_lambert.analytical_model_Lambert.Fp2Fs(dataX, co1=Co1_l, co2=Co2_l, params=upper_l)

# plot the data and model
fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(dataX, dataY, yerr=mcmc.sigma, fmt='o', color='k', label='Data',markersize=3)
# 绘制Specular和Lambert模型的拟合曲线
ax.plot(dataX, data_model, '-', color='red', linewidth=2, label='Specular Model')
ax.plot(dataX, data_model_l, '-', color='blue', linewidth=2, label='Lambert Model')
# 绘制95%置信区间
ax.fill_between(dataX, data_model_lower, data_model_upper,
                alpha=0.5, color='red', label='Specular Model 95% CI')
ax.fill_between(dataX, data_model_l_lower, data_model_l_upper,
                alpha=0.5, color='blue', label='Lambert Model 95% CI')
ax.set_xlabel('Orbital phase')
ax.set_ylabel('Fp/Fs (ppm)')
ax.legend()

plt.savefig('./output/Kepler_specular_vs_lambert.png')
ax.set_ylim(-20, (np.max(dataY)+mcmc.sigma) *1.1)  # 设置下限为-10，上限自动调整
plt.savefig('./output/Kepler_specular_vs_lambert_nT.png')
plt.show()
plt.close()



