import os
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from core_comb.analytical_comb import Fp2Fs
import arviz as az
from multiprocessing import Pool
os.environ["OMP_NUM_THREADS"] = "1"
from scipy.stats import truncnorm
from parameters import PPs

class MCMC:
    def __init__(self, target_name, file_name,  sigma=10, ndim=9, nwalkers=120, nsteps=2000, burnin=1000):
        """
        初始化 MCMC 类。
        
        :param file_name: 数据文件名（不含扩展名)
        :param sigma: 噪声标准差（默认 10 ppm)
        :param ndim: 参数维度（默认 3)
        :param nwalkers: 游走者数量
        :param nsteps: 采样步数
        :param burnin: 烧入期步数
        """
        self.file_name = file_name
        self.target_name = target_name
        self.sigma = sigma
        self.ndim = ndim
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.burnin = burnin
        self.labels = ["A_diffuse", "A_specular", "alpha_ellip", "delta", "Tss", "Rp/Rs", "F", "inc", "alpha"]
        self.Co1, self.Co2 = PPs.Coefficents
        
        # 加载数据, 使用 os.path.join 构建跨平台的文件路径
        folder = os.path.join('Target', target_name)
        os.environ['FOLDER_PATH'] = folder
        path = os.path.join('Target', target_name, f'{file_name}.txt')
        data = np.loadtxt(path, delimiter=',')
        self.data_X = data[:, 0] * 2 * np.pi
        # 为了避免出现负值，将所有小于0的数加上 2 pi
        self.data_X = np.where(self.data_X < 0, self.data_X + 2 * np.pi, self.data_X)
        self.data_Y = data[:, 1]

    def log_likelihood(self, params):
        """对数似然函数"""
        model = Fp2Fs(self.data_X, co1= self.Co1, co2= self.Co2, params=params)
        return -0.5 * np.sum((self.data_Y - model) ** 2 / self.sigma**2 + np.log(2 * np.pi * self.sigma**2))
    
    def log_prior(self, params):
        """对数先验函数"""
        AB_D, AB_S, alpha_ellip, delta, Tss, Rp2Rs, F, inc, alpha = params
        
        # AB_D: 均匀分布 [0, 0.7]
        if not (0 <= AB_D <= 0.7):
            return -np.inf
        log_prior_ABD = 0.0  # 均匀分布的对数概率为常数
        
        # AB_S: 均匀分布 [0, 0.3]
        if not (0 <= AB_S <= 0.7):
            return -np.inf
        log_prior_ABS = 0.0  # 均匀分布的对数概率为常数
        
        if not (0 <= AB_D + AB_S < 1):  # 确保 AB_D + AB_S < 1
            return -np.inf
        
        # alpha_ellip: 非负正态分布，mu=5, sigma=5
        if alpha_ellip < 0:
            return -np.inf  # 确保非负
        mu, sigma = 3.0, 1.0
        log_prior_alpha_ellip = -0.5 * ((alpha_ellip - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))
        # 注意：这里未完全归一化截断正态分布，但对 MCMC 影响不大（仅影响常数项）
        
        # delta: 正态分布，mu=-5, sigma=3
        mu, sigma = 0, 1.0
        log_prior_delta = -0.5 * ((delta - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))
        
        # Tss: 正态分布，mu=Tss_ref, sigma=200
        mu, sigma = PPs.Tss*0.9, 150
        if Tss > PPs.Tss *1.1:
            return -np.inf
        log_prior_Tss = -0.5 * ((Tss - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))
        
        # Rp2Rs: 正态分布，mu=PPs.Rp2Rs, sigma=0.1
        mu, sigma = PPs.Rp2Rs, 0.02 * PPs.Rp2Rs
        log_prior_Rp2Rs = -0.5 * ((Rp2Rs - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))
        
        # F: 非负正态分布，mu=0, sigma=0.05
        if F < 0 or F > 0.5:
            return -np.inf  # 确保非负
        mu, sigma = 0, 0.03
        log_prior_F = -0.5 * ((F - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))
        
        # inc: inc<90的半正态分布，mu=90, sigma=5
        if inc < 75 or inc > 90:
            return -np.inf
        mu, sigma = 86.3, 3.1
        log_prior_inc = -0.5 * ((inc - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))
        
        # alpha: 正态分布，mu=PPs.alpha, sigma=0.02*PPs.alpha
        mu, sigma = PPs.alpha, 0.02 * PPs.alpha
        log_prior_alpha = -0.5 * ((alpha - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))

        return log_prior_ABD+ log_prior_ABS + log_prior_alpha_ellip + log_prior_delta + log_prior_Tss + log_prior_Rp2Rs + log_prior_F + log_prior_inc + log_prior_alpha
    
    def log_posterior(self, params):
        """对数后验函数"""
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(params)
    
    def sample(self):
        """运行 MCMC 采样并保存样本"""
        # initialize the walkers positions
        initial = np.zeros((self.nwalkers, self.ndim))
        initial[:, 0] = np.random.uniform(0, 0.5, self.nwalkers)  # AB_D
        initial[:, 1] = np.random.uniform(0, 0.5, self.nwalkers)  # AB_S
        initial[:, 2] = np.abs(np.random.normal(loc=3.0, scale=1.0, size = self.nwalkers))  # alpha_elips
        # delta
        mu, sigma = 0, 1.0
        initial[:, 3] = np.random.normal(loc=mu, scale=sigma, size=self.nwalkers)
        # Tss
        mu, sigma = PPs.Tss *0.9, 150
        a = (0 - mu) / sigma
        b = (PPs.Tss * 1.1 - mu) / sigma
        initial[:, 4] = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=self.nwalkers)
        # Rp2Rs
        mu, sigma = PPs.Rp2Rs, 0.02 * PPs.Rp2Rs
        initial[:, 5] = np.random.normal(loc=mu, scale=sigma, size=self.nwalkers)
        # F
        mu, sigma = 0, 0.03
        initial[:, 6] = np.abs(np.random.normal(loc=mu, scale=sigma, size=self.nwalkers))
        # inc
        mu, sigma = 86.3, 3.1
        a = (75 - mu) / sigma
        b = (90 - mu) / sigma
        initial[:, 7] = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=self.nwalkers)
        # alpha
        mu, sigma = PPs.alpha, 0.02 * PPs.alpha
        initial[:, 8] = np.random.normal(loc=mu, scale=sigma, size=self.nwalkers)

        # Create the EnsembleSampler object using a multiprocessing pool
        with Pool() as pool:  # multiprocessing 多进程池
            sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_posterior, pool=pool)
            sampler.run_mcmc(initial, self.nsteps, progress=True)
            
        # 保存样本（展平后的样本，去除烧入期)
        samples = sampler.get_chain(discard=self.burnin, flat=True)
        path = os.path.join('Target', self.target_name, f'{self.file_name}_mcmc_samples.npy')
        np.save(path, samples)
        
        # 保存整个链以便绘制迹线图
        self.chain = sampler.get_chain()
        # print("type of chain: ", type(self.chain))

        # 计算 Gelman-Rubin 统计量以评估收敛性
        self.sampler = sampler
        trace = az.from_emcee(sampler)
        self.r_hat = az.rhat(trace)
        r_hat_values = self.r_hat.to_array().values
        print("R_hat for each parameter:", r_hat_values)
        
        return samples
    
    def load_samples(self):
        """加载保存的 MCMC 样本"""
        path = os.path.join('Target', self.target_name, f'{self.file_name}_mcmc_samples.npy')
        return np.load(path)
    
    def plot_fit(self, samples=None, num_samples=100):
        """绘制观测数据与模型预测的拟合图"""
        if samples is None:
            samples = self.load_samples()
        
        inds = np.random.randint(len(samples), size=num_samples)
        plt.figure(figsize=(10, 6))
        for ind in inds:
            sample = samples[ind]
            model_pred = Fp2Fs(self.data_X, co1=self.Co1, co2=self.Co2, params=sample)
            plt.plot(self.data_X / (2 * np.pi), model_pred, "C1", alpha=0.1)
        plt.errorbar(self.data_X / (2 * np.pi), self.data_Y, yerr=self.sigma, fmt=".k", capsize=0, label="Data")
        plt.xlabel("Phase (normalized)")
        plt.ylabel("Flux")
        plt.legend()
        path = os.path.join('Target', self.target_name, f'{self.file_name}_model_fit.pdf')
        plt.savefig(path, format='pdf')
        plt.close()
    
    def plot_trace(self):
        """绘制迹线图以检查收敛性"""
        if not hasattr(self, 'chain'):
            print("请先运行 sample() 方法以获取链。")
            return
        
        plt.figure(figsize=(10, 8))
        for i in range(self.ndim):
            plt.subplot(self.ndim, 1, i+1)
            plt.plot(self.chain[:, :, i], "k", alpha=0.3)
            plt.ylabel(self.labels[i])
        plt.xlabel("Step number")
        path = os.path.join('Target', self.target_name, f'{self.file_name}_trace.pdf')
        plt.savefig(path, format='pdf')
        plt.close()
    
    def plot_corner(self, samples=None):
        """绘制角图展示后验分布"""
        if samples is None:
            samples = self.load_samples()
        
        fig = corner.corner(samples, labels=self.labels)
        path = os.path.join('Target', self.target_name, f'{self.file_name}_corner.pdf')
        plt.savefig(path, format='pdf')
        plt.close()
        
    def compute_rhat(self):
        """独立计算 Gelman-Rubin 统计量以评估收敛性"""
        if not hasattr(self, 'chain'):
            print("请先运行 sample() 方法以获取链。")
            return None
        
        # 从 emcee 的 sampler 中创建 ArviZ 兼容的 trace 对象
        trace = az.from_emcee(self.sampler)
        # 计算 Gelman-Rubin 统计量
        r_hat = az.rhat(trace)
        print("R_hat for each parameter:")
        for param, value in r_hat.items():
            print(f"{param}: {value:.3f}")
        return r_hat
    
    def estimate_parameters(self, samples=None):
        """
        根据 samples 估计参数值以及 16% 和 84% 分位数作为上下不确定度，并打印结果。
        
        如果未传入 samples, 则调用 load_samples 加载保存的样本。
        """
        if samples is None:
            samples = self.load_samples()  # shape: (num_samples, ndim)

        medians = np.percentile(samples, 50, axis=0)
        lower = np.percentile(samples, 16, axis=0)
        upper = np.percentile(samples, 84, axis=0)
        # save the results to MCMC
        self.medians = medians

        lower_errors = medians - lower
        upper_errors = upper - medians

        for i in range(self.ndim):
            print(f"{self.labels[i]}: {medians[i]:.4f} -{lower_errors[i]:.4f} / +{upper_errors[i]:.4f}")
        return medians, lower, upper