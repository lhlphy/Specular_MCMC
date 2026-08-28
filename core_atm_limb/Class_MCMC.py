import os
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from core_atm_limb.analytical_model_atm import Fp2Fs
import arviz as az
from multiprocessing import Pool
from scipy.stats import truncnorm
from parameters import PPs

os.environ["OMP_NUM_THREADS"] = "1"


class MCMC:
    def __init__(self, target_name, file_name, sigma=10, ndim=13, nwalkers=120, nsteps=2000, burnin=1000):
        self.file_name = file_name
        self.target_name = target_name
        self.ndim = ndim
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.burnin = burnin
        self.labels = [r"$\omega$", "g", r"$H\ ({\rm km})$", r"$T_{\rm sub}$", "Rp/Rs", "F", "inc", r"$\alpha$", "u1", "u2", "delta", r"$A_{\rm th}$", r"$x_{\rm offset}$"]
        self.Co1, self.Co2 = PPs.Coefficents
        self.coeff_sigma = 0.1

        folder = os.path.join('Target', target_name)
        os.environ['FOLDER_PATH'] = folder
        path = os.path.join('Target', target_name, f'{file_name}.txt')
        data, self.sigma = self._load_observations(path, sigma)
        self.data_X = data[:, 0] * 2 * np.pi
        self.data_X = np.where(self.data_X < 0, self.data_X + 2 * np.pi, self.data_X)
        self.data_Y = data[:, 1]
        print(f"Observation data: {path}")
        print(f"Errorbar source: {self.errorbar_source}")

    def _load_observations(self, path, main_sigma):
        rows = []
        with open(path, "r", encoding="utf-8-sig") as handle:
            for raw_line in handle:
                text = raw_line.strip()
                if not text or text.startswith("#"):
                    continue
                parts = text.replace(",", " ").split()
                if len(parts) < 2:
                    continue
                try:
                    rows.append([float(value) for value in parts[:3]])
                except ValueError:
                    continue

        if not rows:
            raise ValueError(f"No numeric observation rows found in {path}.")

        column_count = max(len(row) for row in rows)
        if column_count not in (2, 3) or any(len(row) != column_count for row in rows):
            raise ValueError(f"{path} must contain consistently two or three numeric columns.")

        data = np.asarray(rows, dtype=float)
        if column_count == 3:
            if np.any(data[:, 2] <= 0):
                raise ValueError(f"{path} contains non-positive errorbar values.")
            self.errorbar_source = f"third column in {os.path.basename(path)}"
            print(
                f"WARNING: {path} has three columns; using the third-column errorbar. "
                f"The errorbar specified in main.py (sigma={main_sigma}) is ignored."
            )
            return data[:, :2], data[:, 2]

        self.errorbar_source = f"main.py sigma={main_sigma}"
        return data, float(main_sigma)

    def _samples_path(self):
        return os.path.join('Target', self.target_name, f'{self.file_name}_mcmc_samples.npy')

    def _full_chain_path(self):
        return os.path.join('Target', self.target_name, f'{self.file_name}_mcmc_chain.npy')

    def _sample_log_likelihood_path(self):
        return os.path.join('Target', self.target_name, f'{self.file_name}_mcmc_log_likelihood.npy')

    def _sample_log_posterior_path(self):
        return os.path.join('Target', self.target_name, f'{self.file_name}_mcmc_log_posterior.npy')

    def _full_log_likelihood_path(self):
        return os.path.join('Target', self.target_name, f'{self.file_name}_mcmc_log_likelihood_chain.npy')

    def _full_log_posterior_path(self):
        return os.path.join('Target', self.target_name, f'{self.file_name}_mcmc_log_posterior_chain.npy')

    def load_full_chain(self):
        return np.load(self._full_chain_path())

    def _save_chain_products(self, chain, log_posterior_chain=None, log_likelihood_chain=None):
        np.save(self._full_chain_path(), chain)
        samples = chain[self.burnin:, :, :].reshape(-1, self.ndim)
        np.save(self._samples_path(), samples)
        if log_posterior_chain is not None and log_likelihood_chain is not None:
            np.save(self._full_log_posterior_path(), log_posterior_chain)
            np.save(self._full_log_likelihood_path(), log_likelihood_chain)
            np.save(self._sample_log_posterior_path(), log_posterior_chain[self.burnin:, :].reshape(-1))
            np.save(self._sample_log_likelihood_path(), log_likelihood_chain[self.burnin:, :].reshape(-1))
        return samples

    def _compute_rhat_from_chain(self, chain):
        post_burnin = chain[self.burnin:, :, :]
        if post_burnin.shape[0] < 2:
            raise ValueError("Need at least two post-burn-in draws to compute R_hat.")
        dataset = az.convert_to_dataset({"posterior": np.transpose(post_burnin, (1, 0, 2))})
        return az.rhat(dataset)["posterior"].values

    def _score_chain(self, chain):
        flat_chain = chain.reshape(-1, self.ndim)
        scored = [self.log_posterior(params) for params in flat_chain]
        log_posterior, log_likelihood = np.asarray(scored, dtype=float).T
        return (
            log_posterior.reshape(chain.shape[:2]),
            log_likelihood.reshape(chain.shape[:2]),
        )

    def log_likelihood(self, params):
        model = Fp2Fs(self.data_X, co1=self.Co1, co2=self.Co2, params=params)
        return -0.5 * np.sum((self.data_Y - model) ** 2 / self.sigma**2 + np.log(2 * np.pi * self.sigma**2))

    def log_prior(self, params):
        omega, g, H, Tss, Rp2Rs, F, inc, alpha, co1, co2, delta, A_th, x_offset = params

        if not (0 <= omega <= 1):
            return -np.inf
        log_prior_omega = 0.0

        if not (-0.999 <= g <= 0.999):
            return -np.inf
        log_prior_g = 0.0

        if not (27 <= H <= 89.8):
            return -np.inf
        log_prior_H = 0.0

        mu, sigma = 0.95 * PPs.Tss, 45 * 2
        if Tss > PPs.Tss * 1.2:
            return -np.inf
        log_prior_Tss = -0.5 * ((Tss - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))

        mu, sigma = PPs.Rp2Rs, 0.02662 * PPs.Rp2Rs
        log_prior_Rp2Rs = -0.5 * ((Rp2Rs - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))

        if F < 0.3 or F > 0.5:
            return -np.inf
        log_prior_F = 0.0

        if inc < 70 or inc > 90:
            return -np.inf
        mu, sigma = 75.2, 2.4
        log_prior_inc = -0.5 * ((inc - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))

        mu, sigma = PPs.alpha, 0.02312 * PPs.alpha
        log_prior_alpha = -0.5 * ((alpha - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))

        if co1 < 0 or co2 < 0:
            return -np.inf
        a1 = (0 - self.Co1) / self.coeff_sigma
        a2 = (0 - self.Co2) / self.coeff_sigma
        log_prior_co1 = truncnorm.logpdf(co1, a1, np.inf, loc=self.Co1, scale=self.coeff_sigma)
        log_prior_co2 = truncnorm.logpdf(co2, a2, np.inf, loc=self.Co2, scale=self.coeff_sigma)

        if not (-10 <= delta <= 10):
            return -np.inf
        log_prior_delta = 0.0

        if not (0 <= A_th <= 0.7):
            return -np.inf
        log_prior_A_th = 0.0

        if not (-0.03 <= x_offset <= 0.03):
            return -np.inf
        log_prior_x_offset = 0.0

        return (
            log_prior_omega
            + log_prior_g
            + log_prior_H
            + log_prior_Tss
            + log_prior_Rp2Rs
            + log_prior_F
            + log_prior_inc
            + log_prior_alpha
            + log_prior_co1
            + log_prior_co2
            + log_prior_delta
            + log_prior_A_th
            + log_prior_x_offset
        )

    def log_posterior(self, params):
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf, -np.inf
        ll = self.log_likelihood(params)
        return lp + ll, ll

    def sample(self, resume=False, nsteps=None):
        run_steps = self.nsteps if nsteps is None else int(nsteps)
        prior_chain = None
        prior_log_posterior = None
        prior_log_likelihood = None
        if resume:
            prior_chain = self.load_full_chain()
            try:
                prior_log_posterior = np.load(self._full_log_posterior_path())
                prior_log_likelihood = np.load(self._full_log_likelihood_path())
            except FileNotFoundError:
                print("Saved log arrays not found; scoring existing chain before resume.")
                prior_log_posterior, prior_log_likelihood = self._score_chain(prior_chain)
            initial = prior_chain[-1]
            if initial.shape != (self.nwalkers, self.ndim):
                raise ValueError("Saved chain shape does not match nwalkers/ndim.")
        else:
            initial = np.zeros((self.nwalkers, self.ndim))
        initial[:, 0] = np.random.uniform(low=0.0, high=1.0, size=self.nwalkers)
        initial[:, 1] = np.random.uniform(low=-0.999, high=0.999, size=self.nwalkers)
        initial[:, 2] = np.random.uniform(low=27.0, high=89.8, size=self.nwalkers)

        mu, sigma = 0.95 * PPs.Tss, 45 * 2
        a = (0 - mu) / sigma
        b = (PPs.Tss * 1.2 - mu) / sigma
        initial[:, 3] = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=self.nwalkers)

        mu, sigma = PPs.Rp2Rs, 0.02662 * PPs.Rp2Rs
        initial[:, 4] = np.random.normal(loc=mu, scale=sigma, size=self.nwalkers)

        initial[:, 5] = np.random.uniform(low=0.3, high=0.5, size=self.nwalkers)

        mu, sigma = 75.2, 2.4
        a = (70 - mu) / sigma
        b = (90 - mu) / sigma
        initial[:, 6] = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=self.nwalkers)

        mu, sigma = PPs.alpha, 0.02312 * PPs.alpha
        initial[:, 7] = np.random.normal(loc=mu, scale=sigma, size=self.nwalkers)

        a = (0 - self.Co1) / self.coeff_sigma
        initial[:, 8] = truncnorm.rvs(a, np.inf, loc=self.Co1, scale=self.coeff_sigma, size=self.nwalkers)

        a = (0 - self.Co2) / self.coeff_sigma
        initial[:, 9] = truncnorm.rvs(a, np.inf, loc=self.Co2, scale=self.coeff_sigma, size=self.nwalkers)

        initial[:, 10] = np.random.uniform(low=-10.0, high=10.0, size=self.nwalkers)
        initial[:, 11] = np.random.uniform(low=0.0, high=0.7, size=self.nwalkers)
        initial[:, 12] = np.random.uniform(low=-0.03, high=0.03, size=self.nwalkers)

        with Pool() as pool:
            sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_posterior, pool=pool)
            sampler.run_mcmc(initial, run_steps, progress=True)

        new_chain = sampler.get_chain()
        new_log_posterior = sampler.get_log_prob()
        new_log_likelihood = np.asarray(sampler.get_blobs(), dtype=float)

        if prior_chain is not None:
            chain = np.concatenate([prior_chain, new_chain], axis=0)
            log_posterior_chain = np.concatenate([prior_log_posterior, new_log_posterior], axis=0)
            log_likelihood_chain = np.concatenate([prior_log_likelihood, new_log_likelihood], axis=0)
        else:
            chain = new_chain
            log_posterior_chain = new_log_posterior
            log_likelihood_chain = new_log_likelihood

        samples = self._save_chain_products(chain, log_posterior_chain, log_likelihood_chain)

        self.chain = chain
        self.sampler = sampler
        try:
            self.r_hat = self._compute_rhat_from_chain(chain)
            print("R_hat for each parameter:", self.r_hat)
        except ValueError as exc:
            print(f"R_hat skipped: {exc}")

        return samples

    def load_samples(self):
        return np.load(self._samples_path())

    def load_log_likelihood(self):
        return np.load(self._sample_log_likelihood_path())

    def load_log_posterior(self):
        return np.load(self._sample_log_posterior_path())

    def chi2_cal(self, dataY, data_Model, errorbar):
        return np.sum(((dataY - data_Model) / errorbar) ** 2) / len(dataY)

    def plot_fit(self, samples=None, num_samples=100):
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
        path = os.path.join('Target', self.target_name, f'{self.file_name}_model_fit1.pdf')
        plt.savefig(path, format='pdf')
        plt.ylim([-10, np.max(self.data_Y) * 1.1])
        path = os.path.join('Target', self.target_name, f'{self.file_name}_model_fit2.pdf')
        plt.savefig(path, format='pdf')
        plt.close()

    def plot_trace(self):
        if not hasattr(self, 'chain'):
            print("Please run sample() first.")
            return

        plt.figure(figsize=(10, 8))
        for i in range(self.ndim):
            plt.subplot(self.ndim, 1, i + 1)
            plt.plot(self.chain[:, :, i], "k", alpha=0.3)
            plt.ylabel(self.labels[i])
        plt.xlabel("Step number")
        path = os.path.join('Target', self.target_name, f'{self.file_name}_trace.pdf')
        plt.savefig(path, format='pdf')
        plt.close()

    def plot_corner(self, samples=None):
        if samples is None:
            samples = self.load_samples()

        fig = corner.corner(samples, labels=self.labels)
        for ax in fig.get_axes():
            ax.tick_params(axis='both', labelsize=11.5)
            ax.xaxis.label.set_size(17)
            ax.yaxis.label.set_size(17)
        path = os.path.join('Target', self.target_name, f'{self.file_name}_corner.pdf')
        plt.savefig(path, format='pdf')
        plt.close()

    def plot_omega_distribution(self, samples=None):
        if samples is None:
            samples = self.load_samples()

        subset_samples = samples[:, [0]]
        fig = corner.corner(subset_samples, labels=[self.labels[0]])
        ax = fig.axes[0]
        ax.set_ylabel("Density", fontsize=14)
        ax.tick_params(axis='both', labelsize=11.5)
        ax.xaxis.label.set_size(17)
        y_min, y_max = ax.get_ylim()
        yticks = np.linspace(y_min, y_max, 5)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{(y / 115200):.2f}" for y in yticks], fontsize=11.5)
        path = os.path.join('Target', self.target_name, f'{self.file_name}_omega_distribution.pdf')
        plt.savefig(path, format='pdf', bbox_inches='tight')
        plt.close()

    def compute_rhat(self):
        if not hasattr(self, 'chain'):
            self.chain = self.load_full_chain()

        r_hat = self._compute_rhat_from_chain(self.chain)
        print("R_hat for each parameter:")
        for i, value in enumerate(r_hat):
            print(f"var_{i}: {value:.3f}")
        return r_hat

    def estimate_parameters(self, samples=None):
        if samples is None:
            samples = self.load_samples()

        medians = np.percentile(samples, 50, axis=0)
        lower = np.percentile(samples, 16, axis=0)
        upper = np.percentile(samples, 84, axis=0)
        self.medians = medians

        model_pred = Fp2Fs(self.data_X, co1=self.Co1, co2=self.Co2, params=medians)
        self.chi2 = self.chi2_cal(self.data_Y, model_pred, self.sigma)
        print("Chi2: ", self.chi2)

        lower_errors = medians - lower
        upper_errors = upper - medians

        for i in range(self.ndim):
            print(f"{self.labels[i]}: {medians[i]:.5f} -{lower_errors[i]:.5f} / +{upper_errors[i]:.5f}")
        return medians, lower, upper
