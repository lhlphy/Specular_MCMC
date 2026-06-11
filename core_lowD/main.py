import os
import sys
import warnings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from Class_MCMC import MCMC


if __name__ == '__main__':
    resume = os.environ.get("MCMC_RESUME", "0") == "1"
    extra_steps = os.environ.get("MCMC_EXTRA_STEPS")
    run_steps = int(extra_steps) if extra_steps else None

    mcmc = MCMC('K2-141b_lowD', 'Kepler', sigma=7.05, ndim=6, nwalkers=64, nsteps=4000, burnin=1500)
    mcmc.sample(resume=resume, nsteps=run_steps)
    mcmc.plot_trace()
    mcmc.plot_corner()
    mcmc.compute_rhat()
    mcmc.estimate_parameters()
    mcmc.plot_fit()

    samples = mcmc.load_samples()
    print("sample shape:", samples.shape)
