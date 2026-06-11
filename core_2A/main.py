import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Class_MCMC import MCMC

TARGET_NAME = os.environ.get("MCMC_TARGET_NAME", "K2-141b_2A")
OBSERVATION_NAME = os.environ.get("MCMC_OBSERVATION_NAME", "Kepler")


if __name__ == '__main__':
    resume = os.environ.get("MCMC_RESUME", "0") == "1"
    extra_steps = os.environ.get("MCMC_EXTRA_STEPS")
    run_steps = int(extra_steps) if extra_steps else None

    print(f"Running target: {TARGET_NAME}")
    print(f"Running observation data: {OBSERVATION_NAME}")
    mcmc = MCMC(TARGET_NAME, OBSERVATION_NAME, sigma=7.05, ndim=10, nwalkers=64, nsteps=4000, burnin=1500)
    mcmc.sample(resume=resume, nsteps=run_steps)
    mcmc.plot_trace()
    mcmc.plot_corner()
    mcmc.compute_rhat()
    mcmc.estimate_parameters()
    mcmc.plot_fit()

    samples = mcmc.load_samples()
    print("sample shape:", samples.shape)
