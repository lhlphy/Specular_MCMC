# Nested Sampling for K2-141b Limb Models

This is a self-contained `dynesty` project for comparing three K2-141b limb models with Bayesian evidence and posterior predictive checks. It does not import modules from the parent MCMC project at runtime.

## Install

Use the same scientific Python environment as the MCMC project, then install the extra dependency:

```powershell
cd F:\26_5\MCMC\Nested_Sampling
C:\Users\dell\.conda\envs\MCMC\python.exe -m pip install -r requirements.txt
```

## Quick Check

This verifies that the copied models, priors, data files, and likelihoods are internally consistent:

```powershell
cd F:\26_5\MCMC\Nested_Sampling
C:\Users\dell\.conda\envs\MCMC\python.exe -m nested_sampler.quick_check
```

## Run

Run all three limb models with the default medium-precision settings:

```powershell
cd F:\26_5\MCMC\Nested_Sampling
C:\Users\dell\.conda\envs\MCMC\python.exe -m nested_sampler.run --model all
```

Use local multiprocessing workers:

```powershell
C:\Users\dell\.conda\envs\MCMC\python.exe -m nested_sampler.run --model all --workers 8
```

Submit the 32-core server job from this directory:

```bash
sbatch run_nested_32core.sh
```

Run a fast single-model smoke test:

```powershell
C:\Users\dell\.conda\envs\MCMC\python.exe -m nested_sampler.run --model specular_limb --nlive-init 50 --nlive-batch 25 --dlogz-init 1.0 --maxbatch 0 --max-data-points 5 --quiet
```

## Outputs

Each model writes to `results/<model>/`:

- `nested_results.npz`
- `posterior_samples.csv`
- `summary.json`
- `corner.pdf`
- `posterior_predictive.pdf`

The three-model comparison writes to `results/comparison/`:

- `evidence_summary.csv`
- `delta_logz.csv`
- `model_compare.pdf`
