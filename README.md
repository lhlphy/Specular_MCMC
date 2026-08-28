## Markov Chain Monte Carlo (MCMC) Analysis of Phase Curve Models and Observational Data for Exoplanets

Predefine Lambert sacttering, specular reflection and atmospheric scattering models.

### Inputs

- **Phase curve model:** Spectral model: `./core_limb`; Lambert model: `./core_lambert_limb`; Atmospheric model: `./core_atm_limb`
- **Observation target:** Defined in: `parameters.py`, `Sampling.py`
- **Observation data:** Kepler data for the target
- **Prior distribution of parameters**

### Attention

- Read and follow `./check_list.py` to verify settings after changing the target.

### Outputs

- **Posterior distribution of parameters**

### Run
- run_specular.sh : Runs the MCMC analysis for the specular-reflection model.
- run_diffuse.sh : Runs the MCMC analysis for the diffuse-reflection model.
- run_atmosphere.sh : Runs the MCMC analysis for the atmospheric model.