# Atmospheric Scattering MCMC Core

This folder mirrors the existing `core` and `core_lambert` MCMC workflows, but
uses atmospheric scattering for the reflected-light component.

Parameter order:

1. `omega`: single-scattering albedo
2. `g`: Henyey-Greenstein asymmetry factor
3. `Tss`
4. `Rp/Rs`
5. `F`
6. `inc`
7. `alpha`
8. `alpha_ellip`

The default entry point is:

```powershell
python core_atm\main.py
```

Outputs are written under `Target\K2-141b_atm`.
