import os

import numpy as np
from scipy.interpolate import interp1d


H_PLANCK = 6.626e-34
C_LIGHT = 3.0e8
K_BOLTZMANN = 1.38e-23
DEFAULT_TABLE_SIZE = int(os.environ.get("MCMC_THERMAL_TABLE_SIZE", "4096"))
USE_TABLE = os.environ.get("MCMC_THERMAL_USE_TABLE", "1") != "0"

_response_cache = {}
_band_table_cache = {}


def _response_path():
    folder_path = os.environ.get("FOLDER_PATH")
    if folder_path:
        return os.path.join(folder_path, "Response.txt")
    return "Response.txt"


def _load_response_data():
    path = _response_path()
    key = os.path.abspath(path)
    cached = _response_cache.get(key)
    if cached is not None:
        return cached
    try:
        data = np.loadtxt(path, delimiter=",")
    except FileNotFoundError:
        data = None
    _response_cache[key] = data
    return data


def response_values(lam, clip_negative=False):
    data = _load_response_data()
    if data is None:
        return np.ones_like(np.asarray(lam, dtype=float), dtype=float)
    spl = interp1d(
        data[:, 0],
        data[:, 1],
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )
    values = spl(np.asarray(lam, dtype=float) * 1e6)
    if clip_negative:
        values = np.where(values < 0, 0, values)
    return values


def blackbody(lam, temperature):
    lam = np.asarray(lam, dtype=float)
    temperature = np.asarray(temperature, dtype=float)
    safe_temperature = np.where(temperature < 10, 10.0, temperature)
    exponent = H_PLANCK * C_LIGHT / lam / K_BOLTZMANN / safe_temperature
    exponent = np.clip(exponent, None, 700.0)
    values = 2 * H_PLANCK * C_LIGHT**2 / lam**5 / (np.exp(exponent) - 1)
    return np.where(temperature < 10, 0.0, values)


def _direct_band_integral(temperature, lam1, lam2, n_lam, clip_negative):
    lam_grid = np.linspace(lam1, lam2, n_lam)
    response = response_values(lam_grid, clip_negative=clip_negative)
    values = blackbody(lam_grid, np.asarray(temperature)[..., np.newaxis]) * response
    return np.sum(values, axis=-1) * (lam_grid[1] - lam_grid[0])


def band_integral(temperature, lam1, lam2, n_lam=8, clip_negative=False):
    if not USE_TABLE:
        return _direct_band_integral(temperature, lam1, lam2, n_lam, clip_negative)

    temperature_array = np.asarray(temperature, dtype=float)
    finite = temperature_array[np.isfinite(temperature_array)]
    max_temperature = max(10.0, float(np.max(finite)) if finite.size else 10.0)
    table_max = max(5000.0, max_temperature * 1.05)
    response_key = os.path.abspath(_response_path())
    cache_key = (
        response_key,
        float(lam1),
        float(lam2),
        int(n_lam),
        bool(clip_negative),
        DEFAULT_TABLE_SIZE,
        int(np.ceil(table_max / 100.0) * 100),
    )
    cached = _band_table_cache.get(cache_key)
    if cached is None:
        t_max = cache_key[-1]
        t_grid = np.linspace(10.0, t_max, DEFAULT_TABLE_SIZE)
        flux_grid = _direct_band_integral(
            t_grid,
            lam1,
            lam2,
            n_lam,
            clip_negative,
        )
        cached = (t_grid, flux_grid)
        _band_table_cache[cache_key] = cached

    t_grid, flux_grid = cached
    clipped_temperature = np.where(temperature_array < 10, 10.0, temperature_array)
    interpolated = np.interp(clipped_temperature, t_grid, flux_grid)
    return np.where(temperature_array < 10, 0.0, interpolated)
