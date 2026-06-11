import os

import numpy as np


TABLE_NAME = "fresnel_lookup_table.npz"
A_SIZE = int(os.environ.get("MCMC_FRESNEL_A_SIZE", os.environ.get("MCMC_FRESNEL_TABLE_SIZE", "1025")))
MU_SIZE = int(os.environ.get("MCMC_FRESNEL_MU_SIZE", "512"))
A_MAX = float(os.environ.get("MCMC_FRESNEL_A_MAX", "0.999999"))
S_MAX = np.sqrt(A_MAX)
MU_MIN = 0.0
MU_MAX = 1.0
USE_TABLE = os.environ.get("MCMC_FRESNEL_USE_TABLE", "1") != "0"

_table_cache = None


def _table_path():
    return os.path.join(os.path.dirname(__file__), TABLE_NAME)


def _direct_fresnel_from_mu(a_normal, mu):
    a_normal = np.asarray(a_normal, dtype=float)
    mu = np.asarray(mu, dtype=float)
    a_normal = np.clip(a_normal, 0.0, A_MAX)
    mu = np.clip(mu, MU_MIN, MU_MAX)

    cosi = mu
    sini = np.sqrt(np.maximum(0.0, 1.0 - cosi**2))
    n_refr = 2 / (1 - np.sqrt(a_normal)) - 1
    co1 = np.sqrt(n_refr**2 - sini**2)

    den_s = cosi + co1
    den_p = co1 + n_refr**2 * cosi
    rs = np.divide((cosi - co1) ** 2, den_s**2, out=np.ones_like(co1), where=den_s != 0)
    rp = np.divide((co1 - n_refr**2 * cosi) ** 2, den_p**2, out=np.ones_like(co1), where=den_p != 0)
    return np.where(cosi == 0, 1.0, (rs + rp) / 2)


def _direct_fresnel(a_normal, i_angle):
    return _direct_fresnel_from_mu(a_normal, np.cos(i_angle))


def build_table(path=None, a_size=A_SIZE, mu_size=MU_SIZE):
    path = path or _table_path()
    s_grid = np.linspace(0.0, S_MAX, a_size, dtype=np.float64)
    a_grid = s_grid**2
    mu_grid = np.linspace(MU_MIN, MU_MAX, mu_size, dtype=np.float64)
    values = np.empty((a_size, mu_size), dtype=np.float64)

    chunk = int(os.environ.get("MCMC_FRESNEL_BUILD_CHUNK", "64"))
    mu_row = mu_grid[np.newaxis, :]
    for start in range(0, a_size, chunk):
        stop = min(start + chunk, a_size)
        values[start:stop] = _direct_fresnel_from_mu(a_grid[start:stop, np.newaxis], mu_row)

    tmp_path = f"{path}.tmp.{os.getpid()}.npz"
    np.savez(tmp_path, s_grid=s_grid, mu_grid=mu_grid, values=values)
    if os.path.exists(path):
        os.remove(path)
    os.replace(tmp_path, path)
    return s_grid, mu_grid, values


def load_table():
    global _table_cache
    if _table_cache is not None:
        return _table_cache

    path = _table_path()
    if not os.path.exists(path):
        _table_cache = build_table(path)
        return _table_cache

    try:
        with np.load(path) as data:
            s_grid = data["s_grid"]
            mu_grid = data["mu_grid"]
            values = data["values"]
            if values.shape != (A_SIZE, MU_SIZE):
                raise ValueError("Fresnel lookup table shape does not match configured grid")
            _table_cache = (s_grid, mu_grid, values)
    except Exception:
        _table_cache = build_table(path)
    return _table_cache


def lookup_fresnel(a_normal, i_angle):
    if not USE_TABLE:
        return _direct_fresnel(a_normal, i_angle)

    s_grid, mu_grid, values = load_table()
    a_normal = np.asarray(a_normal, dtype=float)
    i_angle = np.asarray(i_angle, dtype=float)
    s_clipped = np.sqrt(np.clip(a_normal, 0.0, A_MAX))
    mu_clipped = np.clip(np.cos(i_angle), mu_grid[0], mu_grid[-1])

    a_pos = (s_clipped - s_grid[0]) / (s_grid[-1] - s_grid[0]) * (len(s_grid) - 1)
    mu_pos = (mu_clipped - mu_grid[0]) / (mu_grid[-1] - mu_grid[0]) * (len(mu_grid) - 1)

    a0 = np.floor(a_pos).astype(np.int64)
    m0 = np.floor(mu_pos).astype(np.int64)
    a0 = np.clip(a0, 0, len(s_grid) - 2)
    m0 = np.clip(m0, 0, len(mu_grid) - 2)
    a1 = a0 + 1
    m1 = m0 + 1

    wa = a_pos - a0
    wm = mu_pos - m0

    if np.ndim(a_normal) == 0:
        row0 = values[int(a0), :]
        row1 = values[int(a1), :]
        row = (1 - float(wa)) * row0 + float(wa) * row1
        flat_mu = np.ravel(mu_clipped)
        flat_pos = np.ravel(mu_pos)
        flat_m0 = np.clip(np.floor(flat_pos).astype(np.int64), 0, len(mu_grid) - 2)
        flat_wm = flat_pos - flat_m0
        out = (1 - flat_wm) * row[flat_m0] + flat_wm * row[flat_m0 + 1]
        return out.reshape(np.shape(mu_clipped))

    v00 = values[a0, m0]
    v10 = values[a1, m0]
    v01 = values[a0, m1]
    v11 = values[a1, m1]
    return (
        (1 - wa) * (1 - wm) * v00
        + wa * (1 - wm) * v10
        + (1 - wa) * wm * v01
        + wa * wm * v11
    )


def direct_fresnel(a_normal, i_angle):
    return _direct_fresnel(a_normal, i_angle)
