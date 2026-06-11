from pathlib import Path
import sys

import numpy as np
from functools import wraps

PARENT_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

try:
    from .parameters import PPs
except ImportError:
    from parameters import PPs


def normalize_0_2pi(num_points=1000):
    def decorator(func):
        cache = {}

        @wraps(func)
        def wrapper(x, *args, normalize=True, **kwargs):
            if not normalize:
                return func(x, *args, **kwargs)

            key = (args, tuple(sorted(kwargs.items())))
            if key not in cache:
                x_grid = np.linspace(0.0, 2.0 * np.pi, num_points)
                y_grid = func(x_grid, *args, **kwargs)
                integral = np.trapz(y_grid, x_grid)
                if integral == 0:
                    raise ValueError("Integral is zero, cannot normalize.")
                cache[key] = integral

            return func(x, *args, **kwargs) / cache[key]

        return wrapper

    return decorator


def fresnel_albedo(theta=0.0, a_normal=0.0, incidence_angle=-1.0, inc=90.0):
    theta = np.asarray(theta, dtype=float)
    theta_obs = np.arccos(np.cos(theta) * np.sin(np.deg2rad(inc)))
    default_incidence = np.abs((np.pi - theta_obs) / 2.0)
    angle = np.where(incidence_angle == -1.0, default_incidence, incidence_angle)
    angle = np.clip(angle, 0.0, np.pi / 2.0)

    sin_i = np.sin(angle)
    cos_i = np.cos(angle)
    n = 2.0 / (1.0 - np.sqrt(a_normal)) - 1.0
    cos_t = np.sqrt(n**2 - sin_i**2)

    rs = ((cos_i - cos_t) / (cos_i + cos_t)) ** 2
    rp = ((cos_t - n**2 * cos_i) / (cos_t + n**2 * cos_i)) ** 2
    return (rs + rp) / 2.0


@normalize_0_2pi()
def specular_reflection_phase_curve(
    theta_array,
    a_normal,
    rp2rs=PPs.Rp2Rs,
    inc=90.0,
    alpha=PPs.alpha,
):
    theta_array = np.asarray(theta_array, dtype=float)
    flux = rp2rs**2 * np.sin(alpha / 2.0) ** 2

    theta_obs = np.arccos(np.cos(theta_array) * np.sin(np.deg2rad(inc)))
    tx = np.abs(np.pi - np.abs(theta_obs)) / 2.0
    fresnel_tx = fresnel_albedo(incidence_angle=tx, a_normal=a_normal, inc=inc)
    fresnel_edge = fresnel_albedo(
        incidence_angle=tx - alpha / 3.0,
        a_normal=a_normal,
        inc=inc,
    )

    edge_mask = tx > (np.pi / 2.0 - alpha / 2.0)
    blended = fresnel_tx * (np.pi - 2.0 * tx) / alpha
    blended += fresnel_edge * (2.0 * tx - np.pi + alpha) / alpha

    flux = flux * np.where(edge_mask, blended, fresnel_tx)
    return flux * 1e6


F_specular = specular_reflection_phase_curve


def main():
    import matplotlib.pyplot as plt

    theta = np.linspace(0.0, 2.0 * np.pi, 2000)
    flux = specular_reflection_phase_curve(theta, a_normal=0.1, normalize=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(theta / (2.0 * np.pi), flux, linewidth=2)
    ax.set_xlabel("Orbital phase")
    ax.set_ylabel("Intensity (ppm)")
    ax.set_title("Specular reflection phase curve")
    ax.set_xlim(0.0, 1.0)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
