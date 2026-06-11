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


@normalize_0_2pi()
def lambert_diffuse_phase_curve(
    theta_array,
    bond_albedo,
    rp2rs=PPs.Rp2Rs,
    alpha=PPs.alpha,
):
    theta_array = np.asarray(theta_array, dtype=float)
    zeta = np.arccos(-np.cos(theta_array))
    phase_function = (
        bond_albedo
        * 2.0
        / 3.0
        * (np.sin(zeta) + (np.pi - zeta) * np.cos(zeta))
        / np.pi
    )
    return rp2rs**2 * np.sin(alpha) ** 2 * phase_function * 1e6


F_lambert = lambert_diffuse_phase_curve


def main():
    import matplotlib.pyplot as plt

    theta = np.linspace(0.0, 2.0 * np.pi, 2000)
    flux = lambert_diffuse_phase_curve(theta, bond_albedo=0.1, normalize=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(theta / (2.0 * np.pi), flux, linewidth=2, color="black")
    ax.set_xlabel("Orbital phase")
    ax.set_ylabel("Intensity (ppm)")
    ax.set_title("Lambert diffuse phase curve")
    ax.set_xlim(0.0, 1.0)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
