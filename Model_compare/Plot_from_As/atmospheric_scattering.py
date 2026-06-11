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


def _integral_phase_function(psi, sin_abs_sort_alpha, sort_alpha, sort):
    return np.trapz(psi[sort] * sin_abs_sort_alpha, sort_alpha)


def heng_reflected_phase_curve(xi, omega, g, a_rp):
    xi = np.asarray(xi, dtype=float)
    phases = (xi + np.pi) / (2.0 * np.pi)
    alpha = 2.0 * np.pi * phases - np.pi
    abs_alpha = np.abs(alpha)
    alpha_sort_order = np.argsort(alpha)
    sin_abs_sort_alpha = np.sin(abs_alpha[alpha_sort_order])
    sort_alpha = alpha[alpha_sort_order]

    gamma = np.sqrt(1.0 - omega)
    eps = (1.0 - gamma) / (1.0 + gamma)

    p_star = (1.0 - g**2) / (1.0 + g**2 + 2.0 * g * np.cos(alpha)) ** 1.5
    p_0 = (1.0 - g) / (1.0 + g) ** 2

    rho_s = p_star - 1.0 + 0.25 * ((1.0 + eps) * (2.0 - eps)) ** 2
    rho_s_0 = p_0 - 1.0 + 0.25 * ((1.0 + eps) * (2.0 - eps)) ** 2
    rho_l = 0.5 * eps * (2.0 - eps) * (1.0 + eps) ** 2
    rho_c = eps**2 * (1.0 + eps) ** 2

    alpha_plus = np.sin(abs_alpha / 2.0) + np.cos(abs_alpha / 2.0)
    alpha_minus = np.sin(abs_alpha / 2.0) - np.cos(abs_alpha / 2.0)

    valid = (
        (alpha_minus != -1.0)
        & (alpha_plus != 1.0)
        & (alpha_plus != -1.0)
        & (alpha_minus != 1.0)
    )
    num1 = np.where(valid, 1.0 + alpha_minus, 1.0)
    num2 = np.where(valid, alpha_plus - 1.0, 1.0)
    den1 = np.where(valid, 1.0 + alpha_plus, 1.0)
    den2 = np.where(valid, 1.0 - alpha_minus, 1.0)

    psi_0 = np.where(valid, np.log(num1 * num2 / den1 / den2), 0.0)
    psi_s = 1.0 - 0.5 * (np.cos(abs_alpha / 2.0) - 1.0 / np.cos(abs_alpha / 2.0)) * psi_0
    psi_l = (np.sin(abs_alpha) + (np.pi - abs_alpha) * np.cos(abs_alpha)) / np.pi
    psi_c = (
        -1.0
        + 5.0 / 3.0 * np.cos(abs_alpha / 2.0) ** 2
        - 0.5 * np.tan(abs_alpha / 2.0) * np.sin(abs_alpha / 2.0) ** 3 * psi_0
    )

    psi_s = np.where(abs_alpha == 0.0, 1.0, psi_s)
    psi_c = np.where(abs_alpha == 0.0, 2.0 / 3.0, psi_c)
    psi_s = np.where(abs_alpha == np.pi, 0.0, psi_s)
    psi_c = np.where(abs_alpha == np.pi, 0.0, psi_c)

    a_g = omega / 8.0 * (p_0 - 1.0) + eps / 2.0 + eps**2 / 6.0 + eps**3 / 24.0
    psi = (
        12.0 * rho_s * psi_s + 16.0 * rho_l * psi_l + 9.0 * rho_c * psi_c
    ) / (12.0 * rho_s_0 + 16.0 * rho_l + 6.0 * rho_c)
    flux_ratio_ppm = 1e6 * (a_rp**-2 * a_g * psi)
    q = _integral_phase_function(psi, sin_abs_sort_alpha, sort_alpha, alpha_sort_order)
    return flux_ratio_ppm, a_g, q


def hg_point_star_phase_function(alpha, g):
    return (1.0 - g**2) / (
        4.0 * np.pi * (1.0 + g**2 - 2.0 * g * np.cos(alpha)) ** 1.5
    )


def garcia_munoz_limb_flux(alpha, h, rp, semi_major_axis, omega, g):
    return (
        2.0
        * np.pi
        * h
        * rp
        / semi_major_axis**2
        * omega
        * hg_point_star_phase_function(alpha, g)
        * 1e6
    )


def single_pass_atmospheric_scattering(
    orbital_angle,
    h=100.0 / 696340.0,
    rp=PPs.Rp / 696340.0,
    semi_major_axis=PPs.semi_axis / 696340.0,
    omega=0.8,
    g=0.77,
):
    orbital_angle = np.asarray(orbital_angle, dtype=float)
    heng_hg, _, _ = heng_reflected_phase_curve(
        np.pi - orbital_angle,
        omega,
        g,
        semi_major_axis / rp,
    )
    limb = garcia_munoz_limb_flux(
        orbital_angle,
        h=h,
        rp=rp,
        semi_major_axis=semi_major_axis,
        omega=omega,
        g=g,
    )
    return heng_hg + limb


@normalize_0_2pi()
def atmospheric_scattering_phase_curve(
    theta_array,
    h=100.0 / 696340.0,
    rp=PPs.Rp / 696340.0,
    semi_major_axis=PPs.semi_axis / 696340.0,
    omega=0.8,
    g=0.77,
):
    theta_array = np.asarray(theta_array, dtype=float)
    orbital_angle = np.where(theta_array <= np.pi, theta_array, 2.0 * np.pi - theta_array)
    return single_pass_atmospheric_scattering(
        orbital_angle,
        h=h,
        rp=rp,
        semi_major_axis=semi_major_axis,
        omega=omega,
        g=g,
    )


def main():
    import matplotlib.pyplot as plt

    theta = np.linspace(0.0, 2.0 * np.pi, 2000)
    flux = atmospheric_scattering_phase_curve(theta, normalize=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(theta / (2.0 * np.pi), flux, linewidth=2)
    ax.set_xlabel("Orbital phase")
    ax.set_ylabel("Intensity (ppm)")
    ax.set_title("Atmospheric scattering phase curve")
    ax.set_xlim(0.0, 1.0)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
