'''
DKOLL: downloaded from: https://github.com/bmorris3/kelp/blob/main/kelp/core.py
Strip away anything except the reflected light curves
'''

from math import sin, cos

import numpy as np
from scipy.integrate import dblquad
from scipy.interpolate import RectBivariateSpline
from scipy.special import hermite
from functools import wraps
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
                x_grid = np.linspace(0, 2*np.pi, num_points)
                result = func(x_grid, *args, **kwargs)

                y_grid = result[0] if isinstance(result, tuple) else result
                integral = np.trapz(y_grid, x_grid)

                if integral == 0:
                    raise ValueError("Integral is zero.")

                cache[key] = integral

            result = func(x, *args, **kwargs)

            if isinstance(result, tuple):
                y = result[0] / cache[key]
                return (y, *result[1:])
            else:
                return result / cache[key]

        return wrapper
    return decorator


def trapz2d(z, x, y):
    """
    Integrates a regularly spaced 2D grid using the composite trapezium rule.

    Source: https://github.com/tiagopereira/python_tips/blob/master/code/trapz2d.py

    Parameters
    ----------
    z : `~numpy.ndarray`
        2D array
    x : `~numpy.ndarray`
        grid values for x (1D array)
    y : `~numpy.ndarray`
        grid values for y (1D array)

    Returns
    -------
    t : `~numpy.ndarray`
        Trapezoidal approximation to the integral under z
    """
    m = z.shape[0] - 1
    n = z.shape[1] - 1
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    s1 = z[0, 0, :] + z[m, 0, :] + z[0, n, :] + z[m, n, :]
    s2 = (np.sum(z[1:m, 0, :], axis=0) + np.sum(z[1:m, n, :], axis=0) +
          np.sum(z[0, 1:n, :], axis=0) + np.sum(z[m, 1:n, :], axis=0))
    s3 = np.sum(np.sum(z[1:m, 1:n, :], axis=0), axis=0)
    return dx * dy * (s1 + 2 * s2 + 4 * s3) / 4


def mu(theta):
    r"""
    Angle :math:`\mu = \cos(\theta)`

    Parameters
    ----------
    theta : `~numpy.ndarray`
        Angle :math:`\theta`
    """
    return np.cos(theta)


def tilda_mu(theta, alpha):
    r"""
    The normalized quantity
    :math:`\tilde{\mu} = \alpha \mu(\theta)`

    Parameters
    ----------
    theta : `~numpy.ndarray`
        Angle :math:`\theta`
    alpha : float
        Dimensionless fluid number :math:`\alpha`
    """
    return alpha * mu(theta)


def H(lmax, theta, alpha):
    r"""
    Hermite Polynomials in :math:`\tilde{\mu}(\theta)`.

    Parameters
    ----------
    lmax : int
        Maximum spherical harmonic degree.
    theta : float
        Angle :math:`\theta`
    alpha : float
        Dimensionless fluid number :math:`\alpha`

    Returns
    -------
    result : `~numpy.ndarray`
        Hermite Polynomial evaluated at angles :math:`\theta`.
    """
    return np.sum([a * tilda_mu(theta, alpha) ** l for l, a in
        zip(range(0, lmax + 1)[::-1], list(hermite(n=lmax)))], axis=0
    )


def _integral_phase_function(Psi, sin_abs_sort_alpha, sort_alpha, sort):
    """
    Integral phase function q for a generic, possibly asymmetric reflectivity
    map
    """
    return np.trapz(Psi[sort] * sin_abs_sort_alpha, sort_alpha)

@normalize_0_2pi()
def reflected_phase_curve(xi, omega, g, a_rp):
    """
    Reflected light phase curve for a homogeneous sphere by
    Heng, Morris & Kitzmann (2021).

    Parameters
    ----------
    xi : `~np.ndarray`
        Orbital phases of each observation defined on (-pi, pi)
    omega : tensor-like
        Single-scattering albedo as defined in
    g : tensor-like
        Scattering asymmetry factor, ranges from (-1, 1).
    a_rp : float, tensor-like
        Semimajor axis scaled by the planetary radius

    Returns
    -------
    flux_ratio_ppm : `~np.ndarray`
        Flux ratio between the reflected planetary flux and the stellar flux in
        units of ppm.
    A_g : float
        Geometric albedo derived for the planet given {omega, g}.
    q : float
        Integral phase function
    """
    phases = (xi + np.pi) / 2 / np.pi

    # Convert orbital phase on (0, 1) to "alpha" on (0, np.pi)
    alpha = (2 * np.pi * phases - np.pi)
    abs_alpha = np.abs(alpha)
    alpha_sort_order = np.argsort(alpha)
    sin_abs_sort_alpha = np.sin(abs_alpha[alpha_sort_order])
    sort_alpha = alpha[alpha_sort_order]

    gamma = np.sqrt(1 - omega)
    eps = (1 - gamma) / (1 + gamma)

    # Equation 34 for Henyey-Greestein
    P_star = (1 - g ** 2) / (1 + g ** 2 +
                             2 * g * np.cos(alpha)) ** 1.5
    # Equation 36
    P_0 = (1 - g) / (1 + g) ** 2

    # Equation 10:
    Rho_S = P_star - 1 + 0.25 * ((1 + eps) * (2 - eps)) ** 2
    Rho_S_0 = P_0 - 1 + 0.25 * ((1 + eps) * (2 - eps)) ** 2
    Rho_L = 0.5 * eps * (2 - eps) * (1 + eps) ** 2
    Rho_C = eps ** 2 * (1 + eps) ** 2

    alpha_plus = np.sin(abs_alpha / 2) + np.cos(abs_alpha / 2)
    alpha_minus = np.sin(abs_alpha / 2) - np.cos(abs_alpha / 2)

    valid_conditions = (
        (alpha_minus != -1) & (alpha_plus != 1) & (alpha_plus != -1) &
        (alpha_minus != 1)
    )
    num1 = np.where(
        valid_conditions,
        (1 + alpha_minus),
        1
    )
    num2 = np.where(
        valid_conditions,
        (alpha_plus - 1),
        1
    )
    denom1 = np.where(
        valid_conditions,
        (1 + alpha_plus),
        1
    )
    denom2 = np.where(
        valid_conditions,
        (1 - alpha_minus),
        1
    )

    # Equation 11:
    Psi_0 = np.where(
        valid_conditions,
        np.log(num1 * num2 / denom1 / denom2),
        0
    )

    Psi_S = 1 - 0.5 * (np.cos(abs_alpha / 2) -
                       1.0 / np.cos(abs_alpha / 2)) * Psi_0
    Psi_L = (np.sin(abs_alpha) + (np.pi - abs_alpha) *
             np.cos(abs_alpha)) / np.pi
    Psi_C = (-1 + 5 / 3 * np.cos(abs_alpha / 2) ** 2 - 0.5 *
             np.tan(abs_alpha / 2) * np.sin(abs_alpha / 2) ** 3 * Psi_0)

    # Fix the case when the phase angle is exactly 0 or pi
    Psi_S[abs_alpha == 0] = 1
    Psi_C[abs_alpha == 0] = 2 / 3
    Psi_S[abs_alpha == np.pi] = 0
    Psi_C[abs_alpha == np.pi] = 0

    # Equation 8:
    A_g = omega / 8 * (P_0 - 1) + eps / 2 + eps ** 2 / 6 + eps ** 3 / 24

    # Equation 9:
    Psi = ((12 * Rho_S * Psi_S + 16 * Rho_L *
            Psi_L + 9 * Rho_C * Psi_C) /
           (12 * Rho_S_0 + 16 * Rho_L + 6 * Rho_C))

    flux_ratio_ppm = 1e6 * (a_rp ** -2 * A_g * Psi)

    q = _integral_phase_function(
        Psi, sin_abs_sort_alpha, sort_alpha, alpha_sort_order
    )

    return flux_ratio_ppm, A_g, q




# DKOLL: reflected light phase curve, for the single scattering component only!
@normalize_0_2pi()
def reflected_phase_curve_SS(xi, omega, g, a_rp):
    """
    Reflected light phase curve for a homogeneous sphere by
    Heng, Morris & Kitzmann (2021).

    Parameters
    ----------
    xi : `~np.ndarray`
        Orbital phases of each observation defined on (-pi, pi)
    omega : tensor-like
        Single-scattering albedo as defined in
    g : tensor-like
        Scattering asymmetry factor, ranges from (-1, 1).
    a_rp : float, tensor-like
        Semimajor axis scaled by the planetary radius

    Returns
    -------
    flux_ratio_ppm : `~np.ndarray`
        Flux ratio between the reflected planetary flux and the stellar flux in
        units of ppm.
    A_g : float
        Geometric albedo derived for the planet given {omega, g}.
    q : float
        Integral phase function
    """
    phases = (xi + np.pi) / 2 / np.pi

    # Convert orbital phase on (0, 1) to "alpha" on (0, np.pi)
    alpha = (2 * np.pi * phases - np.pi)
    abs_alpha = np.abs(alpha)
    alpha_sort_order = np.argsort(alpha)
    sin_abs_sort_alpha = np.sin(abs_alpha[alpha_sort_order])
    sort_alpha = alpha[alpha_sort_order]

    gamma = np.sqrt(1 - omega)
    eps = (1 - gamma) / (1 + gamma)

    # Equation 34 for Henyey-Greestein
    P_star = (1 - g ** 2) / (1 + g ** 2 +
                             2 * g * np.cos(alpha)) ** 1.5
    # Equation 36
    P_0 = (1 - g) / (1 + g) ** 2

    # Equation 10:
    Rho_S = P_star - 1 + 0.25 * ((1 + eps) * (2 - eps)) ** 2
    Rho_S_0 = P_0 - 1 + 0.25 * ((1 + eps) * (2 - eps)) ** 2
    Rho_L = 0.5 * eps * (2 - eps) * (1 + eps) ** 2
    Rho_C = eps ** 2 * (1 + eps) ** 2

    alpha_plus = np.sin(abs_alpha / 2) + np.cos(abs_alpha / 2)
    alpha_minus = np.sin(abs_alpha / 2) - np.cos(abs_alpha / 2)

    valid_conditions = (
        (alpha_minus != -1) & (alpha_plus != 1) & (alpha_plus != -1) &
        (alpha_minus != 1)
    )
    num1 = np.where(
        valid_conditions,
        (1 + alpha_minus),
        1
    )
    num2 = np.where(
        valid_conditions,
        (alpha_plus - 1),
        1
    )
    denom1 = np.where(
        valid_conditions,
        (1 + alpha_plus),
        1
    )
    denom2 = np.where(
        valid_conditions,
        (1 - alpha_minus),
        1
    )

    # Equation 11:
    Psi_0 = np.where(
        valid_conditions,
        np.log(num1 * num2 / denom1 / denom2),
        0
    )

    Psi_S = 1 - 0.5 * (np.cos(abs_alpha / 2) -
                       1.0 / np.cos(abs_alpha / 2)) * Psi_0
    Psi_L = (np.sin(abs_alpha) + (np.pi - abs_alpha) *
             np.cos(abs_alpha)) / np.pi
    Psi_C = (-1 + 5 / 3 * np.cos(abs_alpha / 2) ** 2 - 0.5 *
             np.tan(abs_alpha / 2) * np.sin(abs_alpha / 2) ** 3 * Psi_0)

    # Fix the case when the phase angle is exactly 0 or pi
    Psi_S[abs_alpha == 0] = 1
    Psi_C[abs_alpha == 0] = 2 / 3
    Psi_S[abs_alpha == np.pi] = 0
    Psi_C[abs_alpha == np.pi] = 0

    # Equation 8:
    #A_g = omega / 8 * (P_0 - 1) + eps / 2 + eps ** 2 / 6 + eps ** 3 / 24
    A_g_ss = omega * P_0/ 8   # DKOLL: eqn 22

    # Equation 9: # DKOLL: simplified for SS component only
    #Psi = ((12 * Rho_S * Psi_S + 16 * Rho_L *
    #        Psi_L + 9 * Rho_C * Psi_C) /
    #       (12 * Rho_S_0 + 16 * Rho_L + 6 * Rho_C))
    Psi_ss = (Rho_S * Psi_S) / Rho_S_0

    flux_ratio_ppm = 1e6 * (a_rp ** -2 * A_g_ss * Psi_ss)

    q = _integral_phase_function(
        Psi_ss, sin_abs_sort_alpha, sort_alpha, alpha_sort_order
    )
    flux_ratio_ppm[np.abs(xi) < PPs.alpha] = 0
    return flux_ratio_ppm, A_g_ss, q
