import numpy as np

from hengmorris_analytic_phasecurves import reflected_phase_curve
from parameters import PPs


# DEFAULT_RP = 0.01409075
# DEFAULT_A = 1.53821517
DEFAULT_RP = PPs.Rp / 696340
DEFAULT_A = PPs.semi_axis / 696340
DEFAULT_H = 100/696340  # 100 km
DEFAULT_OMEGA = 0.8
DEFAULT_G = 0.77


def hg_point_star_phase_function(alpha, g=DEFAULT_G):
    """Henyey-Greenstein point-star phase function used by the notebook."""
    return (
        (1.0 - g**2)
        / (4.0 * np.pi * (1.0 + g**2 - 2.0 * g * np.cos(alpha)) ** 1.5)
    )


def garcia_munoz_limb_flux(
    alpha,
    h=DEFAULT_H,
    rp=DEFAULT_RP,
    semi_major_axis=DEFAULT_A,
    omega=DEFAULT_OMEGA,
    g=DEFAULT_G,
):
    """Garcia-Munoz limb forward-scattering contribution in ppm.

    This matches the Mathematica notebook's FpOverFstarGMHG:
    (2*pi*H*Rp/a^2) * omega * GetPaPointStar[alpha, g] * 1e6.
    The radius, scale height, and orbital distance must use the same units.
    """
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


def mathematica_atmospheric_forward_scattering(
    x,
    h=DEFAULT_H,
    rp=DEFAULT_RP,
    semi_major_axis=DEFAULT_A,
    omega=DEFAULT_OMEGA,
    g=DEFAULT_G,
):
    """Atmospheric curve matching the Mathematica notebook.

    The notebook plots:
    FpOverFstarHengHG[pi - x, Rp, a, omega, g]
    + FpOverFstarGMHG[x, H, Rp, a, omega, g].
    """
    heng_hg, _, _ = reflected_phase_curve(
        np.pi - x,
        omega,
        g,
        semi_major_axis / rp,
        normalize=False,
    )
    limb = garcia_munoz_limb_flux(
        x,
        h=h,
        rp=rp,
        semi_major_axis=semi_major_axis,
        omega=omega,
        g=g,
    )
    return heng_hg + limb
