from pathlib import Path
import sys

import numpy as np

PARENT_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

try:
    from .parameters import PPs
except ImportError:
    from parameters import PPs

try:
    from .lambert_diffuse import lambert_diffuse_phase_curve
except ImportError:
    from lambert_diffuse import lambert_diffuse_phase_curve


def _to_dimensionless_flux(flux, flux_in_ppm):
    flux = np.asarray(flux, dtype=float)
    if flux_in_ppm:
        return flux * 1e-6
    return flux


def _prepare_arrays(theta, flux):
    theta = np.asarray(theta, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if theta.shape != flux.shape:
        raise ValueError("theta and flux must have the same shape.")
    order = np.argsort(theta)
    return theta[order], flux[order]


def _interp_full_phase_flux(theta, flux):
    tolerance = 1e-12
    if theta[0] > np.pi + tolerance or theta[-1] < np.pi - tolerance:
        raise ValueError("theta array must cover theta = pi to infer A_g.")
    return np.interp(np.pi, theta, flux)


def _full_orbit_phase_average(alpha, theta, flux):
    left_flux = np.interp(np.pi - alpha, theta, flux)
    right_flux = np.interp(np.pi + alpha, theta, flux)
    return 0.5 * (left_flux + right_flux)


def albedo_from_reflection_phase_curve(
    theta,
    reflection_phase_curve,
    a_over_rp=None,
    flux_in_ppm=True,
    full_orbit=None,
):
    """
    Compute geometric albedo A_g and spherical albedo A_s from a reflection
    phase curve following reflection_phase_curve_albedo.md.

    Parameters
    ----------
    theta : array-like
        Orbital angle in radians. This code assumes theta = 0 at transit and
        theta = pi at full phase / secondary eclipse.
    reflection_phase_curve : array-like
        Reflected-light phase curve, either in ppm or in dimensionless Fp/F*.
    a_over_rp : float, optional
        Orbital distance divided by planetary radius. Defaults to PPs.semi_axis
        / PPs.Rp.
    flux_in_ppm : bool, optional
        If True, convert the input curve from ppm to dimensionless flux ratio.
    full_orbit : bool, optional
        If True, treat the input as a 0..2pi orbit and use the averaged
        full-orbit formula. If False, treat the input as a 0..pi half orbit.
        If None, infer from the theta coverage.
    """
    if a_over_rp is None:
        a_over_rp = PPs.semi_axis / PPs.Rp

    theta, flux = _prepare_arrays(theta, _to_dimensionless_flux(reflection_phase_curve, flux_in_ppm))
    theta_span = theta[-1] - theta[0]

    if full_orbit is None:
        full_orbit = theta_span > 1.5 * np.pi

    if full_orbit:
        tolerance = 1e-12
        if theta[0] > tolerance or theta[-1] < 2.0 * np.pi - tolerance:
            raise ValueError("Full-orbit mode requires theta coverage from 0 to 2*pi.")

        alpha = np.linspace(0.0, np.pi, theta.size)
        averaged_flux = _full_orbit_phase_average(alpha, theta, flux)
        full_phase_flux = averaged_flux[0]
        phase_function = averaged_flux / full_phase_flux
        q = 2.0 * np.trapz(phase_function * np.sin(alpha), alpha)
        ag = full_phase_flux * a_over_rp**2
        a_s_integral = 2.0 * a_over_rp**2 * np.trapz(averaged_flux * np.sin(alpha), alpha)
        a_s_from_q = q * ag
        return {
            "Ag": ag,
            "As": a_s_integral,
            "As_from_qAg": a_s_from_q,
            "q": q,
            "full_phase_flux_ratio": full_phase_flux,
            "alpha_grid": alpha,
            "averaged_flux_ratio": averaged_flux,
            "phase_function": phase_function,
        }

    tolerance = 1e-12
    if theta[0] > tolerance or theta[-1] < np.pi - tolerance:
        raise ValueError("Half-orbit mode requires theta coverage from 0 to pi.")

    full_phase_flux = _interp_full_phase_flux(theta, flux)
    phase_function = flux / full_phase_flux
    q = 2.0 * np.trapz(phase_function * np.sin(theta), theta)
    ag = full_phase_flux * a_over_rp**2
    a_s_integral = 2.0 * a_over_rp**2 * np.trapz(flux * np.sin(theta), theta)
    a_s_from_q = q * ag
    return {
        "Ag": ag,
        "As": a_s_integral,
        "As_from_qAg": a_s_from_q,
        "q": q,
        "full_phase_flux_ratio": full_phase_flux,
        "theta_grid": theta,
        "phase_function": phase_function,
    }


def validate_lambert_relation(bond_albedo=0.1, num_points=4001):
    theta = np.linspace(0.0, 2.0 * np.pi, num_points)
    lambert_curve_ppm = lambert_diffuse_phase_curve(
        theta,
        bond_albedo=bond_albedo,
        normalize=False,
    )
    result = albedo_from_reflection_phase_curve(
        theta,
        lambert_curve_ppm,
        flux_in_ppm=True,
        full_orbit=True,
    )

    expected_ag_exact_lambert = 2.0 * bond_albedo / 3.0
    expected_as_exact_lambert = bond_albedo

    return {
        "bond_albedo_input": bond_albedo,
        "Ag_measured": result["Ag"],
        "Ag_expected_exact_lambert": expected_ag_exact_lambert,
        "As_measured": result["As"],
        "As_expected_exact_lambert": expected_as_exact_lambert,
        "q_measured": result["q"],
        "q_expected": 1.5,
        "As_over_Ag_measured": result["As"] / result["Ag"],
        "As_over_Ag_expected": 1.5,
        "Ag_abs_error_vs_exact_lambert": result["Ag"] - expected_ag_exact_lambert,
        "As_abs_error_vs_exact_lambert": result["As"] - expected_as_exact_lambert,
    }


def main():
    validation = validate_lambert_relation(bond_albedo=0.1)
    for key, value in validation.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
