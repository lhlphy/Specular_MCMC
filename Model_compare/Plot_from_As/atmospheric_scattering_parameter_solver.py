from pathlib import Path
import sys
import argparse

import numpy as np
from scipy.optimize import least_squares

PARENT_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

try:
    from .an_from_reflection_albedo import an_from_reflection_albedo
    from .atmospheric_scattering import atmospheric_scattering_phase_curve
    from .specular_reflection import specular_reflection_phase_curve
except ImportError:
    from an_from_reflection_albedo import an_from_reflection_albedo
    from atmospheric_scattering import atmospheric_scattering_phase_curve
    from specular_reflection import specular_reflection_phase_curve

try:
    from .parameters import PPs
except ImportError:
    from parameters import PPs


DEFAULT_THETA_GRID = np.linspace(0.0, np.pi, 4001)


def atmospheric_scattering_ag_as(
    omega,
    g,
    theta_grid=DEFAULT_THETA_GRID,
    a_over_rp=None,
):
    if a_over_rp is None:
        a_over_rp = PPs.semi_axis / PPs.Rp

    theta_grid = np.asarray(theta_grid, dtype=float)
    flux_ppm = atmospheric_scattering_phase_curve(
        theta_grid,
        omega=omega,
        g=g,
        normalize=False,
    )
    flux_ratio = flux_ppm * 1e-6
    ag = flux_ratio[-1] * a_over_rp**2
    a_s = 2.0 * a_over_rp**2 * np.trapz(flux_ratio * np.sin(theta_grid), theta_grid)
    q = a_s / ag if ag > 0.0 else np.nan
    return {
        "Ag": float(ag),
        "As": float(a_s),
        "q": float(q),
    }


def atmospheric_transit_flux_ppm(omega, g):
    return float(
        atmospheric_scattering_phase_curve(
            np.array([0.0]),
            omega=omega,
            g=g,
            normalize=False,
        )[0]
    )


def specular_transit_flux_ppm(a_normal):
    return float(
        specular_reflection_phase_curve(
            np.array([0.0]),
            a_normal,
            normalize=False,
        )[0]
    )


def infer_specular_an_for_reference(
    target_as,
    specular_an=None,
    specular_albedo_value=None,
    specular_albedo_type="As",
):
    if specular_an is not None:
        return float(specular_an)

    if specular_albedo_value is None:
        specular_albedo_value = target_as

    return float(
        an_from_reflection_albedo(
            specular_albedo_value,
            albedo_type=specular_albedo_type,
            reflection_model="specular",
        )
    )


def solve_atmospheric_omega_g(
    target_as,
    specular_an=None,
    specular_albedo_value=None,
    specular_albedo_type="As",
    initial_guess=(1.0, 0.999),
    theta_grid=DEFAULT_THETA_GRID,
    omega_bounds=(1e-8, 1.0),
    g_bounds=(-0.999, 0.999999),
    max_nfev=200,
):
    specular_an_ref = infer_specular_an_for_reference(
        target_as,
        specular_an=specular_an,
        specular_albedo_value=specular_albedo_value,
        specular_albedo_type=specular_albedo_type,
    )
    target_transit_ppm = specular_transit_flux_ppm(specular_an_ref)
    as_scale = max(abs(float(target_as)), 1.0)
    transit_scale = max(abs(target_transit_ppm), 1.0)

    lower_bounds = np.array([omega_bounds[0], g_bounds[0]], dtype=float)
    upper_bounds = np.array([omega_bounds[1], g_bounds[1]], dtype=float)
    x0 = np.clip(np.asarray(initial_guess, dtype=float), lower_bounds, upper_bounds)

    def residuals(params):
        omega, g = params
        albedo = atmospheric_scattering_ag_as(
            omega,
            g,
            theta_grid=theta_grid,
        )
        transit_ppm = atmospheric_transit_flux_ppm(omega, g)
        return np.array(
            [
                (albedo["As"] - target_as) / as_scale,
                (transit_ppm - target_transit_ppm) / transit_scale,
            ],
            dtype=float,
        )

    result = least_squares(
        residuals,
        x0,
        bounds=(lower_bounds, upper_bounds),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=max_nfev,
    )

    omega, g = result.x
    albedo = atmospheric_scattering_ag_as(omega, g, theta_grid=theta_grid)
    transit_ppm = atmospheric_transit_flux_ppm(omega, g)

    return {
        "omega": float(omega),
        "g": float(g),
        "target_As": float(target_as),
        "atmospheric_As": albedo["As"],
        "atmospheric_Ag": albedo["Ag"],
        "atmospheric_q": albedo["q"],
        "target_transit_ppm": target_transit_ppm,
        "atmospheric_transit_ppm": transit_ppm,
        "specular_reference_An": specular_an_ref,
        "success": bool(result.success),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "nfev": int(result.nfev),
        "message": result.message,
    }


def _build_argument_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Solve atmospheric scattering omega and g for a target As and a "
            "transit flux matched to specular reflection."
        )
    )
    parser.add_argument("--target-as", type=float, required=True)
    parser.add_argument("--specular-an", type=float, default=None)
    parser.add_argument("--specular-albedo-value", type=float, default=None)
    parser.add_argument(
        "--specular-albedo-type",
        choices=["Ag", "As", "ag", "as"],
        default="As",
    )
    parser.add_argument("--omega0", type=float, default=1.0)
    parser.add_argument("--g0", type=float, default=0.999)
    return parser


def main():
    parser = _build_argument_parser()
    args = parser.parse_args()
    result = solve_atmospheric_omega_g(
        args.target_as,
        specular_an=args.specular_an,
        specular_albedo_value=args.specular_albedo_value,
        specular_albedo_type=args.specular_albedo_type,
        initial_guess=(args.omega0, args.g0),
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
