from pathlib import Path
import sys
import argparse
import csv

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PARENT_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

try:
    from .an_from_reflection_albedo import an_from_reflection_albedo
    from .atmospheric_scattering import atmospheric_scattering_phase_curve
    from .atmospheric_scattering_parameter_solver import solve_atmospheric_omega_g
    from .lambert_diffuse import lambert_diffuse_phase_curve
    from .specular_reflection import specular_reflection_phase_curve
except ImportError:
    from an_from_reflection_albedo import an_from_reflection_albedo
    from atmospheric_scattering import atmospheric_scattering_phase_curve
    from atmospheric_scattering_parameter_solver import solve_atmospheric_omega_g
    from lambert_diffuse import lambert_diffuse_phase_curve
    from specular_reflection import specular_reflection_phase_curve


DEFAULT_AS_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "figs"


def format_as_for_filename(a_s):
    return f"{a_s:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def compute_reflection_curves_for_as(a_s, theta_grid, solver_theta_points=4001):
    diffuse_an = a_s
    specular_an = an_from_reflection_albedo(
        a_s,
        albedo_type="As",
        reflection_model="specular",
    )
    solver_theta_grid = np.linspace(0.0, np.pi, solver_theta_points)
    atmospheric_solution = solve_atmospheric_omega_g(
        a_s,
        theta_grid=solver_theta_grid,
    )

    diffuse_ppm = lambert_diffuse_phase_curve(
        theta_grid,
        diffuse_an,
        normalize=False,
    )
    specular_ppm = specular_reflection_phase_curve(
        theta_grid,
        specular_an,
        normalize=False,
    )
    atmospheric_ppm = atmospheric_scattering_phase_curve(
        theta_grid,
        omega=atmospheric_solution["omega"],
        g=atmospheric_solution["g"],
        normalize=False,
    )

    return {
        "As": a_s,
        "diffuse_an": diffuse_an,
        "specular_an": specular_an,
        "omega": atmospheric_solution["omega"],
        "g": atmospheric_solution["g"],
        "atmospheric_As": atmospheric_solution["atmospheric_As"],
        "atmospheric_Ag": atmospheric_solution["atmospheric_Ag"],
        "target_transit_ppm": atmospheric_solution["target_transit_ppm"],
        "atmospheric_transit_ppm": atmospheric_solution["atmospheric_transit_ppm"],
        "diffuse_ppm": diffuse_ppm,
        "specular_ppm": specular_ppm,
        "atmospheric_ppm": atmospheric_ppm,
    }


def plot_reflection_curves(result, theta_grid, output_dir=DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase = theta_grid / (2.0 * np.pi)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(
        phase,
        result["diffuse_ppm"],
        color="black",
        linewidth=2.2,
        label=f"Lambert diffuse (An={result['diffuse_an']:.3f})",
    )
    ax.plot(
        phase,
        result["specular_ppm"],
        color="#d55e00",
        linewidth=2.2,
        label=f"Specular reflection (An={result['specular_an']:.4f})",
    )
    ax.plot(
        phase,
        result["atmospheric_ppm"],
        color="#0072b2",
        linewidth=2.2,
        label=f"Atmospheric scattering (omega={result['omega']:.4f}, g={result['g']:.4f})",
    )

    ax.set_title(f"Reflection phase curves, As={result['As']:.1f}", fontsize=15)
    ax.set_xlabel("Orbital phase", fontsize=13)
    ax.set_ylabel("Intensity (ppm)", fontsize=13)
    ax.set_xlim(0.0, 1.0)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(loc="best", fontsize=10, frameon=False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    fig.tight_layout()

    output_path = output_dir / f"reflection_models_As_{format_as_for_filename(result['As'])}.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def save_summary_csv(results, output_dir=DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "reflection_models_by_as_summary.csv"
    fieldnames = [
        "As",
        "diffuse_an",
        "specular_an",
        "omega",
        "g",
        "atmospheric_As",
        "atmospheric_Ag",
        "target_transit_ppm",
        "atmospheric_transit_ppm",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in fieldnames})
    return output_path


def plot_reflection_models_by_as(
    as_values=DEFAULT_AS_VALUES,
    output_dir=DEFAULT_OUTPUT_DIR,
    curve_points=2000,
    solver_theta_points=4001,
):
    theta_grid = np.linspace(0.0, 2.0 * np.pi, curve_points)
    results = []
    figure_paths = []
    for a_s in as_values:
        result = compute_reflection_curves_for_as(
            float(a_s),
            theta_grid,
            solver_theta_points=solver_theta_points,
        )
        results.append(result)
        figure_paths.append(plot_reflection_curves(result, theta_grid, output_dir=output_dir))

    summary_path = save_summary_csv(results, output_dir=output_dir)
    return figure_paths, summary_path


def _build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Plot diffuse, specular, and atmospheric reflection curves for selected As values."
    )
    parser.add_argument(
        "--as-values",
        nargs="+",
        type=float,
        default=DEFAULT_AS_VALUES,
        help="Spherical albedo values to plot.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for output figures.",
    )
    parser.add_argument("--curve-points", type=int, default=2000)
    parser.add_argument("--solver-theta-points", type=int, default=4001)
    return parser


def main():
    parser = _build_argument_parser()
    args = parser.parse_args()
    figure_paths, summary_path = plot_reflection_models_by_as(
        as_values=args.as_values,
        output_dir=args.output_dir,
        curve_points=args.curve_points,
        solver_theta_points=args.solver_theta_points,
    )
    for path in figure_paths:
        print(path)
    print(summary_path)


if __name__ == "__main__":
    main()
