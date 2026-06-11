import os

import matplotlib.pyplot as plt
import numpy as np

if not hasattr(np, "acos"):
    np.acos = np.arccos

import analytical_model as analytic
from analytical_model import F_specular, F_thermal
from garcia_munoz_limb import mathematica_atmospheric_forward_scattering
from analytical_model_Lambert import F_lambert
from parameters import PPs


def build_atmospheric_forward_scattering(normalize=True):
    x = np.linspace(0, np.pi, num=1000)
    atmosphere = mathematica_atmospheric_forward_scattering(x)
    atmosphere = np.array([atmosphere, atmosphere[::-1]]).flatten()
    if not normalize:
        return atmosphere
    theta = np.linspace(0, 2 * np.pi, atmosphere.size)
    integral = np.trapz(atmosphere, theta)
    if integral == 0:
        raise ValueError("Integral is zero, cannot normalize.")
    return atmosphere / integral


def style_axis(ax, title, ylabel, ylim=None):
    ax.set_title(title, fontsize=17)
    ax.set_xlabel("Orbital Phase", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlim(0, 1)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="both", labelsize=14)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def main():
    theta_array = np.linspace(0, 2 * np.pi, 2000)
    phase = theta_array / (2 * np.pi)
    ab = 0.1
    i0 = (PPs.Rp / PPs.semi_axis) ** 2 * 1e6
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    if "FOLDER_PATH" not in os.environ:
        analytic.Response = lambda lam: np.ones_like(lam, dtype=float)

    specular_normalized = F_specular(theta_array, ab, PPs.Rp2Rs)
    diffuse_normalized = F_lambert(theta_array, ab, PPs.Rp2Rs)
    specular_absolute = F_specular(theta_array, ab, PPs.Rp2Rs, normalize=False)
    diffuse_absolute = F_lambert(theta_array, ab, PPs.Rp2Rs, normalize=False)
    specular = specular_normalized * i0
    diffuse = diffuse_normalized * i0
    atmosphere = build_atmospheric_forward_scattering(normalize=True) * i0
    atmosphere_absolute = build_atmospheric_forward_scattering(normalize=False)
    thermal_thick = F_thermal(theta_array, ab, F=0.5, Rp2Rs=PPs.Rp2Rs)
    thermal_bare = F_thermal(theta_array, ab, F=0, Rp2Rs=PPs.Rp2Rs)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    axes[0].plot(
        phase,
        specular,
        color=colors[1],
        label="Specular (lava ocean)",
        linewidth=2,
    )
    axes[0].plot(
        phase,
        diffuse,
        color="black",
        label="Diffuse (rough surface)",
        linewidth=2,
    )
    axes[0].plot(
        phase,
        atmosphere,
        color=colors[0],
        label="Atmospheric forward scattering",
        linewidth=2,
    )
    axes[0].legend(loc="upper center", fontsize=12, frameon=False)
    style_axis(axes[0], "Reflection", "Normalized intensity")

    axes[1].plot(
        phase,
        thermal_thick,
        "k-",
        label="Thick atmosphere",
        linewidth=2,
    )
    axes[1].plot(
        phase,
        thermal_bare,
        "k--",
        label="Bare surface",
        linewidth=2,
    )
    axes[1].legend(loc="upper center", fontsize=12, frameon=False)
    style_axis(axes[1], "Thermal emission", "Intensity (ppm)", ylim=(0, 22))

    axes[2].plot(
        phase,
        specular_absolute + thermal_bare,
        "--",
        linewidth=2,
        color=colors[1],
        label="Specular + bare surface",
    )
    axes[2].plot(
        phase,
        diffuse_absolute + thermal_thick,
        linewidth=2,
        color=colors[0],
        label="Diffuse + thick atmosphere",
    )
    axes[2].plot(
        phase,
        atmosphere_absolute + thermal_thick,
        linewidth=2,
        color=colors[2],
        label="Forward scattering + thick atmosphere",
    )
    axes[2].legend(loc="upper center", fontsize=12, frameon=False)
    style_axis(axes[2], "Total flux", "Intensity (ppm)", ylim=(0, 20))

    os.makedirs("figures", exist_ok=True)
    output_name = os.environ.get("PLOT_PAPER_OUTPUT", "plot_paper.pdf")
    fig.savefig(os.path.join("figures", output_name), format="pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
