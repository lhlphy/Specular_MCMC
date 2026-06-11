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


def build_atmospheric_forward_scattering():
    x = np.linspace(0, np.pi, num=1000)
    Theta_array = np.linspace(0, 2 * np.pi, 2000)
    atmosphere = mathematica_atmospheric_forward_scattering(x)
    F_atmos = np.array([atmosphere, atmosphere[::-1]]).flatten()
    F_atmos[np.abs(Theta_array - np.pi) < PPs.alpha] = 0 # eclipse
    return F_atmos 


def style_axis(ax, title, ylabel=None, ylim=None, yticks=None):
    ax.set_title(title, fontsize=17)
    ax.set_xlabel("Orbital Phase", fontsize=16)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlim(0, 1)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.tick_params(axis="both", labelsize=14)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def main():
    theta_array = np.linspace(0, 2 * np.pi, 2000)
    phase = theta_array / (2 * np.pi)
    ab = 0.1
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    if "FOLDER_PATH" not in os.environ:
        analytic.Response = lambda lam: np.ones_like(lam, dtype=float)

    specular_absolute = F_specular(theta_array, ab, PPs.Rp2Rs, normalize=False)
    diffuse_absolute = F_lambert(theta_array, ab, PPs.Rp2Rs, normalize=False)
    atmosphere_absolute = build_atmospheric_forward_scattering()
    thermal_thick = F_thermal(theta_array, ab, F=0.5, Rp2Rs=PPs.Rp2Rs)
    thermal_bare = F_thermal(theta_array, ab, F=0, Rp2Rs=PPs.Rp2Rs)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    axes[0].plot(
        phase,
        specular_absolute,
        color=colors[1],
        label="Specular (lava ocean)",
        linewidth=2,
    )
    axes[0].plot(
        phase,
        diffuse_absolute,
        color="black",
        label="Diffuse (rough surface)",
        linewidth=2,
    )
    axes[0].plot(
        phase,
        atmosphere_absolute,
        color=colors[0],
        label="Forward scattering (Thick atmosphere)",
        linewidth=2,
    )
    axes[0].legend(loc="upper center", fontsize=14, frameon=False)
    style_axis(axes[0], "Reflection", "Intensity (ppm)", yticks=[0, 5, 10, 15, 20], ylim=(-1, 20))

    axes[1].plot(
        phase,
        thermal_thick,
        color=colors[0],
        label="Full heat redistribution",
        linewidth=2,
    )
    axes[1].plot(
        phase,
        thermal_bare,
        color=colors[1],
        label="No heat redistribution",
        linewidth=2,
    )
    axes[1].legend(loc="upper center", fontsize=14, frameon=False)
    style_axis(axes[1], "Thermal emission", yticks=[0, 5, 10, 15, 20], ylim=(-1, 20))

    axes[2].plot(
        phase,
        specular_absolute + thermal_bare,
        linewidth=2,
        color=colors[1],
        label="Specular + No heat redistribution",
    )
    axes[2].plot(
        phase,
        diffuse_absolute + thermal_bare,
        linewidth=2,
        color='black',
        label="Diffuse + No heat redistribution",
    )
    axes[2].plot(
        phase,
        atmosphere_absolute + thermal_thick,
        linewidth=2,
        color=colors[0],
        label="Thick atmosphere + Full heat redistribution",
    )
    axes[2].legend(loc="upper center", fontsize=14, frameon=False)
    style_axis(axes[2], "Total flux", yticks=[0, 5, 10, 15, 20, 25, 30], ylim=(-1, 30))

    os.makedirs("figures", exist_ok=True)
    output_name = os.environ.get("PLOT_PAPER_ABS_OUTPUT", "plot_paper_abs.pdf")
    fig.savefig(os.path.join("figures", output_name), format="pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
