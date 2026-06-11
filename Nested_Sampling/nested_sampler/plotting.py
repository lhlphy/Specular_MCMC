from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def posterior_predictive(spec, posterior_samples, outdir: Path, max_curves: int = 500, seed: int | None = None):
    theta, flux = spec.load_data()
    rng = np.random.default_rng(seed)
    ncurves = min(max_curves, len(posterior_samples))
    indices = rng.choice(len(posterior_samples), size=ncurves, replace=False)
    curves = np.array([spec.predict(theta, posterior_samples[index]) for index in indices])

    q2p5, q16, q50, q84, q97p5 = np.percentile(curves, [2.5, 16, 50, 84, 97.5], axis=0)
    phase = theta / (2.0 * np.pi)
    chi2_values = np.sum(((flux[None, :] - curves) / spec.sigma) ** 2, axis=1)
    coverage_68 = float(np.mean((flux >= q16) & (flux <= q84)))
    coverage_95 = float(np.mean((flux >= q2p5) & (flux <= q97p5)))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(phase, flux, yerr=spec.sigma, fmt="o", color="black", markersize=3, label="Kepler data")
    ax.fill_between(phase, q2p5, q97p5, color=spec.color, alpha=0.16, label="95% predictive band")
    ax.fill_between(phase, q16, q84, color=spec.color, alpha=0.30, label="68% predictive band")
    ax.plot(phase, q50, color=spec.color, linewidth=2.2, label=spec.display_name)
    ax.set_xlabel("Orbital phase")
    ax.set_ylabel("Fp/Fs (ppm)")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False)
    fig.savefig(outdir / "posterior_predictive.pdf", bbox_inches="tight")
    fig.savefig(outdir / "posterior_predictive.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        outdir / "posterior_predictive.npz",
        phase=phase,
        flux=flux,
        sigma=spec.sigma,
        q2p5=q2p5,
        q16=q16,
        q50=q50,
        q84=q84,
        q97p5=q97p5,
        chi2=chi2_values,
        coverage_68=coverage_68,
        coverage_95=coverage_95,
    )
    return {
        "predictive_chi2_median": float(np.median(chi2_values)),
        "predictive_chi2_16": float(np.percentile(chi2_values, 16)),
        "predictive_chi2_84": float(np.percentile(chi2_values, 84)),
        "coverage_68": coverage_68,
        "coverage_95": coverage_95,
        "predictive_curves": int(ncurves),
    }


def corner_plot(spec, posterior_samples, outdir: Path):
    import corner

    fig = corner.corner(posterior_samples, labels=spec.parameter_labels, color=spec.color)
    fig.savefig(outdir / "corner.pdf", bbox_inches="tight")
    plt.close(fig)


def comparison_plot(specs, summaries, results_root: Path):
    fig, ax = plt.subplots(figsize=(9, 6))
    first_pred = None
    for spec in specs:
        pred_path = results_root / spec.name / "posterior_predictive.npz"
        if not pred_path.exists():
            continue
        pred = np.load(pred_path)
        if first_pred is None:
            first_pred = pred
            ax.errorbar(pred["phase"], pred["flux"], yerr=float(pred["sigma"]), fmt="o", color="black", markersize=3, label="Kepler data")
        label = f"{spec.display_name}: dlogZ={summaries[spec.name]['delta_logz']:.2f}"
        ax.fill_between(pred["phase"], pred["q16"], pred["q84"], color=spec.color, alpha=0.20)
        ax.plot(pred["phase"], pred["q50"], color=spec.color, linewidth=2.0, label=label)
    ax.set_xlabel("Orbital phase")
    ax.set_ylabel("Fp/Fs (ppm)")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False)
    comparison_dir = results_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(comparison_dir / "model_compare.pdf", bbox_inches="tight")
    fig.savefig(comparison_dir / "model_compare.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

