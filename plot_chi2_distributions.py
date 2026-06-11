"""Plot fit-statistic distributions from saved MCMC log-likelihood arrays."""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np


TARGET_ROOT = Path("Target")
OUTPUT_DIR = Path("output") / "chi2_distributions"

PLANETS = {
    "K2-141b": {"sigma": 7.05, "color": "tab:purple"},
    "Kepler-10b": {"sigma": 2.5, "color": "tab:green"},
    "Kepler-78b": {"sigma": 3.0, "color": "tab:orange"},
}

MODELS = {
    "limb": {"label": "Specular limb", "color": "crimson", "ndim": 9},
    "lambert_limb": {
        "label": "Lambert limb",
        "color": "royalblue",
        "ndim": 9,
    },
    "atm_limb": {
        "label": "Atmosphere limb",
        "color": "darkorange",
        "ndim": 10,
    },
}


def load_log_likelihood(planet, model):
    target = f"{planet}_{model}"
    target_dir = TARGET_ROOT / target
    data = np.loadtxt(target_dir / "Kepler.txt", delimiter=",")
    log_likelihood = np.load(target_dir / "Kepler_mcmc_log_likelihood.npy")
    log_likelihood = np.asarray(log_likelihood, dtype=float).reshape(-1)
    log_likelihood = log_likelihood[np.isfinite(log_likelihood)]
    return log_likelihood, data.shape[0]


def load_chi2(planet, model, sigma):
    log_likelihood, n_data = load_log_likelihood(planet, model)

    # logL = -0.5 * (chi2 + N * log(2*pi*sigma^2))
    normalization = n_data * np.log(2.0 * np.pi * sigma**2)
    chi2 = -2.0 * log_likelihood - normalization
    return chi2, n_data


def summary_row(planet, model, chi2, n_data):
    return {
        "planet": planet,
        "model": model,
        "n_samples": chi2.size,
        "n_data": n_data,
        "chi2_min": np.min(chi2),
        "chi2_p16": np.percentile(chi2, 16),
        "chi2_median": np.median(chi2),
        "chi2_mean": np.mean(chi2),
        "chi2_p84": np.percentile(chi2, 84),
        "chi2_max": np.max(chi2),
    }


def histogram_range(values):
    # Keep rare poor-fit tail samples in the summary while focusing plots on
    # the posterior bulk.
    lower, upper = np.percentile(values, [0.5, 99.0])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
        return None
    return lower, upper


def annotate_distribution(ax, values, best="min"):
    median = np.median(values)
    best_value = np.min(values) if best == "min" else np.max(values)
    ax.axvline(median, color="black", linestyle="--", linewidth=1.0)
    ax.axvline(best_value, color="black", linestyle=":", linewidth=1.0)
    ax.text(
        0.97,
        0.94,
        f"{best} = {best_value:.2f}\nmedian = {median:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )


def plot_grid(distributions):
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
    for row, planet in enumerate(PLANETS):
        for col, model in enumerate(MODELS):
            ax = axes[row, col]
            chi2 = distributions[(planet, model)]
            ax.hist(
                chi2,
                bins=80,
                range=histogram_range(chi2),
                density=True,
                color=MODELS[model]["color"],
                alpha=0.8,
            )
            annotate_distribution(ax, chi2)
            ax.set_title(f"{planet}: {MODELS[model]['label']}")
            ax.set_xlabel(r"$\chi^2$")
            ax.set_ylabel("Probability density")
            ax.grid(alpha=0.2)

    path = OUTPUT_DIR / "chi2_distributions_3x3.png"
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_planet_overlays(distributions):
    for planet in PLANETS:
        fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
        combined = np.concatenate(
            [distributions[(planet, model)] for model in MODELS]
        )
        common_range = histogram_range(combined)

        for model, settings in MODELS.items():
            chi2 = distributions[(planet, model)]
            ax.hist(
                chi2,
                bins=90,
                range=common_range,
                density=True,
                histtype="step",
                linewidth=1.8,
                color=settings["color"],
                label=(
                    f"{settings['label']} "
                    f"(min={np.min(chi2):.1f}, median={np.median(chi2):.1f})"
                ),
            )

        ax.set_title(f"{planet} posterior chi-square distributions")
        ax.set_xlabel(r"$\chi^2$")
        ax.set_ylabel("Probability density")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        path = OUTPUT_DIR / f"{planet}_chi2_distributions.png"
        fig.savefig(path, dpi=220)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)


def plot_bic_distributions():
    """Plot BIC distributions using BIC = k*ln(N) - 2*logL."""
    distributions = {}
    for planet in PLANETS:
        for model, settings in MODELS.items():
            log_likelihood, n_data = load_log_likelihood(planet, model)
            distributions[(planet, model)] = (
                settings["ndim"] * np.log(n_data) - 2.0 * log_likelihood
            )

    fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
    for row, planet in enumerate(PLANETS):
        for col, (model, settings) in enumerate(MODELS.items()):
            ax = axes[row, col]
            bic = distributions[(planet, model)]
            ax.hist(
                bic,
                bins=80,
                range=histogram_range(bic),
                density=True,
                color=settings["color"],
                alpha=0.8,
            )
            annotate_distribution(ax, bic, best="min")
            ax.set_title(f"{planet}: {settings['label']}")
            ax.set_xlabel("BIC")
            ax.set_ylabel("Probability density")
            ax.grid(alpha=0.2)

    path = OUTPUT_DIR / "bic_distributions_3x3.png"
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)

    for planet in PLANETS:
        fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
        combined = np.concatenate(
            [distributions[(planet, model)] for model in MODELS]
        )
        for model, settings in MODELS.items():
            bic = distributions[(planet, model)]
            ax.hist(
                bic,
                bins=90,
                range=histogram_range(combined),
                density=True,
                histtype="step",
                linewidth=1.8,
                color=settings["color"],
                label=(
                    f"{settings['label']} "
                    f"(min={np.min(bic):.1f}, median={np.median(bic):.1f})"
                ),
            )
        ax.set_title(f"{planet} posterior BIC distributions")
        ax.set_xlabel("BIC")
        ax.set_ylabel("Probability density")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        path = OUTPUT_DIR / f"{planet}_bic_distributions.png"
        fig.savefig(path, dpi=220)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)

    return distributions


def plot_logz_distributions():
    """Plot BIC-approximated log-evidence distributions: logZ ~= -BIC/2."""
    distributions = {}
    for planet in PLANETS:
        for model, settings in MODELS.items():
            log_likelihood, n_data = load_log_likelihood(planet, model)
            bic = settings["ndim"] * np.log(n_data) - 2.0 * log_likelihood
            distributions[(planet, model)] = -0.5 * bic

    fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
    for row, planet in enumerate(PLANETS):
        for col, (model, settings) in enumerate(MODELS.items()):
            ax = axes[row, col]
            logz = distributions[(planet, model)]
            ax.hist(
                logz,
                bins=80,
                range=histogram_range(logz),
                density=True,
                color=settings["color"],
                alpha=0.8,
            )
            annotate_distribution(ax, logz, best="max")
            ax.set_title(f"{planet}: {settings['label']}")
            ax.set_xlabel(r"$\log Z_{\mathrm{BIC}}$")
            ax.set_ylabel("Probability density")
            ax.grid(alpha=0.2)

    fig.suptitle(r"BIC approximation: $\log Z_{\mathrm{BIC}}=-\mathrm{BIC}/2$")
    path = OUTPUT_DIR / "logz_bic_distributions_3x3.png"
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)

    for planet in PLANETS:
        fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
        combined = np.concatenate(
            [distributions[(planet, model)] for model in MODELS]
        )
        for model, settings in MODELS.items():
            logz = distributions[(planet, model)]
            ax.hist(
                logz,
                bins=90,
                range=histogram_range(combined),
                density=True,
                histtype="step",
                linewidth=1.8,
                color=settings["color"],
                label=(
                    f"{settings['label']} "
                    f"(max={np.max(logz):.1f}, median={np.median(logz):.1f})"
                ),
            )
        ax.set_title(f"{planet} BIC-approximated logZ distributions")
        ax.set_xlabel(r"$\log Z_{\mathrm{BIC}}$")
        ax.set_ylabel("Probability density")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        path = OUTPUT_DIR / f"{planet}_logz_bic_distributions.png"
        fig.savefig(path, dpi=220)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)

    return distributions


def plot_delta_bic_vs_lambert(random_seed=20260607):
    """Plot delta BIC relative to Lambert using paired posterior resampling."""
    rng = np.random.default_rng(random_seed)
    comparison_models = {
        "limb": {
            "label": r"Specular $-$ Lambert",
            "color": MODELS["limb"]["color"],
        },
        "atm_limb": {
            "label": r"Atmosphere $-$ Lambert",
            "color": MODELS["atm_limb"]["color"],
        },
    }
    distributions = {}

    for planet in PLANETS:
        logl_lambert, n_data = load_log_likelihood(planet, "lambert_limb")
        bic_lambert = (
            MODELS["lambert_limb"]["ndim"] * np.log(n_data)
            - 2.0 * logl_lambert
        )

        for model in comparison_models:
            log_likelihood, model_n_data = load_log_likelihood(planet, model)
            if model_n_data != n_data:
                raise ValueError(
                    f"{planet} data count differs between {model} and Lambert"
                )
            bic_model = (
                MODELS[model]["ndim"] * np.log(n_data)
                - 2.0 * log_likelihood
            )

            n_pairs = max(bic_model.size, bic_lambert.size)
            model_draws = rng.choice(bic_model, size=n_pairs, replace=True)
            lambert_draws = rng.choice(bic_lambert, size=n_pairs, replace=True)
            distributions[(planet, model)] = model_draws - lambert_draws

    fig, axes = plt.subplots(
        len(PLANETS), 2, figsize=(12, 11), constrained_layout=True
    )
    for row, planet in enumerate(PLANETS):
        combined = np.concatenate(
            [distributions[(planet, model)] for model in comparison_models]
        )
        common_range = histogram_range(combined)
        for col, (model, settings) in enumerate(comparison_models.items()):
            ax = axes[row, col]
            delta_bic = distributions[(planet, model)]
            ax.hist(
                delta_bic,
                bins=90,
                range=common_range,
                density=True,
                color=settings["color"],
                alpha=0.8,
            )
            ax.axvline(0.0, color="black", linewidth=1.2)
            ax.axvline(
                np.median(delta_bic),
                color="black",
                linestyle="--",
                linewidth=1.0,
            )
            probability_better = np.mean(delta_bic < 0.0)
            ax.text(
                0.97,
                0.94,
                (
                    f"median = {np.median(delta_bic):.2f}\n"
                    f"P(Delta BIC < 0) = {probability_better:.3f}"
                ),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
            )
            ax.set_title(f"{planet}: {settings['label']}")
            ax.set_xlabel(r"$\Delta\mathrm{BIC}$ relative to Lambert")
            ax.set_ylabel("Probability density")
            ax.grid(alpha=0.2)

    path = OUTPUT_DIR / "delta_bic_vs_lambert_distributions.png"
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)

    for planet in PLANETS:
        fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
        combined = np.concatenate(
            [distributions[(planet, model)] for model in comparison_models]
        )
        for model, settings in comparison_models.items():
            delta_bic = distributions[(planet, model)]
            ax.hist(
                delta_bic,
                bins=90,
                range=histogram_range(combined),
                density=True,
                histtype="step",
                linewidth=1.8,
                color=settings["color"],
                label=(
                    f"{settings['label']} "
                    f"(median={np.median(delta_bic):.1f}, "
                    f"P<0={np.mean(delta_bic < 0.0):.3f})"
                ),
            )
        ax.axvline(0.0, color="black", linewidth=1.2, label="Equal BIC")
        ax.set_title(f"{planet} delta BIC relative to Lambert")
        ax.set_xlabel(r"$\Delta\mathrm{BIC}$")
        ax.set_ylabel("Probability density")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        path = OUTPUT_DIR / f"{planet}_delta_bic_vs_lambert.png"
        fig.savefig(path, dpi=220)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)

    return distributions


def write_summary(rows):
    path = OUTPUT_DIR / "chi2_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    distributions = {}
    rows = []

    for planet, planet_settings in PLANETS.items():
        for model in MODELS:
            chi2, n_data = load_chi2(
                planet, model, sigma=planet_settings["sigma"]
            )
            distributions[(planet, model)] = chi2
            rows.append(summary_row(planet, model, chi2, n_data))

    plot_grid(distributions)
    plot_planet_overlays(distributions)
    plot_bic_distributions()
    plot_logz_distributions()
    plot_delta_bic_vs_lambert()
    write_summary(rows)

    for row in rows:
        print(
            f"{row['planet']:11s} {row['model']:13s} "
            f"min={row['chi2_min']:.3f} "
            f"median={row['chi2_median']:.3f} "
            f"mean={row['chi2_mean']:.3f}"
        )
    print(f"Saved plots and summary to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
