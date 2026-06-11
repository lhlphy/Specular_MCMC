"""Plot chi2, delta-BIC, and BIC-approximated logZ for savgol MCMC outputs."""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np


TARGET_ROOT = Path("Target")
OUTPUT_DIR = Path("output") / "savgol_mcmc_stats"
RANDOM_SEED = 20260608

PLANETS = {
    "K2-141b": {"sigma": 7.05},
    "Kepler-78b": {"sigma": 3.0},
}

MODELS = {
    "limb": {"label": "Specular", "color": "crimson", "ndim": 9},
    "lambert_limb": {"label": "Lambert", "color": "royalblue", "ndim": 9},
    "atm_limb": {"label": "Atmosphere", "color": "darkorange", "ndim": 10},
}

MODEL_ORDER = ("limb", "lambert_limb", "atm_limb")


def target_name(planet, model):
    return f"{planet}_{model}"


def choose_file_prefix(planet, model):
    folder = TARGET_ROOT / target_name(planet, model)
    savgol_prefix = f"{planet}_savgol"
    if (folder / f"{savgol_prefix}_mcmc_log_likelihood.npy").exists():
        return savgol_prefix
    if (folder / "Kepler_mcmc_log_likelihood.npy").exists():
        return "Kepler"
    raise FileNotFoundError(f"No MCMC log-likelihood found in {folder}")


def load_case(planet, model):
    folder = TARGET_ROOT / target_name(planet, model)
    prefix = choose_file_prefix(planet, model)
    data = load_observation_data(folder / f"{prefix}.txt")
    log_likelihood = np.load(folder / f"{prefix}_mcmc_log_likelihood.npy")
    log_likelihood = np.asarray(log_likelihood, dtype=float).reshape(-1)
    log_likelihood = log_likelihood[np.isfinite(log_likelihood)]
    n_data = data.shape[0]
    sigma = PLANETS[planet]["sigma"]
    ndim = MODELS[model]["ndim"]

    chi2 = -2.0 * log_likelihood - n_data * np.log(2.0 * np.pi * sigma**2)
    bic = ndim * np.log(n_data) - 2.0 * log_likelihood
    logz_bic = -0.5 * bic
    return {
        "planet": planet,
        "model": model,
        "prefix": prefix,
        "n_data": n_data,
        "n_samples": log_likelihood.size,
        "chi2": chi2,
        "bic": bic,
        "logz_bic": logz_bic,
    }


def load_observation_data(path):
    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline()
    skiprows = 0 if first_line[:1].isdigit() or first_line[:1] == "-" else 1
    delimiter = "," if "," in first_line else None
    return np.loadtxt(path, delimiter=delimiter, skiprows=skiprows)


def histogram_range(values):
    lower, upper = np.percentile(values, [0.5, 99.0])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
        return None
    return lower, upper


def annotate(ax, values, best):
    if best == "min":
        best_value = np.min(values)
    elif best == "max":
        best_value = np.max(values)
    else:
        raise ValueError(f"Unsupported best={best!r}")
    median = np.median(values)
    ax.axvline(best_value, color="black", linestyle=":", linewidth=1.0)
    ax.axvline(median, color="black", linestyle="--", linewidth=1.0)
    ax.text(
        0.97,
        0.94,
        f"{best} = {best_value:.2f}\nmedian = {median:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )


def plot_stat_grid(cases, stat_key, filename, xlabel, best):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), constrained_layout=True)
    for row, planet in enumerate(PLANETS):
        for col, model in enumerate(MODEL_ORDER):
            ax = axes[row, col]
            case = cases[(planet, model)]
            values = case[stat_key]
            settings = MODELS[model]
            ax.hist(
                values,
                bins=90,
                range=histogram_range(values),
                density=True,
                color=settings["color"],
                alpha=0.8,
            )
            annotate(ax, values, best=best)
            title = f"{planet}: {settings['label']} ({case['prefix']})"
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Probability density")
            ax.grid(alpha=0.2)

    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_delta_bic(cases):
    rng = np.random.default_rng(RANDOM_SEED)
    comparisons = {
        "limb": {
            "label": r"Specular $-$ Lambert",
            "color": MODELS["limb"]["color"],
        },
        "atm_limb": {
            "label": r"Atmosphere $-$ Lambert",
            "color": MODELS["atm_limb"]["color"],
        },
    }
    delta = {}
    for planet in PLANETS:
        lambert = cases[(planet, "lambert_limb")]["bic"]
        for model in comparisons:
            current = cases[(planet, model)]["bic"]
            n_pairs = max(current.size, lambert.size)
            current_draws = rng.choice(current, size=n_pairs, replace=True)
            lambert_draws = rng.choice(lambert, size=n_pairs, replace=True)
            delta[(planet, model)] = current_draws - lambert_draws

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    for row, planet in enumerate(PLANETS):
        combined = np.concatenate([delta[(planet, model)] for model in comparisons])
        common_range = histogram_range(combined)
        for col, (model, settings) in enumerate(comparisons.items()):
            ax = axes[row, col]
            values = delta[(planet, model)]
            ax.hist(
                values,
                bins=90,
                range=common_range,
                density=True,
                color=settings["color"],
                alpha=0.8,
            )
            ax.axvline(0.0, color="black", linewidth=1.2)
            ax.axvline(np.median(values), color="black", linestyle="--", linewidth=1.0)
            ax.text(
                0.97,
                0.94,
                (
                    f"median = {np.median(values):.2f}\n"
                    f"P(Delta BIC < 0) = {np.mean(values < 0.0):.3f}"
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

    path = OUTPUT_DIR / "delta_bic_vs_lambert_savgol.png"
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return delta


def summary_rows(cases, delta_bic):
    rows = []
    for planet in PLANETS:
        for model in MODEL_ORDER:
            case = cases[(planet, model)]
            row = {
                "planet": planet,
                "model": model,
                "file_prefix": case["prefix"],
                "n_data": case["n_data"],
                "n_samples": case["n_samples"],
                "chi2_min": np.min(case["chi2"]),
                "chi2_median": np.median(case["chi2"]),
                "bic_min": np.min(case["bic"]),
                "bic_median": np.median(case["bic"]),
                "logz_bic_max": np.max(case["logz_bic"]),
                "logz_bic_median": np.median(case["logz_bic"]),
                "delta_bic_median_vs_lambert": "",
                "delta_bic_p_lt_0_vs_lambert": "",
            }
            if model in ("limb", "atm_limb"):
                values = delta_bic[(planet, model)]
                row["delta_bic_median_vs_lambert"] = np.median(values)
                row["delta_bic_p_lt_0_vs_lambert"] = np.mean(values < 0.0)
            rows.append(row)
    return rows


def write_summary(rows):
    path = OUTPUT_DIR / "savgol_mcmc_stats_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = {
        (planet, model): load_case(planet, model)
        for planet in PLANETS
        for model in MODEL_ORDER
    }

    plot_stat_grid(cases, "chi2", "chi2_savgol_distributions.png", r"$\chi^2$", best="min")
    plot_stat_grid(cases, "bic", "bic_savgol_distributions.png", "BIC", best="min")
    plot_stat_grid(
        cases,
        "logz_bic",
        "logz_bic_savgol_distributions.png",
        r"$\log Z_{\mathrm{BIC}}$",
        best="max",
    )
    delta_bic = plot_delta_bic(cases)
    rows = summary_rows(cases, delta_bic)
    write_summary(rows)

    for row in rows:
        print(
            f"{row['planet']:10s} {row['model']:13s} "
            f"prefix={row['file_prefix']:17s} "
            f"chi2_min={row['chi2_min']:.3f} "
            f"BIC_min={row['bic_min']:.3f} "
            f"logZ_BIC_max={row['logz_bic_max']:.3f}"
        )
    print(f"Saved savgol MCMC statistic plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
