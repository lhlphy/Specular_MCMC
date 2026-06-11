"""Focused statistic plots for a nested Kepler-78b target folder."""

import os
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np


DATASET = os.environ.get("MCMC_STATS_DATASET", "savgol")
TARGET_ROOT = Path(os.environ.get("MCMC_STATS_TARGET_ROOT", str(Path("Target") / f"Kepler-78b_{DATASET}")))
OUTPUT_DIR = Path(os.environ.get("MCMC_STATS_OUTPUT_DIR", str(TARGET_ROOT)))
PLANET = "Kepler-78b"
PREFIX = os.environ.get("MCMC_STATS_PREFIX", f"Kepler-78b_{DATASET}")
SIGMA = 3.0
RANDOM_SEED = 20260608

MODELS = {
    "limb": {"target": "Kepler-78b_limb", "label": "Specular", "ndim": 9, "color": "crimson"},
    "lambert_limb": {
        "target": "Kepler-78b_lambert_limb",
        "label": "Lambert",
        "ndim": 9,
        "color": "royalblue",
    },
    "atm_limb": {
        "target": "Kepler-78b_atm_limb",
        "label": "Atmosphere",
        "ndim": 10,
        "color": "darkorange",
    },
}
MODEL_ORDER = ("limb", "lambert_limb", "atm_limb")
DELTA_BIC_RANGE = (-50.0, 50.0)


def read_observation(path):
    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline()
    skiprows = 0 if first_line[:1].isdigit() or first_line[:1] == "-" else 1
    delimiter = "," if "," in first_line else None
    return np.loadtxt(path, delimiter=delimiter, skiprows=skiprows)


def focused_range(values):
    lower, upper = np.percentile(values, [1.0, 99.0])
    width = upper - lower
    if not np.isfinite(width) or width <= 0:
        return None
    pad = 0.08 * width
    return lower - pad, upper + pad


def load_case(model_key):
    settings = MODELS[model_key]
    folder = TARGET_ROOT / settings["target"]
    data = read_observation(folder / f"{PREFIX}.txt")
    log_likelihood = np.load(folder / f"{PREFIX}_mcmc_log_likelihood.npy")
    log_likelihood = np.asarray(log_likelihood, dtype=float).reshape(-1)
    log_likelihood = log_likelihood[np.isfinite(log_likelihood)]
    n_data = data.shape[0]
    chi2 = -2.0 * log_likelihood - n_data * np.log(2.0 * np.pi * SIGMA**2)
    bic = settings["ndim"] * np.log(n_data) - 2.0 * log_likelihood
    return {
        "model": model_key,
        "prefix": PREFIX,
        "n_data": n_data,
        "n_samples": log_likelihood.size,
        "chi2": chi2,
        "bic": bic,
        "logz_bic": -0.5 * bic,
    }


def annotate(ax, values, best):
    best_value = np.min(values) if best == "min" else np.max(values)
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
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9), constrained_layout=True)
    for ax, model_key in zip(axes, MODEL_ORDER):
        settings = MODELS[model_key]
        values = cases[model_key][stat_key]
        ax.hist(
            values,
            bins=90,
            range=focused_range(values),
            density=True,
            color=settings["color"],
            alpha=0.82,
        )
        annotate(ax, values, best)
        ax.set_title(f"{PLANET}: {settings['label']} ({PREFIX})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probability density")
        ax.grid(alpha=0.2)

    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_delta_bic(cases):
    rng = np.random.default_rng(RANDOM_SEED)
    baseline = cases["lambert_limb"]["bic"]
    comparisons = ("limb", "atm_limb")
    delta = {}

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), constrained_layout=True)
    for ax, model_key in zip(axes, comparisons):
        settings = MODELS[model_key]
        current = cases[model_key]["bic"]
        n_pairs = max(current.size, baseline.size)
        values = rng.choice(current, size=n_pairs, replace=True) - rng.choice(
            baseline, size=n_pairs, replace=True
        )
        delta[model_key] = values
        in_focus = np.mean((values >= DELTA_BIC_RANGE[0]) & (values <= DELTA_BIC_RANGE[1]))

        ax.hist(
            values,
            bins=100,
            range=DELTA_BIC_RANGE,
            density=True,
            color=settings["color"],
            alpha=0.82,
        )
        ax.axvline(0.0, color="black", linewidth=1.2)
        ax.axvline(np.median(values), color="black", linestyle="--", linewidth=1.0)
        ax.text(
            0.97,
            0.94,
            (
                f"median = {np.median(values):.2f}\n"
                f"P(Delta BIC < 0) = {np.mean(values < 0.0):.3f}\n"
                f"in range = {in_focus:.3f}"
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
        ax.set_title(f"{PLANET}: {settings['label']} - Lambert")
        ax.set_xlabel(r"$\Delta\mathrm{BIC}$ relative to Lambert")
        ax.set_ylabel("Probability density")
        ax.set_xlim(*DELTA_BIC_RANGE)
        ax.grid(alpha=0.2)

    path = OUTPUT_DIR / "delta_bic_vs_lambert_limb_focused.png"
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return delta


def write_summary(cases, delta):
    rows = []
    for model_key in MODEL_ORDER:
        case = cases[model_key]
        row = {
            "planet": PLANET,
            "model": model_key,
            "file_prefix": case["prefix"],
            "n_data": case["n_data"],
            "n_samples": case["n_samples"],
            "chi2_min": np.min(case["chi2"]),
            "chi2_median": np.median(case["chi2"]),
            "bic_min": np.min(case["bic"]),
            "bic_median": np.median(case["bic"]),
            "logz_bic_max": np.max(case["logz_bic"]),
            "logz_bic_median": np.median(case["logz_bic"]),
            "delta_bic_median": "",
            "delta_bic_p_lt_0": "",
            "delta_bic_fraction_in_focused_range": "",
        }
        if model_key in delta:
            values = delta[model_key]
            row["delta_bic_median"] = np.median(values)
            row["delta_bic_p_lt_0"] = np.mean(values < 0.0)
            row["delta_bic_fraction_in_focused_range"] = np.mean(
                (values >= DELTA_BIC_RANGE[0]) & (values <= DELTA_BIC_RANGE[1])
            )
        rows.append(row)

    path = OUTPUT_DIR / "mcmc_statistic_summary_focused.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    cases = {model_key: load_case(model_key) for model_key in MODEL_ORDER}
    plot_stat_grid(cases, "chi2", "chi2_distributions_focused.png", r"$\chi^2$", "min")
    plot_stat_grid(cases, "bic", "bic_distributions_focused.png", "BIC", "min")
    plot_stat_grid(
        cases,
        "logz_bic",
        "logz_bic_distributions_focused.png",
        r"$\log Z_{\mathrm{BIC}}$",
        "max",
    )
    delta = plot_delta_bic(cases)
    rows = write_summary(cases, delta)
    for row in rows:
        print(
            f"{row['planet']:10s} {row['model']:13s} "
            f"prefix={row['file_prefix']:17s} "
            f"chi2_min={float(row['chi2_min']):.3f} "
            f"BIC_min={float(row['bic_min']):.3f} "
            f"logZ_BIC_max={float(row['logz_bic_max']):.3f}"
        )
    print(f"Saved focused statistic plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
