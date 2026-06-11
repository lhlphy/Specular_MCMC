"""Plot nested Kepler limb experiments from the last N MCMC chain steps."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


PLANET = os.environ.get("MCMC_LAST_PLANET", "Kepler-10b")
DATASET = os.environ.get("MCMC_LAST_DATASET", "slf")
CHAIN_PREFIX = os.environ.get("MCMC_LAST_CHAIN_PREFIX", f"{PLANET}_savgol")
TARGET_ROOT = Path(os.environ.get("MCMC_LAST_TARGET_ROOT", str(Path("Target") / f"{PLANET}_{DATASET}")))
OUTPUT_DIR = Path(os.environ.get("MCMC_LAST_OUTPUT_DIR", str(TARGET_ROOT)))
LAST_STEPS = int(os.environ.get("MCMC_LAST_STEPS", "3000"))
SIGMA = float(os.environ.get("MCMC_LAST_SIGMA", "3.0"))
RANDOM_SEED = int(os.environ.get("MCMC_LAST_RANDOM_SEED", "20260609"))
DELTA_BIC_RANGE = (-50.0, 50.0)

os.environ.setdefault("MCMC_PARAMETER_TARGET", PLANET)

import core_limb.Class_MCMC as core_mcmc
import core_limb.analytical_model as core_model
import core_atm_limb.Class_MCMC as atm_mcmc
import core_atm_limb.analytical_model_atm as atm_model
import core_lambert_limb.Class_MCMC as lambert_mcmc
import core_lambert_limb.analytical_model_Lambert as lambert_model


@dataclass(frozen=True)
class Experiment:
    key: str
    target: str
    label: str
    color: str
    ndim: int
    module: object
    model: object


EXPERIMENTS = (
    Experiment("limb", f"{PLANET}_limb", "Specular", "crimson", 9, core_mcmc, core_model),
    Experiment("atm_limb", f"{PLANET}_atm_limb", "Atmosphere", "darkorange", 10, atm_mcmc, atm_model),
    Experiment(
        "lambert_limb",
        f"{PLANET}_lambert_limb",
        "Lambert",
        "royalblue",
        9,
        lambert_mcmc,
        lambert_model,
    ),
)
PARAMETER_LABELS = {
    "Specular": ["A_lambda", "Tsub", "Rp/Rs", "F", "inc", "alpha", "u1", "u2", "delta"],
    "Atmosphere": ["omega", "g", "Tsub", "Rp/Rs", "F", "inc", "alpha", "u1", "u2", "delta"],
    "Lambert": ["A_lambda", "Tsub", "Rp/Rs", "F", "inc", "alpha", "u1", "u2", "delta"],
}
MODE_LABELS = {
    "posterior_median": "posterior median",
    "posterior_mean": "posterior mean",
    "map": "MAP",
    "mle": "maximum likelihood",
}
MODES = ("posterior_median", "posterior_mean", "map", "mle")


def nested_target_name(exp: Experiment) -> str:
    return str(Path(TARGET_ROOT.name) / exp.target)


def target_folder(exp: Experiment) -> Path:
    return TARGET_ROOT / exp.target


def read_observation(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline()
    skiprows = 0 if first_line[:1].isdigit() or first_line[:1] == "-" else 1
    delimiter = "," if "," in first_line else None
    return np.loadtxt(path, delimiter=delimiter, skiprows=skiprows)


def load_last_chain_case(exp: Experiment):
    folder = target_folder(exp)
    chain = np.load(folder / f"{CHAIN_PREFIX}_mcmc_chain.npy", mmap_mode="r")
    logl_chain = np.load(folder / f"{CHAIN_PREFIX}_mcmc_log_likelihood_chain.npy", mmap_mode="r")
    logp_chain = np.load(folder / f"{CHAIN_PREFIX}_mcmc_log_posterior_chain.npy", mmap_mode="r")
    if chain.shape[0] < LAST_STEPS:
        raise ValueError(f"{folder} has only {chain.shape[0]} chain steps, less than {LAST_STEPS}.")
    samples = np.asarray(chain[-LAST_STEPS:]).reshape(-1, chain.shape[-1])
    log_likelihood = np.asarray(logl_chain[-LAST_STEPS:]).reshape(-1)
    log_posterior = np.asarray(logp_chain[-LAST_STEPS:]).reshape(-1)
    finite = np.isfinite(log_likelihood) & np.isfinite(log_posterior)
    samples = samples[finite]
    log_likelihood = log_likelihood[finite]
    log_posterior = log_posterior[finite]

    data = read_observation(folder / f"{CHAIN_PREFIX}.txt")
    n_data = data.shape[0]
    chi2 = -2.0 * log_likelihood - n_data * np.log(2.0 * np.pi * SIGMA**2)
    bic = exp.ndim * np.log(n_data) - 2.0 * log_likelihood
    return {
        "exp": exp,
        "samples": samples,
        "log_likelihood": log_likelihood,
        "log_posterior": log_posterior,
        "n_data": n_data,
        "chi2": chi2,
        "bic": bic,
        "logz_bic": -0.5 * bic,
    }


def focused_range(values):
    lower, upper = np.percentile(values, [1.0, 99.0])
    width = upper - lower
    if not np.isfinite(width) or width <= 0:
        return None
    pad = 0.08 * width
    return lower - pad, upper + pad


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


def savefig(fig, stem: str):
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_stat_grid(cases, stat_key, stem, xlabel, best):
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9), constrained_layout=True)
    for ax, case in zip(axes, cases):
        exp = case["exp"]
        values = case[stat_key]
        ax.hist(
            values,
            bins=90,
            range=focused_range(values),
            density=True,
            color=exp.color,
            alpha=0.82,
        )
        annotate(ax, values, best)
        ax.set_title(f"{PLANET}: {exp.label} ({CHAIN_PREFIX}, last {LAST_STEPS})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probability density")
        ax.grid(alpha=0.2)
    savefig(fig, stem)


def plot_delta_bic(cases):
    by_key = {case["exp"].key: case for case in cases}
    baseline = by_key["lambert_limb"]["bic"]
    rng = np.random.default_rng(RANDOM_SEED)
    delta = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), constrained_layout=True)
    for ax, key in zip(axes, ("limb", "atm_limb")):
        case = by_key[key]
        current = case["bic"]
        n_pairs = max(current.size, baseline.size)
        values = rng.choice(current, size=n_pairs, replace=True) - rng.choice(
            baseline, size=n_pairs, replace=True
        )
        delta[key] = values
        in_focus = np.mean((values >= DELTA_BIC_RANGE[0]) & (values <= DELTA_BIC_RANGE[1]))
        exp = case["exp"]
        ax.hist(values, bins=100, range=DELTA_BIC_RANGE, density=True, color=exp.color, alpha=0.82)
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
        ax.set_title(f"{PLANET}: {exp.label} - Lambert")
        ax.set_xlabel(r"$\Delta\mathrm{BIC}$ relative to Lambert")
        ax.set_ylabel("Probability density")
        ax.set_xlim(*DELTA_BIC_RANGE)
        ax.grid(alpha=0.2)
    savefig(fig, f"delta_bic_vs_lambert_limb_last{LAST_STEPS}")
    return delta


def instantiate_mcmc(exp: Experiment):
    return exp.module.MCMC(
        nested_target_name(exp),
        CHAIN_PREFIX,
        sigma=SIGMA,
        ndim=exp.ndim,
        nwalkers=64,
        nsteps=LAST_STEPS,
        burnin=0,
    )


def select_params(case, mode):
    samples = case["samples"]
    if mode == "posterior_median":
        idx = None
        params = np.percentile(samples, 50, axis=0)
    elif mode == "posterior_mean":
        idx = None
        params = np.mean(samples, axis=0)
    elif mode == "map":
        idx = int(np.argmax(case["log_posterior"]))
        params = samples[idx]
    elif mode == "mle":
        idx = int(np.argmax(case["log_likelihood"]))
        params = samples[idx]
    else:
        raise ValueError(mode)
    return params, idx


def evaluate_model(exp: Experiment, mcmc, params):
    os.environ["FOLDER_PATH"] = str(target_folder(exp))
    return exp.model.Fp2Fs(mcmc.data_X, co1=mcmc.Co1, co2=mcmc.Co2, params=params)


def chi2(data_y, model, sigma):
    return float(np.sum(((data_y - model) / sigma) ** 2))


def plot_model_mode(cases, mode):
    results = []
    for case in cases:
        exp = case["exp"]
        mcmc = instantiate_mcmc(exp)
        params, idx = select_params(case, mode)
        model = evaluate_model(exp, mcmc, params)
        if idx is None:
            log_likelihood = -0.5 * np.sum(
                (mcmc.data_Y - model) ** 2 / mcmc.sigma**2
                + np.log(2.0 * np.pi * mcmc.sigma**2)
            )
            log_posterior = log_likelihood
        else:
            log_likelihood = float(case["log_likelihood"][idx])
            log_posterior = float(case["log_posterior"][idx])
        results.append(
            {
                "case": case,
                "mcmc": mcmc,
                "params": params,
                "model": model,
                "chi2": chi2(mcmc.data_Y, model, mcmc.sigma),
                "log_likelihood": log_likelihood,
                "log_posterior": log_posterior,
                "idx": idx,
            }
        )

    reference = results[0]["mcmc"]
    sort_idx = np.argsort(reference.data_X)
    phase = reference.data_X[sort_idx] / (2.0 * np.pi)
    data_y = reference.data_Y[sort_idx]

    fig = plt.figure(figsize=(9, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax1 = fig.add_subplot(gs[0])
    ax1.errorbar(
        phase,
        data_y,
        yerr=reference.sigma,
        fmt="o",
        color="black",
        markersize=3,
        label="Kepler data",
        zorder=4,
    )
    for result in results:
        exp = result["case"]["exp"]
        ax1.plot(
            phase,
            result["model"][sort_idx],
            "-",
            color=exp.color,
            linewidth=2.4,
            label=f"{exp.label}: chi2={result['chi2']:.2f}",
            zorder=5,
        )
    ax1.set_ylabel("Fp/Fs (ppm)", fontsize=15)
    ax1.tick_params(axis="both", labelsize=12)
    ax1.legend(fontsize=10, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-20, (np.max(data_y) + float(np.max(reference.sigma))) * 1.15)
    ax1.set_xticklabels([])

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    max_residual = 0.0
    for result in results:
        exp = result["case"]["exp"]
        residuals = data_y - result["model"][sort_idx]
        max_residual = max(max_residual, float(np.max(np.abs(residuals))))
        ax2.plot(phase, residuals, "o", color=exp.color, markersize=3, alpha=0.7)
    ax2.axhline(0, color="black", linestyle="--", linewidth=1)
    ax2.set_xlabel("Orbital phase", fontsize=15)
    ax2.set_ylabel("Residuals (ppm)", fontsize=15)
    ax2.tick_params(axis="both", labelsize=12)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-max_residual * 1.2, max_residual * 1.2)
    savefig(fig, f"{PLANET}_limb_model_compare_{CHAIN_PREFIX}_last{LAST_STEPS}_{mode}")

    print(f"\nchi2 summary ({MODE_LABELS[mode]}, last {LAST_STEPS}):")
    for result in results:
        exp = result["case"]["exp"]
        idx_text = "" if result["idx"] is None else f", index={result['idx']}"
        print(
            f"{exp.label}: chi2={result['chi2']:.4f}, "
            f"logL={result['log_likelihood']:.4f}, "
            f"logPost={result['log_posterior']:.4f}{idx_text}"
        )


def plot_albedo_distribution(cases):
    import corner

    fig = None
    for case in cases:
        exp = case["exp"]
        label = r"$\omega$" if exp.key == "atm_limb" else r"$A_{\lambda}$"
        fig = corner.corner(
            case["samples"][:, [0]],
            fig=fig,
            labels=[label],
            color=exp.color,
            hist_kwargs={"histtype": "step", "linewidth": 1.4},
        )
    ax = fig.axes[0]
    ax.set_ylabel("Sample count", fontsize=10)
    ax.set_xlabel(r"$A_{\lambda}$ or $\omega$", fontsize=10)
    ax.tick_params(axis="both", labelsize=8)
    ax.xaxis.set_label_coords(0.5, -0.12)
    savefig(fig, f"A_lambda_omega_compare_{CHAIN_PREFIX}_last{LAST_STEPS}")


def write_summary(cases, delta):
    rows = []
    for case in cases:
        exp = case["exp"]
        row = {
            "planet": PLANET,
            "model": exp.key,
            "file_prefix": CHAIN_PREFIX,
            "last_steps": LAST_STEPS,
            "n_data": case["n_data"],
            "n_samples": len(case["samples"]),
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
        if exp.key in delta:
            values = delta[exp.key]
            row["delta_bic_median"] = np.median(values)
            row["delta_bic_p_lt_0"] = np.mean(values < 0.0)
            row["delta_bic_fraction_in_focused_range"] = np.mean(
                (values >= DELTA_BIC_RANGE[0]) & (values <= DELTA_BIC_RANGE[1])
            )
        rows.append(row)

    path = OUTPUT_DIR / f"mcmc_statistic_summary_{CHAIN_PREFIX}_last{LAST_STEPS}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = [load_last_chain_case(exp) for exp in EXPERIMENTS]
    plot_stat_grid(cases, "chi2", f"chi2_distributions_{CHAIN_PREFIX}_last{LAST_STEPS}", r"$\chi^2$", "min")
    plot_stat_grid(cases, "bic", f"bic_distributions_{CHAIN_PREFIX}_last{LAST_STEPS}", "BIC", "min")
    plot_stat_grid(
        cases,
        "logz_bic",
        f"logz_bic_distributions_{CHAIN_PREFIX}_last{LAST_STEPS}",
        r"$\log Z_{\mathrm{BIC}}$",
        "max",
    )
    delta = plot_delta_bic(cases)
    rows = write_summary(cases, delta)
    for mode in MODES:
        plot_model_mode(cases, mode)
    plot_albedo_distribution(cases)
    for row in rows:
        print(
            f"{row['planet']:10s} {row['model']:13s} "
            f"prefix={row['file_prefix']:17s} "
            f"last={row['last_steps']} "
            f"chi2_min={float(row['chi2_min']):.3f} "
            f"BIC_min={float(row['bic_min']):.3f} "
            f"logZ_BIC_max={float(row['logz_bic_max']):.3f}"
        )
    print(f"\nSaved last-{LAST_STEPS} plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
