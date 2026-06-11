import os
import warnings
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

COMPARE_PLANET = os.environ.get("MCMC_COMPARE_PLANET", "K2-141b")
PARAMETER_TARGETS = {
    "K2-141b": "K2-141b",
    "Kepler-10b": "Kepler-10b",
    "Kepler-78b": "Kepler-78b",
}
if "MCMC_PARAMETER_TARGET" not in os.environ:
    os.environ["MCMC_PARAMETER_TARGET"] = PARAMETER_TARGETS.get(COMPARE_PLANET, COMPARE_PLANET)

import core_limb.Class_MCMC as core_mcmc
import core_limb.analytical_model as core_model
import core_atm_limb.Class_MCMC as atm_mcmc
import core_atm_limb.analytical_model_atm as atm_model
import core_lambert_limb.Class_MCMC as lambert_mcmc
import core_lambert_limb.analytical_model_Lambert as lambert_model

warnings.filterwarnings("ignore")


PLANET_CONFIGS = {
    "K2-141b": {
        "output_dir": os.path.join("output", "K2-141b_limb"),
        "file_name": "Kepler",
        "sigma": 7.05,
        "base_name": "K2-141b_limb_model_compare",
        "experiments": {
            "Specular": {"name": "K2-141b_limb", "nsteps": 4000, "burnin": 1500},
            "Atmosphere": {"name": "K2-141b_atm_limb", "nsteps": 5000, "burnin": 2500},
            "Lambert": {"name": "K2-141b_lambert_limb", "nsteps": 4000, "burnin": 1500},
        },
    },
    "Kepler-78b": {
        "output_dir": os.path.join("output", "Kepler-78b_limb"),
        "file_name": "Kepler-78b_savgol",
        "sigma": 3.0,
        "base_name": "Kepler-78b_limb_model_compare",
        "experiments": {
            "Specular": {"name": "Kepler-78b_limb", "nsteps": 6000, "burnin": 3000},
            "Atmosphere": {"name": "Kepler-78b_atm_limb", "nsteps": 5000, "burnin": 2500},
            "Lambert": {"name": "Kepler-78b_lambert_limb", "nsteps": 6000, "burnin": 3000},
        },
    },
    "Kepler-10b": {
        "output_dir": os.path.join("output", "Kepler-10b_limb"),
        "file_name": "Kepler-10b_savgol",
        "sigma": 3.0,
        "base_name": "Kepler-10b_limb_model_compare",
        "experiments": {
            "Specular": {"name": "Kepler-10b_limb", "nsteps": 6000, "burnin": 3000},
            "Atmosphere": {"name": "Kepler-10b_atm_limb", "nsteps": 5000, "burnin": 2500},
            "Lambert": {"name": "Kepler-10b_lambert_limb", "nsteps": 6000, "burnin": 3000},
        },
    },
}

if COMPARE_PLANET not in PLANET_CONFIGS:
    valid_planets = ", ".join(sorted(PLANET_CONFIGS))
    raise ValueError(f"Unknown MCMC_COMPARE_PLANET={COMPARE_PLANET!r}. Expected one of: {valid_planets}")

PLANET_CONFIG = PLANET_CONFIGS[COMPARE_PLANET]
OUTPUT_DIR = os.environ.get("MCMC_COMPARE_OUTPUT_DIR", PLANET_CONFIG["output_dir"])
TARGET_ROOT = os.environ.get("MCMC_COMPARE_TARGET_ROOT", "Target")
TARGET_NAME_PREFIX = os.path.relpath(TARGET_ROOT, "Target")
if TARGET_NAME_PREFIX == ".":
    TARGET_NAME_PREFIX = ""
FILE_NAME = os.environ.get("MCMC_COMPARE_FILE_NAME", PLANET_CONFIG["file_name"])
SIGMA = float(os.environ.get("MCMC_COMPARE_SIGMA", str(PLANET_CONFIG["sigma"])))
OUTPUT_BASE_NAME = os.environ.get("MCMC_COMPARE_BASE_NAME", PLANET_CONFIG["base_name"])
CURVE_INTERVAL_SAMPLE_COUNT = int(os.environ.get("MCMC_COMPARE_INTERVAL_SAMPLES", "1000"))
CURVE_INTERVAL_RANDOM_SEED = int(os.environ.get("MCMC_COMPARE_RANDOM_SEED", "20260606"))
CURVE_INTERVAL_WORKERS = int(os.environ.get("MCMC_COMPARE_INTERVAL_WORKERS", "8"))
DRAW_CURVE_INTERVAL = os.environ.get("MCMC_COMPARE_DRAW_INTERVAL", "1") != "0"
DEFAULT_ESTIMATE_MODES = ("posterior_median", "posterior_mean", "map", "mle")
ESTIMATE_MODES = tuple(
    mode.strip()
    for mode in os.environ.get("MCMC_COMPARE_MODES", ",".join(DEFAULT_ESTIMATE_MODES)).split(",")
    if mode.strip()
)


@dataclass
class Experiment:
    name: str
    label: str
    color: str
    ndim: int
    nsteps: int
    burnin: int
    mcmc_module: object
    model_module: object


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


MODEL_MODULES_BY_LABEL = {
    "Specular": core_model,
    "Atmosphere": atm_model,
    "Lambert": lambert_model,
}


EXPERIMENT_DEFAULTS = {
    "Specular": {
        "color": "crimson",
        "ndim": 9,
        "mcmc_module": core_mcmc,
        "model_module": core_model,
    },
    "Atmosphere": {
        "color": "darkorange",
        "ndim": 10,
        "mcmc_module": atm_mcmc,
        "model_module": atm_model,
    },
    "Lambert": {
        "color": "royalblue",
        "ndim": 9,
        "mcmc_module": lambert_mcmc,
        "model_module": lambert_model,
    },
}


def build_experiments():
    experiments = []
    for label in ("Specular", "Atmosphere", "Lambert"):
        planet_settings = PLANET_CONFIG["experiments"][label]
        defaults = EXPERIMENT_DEFAULTS[label]
        target_name = planet_settings["name"]
        if TARGET_NAME_PREFIX:
            target_name = os.path.join(TARGET_NAME_PREFIX, target_name)
        experiments.append(
            Experiment(
                name=target_name,
                label=label,
                color=defaults["color"],
                ndim=defaults["ndim"],
                nsteps=planet_settings["nsteps"],
                burnin=planet_settings["burnin"],
                mcmc_module=defaults["mcmc_module"],
                model_module=defaults["model_module"],
            )
        )
    return experiments


EXPERIMENTS = build_experiments()


def chi2(data_y, data_model, errorbar):
    return np.sum(((data_y - data_model) / errorbar) ** 2)


def max_errorbar(errorbar):
    return float(np.max(errorbar))


def target_folder(target_name):
    return os.path.join("Target", target_name)


def sample_path(target_name):
    return os.path.join(target_folder(target_name), f"{FILE_NAME}_mcmc_samples.npy")


def score_path(target_name, score_name):
    return os.path.join(target_folder(target_name), f"{FILE_NAME}_mcmc_{score_name}.npy")


def curve_interval_cache_path(exp):
    return os.path.join(
        OUTPUT_DIR,
        f"{exp.name}_{FILE_NAME}_curve_interval_{CURVE_INTERVAL_SAMPLE_COUNT}_{CURVE_INTERVAL_RANDOM_SEED}.npz",
    )


def estimate_interval(samples):
    lower = np.percentile(samples, 16, axis=0)
    upper = np.percentile(samples, 84, axis=0)
    return lower, upper


def evaluate_curve_worker(args):
    label, folder_path, data_x, co1, co2, params = args
    os.environ["FOLDER_PATH"] = folder_path
    return MODEL_MODULES_BY_LABEL[label].Fp2Fs(
        data_x,
        co1=co1,
        co2=co2,
        params=params,
    )


def calculate_curve_interval(exp, mcmc, samples, random_seed):
    cache_path = curve_interval_cache_path(exp)
    if os.path.exists(cache_path):
        cache = np.load(cache_path)
        if (
            int(cache["sample_count"]) == min(CURVE_INTERVAL_SAMPLE_COUNT, len(samples))
            and int(cache["random_seed"]) == random_seed
            and cache["lower"].shape == mcmc.data_X.shape
            and cache["upper"].shape == mcmc.data_X.shape
        ):
            print(f"  loaded phase-curve 1-sigma interval cache: {cache_path}")
            return {
                "lower": cache["lower"],
                "upper": cache["upper"],
                "sample_count": int(cache["sample_count"]),
            }

    sample_count = min(CURVE_INTERVAL_SAMPLE_COUNT, len(samples))
    rng = np.random.default_rng(random_seed)
    sample_indices = rng.choice(len(samples), size=sample_count, replace=False)
    tasks = [
        (
            exp.label,
            target_folder(exp.name),
            mcmc.data_X,
            mcmc.Co1,
            mcmc.Co2,
            samples[sample_idx],
        )
        for sample_idx in sample_indices
    ]

    os.environ["FOLDER_PATH"] = target_folder(exp.name)
    max_workers = max(1, min(CURVE_INTERVAL_WORKERS, os.cpu_count() or 1))
    print(
        f"  calculating phase-curve 1-sigma interval from {sample_count} samples "
        f"with {max_workers} workers"
    )
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        curves = np.array(list(executor.map(evaluate_curve_worker, tasks, chunksize=4)))

    lower = np.percentile(curves, 16, axis=0)
    upper = np.percentile(curves, 84, axis=0)
    np.savez(
        cache_path,
        lower=lower,
        upper=upper,
        sample_count=sample_count,
        random_seed=random_seed,
        sample_indices=sample_indices,
    )

    return {
        "lower": lower,
        "upper": upper,
        "sample_count": sample_count,
    }


def load_saved_scores(exp, samples):
    scores = {}
    for score_name in ("log_likelihood", "log_posterior"):
        path = score_path(exp.name, score_name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{exp.name} is missing {path}. MAP/MLE plotting requires saved "
                "scores aligned with Kepler_mcmc_samples.npy; scores are not recomputed."
            )
        values = np.load(path)
        if values.ndim != 1 or len(values) != len(samples):
            raise ValueError(
                f"{path} has shape {values.shape}, but samples have shape {samples.shape}. "
                "MAP/MLE plotting requires one saved score per sample."
            )
        scores[score_name] = values
    return scores


def saved_score_parameter_estimate(exp, mcmc, samples, mode):
    scores = load_saved_scores(exp, samples)
    selection_score = scores["log_posterior"] if mode == "map" else scores["log_likelihood"]
    finite_idx = np.flatnonzero(np.isfinite(selection_score))
    if finite_idx.size == 0:
        raise ValueError(f"{exp.name} has no finite saved {mode.upper()} scores.")

    sample_idx = finite_idx[np.argmax(selection_score[finite_idx])]
    params = samples[sample_idx]
    os.environ["FOLDER_PATH"] = target_folder(exp.name)
    model = exp.model_module.Fp2Fs(
        mcmc.data_X,
        co1=mcmc.Co1,
        co2=mcmc.Co2,
        params=params,
    )
    print(f"  selected saved sample index {sample_idx} for {mode.upper()}")
    return (
        params,
        model,
        float(scores["log_likelihood"][sample_idx]),
        float(scores["log_posterior"][sample_idx]),
    )


def choose_parameters(exp, mcmc, samples, mode):
    lower, upper = estimate_interval(samples)

    if mode == "posterior_median":
        params = np.percentile(samples, 50, axis=0)
        os.environ["FOLDER_PATH"] = target_folder(exp.name)
        model = exp.model_module.Fp2Fs(mcmc.data_X, co1=mcmc.Co1, co2=mcmc.Co2, params=params)
        log_likelihood = -0.5 * np.sum(
            (mcmc.data_Y - model) ** 2 / mcmc.sigma**2
            + np.log(2 * np.pi * mcmc.sigma**2)
        )
        log_prior = mcmc.log_prior(params)
        log_posterior = log_prior + log_likelihood if np.isfinite(log_prior) else -np.inf
    elif mode == "posterior_mean":
        params = np.mean(samples, axis=0)
        os.environ["FOLDER_PATH"] = target_folder(exp.name)
        model = exp.model_module.Fp2Fs(mcmc.data_X, co1=mcmc.Co1, co2=mcmc.Co2, params=params)
        log_likelihood = -0.5 * np.sum(
            (mcmc.data_Y - model) ** 2 / mcmc.sigma**2
            + np.log(2 * np.pi * mcmc.sigma**2)
        )
        log_prior = mcmc.log_prior(params)
        log_posterior = log_prior + log_likelihood if np.isfinite(log_prior) else -np.inf
    elif mode in {"map", "mle"}:
        params, model, log_likelihood, log_posterior = saved_score_parameter_estimate(
            exp, mcmc, samples, mode
        )
    else:
        raise ValueError(f"Unknown estimate mode: {mode}")

    return {
        "params": params,
        "lower": lower,
        "upper": upper,
        "model": model,
        "chi2": chi2(mcmc.data_Y, model, mcmc.sigma),
        "log_likelihood": log_likelihood,
        "log_posterior": log_posterior,
    }


def print_parameter_diagnostics(result):
    exp = result["experiment"]
    labels = PARAMETER_LABELS[exp.label]
    params = result["params"]
    lower = result["lower"]
    upper = result["upper"]

    print(f"\n{exp.label} plotting parameters vs posterior ranges:")
    for label, value, lo, hi in zip(labels, params, lower, upper):
        print(f"  {label}: plot={value:.6f}, p16={lo:.6f}, p84={hi:.6f}")


def load_experiment(exp, mode, curve_interval_cache, random_seed):
    mcmc = exp.mcmc_module.MCMC(
        exp.name,
        FILE_NAME,
        sigma=SIGMA,
        ndim=exp.ndim,
        nwalkers=64,
        nsteps=exp.nsteps,
        burnin=exp.burnin,
    )

    print(f"\nThe parameters of the {exp.label} model ({MODE_LABELS[mode]}):")
    samples = mcmc.load_samples()
    if DRAW_CURVE_INTERVAL and exp.name not in curve_interval_cache:
        curve_interval_cache[exp.name] = calculate_curve_interval(
            exp,
            mcmc,
            samples,
            random_seed,
        )
    estimate = choose_parameters(exp, mcmc, samples, mode)

    return {
        "experiment": exp,
        "mode": mode,
        "mcmc": mcmc,
        "samples": samples,
        "params": estimate["params"],
        "lower": estimate["lower"],
        "upper": estimate["upper"],
        "model": estimate["model"],
        "chi2": estimate["chi2"],
        "log_likelihood": estimate["log_likelihood"],
        "log_posterior": estimate["log_posterior"],
        "curve_lower": curve_interval_cache[exp.name]["lower"] if DRAW_CURVE_INTERVAL else None,
        "curve_upper": curve_interval_cache[exp.name]["upper"] if DRAW_CURVE_INTERVAL else None,
        "curve_interval_sample_count": (
            curve_interval_cache[exp.name]["sample_count"] if DRAW_CURVE_INTERVAL else 0
        ),
    }


def plot_model_comparison(results):
    reference = results[0]["mcmc"]
    mode = results[0]["mode"]
    sort_idx = np.argsort(reference.data_X)
    phase = reference.data_X[sort_idx] / (2 * np.pi)
    data_y = reference.data_Y
    data_y_sorted = data_y[sort_idx]

    fig = plt.figure(figsize=(9, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)

    ax1 = fig.add_subplot(gs[0])

    if DRAW_CURVE_INTERVAL:
        # Draw uncertainty bands first. Atmosphere is last in this layer so its
        # orange band remains visible where the three model bands overlap.
        fill_results = sorted(
            results,
            key=lambda item: item["experiment"].label == "Atmosphere",
        )
        for result in fill_results:
            exp = result["experiment"]
            curve_lower_sorted = result["curve_lower"][sort_idx]
            curve_upper_sorted = result["curve_upper"][sort_idx]
            ax1.fill_between(
                phase,
                curve_lower_sorted,
                curve_upper_sorted,
                color=exp.color,
                alpha=0.2,
                linewidth=0,
                zorder=1,
            )

    ax1.errorbar(
        phase,
        data_y_sorted,
        yerr=reference.sigma,
        fmt="o",
        color="black",
        markersize=3,
        label="Kepler data",
        zorder=4,
    )

    for result in results:
        exp = result["experiment"]
        model_sorted = result["model"][sort_idx]
        ax1.plot(
            phase,
            model_sorted,
            "-",
            color=exp.color,
            linewidth=2.4,
            label=f"{exp.label}: chi2={result['chi2']:.2f}",
            zorder=5,
        )

    ax1.set_ylabel("Fp/Fs (ppm)", fontsize=15)
    ax1.tick_params(axis="both", labelsize=12)
    ax1.legend(
        fontsize=10,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
    )
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-20, (np.max(data_y_sorted) + max_errorbar(reference.sigma)) * 1.15)
    ax1.set_xticklabels([])

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    max_residual = 0.0
    for result in results:
        exp = result["experiment"]
        model_sorted = result["model"][sort_idx]
        residuals = data_y_sorted - model_sorted
        max_residual = max(max_residual, np.max(np.abs(residuals)))
        ax2.plot(
            phase,
            residuals,
            "o",
            color=exp.color,
            markersize=3,
            alpha=0.7,
            label=exp.label,
        )

    ax2.axhline(0, color="black", linestyle="--", linewidth=1)
    ax2.set_xlabel("Orbital phase", fontsize=15)
    ax2.set_ylabel("Residuals (ppm)", fontsize=15)
    ax2.tick_params(axis="both", labelsize=12)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-max_residual * 1.2, max_residual * 1.2)

    base_name = f"{OUTPUT_BASE_NAME}_{mode}"
    fig.savefig(os.path.join(OUTPUT_DIR, f"{base_name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUTPUT_DIR, f"{base_name}.png"), dpi=200, bbox_inches="tight")

    ax1.set_ylim(-20, (np.max(data_y_sorted) + max_errorbar(reference.sigma)) * 1.15)
    fig.savefig(os.path.join(OUTPUT_DIR, f"{base_name}_nT.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_albedo_distribution(results):
    import corner

    fig = None
    for result in results:
        exp = result["experiment"]
        label = r"$\omega$" if "_atm" in exp.name else r"$A_{\lambda}$"
        data = result["samples"][:, [0]]
        fig = corner.corner(
            data,
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
    fig.savefig(os.path.join(OUTPUT_DIR, "A_lambda_omega_compare.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUTPUT_DIR, "A_lambda_omega_compare.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = {}
    curve_interval_cache = {}

    for mode in ESTIMATE_MODES:
        if mode not in MODE_LABELS:
            raise ValueError(f"Unknown mode in MCMC_COMPARE_MODES: {mode}")
        if mode in {"map", "mle"}:
            for exp in EXPERIMENTS:
                samples = np.load(sample_path(exp.name), mmap_mode="r")
                load_saved_scores(exp, samples)
        print(f"\n=== Plotting with {MODE_LABELS[mode]} parameters ===")
        results = [
            load_experiment(
                exp,
                mode,
                curve_interval_cache,
                CURVE_INTERVAL_RANDOM_SEED + exp_idx,
            )
            for exp_idx, exp in enumerate(EXPERIMENTS)
        ]
        all_results[mode] = results
        for result in results:
            print_parameter_diagnostics(result)

        print(f"\nchi2 summary ({MODE_LABELS[mode]}):")
        for result in results:
            exp = result["experiment"]
            print(
                f"{exp.label}: chi2={result['chi2']:.4f}, "
                f"logL={result['log_likelihood']:.4f}, "
                f"logPost={result['log_posterior']:.4f}"
            )

        best = min(results, key=lambda item: item["chi2"])
        print(f"Best model ({MODE_LABELS[mode]}): {best['experiment'].label}")
        plot_model_comparison(results)

    albedo_mode = "posterior_median" if "posterior_median" in all_results else ESTIMATE_MODES[0]
    plot_albedo_distribution(all_results[albedo_mode])
    print(f"\nSaved comparison figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
