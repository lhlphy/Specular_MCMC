from __future__ import annotations

import csv
import json
from multiprocessing import get_context
from pathlib import Path
from contextlib import nullcontext

import numpy as np

from .plotting import comparison_plot, corner_plot, posterior_predictive


def _write_posterior_csv(path: Path, parameter_names: list[str], samples: np.ndarray, max_rows: int = 20000):
    rows = samples[:max_rows]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(parameter_names)
        writer.writerows(rows)


def _equal_weight_samples(results, seed: int | None, max_samples: int = 20000) -> np.ndarray:
    from dynesty.utils import resample_equal

    weights = np.exp(results.logwt - results.logz[-1])
    samples = resample_equal(results.samples, weights, rstate=np.random.default_rng(seed))
    if len(samples) > max_samples:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(samples), size=max_samples, replace=False)
        samples = samples[indices]
    return np.asarray(samples)


def run_nested_model(
    spec,
    results_root: Path,
    nlive_init: int = 500,
    nlive_batch: int = 250,
    dlogz_init: float = 0.1,
    sample: str = "rwalk",
    bound: str = "multi",
    seed: int | None = None,
    maxiter: int | None = None,
    maxcall: int | None = None,
    maxbatch: int | None = None,
    workers: int = 1,
    print_progress: bool = True,
):
    from dynesty import DynamicNestedSampler

    outdir = results_root / spec.name
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    workers = max(1, int(workers))
    pool_context = (
        get_context("spawn").Pool(processes=workers)
        if workers > 1
        else nullcontext(None)
    )

    with pool_context as pool:
        sampler = DynamicNestedSampler(
            spec.loglikelihood,
            spec.prior_transform,
            spec.ndim,
            bound=bound,
            sample=sample,
            rstate=rng,
            pool=pool,
            queue_size=workers if workers > 1 else None,
        )
        sampler.run_nested(
            nlive_init=nlive_init,
            nlive_batch=nlive_batch,
            dlogz_init=dlogz_init,
            maxiter=maxiter,
            maxcall=maxcall,
            maxbatch=maxbatch,
            print_progress=print_progress,
        )
    results = sampler.results
    posterior_samples = _equal_weight_samples(results, seed=seed)

    np.savez(
        outdir / "nested_results.npz",
        samples=results.samples,
        logl=results.logl,
        logwt=results.logwt,
        logz=results.logz,
        logzerr=results.logzerr,
        posterior_samples=posterior_samples,
        parameter_names=np.array(spec.parameter_names),
    )
    _write_posterior_csv(outdir / "posterior_samples.csv", spec.parameter_names, posterior_samples)
    predictive = posterior_predictive(spec, posterior_samples, outdir, seed=seed)
    corner_plot(spec, posterior_samples, outdir)

    summary = {
        "model": spec.name,
        "display_name": spec.display_name,
        "target_name": spec.target_name,
        "ndim": spec.ndim,
        "sigma": spec.sigma,
        "nlive_init": nlive_init,
        "nlive_batch": nlive_batch,
        "dlogz_init": dlogz_init,
        "sample": sample,
        "bound": bound,
        "maxiter": maxiter,
        "maxcall": maxcall,
        "maxbatch": maxbatch,
        "workers": workers,
        "logz": float(results.logz[-1]),
        "logzerr": float(results.logzerr[-1]),
        "niter": int(results.niter),
        "ncall": int(np.sum(results.ncall)),
        "posterior_samples": int(len(posterior_samples)),
        **predictive,
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def write_comparison(specs, summaries: list[dict], results_root: Path):
    comparison_dir = results_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    best_logz = max(summary["logz"] for summary in summaries)
    by_name = {}
    rows = []
    for summary in summaries:
        row = dict(summary)
        row["delta_logz"] = row["logz"] - best_logz
        rows.append(row)
        by_name[row["model"]] = row
    rows.sort(key=lambda item: item["logz"], reverse=True)

    fields = [
        "model",
        "display_name",
        "ndim",
        "logz",
        "logzerr",
        "delta_logz",
        "predictive_chi2_median",
        "predictive_chi2_16",
        "predictive_chi2_84",
        "coverage_68",
        "coverage_95",
    ]
    for filename in ("evidence_summary.csv", "delta_logz.csv"):
        with (comparison_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    comparison_plot(specs, by_name, results_root)
    return rows
