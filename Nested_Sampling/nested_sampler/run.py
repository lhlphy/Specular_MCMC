from __future__ import annotations

import argparse
from pathlib import Path

from .runner import run_nested_model, write_comparison
from .specs import PROJECT_ROOT, build_model_specs


def parse_args():
    parser = argparse.ArgumentParser(description="Run dynesty nested sampling for K2-141b limb models.")
    parser.add_argument("--model", default="all", choices=["all", "specular_limb", "atmosphere_limb", "lambert_limb"])
    parser.add_argument("--nlive-init", type=int, default=500)
    parser.add_argument("--nlive-batch", type=int, default=250)
    parser.add_argument("--dlogz-init", type=float, default=0.1)
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument("--maxcall", type=int, default=None)
    parser.add_argument("--maxbatch", type=int, default=None)
    parser.add_argument("--sample", default="rwalk")
    parser.add_argument("--bound", default="multi")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--workers", type=int, default=1, help="Number of multiprocessing workers for dynesty.")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--max-data-points", type=int, default=None, help="Debug only: use the first N data points.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    specs_by_name = build_model_specs()
    selected = list(specs_by_name.values()) if args.model == "all" else [specs_by_name[args.model]]
    selected = [spec.with_data_limit(args.max_data_points) for spec in selected]
    args.outdir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for spec in selected:
        print(f"Running {spec.name} ({spec.ndim}D)")
        summaries.append(
            run_nested_model(
                spec,
                args.outdir,
                nlive_init=args.nlive_init,
                nlive_batch=args.nlive_batch,
                dlogz_init=args.dlogz_init,
                sample=args.sample,
                bound=args.bound,
                seed=args.seed,
                maxiter=args.maxiter,
                maxcall=args.maxcall,
                maxbatch=args.maxbatch,
                workers=args.workers,
                print_progress=not args.quiet,
            )
        )

    if len(summaries) > 1:
        rows = write_comparison(selected, summaries, args.outdir)
        print("\nEvidence ranking:")
        for row in rows:
            print(f"{row['model']}: logZ={row['logz']:.3f} +/- {row['logzerr']:.3f}, dlogZ={row['delta_logz']:.3f}")


if __name__ == "__main__":
    main()
