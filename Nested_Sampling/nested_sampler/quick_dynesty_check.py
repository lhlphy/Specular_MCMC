from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .priors import UniformPrior, transform_unit_cube
from .runner import run_nested_model, write_comparison
from .specs import PROJECT_ROOT


@dataclass(frozen=True)
class ToySpec:
    name: str = "toy_gaussian"
    display_name: str = "Toy Gaussian"
    target_name: str = "toy"
    color: str = "black"
    sigma: float = 0.1
    priors: tuple = (
        UniformPrior("offset", "offset", -2.0, 2.0),
        UniformPrior("slope", "slope", -2.0, 2.0),
    )

    @property
    def ndim(self):
        return len(self.priors)

    @property
    def parameter_names(self):
        return [prior.name for prior in self.priors]

    @property
    def parameter_labels(self):
        return [prior.label for prior in self.priors]

    def load_data(self):
        x = np.linspace(-1.0, 1.0, 12)
        y = 0.3 + 0.7 * x
        return x, y

    def prior_transform(self, unit_cube):
        return transform_unit_cube(unit_cube, self.priors)

    def predict(self, theta_grid, params):
        offset, slope = params
        return offset + slope * theta_grid

    def loglikelihood(self, params):
        x, y = self.load_data()
        model = self.predict(x, params)
        return float(-0.5 * np.sum(((y - model) / self.sigma) ** 2 + np.log(2.0 * np.pi * self.sigma**2)))


def main():
    outdir = PROJECT_ROOT / "results" / "dynesty_api_check"
    if outdir.exists():
        shutil.rmtree(outdir)
    spec = ToySpec()
    summary = run_nested_model(
        spec,
        outdir,
        nlive_init=40,
        nlive_batch=30,
        dlogz_init=1.0,
        seed=123,
        maxbatch=0,
        workers=2,
        print_progress=False,
    )
    write_comparison([spec], [summary], outdir)
    print(f"toy_gaussian: logZ={summary['logz']:.3f} +/- {summary['logzerr']:.3f}")


if __name__ == "__main__":
    main()
