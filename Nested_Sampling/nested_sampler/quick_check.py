from __future__ import annotations

import numpy as np

from .specs import build_model_specs


def main():
    for spec in build_model_specs().values():
        theta, flux = spec.load_data()
        theta = theta[:5]
        flux = flux[:5]
        params = spec.prior_transform(np.full(spec.ndim, 0.5))
        model = spec.predict(theta, params)
        resid = flux - model
        logl = -0.5 * np.sum((resid / spec.sigma) ** 2 + np.log(2.0 * np.pi * spec.sigma**2))
        if model.shape != flux.shape or not np.all(np.isfinite(model)) or not np.isfinite(logl):
            raise RuntimeError(f"{spec.name} failed finite model/loglikelihood check.")
        print(f"{spec.name}: ndim={spec.ndim}, ndata={len(flux)}, logl={logl:.3f}")


if __name__ == "__main__":
    main()
