from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np

from parameters import PPs

from .priors import NormalPrior, Prior, TruncatedNormalPrior, UniformPrior, transform_unit_cube


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "targets"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    target_name: str
    model_module: str
    color: str
    priors: tuple[Prior, ...]
    sigma: float = 7.05
    file_name: str = "Kepler"
    max_data_points: int | None = None

    @property
    def ndim(self) -> int:
        return len(self.priors)

    @property
    def parameter_names(self) -> list[str]:
        return [prior.name for prior in self.priors]

    @property
    def parameter_labels(self) -> list[str]:
        return [prior.label for prior in self.priors]

    @property
    def target_dir(self) -> Path:
        return DATA_ROOT / self.target_name

    @property
    def data_path(self) -> Path:
        return self.target_dir / f"{self.file_name}.txt"

    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        data = np.loadtxt(self.data_path, delimiter=",")
        if self.max_data_points is not None:
            data = data[: self.max_data_points]
        theta = data[:, 0] * 2.0 * np.pi
        theta = np.where(theta < 0.0, theta + 2.0 * np.pi, theta)
        return theta, data[:, 1]

    def with_data_limit(self, max_data_points: int | None):
        return replace(self, max_data_points=max_data_points)

    def prior_transform(self, unit_cube) -> np.ndarray:
        return transform_unit_cube(unit_cube, self.priors)

    def model_function(self) -> Callable:
        return importlib.import_module(self.model_module).Fp2Fs

    def predict(self, theta_grid, params) -> np.ndarray:
        os.environ["FOLDER_PATH"] = str(self.target_dir)
        return self.model_function()(theta_grid, co1=PPs.Coefficents[0], co2=PPs.Coefficents[1], params=params)

    def loglikelihood(self, params) -> float:
        theta, flux = self.load_data()
        model = self.predict(theta, params)
        resid = flux - model
        return float(-0.5 * np.sum((resid / self.sigma) ** 2 + np.log(2.0 * np.pi * self.sigma**2)))


def _common_limb_priors(first: Prior) -> tuple[Prior, ...]:
    return (
        first,
        TruncatedNormalPrior("T_sub", r"$T_{\rm sub}$", PPs.Tss, 64.2 * 2.0, None, PPs.Tss * 1.2),
        NormalPrior("Rp2Rs", "Rp/Rs", PPs.Rp2Rs, 0.02258 * PPs.Rp2Rs),
        TruncatedNormalPrior("F", "F", 0.156, 0.120, 0.0, 0.5),
        TruncatedNormalPrior("inc", "inc", 86.3, 3.1, 75.0, 90.0),
        NormalPrior("alpha", r"$\alpha$", PPs.alpha, 0.02636 * PPs.alpha),
        TruncatedNormalPrior("u1", "u1", PPs.Coefficents[0], 0.1, 0.0, None),
        TruncatedNormalPrior("u2", "u2", PPs.Coefficents[1], 0.1, 0.0, None),
        UniformPrior("delta", "delta", -10.0, 10.0),
    )


def build_model_specs() -> dict[str, ModelSpec]:
    specular = ModelSpec(
        name="specular_limb",
        display_name="Specular limb",
        target_name="K2-141b_limb",
        model_module="models.specular_limb",
        color="crimson",
        priors=_common_limb_priors(UniformPrior("A_lambda", r"$A_{\lambda}$", 0.0, 0.7)),
    )
    atmosphere = ModelSpec(
        name="atmosphere_limb",
        display_name="Atmosphere limb",
        target_name="K2-141b_atm_limb",
        model_module="models.atmosphere_limb",
        color="darkorange",
        priors=(
            UniformPrior("omega", r"$\omega$", 0.0, 1.0),
            UniformPrior("g", "g", -0.999, 0.999),
            TruncatedNormalPrior("T_sub", r"$T_{\rm sub}$", PPs.Tss, 64.2 * 2.0, None, PPs.Tss * 1.2),
            NormalPrior("Rp2Rs", "Rp/Rs", PPs.Rp2Rs, 0.02258 * PPs.Rp2Rs),
            TruncatedNormalPrior("F", "F", 0.156, 0.120, 0.0, 0.5),
            TruncatedNormalPrior("inc", "inc", 86.3, 3.1, 75.0, 90.0),
            NormalPrior("alpha", r"$\alpha$", PPs.alpha, 0.02636 * PPs.alpha),
            TruncatedNormalPrior("u1", "u1", PPs.Coefficents[0], 0.1, 0.0, None),
            TruncatedNormalPrior("u2", "u2", PPs.Coefficents[1], 0.1, 0.0, None),
            UniformPrior("delta", "delta", -10.0, 10.0),
        ),
    )
    lambert = ModelSpec(
        name="lambert_limb",
        display_name="Lambert limb",
        target_name="K2-141b_lambert_limb",
        model_module="models.lambert_limb",
        color="royalblue",
        priors=_common_limb_priors(UniformPrior("A_lambda", r"$A_{\lambda}$", 0.0, 0.7)),
    )
    return {spec.name: spec for spec in (specular, atmosphere, lambert)}
