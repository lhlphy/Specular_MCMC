from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import norm, truncnorm


EPS = np.finfo(float).eps


def _clip_unit(u: float) -> float:
    return float(np.clip(u, EPS, 1.0 - EPS))


@dataclass(frozen=True)
class Prior:
    name: str
    label: str

    def transform(self, u: float) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class UniformPrior(Prior):
    low: float
    high: float

    def transform(self, u: float) -> float:
        return self.low + _clip_unit(u) * (self.high - self.low)


@dataclass(frozen=True)
class NormalPrior(Prior):
    mu: float
    sigma: float

    def transform(self, u: float) -> float:
        return float(norm.ppf(_clip_unit(u), loc=self.mu, scale=self.sigma))


@dataclass(frozen=True)
class TruncatedNormalPrior(Prior):
    mu: float
    sigma: float
    low: Optional[float] = None
    high: Optional[float] = None

    def transform(self, u: float) -> float:
        low = -np.inf if self.low is None else self.low
        high = np.inf if self.high is None else self.high
        a = (low - self.mu) / self.sigma
        b = (high - self.mu) / self.sigma
        return float(truncnorm.ppf(_clip_unit(u), a, b, loc=self.mu, scale=self.sigma))


def transform_unit_cube(unit_cube, priors):
    unit_cube = np.asarray(unit_cube, dtype=float)
    if unit_cube.shape[0] != len(priors):
        raise ValueError(f"Expected {len(priors)} unit-cube values, got {unit_cube.shape[0]}.")
    return np.array([prior.transform(value) for prior, value in zip(priors, unit_cube)], dtype=float)

