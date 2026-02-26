from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np


@dataclass(frozen=True)
class StreamConfig:
    n: int = 2000
    seed: int = 42

    # Phase boundaries (indices)
    stable_end: int = 700          # [0, stable_end)
    sudden_end: int = 1200         # [stable_end, sudden_end)
    drift_end: int = 2000          # [sudden_end, drift_end) -> should equal n

    # Distributions
    stable_mean: float = 0.0
    stable_std: float = 1.0

    sudden_mean: float = 2.5       # sudden shift in mean
    sudden_std: float = 1.0

    drift_mean_start: float = 2.5  # gradual drift back down
    drift_mean_end: float = 0.5
    drift_std_start: float = 1.0
    drift_std_end: float = 1.8     # gradual variance drift


def generate_stream(cfg: StreamConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      x: shape (n,), stream values
      phase: shape (n,), integer phase labels:
          0 = stable, 1 = sudden shift, 2 = gradual drift
    """
    if cfg.drift_end != cfg.n:
        raise ValueError("drift_end should equal n for simplicity.")
    if not (0 <= cfg.stable_end <= cfg.sudden_end <= cfg.drift_end):
        raise ValueError("Phase boundaries must satisfy stable_end <= sudden_end <= drift_end.")

    rng = np.random.default_rng(cfg.seed)
    x = np.zeros(cfg.n, dtype=np.float64)
    phase = np.zeros(cfg.n, dtype=np.int64)

    # Phase 0: stable
    i0, i1 = 0, cfg.stable_end
    x[i0:i1] = rng.normal(cfg.stable_mean, cfg.stable_std, size=i1 - i0)
    phase[i0:i1] = 0

    # Phase 1: sudden shift
    i0, i1 = cfg.stable_end, cfg.sudden_end
    x[i0:i1] = rng.normal(cfg.sudden_mean, cfg.sudden_std, size=i1 - i0)
    phase[i0:i1] = 1

    # Phase 2: gradual drift (mean and std interpolate)
    i0, i1 = cfg.sudden_end, cfg.drift_end
    t = np.linspace(0.0, 1.0, num=i1 - i0, endpoint=False)
    means = cfg.drift_mean_start + t * (cfg.drift_mean_end - cfg.drift_mean_start)
    stds = cfg.drift_std_start + t * (cfg.drift_std_end - cfg.drift_std_start)
    x[i0:i1] = rng.normal(means, stds)
    phase[i0:i1] = 2

    return x, phase
