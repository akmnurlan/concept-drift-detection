from __future__ import annotations

import numpy as np

from src.detectors import RollingKSDetector


def test_no_drift_on_stationary_stream():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=1500)

    det = RollingKSDetector(window_a=80, window_b=80, alpha=1e-4, n_consecutive=3)
    det.run(x)

    # Very strict alpha; should rarely/never fire on stationary data
    assert len(det.events) == 0


def test_detects_clear_mean_shift():
    rng = np.random.default_rng(1)
    x1 = rng.normal(0.0, 1.0, size=800)
    x2 = rng.normal(3.0, 1.0, size=800)
    x = np.concatenate([x1, x2])

    det = RollingKSDetector(window_a=60, window_b=60, alpha=0.01, n_consecutive=2)
    det.run(x)

    assert len(det.events) >= 1
    # Drift should occur after shift, not before windows are full
    assert det.events[0].t >= det.total_window - 1
