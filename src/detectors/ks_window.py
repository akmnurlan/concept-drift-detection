from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import ks_2samp


@dataclass
class DriftEvent:
    t: int
    p_value: float
    statistic: float


class RollingKSDetector:
    """
    Rolling-window drift detector using the two-sample Kolmogorov–Smirnov test.

    Maintain a buffer of size (window_a + window_b). At each update when full:
      A = first window_a samples (reference / older)
      B = last  window_b samples (current / newer)

    Run KS(A, B). If p < alpha for n_consecutive checks => drift alert.

    Cooldown:
      After an alert, the detector enters a cooldown period during which it still
      computes p-values (for plotting) but does not emit new alerts. This avoids
      repeated alerts during the same drift episode.

    Parameters
    ----------
    window_a: int
        Size of reference window.
    window_b: int
        Size of current window.
    alpha: float
        Significance threshold.
    n_consecutive: int
        Number of consecutive p-values below alpha required to trigger drift.
    min_updates: int
        Minimum number of full-window checks before allowing alerts (warm-up).
    cooldown: int
        Number of steps to wait after an alert before allowing another alert.
    """

    def __init__(
        self,
        window_a: int = 100,
        window_b: int = 100,
        alpha: float = 0.01,
        n_consecutive: int = 3,
        min_updates: int = 1,
        cooldown: int = 200,
    ) -> None:
        if window_a <= 0 or window_b <= 0:
            raise ValueError("window_a and window_b must be positive.")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        if n_consecutive <= 0:
            raise ValueError("n_consecutive must be positive.")
        if min_updates <= 0:
            raise ValueError("min_updates must be positive.")
        if cooldown < 0:
            raise ValueError("cooldown must be >= 0.")

        self.window_a = int(window_a)
        self.window_b = int(window_b)
        self.alpha = float(alpha)
        self.n_consecutive = int(n_consecutive)
        self.min_updates = int(min_updates)
        self.cooldown = int(cooldown)

        self._buffer: List[float] = []
        self._t: int = -1
        self._checks_done: int = 0
        self._below_count: int = 0
        self._cooldown_left: int = 0

        self.p_values: List[Optional[float]] = []
        self.statistics: List[Optional[float]] = []
        self.events: List[DriftEvent] = []

    @property
    def total_window(self) -> int:
        return self.window_a + self.window_b

    def reset(self) -> None:
        self._buffer.clear()
        self._t = -1
        self._checks_done = 0
        self._below_count = 0
        self._cooldown_left = 0
        self.p_values.clear()
        self.statistics.clear()
        self.events.clear()

    def update(self, x: float) -> Tuple[bool, Optional[float]]:
        """
        Add one observation.

        Returns
        -------
        drift_detected : bool
            True if a drift event is emitted at this time step.
        p_value_if_checked : Optional[float]
            KS p-value if a full window is available, else None.
        """
        self._t += 1
        self._buffer.append(float(x))
        if len(self._buffer) > self.total_window:
            self._buffer.pop(0)

        # Default: no check yet
        self.p_values.append(None)
        self.statistics.append(None)

        if len(self._buffer) < self.total_window:
            return False, None

        # Full window available => run KS test
        a = np.asarray(self._buffer[: self.window_a], dtype=np.float64)
        b = np.asarray(self._buffer[self.window_a :], dtype=np.float64)

        res = ks_2samp(a, b, alternative="two-sided", mode="auto")
        p = float(res.pvalue)
        stat = float(res.statistic)

        self._checks_done += 1
        self.p_values[-1] = p
        self.statistics[-1] = stat

        # Update cooldown timer (still compute p-values while cooling down)
        if self._cooldown_left > 0:
            self._cooldown_left -= 1

        # Track consecutive low p-values
        if p < self.alpha:
            self._below_count += 1
        else:
            self._below_count = 0

        drift = False
        can_alert = (self._checks_done >= self.min_updates) and (self._cooldown_left == 0)

        if can_alert and (self._below_count >= self.n_consecutive):
            drift = True
            self.events.append(DriftEvent(t=self._t, p_value=p, statistic=stat))
            self._below_count = 0
            self._cooldown_left = self.cooldown

        return drift, p

    def run(self, xs: np.ndarray) -> List[DriftEvent]:
        """
        Convenience method to process a whole stream.
        """
        for x in xs:
            self.update(float(x))
        return self.events
