from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt

from src.detectors.ks_window import DriftEvent


def plot_drift(
    x: np.ndarray,
    p_values: List[Optional[float]],
    events: List[DriftEvent],
    alpha: float,
    out_path: str = "figures/drift_demo.png",
    title: str = "Concept Drift Detection (Rolling KS-Test)",
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    t = np.arange(len(x))

    # Convert p_values (with None) to numeric arrays
    p = np.array([np.nan if v is None else float(v) for v in p_values], dtype=np.float64)

    fig = plt.figure(figsize=(11, 7))

    # Top: stream values + drift points
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(t, x, linewidth=1.0)
    ax1.set_title(title)
    ax1.set_ylabel("stream value")

    if events:
        ev_t = [e.t for e in events]
        ev_y = [x[e.t] for e in events]
        ax1.scatter(ev_t, ev_y, marker="x")
        for et in ev_t:
            ax1.axvline(et, linestyle="--", linewidth=1.0)

    # Bottom: p-values
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(t, p, linewidth=1.0)
    ax2.axhline(alpha, linestyle="--", linewidth=1.0)
    ax2.set_ylabel("KS p-value")
    ax2.set_xlabel("time")
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close(fig)
