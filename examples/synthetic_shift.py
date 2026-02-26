from __future__ import annotations

from src.stream import StreamConfig, generate_stream
from src.detectors import RollingKSDetector
from src.visualize import plot_drift


def main() -> None:
    cfg = StreamConfig(
        n=2000,
        seed=42,
        stable_end=700,
        sudden_end=1200,
        drift_end=2000,
    )

    x, phase = generate_stream(cfg)

    detector = RollingKSDetector(
        window_a=120,
        window_b=120,
        alpha=0.01,
        n_consecutive=3,
        min_updates=2,
    )

    detector.run(x)

    print(f"Detected {len(detector.events)} drift event(s).")
    for e in detector.events[:5]:
        print(f"  t={e.t}  p={e.p_value:.3e}  stat={e.statistic:.3f}")

    plot_drift(
        x=x,
        p_values=detector.p_values,
        events=detector.events,
        alpha=detector.alpha,
        out_path="figures/drift_demo.png",
    )
    print("Saved: figures/drift_demo.png")


if __name__ == "__main__":
    main()
