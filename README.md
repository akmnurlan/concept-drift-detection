# Concept Drift Detection (Rolling KS-Test)

Concept drift occurs when the data distribution changes over time, causing model performance to degrade in deployed ML systems.
This repository implements a simple, research-friendly **rolling-window drift detector** using the **Kolmogorov–Smirnov (KS) test**
and demonstrates detection on synthetic streaming data with clear visualizations.

## Method (Rolling KS Window)
We maintain two sliding windows:
- **Reference window A** (older data)
- **Current window B** (newer data)

At each step we run a two-sample KS test between A and B:
- If `p-value < alpha` for `n_consecutive` checks → **drift alert**

This is intentionally minimal, interpretable, and easy to extend.

## Quickstart

```bash
pip install -e .
python examples/synthetic_shift.py
```
Output figure is saved to:

figures/drift_demo.png

Project Structure
src/
  stream.py                 # synthetic streaming generator
  detectors/ks_window.py    # rolling KS detector
  visualize.py              # plotting utilities
examples/
  synthetic_shift.py        # demo script
tests/
  test_ks_detector.py       # basic correctness tests
figures/
  drift_demo.png            # generated output
Notes

This is a univariate detector (one feature stream). For multivariate drift, you can run per-feature detection and aggregate.

The KS test is nonparametric and sensitive to distributional changes (mean/variance/shape).


---

## `src/__init__.py`

```python
# Package marker
