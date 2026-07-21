"""Rhythmic walking detection and PW-to-unified transition helpers.

Villarreal et al. 2017 detect steady rhythmic locomotion from hip motion and
switch to the unified phase variable only when unified crosses PW.
"""

from __future__ import annotations

import numpy as np


def coefficient_of_variation(values: list[float]) -> float:
    """Coefficient of variation (std / mean) for a list of stride durations."""
    if len(values) < 2:
        return float("inf")
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    if mean <= 0:
        return float("inf")
    return float(np.std(arr) / mean)


def detect_rhythmic_walking(
    stride_durations: list[float],
    min_strides: int = 4,
    cv_threshold: float = 0.10,
) -> bool:
    """Return True when recent stride times are steady (low CV)."""
    if len(stride_durations) < min_strides:
        return False
    return coefficient_of_variation(stride_durations) < cv_threshold


def detect_phase_crossing(
    phi_pw_prev: float,
    phi_unified_prev: float,
    phi_pw: float,
    phi_unified: float,
) -> bool:
    """Return True when unified and PW phase variables cross between steps."""
    diff_prev = phi_unified_prev - phi_pw_prev
    diff_curr = phi_unified - phi_pw
    if diff_prev == 0.0:
        return diff_curr == 0.0
    return diff_prev * diff_curr <= 0.0
