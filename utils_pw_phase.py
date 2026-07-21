"""Piecewise (PW) phase variable — Villarreal et al. 2017, ICORR, Eq. (1).

The PW phase variable maps normalized hip angle to gait phase separately in
stance and swing, switched by ground-contact sensing.
"""

from __future__ import annotations

import numpy as np

DEFAULT_STANCE_TRANSITION = 0.57


def compute_pw_phase(x: float, stance: bool, s: float = DEFAULT_STANCE_TRANSITION) -> float:
    """Compute PW phase variable phi in [0, 1].

    Stance:  phi = 2 * s * (0.5 - x)
    Swing:   phi = 2 * x * (1 - s) + s

    x is the normalized hip angle after saturation and +0.25 offset
    s is the desired stance-to-swing phase transition, s = 0.57
    """
    if stance:
        phi = 2.0 * s * (0.5 - x)
    else:
        phi = 2.0 * x * (1.0 - s) + s
    return float(np.clip(phi, 0.0, 1.0))
