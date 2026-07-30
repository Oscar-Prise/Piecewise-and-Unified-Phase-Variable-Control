"""Motor-encoder hip angle preprocessing for phase-variable gait segmentation.

Villarreal et al. 2017 use a global hip angle from an IMU. Here the hip angle
is approximated by the exoskeleton motor encoder (degrees, zero-calibrated at
startup). Optional low-pass filtering reduces CAN noise before phase mapping.
"""

from __future__ import annotations

import numpy as np


class HipAngleProcessor:
    """Stateful per-leg hip angle filter and PW normalization."""

    def __init__(
        self,
        extension_deg: float = -15.0,
        flexion_deg: float = 40.0,
        neutral_deg: float = 0.0,
        cutoff_hz: float = 6.0,
        control_freq_Hz: float = 100.0,
    ) -> None:
        if flexion_deg <= extension_deg:
            raise ValueError("flexion_deg must be greater than extension_deg")

        self.extension_deg = extension_deg
        self.flexion_deg = flexion_deg
        self.rom_deg = flexion_deg - extension_deg
        self.neutral_deg = neutral_deg

        # First-order low-pass alpha = dt / (RC + dt); alpha = 1 disables filtering.
        if cutoff_hz <= 0 or control_freq_Hz <= 0:
            self.alpha = 1.0
        else:
            dt = 1.0 / control_freq_Hz
            rc = 1.0 / (2.0 * np.pi * cutoff_hz)
            self.alpha = dt / (rc + dt)

        self.filtered_deg = 0.0
        self.qH_deg = 0.0
        self.x_pw = 0.25
        self.x_unclipped = 0.0
        self.saturated = False
        self._initialized = False

    def reset(self, neutral_deg: float | None = None) -> None:
        if neutral_deg is not None:
            self.neutral_deg = neutral_deg
        self.filtered_deg = 0.0
        self.qH_deg = 0.0
        self.x_pw = 0.25
        self.x_unclipped = 0.0
        self.saturated = False
        self._initialized = False

    def update(self, raw_encoder_deg: float) -> tuple[float, float, bool]:
        """Filter encoder reading and return (qH_deg, x_pw, saturated)."""
        # Express encoder reading relative to the neutral standing angle.
        qH = raw_encoder_deg - self.neutral_deg

        # Single-step first-order low-pass; seed with the first sample.
        if not self._initialized:
            self.filtered_deg = qH
            self._initialized = True
        else:
            self.filtered_deg = self.alpha * qH + (1.0 - self.alpha) * self.filtered_deg

        self.qH_deg = self.filtered_deg

        # Map the anatomical range asymmetrically onto PW x in [0, 0.5]:
        # maximum extension -> 0, maximum flexion -> 0.5.
        self.x_unclipped = 0.5 * (
            (self.qH_deg - self.extension_deg) / self.rom_deg
        )
        self.saturated = (
            self.qH_deg < self.extension_deg or self.qH_deg > self.flexion_deg
        )
        self.x_pw = float(np.clip(self.x_unclipped, 0.0, 0.5))

        return self.qH_deg, self.x_pw, self.saturated
