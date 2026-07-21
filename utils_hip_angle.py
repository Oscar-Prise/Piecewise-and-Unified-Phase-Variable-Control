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
        rom_deg: float = 40.0,
        neutral_deg: float = 0.0,
        cutoff_hz: float = 6.0,
        control_freq_Hz: float = 100.0,
    ) -> None:
        if rom_deg <= 0:
            raise ValueError("rom_deg must be positive")

        self.rom_deg = rom_deg
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
        self.x_before_offset = 0.0
        self.saturated = False
        self._initialized = False

    def reset(self, neutral_deg: float | None = None) -> None:
        if neutral_deg is not None:
            self.neutral_deg = neutral_deg
        self.filtered_deg = 0.0
        self.qH_deg = 0.0
        self.x_pw = 0.25
        self.x_before_offset = 0.0
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

        # Map hip angle to PW normalized variable x (Villarreal 2017, Fig. 4):
        # divide by 2*RoM, saturate to [-0.25, 0.25], then offset by +0.25 so x
        # starts near zero at heel strike.
        self.x_before_offset = self.qH_deg / (2.0 * self.rom_deg)
        self.saturated = self.x_before_offset > 0.25 or self.x_before_offset < -0.25
        self.x_pw = float(np.clip(self.x_before_offset, -0.25, 0.25)) + 0.25

        return self.qH_deg, self.x_pw, self.saturated
