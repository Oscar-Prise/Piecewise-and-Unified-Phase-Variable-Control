"""Piecewise and unified phase-variable gait segmentation (Villarreal et al. 2017).

Hip angle is read from the exoskeleton motor encoder (via utils_hip_angle).
Ground contact comes from Vicon COP until onboard FSR is available; each
contact False→True edge defines heel strike.

Output percent_gc is in [0, 1], compatible with HipTorqueProfile / t2_spline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from utils_hip_angle import HipAngleProcessor
from utils_pw_phase import DEFAULT_STANCE_TRANSITION, compute_pw_phase
from utils_rhythmic import detect_phase_crossing, detect_rhythmic_walking
from utils_unified_phase import (
    DEFAULT_K_SCALE,
    HipIntegralMethod,
    compute_adjusted_hip_angle,
    compute_drift_increment,
    compute_k_scale,
    compute_unified_phase,
    integrate_hip_angle,
    reset_stride_extrema,
    update_stride_extrema,
)


class PhaseOutputMode(str, Enum):
    PW = "pw"
    UNIFIED = "unified"
    AUTO = "auto"


@dataclass
class PhaseSideState:
    """Runtime state for one leg's phase-variable estimator."""

    hip: HipAngleProcessor = field(default_factory=HipAngleProcessor)

    # PW
    phi_pw: float = 0.0
    phi_pw_prev: float = 0.0
    stance: bool = False

    # Unified (Eq. 2–5)
    phi_unified: float = 0.0
    phi_unified_prev: float = 0.0
    theta: float = 0.0
    theta_prev: float = 0.0
    theta_int: float = 0.0
    k: float = DEFAULT_K_SCALE
    x0: float = 0.0
    stride_count: int = 0
    theta_max: float = 0.0
    theta_min: float = 0.0
    theta_int_max: float = 0.0
    theta_int_min: float = 0.0
    # Heel strike / stride timing
    time_hs: float = 0.0
    prev_time_hs: float = 0.0
    stride_durations: list[float] = field(default_factory=list)
    heel_strike_pulse: bool = False

    # Mode switching
    active_mode: PhaseOutputMode = PhaseOutputMode.PW
    rhythmic: bool = False
    pending_unified: bool = False
    unified_ready: bool = False

    # Output
    percent_gc: float = 0.0
    saturated: bool = False

    def reset(self) -> None:
        self.hip.reset()
        self.phi_pw = 0.0
        self.phi_pw_prev = 0.0
        self.stance = False
        self.phi_unified = 0.0
        self.phi_unified_prev = 0.0
        self.theta = 0.0
        self.theta_prev = 0.0
        self.theta_int = 0.0
        self.k = DEFAULT_K_SCALE
        self.x0 = 0.0
        self.stride_count = 0
        self.theta_max, self.theta_min, self.theta_int_max, self.theta_int_min = (
            reset_stride_extrema(0.0)
        )
        self.time_hs = 0.0
        self.prev_time_hs = 0.0
        self.stride_durations = []
        self.heel_strike_pulse = False
        self.active_mode = PhaseOutputMode.PW
        self.rhythmic = False
        self.pending_unified = False
        self.unified_ready = False
        self.percent_gc = 0.0
        self.saturated = False


class PhaseVariableSegmenter:
    """Bilateral PW / unified phase-variable estimator."""

    def __init__(
        self,
        extension_deg: float = -15.0,
        flexion_deg: float = 40.0,
        neutral_deg_l: float = 0.0,
        neutral_deg_r: float = 0.0,
        stance_transition_s: float = DEFAULT_STANCE_TRANSITION,
        control_freq_Hz: float = 100.0,
        hip_cutoff_hz: float = 6.0,
        rhythmic_min_strides: int = 4,
        rhythmic_cv_threshold: float = 0.20,
        rhythmic_cv_exit_threshold: float = 0.40,
        stride_duration_window: int = 4,
        output_mode: PhaseOutputMode = PhaseOutputMode.AUTO,
        hip_integral_method: HipIntegralMethod = HipIntegralMethod.TRAPEZOID_POSITION,
    ) -> None:
        self.stance_transition_s = stance_transition_s
        self.control_freq_Hz = control_freq_Hz
        self.dt = 1.0 / control_freq_Hz
        self.hip_integral_method = hip_integral_method
        self.rhythmic_min_strides = rhythmic_min_strides
        self.rhythmic_cv_threshold = rhythmic_cv_threshold
        self.rhythmic_cv_exit_threshold = max(
            rhythmic_cv_exit_threshold, rhythmic_cv_threshold
        )
        self.stride_duration_window = stride_duration_window
        self.output_mode = output_mode

        self.left = PhaseSideState(
            hip=HipAngleProcessor(
                extension_deg=extension_deg,
                flexion_deg=flexion_deg,
                neutral_deg=neutral_deg_l,
                cutoff_hz=hip_cutoff_hz,
                control_freq_Hz=control_freq_Hz,
            )
        )
        self.right = PhaseSideState(
            hip=HipAngleProcessor(
                extension_deg=extension_deg,
                flexion_deg=flexion_deg,
                neutral_deg=neutral_deg_r,
                cutoff_hz=hip_cutoff_hz,
                control_freq_Hz=control_freq_Hz,
            )
        )

    def reset(self) -> None:
        self.left.reset()
        self.right.reset()

    def both_unified_ready(self) -> bool:
        return self.left.unified_ready and self.right.unified_ready

    def _handle_heel_strike(self, side: PhaseSideState, timestamp: float) -> None:
        """Heel-strike bookkeeping: k update, drift correction, integral reset."""
        if side.time_hs > 0:
            stride_duration = timestamp - side.time_hs
            if stride_duration > 0:
                side.stride_durations.append(stride_duration)
                side.stride_durations = side.stride_durations[-self.stride_duration_window :]

                if side.stride_count > 0:
                    side.k = compute_k_scale(
                        side.theta_max,
                        side.theta_min,
                        side.theta_int_max,
                        side.theta_int_min,
                        default_k=side.k,
                    )
                    side.x0 += compute_drift_increment(side.theta_int, stride_duration)

        side.prev_time_hs = side.time_hs
        side.time_hs = timestamp
        side.stride_count += 1
        side.theta_int = 0.0
        side.unified_ready = side.stride_count >= 2

        side.theta_max, side.theta_min, side.theta_int_max, side.theta_int_min = (
            reset_stride_extrema(side.theta)
        )

    def _update_mode_switch(self, side: PhaseSideState) -> None:
        """PW default; transition to unified on cross when rhythmic (Villarreal Sec. II-A).

        Hysteresis: entering unified uses rhythmic_cv_threshold; leaving unified
        only happens if stride CV exceeds the higher rhythmic_cv_exit_threshold.
        """
        if self.output_mode == PhaseOutputMode.PW:
            side.active_mode = PhaseOutputMode.PW
            return
        if self.output_mode == PhaseOutputMode.UNIFIED:
            side.active_mode = PhaseOutputMode.UNIFIED
            return

        # Higher CV bar to drop out of unified than to enter it.
        cv_for_rhythmic = (
            self.rhythmic_cv_exit_threshold
            if side.active_mode == PhaseOutputMode.UNIFIED
            else self.rhythmic_cv_threshold
        )
        side.rhythmic = detect_rhythmic_walking(
            side.stride_durations,
            min_strides=self.rhythmic_min_strides,
            cv_threshold=cv_for_rhythmic,
        )

        if not side.rhythmic:
            side.active_mode = PhaseOutputMode.PW
            side.pending_unified = False
            return

        if side.active_mode == PhaseOutputMode.PW:
            if not side.pending_unified:
                side.pending_unified = True
            elif side.unified_ready and detect_phase_crossing(
                side.phi_pw_prev,
                side.phi_unified_prev,
                side.phi_pw,
                side.phi_unified,
            ):
                side.active_mode = PhaseOutputMode.UNIFIED
                side.pending_unified = False

    def _select_output_phase(self, side: PhaseSideState) -> float:
        if side.active_mode == PhaseOutputMode.UNIFIED and side.unified_ready:
            return side.phi_unified
        return side.phi_pw

    def update_side(
        self,
        side: PhaseSideState,
        raw_hip_encoder_deg: float,
        on_plate: bool,
        timestamp: float | None = None,
        raw_hip_velocity_deg_s: float | None = None,
    ) -> None:
        """Update one leg from motor encoder angle and ground-contact events.

        Unified phase uses encoder angle q_H for atan2. The integral q̃_H is
        accumulated from angle samples (default) or optionally from motor ω.
        """
        if timestamp is None:
            timestamp = time.time()

        qH_deg, x_pw, saturated = side.hip.update(raw_hip_encoder_deg)
        side.saturated = saturated
        was_stance = side.stance
        side.stance = bool(on_plate)

        side.phi_pw_prev = side.phi_pw
        side.phi_pw = compute_pw_phase(x_pw, side.stance, s=self.stance_transition_s)

        side.theta_prev = side.theta
        side.theta = compute_adjusted_hip_angle(qH_deg, side.x0)
        side.theta_int = integrate_hip_angle(
            side.theta_int,
            side.theta,
            self.dt,
            method=self.hip_integral_method,
            theta_prev=side.theta_prev,
            omega_deg_s=raw_hip_velocity_deg_s,
        )
        (
            side.theta_max,
            side.theta_min,
            side.theta_int_max,
            side.theta_int_min,
        ) = update_stride_extrema(
            side.theta,
            side.theta_int,
            side.theta_max,
            side.theta_min,
            side.theta_int_max,
            side.theta_int_min,
        )

        # A contact False→True edge is the authoritative heel-strike event.
        pulse = side.stance and not was_stance
        side.heel_strike_pulse = pulse
        if pulse:
            self._handle_heel_strike(side, timestamp)

        side.phi_unified_prev = side.phi_unified
        if pulse and side.unified_ready:
            # Contact defines the gait-cycle boundary even if the encoder
            # phase portrait has accumulated a small amount of drift.
            side.phi_unified = 0.0
        elif side.unified_ready:
            side.phi_unified = compute_unified_phase(side.theta, side.theta_int, side.k)
        else:
            side.phi_unified = side.phi_pw

        self._update_mode_switch(side)
        side.percent_gc = self._select_output_phase(side)

    def update_lr(
        self,
        raw_hip_encoder_l: float,
        raw_hip_encoder_r: float,
        on_plate_l: bool,
        on_plate_r: bool,
        timestamp: float | None = None,
        raw_hip_velocity_l_deg_s: float | None = None,
        raw_hip_velocity_r_deg_s: float | None = None,
    ) -> tuple[float, float]:
        """Update both legs and return (percent_gcL, percent_gcR) in [0, 1]."""
        self.update_side(
            self.left,
            raw_hip_encoder_l,
            on_plate_l,
            timestamp,
            raw_hip_velocity_deg_s=raw_hip_velocity_l_deg_s,
        )
        self.update_side(
            self.right,
            raw_hip_encoder_r,
            on_plate_r,
            timestamp,
            raw_hip_velocity_deg_s=raw_hip_velocity_r_deg_s,
        )
        return self.left.percent_gc, self.right.percent_gc
