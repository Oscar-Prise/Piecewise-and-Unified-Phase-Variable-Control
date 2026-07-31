"""Unified phase variable — Villarreal et al. 2017, ICORR, Eq. (2)–(5).

The unified phase variable uses a phase portrait (hip angle vs. its integral)
to provide a continuous monotonic phase over the full gait cycle during
rhythmic walking.

No IMU or separate angular-velocity sensor is required. The hip integral
q̃_H is built from motor encoder angle at the control rate (100 Hz):

    q̃_H[k+1] = q̃_H[k] + q_H[k] * dt        (Euler, default)
    q̃_H[k+1] = q̃_H[k] + (q_H[k]+q_H[k-1])/2 * dt   (trapezoid)

Optionally, motor-reported velocity can drive the integral (ω * dt) for
comparison; atan2 still uses encoder angle q_H as the x-axis of the portrait.
"""

from __future__ import annotations

import math
from enum import Enum

DEFAULT_K_SCALE = 1.0
# Portrait typically wraps ~0.12 cycle before force-plate contact (AB01 walking).
DEFAULT_PHASE_OFFSET = 0.12


class HipIntegralMethod(str, Enum):
    """How to accumulate ∫q_H dt for unified phase (encoder-only by default)."""

    EULER_POSITION = "euler"       # q̃ += q_H * dt  — angle only
    TRAPEZOID_POSITION = "trapezoid"  # q̃ += (q_H + q_H_prev)/2 * dt
    MOTOR_VELOCITY = "velocity"    # q̃ += ω_motor * dt  — uses CAN velocity


def compute_k_scale(
    theta_max: float,
    theta_min: float,
    theta_int_max: float,
    theta_int_min: float,
    default_k: float = DEFAULT_K_SCALE,
) -> float:
    """Scaling factor k from the previous gait cycle (Eq. 3)."""
    theta_range = abs(theta_max - theta_min)
    int_range = abs(theta_int_max - theta_int_min)
    if int_range < 1e-9 or theta_range < 1e-9:
        return default_k
    return theta_range / int_range


def compute_unified_phase(theta: float, theta_int: float, k: float) -> float:
    """Raw unified phase in [0, 1) from the encoder phase portrait.

    Villarreal Eq. (2)/(4) places phase zero on the negative hip-angle axis.
    Our encoder convention is positive in flexion, so wrapping the raw portrait
    angle rotates that origin by half a cycle. Contact alignment is applied
    separately via ``apply_phase_offset``.
    """
    angle = math.atan2(k * theta_int, theta)
    return float((angle % math.tau) / math.tau)


def apply_phase_offset(phi_portrait: float, offset: float) -> float:
    """Shift portrait phase so contact heel strike maps near zero."""
    return float((phi_portrait - offset) % 1.0)


def compute_adjusted_hip_angle(qH_deg: float, x0: float) -> float:
    """Drift-corrected hip angle theta(t) = qH(t) - x0(TN) (Eq. 4)."""
    return qH_deg - x0


def compute_drift_increment(theta_int_at_hs: float, stride_duration: float) -> float:
    """Single-stride drift correction term sgn(integral_at_HS / T) (Eq. 5)."""
    if stride_duration <= 0:
        return 0.0
    ratio = theta_int_at_hs / stride_duration
    if ratio > 0:
        return 1.0
    if ratio < 0:
        return -1.0
    return 0.0


def integrate_hip_angle_euler(theta_int: float, theta: float, dt: float) -> float:
    """Euler integration: q̃ += q_H * dt (encoder angle only, no IMU)."""
    return theta_int + theta * dt


def integrate_hip_angle_trapezoid(
    theta_int: float, theta: float, theta_prev: float, dt: float
) -> float:
    """Trapezoidal integration from consecutive encoder samples."""
    return theta_int + 0.5 * (theta + theta_prev) * dt


def integrate_hip_angle_velocity(theta_int: float, omega_deg_s: float, dt: float) -> float:
    """Integrate using motor-reported angular velocity: q̃ += ω * dt."""
    return theta_int + omega_deg_s * dt


def integrate_hip_angle(
    theta_int: float,
    theta: float,
    dt: float,
    method: HipIntegralMethod = HipIntegralMethod.EULER_POSITION,
    theta_prev: float | None = None,
    omega_deg_s: float | None = None,
) -> float:
    """Dispatch hip-angle integration for unified phase portrait."""
    if method == HipIntegralMethod.TRAPEZOID_POSITION:
        if theta_prev is None:
            return integrate_hip_angle_euler(theta_int, theta, dt)
        return integrate_hip_angle_trapezoid(theta_int, theta, theta_prev, dt)

    if method == HipIntegralMethod.MOTOR_VELOCITY:
        if omega_deg_s is None:
            return integrate_hip_angle_euler(theta_int, theta, dt)
        return integrate_hip_angle_velocity(theta_int, omega_deg_s, dt)

    return integrate_hip_angle_euler(theta_int, theta, dt)


def reset_stride_extrema(theta: float) -> tuple[float, float, float, float]:
    """Initialize min/max trackers for theta and its integral at heel strike."""
    return theta, theta, 0.0, 0.0


def update_stride_extrema(
    theta: float,
    theta_int: float,
    theta_max: float,
    theta_min: float,
    theta_int_max: float,
    theta_int_min: float,
) -> tuple[float, float, float, float]:
    """Track running extrema within the current stride."""
    return (
        max(theta_max, theta),
        min(theta_min, theta),
        max(theta_int_max, theta_int),
        min(theta_int_min, theta_int),
    )
