"""Ground-contact helpers for phase-variable gait segmentation.

Villarreal et al. 2017 use an FSR on the prosthetic foot for stance/swing
detection and heel-strike resets. In the Vicon protocol, equivalent events
come from force-plate COP (on_plate, heel_strike) streamed by Vicon-Computer.
"""

from __future__ import annotations


def is_stance(on_plate: bool) -> bool:
    """Return True when the foot is in contact with the ground (stance)."""
    return bool(on_plate)


def is_swing(on_plate: bool) -> bool:
    """Return True when the foot is in swing (no ground contact)."""
    return not bool(on_plate)


def consume_heel_strike_pulse(heel_strike_flag: bool, latched: bool) -> tuple[bool, bool]:
    """Convert a streaming heel-strike flag into a one-shot pulse.

    Returns (pulse_this_step, new_latched_state).
    """
    if heel_strike_flag and not latched:
        return True, True
    if not heel_strike_flag:
        return False, False
    return False, latched
