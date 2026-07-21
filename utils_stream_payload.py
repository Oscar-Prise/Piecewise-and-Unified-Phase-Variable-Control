"""Build TCP JSON payloads for Vicon-Computer → Jetson phase-variable streaming.

Add these fields to the treadmill_data dict in Vicon_side.py alongside percent_gcL/R.
"""

from __future__ import annotations

import json


def build_phase_stream_payload(
    vicon_timestamp: float,
    percent_gc_l: float,
    percent_gc_r: float,
    on_plate_l: bool,
    on_plate_r: bool,
    heel_strike_l: bool,
    heel_strike_r: bool,
) -> str:
    """Return newline-ready JSON for StreamJetson.send_data()."""
    payload = {
        "vicon_timestamp": vicon_timestamp,
        "percent_gcL": percent_gc_l,
        "percent_gcR": percent_gc_r,
        "on_plateL": on_plate_l,
        "on_plateR": on_plate_r,
        "heel_strikeL": heel_strike_l,
        "heel_strikeR": heel_strike_r,
    }
    return json.dumps(payload)
