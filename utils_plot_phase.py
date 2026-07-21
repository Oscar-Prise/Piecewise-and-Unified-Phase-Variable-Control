"""Offline replay and comparison plots for phase-variable segmentation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils_phase_seg import PhaseOutputMode, PhaseVariableSegmenter


def replay_motor_log(
    motor_csv: str | Path,
    contact_csv: str | Path | None = None,
    rom_deg: float = 40.0,
    control_freq_Hz: float = 100.0,
    output_mode: PhaseOutputMode = PhaseOutputMode.AUTO,
) -> pd.DataFrame:
    """Replay logged motor/contact data through the phase-variable estimator."""
    motor_df = pd.read_csv(motor_csv)
    dt = 1.0 / control_freq_Hz

    if contact_csv is not None:
        contact_df = pd.read_csv(contact_csv)
        min_len = min(len(motor_df), len(contact_df))
        motor_df = motor_df.iloc[:min_len].reset_index(drop=True)
        contact_df = contact_df.iloc[:min_len].reset_index(drop=True)
    else:
        contact_df = pd.DataFrame(
            {
                "on_plateL": np.zeros(len(motor_df), dtype=bool),
                "on_plateR": np.zeros(len(motor_df), dtype=bool),
                "heel_strikeL": np.zeros(len(motor_df), dtype=bool),
                "heel_strikeR": np.zeros(len(motor_df), dtype=bool),
            }
        )

    segmenter = PhaseVariableSegmenter(
        rom_deg=rom_deg,
        control_freq_Hz=control_freq_Hz,
        output_mode=output_mode,
    )
    segmenter.reset()

    rows = []
    t0 = float(motor_df["time"].iloc[0]) if "time" in motor_df.columns else 0.0

    for i in range(len(motor_df)):
        timestamp = t0 + i * dt
        pos_l = float(motor_df["mtr_pos_L"].iloc[i])
        pos_r = float(motor_df["mtr_pos_R"].iloc[i])

        phi_l, phi_r = segmenter.update_lr(
            raw_hip_encoder_l=pos_l,
            raw_hip_encoder_r=-pos_r,
            on_plate_l=bool(contact_df["on_plateL"].iloc[i]),
            on_plate_r=bool(contact_df["on_plateR"].iloc[i]),
            heel_strike_l=bool(contact_df["heel_strikeL"].iloc[i]),
            heel_strike_r=bool(contact_df["heel_strikeR"].iloc[i]),
            timestamp=timestamp,
        )

        rows.append(
            {
                "time": timestamp - t0,
                "percent_gcL": phi_l,
                "percent_gcR": phi_r,
                "phi_pwL": segmenter.left.phi_pw,
                "phi_pwR": segmenter.right.phi_pw,
                "phi_unifiedL": segmenter.left.phi_unified,
                "phi_unifiedR": segmenter.right.phi_unified,
                "modeL": segmenter.left.active_mode.value,
                "modeR": segmenter.right.active_mode.value,
            }
        )

    return pd.DataFrame(rows)


def plot_phase_comparison(df: pd.DataFrame, save_path: str | None = None, show: bool = True):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(df["time"], df["phi_pwL"], label="PW L", alpha=0.8)
    axes[0].plot(df["time"], df["phi_unifiedL"], label="Unified L", alpha=0.8)
    axes[0].plot(df["time"], df["percent_gcL"], label="Active L", linewidth=2)
    axes[0].set_ylabel("Phase L")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(df["time"], df["phi_pwR"], label="PW R", alpha=0.8)
    axes[1].plot(df["time"], df["phi_unifiedR"], label="Unified R", alpha=0.8)
    axes[1].plot(df["time"], df["percent_gcR"], label="Active R", linewidth=2)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Phase R")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motor_csv", help="Path to *_input_motor.csv")
    parser.add_argument("--contact-csv", default=None, help="Optional contact event log CSV")
    parser.add_argument("--rom-deg", type=float, default=40.0)
    parser.add_argument(
        "--mode",
        choices=["pw", "unified", "auto"],
        default="auto",
    )
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    df = replay_motor_log(
        args.motor_csv,
        contact_csv=args.contact_csv,
        rom_deg=args.rom_deg,
        output_mode=PhaseOutputMode(args.mode),
    )
    plot_phase_comparison(df, save_path=args.save)
