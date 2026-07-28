"""Offline replay and comparison plots for phase-variable segmentation.

To actually plot something, write something like this in the terminal:
python utils_plot_phase.py "test_run/AB01_1_phase_auto_scale_0.5_input_motor.csv"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils_phase_seg import PhaseOutputMode, PhaseVariableSegmenter
from utils_pw_phase import DEFAULT_STANCE_TRANSITION


def _rising_edges(flags: pd.Series) -> np.ndarray:
    """True on False→True transitions (synthetic heel-strike pulses)."""
    values = flags.astype(bool).to_numpy()
    prev = np.concatenate(([False], values[:-1]))
    return (~prev) & values


def _contact_from_motor_log(motor_df: pd.DataFrame) -> pd.DataFrame:
    """Build contact flags from columns already stored in *_input_motor.csv."""
    n = len(motor_df)
    on_l = (
        motor_df["on_plateL"].astype(bool)
        if "on_plateL" in motor_df.columns
        else pd.Series(np.zeros(n, dtype=bool))
    )
    on_r = (
        motor_df["on_plateR"].astype(bool)
        if "on_plateR" in motor_df.columns
        else pd.Series(np.zeros(n, dtype=bool))
    )

    if "heel_strikeL" in motor_df.columns:
        hs_l = motor_df["heel_strikeL"].astype(bool).to_numpy()
    else:
        hs_l = _rising_edges(on_l)

    if "heel_strikeR" in motor_df.columns:
        hs_r = motor_df["heel_strikeR"].astype(bool).to_numpy()
    else:
        hs_r = _rising_edges(on_r)

    return pd.DataFrame(
        {
            "on_plateL": on_l.to_numpy(),
            "on_plateR": on_r.to_numpy(),
            "heel_strikeL": hs_l,
            "heel_strikeR": hs_r,
        }
    )


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
        # Prefer contact already logged in the motor CSV (on_plate*). Otherwise
        # stance stays False forever and PW only uses the swing branch.
        contact_df = _contact_from_motor_log(motor_df)

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
        # mtr_pos_* were saved in the same sign convention already passed to
        # PhaseVariableSegmenter during the live run — do not re-negate.
        pos_l = float(motor_df["mtr_pos_L"].iloc[i])
        pos_r = float(motor_df["mtr_pos_R"].iloc[i])

        phi_l, phi_r = segmenter.update_lr(
            raw_hip_encoder_l=pos_l,
            raw_hip_encoder_r=pos_r,
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


def _shade_mode_regions(ax, time: np.ndarray, mode: pd.Series) -> None:
    """Background bands: PW = cool gray-blue, Unified = warm sand."""
    colors = {"pw": "#d9e8f5", "unified": "#f5e6d3"}
    labels_used = set()
    modes = mode.astype(str).to_numpy()
    if len(modes) == 0:
        return

    start = 0
    for i in range(1, len(modes) + 1):
        if i < len(modes) and modes[i] == modes[start]:
            continue
        m = modes[start]
        t0 = float(time[start])
        t1 = float(time[min(i, len(time) - 1)])
        label = None
        if m not in labels_used:
            label = f"Mode: {m}"
            labels_used.add(m)
        ax.axvspan(t0, t1, facecolor=colors.get(m, "#eeeeee"), alpha=0.85, lw=0, label=label)
        start = i


def _plot_side(
    ax,
    time: np.ndarray,
    active: np.ndarray,
    mode: pd.Series,
    ylabel: str,
) -> None:
    """Plot only the commanded/active phase; shade by which estimator is selected."""
    _shade_mode_regions(ax, time, mode)
    ax.plot(time, active, color="#1a1a1a", linewidth=2.0, label="Active phase")
    # PW stance→swing boundary (only meaningful while mode is pw).
    ax.axhline(
        DEFAULT_STANCE_TRANSITION,
        color="#c0392b",
        linestyle="--",
        linewidth=1.2,
        alpha=0.85,
        label=f"PW stance→swing ({DEFAULT_STANCE_TRANSITION})",
    )
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right")


def plot_phase_comparison(df: pd.DataFrame, save_path: str | None = None, show: bool = True):
    """Plot the active phase only, with background color showing PW vs unified mode.

    Both estimators always run in parallel internally; only Active is used for
    torque. Overlaying PW + unified + Active made the switch hard to see.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)
    time = df["time"].to_numpy()

    mode_l = df["modeL"] if "modeL" in df.columns else pd.Series(["pw"] * len(df))
    mode_r = df["modeR"] if "modeR" in df.columns else pd.Series(["pw"] * len(df))

    _plot_side(axes[0], time, df["percent_gcL"].to_numpy(), mode_l, "Phase L")
    _plot_side(axes[1], time, df["percent_gcR"].to_numpy(), mode_r, "Phase R")
    axes[1].set_xlabel("Time (s)")

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
