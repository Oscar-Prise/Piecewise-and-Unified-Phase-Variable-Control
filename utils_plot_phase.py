"""Offline replay and comparison plots for phase-variable segmentation.

Examples:
  python utils_plot_phase.py "test_run\\AB01_1_phase_auto_scale_0.5_input_motor.csv"
  python utils_plot_phase.py "test_run\\AB01_1_phase_auto_scale_0.5_input_motor.csv" --everything
  python utils_plot_phase.py "test_run\\AB01_1_phase_auto_scale_0.5_input_motor.csv" --everything --save out.png
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


def _set_full_time_xlim(axes, time: np.ndarray) -> None:
    """Force x-axis to the full trial span (no default padding / clipping)."""
    if len(time) == 0:
        return
    t0 = float(np.min(time))
    t1 = float(np.max(time))
    if t1 <= t0:
        t1 = t0 + 1e-3
    for ax in np.atleast_1d(axes):
        ax.set_xlim(t0, t1)


def _shade_mode_regions(ax, time: np.ndarray, mode: pd.Series) -> None:
    """Background bands: PW = light blue (matches pw line), Unified = light orange (matches unified line)."""
    colors = {"pw": "#cfe2f3", "unified": "#fce4c7"}
    labels_used = set()
    modes = mode.astype(str).str.lower().to_numpy()
    if len(modes) == 0:
        return

    start = 0
    for i in range(1, len(modes) + 1):
        if i < len(modes) and modes[i] == modes[start]:
            continue
        m = modes[start]
        t0 = float(time[start])
        t1 = float(time[min(i, len(time) - 1)])
        # Avoid zero-width spans when a mode lasts only one sample at the end.
        if t1 <= t0 and i < len(time):
            t1 = float(time[i]) if i < len(time) else t0
        label = None
        if m not in labels_used:
            label = f"Mode: {m}"
            labels_used.add(m)
        ax.axvspan(
            t0,
            t1,
            facecolor=colors.get(m, "#eeeeee"),
            alpha=0.9,
            lw=0,
            zorder=0,
            label=label,
        )
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
    _set_full_time_xlim(axes, time)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()
    return fig


def _mode_series(df: pd.DataFrame, side: str) -> pd.Series:
    """Prefer replay mode* (current auto rules); fall back to live phase_mode*."""
    for key in (f"mode{side}", f"phase_mode{side}"):
        if key in df.columns:
            return df[key]
    return pd.Series(["pw"] * len(df))


def plot_everything(
    df: pd.DataFrame,
    save_path: str | None = None,
    show: bool = True,
):
    """Full diagnostic stack: phase L/R (active + PW + unified), hip encoders, contact.

    Expects a motor log DataFrame (or equivalent) with columns like those written by
    main.py: time, percent_gc*, phi_pw*, phi_unified*, mtr_pos_*, on_plate*, and
    optionally phase_mode* / mode*.
    """
    time = df["time"].to_numpy()
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    for ax, side in zip(axes[:2], ("L", "R")):
        mode = _mode_series(df, side)
        _shade_mode_regions(ax, time, mode)
        ax.plot(
            time,
            df[f"percent_gc{side}"].to_numpy(),
            color="#1a1a1a",
            linewidth=1.4,
            label="active",
        )
        if f"phi_pw{side}" in df.columns:
            ax.plot(
                time,
                df[f"phi_pw{side}"].to_numpy(),
                color="C0",
                alpha=0.55,
                linewidth=1.0,
                label="pw",
            )
        if f"phi_unified{side}" in df.columns:
            ax.plot(
                time,
                df[f"phi_unified{side}"].to_numpy(),
                color="C1",
                alpha=0.55,
                linewidth=1.0,
                label="unified",
            )
        ax.axhline(
            DEFAULT_STANCE_TRANSITION,
            color="#c0392b",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label=f"PW stance→swing ({DEFAULT_STANCE_TRANSITION})",
        )
        ax.set_ylabel(f"Phase {side}")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.35)
        ax.legend(loc="upper right", ncol=2, fontsize=8)

    if "mtr_pos_L" in df.columns and "mtr_pos_R" in df.columns:
        axes[2].plot(time, df["mtr_pos_L"].to_numpy(), label="pos L")
        axes[2].plot(time, df["mtr_pos_R"].to_numpy(), label="pos R")
        axes[2].set_ylabel("Hip enc")
        axes[2].grid(True, alpha=0.35)
        axes[2].legend(loc="upper right")
    else:
        axes[2].set_ylabel("Hip enc (missing)")

    if "on_plateL" in df.columns and "on_plateR" in df.columns:
        axes[3].plot(time, df["on_plateL"].astype(float).to_numpy(), label="on L")
        axes[3].plot(
            time,
            df["on_plateR"].astype(float).to_numpy() + 1.1,
            label="on R (+1.1)",
        )
        axes[3].set_ylabel("Contact")
        axes[3].set_yticks([0, 1, 1.1, 2.1])
        axes[3].set_yticklabels(["0", "1", "0", "1"])
        axes[3].grid(True, alpha=0.35)
        axes[3].legend(loc="upper right")
    else:
        axes[3].set_ylabel("Contact (missing)")

    axes[3].set_xlabel("Time (s)")
    _set_full_time_xlim(axes, time)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
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
    parser.add_argument(
        "--everything",
        action="store_true",
        help="Plot live-log phase + PW + unified + hip encoders + contact",
    )
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    # Avoid blocking on plt.show() when writing a file from the CLI.
    show = args.save is None

    if args.everything:
        live = pd.read_csv(args.motor_csv)
        # Live phase_mode* may be all-pw on older trials; replay with current auto
        # rules so Mode shading (blue PW / orange unified) reflects the switch.
        replayed = replay_motor_log(
            args.motor_csv,
            contact_csv=args.contact_csv,
            rom_deg=args.rom_deg,
            output_mode=PhaseOutputMode(args.mode),
        )
        live = live.copy()
        live["modeL"] = replayed["modeL"].to_numpy()
        live["modeR"] = replayed["modeR"].to_numpy()
        # Prefer replayed phase traces so active matches the shaded mode bands.
        for col in (
            "percent_gcL",
            "percent_gcR",
            "phi_pwL",
            "phi_pwR",
            "phi_unifiedL",
            "phi_unifiedR",
        ):
            if col in replayed.columns:
                live[col] = replayed[col].to_numpy()
        plot_everything(live, save_path=args.save, show=show)
    else:
        df = replay_motor_log(
            args.motor_csv,
            contact_csv=args.contact_csv,
            rom_deg=args.rom_deg,
            output_mode=PhaseOutputMode(args.mode),
        )
        plot_phase_comparison(df, save_path=args.save, show=show)
