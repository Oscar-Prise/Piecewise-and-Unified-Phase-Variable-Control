"""Offline replay and comparison plots for phase-variable segmentation.

Examples (all options can also be combined with --save out.png):
  python utils_plot_phase.py "test_run\\AB01_1_phase_auto_scale_0.5_input_motor.csv"
  python utils_plot_phase.py "test_run\\AB01_1_phase_auto_scale_0.5_input_motor.csv" --everything
  python utils_plot_phase.py "test_run\\AB01_1_phase_auto_scale_0.5_input_motor.csv" --torque-only

Selectable signal suffixes:
  --phase, --enc, --torque, and --contact may be combined in any order.
  -L or -R selects a side; omitting both selects both sides in separate columns.
  --overlay draws selected left/right signals on the same subplot.

  python utils_plot_phase.py INPUT.csv --torque --enc -R
  python utils_plot_phase.py INPUT.csv --phase --enc --torque -L -R
  python utils_plot_phase.py INPUT.csv --phase --torque --overlay

Time-range suffixes:
  A final four-digit suffix uses two digits per endpoint: -4060 plots 40-60 s.
  For decimals or times above 99 s, use --time-range START END.

  python utils_plot_phase.py INPUT.csv --everything -4060
  python utils_plot_phase.py INPUT.csv --torque --enc -R --time-range 40.5 60.5
  python utils_plot_phase.py INPUT.csv --phase -L -4060 --save out.png
"""

from __future__ import annotations

import argparse
import re
import sys
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
    extension_deg: float = -15.0,
    flexion_deg: float = 40.0,
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
        extension_deg=extension_deg,
        flexion_deg=flexion_deg,
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


def _set_time_xlim(
    axes,
    time: np.ndarray,
    time_range: tuple[float, float] | None = None,
) -> None:
    """Set either a requested time window or the full trial span."""
    if time_range is None:
        _set_full_time_xlim(axes, time)
        return
    for ax in np.atleast_1d(axes):
        ax.set_xlim(*time_range)


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


def plot_phase_comparison(
    df: pd.DataFrame,
    save_path: str | None = None,
    show: bool = True,
    time_range: tuple[float, float] | None = None,
):
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
    _set_time_xlim(axes, time, time_range)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()
    return fig


def _mode_series(df: pd.DataFrame, side: str) -> pd.Series:
    """Use generic mode* when supplied; otherwise use the live phase_mode* log."""
    for key in (f"mode{side}", f"phase_mode{side}"):
        if key in df.columns:
            return df[key]
    return pd.Series(["pw"] * len(df))


def _plot_torque_axis(
    ax,
    torque_df: pd.DataFrame,
    side: str | None = None,
) -> np.ndarray:
    """Plot commanded and measured torque, optionally for only one side."""
    torque_time = torque_df["time"].to_numpy()
    if side is None:
        torque_columns = [
            ("mtr_cmd_L", "command L", "C0"),
            ("mtr_cmd_R", "command R", "C1"),
            ("actual_torque_L", "measured L", "C0"),
            ("actual_torque_R", "measured R", "C1"),
        ]
    else:
        torque_columns = [
            (f"mtr_cmd_{side}", "command", "C0"),
            (f"actual_torque_{side}", "measured", "C1"),
        ]
    plotted = False
    for column, label, color in torque_columns:
        if column in torque_df.columns:
            is_measured = column.startswith("actual_")
            ax.plot(
                torque_time,
                torque_df[column].to_numpy(),
                label=label,
                color=color,
                linestyle="--" if is_measured else "-",
                alpha=0.65 if is_measured else 1.0,
            )
            plotted = True

    side_label = f" {side}" if side else ""
    ax.set_ylabel(
        f"Torque{side_label} (Nm)" if plotted else f"Torque{side_label} (missing)"
    )
    if plotted:
        ax.axhline(0.0, color="0.3", linewidth=0.8)
        ax.grid(True, alpha=0.35)
        ax.legend(loc="upper right", ncol=2, fontsize=8)
    return torque_time


def plot_torque_profile(
    torque_df: pd.DataFrame,
    save_path: str | None = None,
    show: bool = True,
    time_range: tuple[float, float] | None = None,
):
    """Plot commanded and measured torque in separate left/right subplots."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    time = torque_df["time"].to_numpy()
    _plot_torque_axis(axes[0], torque_df, side="L")
    _plot_torque_axis(axes[1], torque_df, side="R")
    axes[1].set_xlabel("Time (s)")
    _set_time_xlim(axes, time, time_range)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    return fig


def plot_selected(
    df: pd.DataFrame,
    panels: list[str],
    sides: tuple[str, ...] = ("L", "R"),
    torque_df: pd.DataFrame | None = None,
    overlay: bool = False,
    save_path: str | None = None,
    show: bool = True,
    time_range: tuple[float, float] | None = None,
):
    """Plot a selectable grid of phase, encoder, torque, and contact signals."""
    if not panels:
        raise ValueError("At least one panel must be selected")
    if not sides:
        raise ValueError("At least one side must be selected")

    time = df["time"].to_numpy()
    panel_sides = (sides,) if overlay else tuple((side,) for side in sides)
    fig, axes = plt.subplots(
        len(panels),
        len(panel_sides),
        figsize=(7 * len(panel_sides), max(3.0 * len(panels), 4.0)),
        sharex=True,
        squeeze=False,
    )
    side_colors = {"L": "C0", "R": "C1"}

    for row, panel in enumerate(panels):
        for col, plotted_sides in enumerate(panel_sides):
            ax = axes[row, col]
            side_title = " + ".join(plotted_sides)

            if panel == "phase":
                if len(plotted_sides) == 1:
                    side = plotted_sides[0]
                    _shade_mode_regions(ax, time, _mode_series(df, side))
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
                            label="pw",
                        )
                    if f"phi_unified{side}" in df.columns:
                        ax.plot(
                            time,
                            df[f"phi_unified{side}"].to_numpy(),
                            color="C1",
                            alpha=0.55,
                            label="unified",
                        )
                else:
                    for side in plotted_sides:
                        color = side_colors[side]
                        ax.plot(
                            time,
                            df[f"percent_gc{side}"].to_numpy(),
                            color=color,
                            linewidth=1.5,
                            label=f"active {side}",
                        )
                        if f"phi_pw{side}" in df.columns:
                            ax.plot(
                                time,
                                df[f"phi_pw{side}"].to_numpy(),
                                color=color,
                                linestyle=":",
                                alpha=0.55,
                                label=f"pw {side}",
                            )
                        if f"phi_unified{side}" in df.columns:
                            ax.plot(
                                time,
                                df[f"phi_unified{side}"].to_numpy(),
                                color=color,
                                linestyle="--",
                                alpha=0.55,
                                label=f"unified {side}",
                            )
                ax.axhline(
                    DEFAULT_STANCE_TRANSITION,
                    color="#c0392b",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.7,
                    label=f"stance→swing ({DEFAULT_STANCE_TRANSITION})",
                )
                ax.set_ylim(-0.05, 1.05)
                ax.set_ylabel("Phase")

            elif panel == "encoder":
                for side in plotted_sides:
                    column = f"mtr_pos_{side}"
                    if column in df.columns:
                        ax.plot(
                            time,
                            df[column].to_numpy(),
                            color=side_colors[side],
                            label=f"encoder {side}",
                        )
                ax.set_ylabel("Hip encoder (deg)")

            elif panel == "torque":
                torque_data = torque_df if torque_df is not None else df
                torque_time = torque_data["time"].to_numpy()
                for side in plotted_sides:
                    color = side_colors[side]
                    command = f"mtr_cmd_{side}"
                    measured = f"actual_torque_{side}"
                    if command in torque_data.columns:
                        ax.plot(
                            torque_time,
                            torque_data[command].to_numpy(),
                            color=color,
                            label=f"command {side}",
                        )
                    if measured in torque_data.columns:
                        ax.plot(
                            torque_time,
                            torque_data[measured].to_numpy(),
                            color=color,
                            linestyle="--",
                            alpha=0.65,
                            label=f"measured {side}",
                        )
                ax.axhline(0.0, color="0.3", linewidth=0.8)
                ax.set_ylabel("Torque (Nm)")

            elif panel == "contact":
                for side in plotted_sides:
                    column = f"on_plate{side}"
                    if column in df.columns:
                        ax.step(
                            time,
                            df[column].astype(float).to_numpy(),
                            where="post",
                            color=side_colors[side],
                            label=f"contact {side}",
                        )
                ax.set_ylabel("Contact")
                ax.set_yticks([0, 1])

            ax.set_title(f"{panel.capitalize()} — {side_title}")
            ax.grid(True, alpha=0.35)
            ax.legend(loc="upper right", ncol=2, fontsize=8)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")
    _set_time_xlim(axes.ravel(), time, time_range)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    return fig


def plot_everything(
    df: pd.DataFrame,
    torque_df: pd.DataFrame | None = None,
    save_path: str | None = None,
    show: bool = True,
    time_range: tuple[float, float] | None = None,
):
    """Full diagnostic stack: phase L/R, hip encoders, torque, and contact.

    Expects a motor log DataFrame (or equivalent) with columns like those written by
    main.py: time, percent_gc*, phi_pw*, phi_unified*, mtr_pos_*, on_plate*, and
    optionally phase_mode* / mode*. Torque can be supplied separately from the
    corresponding *_output_torque.csv, or included in ``df``.
    """
    time = df["time"].to_numpy()
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

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

    torque_data = torque_df if torque_df is not None else df
    _plot_torque_axis(axes[3], torque_data)

    if "on_plateL" in df.columns and "on_plateR" in df.columns:
        axes[4].plot(time, df["on_plateL"].astype(float).to_numpy(), label="on L")
        axes[4].plot(
            time,
            df["on_plateR"].astype(float).to_numpy() + 1.1,
            label="on R (+1.1)",
        )
        axes[4].set_ylabel("Contact")
        axes[4].set_yticks([0, 1, 1.1, 2.1])
        axes[4].set_yticklabels(["0", "1", "0", "1"])
        axes[4].grid(True, alpha=0.35)
        axes[4].legend(loc="upper right")
    else:
        axes[4].set_ylabel("Contact (missing)")

    axes[4].set_xlabel("Time (s)")
    _set_time_xlim(axes, time, time_range)
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
    parser.add_argument(
        "--extension-deg",
        type=float,
        default=-15.0,
        help="Maximum hip extension angle used by PW normalization",
    )
    parser.add_argument(
        "--flexion-deg",
        type=float,
        default=40.0,
        help="Maximum hip flexion angle used by PW normalization",
    )
    parser.add_argument(
        "--mode",
        choices=["pw", "unified", "auto"],
        default="auto",
    )
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--everything",
        action="store_true",
        help="Plot phase + PW + unified + hip encoders + torque + contact",
    )
    plot_group.add_argument(
        "--torque-only",
        action="store_true",
        help="Plot only the corresponding *_output_torque.csv",
    )
    parser.add_argument("--phase", action="store_true", help="Include phase panels")
    parser.add_argument("--enc", action="store_true", help="Include hip encoder panels")
    parser.add_argument("--torque", action="store_true", help="Include torque panels")
    parser.add_argument("--contact", action="store_true", help="Include contact panels")
    parser.add_argument("-L", "--left", action="store_true", help="Plot the left side")
    parser.add_argument("-R", "--right", action="store_true", help="Plot the right side")
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Overlay selected left/right signals instead of using two columns",
    )
    parser.add_argument(
        "--time-range",
        nargs=2,
        type=float,
        metavar=("START", "END"),
        help="Only show the specified time range in seconds",
    )
    parser.add_argument("--save", default=None)

    # Shorthand suffix: "-4060" means "--time-range 40 60". This compact
    # form intentionally uses two digits for each endpoint; use --time-range
    # for decimal, negative, or three-digit times.
    argv = sys.argv[1:]
    if argv:
        compact_range = re.fullmatch(r"-(\d{2})(\d{2})", argv[-1])
        if compact_range:
            argv[-1:] = [
                "--time-range",
                compact_range.group(1),
                compact_range.group(2),
            ]
    args = parser.parse_args(argv)
    time_range = tuple(args.time_range) if args.time_range else None
    if time_range is not None and time_range[1] <= time_range[0]:
        parser.error("Time range END must be greater than START")
    selected_panels = [
        panel
        for panel, selected in (
            ("phase", args.phase),
            ("encoder", args.enc),
            ("torque", args.torque),
            ("contact", args.contact),
        )
        if selected
    ]
    if (args.left or args.right) and not selected_panels:
        selected_panels = ["phase"]
    if selected_panels and (args.everything or args.torque_only):
        parser.error(
            "Signal selections cannot be combined with --everything or --torque-only"
        )
    if args.overlay and not selected_panels:
        parser.error("--overlay requires a signal selection such as --phase or --torque")
    selected_sides = tuple(
        side
        for side, selected in (("L", args.left), ("R", args.right))
        if selected
    ) or ("L", "R")

    # Avoid blocking on plt.show() when writing a file from the CLI.
    show = args.save is None

    torque_path = Path(args.motor_csv)
    if "_output_torque.csv" not in torque_path.name:
        torque_path = torque_path.with_name(
            torque_path.name.replace("_input_motor.csv", "_output_torque.csv")
        )

    if selected_panels:
        live = pd.read_csv(args.motor_csv)
        selected_torque_df = None
        if "torque" in selected_panels:
            if not torque_path.exists():
                parser.error(f"Torque CSV not found: {torque_path}")
            selected_torque_df = pd.read_csv(torque_path)
        plot_selected(
            live,
            panels=selected_panels,
            sides=selected_sides,
            torque_df=selected_torque_df,
            overlay=args.overlay,
            save_path=args.save,
            show=show,
            time_range=time_range,
        )
    elif args.torque_only:
        if not torque_path.exists():
            parser.error(f"Torque CSV not found: {torque_path}")
        plot_torque_profile(
            pd.read_csv(torque_path),
            save_path=args.save,
            show=show,
            time_range=time_range,
        )
    elif args.everything:
        live = pd.read_csv(args.motor_csv)
        torque_df = pd.read_csv(torque_path) if torque_path.exists() else None
        plot_everything(
            live,
            torque_df=torque_df,
            save_path=args.save,
            show=show,
            time_range=time_range,
        )
    else:
        df = replay_motor_log(
            args.motor_csv,
            contact_csv=args.contact_csv,
            extension_deg=args.extension_deg,
            flexion_deg=args.flexion_deg,
            output_mode=PhaseOutputMode(args.mode),
        )
        plot_phase_comparison(
            df,
            save_path=args.save,
            show=show,
            time_range=time_range,
        )
