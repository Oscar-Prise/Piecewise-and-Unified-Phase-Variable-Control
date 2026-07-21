"""Map gait-cycle percentage to hip torque using the TBE spline profile."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicHermiteSpline

# LG, from Supplementary Item of https://doi.org/10.1109/TNSRE.2022.3196665
DEFAULT_SPLINE_PARAMS = {
    "peak_time": 34,
    "rise_time": 18.8,
    "mid_time": 47.1,
    "mid_dur": 1.3,
    "peak_time_2": 82.2,
    "fall_time": 21.4,
    "gp_offset": 16,
    "peak_extension_torque": -0.370,
    "peak_flexion_torque": 0.221,
}


def spline_generator(
    peak_time,
    rise_time,
    mid_time,
    mid_dur,
    peak_time_2,
    fall_time,
    gp_offset,
    peak_extension_torque,
    peak_flexion_torque,
    scale_factor=0.5,
    control_Hz=100,
):
    spline_x = np.array(
        [
            peak_time - rise_time - gp_offset,
            peak_time - gp_offset,
            mid_time - mid_dur / 2 - gp_offset,
            mid_time + mid_dur / 2 - gp_offset,
            peak_time_2 - gp_offset,
            peak_time_2 + fall_time - gp_offset,
            100 + peak_time - rise_time - gp_offset,
        ]
    )
    spline_y = (
        np.array([0.0, peak_extension_torque, 0.0, 0.0, peak_flexion_torque, 0.0, 0.0])
        * scale_factor
    )
    spline_dydx = np.array([0, 0.0, 0.0, 0, 0.0, 0.0, 0])
    spline_profile = CubicHermiteSpline(
        spline_x, spline_y, spline_dydx, extrapolate="periodic"
    )

    spline_profile_arr = spline_profile(np.linspace(0, 99, control_Hz))

    return spline_profile_arr


def percent_gc_to_index(percent_gc: float, delay_percent_gc: float = 0.0) -> int:
    """Convert gait-cycle percentage [0, 100] to a spline lookup index [0, 99].

    delay_percent_gc shifts the profile along the gait cycle (x-axis). A positive
    delay samples an earlier point on the curve, so peaks occur later in the stride.
    """
    shifted_gc = (percent_gc - delay_percent_gc) % 100.0
    return int(np.clip(shifted_gc, 0, 99))


def plot_default_spline(
    body_mass_kg: float = 80.0,
    scale_factor: float = 1.0,
    delay_percent_gc: float = 0.0,
    control_freq_Hz: float = 100,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot the TBE hip torque spline, including scale and delay modifiers."""
    profile = HipTorqueProfile(
        body_mass_kg=body_mass_kg,
        control_freq_Hz=control_freq_Hz,
        scale_factor=scale_factor,
        delay_percent_gc=delay_percent_gc,
    )
    percent_gc = np.linspace(0, 99, len(profile.spline))
    torque_nm = np.array([profile.torque_from_percent_gc(p) for p in percent_gc])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(percent_gc, torque_nm, color="tab:blue")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Gait cycle (%)")
    ax.set_ylabel("Hip torque (Nm)")
    ax.set_title(
        f"TBE hip torque spline (scale={scale_factor:.2f}, delay={delay_percent_gc:.1f}%)"
    )
    ax.set_xlim(0, 99)
    ax.grid(True)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)

    if show:
        plt.show()

    return fig


class HipTorqueProfile:
    """Look up hip torque (Nm) from Vicon gait-cycle percentage."""

    def __init__(
        self,
        body_mass_kg: float,
        control_freq_Hz: float = 100,
        scale_factor: float = 1.0,
        delay_percent_gc: float = 0.0,
        spline_params: dict | None = None,
    ):
        self.body_mass_kg = body_mass_kg
        self.control_freq_Hz = control_freq_Hz
        self.scale_factor = scale_factor
        self.delay_percent_gc = delay_percent_gc

        params = DEFAULT_SPLINE_PARAMS if spline_params is None else spline_params
        self.spline = spline_generator(
            peak_time=params["peak_time"],
            rise_time=params["rise_time"],
            mid_time=params["mid_time"],
            mid_dur=params["mid_dur"],
            peak_time_2=params["peak_time_2"],
            fall_time=params["fall_time"],
            gp_offset=params["gp_offset"],
            peak_extension_torque=params["peak_extension_torque"],
            peak_flexion_torque=params["peak_flexion_torque"],
            control_Hz=int(control_freq_Hz),
        )

    def torque_from_percent_gc(self, percent_gc: float) -> float:
        """Return hip torque (Nm) for one leg at the given gait-cycle percentage."""
        idx = percent_gc_to_index(percent_gc, self.delay_percent_gc)
        return self.spline[idx] * self.body_mass_kg * self.scale_factor

    def torque_from_percent_gc_lr(
        self, percent_gc_l: float, percent_gc_r: float
    ) -> tuple[float, float]:
        """Return (cmd_L, cmd_R) hip torques (Nm) for both legs."""
        return (
            self.torque_from_percent_gc(percent_gc_l * 100),
            self.torque_from_percent_gc(percent_gc_r * 100),
        )


if __name__ == "__main__":
    import argparse
    plot_default_spline()
# cd "c:\CMU\MetaMobility Lab\Personal Code\git_clone\Vicon"
# python utils_hip_torque.py