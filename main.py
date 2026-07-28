# Phase-variable exoskeleton protocol (Villarreal et al. 2017 PW / unified)
import time
import signal
import os
import atexit
import gc
import threading
import csv

import numpy as np
import pandas as pd

from Header_Mocap_trigger_protocolTest import Mocap_trigger
from utils_motors import RobStrideMotorGroup
from utils_gpio import GpioPulse, SyncPulse
from utils_teleplot import Teleplot
from t2_spline import HipTorqueProfile
from utils_phase_seg import PhaseOutputMode, PhaseVariableSegmenter
from utils_unified_phase import HipIntegralMethod

# =============================================================================
# Configuration — edit before each trial
# =============================================================================

# Trial
subject = "AB01"
trial_start_sec = 1
target_duration_sec = 31
target_time_range = 31
exo_ON = True
scale_factor = 0.5
delay_factor = 0
body_mass_kg = 70

# Phase-variable method: "pw", "unified", or "auto" (Villarreal Sec. II-A)
phase_method = "auto"

# Hip angle from motor encoder (replaces IMU global hip angle)
hip_rom_deg = 40.0
hip_neutral_deg_l = 0.0
hip_neutral_deg_r = 0.0
hip_cutoff_hz = 6.0
stance_transition_s = 0.57
# Unified ∫q_H dt: "trapezoid" (encoder angle, default), "euler", or "velocity" (CAN ω)
hip_integral_method = "trapezoid"

# Rhythmic detection for auto PW → unified transition
# Looser than Villarreal defaults so contact noise / extra HS don't block unified as long.
rhythmic_min_strides = 4
rhythmic_cv_threshold = 0.20
stride_duration_window = 4

# Trigger: "mocap" or "typing"
trigger_type = "mocap"

# Output paths
OUTPUT_DIR = "test_run"
mocap_log_csv = "output_phase.csv"

# GPIO sync pulses
gpio_pin = 7
gpio_first_pulse_sec = 2.0
gpio_pulse_duration_sec = 0.05

# RobStride motors
can_id_L = 1
can_id_R = 2
motor_channel = "can0"
torque_limit = 17.0
offset_samples = 50
control_freq_Hz = 100
frame_length = 95

# Teleplot
teleplot_host = "127.0.0.1"
teleplot_port = 47269

# Vicon / mocap server (Vicon-Computer must stream on_plate* and heel_strike*)
mocap_server_ip = "172.24.44.177"
mocap_port = 11

# =============================================================================
# Runtime state
# =============================================================================
trial_num = None
trial_name = None

# data to be saved (changed to lists for efficient appending)
data_to_save = {
    "timestamp": [],
    "mtr_cmd_L": [], "mtr_cmd_R": [],
    "mtr_pos_L": [], "mtr_pos_R": [],
    "mtr_vel_L": [], "mtr_vel_R": [],
    "actual_torque_L": [], "actual_torque_R": [],
    # phase-variable logging (Villarreal)
    "percent_gcL": [], "percent_gcR": [],
    "phi_pwL": [], "phi_pwR": [],
    "phi_unifiedL": [], "phi_unifiedR": [],
    "phase_modeL": [], "phase_modeR": [],
    "on_plateL": [], "on_plateR": [],
    "hip_satL": [], "hip_satR": [],
    "gpio_output": [],  # GPIO
}

# Global variables
mocap_trigger = None  # Will be initialized in __main__
gpio_pulse = None
motors = None
teleplot = None
phase_segmenter = None


# Function to save all collected data
def save_data(start_rec_sec=0, trial_time_sec=None):
    global data_to_save

    # Convert lists to NumPy arrays
    data_np = {k: np.array(v) if isinstance(v, list) else v for k, v in data_to_save.items()}

    # Determine minimum length
    min_len = min(len(data_np["timestamp"]), len(data_np["mtr_pos_L"]), len(data_np["mtr_pos_R"]),
                  len(data_np["mtr_vel_L"]), len(data_np["mtr_vel_R"]),
                  len(data_np["actual_torque_L"]), len(data_np["actual_torque_R"]),
                  len(data_np["gpio_output"]) if "gpio_output" in data_np else float('inf'),
                  len(data_np["mtr_cmd_L"]), len(data_np["mtr_cmd_R"]))

    print(f'Total data length collected: {min_len}')

    if min_len == 0:
        print("ERROR: No data collected! min_len is 0")
        return

    # Calculate start and end indices for slicing
    start_idx = int(start_rec_sec * control_freq_Hz)
    end_idx = min_len

    if trial_time_sec:
        end_idx = min(min_len, int((start_rec_sec + trial_time_sec) * control_freq_Hz))

    print(f'Slicing data from {start_rec_sec}s to {(start_rec_sec + (trial_time_sec or (min_len/100 - start_rec_sec)))}s')
    print(f'Index range: {start_idx} to {end_idx}')

    # Slice data with start offset (phase keys added for Villarreal phase-variable logging)
    phase_keys = [
        'percent_gcL', 'percent_gcR',
        'phi_pwL', 'phi_pwR',
        'phi_unifiedL', 'phi_unifiedR',
        'phase_modeL', 'phase_modeR',
        'on_plateL', 'on_plateR',
        'hip_satL', 'hip_satR',
    ]
    timestamp_sliced = [t - start_rec_sec for t in data_np["timestamp"][start_idx:end_idx]]
    sliced_data = {
        k: v[start_idx:end_idx]
        if k.startswith('mtr') or k.startswith('actual_torque') or k in ({'gpio_output', 'perturbation_idx'} | set(phase_keys))
        else None
        for k, v in data_np.items()
    }
    sliced_data['time'] = timestamp_sliced

    # Create DataFrames and save to CSV
    motor_data_keys = ['time', 'mtr_pos_L', 'mtr_pos_R', 'mtr_vel_L', 'mtr_vel_R']
    motor_data_keys += phase_keys
    if 'gpio_output' in sliced_data and sliced_data['gpio_output'] is not None:
        motor_data_keys.append('gpio_output')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_mtr = pd.DataFrame({k: sliced_data[k] for k in motor_data_keys})
    df_mtr.to_csv(f'{OUTPUT_DIR}/{trial_name}_input_motor.csv', index=False)
    print(f'Motor Data saved to {trial_name}_input_motor.csv')
    print('Dimensions:', df_mtr.shape)

    # Save motor command data
    df_torque = pd.DataFrame({k: sliced_data[k] for k in ['time', 'mtr_cmd_L', 'mtr_cmd_R', 'actual_torque_L', 'actual_torque_R', 'gpio_output']})
    df_torque.to_csv(f'{OUTPUT_DIR}/{trial_name}_output_torque.csv', index=False)
    print(f'Torque data saved to {trial_name}_output_torque.csv')
    print('Dimensions:', df_torque.shape)


# Signal handler for graceful exit
def exit_signal_handler(sig, frame):
    print("Signal received, initiating shutdown...")

    motors.disconnect()

    save_data(trial_start_sec, target_duration_sec)
    if gpio_pulse is not None:
        gpio_pulse.cleanup()
    if teleplot is not None:
        teleplot.close()
    gc.collect()

    print("Exiting program")
    os._exit(0)


def main():
    global data_to_save, motors, gpio_pulse, phase_segmenter

    gpio_pulse = GpioPulse(pin=gpio_pin)
    gpio_pulse.setup()
    pulse_scheduler = SyncPulse(
        gpio_pulse,
        first_at_sec=gpio_first_pulse_sec,
        second_at_sec=target_time_range,
        pulse_duration_sec=gpio_pulse_duration_sec,
    )

    motors = RobStrideMotorGroup(
        can_id_L=can_id_L,
        can_id_R=can_id_R,
        channel=motor_channel,
        torque_limit=torque_limit,
        offset_samples=offset_samples,
        control_freq_Hz=control_freq_Hz,
        frame_length=frame_length,
    )
    motors.connect()

    hip_torque_profile = HipTorqueProfile(
        body_mass_kg=body_mass_kg,
        control_freq_Hz=control_freq_Hz,
        scale_factor=scale_factor,
        delay_percent_gc=delay_factor,
    )

    phase_segmenter = PhaseVariableSegmenter(
        rom_deg=hip_rom_deg,
        neutral_deg_l=hip_neutral_deg_l,
        neutral_deg_r=hip_neutral_deg_r,
        stance_transition_s=stance_transition_s,
        control_freq_Hz=control_freq_Hz,
        hip_cutoff_hz=hip_cutoff_hz,
        rhythmic_min_strides=rhythmic_min_strides,
        rhythmic_cv_threshold=rhythmic_cv_threshold,
        stride_duration_window=stride_duration_window,
        output_mode=PhaseOutputMode(phase_method),
        hip_integral_method=HipIntegralMethod(hip_integral_method),
    )
    phase_segmenter.reset()

    current_pos_L, current_vel_L = 0.0, 0.0
    current_pos_R, current_vel_R = 0.0, 0.0

    atexit.register(lambda: (motors.disconnect(), gpio_pulse.cleanup()))
    signal.signal(signal.SIGINT, exit_signal_handler)

    logging_started = False
    start_time = None
    start_index = 1
    mocap_data_available = False

    percent_gcL, percent_gcR = 0.0, 0.0
    on_plateL, on_plateR = False, False
    heel_strikeL, heel_strikeR = False, False

    if trigger_type == "mocap":
        print("Waiting for start trigger...\n")
    elif trigger_type == "typing":
        input("Press Enter to start trial...\n")

    if trigger_type == "mocap" and not logging_started:
        print("Waiting for start trigger from server...")
        mocap_trigger.start_logging_event.wait()
        start_time = time.time()
        logging_started = True
        print(f"Started Vicon time: {start_time}")
    elif trigger_type == "typing" and not logging_started:
        start_time = time.time()
        logging_started = True

    while True:
        if not logging_started:
            continue

        if trigger_type == "mocap" and mocap_trigger is not None:
            if mocap_trigger.first_data_received.is_set():
                on_plateL = mocap_trigger.send_on_plateL
                on_plateR = mocap_trigger.send_on_plateR
                heel_strikeL = mocap_trigger.send_heel_strikeL
                heel_strikeR = mocap_trigger.send_heel_strikeR
                mocap_data_available = True

                with open(mocap_log_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            mocap_trigger.send_time,
                            mocap_trigger.recv_time,
                            on_plateR,
                            on_plateL,
                            heel_strikeR,
                            heel_strikeL,
                        ]
                    )
            else:
                mocap_data_available = False

        loop_time = start_time + (start_index - 1) / control_freq_Hz
        percent_gcL, percent_gcR = phase_segmenter.update_lr(
            raw_hip_encoder_l=current_pos_L,
            raw_hip_encoder_r=-current_pos_R,
            on_plate_l=on_plateL,
            on_plate_r=on_plateR,
            heel_strike_l=heel_strikeL,
            heel_strike_r=heel_strikeR,
            timestamp=loop_time,
            raw_hip_velocity_l_deg_s=current_vel_L,
            raw_hip_velocity_r_deg_s=-current_vel_R,
        )

        data_to_save["mtr_pos_L"].append(current_pos_L)
        data_to_save["mtr_pos_R"].append(-current_pos_R)
        data_to_save["mtr_vel_L"].append(current_vel_L)
        data_to_save["mtr_vel_R"].append(-current_vel_R)
        data_to_save["percent_gcL"].append(percent_gcL)
        data_to_save["percent_gcR"].append(percent_gcR)
        data_to_save["phi_pwL"].append(phase_segmenter.left.phi_pw)
        data_to_save["phi_pwR"].append(phase_segmenter.right.phi_pw)
        data_to_save["phi_unifiedL"].append(phase_segmenter.left.phi_unified)
        data_to_save["phi_unifiedR"].append(phase_segmenter.right.phi_unified)
        data_to_save["phase_modeL"].append(phase_segmenter.left.active_mode.value)
        data_to_save["phase_modeR"].append(phase_segmenter.right.active_mode.value)
        data_to_save["on_plateL"].append(int(on_plateL))
        data_to_save["on_plateR"].append(int(on_plateR))
        data_to_save["hip_satL"].append(int(phase_segmenter.left.saturated))
        data_to_save["hip_satR"].append(int(phase_segmenter.right.saturated))

        if exo_ON and mocap_data_available:
            cmdL, cmdR = hip_torque_profile.torque_from_percent_gc_lr(
                percent_gcL, percent_gcR
            )
        else:
            cmdL, cmdR = 0.0, 0.0

        motors.set_torque(-cmdL, cmdR)
        (
            current_pos_L,
            current_vel_L,
            current_torque_L,
            current_pos_R,
            current_vel_R,
            current_torque_R,
        ) = motors.update_readings()

        data_to_save["mtr_cmd_L"].append(cmdL)
        data_to_save["mtr_cmd_R"].append(cmdR)
        data_to_save["actual_torque_L"].append(current_torque_L)
        data_to_save["actual_torque_R"].append(-current_torque_R)

        current_time = time.time() - start_time
        pulse_scheduler.update(current_time)
        data_to_save["gpio_output"].append(gpio_pulse.read_state())

        teleplot.sendValue("pos_L", current_pos_L)
        teleplot.sendValue("pos_R", current_pos_R)
        teleplot.sendValue("gc_L", percent_gcL)
        teleplot.sendValue("gc_R", percent_gcR)
        teleplot.sendValue("cmd_L", cmdL)
        teleplot.sendValue("cmd_R", cmdR)

        if (time.time() - start_time) > (start_index / control_freq_Hz):
            pass
        else:
            while (time.time() - start_time) < (start_index / control_freq_Hz):
                pass

        data_to_save["timestamp"].append(time.time() - start_time)
        start_index += 1


if __name__ == "__main__":
    gc.collect()

    trial_num = int(input("Enter trial number: "))
    trial_name = f"{subject}_{trial_num}_phase_{phase_method}_scale_{scale_factor}"

    teleplot = Teleplot(teleplot_host, teleplot_port)

    if trigger_type == "mocap":
        mocap_trigger = Mocap_trigger(server_ip=mocap_server_ip, port_number=mocap_port)
        mocap_trigger.start_client()
        mocap_thread = threading.Thread(target=mocap_trigger.stream_data, daemon=True)
        mocap_thread.start()
        print("Mocap client connected and streaming thread started")

    main()
