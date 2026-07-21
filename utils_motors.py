"""RobStride RS-02 motor interface for Vicon exoskeleton protocols."""
import os
import time
import numpy as np
from robstride_dynamics import Motor, RobstrideBus

RAD_TO_DEG = 180.0 / np.pi

MOTOR_L = "MOTOR_L"
MOTOR_R = "MOTOR_R"


class RobStrideMotorGroup:
    def __init__(
        self,
        can_id_L: int = 1,
        can_id_R: int = 2,
        channel: str = "can0",
        torque_limit: float = 17.0,
        offset_samples: int = 50,
        control_freq_Hz: float = 100,
        frame_length: int = 95,
    ):
        self.channel = channel
        self.torque_limit = torque_limit
        self.offset_samples = offset_samples
        self.control_freq_Hz = control_freq_Hz
        self.frame_length = frame_length

        # biotorque parameters
        self.scale_factor = 0
        self.delay_factor = 0  # Number of frames to delay the torque command
        self.duration = 0

        self.motors = {
            MOTOR_L: Motor(id=can_id_L, model="rs-02"),
            MOTOR_R: Motor(id=can_id_R, model="rs-02"),
        }

        self._offsets = {MOTOR_L: 0.0, MOTOR_R: 0.0}
        self.bus = None

    def _setup_can(self) -> None:
        cmds = [
            f"sudo ip link set {self.channel} down",
            f"sudo ip link set {self.channel} type can bitrate 1000000 restart-ms 100 berr-reporting off",
            f"sudo ip link set {self.channel} up",
        ]
        for cmd in cmds:
            rc = os.system(cmd)
            if rc != 0:
                raise RuntimeError(f"Command failed (rc={rc}): {cmd}")
        
    def _calibrate_offsets(self) -> None:
        offsets = {MOTOR_L: [], MOTOR_R: []}
        for _ in range(self.offset_samples):
            self.set_torque(0.0, 0.0)   # ← ADD HERE each iteration
            for motor in (MOTOR_L, MOTOR_R):
                pos, _, _, _ = self.bus.read_operation_frame(motor)
                offsets[motor].append(pos)
            time.sleep(0.01)

        for motor in (MOTOR_L, MOTOR_R):
            self._offsets[motor] = float(np.mean(offsets[motor]))

        # print(
        #     "RobStride zero offsets (rad): "
        #     f"left={self._offsets[MOTOR_L]:.4f}, "
        #     f"right={self._offsets[MOTOR_R]:.4f}"
        # )

    def connect(self, prompt: bool = True) -> None:
        if prompt:
            input("Press Enter to initialize RobStride RS-02 motors: ")
        self._setup_can()
        self.bus = RobstrideBus(self.channel, self.motors)
        self.bus.connect()
        self.bus.enable(MOTOR_L)
        self.bus.enable(MOTOR_R)
        time.sleep(0.3)
        self.set_torque(0.0, 0.0)
        self._calibrate_offsets()

    def disconnect(self) -> None:
        if self.bus is not None and self.bus.is_connected:
            self.set_torque(0.0, 0.0)
            time.sleep(0.1)
            self.bus.disconnect(disable_torque=True)

    def set_torque(self, torque_l: float, torque_r: float) -> None:
        torque_l = float(np.clip(torque_l, -self.torque_limit, self.torque_limit))
        torque_r = float(np.clip(torque_r, -self.torque_limit, self.torque_limit))
        self.bus.write_operation_frame(MOTOR_L, 0, 0, 0, 0, torque_l)
        self.bus.write_operation_frame(MOTOR_R, 0, 0, 0, 0, torque_r)

    def update_readings(self, degrees: bool = True):
        """Read position, velocity, and torque for both motors (2 CAN reads)."""
        pos_l, vel_l, tor_l, _ = self.bus.read_operation_frame(MOTOR_L)
        pos_r, vel_r, tor_r, _ = self.bus.read_operation_frame(MOTOR_R)

        pos_l -= self._offsets[MOTOR_L]
        pos_r -= self._offsets[MOTOR_R]

        if degrees:
            pos_l *= RAD_TO_DEG
            vel_l *= RAD_TO_DEG
            pos_r *= RAD_TO_DEG
            vel_r *= RAD_TO_DEG

        return pos_l, vel_l, tor_l, pos_r, vel_r, tor_r
