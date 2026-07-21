"""GPIO pulse control for Vicon trial sync triggers."""

import Jetson.GPIO as GPIO


class GpioPulse:
    def __init__(self, pin: int = 7):
        self.pin = pin

    def setup(self):
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)
        print("GPIO initialized successfully")

    def start(self):
        try:
            GPIO.output(self.pin, GPIO.HIGH)
            print("GPIO pulse started (HIGH)")
        except Exception as e:
            print(f"Error starting GPIO pulse: {e}")

    def end(self):
        try:
            GPIO.output(self.pin, GPIO.LOW)
            print("GPIO pulse ended (LOW)")
        except Exception as e:
            print(f"Error ending GPIO pulse: {e}")

    def read_state(self) -> int:
        try:
            return int(GPIO.input(self.pin))
        except Exception:
            return 0

    def cleanup(self):
        try:
            GPIO.cleanup()
            print("GPIO cleaned up successfully")
        except Exception as e:
            print(f"Error during GPIO cleanup: {e}")


class SyncPulse:
    """Send two short GPIO pulses at configured times after trial start."""

    def __init__(
        self,
        gpio: GpioPulse,
        first_at_sec: float = 2.0,
        second_at_sec: float = 31.0,
        pulse_duration_sec: float = 0.05,
    ):
        self.gpio = gpio
        self.first_at_sec = first_at_sec
        self.second_at_sec = second_at_sec
        self.pulse_duration_sec = pulse_duration_sec
        self._first_sent = False
        self._first_end_time = None
        self._second_sent = False
        self._second_end_time = None

    def update(self, current_time: float):
        if current_time >= self.first_at_sec and not self._first_sent:
            self.gpio.start()
            self._first_sent = True
            self._first_end_time = current_time + self.pulse_duration_sec
            print("First pulse started 2 seconds after mocap trigger")

        if self._first_sent and self._first_end_time and current_time >= self._first_end_time:
            self.gpio.end()
            self._first_end_time = None
            print("First pulse ended")

        if current_time >= self.second_at_sec and not self._second_sent:
            self.gpio.start()
            self._second_sent = True
            self._second_end_time = current_time + self.pulse_duration_sec
            print(f"Second pulse started after {current_time} seconds")

        if self._second_sent and self._second_end_time and current_time >= self._second_end_time:
            self.gpio.end()
            self._second_end_time = None
            print("Second pulse ended")
