# Extended mocap TCP client — parses phase-variable contact fields from Vicon-Computer.
import socket
import json
import time
import csv
import os
import threading


class Mocap_trigger:
    def __init__(self, server_ip, port_number):
        self.server_ip = server_ip
        self.port_number = port_number
        self.client = None

        self.time_sent = 0.0
        self.recv_time = 0.0

        # TBE gait phase (optional comparison / fallback)
        self.send_percent_gcR = 0.0
        self.send_percent_gcL = 0.0

        # Ground contact from Vicon COP (replaces FSR until onboard sensors available)
        self.send_on_plateR = False
        self.send_on_plateL = False
        self.send_heel_strikeR = False
        self.send_heel_strikeL = False

        # Trigger handling
        self.trigger_received = False
        self.trigger_value = None

        # Logging control
        self.start_logging_received = False

        self.lock = threading.Lock()
        self.running = False

        self.first_data_received = threading.Event()
        self.start_logging_event = threading.Event()

        self.latest_vicon_timestamp = None
        self.first_vicon_timestamp = None

    def start_client(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            print("[CONNECTING] Connecting to server...")
            self.client.connect((self.server_ip, self.port_number))
            print(f"[CONNECTED] Connected to server at {self.server_ip}:{self.port_number}")
        except ConnectionRefusedError:
            print(f"[ERROR] Cannot connect to server at {self.server_ip}:{self.port_number}")
            return

    def stream_data(self):
        try:
            print("[STREAMING] Receiving data from server...")
            buffer = ""
            log_file = "latency_log.csv"

            if not os.path.exists(log_file):
                with open(log_file, mode="w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Latency_ms"])

            latencies = []
            self.running = True

            while self.running:
                chunk = self.client.recv(4096).decode("utf-8")
                if not chunk:
                    print("[DISCONNECTED] Server closed the connection.")
                    break

                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()

                    if line == "START_LOGGING":
                        with self.lock:
                            self.start_logging_received = True
                        self.start_logging_event.set()
                        print("[INFO] Start logging signal received from server!")
                        continue

                    if line.isdigit():
                        trigger_value = int(line)
                        with self.lock:
                            self.trigger_received = True
                            self.trigger_value = trigger_value
                        continue

                    try:
                        data = json.loads(line)
                        self.recv_time = time.time()
                        self.send_time = float(data.get("vicon_timestamp", 0.00))
                        self.send_percent_gcR = float(data.get("percent_gcR", 0.00))
                        self.send_percent_gcL = float(data.get("percent_gcL", 0.00))
                        self.send_on_plateR = bool(data.get("on_plateR", False))
                        self.send_on_plateL = bool(data.get("on_plateL", False))
                        self.send_heel_strikeR = bool(data.get("heel_strikeR", False))
                        self.send_heel_strikeL = bool(data.get("heel_strikeL", False))

                        with self.lock:
                            self.time_sent = self.send_time
                            self.latest_vicon_timestamp = self.send_time
                            if self.first_vicon_timestamp is None:
                                self.first_vicon_timestamp = self.send_time

                        latency = (self.recv_time - self.send_time) * 1000

                        with open(log_file, mode="a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([latency])

                        latencies.append(latency)
                        if len(latencies) > 100:
                            latencies.pop(0)

                        if not self.first_data_received.is_set():
                            print("[INFO] First data received. Proceeding...")
                            self.first_data_received.set()

                    except json.JSONDecodeError:
                        print(f"[ERROR] Failed to parse line as JSON: {line}")
        except Exception as e:
            print(f"[ERROR] Streaming error: {e}")
        finally:
            self.running = False
            self.client.close()
            print("[CLOSED] Connection closed.")

    def check_trigger(self):
        with self.lock:
            if self.trigger_received:
                trigger = self.trigger_value
                self.trigger_received = False
                self.trigger_value = None
                return trigger
            return None

    def wait_for_start_logging(self):
        print("[INFO] Waiting for start logging signal from server...")
        self.start_logging_event.wait()
        return True

    def check_start_logging(self):
        with self.lock:
            return self.start_logging_received

    def get_vicon_timestamps(self):
        with self.lock:
            return self.latest_vicon_timestamp, self.first_vicon_timestamp

    def stop_streaming(self):
        self.running = False
        try:
            self.client.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        self.client.close()
