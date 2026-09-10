"""Linux joystick reader used by the Isaac Lab teleoperation script."""

from __future__ import annotations

import os
import struct
import threading


class LinuxJoystick:
    """Read a Linux ``/dev/input/js*`` device in a background thread."""

    def __init__(self, device: str, max_vx: float, max_vy: float, max_wz: float):
        self.device = device
        self.max_vx = float(max_vx)
        self.max_vy = float(max_vy)
        self.max_wz = float(max_wz)
        self.available = False
        self._running = True
        self._lock = threading.Lock()
        self._command = [0.0, 0.0, 0.0]
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        if not os.path.exists(self.device):
            print(f"[Joystick] device not found: {self.device}")
            return
        try:
            with open(self.device, "rb") as joystick:
                self.available = True
                print(f"[Joystick] listening on {self.device}; axes 1=vx, 0=vy, 3=wz")
                event_size = struct.calcsize("IhBB")
                while self._running:
                    event = joystick.read(event_size)
                    if not event:
                        break
                    _, value, event_type, number = struct.unpack("IhBB", event)
                    if event_type & 0x80 or event_type != 0x02:
                        continue
                    value = value / 32767.0
                    if abs(value) < 0.1:
                        value = 0.0
                    with self._lock:
                        if number == 1:
                            self._command[0] = -value * self.max_vx
                        elif number == 0:
                            self._command[1] = -value * self.max_vy
                        elif number == 3:
                            self._command[2] = -value * self.max_wz
        except OSError as exc:
            print(f"[Joystick] read error: {exc}")
        finally:
            self.available = False

    def command(self) -> list[float]:
        with self._lock:
            return list(self._command)

    def stop(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
