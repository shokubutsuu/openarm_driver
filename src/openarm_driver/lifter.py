# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Driver for the OpenArm Cell lifter."""

import logging
import math
import time

import openarm_can as oa

logger = logging.getLogger(__name__)

# Fastest motor speed (rad/s) any move is given.
MAX_SPEED_RAD_S = 30.0
# Speed (rad/s) of calibration and homing.
HOMING_SPEED_RAD_S = MAX_SPEED_RAD_S / 5.0

# The DM4310 reports its angle only from -12.5 to +12.5 rad. Past either end
# the reading jumps to the other end, so it repeats every 25 rad: about 4
# turns, or 20 mm of lift with a 5 mm lead. Counting those jumps gives the
# real position, as long as the reading is taken more often than every
# half of that (12.5 rad, 0.4 s at full speed).
POSITION_WRAP_RAD = 25.0

# Every move aims at a target this far (rad) beyond the stroke, up or down,
# and sets only the speed. The motor takes targets in its own multi-turn
# count, which can be whole 25 rad turns off from the wrapped position it
# reports, so a target computed from the reading can land 20 mm off; one
# this far lands on the right side on any count.
FAR_RAD = 1000.0
# Speed (rad/s) per rad of distance left when driving to a position, so a
# move slows down and stops on it (at 0 rad/s the motor holds still).
DRIVE_GAIN = 5.0
# Fastest speed (rad/s) used to hold a position against drift.
HOLD_SPEED_RAD_S = 1.0

# Moves slow down linearly within this fraction of the stroke from either
# end, to EDGE_SPEED_FACTOR of the commanded speed at the end itself.
EDGE_MARGIN = 0.05
EDGE_SPEED_FACTOR = 0.25

# Torque (Nm) that means the lifter is pushing against the bottom stop.
BOTTOM_TORQUE_NM = 0.7
# Calibration first moves up for RUNWAY_S, then drives down. The torque of
# reversing is not the bottom, so it is ignored for CALIBRATION_SETTLE_S,
# and the bottom is only taken once the torque has stayed over
# BOTTOM_TORQUE_NM for BOTTOM_CONFIRM_SAMPLES in a row.
RUNWAY_S = 0.3
CALIBRATION_SETTLE_S = 0.3
BOTTOM_CONFIRM_SAMPLES = 5

# How often blocking calibration and homing command the motor.
BLOCKING_PERIOD_S = 0.004
# Homing gives up once the position has not moved this much for this long
# without reaching the bottom, e.g. when the motor's power is off.
HOMING_STALL_RAD = 0.1
HOMING_STALL_S = 5.0
# How often blocking calibration and homing log their progress.
PROGRESS_LOG_INTERVAL_S = 5.0


class _Unwrapper:
    """Turn the wrapped position reading into a continuous position."""

    def __init__(self):
        self._prev = None
        self._position = 0.0

    def update(self, raw: float) -> float:
        if self._prev is not None:
            diff = raw - self._prev
            if diff > POSITION_WRAP_RAD / 2.0:
                diff -= POSITION_WRAP_RAD
            elif diff < -POSITION_WRAP_RAD / 2.0:
                diff += POSITION_WRAP_RAD
            self._position += diff
        else:
            self._position = raw
        self._prev = raw
        return self._position


class _BottomDetector:
    """Decide when a descent that began at `started` has reached the bottom."""

    def __init__(self, started: float):
        self._started = started
        self._over = 0

    def update(self, now: float, torque: float) -> bool:
        if (
            now - self._started < CALIBRATION_SETTLE_S
            or abs(torque) <= BOTTOM_TORQUE_NM
        ):
            self._over = 0
            return False
        self._over += 1
        return self._over >= BOTTOM_CONFIRM_SAMPLES


class LifterDriver:
    """Driver for the OpenArm Cell lifter: one motor on a lead screw.

    Heights are in mm above the bottom stop, which `calibrate` finds by
    driving down until the motor pushes against it. Moves (`move`,
    `move_to`, `hold`) are meant to be called every control cycle; each
    reads the motor first, so `state` stays current.
    """

    def __init__(
        self,
        can_interface: str = "can2",
        lead_mm: float = 5.0,
        stroke_mm: float = 300.0,
        motor_type: str = "DM4310",
        send_id: int = 0x0A,
        recv_id: int = 0x1A,
    ):
        """Initialize the lifter motor in position-velocity mode.

        Args:
            can_interface: SocketCAN interface the motor is on.
            lead_mm: Lead of the screw (mm per turn).
            stroke_mm: Travel from the bottom stop to the top.
            motor_type: Name of the openarm_can.MotorType.
            send_id: CAN ID the motor listens on.
            recv_id: CAN ID the motor replies on.

        """
        self.can_interface = can_interface
        self.lead_mm = lead_mm
        self.stroke_mm = stroke_mm
        self.openarm = oa.OpenArm(can_interface, True)
        self.openarm.init_arm_motors(
            [getattr(oa.MotorType, motor_type)],
            [send_id],
            [recv_id],
            [oa.ControlMode.POS_VEL],
        )
        self.openarm.set_callback_mode_all(oa.CallbackMode.STATE)
        self._stroke_rad = self._to_rad(stroke_mm)
        self._unwrapper = _Unwrapper()
        self._offset_rad: float | None = None
        self._bottom: _BottomDetector | None = None
        self._raw_rad = 0.0
        self._position_rad = 0.0
        self._velocity_rad_s = 0.0
        self._torque_nm = 0.0

    @property
    def calibrated(self) -> bool:
        """Return whether the bottom stop has been found."""
        return self._offset_rad is not None

    @property
    def max_speed_mm_s(self) -> float:
        """Return the fastest speed a move is given."""
        return self._to_mm(MAX_SPEED_RAD_S)

    @property
    def state(self) -> dict:
        """Return the latest reading.

        `elevation_mm` and `velocity_mm_s` are None until calibrated;
        `torque_nm` is the motor torque.
        """
        return {
            "elevation_mm": self._to_mm(self._position_rad)
            if self.calibrated
            else None,
            "velocity_mm_s": self._to_mm(self._velocity_rad_s)
            if self.calibrated
            else None,
            "torque_nm": self._torque_nm,
        }

    def enable(self):
        """Enable the motor."""
        self.openarm.enable_all()
        self._update_state()

    def disable(self):
        """Disable the motor; it no longer holds the lifter."""
        self.openarm.disable_all()
        self._update_state()

    def fetch_state(self) -> dict:
        """Read the motor and return `state`."""
        self._update_state()
        return self.state

    def start_calibration(self):
        """Start finding the bottom stop; continue with `step_calibration`.

        Moves up for a moment first, so the descent has room to build up
        the speed and torque the bottom is detected by.
        """
        self._offset_rad = None
        self._send(FAR_RAD, HOMING_SPEED_RAD_S)
        time.sleep(RUNWAY_S)
        # Start down now rather than at the first step, which may come late,
        # or the lifter would keep rising.
        self._send(-FAR_RAD, HOMING_SPEED_RAD_S)
        self._bottom = _BottomDetector(time.monotonic())

    def step_calibration(self) -> bool:
        """Drive down for one cycle; return whether the bottom is found.

        Once found, the lifter holds still at the bottom (elevation 0).
        """
        if self._bottom is None:
            raise RuntimeError("start_calibration() was not called")
        self._send(-FAR_RAD, HOMING_SPEED_RAD_S)
        self.openarm.recv_all()
        raw, velocity, torque = self._read_motor()
        if not self._bottom.update(time.monotonic(), torque):
            return False
        # Start the unwrapper from this very reading. Starting it from the
        # next one instead would miss a jump right at the bottom (e.g. -12.49
        # here, +12.49 next) and read 20 mm at the bottom.
        self._unwrapper = _Unwrapper()
        self._offset_rad = self._unwrapper.update(raw)
        self._raw_rad = raw
        self._position_rad = 0.0
        self._velocity_rad_s = velocity
        self._torque_nm = torque
        self._bottom = None
        self._send(FAR_RAD, 0.0)
        logger.info("lifter: found the bottom on %s", self.can_interface)
        return True

    def calibrate(self, timeout_s: float | None = None) -> bool:
        """Find the bottom stop; return whether it was found in time.

        Args:
            timeout_s: Give up after this long. Defaults to the time a full
                stroke takes plus a margin.

        """
        if timeout_s is None:
            timeout_s = self._full_stroke_s() * 1.1 + 2.0
        self.start_calibration()
        started = time.monotonic()
        while not self.step_calibration():
            if time.monotonic() - started >= timeout_s:
                self._bottom = None
                self._send(FAR_RAD, 0.0)
                logger.warning(
                    "lifter: did not find the bottom in %.0f s (torque %.3f Nm)",
                    timeout_s,
                    self._torque_nm,
                )
                return False
            time.sleep(BLOCKING_PERIOD_S)
        return True

    def move(self, speed_mm_s: float):
        """Move up (positive) or down (negative) at a speed, for one cycle.

        The speed is capped at `max_speed_mm_s` and reduced near either end
        of the stroke, and the lifter stops at the end.
        """
        self._require_calibrated()
        self._update_state()
        target = self._stroke_rad if speed_mm_s > 0 else 0.0
        speed = min(MAX_SPEED_RAD_S, self._to_rad(abs(speed_mm_s)))
        self._drive(target, speed * self._edge_factor())

    def move_to(self, elevation_mm: float, max_speed_mm_s: float):
        """Move toward a height for one cycle, slowing to a stop on it.

        The height is clamped to the stroke.
        """
        self._require_calibrated()
        self._update_state()
        target = self._to_rad(max(0.0, min(self.stroke_mm, elevation_mm)))
        self._drive(target, min(MAX_SPEED_RAD_S, self._to_rad(max_speed_mm_s)))

    def hold(self, elevation_mm: float | None = None):
        """Hold a height (default: the current one) for one cycle.

        Before calibration, holds wherever the lifter is.
        """
        if not self.calibrated:
            self._send(FAR_RAD, 0.0)
            self._update_state()
            return
        self._update_state()
        if elevation_mm is None:
            self._send(FAR_RAD, 0.0)
            return
        target = self._to_rad(max(0.0, min(self.stroke_mm, elevation_mm)))
        self._drive(target, HOLD_SPEED_RAD_S)

    def home(self, timeout_s: float | None = None) -> bool:
        """Drive down to the bottom stop; return whether it got there.

        Gives up when the lifter has not moved for HOMING_STALL_S (e.g. its
        power is off) or after `timeout_s`, by default the time a full
        stroke takes plus a margin. It does not hold or disable afterwards;
        that is the caller's call.
        """
        if timeout_s is None:
            timeout_s = self._full_stroke_s() * 1.1 + 2.0
        logger.info("lifter: homing to the bottom (limit %.0f s)", timeout_s)
        started = time.monotonic()
        moved_at = started
        moved_from = None
        next_log_s = PROGRESS_LOG_INTERVAL_S
        while True:
            self._send(-FAR_RAD, HOMING_SPEED_RAD_S)
            self._update_state()
            raw, torque = self._raw_rad, self._torque_nm
            now = time.monotonic()
            elapsed = now - started
            if moved_from is None or abs(raw - moved_from) > HOMING_STALL_RAD:
                moved_from = raw
                moved_at = now
            if abs(torque) > BOTTOM_TORQUE_NM:
                logger.info("lifter: reached the bottom after %.1f s", elapsed)
                return True
            if now - moved_at >= HOMING_STALL_S:
                logger.warning(
                    "lifter: did not reach the bottom: not moving for %.0f s "
                    "(torque %.3f Nm)",
                    HOMING_STALL_S,
                    torque,
                )
                return False
            if elapsed >= timeout_s:
                logger.warning(
                    "lifter: did not reach the bottom in %.0f s (torque %.3f Nm)",
                    elapsed,
                    torque,
                )
                return False
            if elapsed >= next_log_s:
                logger.info("lifter: homing %.0f s", elapsed)
                next_log_s += PROGRESS_LOG_INTERVAL_S
            time.sleep(BLOCKING_PERIOD_S)

    def _drive(self, target: float, max_speed: float):
        # Aim at FAR_RAD on the target's side and set only the speed, which
        # falls to 0 at the target.
        distance = target - self._position_rad
        speed = min(max_speed, DRIVE_GAIN * abs(distance))
        self._send(FAR_RAD if distance > 0 else -FAR_RAD, speed)

    def _edge_factor(self) -> float:
        margin = self._stroke_rad * EDGE_MARGIN
        to_edge = min(self._position_rad, self._stroke_rad - self._position_rad)
        if to_edge >= margin:
            return 1.0
        return EDGE_SPEED_FACTOR + (1.0 - EDGE_SPEED_FACTOR) * max(
            0.0, to_edge / margin
        )

    def _send(self, q: float, dq: float):
        self.openarm.get_arm().posvel_control_all([oa.PosVelParam(q=q, dq=dq)])

    def _read_motor(self) -> tuple[float, float, float]:
        motor = self.openarm.get_arm().get_motors()[0]
        return motor.get_position(), motor.get_velocity(), motor.get_torque()

    def _update_state(self):
        self.openarm.recv_all()
        raw, velocity, torque = self._read_motor()
        self._raw_rad = raw
        self._velocity_rad_s = velocity
        self._torque_nm = torque
        if self.calibrated:
            self._position_rad = self._unwrapper.update(raw) - self._offset_rad

    def _require_calibrated(self):
        if not self.calibrated:
            raise RuntimeError("the lifter is not calibrated")

    def _full_stroke_s(self) -> float:
        return self._stroke_rad / HOMING_SPEED_RAD_S

    def _to_rad(self, mm: float) -> float:
        return mm / self.lead_mm * 2.0 * math.pi

    def _to_mm(self, rad: float) -> float:
        return rad / (2.0 * math.pi) * self.lead_mm
