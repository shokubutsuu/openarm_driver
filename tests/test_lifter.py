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

import logging
import math

import pytest

from openarm_driver import lifter as lifter_module
from openarm_driver.lifter import LifterDriver

LEAD_MM = 5.0
STROKE_MM = 300.0
STROKE_RAD = STROKE_MM / LEAD_MM * 2.0 * math.pi
TICK_S = 0.004


class LifterSim:
    """A DM4310 on a lead screw, seen through openarm_can.

    `true_rad` is the motor's own multi-turn count, which is also the frame
    it takes position targets in. It reports that count wrapped to
    -12.5..12.5 rad, like the real motor.
    """

    def __init__(self, bottom_rad, start_mm=100.0):
        self.bottom_rad = bottom_rad
        self.true_rad = bottom_rad + start_mm / LEAD_MM * 2.0 * math.pi
        self.q = self.true_rad
        self.dq = 0.0
        self.velocity = 0.0
        self.torque = 0.0
        self.powered = True
        self.spike_reads = 0
        self.jitter_rad = 0.0
        self._reads = 0
        self.now = 0.0

    def step(self, dt):
        self.now += dt
        if not self.powered:
            self.velocity = 0.0
            self.torque = 0.0
            return
        distance = self.q - self.true_rad
        move = math.copysign(min(abs(distance), self.dq * dt), distance)
        top = self.bottom_rad + STROKE_RAD
        new = self.true_rad + move
        if new <= self.bottom_rad and move < 0:
            self.true_rad = self.bottom_rad
            self.velocity = 0.0
            self.torque = -1.0
        elif new >= top and move > 0:
            self.true_rad = top
            self.velocity = 0.0
            self.torque = 1.0
        else:
            self.true_rad = new
            self.velocity = move / dt if dt else 0.0
            self.torque = 0.1 if move else 0.0

    def sleep(self, seconds):
        # Step in ticks so a long sleep cannot jump past the bottom.
        while seconds > 1e-9:
            dt = min(TICK_S, seconds)
            self.step(dt)
            seconds -= dt

    def monotonic(self):
        return self.now

    # openarm_can.Motor
    def get_position(self):
        # Alternate the noise on each read, as a reading at the jump would.
        self._reads += 1
        noise = self.jitter_rad if self._reads % 2 else -self.jitter_rad
        return (self.true_rad + noise + 12.5) % 25.0 - 12.5

    def get_velocity(self):
        return self.velocity

    def get_torque(self):
        if self.spike_reads:
            self.spike_reads -= 1
            return 1.0
        return self.torque


class OpenArmFake:
    """Stands in for openarm_can.OpenArm and its arm collection."""

    def __init__(self, sim):
        self.sim = sim

    def __getattr__(self, name):
        return lambda *args, **kwargs: None

    def get_arm(self):
        return self

    def get_motors(self):
        return [self.sim]

    def posvel_control_all(self, params):
        self.sim.q = params[0].q
        self.sim.dq = params[0].dq


def make_lifter(monkeypatch, sim):
    monkeypatch.setattr(lifter_module.oa, "OpenArm", lambda *a, **k: OpenArmFake(sim))
    monkeypatch.setattr(lifter_module.time, "sleep", sim.sleep)
    monkeypatch.setattr(lifter_module.time, "monotonic", sim.monotonic)
    lifter = LifterDriver(lead_mm=LEAD_MM, stroke_mm=STROKE_MM)
    lifter.enable()
    return lifter


def run(lifter, sim, command, seconds):
    for _ in range(int(seconds / TICK_S)):
        command()
        sim.step(TICK_S)
    return lifter.fetch_state()


# Bottoms in several frames: right at the reading's jump, and whole 25 rad
# turns away from the reported range.
BOTTOMS = [0.0, -12.49, 12.45, 40.0, -77.3]


@pytest.mark.parametrize("bottom", BOTTOMS)
def test_calibrate_reads_zero_at_the_bottom(monkeypatch, bottom):
    sim = LifterSim(bottom)
    lifter = make_lifter(monkeypatch, sim)

    assert lifter.calibrate()

    assert lifter.calibrated
    assert sim.true_rad == pytest.approx(bottom)
    assert lifter.fetch_state()["elevation_mm"] == pytest.approx(0.0, abs=0.05)


def test_calibrate_reads_zero_with_a_noisy_reading_at_the_jump(monkeypatch):
    # The reading at this bottom flips between -12.49 and +12.49.
    sim = LifterSim(-12.49)
    sim.jitter_rad = 0.02
    lifter = make_lifter(monkeypatch, sim)

    assert lifter.calibrate()

    for _ in range(4):
        assert lifter.fetch_state()["elevation_mm"] == pytest.approx(0.0, abs=0.05)


def test_calibrate_ignores_the_reversal_spike(monkeypatch):
    sim = LifterSim(0.0)
    lifter = make_lifter(monkeypatch, sim)
    sim.spike_reads = BOTTOM_CONFIRM = lifter_module.BOTTOM_CONFIRM_SAMPLES

    lifter.start_calibration()
    for _ in range(BOTTOM_CONFIRM):
        assert not lifter.step_calibration()
        sim.step(TICK_S)

    assert lifter.calibrate()
    assert sim.true_rad == pytest.approx(0.0)


def test_calibrate_gives_up_when_the_lifter_does_not_move(monkeypatch):
    sim = LifterSim(0.0)
    sim.powered = False
    lifter = make_lifter(monkeypatch, sim)

    assert not lifter.calibrate(timeout_s=1.0)
    assert not lifter.calibrated


def test_moves_need_calibration(monkeypatch):
    lifter = make_lifter(monkeypatch, LifterSim(0.0))

    with pytest.raises(RuntimeError):
        lifter.move(10.0)
    with pytest.raises(RuntimeError):
        lifter.move_to(10.0, 10.0)


@pytest.mark.parametrize("bottom", BOTTOMS)
@pytest.mark.parametrize("target", [7.0, 150.0, 20.0])
def test_move_to_stops_on_the_height_in_any_frame(monkeypatch, bottom, target):
    sim = LifterSim(bottom)
    lifter = make_lifter(monkeypatch, sim)
    lifter.calibrate()

    state = run(lifter, sim, lambda: lifter.move_to(target, 20.0), 20.0)

    assert state["elevation_mm"] == pytest.approx(target, abs=0.1)
    true_mm = (sim.true_rad - bottom) / (2.0 * math.pi) * LEAD_MM
    assert true_mm == pytest.approx(target, abs=0.1)


@pytest.mark.parametrize("bottom", BOTTOMS)
def test_move_goes_the_right_way_from_the_start(monkeypatch, bottom):
    sim = LifterSim(bottom)
    lifter = make_lifter(monkeypatch, sim)
    lifter.calibrate()
    lifter.move_to(40.0, 20.0)
    run(lifter, sim, lambda: lifter.move_to(40.0, 20.0), 5.0)

    down = run(lifter, sim, lambda: lifter.move(-5.0), 1.0)["elevation_mm"]
    up = run(lifter, sim, lambda: lifter.move(5.0), 1.0)["elevation_mm"]

    assert down == pytest.approx(35.0, abs=0.3)
    assert up == pytest.approx(40.0, abs=0.3)


def test_move_slows_near_the_ends_and_stops_at_them(monkeypatch):
    sim = LifterSim(0.0)
    lifter = make_lifter(monkeypatch, sim)
    lifter.calibrate()
    run(lifter, sim, lambda: lifter.move_to(150.0, 25.0), 15.0)

    lifter.move(lifter.max_speed_mm_s)
    full = sim.dq
    run(lifter, sim, lambda: lifter.move(lifter.max_speed_mm_s), 10.0)
    lifter.move(lifter.max_speed_mm_s)
    at_top = lifter.state["elevation_mm"]

    assert full == pytest.approx(lifter_module.MAX_SPEED_RAD_S)
    assert at_top == pytest.approx(STROKE_MM, abs=0.1)
    assert sim.true_rad < STROKE_RAD
    assert sim.dq < 0.01


def test_hold_keeps_the_height(monkeypatch):
    sim = LifterSim(0.0)
    lifter = make_lifter(monkeypatch, sim)
    lifter.calibrate()
    run(lifter, sim, lambda: lifter.move_to(60.0, 20.0), 10.0)

    sim.true_rad += 0.5
    state = run(lifter, sim, lambda: lifter.hold(60.0), 2.0)

    assert state["elevation_mm"] == pytest.approx(60.0, abs=0.05)


@pytest.mark.parametrize("bottom", BOTTOMS)
def test_home_reaches_the_bottom(monkeypatch, caplog, bottom):
    sim = LifterSim(bottom, start_mm=250.0)
    lifter = make_lifter(monkeypatch, sim)
    caplog.set_level(logging.INFO)

    assert lifter.home()

    assert sim.true_rad == pytest.approx(bottom)
    assert "reached the bottom" in caplog.text


def test_home_gives_up_when_the_lifter_does_not_move(monkeypatch, caplog):
    sim = LifterSim(0.0)
    lifter = make_lifter(monkeypatch, sim)
    sim.powered = False

    assert not lifter.home()

    assert sim.now == pytest.approx(lifter_module.HOMING_STALL_S, abs=0.1)
    assert "not moving" in caplog.text


def test_home_gives_up_after_the_time_limit(monkeypatch, caplog):
    sim = LifterSim(0.0, start_mm=250.0)
    lifter = make_lifter(monkeypatch, sim)

    assert not lifter.home(timeout_s=2.0)

    assert sim.now == pytest.approx(2.0, abs=0.1)
    assert sim.true_rad > 0.0
