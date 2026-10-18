"""Validation 5 (spec §5.5): trajectory vs. the Allen–Eggers ballistic solution.

1. With gravity and curvature switched off, the equations of motion reduce exactly to the
   Allen–Eggers assumptions, so the integrated trajectory must match the analytic one to
   integration accuracy.
2. With full physics (the spec case: V_E = 7.5 km/s, gamma_E = -20 deg, beta = 500 kg/m^2), the
   residual is the along-path gravity term g sin(gamma) that Allen–Eggers neglects: the vehicle is
   slightly faster at every altitude (V/V_E at peak deceleration 0.6127 vs e^-1/2 = 0.6065, +1.0%;
   at peak heating 0.8571 vs e^-1/6 = 0.8465, +1.25%; peak load 52.6 g vs 50.1 g, +4.9%, of which
   about 2 points come from gravity steepening the path).
3. That explanation predicts a residual scaling with g H / V_E^2: doubling V_E cuts it ~4x.
"""

import numpy as np
import pytest

from tps.trajectory import G0, SkipOutError, Vehicle, allen_eggers_peaks, allen_eggers_velocity, fly

VE, GAMMA, BETA = 7500.0, -20.0, 500.0
DECEL = lambda V, g, h, rho: rho * V**2 / (2 * BETA)
HEATING = lambda V, g, h, rho: np.sqrt(rho) * V**3


def peaks(tr):
    ta, a = tr.peak(DECEL)
    tq, _ = tr.peak(HEATING)
    Va, _, ha, _ = tr.state(ta)
    Vq, _, hq, _ = tr.state(tq)
    return dict(Va=Va, a=a, ha=ha, Vq=Vq, hq=hq)


def test_exact_without_gravity_and_curvature():
    tr = fly(Vehicle(BETA), VE, GAMMA, gravity=False, curvature=False)
    ae = allen_eggers_peaks(VE, GAMMA, BETA)
    np.testing.assert_allclose(tr.V, allen_eggers_velocity(tr.h, VE, GAMMA, BETA), rtol=1e-5)
    np.testing.assert_allclose(np.rad2deg(tr.gamma), GAMMA, atol=1e-9)
    p = peaks(tr)
    assert p["Va"] / VE == pytest.approx(np.exp(-0.5), rel=1e-5)
    assert p["Vq"] / VE == pytest.approx(np.exp(-1.0 / 6.0), rel=1e-5)
    assert p["a"] == pytest.approx(ae["decel"]["a"], rel=1e-5)
    assert p["ha"] == pytest.approx(ae["decel"]["h"], abs=1.0)
    assert p["hq"] == pytest.approx(ae["heating"]["h"], abs=1.0)


def test_spec_case_with_full_physics_is_close():
    tr = fly(Vehicle(BETA), VE, GAMMA)
    ae = allen_eggers_peaks(VE, GAMMA, BETA)
    p = peaks(tr)
    dVa = p["Va"] / ae["decel"]["V"] - 1
    dVq = p["Vq"] / ae["heating"]["V"] - 1
    da = p["a"] / ae["decel"]["a"] - 1
    assert 0 < dVa < 0.02 and 0 < dVq < 0.02   # faster than Allen–Eggers, by < 2%
    assert 0 < da < 0.06                         # harder peak load, by < 6%
    assert abs(p["ha"] - ae["decel"]["h"]) < 500.0 and abs(p["hq"] - ae["heating"]["h"]) < 500.0
    assert np.rad2deg(tr.gamma[-1]) < GAMMA      # gravity steepens the path


def test_residual_scales_like_gravity_over_kinetic_energy():
    resid = []
    for v in (7500.0, 15000.0):
        p = peaks(fly(Vehicle(BETA), v, -60.0))
        resid.append(p["Va"] / (v * np.exp(-0.5)) - 1)
    assert 3.0 < resid[0] / resid[1] < 5.0


def test_terminates_at_velocity_cutoff():
    tr = fly(Vehicle(BETA), VE, GAMMA, V_cutoff=500.0)
    assert tr.V[-1] == pytest.approx(500.0, abs=1e-6) and tr.h[-1] > 0
    assert np.all(np.diff(tr.h) < 0)


def test_lifting_entry_flattens_the_path():
    """L/D > 0 adds an upward term to dgamma/dt: a lifting entry pulls out higher, with lower peak g."""
    ballistic = fly(Vehicle(300.0), 7800.0, -6.0)
    lifting = fly(Vehicle(300.0, lift_to_drag=0.3), 7800.0, -6.0)
    assert lifting.g_load.max() < ballistic.g_load.max()
    i_b, i_l = np.argmax(ballistic.accel), np.argmax(lifting.accel)
    assert lifting.h[i_l] > ballistic.h[i_b]


def test_skip_out_is_detected():
    """A very shallow lifting entry at orbital speed bounces off the atmosphere."""
    with pytest.raises(SkipOutError):
        fly(Vehicle(300.0, lift_to_drag=0.5), 7900.0, -0.5)
