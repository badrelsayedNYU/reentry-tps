"""Heating correlation and trajectory -> wall coupling (spec §3, §4). Test-only wall properties."""

import numpy as np
import pytest

from tps.conduction import Layer, Wall
from tps.coupling import run
from tps.heating import CP_AIR, heat_load, hot_wall_factor, sutton_graves, total_enthalpy
from tps.materials import Material
from tps.trajectory import Vehicle, allen_eggers_peaks, fly

T_TABLE = [200.0, 600.0, 1200.0, 2000.0]
SHIELD = Material.from_values("test shield", 300.0, (T_TABLE, [0.05, 0.08, 0.15, 0.30]),
                              (T_TABLE, [700.0, 1100.0, 1400.0, 1550.0]), source="test-only")
GLUE = Material.from_values("test glue", 1400.0, 0.3, 1300.0, source="test-only")
METAL = Material.from_values("test metal", 2800.0, 150.0, 900.0, source="test-only")


def test_sutton_graves_formula_and_scaling():
    q = sutton_graves(1e-4, 7000.0, 1.0)
    assert q == pytest.approx(1.7415e-4 * np.sqrt(1e-4) * 7000.0**3)
    assert sutton_graves(4e-4, 7000.0, 1.0) / q == pytest.approx(2.0)      # ~ sqrt(rho)
    assert sutton_graves(1e-4, 14000.0, 1.0) / q == pytest.approx(8.0)     # ~ V^3
    assert sutton_graves(1e-4, 7000.0, 4.0) / q == pytest.approx(0.5)      # ~ 1/sqrt(R_n)


def test_hot_wall_factor():
    V, T_inf = 7000.0, 220.0
    assert hot_wall_factor(V, T_inf, 0.0) == pytest.approx(1.0)
    T_adiabatic = total_enthalpy(V, T_inf) / CP_AIR   # wall at total temperature: no heat transfer
    assert hot_wall_factor(V, T_inf, T_adiabatic) == pytest.approx(0.0, abs=1e-12)
    assert 0.9 < hot_wall_factor(V, T_inf, 1500.0) < 1.0   # small correction at entry speeds


def test_heat_load_trapezoid():
    t = np.linspace(0, 10, 1001)
    assert heat_load(t, 3.0 * t**2) == pytest.approx(1000.0, rel=1e-5)


def test_peak_heating_time_matches_allen_eggers_without_gravity():
    beta = 500.0
    tr = fly(Vehicle(beta), 7500.0, -20.0, gravity=False, curvature=False)
    q = sutton_graves(tr.rho, tr.V, 0.5)
    i = int(np.argmax(q))
    assert tr.h[i] == pytest.approx(allen_eggers_peaks(7500.0, -20.0, beta)["heating"]["h"], abs=50.0)


@pytest.fixture(scope="module")
def coupled():
    wall = Wall([Layer(SHIELD, 0.03, 50, 1.06), Layer(GLUE, 0.0005, 3), Layer(METAL, 0.002, 4)])
    tr = fly(Vehicle(300.0), 7800.0, -6.0)
    return run(tr, wall, nose_radius=1.0, emissivity=0.85, T_initial=295.0, dt=0.2, soak_time=900.0)


def test_bondline_peaks_after_surface_soak_back(coupled):
    s = coupled.summary()
    assert s["t_peak_T_bondline_s"] > s["t_peak_T_surface_s"] > 0
    assert s["t_peak_T_bondline_s"] > coupled.trajectory.t[-1]  # after the heating has ended
    assert s["peak_T_bondline_K"] < s["peak_T_surface_K"]


def test_truncated_soak_is_flagged(coupled):
    """With a 900 s soak this thick test wall is still heating at the bondline: the summary must say so."""
    assert coupled.summary()["bondline_peak_captured"] is False
    wall = Wall([Layer(SHIELD, 0.01, 30, 1.06), Layer(GLUE, 0.0005, 3), Layer(METAL, 0.002, 4)])
    thin = run(fly(Vehicle(300.0), 7800.0, -6.0), wall, 1.0, 0.85, 295.0, dt=0.5, soak_time=3000.0)
    assert thin.summary()["bondline_peak_captured"] is True


def test_coupled_energy_balance(coupled):
    res = coupled.conduction
    net = coupled.q_conv - coupled.q_rad - res.flux_back
    supplied = np.sum(np.diff(res.t) * net[1:])   # backward Euler weighting
    stored = res.wall.stored_energy(res.T_cells[-1]) - res.wall.stored_energy(res.T_cells[0])
    assert supplied == pytest.approx(stored, rel=1e-6)


def test_heating_matches_trajectory(coupled):
    tr = coupled.trajectory
    q_traj = sutton_graves(tr.rho, tr.V, 1.0)
    assert coupled.summary()["peak_q_W_m2"] == pytest.approx(q_traj.max(), rel=2e-3)
    assert coupled.summary()["heat_load_J_m2"] == pytest.approx(heat_load(tr.t, q_traj), rel=2e-3)
    assert np.all(coupled.q_conv[coupled.t > tr.t[-1]] == 0.0)


def test_hot_wall_reduces_heat_input():
    wall = Wall([Layer(SHIELD, 0.03, 40, 1.06), Layer(GLUE, 0.0005, 3), Layer(METAL, 0.002, 4)])
    tr = fly(Vehicle(300.0), 7800.0, -6.0)
    cold = run(tr, wall, 1.0, 0.85, 295.0, dt=0.5, soak_time=300.0)
    hot = run(tr, wall, 1.0, 0.85, 295.0, dt=0.5, soak_time=300.0, hot_wall=True)
    assert hot.summary()["peak_T_surface_K"] < cold.summary()["peak_T_surface_K"]
    assert hot.summary()["heat_load_J_m2"] < cold.summary()["heat_load_J_m2"]
