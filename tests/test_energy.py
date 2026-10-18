"""Validation 4 (spec §5.4) and the radiation / temperature-dependent-property checks (milestone 2).

Energy conservation: with q_in the applied heating, q_rad = eps sigma (T_s^4 - T_inf^4) and q_back
the flux leaving through the back face,
    integral (q_in - q_rad - q_back) dt = sum_i rho_i dx_i [e(T_i, end) - e(T_i, start)],
where e(T) = integral c_p dT (the energy-conserving form with variable c_p). Time integrals use the
same theta weighting as the scheme.

All properties here are test-only tables, not material data.
"""

import numpy as np
import pytest

from tps.conduction import SIGMA, AdiabaticBC, Layer, RadiatingFluxBC, RadiationBC, TemperatureBC, Wall, solve
from tps.materials import Material

T_TABLE = [200.0, 600.0, 1200.0, 2000.0]
SHIELD = Material.from_values("test shield", rho=300.0, k=(T_TABLE, [0.05, 0.08, 0.15, 0.30]),
                              cp=(T_TABLE, [700.0, 1100.0, 1400.0, 1550.0]), source="test-only")
GLUE = Material.from_values("test glue", rho=1400.0, k=(T_TABLE, [0.25, 0.30, 0.35, 0.40]),
                            cp=(T_TABLE, [1000.0, 1300.0, 1500.0, 1600.0]), source="test-only")
METAL = Material.from_values("test metal", rho=2800.0, k=(T_TABLE, [120.0, 160.0, 180.0, 190.0]),
                             cp=(T_TABLE, [800.0, 1000.0, 1100.0, 1150.0]), source="test-only")


def pulse(t):
    """A heating pulse shaped like an entry: rises to 600 kW/m^2 at t = 60 s, gone by t = 160 s."""
    return 6e5 * np.sin(np.pi * np.clip(t - 10.0, 0, 150) / 150.0) ** 2


def wall():
    return Wall([Layer(SHIELD, 0.03, 60, 1.05), Layer(GLUE, 0.0005, 4), Layer(METAL, 0.003, 8)])


@pytest.mark.parametrize("theta", [1.0, 0.5])
def test_energy_conservation_with_radiation_and_variable_properties(theta):
    front = RadiatingFluxBC(lambda t, Ts: pulse(t), 0.85, 250.0)
    back = RadiationBC(0.6, 295.0)
    w = wall()
    res = solve(w, 295.0, 400.0, 0.25, front, back, theta=theta)

    q_in = pulse(res.t)
    q_rad = front.radiated(res.t, res.T_surface)
    q_back = res.flux_back
    net = q_in - q_rad - q_back
    h = np.diff(res.t)
    supplied = np.sum(h * (theta * net[1:] + (1 - theta) * net[:-1]))
    stored = w.stored_energy(res.T_cells[-1]) - w.stored_energy(res.T_cells[0])

    assert abs(supplied - stored) / stored < 1e-3                   # the spec's 0.1% criterion
    assert abs(supplied - stored) / stored < 1e-6                   # in fact: closes to the iteration tolerance
    assert res.T_surface.max() > 1200.0                             # the T-dependent tables are exercised
    assert np.max(res.iterations) > 1                               # and the nonlinear loop is active


def test_radiative_equilibrium():
    """Constant heating, adiabatic back: the wall heats until re-radiation balances the input,
    eps sigma (T_s^4 - T_inf^4) = q, with a uniform temperature through the wall."""
    q, eps, T_inf = 2e5, 0.85, 250.0
    w = Wall([Layer(SHIELD, 0.01, 20)])
    res = solve(w, 300.0, 5e5, 500.0, RadiatingFluxBC(q, eps, T_inf), AdiabaticBC())
    T_eq = (q / (eps * SIGMA) + T_inf**4) ** 0.25
    assert res.T_surface[-1] == pytest.approx(T_eq, rel=1e-6)
    np.testing.assert_allclose(res.T_cells[-1], T_eq, rtol=1e-6)


def test_temperature_dependent_conductivity_steady():
    """k(T) = a + b T, fixed end temperatures. The Kirchhoff transform K(T) = a T + b T^2 / 2 is
    linear in x at steady state, so T(x) is known exactly (Incropera §3.2 / Carslaw & Jaeger §1.8)."""
    a, b = 0.5, 2e-3
    mat = Material.from_values("k(T) test", rho=1000.0, k=([0.0, 3000.0], [a, a + b * 3000.0]),
                               cp=1000.0, source="test-only")
    L, Th, Tc = 0.02, 1500.0, 300.0
    K = lambda T: a * T + 0.5 * b * T**2
    errors = []
    for n in (100, 200, 400):
        w = Wall([Layer(mat, L, n)])
        res = solve(w, Tc, 1e12, 1e12, TemperatureBC(Th), TemperatureBC(Tc))
        Kx = K(Th) + (K(Tc) - K(Th)) * w.x / L
        T_exact = (-a + np.sqrt(a**2 + 2 * b * Kx)) / b
        errors.append(np.max(np.abs(res.T_cells[-1] - T_exact)))
    # k is evaluated at cell-centre temperatures: a second-order discretization error
    ratios = np.array(errors[:-1]) / np.array(errors[1:])
    assert np.all((ratios > 3.5) & (ratios < 4.5)), f"errors {errors}"
    assert errors[-1] < 0.02  # K, on a 1200 K span


def test_enthalpy_is_exact_integral_of_tabulated_cp():
    T = np.array([150.0, 200.0, 450.0, 1199.0, 2500.0])
    e = SHIELD.enthalpy(T)
    from scipy.integrate import quad
    for Ti, ei in zip(T[1:], e[1:]):
        ref = quad(lambda s: float(SHIELD.cp(s)), T[0], Ti, points=T_TABLE, limit=200)[0]
        assert ei - e[0] == pytest.approx(ref, rel=1e-10)
