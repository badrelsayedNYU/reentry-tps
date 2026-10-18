"""Validation 2 (spec §5.2): finite slab, sudden surface temperature, adiabatic back face.

Slab 0 <= x <= L at T_i; at t = 0+ the face x = 0 is held at T_s, x = L is adiabatic:

    (T - T_s)/(T_i - T_s) = sum_{n>=0} 4/((2n+1) pi) sin(lambda_n x) exp(-lambda_n^2 alpha t),
    lambda_n = (2n+1) pi / (2L)                         (Incropera §5.5; Carslaw & Jaeger §3.3)
"""

import numpy as np
import pytest

from tps.conduction import AdiabaticBC, Layer, TemperatureBC, Wall, solve

L, T_I, T_S = 0.05, 300.0, 1000.0


def exact(x, t, alpha, terms=200):
    n = np.arange(terms)[:, None]
    lam = (2 * n + 1) * np.pi / (2 * L)
    series = np.sum(4 / ((2 * n + 1) * np.pi) * np.sin(lam * x) * np.exp(-lam**2 * alpha * t), axis=0)
    return T_S + (T_I - T_S) * series


@pytest.mark.parametrize("t_check", [10.0, 60.0, 250.0])
def test_profile_matches_fourier_series(slab_material, t_check):
    alpha = 10.0 / (1000.0 * 1000.0)
    wall = Wall([Layer(slab_material, L, 200)])
    res = solve(wall, T_I, t_check, 0.01, TemperatureBC(T_S), AdiabaticBC())
    T_num = res.T_cells[-1]
    T_ex = exact(wall.x, t_check, alpha)
    err = np.max(np.abs(T_num - T_ex)) / (T_S - T_I)
    assert err < 2e-3, f"max error {err:.2e} of the 700 K step"


def test_adiabatic_back_face_has_zero_flux(slab_material):
    wall = Wall([Layer(slab_material, L, 50)])
    res = solve(wall, T_I, 30.0, 0.1, TemperatureBC(T_S), AdiabaticBC())
    np.testing.assert_allclose(res.flux_back, 0.0, atol=1e-8)
