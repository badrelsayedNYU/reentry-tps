"""Validation 3 (spec §5.3): two-layer wall at steady state with fixed temperatures at both ends.

The profile is linear in each layer, and continuity of flux q = -k dT/dx at the interface gives
slope_A / slope_B = k_B / k_A. The interface temperature follows from series thermal resistance:
    q = (T_hot - T_cold) / (L_A/k_A + L_B/k_B),   T_int = T_hot - q L_A / k_A.
"""

import numpy as np
import pytest

from tps.conduction import Layer, TemperatureBC, Wall, solve

T_HOT, T_COLD, LA, LB = 1200.0, 300.0, 0.02, 0.005


@pytest.mark.parametrize("stretch", [1.0, 1.08])
def test_piecewise_linear_steady_state(two_materials, stretch):
    A, B = two_materials
    wall = Wall([Layer(A, LA, 40, stretch), Layer(B, LB, 15, stretch)])
    # one backward-Euler step with an enormous dt is the steady solution
    res = solve(wall, T_COLD, 1e12, 1e12, TemperatureBC(T_HOT), TemperatureBC(T_COLD))
    T = res.T_cells[-1]
    inA, inB = wall.owner == 0, wall.owner == 1

    slope_A = np.polyfit(wall.x[inA], T[inA], 1)
    slope_B = np.polyfit(wall.x[inB], T[inB], 1)
    np.testing.assert_allclose(np.polyval(slope_A, wall.x[inA]), T[inA], atol=1e-8)  # linear
    np.testing.assert_allclose(np.polyval(slope_B, wall.x[inB]), T[inB], atol=1e-8)
    assert slope_A[0] / slope_B[0] == pytest.approx(B.k(0.0) / A.k(0.0), rel=1e-9)

    q = (T_HOT - T_COLD) / (LA / A.k(0.0) + LB / B.k(0.0))
    np.testing.assert_allclose(res.flux_front[-1], q, rtol=1e-9)
    assert np.polyval(slope_A, LA) == pytest.approx(T_HOT - q * LA / A.k(0.0), rel=1e-9)


def test_cells_do_not_straddle_layers(two_materials):
    A, B = two_materials
    wall = Wall([Layer(A, 0.013, 17, 1.1), Layer(B, 0.0041, 9), Layer(A, 0.002, 4)])
    assert np.any(np.isclose(wall.faces, 0.013)) and np.any(np.isclose(wall.faces, 0.0171))
    assert wall.faces[-1] == pytest.approx(0.0191)
    assert wall.dx[1] / wall.dx[0] == pytest.approx(1.1)  # stretched, smallest cell at the hot face
