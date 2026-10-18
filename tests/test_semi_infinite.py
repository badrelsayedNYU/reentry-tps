"""Validation 1 (spec §5.1): semi-infinite solid under a constant surface flux.

    T(0, t) - T_i = (2 q / k) sqrt(alpha t / pi)        (Incropera, Fundamentals of Heat and Mass Transfer, §5.7)

A thick single-layer wall and short times keep the back face unaffected (penetration depth
sqrt(alpha t) = 3 cm << 1 m).
"""

import numpy as np
import pytest

from tps.conduction import AdiabaticBC, FluxBC, Layer, Wall, solve

Q, T_I, T_END = 1.0e5, 300.0, 100.0


def exact_surface(t, mat):
    alpha = mat.k(0.0) / (mat.rho * mat.cp(0.0))
    return T_I + 2 * Q / mat.k(0.0) * np.sqrt(alpha * t / np.pi)


def run(mat, n_cells, dt, theta=1.0, stretch=1.03):
    wall = Wall([Layer(mat, 1.0, n_cells, stretch)])
    return solve(wall, T_I, T_END, dt, FluxBC(Q), AdiabaticBC(), theta=theta)


@pytest.mark.parametrize("theta", [1.0, 0.5])
def test_surface_temperature_matches_exact(slab_material, theta):
    res = run(slab_material, 200, 0.05, theta)
    exact = exact_surface(res.t[1:], slab_material)
    rise = exact - T_I
    err = np.abs(res.T_surface[1:] - exact) / rise
    late = res.t[1:] >= 10.0  # away from the t^(1/2) start-up singularity
    assert err[late].max() < 0.005, f"max relative error {err[late].max():.2e}"


def test_back_face_untouched(slab_material):
    res = run(slab_material, 120, 0.1)
    assert abs(res.T_back[-1] - T_I) < 1e-9


def test_first_order_in_time(slab_material):
    """Backward Euler: halving dt halves the error (fine grid, so space error is negligible)."""
    err = [run(slab_material, 400, dt, stretch=1.02).T_surface[-1] - exact_surface(T_END, slab_material)
           for dt in (0.8, 0.4, 0.2)]
    ratios = np.array(err[:-1]) / np.array(err[1:])
    assert np.all((ratios > 1.8) & (ratios < 2.3)), f"errors {err}"


def test_second_order_in_space(slab_material):
    """Finite volume: doubling the cells divides the error by ~4 (tiny dt, so time error is negligible)."""
    err = []
    for n in (25, 50, 100):
        wall = Wall([Layer(slab_material, 0.3, n)])
        res = solve(wall, T_I, T_END, 0.01, FluxBC(Q), AdiabaticBC())
        err.append(res.T_surface[-1] - exact_surface(T_END, slab_material))
    ratios = np.array(err[:-1]) / np.array(err[1:])
    assert np.all((ratios > 3.5) & (ratios < 4.5)), f"errors {err}"
