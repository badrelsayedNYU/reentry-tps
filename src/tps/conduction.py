"""1D transient conduction through a multi-layer wall: implicit finite volume (spec §4).

    rho c_p(T) dT/dt = d/dx( k(T) dT/dx ),   x = 0 at the hot surface

Discretization
  - Cells never straddle a layer boundary; each layer can be stretched geometrically so that
    cells cluster at its hot face.
  - Conductance between two cell centres is the series (distance-weighted harmonic) mean
        G = 1 / ( dx_i/(2 k_i) + dx_j/(2 k_j) ),
    which reduces to the harmonic mean 2 k_i k_j/(k_i + k_j)/dx on a uniform grid and gives the
    exact flux through a material interface (Patankar, Numerical Heat Transfer and Fluid Flow, §4.2).
  - Each wall surface is a zero-capacity node: its temperature is the true face temperature, set by
    the flux balance at the face. The unknowns are [T_front, T_cell_1..N, T_back] (tridiagonal).
  - Time: theta scheme, theta = 1 backward Euler (default), theta = 1/2 Crank–Nicolson.
  - Energy-conserving form: rho dx [e(T^{n+1}) - e(T^n)]/dt = theta F^{n+1} + (1 - theta) F^n with
    e(T) = integral c_p dT, so the stored energy changes by exactly the net boundary heat input.
  - Nonlinearities (k(T), c_p(T), boundary fluxes depending on T) are iterated within each step
    until max |dT| < tol: Picard on k, Newton on e(T) and on the boundary flux.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_banded


# ---------------------------------------------------------------- geometry

@dataclass
class Layer:
    material: object
    thickness: float
    n_cells: int
    stretch: float = 1.0   # dx_{j+1}/dx_j inside the layer, j = 0 at the layer's hot face


class Wall:
    def __init__(self, layers):
        self.layers = list(layers)
        dx, owner = [], []
        for idx, layer in enumerate(self.layers):
            n, r, L = layer.n_cells, layer.stretch, layer.thickness
            widths = np.full(n, L / n) if r == 1.0 else L * (r - 1) / (r**n - 1) * r ** np.arange(n)
            dx.append(widths)
            owner += [idx] * n
        self.dx = np.concatenate(dx)
        self.owner = np.array(owner)
        self.faces = np.r_[0.0, np.cumsum(self.dx)]
        self.x = 0.5 * (self.faces[:-1] + self.faces[1:])
        self.rho = np.array([self.layers[i].material.rho for i in self.owner])
        self.interfaces = np.cumsum([layer.thickness for layer in self.layers])[:-1]

    @property
    def n_cells(self):
        return self.dx.size

    @property
    def thickness(self):
        return self.faces[-1]

    def _by_layer(self, T, attr):
        out = np.empty_like(T)
        for idx, layer in enumerate(self.layers):
            m = self.owner == idx
            out[m] = getattr(layer.material, attr)(T[m])
        return out

    def k(self, T):
        return self._by_layer(T, "k")

    def cp(self, T):
        return self._by_layer(T, "cp")

    def enthalpy(self, T):
        out = np.empty_like(T)
        for idx, layer in enumerate(self.layers):
            m = self.owner == idx
            out[m] = layer.material.enthalpy(T[m])
        return out

    def stored_energy(self, T_cells):
        """Sum of rho e(T) dx over the cells [J/m^2]."""
        return float(np.sum(self.rho * self.enthalpy(T_cells) * self.dx))

    def conductances(self, T_cells):
        """(g_front, g_interior[N-1], g_back) at cell temperatures T_cells."""
        k = self.k(T_cells)
        half = self.dx / (2.0 * k)
        return 1.0 / half[0], 1.0 / (half[:-1] + half[1:]), 1.0 / half[-1]


# ---------------------------------------------------------------- boundary conditions
# Each BC either fixes the surface temperature (fixed(t) returns a value) or supplies the heat
# flux INTO the wall through that surface as a function of the surface temperature, linearized
# about T*: q(T) ~ q0 + dq (T - T*).

class FluxBC:
    """Prescribed heat flux into the wall, q(t) [W/m^2] (q may be a constant or a callable)."""

    def __init__(self, q):
        self.q = q if callable(q) else (lambda t, q=float(q): q)

    def fixed(self, t):
        return None

    def linearize(self, t, T_star):
        return self.q(t), 0.0


class TemperatureBC:
    """Prescribed surface temperature T(t) [K] (constant or callable)."""

    def __init__(self, T):
        self.T = T if callable(T) else (lambda t, T=float(T): T)

    def fixed(self, t):
        return self.T(t)


class AdiabaticBC(FluxBC):
    """No heat flux through the surface."""

    def __init__(self):
        super().__init__(0.0)


SIGMA = 5.670374419e-8  # Stefan–Boltzmann constant [W/(m^2 K^4)] (CODATA 2018)


class RadiatingFluxBC:
    """Heated, re-radiating surface (spec §3, surface energy balance):

        -k dT/dx|_surface = q(t, T_s) - eps sigma (T_s^4 - T_inf(t)^4)

    q may depend on the surface temperature (e.g. the hot-wall correction); it is then given as
    q(t, T_s) together with its derivative dq_dT(t, T_s). The T^4 term is Newton-linearized:
        eps sigma T^4 ~ eps sigma [T*^4 + 4 T*^3 (T - T*)].
    With emissivity 0 this is a plain flux BC; with q = 0 it is a radiating back wall.
    """

    def __init__(self, q, emissivity, T_inf, dq_dT=None):
        self.q = q if callable(q) else (lambda t, Ts, q=float(q): q)
        self.dq_dT = dq_dT or (lambda t, Ts: 0.0)
        self.eps = float(emissivity)
        self.T_inf = T_inf if callable(T_inf) else (lambda t, T=float(T_inf): T)

    def fixed(self, t):
        return None

    def radiated(self, t, Ts):
        """Net radiated flux eps sigma (T_s^4 - T_inf^4) [W/m^2]."""
        return self.eps * SIGMA * (Ts**4 - self.T_inf(t) ** 4)

    def linearize(self, t, T_star):
        q0 = self.q(t, T_star) - self.radiated(t, T_star)
        dq = self.dq_dT(t, T_star) - 4.0 * self.eps * SIGMA * T_star**3
        return q0, dq


def RadiationBC(emissivity, T_env):
    """Back wall radiating to an environment (e.g. the cabin) at T_env: no other heat input."""
    return RadiatingFluxBC(0.0, emissivity, T_env)


# ---------------------------------------------------------------- solver

@dataclass
class ConductionResult:
    t: np.ndarray                 # time levels, shape (S,)
    T: np.ndarray                 # [T_front, cells..., T_back] at each level, shape (S, N+2)
    wall: Wall
    flux_front: np.ndarray        # heat flux into the wall through the front face at each level
    flux_back: np.ndarray         # heat flux out of the wall through the back face at each level
    energy_in: np.ndarray         # cumulative theta-weighted net heat input [J/m^2]
    iterations: np.ndarray = field(default=None)

    @property
    def T_surface(self):
        return self.T[:, 0]

    @property
    def T_back(self):
        return self.T[:, -1]

    @property
    def T_cells(self):
        return self.T[:, 1:-1]

    def T_at(self, x):
        """Temperature history at depth x, linear in the cell/face values."""
        xs = np.r_[0.0, self.wall.x, self.wall.thickness]
        return np.array([np.interp(x, xs, row) for row in self.T])


def _surface_row(bc, t, T_star, g, T_neighbor_star):
    """Linear equation a_s T_s + a_n T_n = b for a surface node.
    Fixed:  T_s = T_given.   Flux:  q(T_s) - g (T_s - T_n) = 0, q linearized about T_star."""
    T_fix = bc.fixed(t)
    if T_fix is not None:
        return 1.0, 0.0, T_fix
    q0, dq = bc.linearize(t, T_star)
    return dq - g, g, -q0 + dq * T_star


def solve(wall, T_initial, t_end, dt, front, back, theta=1.0, t_start=0.0, tol=1e-6, max_iter=50):
    """March the wall from t_start to t_end with step dt (last step clipped to land on t_end).

    T_initial: scalar or array over cells. Returns a ConductionResult with every time level.
    """
    N = wall.n_cells
    cells = np.full(N, float(T_initial)) if np.isscalar(T_initial) else np.asarray(T_initial, float).copy()
    t = t_start
    u = _initial_state(wall, cells, t, front, back, tol, max_iter)

    times, states, f_front, f_back, iters = [t], [u.copy()], [], [], []
    gf, gi, gb = wall.conductances(u[1:-1])
    f_front.append(gf * (u[0] - u[1]))
    f_back.append(gb * (u[-2] - u[-1]))
    energy = [0.0]

    while t < t_end - 1e-12 * max(1.0, t_end):
        h = min(dt, t_end - t)
        t_new = t + h
        u_old = u
        e_old = wall.enthalpy(u_old[1:-1])
        flux_old = _cell_net_flux(wall, u_old)
        u_star = u_old.copy()
        for it in range(1, max_iter + 1):
            u_new = _step_linear(wall, u_star, e_old, flux_old, h, t_new, theta, front, back)
            change = np.max(np.abs(u_new - u_star))
            u_star = u_new
            if change < tol:
                break
        else:
            raise RuntimeError(f"nonlinear iteration did not converge at t = {t_new:g} (last change {change:.2e} K)")
        u, t = u_star, t_new
        gf, gi, gb = wall.conductances(u[1:-1])
        f_front.append(gf * (u[0] - u[1]))
        f_back.append(gb * (u[-2] - u[-1]))
        energy.append(energy[-1] + h * (theta * (f_front[-1] - f_back[-1]) + (1 - theta) * (f_front[-2] - f_back[-2])))
        times.append(t)
        states.append(u.copy())
        iters.append(it)

    return ConductionResult(np.array(times), np.array(states), wall, np.array(f_front),
                            np.array(f_back), np.array(energy), np.array(iters))


def _cell_net_flux(wall, u):
    """Net heat flux into each cell (left face in minus right face out) at state u."""
    gf, gi, gb = wall.conductances(u[1:-1])
    T = u[1:-1]
    faces = np.r_[gf * (u[0] - T[0]), gi * (T[:-1] - T[1:]), gb * (T[-1] - u[-1])]
    return faces[:-1] - faces[1:]


def _step_linear(wall, u_star, e_old, flux_old, h, t_new, theta, front, back):
    """One linearized solve about u_star (one Picard/Newton iteration)."""
    N = wall.n_cells
    T_star = u_star[1:-1]
    gf, gi, gb = wall.conductances(T_star)
    cap = wall.rho * wall.dx * wall.cp(T_star) / h
    lower = np.zeros(N + 2)   # coefficient of u[j-1] in row j
    diag = np.zeros(N + 2)
    upper = np.zeros(N + 2)   # coefficient of u[j+1] in row j
    rhs = np.zeros(N + 2)

    # front surface node
    a_s, a_n, b = _surface_row(front, t_new, u_star[0], gf, T_star[0])
    diag[0], upper[0], rhs[0] = a_s, a_n, b

    # cells: cap (T - T*) + rho dx (e(T*) - e_old)/h = theta * net_flux(T) + (1 - theta) * flux_old
    g_left = np.r_[gf, gi]
    g_right = np.r_[gi, gb]
    rows = np.arange(1, N + 1)
    diag[rows] = cap + theta * (g_left + g_right)
    lower[rows] = -theta * g_left
    upper[rows] = -theta * g_right
    rhs[rows] = cap * T_star - wall.rho * wall.dx * (wall.enthalpy(T_star) - e_old) / h \
        + (1 - theta) * flux_old

    # back surface node: q_into(T_b) = g_b (T_b - T_N)
    a_s, a_n, b = _surface_row(back, t_new, u_star[-1], gb, T_star[-1])
    diag[-1], lower[-1], rhs[-1] = a_s, a_n, b

    ab = np.zeros((3, N + 2))
    ab[0, 1:] = upper[:-1]
    ab[1] = diag
    ab[2, :-1] = lower[1:]
    return solve_banded((1, 1), ab, rhs)


def _initial_state(wall, cells, t, front, back, tol, max_iter):
    """Surface node temperatures consistent with the BCs and the initial cell temperatures."""
    u = np.r_[cells[0], cells, cells[-1]]
    gf, _, gb = wall.conductances(cells)
    for bc, idx, nb, g in ((front, 0, 1, gf), (back, -1, -2, gb)):
        for _ in range(max_iter):
            a_s, a_n, b = _surface_row(bc, t, u[idx], g, u[nb])
            new = (b - a_n * u[nb]) / a_s
            done = abs(new - u[idx]) < tol
            u[idx] = new
            if done:
                break
    return u
