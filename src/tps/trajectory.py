"""Point-mass entry trajectory and the Allen–Eggers ballistic solution (spec §2).

Planar flight over a non-rotating spherical Earth (Vinh, Busemann & Culp, Hypersonic and
Planetary Entry Flight Mechanics, ch. 2):

    dV/dt     = -D/m - g sin(gamma)
    dgamma/dt = [ L/m - (g - V^2/(R_E + h)) cos(gamma) ] / V
    dh/dt     = V sin(gamma)
    ds/dt     = V R_E/(R_E + h) cos(gamma)            (downrange)

with D/m = rho V^2 / (2 beta), beta = m/(C_D A), L = (L/D) D, g = g0 (R_E/(R_E + h))^2, and the
flight-path angle gamma negative when descending. R_E = 6371 km (mean Earth radius).

Switches for validation: gravity=False sets g = 0; curvature=False removes the V^2/(R_E + h) term
and the R_E/(R_E + h) factors (flat Earth). With both off and L/D = 0, gamma stays constant and
the equations reduce exactly to the Allen–Eggers assumptions.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from tps.atmosphere import H_SCALE, RHO0

R_EARTH = 6.371e6   # m, mean radius
G0 = 9.80665        # m/s^2
H_ENTRY = 120e3     # m, entry interface (spec §2.2)


@dataclass
class Vehicle:
    beta: float                 # ballistic coefficient m/(C_D A) [kg/m^2]
    lift_to_drag: float = 0.0
    nose_radius: float = None   # m (used by the heating model)


@dataclass
class Trajectory:
    t: np.ndarray
    V: np.ndarray
    gamma: np.ndarray           # rad
    h: np.ndarray
    s: np.ndarray
    rho: np.ndarray
    accel: np.ndarray           # aerodynamic load |L + D|/m [m/s^2]
    sol: object                 # solve_ivp solution with dense output
    vehicle: Vehicle
    params: dict

    @property
    def g_load(self):
        return self.accel / G0

    def state(self, t):
        """Dense-output state [V, gamma, h, s] at time(s) t."""
        return self.sol.sol(t)

    def peak(self, f):
        """(t, value) at the maximum of f(V, gamma, h, rho), refined on the dense solution."""
        vals = f(self.V, self.gamma, self.h, self.rho)
        i = int(np.argmax(vals))
        lo, hi = self.t[max(i - 1, 0)], self.t[min(i + 1, self.t.size - 1)]

        def neg(tt):
            V, g, h, _ = self.sol.sol(tt)
            return -f(V, g, h, self.params["rho0"] * np.exp(-h / self.params["H"]))

        r = minimize_scalar(neg, bounds=(lo, hi), method="bounded", options={"xatol": 1e-9})
        return r.x, -r.fun


def fly(vehicle, V_E, gamma_E_deg, h_E=H_ENTRY, H=H_SCALE, rho0=RHO0, V_cutoff=500.0,
        gravity=True, curvature=True, rtol=1e-8, atol=None, n_samples=None):
    """Integrate from the entry interface until h = 0 or V = V_cutoff (RK45, spec §2.2)."""
    beta, LD = vehicle.beta, vehicle.lift_to_drag

    def rhs(t, y):
        V, g_ang, h, s = y
        rho = rho0 * np.exp(-h / H)
        Dm = rho * V**2 / (2.0 * beta)
        r = R_EARTH + h
        g = (G0 * (R_EARTH / r) ** 2 if curvature else G0) if gravity else 0.0
        centrifugal = V**2 / r if curvature else 0.0
        geo = R_EARTH / r if curvature else 1.0
        return [-Dm - g * np.sin(g_ang),
                (LD * Dm - (g - centrifugal) * np.cos(g_ang)) / V,
                V * np.sin(g_ang),
                V * geo * np.cos(g_ang)]

    ground = lambda t, y: y[2]
    ground.terminal, ground.direction = True, -1
    slow = lambda t, y: y[0] - V_cutoff
    slow.terminal, slow.direction = True, -1
    y0 = [V_E, np.deg2rad(gamma_E_deg), h_E, 0.0]
    atol = atol if atol is not None else [1e-6, 1e-12, 1e-6, 1e-6]
    sol = solve_ivp(rhs, (0.0, 1e5), y0, method="RK45", rtol=rtol, atol=atol,
                    events=[ground, slow], dense_output=True)
    if sol.status != 1:
        raise RuntimeError("trajectory did not reach the ground or the velocity cutoff")
    t_end = sol.t[-1]
    n = n_samples or max(4000, int(t_end / 0.05))
    t = np.linspace(0.0, t_end, n)
    V, g_ang, h, s = sol.sol(t)
    rho = rho0 * np.exp(-h / H)
    accel = rho * V**2 / (2.0 * beta) * np.sqrt(1.0 + LD**2)
    params = dict(V_E=V_E, gamma_E_deg=gamma_E_deg, h_E=h_E, H=H, rho0=rho0, gravity=gravity,
                  curvature=curvature)
    return Trajectory(t, V, g_ang, h, s, rho, accel, sol, vehicle, params)


# ---------------------------------------------------------------- Allen–Eggers (spec §2.3)

def allen_eggers_velocity(h, V_E, gamma_E_deg, beta, H=H_SCALE, rho0=RHO0):
    """V(h) = V_E exp[-rho(h) H / (2 beta |sin gamma_E|)]  (Allen & Eggers, NACA Report 1381, 1958)."""
    rho = rho0 * np.exp(-np.asarray(h, dtype=float) / H)
    return V_E * np.exp(-rho * H / (2.0 * beta * abs(np.sin(np.deg2rad(gamma_E_deg)))))


def allen_eggers_peaks(V_E, gamma_E_deg, beta, H=H_SCALE, rho0=RHO0):
    """Peak deceleration and peak stagnation heating (q ~ sqrt(rho) V^3) of the ballistic solution:
      deceleration: rho = beta |sin g|/H,      V = V_E e^(-1/2),  a_max = V_E^2 |sin g| / (2 e H)
      heating:      rho = beta |sin g|/(3 H),  V = V_E e^(-1/6)
    Altitudes follow from h = -H ln(rho/rho0)."""
    sg = abs(np.sin(np.deg2rad(gamma_E_deg)))
    rho_a, rho_q = beta * sg / H, beta * sg / (3.0 * H)
    return {
        "decel": dict(rho=rho_a, h=-H * np.log(rho_a / rho0), V=V_E * np.exp(-0.5),
                      a=V_E**2 * sg / (2.0 * np.e * H)),
        "heating": dict(rho=rho_q, h=-H * np.log(rho_q / rho0), V=V_E * np.exp(-1.0 / 6.0)),
    }
