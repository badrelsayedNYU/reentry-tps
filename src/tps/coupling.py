"""Trajectory -> stagnation heating -> wall conduction (spec §4, time coupling).

The trajectory is solved first. Its Sutton–Graves heat flux q(t) is interpolated onto the
conduction time steps and applied at the outer surface together with re-radiation:

    -k dT/dx|_0 = q_conv(t) [x (1 - h_w/h_0) if hot-wall] - eps sigma (T_s^4 - T_inf^4)

After the trajectory ends (velocity cutoff) the convective heating is zero, but the conduction run
continues for a soak period: the bondline usually peaks well after the surface does.
T_inf(t) is the 1976 standard-atmosphere temperature at the vehicle's altitude (held at its final
value during the soak).
"""

from dataclasses import dataclass

import numpy as np

from tps.atmosphere import temperature
from tps.conduction import AdiabaticBC, RadiatingFluxBC, solve
from tps.heating import CP_AIR, heat_load, sutton_graves, total_enthalpy


@dataclass
class CoupledResult:
    trajectory: object
    conduction: object
    q_conv: np.ndarray        # applied convective heating at the conduction time levels [W/m^2]
    q_rad: np.ndarray         # re-radiated flux at the conduction time levels [W/m^2]
    bondline_depth: float     # x of the heat-shield / bondline interface [m]

    @property
    def t(self):
        return self.conduction.t

    @property
    def T_surface(self):
        return self.conduction.T_surface

    @property
    def T_bondline(self):
        """Temperature at the heat-shield / bondline interface (the bondline's hot face)."""
        return self.conduction.T_at(self.bondline_depth)

    def summary(self):
        i_q = int(np.argmax(self.q_conv))
        T_b = self.T_bondline
        i_b = int(np.argmax(T_b))
        return {
            "peak_q_W_m2": float(self.q_conv[i_q]),
            "t_peak_q_s": float(self.t[i_q]),
            "heat_load_J_m2": heat_load(self.t, self.q_conv),
            "peak_T_surface_K": float(self.T_surface.max()),
            "t_peak_T_surface_s": float(self.t[int(np.argmax(self.T_surface))]),
            "peak_T_bondline_K": float(T_b[i_b]),
            "t_peak_T_bondline_s": float(self.t[i_b]),
            # False if the bondline is still heating when the run ends: the soak was too short and
            # the reported peak is only a lower bound
            "bondline_peak_captured": bool(i_b < len(T_b) - 1 and T_b[-1] < T_b[i_b] - 1e-6),
        }


def run(trajectory, wall, nose_radius, emissivity, T_initial, dt=0.1, soak_time=1800.0,
        hot_wall=False, back=None, theta=1.0):
    """Couple a Trajectory to a Wall (spec §4). The bondline is taken as the interface between the
    first layer (heat shield) and the second."""
    t_traj = trajectory.t
    q_traj = sutton_graves(trajectory.rho, trajectory.V, nose_radius)
    T_inf_traj = temperature(trajectory.h)
    h0_traj = total_enthalpy(trajectory.V, T_inf_traj)
    t_end_heat = t_traj[-1]

    q_cold = lambda t: float(np.interp(t, t_traj, q_traj, right=0.0))
    T_inf = lambda t: float(np.interp(t, t_traj, T_inf_traj))
    h0 = lambda t: float(np.interp(t, t_traj, h0_traj))

    if hot_wall:
        q = lambda t, Ts: q_cold(t) * (1.0 - CP_AIR * Ts / h0(t)) if t <= t_end_heat else 0.0
        dq = lambda t, Ts: -q_cold(t) * CP_AIR / h0(t) if t <= t_end_heat else 0.0
    else:
        q = lambda t, Ts: q_cold(t)
        dq = None
    front = RadiatingFluxBC(q, emissivity, T_inf, dq_dT=dq)
    res = solve(wall, T_initial, t_end_heat + soak_time, dt, front, back or AdiabaticBC(), theta=theta)

    q_applied = np.array([q(t, Ts) for t, Ts in zip(res.t, res.T_surface)])
    q_rad = np.array([front.radiated(t, Ts) for t, Ts in zip(res.t, res.T_surface)])
    return CoupledResult(trajectory, res, q_applied, q_rad, float(wall.interfaces[0]))
