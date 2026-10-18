"""Stagnation-point convective heating (spec §3).

    Sutton–Graves:  q_conv = k sqrt(rho / R_n) V^3,  k = 1.7415e-4 kg^(1/2)/m for Earth air (SI -> W/m^2)
                    (Sutton & Graves, NASA TR R-376, 1971)
    Hot-wall:       q_net = q_conv (1 - h_w/h_0),  h_0 = c_p T_inf + V^2/2,  h_w = c_p T_w
    Heat load:      Q = integral q dt  [J/m^2]

Assumptions (stated in the README): non-ablating wall, no shock-layer radiative heating (valid below
roughly 10 km/s), stagnation point only.
"""

import numpy as np

K_SUTTON_GRAVES_EARTH = 1.7415e-4  # kg^(1/2)/m
CP_AIR = 1004.5                    # J/(kg K)


def sutton_graves(rho, V, nose_radius, k=K_SUTTON_GRAVES_EARTH):
    """Cold-wall stagnation-point heat flux q = k sqrt(rho/R_n) V^3 [W/m^2]."""
    return k * np.sqrt(np.asarray(rho, dtype=float) / nose_radius) * np.asarray(V, dtype=float) ** 3


def total_enthalpy(V, T_inf, cp=CP_AIR):
    """Free-stream total enthalpy h_0 = c_p T_inf + V^2/2 [J/kg]."""
    return cp * np.asarray(T_inf, dtype=float) + 0.5 * np.asarray(V, dtype=float) ** 2


def hot_wall_factor(V, T_inf, T_wall, cp=CP_AIR):
    """(1 - h_w/h_0) with h_w = c_p T_w: the fraction of the cold-wall flux reaching a hot wall."""
    return 1.0 - cp * np.asarray(T_wall, dtype=float) / total_enthalpy(V, T_inf, cp)


def heat_load(t, q):
    """Q = integral q dt [J/m^2] (trapezoidal rule on the given samples)."""
    t, q = np.asarray(t, dtype=float), np.asarray(q, dtype=float)
    return float(np.sum(0.5 * (q[1:] + q[:-1]) * np.diff(t)))
