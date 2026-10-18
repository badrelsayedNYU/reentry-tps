"""Atmosphere models (spec §2.1).

  - Exponential: rho(h) = rho0 exp(-h/H), rho0 = 1.225 kg/m^3, H = 7200 m by default. This is the
    model the trajectory uses (and the one Allen–Eggers assumes).
  - U.S. Standard Atmosphere 1976, 0-86 km: the seven geopotential layers with linear temperature
    (U.S. Standard Atmosphere, 1976, NOAA-S/T 76-1562, Table 4). Used for comparison and for the
    free-stream temperature T_inf in the surface radiation and hot-wall terms. Above 86 km the
    temperature is held at its 86 km value (186.87 K); there T_inf only enters through
    eps sigma T_inf^4 and c_p T_inf, both negligible against the heating.
"""

import numpy as np

RHO0 = 1.225        # kg/m^3
H_SCALE = 7200.0    # m

R_AIR = 287.05287   # J/(kg K)
G0 = 9.80665        # m/s^2
R_EARTH_GEOPOTENTIAL = 6356766.0  # m, used only to convert geometric to geopotential altitude

# base geopotential altitude [m], base temperature [K], lapse rate [K/m], base pressure [Pa]
_LAYERS = np.array([
    [0.0,      288.15,  -0.0065, 101325.0],
    [11000.0,  216.65,   0.0,     22632.06],
    [20000.0,  216.65,   0.0010,   5474.889],
    [32000.0,  228.65,   0.0028,    868.0187],
    [47000.0,  270.65,   0.0,       110.9063],
    [51000.0,  270.65,  -0.0028,     66.93887],
    [71000.0,  214.65,  -0.0020,      3.956420],
])
_H_TOP = 84852.0  # geopotential altitude of 86 km geometric


def exponential_density(h, rho0=RHO0, H=H_SCALE):
    """rho(h) = rho0 exp(-h/H)  (spec §2.1)."""
    return rho0 * np.exp(-np.asarray(h, dtype=float) / H)


def us1976(h):
    """U.S. Standard Atmosphere 1976 at geometric altitude h [m], 0-86 km.
    Returns dict with T [K], p [Pa], rho [kg/m^3]. Above 86 km, T is held constant (see module docstring)
    and p, rho are not returned meaningfully (nan)."""
    h = np.atleast_1d(np.asarray(h, dtype=float))
    Hg = R_EARTH_GEOPOTENTIAL * h / (R_EARTH_GEOPOTENTIAL + h)
    i = np.clip(np.searchsorted(_LAYERS[:, 0], np.minimum(Hg, _H_TOP), side="right") - 1, 0, len(_LAYERS) - 1)
    Hb, Tb, L, pb = _LAYERS[i].T
    dH = np.minimum(Hg, _H_TOP) - Hb
    T = Tb + L * dH
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(L == 0, pb * np.exp(-G0 * dH / (R_AIR * Tb)), pb * (T / Tb) ** (-G0 / (R_AIR * L)))
    above = Hg > _H_TOP + 1.0  # 86 km geometric itself is included
    p = np.where(above, np.nan, p)
    return {"T": T, "p": p, "rho": p / (R_AIR * T)}


def temperature(h):
    """Free-stream temperature T_inf(h) [K] from the 1976 standard (constant above 86 km)."""
    return us1976(h)["T"]
