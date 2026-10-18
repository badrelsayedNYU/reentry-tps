"""Heat-shield sizing (spec section 6).

Sizing: find the smallest heat-shield thickness L for which the peak bondline temperature over the
whole entry plus soak-back equals T_limit - margin. The peak decreases monotonically with L, so the
root is bracketed on [L_lo, L_hi]; Brent's method (bracketed bisection/secant, scipy.optimize.brentq)
finds it in fewer coupled runs than plain bisection. The result is never thinner than needed
and at most `tol` thicker.

Areal mass: m'' = sum_i rho_i L_i over the layers [kg/m^2].
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import brentq

from tps.conduction import Layer, Wall
from tps.coupling import run
from tps.materials import MissingDataError, _todos, load_materials
from tps.trajectory import Vehicle, fly


@dataclass
class Case:
    shield: object            # Material
    bondline: object
    structure: object
    bondline_thickness: float
    structure_thickness: float
    initial_temperature: float
    V_E: float
    gamma_E_deg: float
    beta: float
    lift_to_drag: float
    nose_radius: float
    h_E: float = 120e3
    V_cutoff: float = 500.0
    hot_wall: bool = False
    shield_cells: int = 60
    shield_stretch: float = 1.05
    bondline_cells: int = 4
    structure_cells: int = 6
    dt: float = 0.5
    soak_time: float = 1800.0
    max_soak_time: float = 4 * 3600.0
    T_limit: float = None
    margin: float = None

    @property
    def T_target(self):
        return self.T_limit - self.margin

    def trajectory(self):
        return fly(Vehicle(self.beta, self.lift_to_drag, self.nose_radius), self.V_E, self.gamma_E_deg,
                   h_E=self.h_E, V_cutoff=self.V_cutoff)

    def wall(self, shield_thickness):
        return Wall([Layer(self.shield, shield_thickness, self.shield_cells, self.shield_stretch),
                     Layer(self.bondline, self.bondline_thickness, self.bondline_cells),
                     Layer(self.structure, self.structure_thickness, self.structure_cells)])

    def areal_mass(self, shield_thickness):
        return (self.shield.rho * shield_thickness + self.bondline.rho * self.bondline_thickness
                + self.structure.rho * self.structure_thickness)


def load_case(materials_file, baseline_file, shield=None):
    """Build a Case from the two data files; refuses while any needed value is TODO."""
    import yaml

    with open(baseline_file) as fh:
        base = yaml.safe_load(fh)
    w, m, n = base["wall"], base["mission"], base["numerics"]
    keys = [shield or w["heat_shield"], w["bondline"], w["structure"]]
    missing = _todos({"mission": m, "wall": w}, "baseline")
    try:
        mats, data = load_materials(materials_file, require=keys)
    except MissingDataError as err:
        missing.append(str(err))
        mats, data = None, None
    limits = {} if data is None else data.get("limits", {})
    missing += _todos(limits, "materials.limits")
    if missing:
        raise MissingDataError("values still to be supplied: " + ", ".join(missing))
    return Case(mats[keys[0]], mats[keys[1]], mats[keys[2]], float(w["bondline_thickness"]),
                float(w["structure_thickness"]), float(w["initial_temperature"]), float(m["V_E"]),
                float(m["gamma_E_deg"]), float(m["beta"]), float(m["lift_to_drag"]),
                float(m["nose_radius"]), float(m["h_E"]), float(m["V_cutoff"]), bool(w["hot_wall"]),
                int(n["shield_cells"]), float(n["shield_stretch"]), int(n["bondline_cells"]),
                int(n["structure_cells"]), float(n["dt"]), float(n["soak_time"]),
                T_limit=float(limits["bondline_temperature_limit"]), margin=float(limits["margin"]))


def simulate(case, shield_thickness, trajectory=None):
    """Coupled run, extending the soak window until the bondline peak is captured."""
    tr = trajectory or case.trajectory()
    soak = case.soak_time
    while True:
        res = run(tr, case.wall(shield_thickness), case.nose_radius, case.shield.emissivity,
                  case.initial_temperature, dt=case.dt, soak_time=soak, hot_wall=case.hot_wall)
        if res.summary()["bondline_peak_captured"] or soak >= case.max_soak_time:
            return res
        soak = min(2 * soak, case.max_soak_time)


def peak_bondline(case, shield_thickness, trajectory=None):
    return simulate(case, shield_thickness, trajectory).summary()["peak_T_bondline_K"]


@dataclass
class SizingResult:
    thickness: float          # m
    areal_mass: float         # kg/m^2 (all three layers)
    peak_T_bondline: float    # K at the sized thickness
    peak_T_surface: float     # K at the sized thickness
    iterations: int           # coupled runs used by the root finder


def size(case, trajectory=None, L_lo=0.002, L_hi=0.25, tol=1e-4):
    """Root-find on heat-shield thickness: peak T_bondline(L) = T_limit - margin (spec section 6.1)."""
    tr = trajectory or case.trajectory()
    f = lambda L: peak_bondline(case, L, tr) - case.T_target
    if f(L_hi) > 0:
        raise ValueError(f"even {L_hi * 100:.0f} cm of {case.shield.name} exceeds the bondline target")
    if f(L_lo) <= 0:
        L, it = L_lo, 0   # the thinnest allowed layer already suffices
    else:
        # brentq stops with the root inside a bracket narrower than xtol around `root`, so with
        # xtol = tol/2 the true minimum L* satisfies root - tol/2 <= L* <= root + tol/2:
        # returning root + tol/2 is never too thin and at most tol thicker than needed.
        root, info = brentq(f, L_lo, L_hi, xtol=0.5 * tol, rtol=1e-12, full_output=True)
        L, it = root + 0.5 * tol, info.function_calls
    s = simulate(case, L, tr).summary()
    return SizingResult(L, case.areal_mass(L), s["peak_T_bondline_K"], s["peak_T_surface_K"], it)


def sweep(case, parameter, values, **size_kwargs):
    """Size the shield for each value of a Case parameter (V_E, gamma_E_deg, beta, nose_radius)."""
    out = []
    for v in values:
        c = replace(case, **{parameter: v})
        tr = c.trajectory()
        r = size(c, tr, **size_kwargs)
        s = simulate(c, r.thickness, tr)
        out.append(dict(value=v, thickness=r.thickness, areal_mass=r.areal_mass,
                        peak_q=s.summary()["peak_q_W_m2"], heat_load=s.summary()["heat_load_J_m2"],
                        peak_g=float(tr.g_load.max())))
    return out
